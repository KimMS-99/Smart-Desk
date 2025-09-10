#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, argparse
import numpy as np
import cv2

from iot_client_HB import IoTClient

try:
    from openpose import pyopenpose as op
except Exception as e:
    raise RuntimeError(
        "pyopenpose import 실패. 환경변수 설정 확인:\n"
        "  export PYTHONPATH=~/openpose/build/python:$PYTHONPATH\n"
        "  export LD_LIBRARY_PATH=~/openpose/build/src:$LD_LIBRARY_PATH"
    ) from e

# -------------------- 키포인트 인덱스 --------------------
KP = {
    "COCO": {
        "NOSE": 0, "NECK": 1,
        "R_SHOULDER": 2, "R_ELBOW": 3, "R_WRIST": 4,
        "L_SHOULDER": 5, "L_ELBOW": 6, "L_WRIST": 7,
        "MID_HIP": None,
        "R_HIP": 8, "R_KNEE": 9, "R_ANKLE": 10,
        "L_HIP": 11, "L_KNEE": 12, "L_ANKLE": 13
    },
    "BODY_25": {
        "NOSE": 0, "NECK": 1,
        "R_SHOULDER": 2, "R_ELBOW": 3, "R_WRIST": 4,
        "L_SHOULDER": 5, "L_ELBOW": 6, "L_WRIST": 7,
        "MID_HIP": 8, "R_HIP": 9, "L_HIP": 10,
        "R_KNEE": 11, "L_KNEE": 12, "R_ANKLE": 13, "L_ANKLE": 14
    }
}

# -------------------- 임계값 --------------------
NECK_OFFSET_THR = 0.22
NECK_TILT_THR   = 32.0
BACK_TILT_THR       = 18.0
SHOULDER_SLOPE_THR  = 14.0
LEG_LEGACY_ANKLES_CLOSE_RATIO = 0.08
LEG_LEGACY_ANKLE_YGAP         = 0.06
ANKLES_DIST_NORM_THR = 1.30
PROX_RATIO_THR       = 1.15
ANKLE_Y_GAP_THR      = 0.03
KNEE_SWAP_THR_PIX    = 22.0

# -------------------- 유틸 --------------------
def vangle_deg(v1, v2):
    a = np.asarray(v1, np.float32); b = np.asarray(v2, np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6: return None
    cos_th = float(np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_th)))

def dist(p, q): return float(np.linalg.norm(np.asarray(p, np.float32)-np.asarray(q, np.float32)))
def mid(p, q):  return ((p[0]+q[0])/2.0, (p[1]+q[1])/2.0)

def take(body, idx, thr=0.2):
    if idx is None: return None
    x,y,c = body[idx]
    return (float(x), float(y)) if c >= thr else None

def median_pose(buf): return np.median(np.stack(buf, axis=0), axis=0) if buf else None

def seg_intersect(p1, p2, p3, p4):
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def on_seg(a,b,p):
        return (min(a[0],b[0])-1e-6 <= p[0] <= max(a[0],b[0])+1e-6 and
                min(a[1],b[1])-1e-6 <= p[1] <= max(a[1],b[1])+1e-6)
    c1,c2 = cross(p1,p2,p3), cross(p1,p2,p4)
    c3,c4 = cross(p3,p4,p1), cross(p3,p4,p2)
    if c1==0 and on_seg(p1,p2,p3): return True
    if c2==0 and on_seg(p1,p2,p4): return True
    if c3==0 and on_seg(p3,p4,p1): return True
    if c4==0 and on_seg(p3,p4,p2): return True
    return (c1>0)!=(c2>0) and (c3>0)!=(c4>0)

# -------------------- 기준점 --------------------
def compute_anchors(body, mp, img_h):
    nose = take(body, mp["NOSE"]); neck = take(body, mp["NECK"])
    rs, ls = take(body, mp["R_SHOULDER"]), take(body, mp["L_SHOULDER"])
    rh, lh = take(body, mp["R_HIP"]), take(body, mp["L_HIP"])
    rk, lk = take(body, mp["R_KNEE"]), take(body, mp["L_KNEE"])
    ra, la = take(body, mp["R_ANKLE"]), take(body, mp["L_ANKLE"])
    mid_hip = take(body, mp.get("MID_HIP"))
    if mid_hip is None and (rh and lh): mid_hip = mid(rh, lh)
    sh_mid = mid(rs, ls) if (rs and ls) else None
    torso_len = (dist(neck, mid_hip) if (neck and mid_hip)
                 else (dist(sh_mid, mid_hip) if (sh_mid and mid_hip) else None))
    return {"nose":nose,"neck":neck,"sh_mid":sh_mid,"hip_mid":mid_hip,
            "rs":rs,"ls":ls,"rh":rh,"lh":lh,"rk":rk,"lk":lk,"ra":ra,"la":la,
            "torso_len":torso_len,"img_h":img_h}

# -------------------- 판정 --------------------
def judge_neck(a):
    nose, neck = a["nose"], a["neck"]
    if not (nose and neck): return False
    torso = a["torso_len"] or a["img_h"]
    offset = abs(nose[0]-neck[0]) / max(torso, 1.0)
    tilt = vangle_deg((nose[0]-neck[0], nose[1]-neck[1]), (0.0, -1.0))
    return (offset > NECK_OFFSET_THR) or (tilt is not None and tilt > NECK_TILT_THR)

def judge_slouch(a):
    sh_mid, hip_mid, rs, ls = a["sh_mid"], a["hip_mid"], a["rs"], a["ls"]
    if not (sh_mid and hip_mid and rs and ls): return False
    back_tilt = vangle_deg((hip_mid[0]-sh_mid[0], hip_mid[1]-sh_mid[1]), (0.0, 1.0))
    shoulder_slope = vangle_deg((ls[0]-rs[0], ls[1]-rs[1]), (1.0, 0.0))
    return ((back_tilt is not None and back_tilt >= BACK_TILT_THR) or
            (shoulder_slope is not None and shoulder_slope >= SHOULDER_SLOPE_THR))

def judge_leg_cross(a):
    rh, lh, rk, lk, ra, la, hip_mid = a["rh"], a["lh"], a["rk"], a["lk"], a["ra"], a["la"], a["hip_mid"]
    if not (rh and lh and rk and lk and ra and la and hip_mid): return False
    img_h = a["img_h"]; hip_w = dist(rh, lh) or 1.0

    hip_mid_x = hip_mid[0]
    ankle_right_side = ra[0] > hip_mid_x
    ankle_left_side  = la[0] > hip_mid_x
    crossed_sides = (ankle_right_side == ankle_left_side)
    knees_cross = ((rk[0]-lk[0]) * (ra[0]-la[0]) < 0)
    ankles_y_gap = abs(ra[1]-la[1]) / max(img_h, 1.0)
    ankles_close = dist(ra, la) < LEG_LEGACY_ANKLES_CLOSE_RATIO * img_h
    legacy = (crossed_sides and ankles_close) or knees_cross or (crossed_sides and ankles_y_gap > LEG_LEGACY_ANKLE_YGAP)

    same_side = np.sign(ra[0] - hip_mid_x) == np.sign(la[0] - hip_mid_x)
    ankles_dist_norm = dist(ra, la) / hip_w
    condA = bool(same_side and ankles_dist_norm < ANKLES_DIST_NORM_THR)
    order_knee = rk[0]-lk[0]; order_ankle = ra[0]-la[0]
    condB = bool(np.sign(order_knee) != np.sign(order_ankle) or abs(order_knee - order_ankle) < KNEE_SWAP_THR_PIX)
    condC = seg_intersect(rk, ra, lk, la)
    prox_r = dist(rk, la) / max(dist(rk, ra), 1e-3)
    prox_l = dist(lk, ra) / max(dist(lk, la), 1e-3)
    condD = bool(prox_r < PROX_RATIO_THR or prox_l < PROX_RATIO_THR)
    condE = bool(same_side and ankles_y_gap > ANKLE_Y_GAP_THR)

    return legacy or condA or condB or condC or condD or condE

# -------------------- UI --------------------
def draw_text(img, text, org, color=(0,255,0), scale=0.9, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_labels(img, latched):
    if latched == "neck":
        line, color = "WARN: Neck", (0,0,255)
    elif latched == "back":
        line, color = "WARN: Back", (0,0,255)
    elif latched == "leg":
        line, color = "WARN: Leg Cross", (0,0,255)
    else:
        line, color = "Posture OK", (0,200,0)
    draw_text(img, line, (10, 28), color)
    if line.startswith("WARN"):
        draw_text(img, "(latched)", (10, 58), color)

# -------------------- IoT 전송 헬퍼 --------------------
def client_send_any(client, line: str) -> bool:
    """IoTClient_HB의 실제 전송 메서드 명이 무엇이든 1회만 시도한다(재연결/close 하지 않음)."""
    if not line.endswith("\n"):
        line = line + "\n"
    for name in ("send_line","send_text","send","send_command","send_msg","send_message"):
        if hasattr(client, name):
            try:
                getattr(client, name)(line)
                return True
            except Exception as e:
                print(f"[WARN] client {name} error:", e)
                return False
    # 최후 수단: sock 속성 직접 사용
    try:
        sock = getattr(client, "sock", None)
        if sock:
            sock.sendall(line.encode("utf-8", "ignore"))
            return True
    except Exception as e:
        print("[WARN] direct socket send failed:", e)
    return False

def state_to_msg(state, user):
    if state == "neck": return f"AI:{user}:POSTURE:BAD:neck"
    if state == "back": return f"AI:{user}:POSTURE:BAD:back"
    if state == "leg":  return f"AI:{user}:POSTURE:BAD:leg"
    return f"AI:{user}:POSTURE:OK"

# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.0.158")
    ap.add_argument("--port", type=int, default=5000)

    # 로그인할 계정(서버 DB에 존재하는 계정)과
    # 메시지에 들어갈 사용자명(요청 포맷의 'seol')을 분리
    ap.add_argument("--login_user", default="SEOL_AND")
    ap.add_argument("--passwd",     default="PASSWD")
    ap.add_argument("--user",       default="seol")   # 메시지용 사용자명

    ap.add_argument("--model_pose", choices=["COCO","BODY_25"], default="COCO")
    ap.add_argument("--net_resolution", default="-1x160")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    print(f"[INFO] model_pose={args.model_pose}, net_resolution={args.net_resolution}")

    # IoT: 시작 시 1회만 접속 (전송 실패해도 close/reconnect 안 함)
    client = IoTClient(args.ip, args.port, args.login_user, args.passwd)
    try:
        client.connect()
    except Exception as e:
        print("[WARN] IoT connect failed:", e)

    params = {
        "model_folder": "models/",
        "model_pose": args.model_pose,
        "net_resolution": args.net_resolution,
        "number_people_max": 1,
        "disable_blending": False,
        "render_pose": 1,
        "render_threshold": 0.05,
        "logging_level": 3,
    }
    opw = op.WrapperPython(); opw.configure(params); opw.start()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERR] 카메라를 열 수 없습니다.")
        return

    mp = KP[args.model_pose]
    pose_buf, last_t = [], time.time()

    # 라칭(Sticky) 상태: None | "neck" | "back" | "leg"
    latched_state = None
    last_sent_msg = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            datum = op.Datum(); datum.cvInputData = frame
            datums = op.VectorDatum(); datums.append(datum)
            opw.emplaceAndPop(datums)

            out = datums[0].cvOutputData if datums[0].cvOutputData is not None else frame
            people = datums[0].poseKeypoints

            if isinstance(people, np.ndarray) and people.ndim == 3 and people.shape[0] >= 1:
                person = people[0]
                pose_buf.append(person)
                if len(pose_buf) > 60:
                    pose_buf = pose_buf[-60:]

                now = time.time()
                if now - last_t >= args.interval and pose_buf:
                    med = median_pose(pose_buf)
                    a = compute_anchors(med, mp, out.shape[0])

                    neck_bad  = judge_neck(a)
                    back_bad  = judge_slouch(a)
                    leg_bad   = judge_leg_cross(a)

                    # 읽기 쉬운 줄바꿈 로그
                    ts = time.strftime('%H:%M:%S')
                    print(f"[{ts}]")
                    print(f"neck={neck_bad}")
                    print(f"slouch={back_bad}")
                    print(f"leg_cross={leg_bad}")

                    new_state = ("neck" if neck_bad else
                                 "back" if back_bad else
                                 "leg"  if leg_bad  else
                                 None)

                    if new_state != latched_state:
                        latched_state = new_state
                        msg = state_to_msg(latched_state, args.user)
                        if msg != last_sent_msg:
                            client_send_any(client, msg)
                            last_sent_msg = msg

                    last_t = now
                    pose_buf.clear()
            else:
                pose_buf.clear()  # 미검출 시 라칭은 유지

            draw_labels(out, latched_state)
            cv2.imshow("OpenPose Posture Monitor", out)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break

    finally:
        cap.release(); cv2.destroyAllWindows()
        try: client.close()
        except: pass


if __name__ == "__main__":
    main()
