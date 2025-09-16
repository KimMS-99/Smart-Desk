# YOLO 커스텀 Keypoint 순서(모델 yaml 기준) 적용 버전
# 기존 주석 유지 + Flask + IoT + 스무딩 포함

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2, math, os, time, argparse, threading, socket
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response
try:
    from iot_client_HB import IoTClient
except ImportError:
    IoTClient = None

# -------------------- 키포인트 인덱스 (모델 정의) --------------------
# 첨부된 키포인트 이름/순서 기준 (0~19)
keypoint_index = {
    'neck1': 0,
    'neck2': 1,
    'left_shoulder': 2,
    'right_shoulder': 3,
    'left_arm': 4,
    'right_arm': 5,
    'back1': 6,
    'back2': 7,
    'waist': 8,
    'left_hip': 9,
    'right_hip': 10,
    'left_knee': 11,
    'left_ankle': 12,
    'right_knee': 13,
    'right_ankle': 14,
    'left_eye': 15,
    'nose': 16,
    'right_eye': 17,
    'right_ear': 18,
    'left_ear': 19
}

# 스켈레톤 연결 정의 (모델 맞춤)
skeleton = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_arm'),
    ('right_shoulder', 'right_arm'),
    ('left_shoulder', 'waist'),
    ('right_shoulder', 'waist'),
    ('waist', 'left_hip'),
    ('waist', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('nose','left_eye'), ('nose','right_eye'),
    ('left_eye','left_ear'), ('right_eye','right_ear')
]

angle_targets = {
    'Neck bent': ('neck1','neck2','back1'),
    'Back bent': ('neck2','waist','back2'),
    'Leg twist': ('left_hip','right_hip','right_knee')
}

def calculate_angle(a,b,c):
    if None in (a,b,c): return None
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = abs(ang)
    return ang if ang<180 else 360-ang

def get_center_person_index(results, frame_width):
    if not results: return None
    r0 = results[0]
    kobj = getattr(r0,"keypoints",None)
    if kobj is None: return None
    xy = getattr(kobj,"xy",None)
    centers_x = None
    if xy is not None:
        if hasattr(xy,"shape") and xy.shape[0]>0:
            centers_x = xy[...,0].mean(dim=1).detach().cpu().numpy()
    if centers_x is None:
        data = getattr(kobj,"data",None)
        if data is None or (hasattr(data,"shape") and data.shape[0]==0):
            return None
        centers_x = data[...,0].mean(dim=1).detach().cpu().numpy()
    if centers_x is None: return None
    centers_x = np.asarray(centers_x,dtype=float)
    if centers_x.size==0 or np.all(np.isnan(centers_x)): return None
    valid = ~np.isnan(centers_x)
    if not np.any(valid): return None
    frame_center = frame_width/2.0
    idx = int(np.argmin(np.abs(centers_x[valid]-frame_center)))
    return int(np.arange(len(centers_x))[valid][idx])

# Flask
app = Flask(__name__)
_latest_frame=None
_frame_lock=threading.Lock()

def _set_latest_frame(bgr):
    global _latest_frame
    with _frame_lock:
        _latest_frame = bgr.copy()

def _mjpeg_stream():
    while True:
        with _frame_lock:
            f=None if _latest_frame is None else _latest_frame.copy()
        if f is None:
            time.sleep(0.01); continue
        ok,jpg=cv2.imencode('.jpg',f,[int(cv2.IMWRITE_JPEG_QUALITY),80])
        if not ok: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n')

@app.get('/')
def _root():
    return '<img src="/video" style="max-width:100vw;max-height:100vh">'

@app.get('/video')
def _video():
    return Response(_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 색상 설정
SKEL_COLOR=(255,200,0)
JOINT_COLOR=(30,144,255)
WARN_COLOR=(0,0,255)
OK_COLOR=(0,200,0)
JOINT_RADIUS=3
THICK_LINE=2

_smooth_pts={}
_SMOOTH_ALPHA=0.5
_DIST_RATIO=0.35

def _ema(name,pt):
    if pt is None: return _smooth_pts.get(name)
    prev=_smooth_pts.get(name)
    if prev is None:
        _smooth_pts[name]=pt
    else:
        _smooth_pts[name]=(int(prev[0]*(1-_SMOOTH_ALPHA)+pt[0]*_SMOOTH_ALPHA),
                          int(prev[1]*(1-_SMOOTH_ALPHA)+pt[1]*_SMOOTH_ALPHA))
    return _smooth_pts[name]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='/home/jetson/custom_yolo/best.pt')
    parser.add_argument('--cam',type=int,default=0)
    parser.add_argument('--conf',type=float,default=0.3)
    parser.add_argument('--imgsz',type=int,default=320)
    parser.add_argument('--ip',default='192.168.0.158')
    parser.add_argument('--port',type=int,default=5000)
    parser.add_argument('--login_user',default='AI2')
    parser.add_argument('--passwd',default='PASSWD')
    parser.add_argument('--user',default='seol')
    parser.add_argument('--host',default='0.0.0.0')
    parser.add_argument('--web_port',type=int,default=8081)
    parser.add_argument('--min_kpt_conf',type=float,default=0.5)
    args=parser.parse_args()

    print('[INFO] model',args.model)
    global _SMOOTH_ALPHA,_DIST_RATIO
    _SMOOTH_ALPHA=0.5; _DIST_RATIO=0.35

    client=None; raw_sock=None
    if IoTClient:
        try:
            client=IoTClient(args.ip,args.port,args.login_user,args.passwd)
            client.connect()
            print('[INFO] IoTClient connected')
        except Exception as e:
            print('[WARN] IoTClient connect fail',e)
    if client is None:
        try:
            raw_sock=socket.socket(); raw_sock.settimeout(3)
            raw_sock.connect((args.ip,args.port))
            print('[INFO] Raw TCP connected')
        except Exception as e:
            print('[ERR] TCP connect fail',e)

    def send_msg(m):
        if client:
            try: client.send(m+'\n'); return
            except: pass
        if raw_sock:
            try: raw_sock.sendall((m+'\n').encode()); return
            except: pass

    model=YOLO(args.model)
    cap=cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print('[ERR] camera open fail'); return

    while True:
        ok,frame=cap.read()
        if not ok: continue
        res=model.predict(frame,conf=args.conf,verbose=False)
        if not res or res[0].keypoints is None:
            cv2.imshow('Posture',frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue
        idx=get_center_person_index(res,frame.shape[1])
        if idx is None:
            cv2.imshow('Posture',frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue
        pose=res[0].keypoints.data[idx].cpu().numpy().reshape(-1,3)
        def gp(name):
            i=keypoint_index.get(name)
            if i is not None and pose[i][2]>args.min_kpt_conf:
                return int(pose[i][0]),int(pose[i][1])
            return None
        h,w=frame.shape[:2]; diag2=w*w+h*h
        for a,b in skeleton:
            p1=_ema(a,gp(a)); p2=_ema(b,gp(b))
            if p1 and p2:
                dx,dy=p1[0]-p2[0],p1[1]-p2[1]
                if dx*dx+dy*dy<diag2*(_DIST_RATIO**2):
                    cv2.line(frame,p1,p2,SKEL_COLOR,THICK_LINE)
        for n,i in keypoint_index.items():
            if i is not None and pose[i][2]>args.min_kpt_conf:
                pt=_ema(n,(int(pose[i][0]),int(pose[i][1])))
                if pt: cv2.circle(frame,pt,JOINT_RADIUS,JOINT_COLOR,-1)
        y=30
        for label,(a,b,c) in angle_targets.items():
            ang=calculate_angle(_ema(a,gp(a)),_ema(b,gp(b)),_ema(c,gp(c)))
            if ang is not None:
                color=OK_COLOR if ang>30 else WARN_COLOR
                cv2.putText(frame,f'{label}:{int(ang)}', (10,y), cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                y+=30
        _set_latest_frame(frame)
        cv2.imshow('Posture',frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

if __name__=='__main__':
    t=threading.Thread(target=lambda: app.run(host='0.0.0.0',port=8081,threaded=True,use_reloader=False),daemon=True)
    t.start()
    main()
