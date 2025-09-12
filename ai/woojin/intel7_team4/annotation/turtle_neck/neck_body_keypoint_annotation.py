# 실시간 웹캠에서 
# (귀 - 목,팔, 몸통, 하체)을 keypoint로 잡아줌.


if 0:
    
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    # 얼굴(귀 포함) 제외한 관절들 (몸통, 팔, 다리)
    BODY_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    # 몸 선(팔, 다리 포함) 연결
    BODY_PAIRS = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
        (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
        (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
        (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
        (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
        (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        out = frame.copy()
        h, w = out.shape[:2]

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 점 (몸/팔/다리만)
            for idx in BODY_LANDMARKS:
                if lm[idx].visibility > 0.3:
                    cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(out, (cx, cy), 6, (0, 255, 0), -1)

            # 선 (몸/팔/다리만)
            for s, e in BODY_PAIRS:
                if lm[s].visibility > 0.3 and lm[e].visibility > 0.3:
                    sx, sy = int(lm[s].x * w), int(lm[s].y * h)
                    ex, ey = int(lm[e].x * w), int(lm[e].y * h)
                    cv2.line(out, (sx, sy), (ex, ey), (0, 0, 255), 3)

            # 목(어깨 중심) ↔ 귀 선 추가
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
            if l_sh.visibility > 0.3 and r_sh.visibility > 0.3 and l_ear.visibility > 0.3:
                neck_x = (l_sh.x + r_sh.x) / 2
                neck_y = (l_sh.y + r_sh.y) / 2
                neck_c = (int(neck_x * w), int(neck_y * h))
                ear_c = (int(l_ear.x * w), int(l_ear.y * h))
                cv2.line(out, neck_c, ear_c, (255, 0, 255), 4)
                cv2.circle(out, neck_c, 8, (0, 255, 255), -1)
                cv2.circle(out, ear_c, 8, (255, 0, 0), -1)

        cv2.imshow('Pose (Body Only) + Neck-Ear', out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()





















# 실시간 웹캠에서 
# (귀 - 목, 몸통, 하체)을 keypoint로 잡아줌.

if 0:
    
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    # 몸통/다리만 점 표시 (팔, 얼굴, 귀 제외)
    BODY_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    # 몸통/다리 연결 (팔 제외)
    BODY_PAIRS = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
        (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
        (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
        (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        out = frame.copy()
        h, w = out.shape[:2]

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 점 (몸/다리만)
            for idx in BODY_LANDMARKS:
                if lm[idx].visibility > 0.3:
                    cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(out, (cx, cy), 6, (0, 255, 0), -1)

            # 선 (몸/다리만)
            for s, e in BODY_PAIRS:
                if lm[s].visibility > 0.3 and lm[e].visibility > 0.3:
                    sx, sy = int(lm[s].x * w), int(lm[s].y * h)
                    ex, ey = int(lm[e].x * w), int(lm[e].y * h)
                    cv2.line(out, (sx, sy), (ex, ey), (0, 0, 255), 3)

            # 목(어깨 중심) ↔ 귀 선 추가
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
            if l_sh.visibility > 0.3 and r_sh.visibility > 0.3 and l_ear.visibility > 0.3:
                neck_x = (l_sh.x + r_sh.x) / 2
                neck_y = (l_sh.y + r_sh.y) / 2
                neck_c = (int(neck_x * w), int(neck_y * h))
                ear_c = (int(l_ear.x * w), int(l_ear.y * h))
                cv2.line(out, neck_c, ear_c, (255, 0, 255), 4)
                cv2.circle(out, neck_c, 8, (0, 255, 255), -1)
                cv2.circle(out, ear_c, 8, (255, 0, 0), -1)

        cv2.imshow('Pose (Torso/Legs Only) + Neck-Ear', out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    























# img_dir = './sitting_dataset' 데이터 셋 경로를 넣어주면
# 이미지에 관절을 추가해서 하나씩 띄워줌
# s : 저장
# d : 저장 안하고 패스
# esc : 강제 종료

# 거북목 판단 X

if 0 :
    
    import cv2
    import mediapipe as mp
    import os

    img_dir = './dataset/sitting_dataset'
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    mp_pose = mp.solutions.pose
    
    # model_complexity 값을 변경해서 모델의 정확도 수정 가능.
    # Lite: 모바일/임베디드 용, 가장 가벼움 (정확도 낮음) = 0
    # Full: 일반적으로 가장 많이 쓰임 (성능/속도 밸런스) = 1
    # Heavy: 데스크탑/서버에서 더 높은 정확도, 느림 = 2
    pose = mp_pose.Pose(model_complexity=2, 
                        static_image_mode=True, 
                        min_detection_confidence=0.5)

    
    # mp_pose.PoseLandmark.LEFT_SHOULDER.value
    # LANDMARK의 좌표값을 불러옴.
    BODY_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]
    BODY_PAIRS = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
        (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
        (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
        (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    ]
    save_dir = './saved_with_keypoint'
    os.makedirs(save_dir, exist_ok=True)
    winname = 'Pose + Neck-Ear [s:저장 d:넘김 esc:종료]'

    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f'[ERROR] 이미지 읽기 실패: {img_path}')
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        out = img.copy()
        h, w = out.shape[:2]

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            for idx in BODY_LANDMARKS:
                if lm[idx].visibility > 0.3:
                    cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(out, (cx, cy), 6, (0, 255, 0), -1)
            for s, e in BODY_PAIRS:
                if lm[s].visibility > 0.3 and lm[e].visibility > 0.3:
                    sx, sy = int(lm[s].x * w), int(lm[s].y * h)
                    ex, ey = int(lm[e].x * w), int(lm[e].y * h)
                    cv2.line(out, (sx, sy), (ex, ey), (0, 0, 255), 3)
            # 목(어깨 중심) ↔ 귀 선 추가
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
            if l_sh.visibility > 0.3 and r_sh.visibility > 0.3 and l_ear.visibility > 0.3:
                neck_x = (l_sh.x + r_sh.x) / 2
                neck_y = (l_sh.y + r_sh.y) / 2
                neck_c = (int(neck_x * w), int(neck_y * h))
                ear_c = (int(l_ear.x * w), int(l_ear.y * h))
                cv2.line(out, neck_c, ear_c, (255, 0, 255), 4)
                cv2.circle(out, neck_c, 8, (0, 255, 255), -1)
                cv2.circle(out, ear_c, 8, (255, 0, 0), -1)
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            print(left_shoulder.x, left_shoulder.y, left_shoulder.visibility)
        # 윈도우 크기 자동 조절, 창 자동 닫기
        cv2.destroyAllWindows()
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

        disp = out.copy()
        if max(disp.shape[:2]) > 1200:
            scale = 1200.0 / max(disp.shape[:2])
            disp = cv2.resize(disp, (int(disp.shape[1]*scale), int(disp.shape[0]*scale)))

        cv2.imshow(winname, disp)
        print(f"[{fname}]   (s: 저장 / d: 넘김 / esc: 종료)")
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            save_path = os.path.join(save_dir, fname)
            cv2.imwrite(save_path, out)
            print(f"저장됨: {save_path}")
        elif key == ord('d'):
            print("넘김")
            continue

    pose.close()
    cv2.destroyAllWindows()




# 기능
# 이미지를 불러와서
# 관절표시
# 거북목 판단

if 0:
    import cv2
    import mediapipe as mp
    import os
    import math

    # ====== 설정 ======
    img_dir = './dataset/sitting_dataset'   # 이미지 폴더명(경로)
    save_dir = './saved_with_keypoint'
    os.makedirs(save_dir, exist_ok=True)

    angle_threshold = 15  # 각도(도, deg) 기준값 (수정 가능),   거북목 판단 각도
    # ===================

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2, 
                        static_image_mode=True, 
                        min_detection_confidence=0.5)

    BODY_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]
    BODY_PAIRS = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
        (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
        (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
        (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    ]

    winname = 'Pose + Neck-Ear [s:save d:skip esc:exit]'
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f'[ERROR] 이미지 읽기 실패: {img_path}')
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        out = img.copy()
        h, w = out.shape[:2]
        angle_deg = None
        neck_c, ear_c = None, None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 점
            for idx in BODY_LANDMARKS:
                if lm[idx].visibility > 0.3:
                    cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(out, (cx, cy), 6, (0, 255, 0), -1)
            # 선
            for s, e in BODY_PAIRS:
                if lm[s].visibility > 0.3 and lm[e].visibility > 0.3:
                    sx, sy = int(lm[s].x * w), int(lm[s].y * h)
                    ex, ey = int(lm[e].x * w), int(lm[e].y * h)
                    cv2.line(out, (sx, sy), (ex, ey), (0, 0, 255), 3)
            # 목(어깨 중심) ↔ 귀 선, 각도 계산
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
            if l_sh.visibility > 0.3 and r_sh.visibility > 0.3 and l_ear.visibility > 0.3:
                neck_x = (l_sh.x + r_sh.x) / 2
                neck_y = (l_sh.y + r_sh.y) / 2
                neck_c = (int(neck_x * w), int(neck_y * h))
                ear_c = (int(l_ear.x * w), int(l_ear.y * h))
                # 목-귀 벡터
                dx = l_ear.x - neck_x
                dy = l_ear.y - neck_y
                # x축 기준 각도 (수직선 = 0, 앞으로 나갈수록 각도 커짐)
                angle_rad = math.atan2(dx, -dy)
                angle_deg = abs(math.degrees(angle_rad))
                # 판정 및 텍스트
                if angle_deg > angle_threshold:
                    main_text = "Turtle Neck!"
                    color = (0, 0, 255)
                else:
                    main_text = "Good"
                    color = (0, 255, 0)
                cv2.line(out, neck_c, ear_c, (255, 0, 255), 4)
                cv2.circle(out, neck_c, 8, (0, 255, 255), -1)
                cv2.circle(out, ear_c, 8, (255, 0, 0), -1)
                # 텍스트 크기 줄이고 두 줄로
                cv2.putText(out, main_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                if angle_deg is not None:
                    cv2.putText(out, f"angle = {angle_deg:.1f} deg", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # 창 크기 자동조절
        disp = out.copy()
        if max(disp.shape[:2]) > 1200:
            scale = 1200.0 / max(disp.shape[:2])
            disp = cv2.resize(disp, (int(disp.shape[1]*scale), int(disp.shape[0]*scale)))
        cv2.imshow(winname, disp)
        print(f"[{fname}]   (s: save / d: skip / esc: exit)")
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            save_path = os.path.join(save_dir, fname)
            cv2.imwrite(save_path, out)
            print(f"저장됨: {save_path}")
        elif key == ord('d'):
            print("넘김")
            continue

    pose.close()
    cv2.destroyAllWindows()






























"""
fix 가능한 코드 확인 필요
"""

# 기능
# 웹캠에서
# 관절표시
# 거북목 판단

if 0:
    import cv2
    import mediapipe as mp
    import math

    angle_threshold = 15  # 각도(도, deg) 기준값 (수정 가능, 거북목 판정)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2, static_image_mode=False, min_detection_confidence=0.5)

    BODY_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]
    BODY_PAIRS = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
        (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
        (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
        (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    ]

    winname = 'Webcam: Turtle Neck Detection'
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        out = img.copy()

        results = pose.process(img_rgb)
        angle_deg = None
        neck_c, ear_c = None, None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 점
            for idx in BODY_LANDMARKS:
                if lm[idx].visibility > 0.3:
                    cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(out, (cx, cy), 6, (0, 255, 0), -1)
            # 선
            for s, e in BODY_PAIRS:
                if lm[s].visibility > 0.3 and lm[e].visibility > 0.3:
                    sx, sy = int(lm[s].x * w), int(lm[s].y * h)
                    ex, ey = int(lm[e].x * w), int(lm[e].y * h)
                    cv2.line(out, (sx, sy), (ex, ey), (0, 0, 255), 3)
            # 목(어깨 중심) ↔ 귀 선, 각도 계산
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
            if l_sh.visibility > 0.3 and r_sh.visibility > 0.3 and l_ear.visibility > 0.3:
                neck_x = (l_sh.x + r_sh.x) / 2
                neck_y = (l_sh.y + r_sh.y) / 2
                neck_c = (int(neck_x * w), int(neck_y * h))
                ear_c = (int(l_ear.x * w), int(l_ear.y * h))
                # 목-귀 벡터
                dx = l_ear.x - neck_x
                dy = l_ear.y - neck_y
                angle_rad = math.atan2(dx, -dy)
                angle_deg = abs(math.degrees(angle_rad))
                if angle_deg > angle_threshold:
                    main_text = "Turtle Neck!"
                    color = (0, 0, 255)
                else:
                    main_text = "Good"
                    color = (0, 255, 0)
                cv2.line(out, neck_c, ear_c, (255, 0, 255), 4)
                cv2.circle(out, neck_c, 8, (0, 255, 255), -1)
                cv2.circle(out, ear_c, 8, (255, 0, 0), -1)
                # 텍스트
                cv2.putText(out, f"{main_text}   angle={angle_deg:.1f} deg", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        disp = out
        if max(disp.shape[:2]) > 1200:
            scale = 1200.0 / max(disp.shape[:2])
            disp = cv2.resize(disp, (int(disp.shape[1]*scale), int(disp.shape[0]*scale)))
        cv2.imshow(winname, disp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


# 기능
# 웹캠에서
# 관절표시
# 거북목 판단 ( 왼쪽 귀, 오른쪽 귀, 코의 좌표를 잇는 중간지점과 목을 이어서 거북목을 판단)
# 정면과 측면에서는 충분한 성능이 나옴.

if 1:
    import cv2
    import mediapipe as mp
    import time
    import math

    # ====== SETTINGS (changeable) ======
    ANGLE_THRESHOLD = 15.0       # degree: 이 값 초과면 Turtle Neck 판단
    VISIBILITY_THRESH = 0.35     # landmark visibility threshold to consider a point valid
    SMOOTH_ALPHA = 0.2           # EMA alpha for smoothing head centroid (0..1). 0=no update, 1=raw
    CAM_INDEX = 0                # camera index
    # ===================================

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1,
                        static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open camera {CAM_INDEX}")

    prev_time = time.time()
    frame_count = 0
    fps = 0

    # EMA smoothed centroid (in normalized coords). Start None.
    smoothed_head = None

    winname = "Webcam: Turtle Neck Detection (centroid head-point)"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, exiting.")
                break

            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count
                frame_count = 0
                prev_time = now

            img = cv2.flip(frame, 1)  # mirror for natural webcam behavior
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = pose.process(img_rgb)

            used_points = 0
            centroid = None

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Collect candidate head points (nose, left_ear, right_ear)
                candidates = []
                # nose
                n = lm[mp_pose.PoseLandmark.NOSE.value]
                if n.visibility > VISIBILITY_THRESH:
                    candidates.append((n.x, n.y))
                # left ear
                le = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
                if le.visibility > VISIBILITY_THRESH:
                    candidates.append((le.x, le.y))
                # right ear
                re = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
                if re.visibility > VISIBILITY_THRESH:
                    candidates.append((re.x, re.y))

                used_points = len(candidates)

                if used_points > 0:
                    # centroid in normalized coords
                    sx = sum([p[0] for p in candidates]) / used_points
                    sy = sum([p[1] for p in candidates]) / used_points
                    centroid = (sx, sy)

                    # EMA smoothing
                    if smoothed_head is None:
                        smoothed_head = centroid
                    else:
                        smoothed_head = (SMOOTH_ALPHA * centroid[0] + (1 - SMOOTH_ALPHA) * smoothed_head[0],
                                        SMOOTH_ALPHA * centroid[1] + (1 - SMOOTH_ALPHA) * smoothed_head[1])

                    # neck point = midpoint of shoulders (normalized)
                    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    if l_sh.visibility > VISIBILITY_THRESH and r_sh.visibility > VISIBILITY_THRESH:
                        neck_x = (l_sh.x + r_sh.x) / 2.0
                        neck_y = (l_sh.y + r_sh.y) / 2.0
                        neck = (neck_x, neck_y)

                        # vector from neck -> head (use smoothed_head if available)
                        head_x, head_y = smoothed_head
                        vx = head_x - neck_x
                        vy = head_y - neck_y
                        norm = math.hypot(vx, vy)
                        angle_deg = 0.0
                        status = "No Angle"
                        color = (0, 255, 0)

                        if norm > 1e-6:
                            # angle between vector and vertical axis (pointing up)
                            # vertical vector (0,-1). angle = atan2(vx, -vy)
                            angle_rad = math.atan2(vx, -vy)
                            angle_deg = abs(math.degrees(angle_rad))

                            if angle_deg > ANGLE_THRESHOLD:
                                status = "Turtle Neck!"
                                color = (0, 0, 255)
                            else:
                                status = "Good"
                                color = (0, 255, 0)

                            # draw line and points on image (convert normalized to pixels)
                            hx, hy = int(head_x * w), int(head_y * h)
                            nx, ny = int(neck_x * w), int(neck_y * h)
                            cv2.line(img, (nx, ny), (hx, hy), (255, 0, 255), 3)
                            cv2.circle(img, (nx, ny), 6, (0, 255, 255), -1)
                            cv2.circle(img, (hx, hy), 6, (255, 0, 0), -1)

                            # also draw small markers for each used landmark
#                            for (px, py) in candidates:
#                                cx, cy = int(px * w), int(py * h)
#                                cv2.circle(img, (cx, cy), 4, (0, 200, 200), -1)

                            # top-left status text: status + angle + used count (English only)
                            text = f"{status}  angle={angle_deg:.1f}deg  used={used_points}"
                            cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        else:
                            cv2.putText(img, "Head-Neck too close", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
                    else:
                        cv2.putText(img, "Shoulders not visible", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
                else:
                    cv2.putText(img, "Head landmarks not visible", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2)
            else:
                cv2.putText(img, "No pose detected", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2)

            # FPS
            cv2.putText(img, f"FPS: {fps}", (12, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,50), 2)

            cv2.imshow(winname, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()

