import cv2
import mediapipe as mp
import numpy as np
import pyautogui  # Dùng để lấy kích thước màn hình
from pynput.mouse import Controller
from collections import deque

# 1. Cấu hình ban đầu
mouse = Controller()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Lấy kích thước màn hình máy tính
w_scr, h_scr = pyautogui.size()
cam_w, cam_h = 640, 480  # Cấu hình camera (thường là 640x480)

cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

# 2. Tạo hàng đợi để lưu lịch sử tọa độ (Smoothing)
# maxlen=5 nghĩa là chỉ nhớ 5 vị trí gần nhất
lm_history = deque(maxlen=7)


def get_smoothed_coords(new_x, new_y):
    """Hàm tính trung bình tọa độ dựa trên lịch sử"""
    lm_history.append((new_x, new_y))

    # Tính trung bình các tọa độ trong hàng đợi
    avg_x = sum([pt[0] for pt in lm_history]) / len(lm_history)
    avg_y = sum([pt[1] for pt in lm_history]) / len(lm_history)
    return int(avg_x), int(avg_y)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lật ảnh để không bị ngược tay
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Vẽ khung hình chữ nhật ảo để giới hạn vùng điều khiển (giúp chuột ra mép màn hình dễ hơn)
    frame_redux = 100  # Giảm biên mỗi bên 100 pixel
    cv2.rectangle(frame, (frame_redux, frame_redux),
                  (cam_w - frame_redux, cam_h - frame_redux), (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ các khớp tay
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            lm_target = hand_landmarks.landmark[9]

            # Tọa độ thô trên webcam
            cx, cy = int(lm_target.x * cam_w), int(lm_target.y * cam_h)

            # 3. Chuyển đổi tọa độ (Mapping) từ Webcam sang Màn hình
            # Dùng np.interp để map giá trị từ khoảng này sang khoảng kia
            screen_x = np.interp(
                cx, (frame_redux, cam_w - frame_redux), (0, w_scr))
            screen_y = np.interp(
                cy, (frame_redux, cam_h - frame_redux), (0, h_scr))

            # 4. Làm mượt (Smoothing)
            sm_x, sm_y = get_smoothed_coords(screen_x, screen_y)

            # Di chuyển chuột
            # Dùng try-except để tránh lỗi khi tọa độ vượt quá màn hình
            try:
                mouse.position = (sm_x, sm_y)
            except:
                pass

            # Vẽ vòng tròn tại điểm điều khiển để debug
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
