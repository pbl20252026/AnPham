import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui
from pynput.mouse import Controller, Button

# ========== CONFIG ==========
# Ngưỡng kích hoạt chụm ngón tay
nguong_cham_left = 30
nguong_cham_right = 30

# Cấu hình chuột & Màn hình
smooth_factor = 5       # Độ mượt
frame_reduction = 100   # Bo viền camera
screen_w, screen_h = pyautogui.size()

# Cấu hình Logic nâng cao
# Giới hạn thời gian click trái (để phân biệt drag)
max_click_duration_left = 0.3
# (QUAN TRỌNG) Biên độ rung: Di chuyển quá 20px mới tính là Hover
hover_threshold = 20

# ========== KHỞI TẠO ==========
mouse = Controller()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Biến toàn cục lưu trạng thái
mouse_down_left = False
start_time_left = 0

mouse_down_right = False
start_pos_right = (0, 0)  # Lưu tọa độ lúc bắt đầu chụm phải
is_right_hovering = False  # Cờ đánh dấu xem có phải đang hover không

# Biến làm mượt tọa độ
plocX, plocY = 0, 0
clocX, clocY = 0, 0


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ========== LOGIC XỬ LÝ ==========


def handle_gestures(x4, y4, x8, y8, x12, y12):
    global mouse_down_left, start_time_left
    global mouse_down_right, start_pos_right, is_right_hovering

    # 1. TÍNH KHOẢNG CÁCH
    dist_left = distance((x4, y4), (x8, y8))   # Cái - Trỏ
    dist_right = distance((x4, y4), (x12, y12))  # Cái - Giữa

    # ========================================================
    # LOGIC LEFT CLICK (DRAG & DROP)
    # ========================================================
    if dist_left < nguong_cham_left:
        if not mouse_down_left:
            mouse_down_left = True
            start_time_left = time.time()
            mouse.press(Button.left)  # Giữ chuột trái để Drag
            print("Left: DOWN (Drag Start)")
    else:
        if mouse_down_left:
            mouse.release(Button.left)  # Nhả chuột (Drop)
            mouse_down_left = False

            # Logic click nhanh
            if time.time() - start_time_left <= max_click_duration_left:
                mouse.click(Button.left)
                print("Left: Click (Fast Tap)")
            else:
                print("Left: Drop Finished")

    # ========================================================
    # LOGIC RIGHT CLICK (HOVER vs CLICK)
    # ========================================================
    if dist_right < nguong_cham_right:
        # --- LÚC BẮT ĐẦU CHỤM PHẢI ---
        if not mouse_down_right:
            mouse_down_right = True
            start_pos_right = (x4, y4)  # Lưu vị trí gốc
            is_right_hovering = False  # Reset trạng thái hover
            print("Right: Gesture Start (Waiting for move...)")

        # --- TRONG LÚC ĐANG GIỮ ---
        else:
            # Kiểm tra xem đã di chuyển vượt quá "biên độ rung" chưa
            dist_move = distance((x4, y4), start_pos_right)

            if dist_move > hover_threshold:
                if not is_right_hovering:
                    is_right_hovering = True
                    print(
                        f"Right: HOVER MODE ACTIVATED (Moved {int(dist_move)}px)")

                # Lưu ý: Ở chế độ này, ta KHÔNG press chuột phải
                # Ta chỉ dùng gesture này để di chuyển con trỏ (xem phần map tọa độ dưới)

    else:
        # --- LÚC NHẢ TAY PHẢI ---
        if mouse_down_right:
            mouse_down_right = False

            # Chỉ click nếu KHÔNG di chuyển nhiều (Hovering = False)
            if not is_right_hovering:
                mouse.click(Button.right)
                print("Right: CLICK EXECUTED")
            else:
                print("Right: Hover Finished (No Click)")


# ========== CHƯƠNG TRÌNH CHÍNH ==========
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Lấy tọa độ pixel
            lm4 = hand_landmarks.landmark[4]  # Cái
            lm8 = hand_landmarks.landmark[8]  # Trỏ
            lm12 = hand_landmarks.landmark[12]  # Giữa

            x4, y4 = int(lm4.x * w), int(lm4.y * h)
            x8, y8 = int(lm8.x * w), int(lm8.y * h)
            x12, y12 = int(lm12.x * w), int(lm12.y * h)

            # Gọi hàm xử lý logic cử chỉ
            handle_gestures(x4, y4, x8, y8, x12, y12)

            # ============================================================
            # ĐIỀU HƯỚNG TỌA ĐỘ CHUỘT
            # ============================================================
            # Logic:
            # 1. Nếu đang Drag Trái -> Dùng Ngón Cái
            # 2. Nếu đang Giữ Phải (Hover/Click) -> Dùng Ngón Cái
            # 3. Còn lại -> Dùng Ngón Trỏ

            if mouse_down_left or mouse_down_right:
                target_x, target_y = x4, y4
                color_point = (0, 255, 0) if mouse_down_left else (
                    0, 255, 255)  # Lục: Drag, Vàng: Right Hover
            else:
                target_x, target_y = x8, y8
                color_point = (255, 0, 255)  # Tím: Normal Hover

            # Mapping tọa độ
            screen_x = np.interp(
                target_x, (frame_reduction, w - frame_reduction), (0, screen_w))
            screen_y = np.interp(
                target_y, (frame_reduction, h - frame_reduction), (0, screen_h))

            # Smoothing
            clocX = plocX + (screen_x - plocX) / smooth_factor
            clocY = plocY + (screen_y - plocY) / smooth_factor

            # Di chuyển chuột
            mouse.position = (clocX, clocY)
            plocX, plocY = clocX, clocY

            # Vẽ Visual
            cv2.circle(frame, (target_x, target_y),
                       15, color_point, cv2.FILLED)

            # Vẽ vòng tròn biên độ rung để debug (khi đang giữ chuột phải)
            if mouse_down_right and not is_right_hovering:
                cv2.circle(frame, start_pos_right,
                           hover_threshold, (255, 255, 255), 1)

            cv2.rectangle(frame, (frame_reduction, frame_reduction),
                          (w - frame_reduction, h - frame_reduction), (255, 0, 0), 2)
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Smart Hover & Drag", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
