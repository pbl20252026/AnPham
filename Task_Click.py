import cv2
import mediapipe as mp
import math
import time


# CẤU HÌNH THAM SỐ NHẬN DẠNG
nguong_cham = 35          # Ngưỡng khoảng cách (pixel) để coi là "chạm"
delay = 0.1             # Thời gian chờ giữa 2 lần click (giây)

last_left_time = 0            # Lưu thời điểm click trái gần nhất
last_right_time = 0           # Lưu thời điểm click phải gần nhất


# KHỞI TẠO MEDIAPIPE HANDS

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,    # Chế độ ảnh động hay video (False)
    max_num_hands=1,              # Chỉ detect tối đa 1 bàn tay
    min_detection_confidence=0.7,  # Độ tin cậy khi phát hiện tay
    min_tracking_confidence=0.7   # Độ tin cậy khi theo dõi landmark
)

mp_draw = mp.solutions.drawing_utils   # Dùng để vẽ khung xương bàn tay


# KHỞI TẠO CAMERA WEBCAM

cap = cv2.VideoCapture(0)     # 0 = webcam mặc định


# HÀM TÍNH KHOẢNG CÁCH 2 ĐIỂM

def distance(p1, p2):
    # Công thức khoảng cách Euclid giữa 2 điểm (x1,y1) và (x2,y2)
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# HÀM PHÁT HIỆN LEFT CLICK (NGÓN CÁI + NGÓN TRỎ)
def left_click(landmarks, w, h):
    global last_left_time

    # Lấy landmark đầu ngón cái (id = 4)
    thumb_tip = landmarks[4]

    # Lấy landmark đầu ngón trỏ (id = 8)
    index_finger_tip = landmarks[8]

    # Chuyển tọa độ chuẩn hóa (0–1) sang pixel thật
    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
    ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

    # Tính khoảng cách giữa 2 đầu ngón
    dist = distance((tx, ty), (ix, iy))

    # Lấy thời gian hiện tại
    now = time.time()

    # Nếu khoảng cách nhỏ hơn ngưỡng và đã qua thời gian delay
    if dist < nguong_cham and now - last_left_time > delay:
        last_left_time = now
        return "LEFT_CLICK"

    return None


# HÀM PHÁT HIỆN RIGHT CLICK (NGÓN CÁI + NGÓN GIỮA)
def right_click(landmarks, w, h):
    global last_right_time

    # Lấy landmark đầu ngón cái (id = 4)
    thumb_tip = landmarks[4]

    # Lấy landmark đầu ngón giữa (id = 12)
    middle_tip = landmarks[12]

    # Chuyển tọa độ chuẩn hóa sang pixel
    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
    mx, my = int(middle_tip.x * w), int(middle_tip.y * h)

    # Tính khoảng cách giữa 2 đầu ngón
    dist = distance((tx, ty), (mx, my))

    # Lấy thời gian hiện tại
    now = time.time()

    # Điều kiện phát hiện click phải
    if dist < nguong_cham and now - last_right_time > delay:
        last_right_time = now
        return "RIGHT_CLICK"

    return None


# VÒNG LẶP XỬ LÝ CAMERA
while True:

    # Đọc 1 frame từ webcam
    success, frame = cap.read()
    if not success:
        continue

    # Lật ảnh để giống gương
    frame = cv2.flip(frame, 1)

    # Chuyển BGR -> RGB cho MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe xử lý và tìm landmark bàn tay
    results = hands.process(rgb)

    # Lấy kích thước ảnh
    h, w, _ = frame.shape

    # Nếu phát hiện có bàn tay
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            # Vẽ skeleton bàn tay lên màn hình
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Gọi hàm kiểm tra left click
            label_left = left_click(hand.landmark, w, h)

            # Gọi hàm kiểm tra right click
            label_right = right_click(hand.landmark, w, h)

            # In kết quả ra terminal để test
            if label_left:
                print(label_left)

            if label_right:
                print(label_right)

    # Hiển thị cửa sổ camera
    cv2.imshow("Gesture Detection Only", frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break


# GIẢI PHÓNG TÀI NGUYÊN

cap.release()
cv2.destroyAllWindows()
