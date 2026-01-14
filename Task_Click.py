import cv2
import mediapipe as mp
import math
import time


# Cấu hình ngưỡng và thời gian

# Ngưỡng khoảng cách cho click trái (nhạy hơn để bắt double click)
nguong_left = 23
nguong_right = 30

# Thời gian tối đa giữa 2 lần click để được xem là double click
time_doubleclick = 0.25
max_press_time = 0.35
hold_timeout = 3.0
none_debounce = 0.3


# Các trạng thái của máy trạng thái hữu hạn (FSM)

state_idle = "idle"                 # Không có thao tác
state_pressing = "pressing"         # Đang nhấn
state_wait_double = "wait_double"   # Chờ click lần 2
state_reset_lock = "reset_lock"     # Khóa sau reset

state = state_idle                 # Trạng thái hiện tại

active_button = None               # Nút đang được nhấn (left / right)
# Thời điểm click lần đầu (dùng cho double click)
last_click_time = 0
press_time = 0                     # Thời điểm bắt đầu nhấn

prev_touching = None               # Trạng thái touching của frame trước
last_output = None                 # Giá trị output đã in lần gần nhất
last_none_time = 0                 # Thời điểm in None gần nhất

# Cho phép in Waiting hay chưa (sau reset phải thoát ngưỡng)
allow_waiting = True


# Khởi tạo Mediapipe và camera

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


# Hàm tính khoảng cách Euclid giữa hai điểm

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# Hàm phát hiện ngón tay đang chạm để xác định left / right click

def detect_touch(landmarks, w, h):
    thumb = landmarks[4]           # Ngón cái
    index_finger = landmarks[8]    # Ngón trỏ
    middle = landmarks[12]         # Ngón giữa

    # Quy đổi tọa độ chuẩn hóa sang pixel
    tx, ty = int(thumb.x * w), int(thumb.y * h)
    ix, iy = int(index_finger.x * w), int(index_finger.y * h)
    mx, my = int(middle.x * w), int(middle.y * h)

    # Tính khoảng cách giữa ngón cái và các ngón khác
    d_left = distance((tx, ty), (ix, iy))
    d_right = distance((tx, ty), (mx, my))

    # Nếu nhỏ hơn ngưỡng thì xem là chạm
    if d_left < nguong_left:
        return "left"
    if d_right < nguong_right:
        return "right"

    # Có tay nhưng không chạm
    return None


# Hàm in trạng thái ra console có chống spam

def emit(msg):
    global last_output, last_none_time

    now = time.time()

    # Chống spam khi in None liên tục
    if msg.lower() == "none":
        if now - last_none_time < none_debounce:
            return
        last_none_time = now

    # Chuẩn hóa format hiển thị
    format_map = {
        "waiting": "Waiting",
        "none": "None",
        "reset": "Reset",
    }

    output = format_map.get(msg, msg)

    # Chỉ in khi giá trị thay đổi
    if output != last_output:
        print(output)
        last_output = output


# Hàm reset toàn bộ trạng thái FSM

def reset_fsm(lock_waiting=True):
    global state, active_button, press_time, allow_waiting

    state = state_idle
    active_button = None
    press_time = 0

    # Nếu reset do timeout thì khóa waiting cho tới khi tay thoát hẳn
    if lock_waiting:
        allow_waiting = False
    else:
        allow_waiting = True


# Vòng lặp chính xử lý camera và logic

while True:
    success, frame = cap.read()
    if not success:
        continue

    # Lật ảnh cho giống gương
    frame = cv2.flip(frame, 1)

    # Chuyển ảnh sang RGB cho Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape
    now = time.time()

    touching = None
    hand_detected = False

    # Nếu phát hiện có tay
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            touching = detect_touch(hand.landmark, w, h)

    # Phát hiện cạnh nhấn và nhả (edge detection)
    just_pressed = touching and not prev_touching
    just_released = not touching and prev_touching
    prev_touching = touching

    # Trạng thái idle: chưa có thao tác

    if state == state_idle:

        # Có thao tác chạm hợp lệ
        if just_pressed and touching in ("left", "right"):
            state = state_pressing
            active_button = touching
            press_time = now
            emit(f"mouseDown {active_button.capitalize()}")

        # Có tay nhưng không có thao tác hợp lệ
        elif hand_detected and touching is None and allow_waiting:
            emit("waiting")

        # Không có tay trong camera
        elif not hand_detected:
            emit("none")

    # Trạng thái pressing: đang nhấn

    elif state == state_pressing:

        # Nếu giữ quá lâu mà chưa nhả thì reset
        if now - press_time > hold_timeout:
            emit("reset")
            state = state_reset_lock
            allow_waiting = False

        # Khi nhả tay
        elif just_released:
            press_duration = now - press_time

            # Click phải hoặc giữ lâu → click đơn
            if active_button == "right" or press_duration > max_press_time:
                emit(
                    "mouseUp rightClick" if active_button == "right"
                    else "mouseUp leftClick"
                )
                reset_fsm(lock_waiting=False)
                emit("waiting")

            # Click trái ngắn → chuyển sang chờ double click
            else:
                state = state_wait_double
                last_click_time = now

    # Trạng thái chờ double click

    elif state == state_wait_double:

        # Nếu click lần 2 trong thời gian cho phép
        if just_pressed and touching == "left":
            if now - last_click_time <= time_doubleclick:
                emit("mouseUp doubleClick")
                reset_fsm(lock_waiting=False)
                emit("waiting")

        # Nếu hết thời gian chờ thì coi là click đơn
        elif now - last_click_time > time_doubleclick:
            emit("mouseUp leftClick")
            reset_fsm(lock_waiting=False)
            emit("waiting")

    # Trạng thái khóa sau reset

    elif state == state_reset_lock:

        # Chỉ khi tay thoát hoàn toàn khỏi ngưỡng mới cho phép hoạt động lại
        if touching is None:
            allow_waiting = True
            reset_fsm(lock_waiting=False)

    # Hiển thị camera
    cv2.imshow("Click", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
