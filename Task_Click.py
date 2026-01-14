import cv2
import mediapipe as mp
import math
import time


# ================= config =================

# Ngưỡng khoảng cách cho click trái (nhạy hơn để bắt double click)
nguong_left = 20
nguong_right = 30

# Thời gian tối đa giữa 2 lần click để được xem là double click
time_doubleclick = 0.25
max_press_time = 0.35
hold_timeout = 3.0
none_debounce = 0.3


# ================= fsm state =================

state_idle = "idle"
state_pressing = "pressing"
state_wait_double = "wait_double"
state_reset_lock = "reset_lock"

state = state_idle

active_button = None
last_click_time = 0
press_time = 0

prev_touching = None
last_output = None
last_none_time = 0

# cho phép emit waiting hay chưa (sau reset phải thoát ngưỡng)
allow_waiting = True


# ================= mediapipe =================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


# ================= utils =================

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def detect_touch(landmarks, w, h):
    thumb = landmarks[4]
    index_finger = landmarks[8]
    middle = landmarks[12]

    tx, ty = int(thumb.x * w), int(thumb.y * h)
    ix, iy = int(index_finger.x * w), int(index_finger.y * h)
    mx, my = int(middle.x * w), int(middle.y * h)

    d_left = distance((tx, ty), (ix, iy))
    d_right = distance((tx, ty), (mx, my))

    if d_left < nguong_left:
        return "left"
    if d_right < nguong_right:
        return "right"
    return None


def emit(msg):
    global last_output, last_none_time

    now = time.time()

    # debounce cho none
    if msg.lower() == "none":
        if now - last_none_time < none_debounce:
            return
        last_none_time = now

    # format output
    format_map = {
        "waiting": "Waiting",
        "none": "None",
        "reset": "Reset",
    }

    output = format_map.get(msg, msg)

    if output != last_output:
        print(output)
        last_output = output


def reset_fsm(lock_waiting=True):
    global state, active_button, press_time, allow_waiting
    state = state_idle
    active_button = None
    press_time = 0

    # reset do timeout → khóa waiting cho tới khi thoát ngưỡng
    if lock_waiting:
        allow_waiting = False
    else:
        allow_waiting = True


# ================= main loop =================

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape
    now = time.time()

    touching = None
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            touching = detect_touch(hand.landmark, w, h)

    # -------- edge detect --------
    just_pressed = touching and not prev_touching
    just_released = not touching and prev_touching
    prev_touching = touching

    # ================= fsm =================

    # -------- idle --------
    if state == state_idle:

        # có thao tác chạm hợp lệ
        if just_pressed and touching in ("left", "right"):
            state = state_pressing
            active_button = touching
            press_time = now
            emit(f"mouseDown {active_button.capitalize()}")

        # có tay nhưng không có thao tác hợp lệ
        elif hand_detected and touching is None and allow_waiting:
            emit("waiting")

        # không có tay trong camera
        elif not hand_detected:
            emit("none")

    # -------- pressing --------
    elif state == state_pressing:

        # giữ quá lâu → reset
        if now - press_time > hold_timeout:
            emit("reset")
            state = state_reset_lock
            allow_waiting = False

        # nhả tay
        elif just_released:
            press_duration = now - press_time

            # right hoặc giữ lâu → single click
            if active_button == "right" or press_duration > max_press_time:
                emit(
                    "mouseUp rightClick" if active_button == "right"
                    else "mouseUp leftClick"
                )
                reset_fsm(lock_waiting=False)
                emit("waiting")

            # left ngắn → chờ double click
            else:
                state = state_wait_double
                last_click_time = now

    # -------- wait double --------
    elif state == state_wait_double:

        # click lần 2 (không cần mouseUp)
        if just_pressed and touching == "left":
            if now - last_click_time <= time_doubleclick:
                emit("mouseUp doubleClick")
                reset_fsm(lock_waiting=False)
                emit("waiting")

        # hết thời gian → single click
        elif now - last_click_time > time_doubleclick:
            emit("mouseUp leftClick")
            reset_fsm(lock_waiting=False)
            emit("waiting")

    # -------- reset lock --------
    elif state == state_reset_lock:

        # chỉ khi tay thoát hoàn toàn khỏi ngưỡng
        if touching is None:
            allow_waiting = True
            reset_fsm(lock_waiting=False)

    # ================= display =================

    cv2.imshow("Click", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
