import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import platform

# Optional: For cross-platform brightness control
try:
    if platform.system() == "Windows" or platform.system() == "Linux":
        import screen_brightness_control as sbc
    elif platform.system() == "Darwin":
        import brightness
    else:
        sbc = None
        brightness = None
except ImportError:
    sbc = None
    brightness = None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

screen_w, screen_h = pyautogui.size()
prev_loc_x, prev_loc_y = 0, 0
smoothening = 0.20000  # Smoothing factor for EMA (higher = more smoothing, e.g., 0.9)

last_click_time = 0
click_delay = 0.15
last_vol_time = 0
vol_delay = 0.15
last_brightness_time = 0
brightness_delay = 0.3

# ========== BRIGHTNESS CONTROL ==========
def brightness_control(lm, frame_shape, now, last_brightness_time, brightness_delay, prev_brightness):
    # Pinch detection for brightness control (thumb tip and index tip)
    h, w, _ = frame_shape
    x1 = int(lm.landmark[4].x * w)
    y1 = int(lm.landmark[4].y * h)
    x2 = int(lm.landmark[8].x * w)
    y2 = int(lm.landmark[8].y * h)
    
    # Calculate distance between thumb and index finger
    length = np.hypot(x2 - x1, y2 - y1)
    
    # Adjusted these values to ensure full 0-100% range
    min_length, max_length = 15, 306  # Increased max_length to ensure full range
    
    # Map the distance to brightness percentage
    brightness_val = np.interp(length, [min_length, max_length], [0, 104])
    
    # Smoothing to avoid abrupt changes
    smooth_brightness = (0.8 * prev_brightness) + (0.2 * brightness_val)
    
    if (now - last_brightness_time) > brightness_delay:
        try:
            # Ensure value is between 0 and 100
            value = max(0, min(100, int(smooth_brightness)))
            
            if platform.system() == "Windows" or platform.system() == "Linux":
                if sbc:
                    sbc.set_brightness(value)
                    print(f"Setting brightness to {value}%")  # Debug output
            elif platform.system() == "Darwin":
                if brightness:
                    brightness.set_brightness(float(value) / 100)
        except Exception as e:
            print("Brightness control failed:", e)
        last_brightness_time = now
    
    return int(smooth_brightness), last_brightness_time, (x1, y1, x2, y2)

# ========== VOLUME UP/DOWN CONTROL ==========
def volume_control(fingers, now, last_vol_time, vol_delay, prev_vol_action):
    # prev_vol_action: "up", "down", or None
    action = None
    if fingers == [1,0,0,0,0] and (now - last_vol_time) > vol_delay:
        pyautogui.press("volumeup")
        action = "up"
        last_vol_time = now
    elif fingers == [0,0,0,0,0] and (now - last_vol_time) > vol_delay:
        pyautogui.press("volumedown")
        action = "down"
        last_vol_time = now
    else:
        action = prev_vol_action
    return action, last_vol_time

# ========== MOUSE MOVEMENT ==========
def mouse_move(lm, frame_shape, prev_loc_x, prev_loc_y, smoothening, screen_w, screen_h):
    h, w, _ = frame_shape
    x = int(lm.landmark[8].x * w)
    y = int(lm.landmark[8].y * h)
    screen_x = np.interp(x, (0, w), (0, screen_w))
    screen_y = np.interp(y, (0, h), (0, screen_h))
    cur_loc_x = (1 - smoothening) * prev_loc_x + smoothening * screen_x
    cur_loc_y = (1 - smoothening) * prev_loc_y + smoothening * screen_y
    if np.hypot(cur_loc_x - prev_loc_x, cur_loc_y - prev_loc_y) > 1:
        pyautogui.moveTo(cur_loc_x, cur_loc_y, duration=0.01)
    return cur_loc_x, cur_loc_y

# Helper for finger state
def fingers_up(landmarks, frame_shape):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    if landmarks.landmark[4].x < landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for i in range(1, 5):
        if landmarks.landmark[tips[i]].y < landmarks.landmark[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Helper for brightness gesture
def is_brightness_control(fingers):
    return fingers[0]==1 and fingers[1]==1 and sum(fingers[2:])==0

current_brightness = 50  # Start at mid value
current_vol_action = None

cv2.namedWindow("HandSense - Gesture Control", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    gesture = "NO ACTION"
    now = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks
            fingers = fingers_up(lm, frame.shape)

            # Mouse Movement (Index finger only up)
            if fingers == [0,1,0,0,0]:
                prev_loc_x, prev_loc_y = mouse_move(lm, frame.shape, prev_loc_x, prev_loc_y, smoothening, screen_w, screen_h)
                gesture = "CURSOR MOVE"

            # Volume Up/Down (Thumb up OR fist)
            elif fingers == [1,0,0,0,0] or fingers == [0,0,0,0,0]:
                current_vol_action, last_vol_time = volume_control(fingers, now, last_vol_time, vol_delay, current_vol_action)
                gesture = f"VOLUME {'UP' if current_vol_action == 'up' else 'DOWN'}"

            # Brightness Control (Thumb and index up)
            elif is_brightness_control(fingers):
                current_brightness, last_brightness_time, (x1, y1, x2, y2) = brightness_control(
                    lm, frame.shape, now, last_brightness_time, brightness_delay, current_brightness)
                gesture = f"BRIGHTNESS: {int(current_brightness)}%"
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.circle(frame, (x1, y1), 8, (0,255,255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 8, (0,255,255), cv2.FILLED)

            cv2.putText(frame, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            break  # Only process first hand

    cv2.imshow("HandSense - Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()