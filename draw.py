# 

import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== CONSTANTS ==================
FINGER_TIPS = [4, 8, 12, 16, 20]
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BZIER_STEPS = 30  # Increased for smoother curves
ERASER_THICKNESS = 5
MAX_POINTS_BUFFER = 20  # Keep recent points for smoother drawing
FPS_TARGET = 30
FRAME_TIME = 1000 // FPS_TARGET  # milliseconds

# ================== INIT MODEL ==================
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# ================== CANVAS ==================
canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

# ================== STATE ==================
mode = "draw"
color = (255, 0, 0)
points = []  # for bezier smoothing
erase_prev = (0, 0)
last_drawn_point = None
frame_time = 1

# ================== UI LAYER ==================
ui_layer = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
cv2.rectangle(ui_layer, (0,0), (100,50), (255,0,0), -1)
cv2.rectangle(ui_layer, (100,0), (200,50), (0,255,0), -1)
cv2.rectangle(ui_layer, (200,0), (300,50), (0,0,255), -1)

# ================== FUNCTIONS ==================
def get_finger_states(landmarks):
    fingers = [1 if landmarks[4].x < landmarks[3].x else 0]
    for i in range(1, 5):
        tip_idx = FINGER_TIPS[i]
        fingers.append(1 if landmarks[tip_idx].y < landmarks[tip_idx-2].y else 0)
def get_finger_states(landmarks):
    fingers = [1 if landmarks[4].x < landmarks[3].x else 0]
    for i in range(1, 5):
        tip_idx = FINGER_TIPS[i]
        fingers.append(1 if landmarks[tip_idx].y < landmarks[tip_idx-2].y else 0)
    return fingers

def get_distance(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return int(np.hypot(x2 - x1, y2 - y1))

def calculate_velocity(p1, p2):
    """Calculate velocity between two points for stroke thickness"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.hypot(dx, dy)

def get_variable_thickness(velocity):
    """Adaptive thickness based on velocity - slower = thicker"""
    base_thickness = 4
    max_thickness = 8
    return max(base_thickness, min(max_thickness, int(base_thickness + (10 - velocity) / 3)))

# ======== BEZIER DRAW FUNCTION ========
def draw_bezier(canvas, p0, p1, p2, color):
    """Draw smooth quadratic Bezier curve with velocity-based thickness"""
    prev_point = p0
    t_values = np.linspace(0, 1, BZIER_STEPS)
    velocity = calculate_velocity(p0, p2)
    thickness = get_variable_thickness(velocity)
    
    for t in t_values:
        one_minus_t = 1 - t
        x = int(one_minus_t**2 * p0[0] + 2*one_minus_t*t * p1[0] + t**2 * p2[0])
        y = int(one_minus_t**2 * p0[1] + 2*one_minus_t*t * p1[1] + t**2 * p2[1])
        cv2.line(canvas, prev_point, (x, y), color, thickness, cv2.LINE_AA)
        prev_point = (x, y)

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

frame_clock = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.add(frame, ui_layer)

    # ================== HAND DETECTION ==================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result and result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        index_x = int(landmarks[8].x * FRAME_WIDTH)
        index_y = int(landmarks[8].y * FRAME_HEIGHT)

        fingers = get_finger_states(landmarks)
        total = sum(fingers)

        # ================== MODE SWITCH ==================
        if fingers[1] == 1 and total == 1:
            mode = "draw"
        elif fingers[1] == 1 and fingers[2] == 1 and total == 2:
            mode = "color"
        elif total == 5:
            mode = "erase"

        # ================== DRAW (BEZIER) ==================
        if mode == "draw":
            current_point = (index_x, index_y)
            points.append(current_point)
            
            # Keep buffer manageable (circular buffer effect)
            if len(points) > MAX_POINTS_BUFFER:
                points.pop(0)

            # Draw smooth curves with sufficient points
            if len(points) >= 4:
                # Use better control point - average of surrounding points
                p0 = points[-4]
                p1_start = points[-3]
                p1_end = points[-2]
                p2 = points[-1]
                
                control = (
                    (p1_start[0] + p1_end[0]) // 2,
                    (p1_start[1] + p1_end[1]) // 2
                )
                draw_bezier(canvas, p0, control, p2, color)

            erase_prev = (0, 0)

        # ================== COLOR ==================
        elif mode == "color":
            points = []
            erase_prev = (0, 0)

            if index_y < 50:
                if 0 < index_x < 100:
                    color = (255, 0, 0)
                elif 100 < index_x < 200:
                    color = (0, 255, 0)
                elif 200 < index_x < 300:
                    color = (0, 0, 255)

        # ================== ERASE ==================
        elif mode == "erase":
            points = []

            prev_x, prev_y = erase_prev

            dist = get_distance(landmarks[8], landmarks[12], FRAME_WIDTH, FRAME_HEIGHT)
            eraser_size = max(10, min(50, dist // 2))
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = index_x, index_y

            cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0,0,0), eraser_size)

            erase_prev = (index_x, index_y)

            cv2.circle(frame, (index_x, index_y), eraser_size, (255,255,255), 2)

    else:
        points = []
        erase_prev = (0, 0)

    # ================== MERGE ==================
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    # ================== UI ==================
    cv2.putText(frame, f"Mode: {mode}", (10,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.circle(frame, (600,30), 20, color, -1)

    # ================== SHOW ==================
    cv2.imshow("Bezier Air Drawing", frame)

    # Frame rate limiting
    elapsed = (time.time() - frame_clock) * 1000
    wait_time = max(1, int(FRAME_TIME - elapsed))
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break
    frame_clock = time.time()

cap.release()
cv2.destroyAllWindows()