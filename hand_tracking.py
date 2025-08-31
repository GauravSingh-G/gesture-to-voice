import cv2
import mediapipe as mp
import time
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

prev_time = None
prev_pos = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Tip of the index finger
            x = hand_landmarks.landmark[8].x * frame.shape[1]
            y = hand_landmarks.landmark[8].y * frame.shape[0]
            curr_pos = (x, y)
            curr_time = time.time()

            if prev_pos is not None and prev_time is not None:
                dt = curr_time - prev_time
                if dt > 0:  # Prevent division by zero
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    distance = math.sqrt(dx**2 + dy**2)
                    speed = distance / dt
                    cv2.putText(frame, f'Speed: {int(speed)} px/s', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            prev_pos = curr_pos
            prev_time = curr_time

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
