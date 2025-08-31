import cv2
import mediapipe as mp
import csv
import os

# Configuration
LABEL = "A"  # Change this to B, C, ... when collecting other signs
DATA_FILE = f"{LABEL}_data.csv"

# Init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Prepare CSV
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
        writer.writerow(header)

print(f"📸 Starting data collection for letter: {LABEL}")
print("👉 Press 's' to save a frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # If 's' key is pressed, save landmarks
            key = cv2.waitKey(1)
            if key == ord('s'):
                landmarks = hand_landmarks.landmark
                x = [lm.x for lm in landmarks]
                y = [lm.y for lm in landmarks]
                z = [lm.z for lm in landmarks]
                row = x + y + z + [LABEL]

                with open(DATA_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print(f"✅ Frame saved for '{LABEL}'")

            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("👋 Finished data collection.")
                exit()

    cv2.putText(frame, f"Letter: {LABEL} | Press 's' to save, 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Collecting Hand Gesture", frame)
