import cv2
import mediapipe as mp
import csv
import os

# ===== 1. GET GESTURE LABEL FROM USER =====
LABEL = input("🔤 Enter gesture label (e.g., A-Z, END, ERASE, SELECT_1, SELECT_2, SELECT_3): ").strip().upper()

if not LABEL.replace("_", "").isalnum() or len(LABEL) < 1:
    print("❌ Invalid input. Use labels like 'A', 'B', 'END', 'ERASE', 'SELECT_1', 'SELECT_2'.")
    exit()

DATA_FILE = f"{LABEL}_data.csv"

# ===== 2. INIT MEDIAPIPE & CAMERA =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera could not be opened. Check your webcam or close other apps using it.")
    exit()

# ===== 3. CREATE CSV IF NOT EXISTS =====
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
        writer.writerow(header)

print(f"\n📸 Starting data collection for gesture '{LABEL}'")
print("👉 Press 's' to save frame, 'q' or 'Esc' to quit.\n")

# ===== 4. MAIN LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not grab frame from camera.")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raw = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            base_x, base_y, base_z = raw[0]
            normalized = [(x - base_x, y - base_y, z - base_z) for x, y, z in raw]
            row = [val for pt in normalized for val in pt] + [LABEL]

            key = cv2.waitKey(1)
            if key == ord('s'):
                with open(DATA_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"✅ Saved frame for '{LABEL}'")

            elif key == ord('q'):
                print("👋 Quitting...")
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # Display status
    cv2.putText(frame, f"Gesture: {LABEL} | Press 's' to save, 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("✋ Collecting Hand Gesture", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc key
        break

# ===== 5. RELEASE CAMERA =====
cap.release()
cv2.destroyAllWindows()
