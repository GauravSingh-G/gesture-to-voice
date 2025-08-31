import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from gtts import gTTS
import tempfile
import platform

# -------- SPEAK FUNCTION (Cross-platform safe) --------
def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)

        if platform.system() == "Windows":
            os.system(f'start /min wmplayer /play /close "{temp_file.name}"')
        elif platform.system() == "Darwin":
            os.system(f'afplay "{temp_file.name}"')
        else:
            os.system(f'mpg123 "{temp_file.name}"')
    except Exception as e:
        print(f"⚠️ Error in speak(): {e}")

# -------- Load CSV Data --------
dataframes = []
labels_loaded = []

for filename in os.listdir():
    if filename.endswith("_data.csv") and os.path.getsize(filename) > 0:
        try:
            df = pd.read_csv(filename)
            if "label" in df.columns and df.shape[1] == 64:
                dataframes.append(df)
                labels_loaded.append(df['label'].iloc[0])
        except Exception as e:
            print(f"⚠️ Skipping file {filename}: {e}")

if not dataframes:
    print("❌ No valid gesture CSV files found.")
    exit()

data = pd.concat(dataframes, ignore_index=True)
X = data.drop("label", axis=1)
y = data["label"]

# -------- Train Model --------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)
joblib.dump(clf, "gesture_model.pkl")

print("✅ Model trained on:", sorted(y.unique().tolist()))

# -------- Mediapipe & Webcam --------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📷 Show a trained gesture (like A or B)... Press Esc to quit.")

last_prediction = ""
spoken_once = False

def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0]
    normalized = [(x - base_x, y - base_y, z - base_z) for x, y, z in landmarks]
    return [val for triplet in normalized for val in triplet]

# -------- Main Loop --------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera read failed. Exiting.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            if len(landmarks) != 21:
                continue

            features = normalize_landmarks(landmarks)
            prediction = clf.predict([features])[0]
            confidence = clf.predict_proba([features])[0].max()

            print(f"Prediction: {prediction} | Confidence: {round(confidence, 2)}")

            if confidence > 0.7:
                cv2.putText(frame, f"{prediction} ({int(confidence*100)}%)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if prediction != last_prediction:
                    spoken_once = False

                if not spoken_once:
                    speak(prediction)
                    spoken_once = True

                last_prediction = prediction
            else:
                cv2.putText(frame, "Low confidence", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("🖐 Live Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
