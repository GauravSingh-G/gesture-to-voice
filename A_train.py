import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("A_data.csv")
X = data.drop("label", axis=1)
y = data["label"]

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save model (optional)
joblib.dump(clf, "gesture_A_model.pkl")

# Setup Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

print("📷 Showing live camera. Make 'A' gesture to test... Press Esc to quit.")

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

            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            if len(features) == 63:
                prediction = clf.predict([features])[0]
                confidence = clf.predict_proba([features])[0].max()
                if prediction == "A":
                    cv2.putText(frame, f"Detected: A ({int(confidence*100)}%)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc key
        break

cap.release()
cv2.destroyAllWindows()
