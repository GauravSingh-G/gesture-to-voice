# ✅ Gesture to Voice with Clean GPT Predictions + Dictionary Filtering

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import time
import torch
from sklearn.ensemble import RandomForestClassifier
from gtts import gTTS
import pygame
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import difflib

# -------- LOAD LOCAL GPT-2 MODEL --------
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()

# -------- LOAD DICTIONARY --------
with open("dictionary.txt", "r") as f:
    DICTIONARY = [word.strip().upper() for word in f.readlines()]

# -------- IMPROVED NEXT WORD PREDICTOR --------
def get_next_word_predictions(prompt_text, top_k=10):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    outputs = model(**inputs)
    next_token_logits = outputs.logits[0, -1, :]
    top_k_probs = torch.topk(next_token_logits, top_k)
    predicted_tokens = top_k_probs.indices.tolist()
    predictions = [tokenizer.decode([token]).strip().upper() for token in predicted_tokens]
    valid_predictions = [word for word in predictions if word in DICTIONARY and word.isalpha() and len(word) > 1]
    return valid_predictions[:3]

# -------- SPEAK FUNCTION --------
def speak(text):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
        tts = gTTS(text=text, lang='en')
        tts.save(temp_path)
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.stop()
        pygame.quit()
        os.remove(temp_path)
    except Exception as e:
        print(f"⚠️ Error in speak(): {e}")

# -------- TOP-N SUGGESTOR --------
def suggest_top_n(prefix, dictionary=DICTIONARY, n=3):
    if len(prefix) < 2:
        return []
    return difflib.get_close_matches(prefix.upper(), dictionary, n=n, cutoff=0.6)

# -------- LOAD CSV TRAINING DATA --------
dataframes = []
for filename in os.listdir():
    if filename.endswith("_data.csv") and os.path.getsize(filename) > 0:
        df = pd.read_csv(filename)
        if "label" in df.columns and df.shape[1] == 64:
            dataframes.append(df)

if not dataframes:
    print("❌ No valid gesture CSV files found.")
    exit()

data = pd.concat(dataframes, ignore_index=True)
X = data.drop("label", axis=1)
y = data["label"]

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)
joblib.dump(clf, "gesture_model.pkl")
print("✅ Model trained on:", sorted(y.unique().tolist()))

# -------- INIT MEDIAPIPE --------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📷 Show gestures... Press Esc to quit. Use SELECT_1/2/3 to choose suggestion.")

# -------- VARIABLES --------
buffer = ""
full_sentence = []
last_pred = ""
last_time = 0
suggestions = []
next_word_mode = False
next_word_suggestions = []

# -------- NORMALIZATION --------
def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    normalized = [(lm.x - base_x, lm.y - base_y, lm.z - base_z) for lm in landmarks]
    return [val for triplet in normalized for val in triplet]

# -------- MAIN LOOP --------
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
            landmarks = hand_landmarks.landmark
            if len(landmarks) != 21:
                continue

            features = normalize_landmarks(landmarks)
            prediction = clf.predict([features])[0]
            confidence = clf.predict_proba([features])[0].max()

            if confidence > 0.75:
                now = cv2.getTickCount()
                if prediction != last_pred or (now - last_time)/cv2.getTickFrequency() > 1.2:
                    print(f"🔤 Predicted: {prediction} | Buffer: {buffer}")
                    last_pred = prediction
                    last_time = now

                    if prediction == "END":
                        if buffer:
                            word_to_add = buffer.upper()
                            if word_to_add not in DICTIONARY and suggestions:
                                word_to_add = suggestions[0]
                                print(f"⚠️ Buffer '{buffer}' not in dictionary. Using suggestion: {word_to_add}")
                            full_sentence.append(word_to_add)
                            speak(word_to_add)
                        buffer = ""
                        sentence_str = " ".join(full_sentence)
                        next_word_suggestions = get_next_word_predictions(sentence_str)
                        next_word_mode = True

                    elif prediction.startswith("SELECT_"):
                        index = int(prediction[-1]) - 1
                        selected_word = None

                        if next_word_mode:
                            if index < len(next_word_suggestions):
                                selected_word = next_word_suggestions[index]
                        else:
                            if not buffer and suggestions:
                                selected_word = suggestions[0]
                            elif index < len(suggestions):
                                selected_word = suggestions[index]
                            elif buffer.upper() not in DICTIONARY and suggestions:
                                selected_word = suggestions[0]

                        if selected_word:
                            full_sentence.append(selected_word.upper())
                            speak(selected_word)
                            buffer = ""
                            sentence_str = " ".join(full_sentence)
                            next_word_suggestions = get_next_word_predictions(sentence_str)
                            next_word_mode = True
                        else:
                            print("⚠️ Invalid selection")

                    elif prediction == "ERASE":
                        buffer = buffer[:-1]
                        next_word_mode = False

                    else:
                        buffer += prediction
                        next_word_mode = False

    # -------- Display --------
    y_offset = 25
    cv2.putText(frame, f"Buffer: {buffer}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    y_offset += 30
    sentence_str = " ".join(full_sentence)
    cv2.putText(frame, f"Sentence: {sentence_str}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    y_offset += 30

    if next_word_mode:
        for i, word in enumerate(next_word_suggestions):
            cv2.putText(frame, f"NEXT {i+1}: {word}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
            y_offset += 25
    else:
        suggestions = suggest_top_n(buffer)
        for i, word in enumerate(suggestions):
            cv2.putText(frame, f"{i+1}. {word}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 25

    cv2.imshow("🤖 Real-Time Sentence Builder", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()