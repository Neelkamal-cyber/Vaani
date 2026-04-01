import os
import cv2
import json
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import queue
from tensorflow.keras.models import load_model

# --- 1. CONFIGURATION ---
MODEL_PATH = "vaani_enc_dce.h5"     # The model we just trained
LABEL_MAP_FILE = "label_map.json"   # The JSON file we generated earlier
SEQUENCE_LENGTH = 60
CONFIDENCE_THRESHOLD = 0.85         # Slightly lowered for the new architecture

# --- 2. TEXT-TO-SPEECH BACKGROUND WORKER ---
speech_queue = queue.Queue()

def tts_worker():
    """Background worker that speaks without freezing the camera."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start the audio worker immediately
threading.Thread(target=tts_worker, daemon=True).start()

def speak_word(text):
    """Tosses the word into the queue."""
    speech_queue.put(text)

# --- 3. LOAD LABELS & MODEL ---
# Load the JSON and flip it so we can look up numbers (e.g., 0 -> "AGAIN")
with open(LABEL_MAP_FILE, 'r') as f:
    label_map = json.load(f)
idx_to_word = {v: k for k, v in label_map.items()}

print("🔄 Loading Encoder-Decoder Model...")
model = load_model(MODEL_PATH)
print(f"✅ Model Loaded! Ready to recognize {len(idx_to_word)} signs.")

# --- 4. MEDIAPIPE SETUP ---
mp_holistic = mp.solutions.holistic

def extract_and_normalize_features(results):
    """Shoulder-normalization to keep signs consistent at any distance."""
    shoulder_x, shoulder_y = 0.0, 0.0
    if results.pose_landmarks:
        left = results.pose_landmarks.landmark[11]
        right = results.pose_landmarks.landmark[12]
        shoulder_x = (left.x + right.x) / 2
        shoulder_y = (left.y + right.y) / 2

    pose = np.array([[res.x - shoulder_x, res.y - shoulder_y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x - shoulder_x, res.y - shoulder_y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x - shoulder_x, res.y - shoulder_y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])

# --- 5. REAL-TIME ENGINE ---
sequence = []
predictions = []
current_word = "Initializing..."
sentence = []
frames_since_last_sign = 0

# Open webcam (Forcing widescreen 720p to fix the cropped view!)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Skeletons
        mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Buffer the last 60 frames
        keypoints = extract_and_normalize_features(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        # --- PREDICTION LOGIC ---
        if len(sequence) == SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_index = np.argmax(res)
            confidence = res[predicted_index]
            
            if confidence > CONFIDENCE_THRESHOLD:
                predictions.append(predicted_index)
                predictions = predictions[-15:] # Look at last 15 frames
                
                # If it predicted the same thing 10 times, it's a solid guess
                if predictions.count(predicted_index) > 10:
                    detected_word = idx_to_word[predicted_index] # Use our JSON map!
                    
                    if current_word != detected_word:
                        current_word = detected_word
                        frames_since_last_sign = 0
                        
                        if len(sentence) == 0 or sentence[-1] != current_word:
                            sentence.append(current_word)
                            speak_word(current_word) 
            else:
                frames_since_last_sign += 1

        # Timeout: Clear the current word if hands are down for 2 seconds
        if frames_since_last_sign > 60:
            current_word = ""

        # Keep sentence length manageable on screen
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # --- UI OVERLAY ---
        cv2.rectangle(image, (0,0), (1280, 50), (40, 40, 40), -1)
        cv2.putText(image, f"Sign: {current_word}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.rectangle(image, (0, 670), (1280, 720), (200, 50, 50), -1)
        cv2.putText(image, " ".join(sentence), (10, 705), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('VAANI Encoder-Decoder Live', image)

        # Hotkeys
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []

    cap.release()
    cv2.destroyAllWindows()