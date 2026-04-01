import os
import cv2
import json
import numpy as np
import mediapipe as mp

# --- CONFIGURATION ---
SOURCE_VIDEO_PATH = r"D:\Self_made database\30X30_data"   # Where your raw videos live
TARGET_NPY_PATH = r"D:\Self_made database\best_accuracy"   # Where the .npy files will go
LABEL_MAP_FILE = "label_map.json"             # The file to save your labels
SEQUENCE_LENGTH = 60

mp_holistic = mp.solutions.holistic

def extract_normalized_features(results):
    """Extracts and normalizes the 258 landmarks based on shoulder position."""
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

def build_dataset_and_labels():
    os.makedirs(TARGET_NPY_PATH, exist_ok=True)
    
    # 1. Create and Save the Label Map
    words = sorted([w for w in os.listdir(SOURCE_VIDEO_PATH) if os.path.isdir(os.path.join(SOURCE_VIDEO_PATH, w))])
    label_map = {word: idx for idx, word in enumerate(words)}
    
    with open(LABEL_MAP_FILE, 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"✅ Label Map saved to {LABEL_MAP_FILE} with {len(words)} words.")

    # 2. Extract .npy files
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for word in words:
            word_dir = os.path.join(SOURCE_VIDEO_PATH, word)
            target_dir = os.path.join(TARGET_NPY_PATH, word)
            os.makedirs(target_dir, exist_ok=True)
            
            videos = [v for v in os.listdir(word_dir) if v.endswith('.mp4')]
            print(f"Processing '{word}' ({len(videos)} videos)...")
            
            for video_name in videos:
                cap = cv2.VideoCapture(os.path.join(word_dir, video_name))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 10:
                    continue
                
                indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH).astype(int)
                sequence_data, frame_idx = [], 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_idx in indices:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(image)
                        sequence_data.append(extract_normalized_features(results))
                    frame_idx += 1
                cap.release()
                
                # Ensure we have exactly 60 frames, then save
                if len(sequence_data) == SEQUENCE_LENGTH:
                    np.save(os.path.join(target_dir, video_name.replace('.mp4', '.npy')), sequence_data)

    print("🎉 Dataset extraction complete!")

if __name__ == "__main__":
    build_dataset_and_labels()