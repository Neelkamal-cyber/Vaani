import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
DATA_PATH = r"D:\Self_made database\best_accuracy"
LABEL_MAP_FILE = "label_map.json"
MODEL_PATH = "vaani_enc_dce.h5"

print("🔄 Loading Model and Labels...")
with open(LABEL_MAP_FILE, 'r') as f:
    label_map = json.load(f)

# Reverse map for the graph labels (0 -> "AGAIN")
idx_to_word = {v: k for k, v in label_map.items()}
class_names = [idx_to_word[i] for i in range(len(idx_to_word))]

model = load_model(MODEL_PATH)

print("🔄 Loading Test Data...")
X_test, y_true = [], []
for word, label_idx in label_map.items():
    word_path = os.path.join(DATA_PATH, word)
    if os.path.exists(word_path):
        for npy_file in os.listdir(word_path):
            X_test.append(np.load(os.path.join(word_path, npy_file)))
            y_true.append(label_idx)

X_test = np.array(X_test)
y_true = np.array(y_true)

print("🧠 Generating Predictions...")
# Get the model's guesses
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- GENERATE CONFUSION MATRIX ---
print("📊 Drawing Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(15, 12))
# Using seaborn to make it look professional
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Vaani Encoder-Decoder: Sign Language Confusion Matrix")
plt.ylabel("Actual Sign (What the user actually did)")
plt.xlabel("Predicted Sign (What the AI guessed)")
plt.xticks(rotation=90)
plt.tight_layout()

# Save the image for your project report!
plt.savefig("confusion_matrix_report.png", dpi=300)
plt.show()

# Print detailed text report
print("\n📝 Detailed Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))