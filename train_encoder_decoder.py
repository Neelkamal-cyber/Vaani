import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- CONFIGURATION ---
DATA_PATH = r"D:\Self_made database\best_accuracy"
LABEL_MAP_FILE = "label_map.json"
SEQUENCE_LENGTH = 60
FEATURE_DIM = 258

# 1. Load the Label Map
with open(LABEL_MAP_FILE, 'r') as f:
    label_map = json.load(f)
num_classes = len(label_map)
print(f"🎯 Loaded {num_classes} words from Label Map.")

# 2. Load Data into Memory
sequences, labels = [], []
for word, label_idx in label_map.items():
    word_path = os.path.join(DATA_PATH, word)
    if os.path.exists(word_path):
        for npy_file in os.listdir(word_path):
            sequences.append(np.load(os.path.join(word_path, npy_file)))
            labels.append(label_idx)

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

print("\n🧠 Building Encoder-Decoder Architecture...")

# --- THE ENCODER ---
# Reads the video and creates the "Context Vector" (state_h, state_c)
encoder_inputs = Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM), name="Video_Input")
encoder_lstm = LSTM(128, return_state=True, activation='tanh', name="Encoder_LSTM")
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# --- THE BRIDGE ---
# We duplicate the Context Vector 60 times so the Decoder has enough data to work with
decoder_bridge = RepeatVector(SEQUENCE_LENGTH, name="Context_Vector_Bridge")(encoder_outputs)

# --- THE DECODER ---
# Unpacks the Context Vector using the Encoder's exact memory states
decoder_lstm = LSTM(128, return_sequences=False, activation='tanh', name="Decoder_LSTM")
decoder_outputs = decoder_lstm(decoder_bridge, initial_state=[state_h, state_c])
decoder_dropout = Dropout(0.3)(decoder_outputs)

# --- CLASSIFIER ---
dense_layer = Dense(128, activation='relu', name="Dense_Processor")(decoder_dropout)
final_output = Dense(num_classes, activation='softmax', name="Final_Prediction")(dense_layer)

# --- COMPILE MODEL ---
model = Model(inputs=encoder_inputs, outputs=final_output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])

model.summary()

# --- TRAINING ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    ModelCheckpoint('vaani_enc_dce.h5', monitor='val_categorical_accuracy', save_best_only=True)
]

print("\n🚀 Starting Encoder-Decoder Training...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=32, callbacks=callbacks)

print("\n🎉 Model saved as 'vaani_enc_dec.h5'")