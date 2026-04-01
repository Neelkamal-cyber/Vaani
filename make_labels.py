import os
import json

# Your exact dataset path from the screenshots
DATA_PATH = r"D:\Self_made database\best_accuracy"
LABEL_MAP_FILE = "label_map.json"

# 1. Look inside the folder and get all the word names (ignoring stray files)
words = sorted([w for w in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, w))])

# 2. Assign a number to each word
label_map = {word: idx for idx, word in enumerate(words)}

# 3. Save it to a JSON file
with open(LABEL_MAP_FILE, 'w') as f:
    json.dump(label_map, f, indent=4)

print(f"✅ Success! 'label_map.json' created with {len(words)} words.")