import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import random  # for shuffling

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Debug: List available folders
available_folders = sorted(os.listdir(DATA_DIR))
print(f"🗂 Available Folders in {DATA_DIR}: {available_folders}")

# Expected class folders in order
expected_classes = ['0', '1', '2']
missing_classes = [c for c in expected_classes if c not in available_folders]
if missing_classes:
    print(f"⚠ Warning: Missing classes {missing_classes}! Check dataset structure.")
    exit()

# Process each class folder in expected order (0, then 1, then 2)
for dir_ in expected_classes:
    dir_path = os.path.join(DATA_DIR, dir_)
    class_label = int(dir_)  # Convert folder name to integer label

    # Debug: Check number of images in the folder
    image_files = os.listdir(dir_path)
    print(f"📂 Processing Class {class_label}: {len(image_files)} images")

    for img_path in image_files:
        data_aux = []
        full_img_path = os.path.join(dir_path, img_path)
        img = cv2.imread(full_img_path)

        # Skip if the image cannot be read
        if img is None:
            print(f"⚠ Warning: Could not read image {full_img_path}. Skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x)
                    data_aux.append(lm.y)
            # Only add sample if landmarks were found
            data.append(data_aux)
            labels.append(class_label)
            print(f"📝 Assigned Label: {class_label} for image {img_path}")
        else:
            print(f"⚠ No hand landmarks detected in {img_path}. Skipping this image.")

# Optionally, shuffle the dataset so the labels are mixed
combined = list(zip(data, labels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)
labels = list(labels)  # Convert back to list if needed

# Save the dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Dataset created successfully: data.pickle")
print(f"🔍 First 10 Labels in Dataset: {labels[:10]} (Should be a mix of 0, 1, 2)")
