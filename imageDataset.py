import cv2
import mediapipe as mp
import os
from datetime import datetime

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Directory to store dataset
DATA_DIR = r"C:\Users\roanm\Desktop\Cmpe 246 project\ASL_Dataset"
os.makedirs(DATA_DIR, exist_ok=True)

webcam = cv2.VideoCapture(0)

def save_image(img, label, hand_type, flipped=False):
    """Save the captured image with the appropriate label and hand type."""
    # Adjust hand type if flipped
    if flipped:
        hand_type = "Left" if hand_type == "Right" else "Right"

    hand_dir_path = os.path.join(DATA_DIR, hand_type, label)
    os.makedirs(hand_dir_path, exist_ok=True)
    
    # Add flipped indication to the filename
    file_name = os.path.join(hand_dir_path, f"{len(os.listdir(hand_dir_path))}_flipped.jpg" if flipped else f"{len(os.listdir(hand_dir_path))}.jpg")
    cv2.imwrite(file_name, img)
    print(f"Saved {file_name}")

while webcam.isOpened():
    success, img = webcam.read()
    if not success:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label  # 'Left' or 'Right'
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Data Collection", img)

    key = cv2.waitKey(5) & 0xFF
    if key == ord("s"):  # Press 's' to save an image
        label = input("Enter letter label (A-Z): ").upper()
        
        # Save original image based on hand type
        save_image(img, label, hand_type)
        
        # Save flipped image (mirror image of the hand)
        img_flipped = cv2.flip(img, 1)
        save_image(img_flipped, label, hand_type, flipped=True)

    if key == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
