import cv2
import mediapipe as mp
import os

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Directory to store dataset
DATA_DIR = r"C:\Users\roanm\Desktop\Cmpe 246 project\ASL_Dataset"
os.makedirs(DATA_DIR, exist_ok=True)

webcam = cv2.VideoCapture(0)

def save_gesture_frames(label, frames, hand_type):
    """Save frames of a gesture to the appropriate folder."""
    hand_dir_path = os.path.join(DATA_DIR, hand_type, label)
    os.makedirs(hand_dir_path, exist_ok=True)

    for i, frame in enumerate(frames):
        # Save original frame
        file_name = os.path.join(hand_dir_path, f"{i}.jpg")
        cv2.imwrite(file_name, frame)
        print(f"Saved {file_name}")

        # Save flipped (mirrored) frame
        frame_flipped = cv2.flip(frame, 1)
        file_name_flipped = os.path.join(hand_dir_path, f"{i}_flipped.jpg")
        cv2.imwrite(file_name_flipped, frame_flipped)
        print(f"Saved {file_name_flipped}")

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
    if key == ord("s"):  # Press 's' to start recording a gesture
        label = input("Enter letter label (A-Z): ").upper()

        if label in ["J", "Z"]:  # For J and Z, we need to capture multiple frames for the gesture
            frames = []
            print(f"Recording gesture {label}...")
            for _ in range(10):  # Record 10 frames for the gesture
                success, img = webcam.read()
                if not success:
                    break
                frames.append(img)  # Add each frame of the gesture
                cv2.imshow("Recording Gesture", img)
                cv2.waitKey(500)  # Wait for 500ms between frames
            save_gesture_frames(label, frames, hand_type)

    if key == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
