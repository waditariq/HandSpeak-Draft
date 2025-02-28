import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load Trained Model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize Camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label Mapping
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Capture Frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame. Skipping...")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Hand Landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Process Only One Hand (Avoid Feature Mismatch)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw Landmarks
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Extract Features (Ensure Consistency)
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))  # Normalize x
            data_aux.append(y - min(y_))  # Normalize y

        # Check Feature Size Before Prediction
        print(f"🔍 Extracted {len(data_aux)} features (Expected: 42)")

        if len(data_aux) != 42:
            print("⚠ Warning: Feature size mismatch! Skipping this frame.")
        else:
            # Make Prediction
            raw_prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(raw_prediction[0])]

            # Print Debugging Info
            print(f"🤖 Raw Prediction: {raw_prediction} -> Mapped Label: {predicted_character}")

            # Draw Bounding Box & Label
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Show Video Feed
    cv2.imshow('frame', frame)

    # Exit on 'Q' Key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
