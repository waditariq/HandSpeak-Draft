import pickle
import sys
import cv2
import mediapipe as mp
import numpy as np
import os

if getattr(sys, 'frozen', False):
    # Running as .exe
    base_path = sys._MEIPASS
else:
    # Running as .py
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, 'model.p')
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

label_map = [chr(i) for i in range(65, 91)]  # A-Z

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()
# Variable to track whether menu is shown
show_menu_flag = False

def show_menu(frame):
    """Show a simple menu on the frame when ESC is pressed"""
    menu_texts = [
        "=== MENU ===",
        "Press ESC to resume",
        "Press Q to exit"
    ]
    y_pos = 30
    for text in menu_texts:
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        y_pos += 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []

    if not show_menu_flag:  # If the menu is not being shown, continue with predictions
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            z_vals = [lm.z for lm in hand_landmarks.landmark]

            min_x, min_y = min(x_vals), min(y_vals)
            max_x, max_y = max(x_vals), max(y_vals)
            width = max_x - min_x
            height = max_y - min_y

            if width == 0 or height == 0:
                continue

            for lm in hand_landmarks.landmark:
                norm_x = (lm.x - min_x) / width
                norm_y = (lm.y - min_y) / height
                data_aux.extend([norm_x, norm_y, lm.z])

            if len(data_aux) == model.n_features_in_:
                prediction_probs = model.predict_proba([np.asarray(data_aux)])[0]
                predicted_index = np.argmax(prediction_probs)
                confidence = prediction_probs[predicted_index]

                if confidence > 0.2:
                    predicted_letter = label_map[predicted_index]
                    display_text = f'{predicted_letter}'
                    cv2.putText(frame, f'Prediction: {display_text}', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

    else:  # If the menu flag is true, show the menu
        show_menu(frame)

    # Show the frame with or without the menu
    cv2.imshow('ASL Prediction', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key
        if show_menu_flag == True:
            show_menu_flag = False
        else:
            show_menu_flag = True


    if key == ord('q') and show_menu_flag == True:  # Q to quit when menu is shown
        break

cap.release()
cv2.destroyAllWindows()
