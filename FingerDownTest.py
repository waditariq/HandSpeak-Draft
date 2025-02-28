import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    success, img = webcam.read()
    if not success:
        continue

    img = cv2.flip(img, 1)  # Flip image for a mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine if it's Left or Right hand
            hand_label = "Right" if handedness.classification[0].label == "Right" else "Left"

            # Get landmark positions
            landmarks = hand_landmarks.landmark
            fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            finger_tips = [4, 8, 12, 16, 20]
            finger_pips = [2, 6, 10, 14, 18]

            fingers_down = []

            for i in range(5):
                h, w, _ = img.shape
                x, y = int(landmarks[finger_tips[i]].x * w), int(landmarks[finger_tips[i]].y * h)

                if i == 0:  # Special handling for the thumb
                    thumb_tip_x = landmarks[4].x
                    thumb_mcp_x = landmarks[2].x

                    if hand_label == "Right":  
                        # Right hand: thumb is "down" if tip is to the RIGHT of MCP
                        if thumb_tip_x > thumb_mcp_x:
                            fingers_down.append(fingers[i])
                    else:  
                        # Left hand: thumb is "down" if tip is to the LEFT of MCP
                        if thumb_tip_x < thumb_mcp_x:
                            fingers_down.append(fingers[i])
                else:
                    # Normal finger detection based on vertical position
                    if landmarks[finger_tips[i]].y > landmarks[finger_pips[i]].y:
                        fingers_down.append(fingers[i])

                # Display text near the finger
                if fingers[i] in fingers_down:
                    cv2.putText(img, f"{hand_label} {fingers[i]}", (x - 30, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Hand Tracking', img)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
