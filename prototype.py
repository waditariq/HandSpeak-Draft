import cv2
import mediapipe as mp

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands()

webcam = cv2.VideoCapture(0)
while webcam.isOpened():
    success, img = webcam.read()

    # apply hand tracking
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(img)

    # Draw on image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for  hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img,hand_landmarks,connections=mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Prototype', img)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
webcam.release()
cv2.destroyAllWindows()