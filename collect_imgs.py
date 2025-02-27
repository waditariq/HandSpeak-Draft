import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Initialize camera (try different indices if 0 doesn't work)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access camera. Try changing the index (0, 1, 2, etc.)")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame during capture.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
