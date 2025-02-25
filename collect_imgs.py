import cv2
import os
import time

# Create folder 'DATA' if it doesn't exist
folder_name = "DATA"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

num_images = 30
capture_duration = 4  # seconds
capture_interval = capture_duration / num_images

capturing = False
image_count = 0
next_capture_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    key = cv2.waitKey(1) & 0xFF
    
    cv2.putText(frame, "Press 'q' to quit", (80, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If not capturing, display prompt overlay on the camera feed
    if not capturing:
        cv2.putText(frame, "Press 's' to start capturing", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if key == ord('s'):
            capturing = True
            image_count = 0
            next_capture_time = time.time()  # capture immediately
    else:
        # While capturing, overlay count info
        cv2.putText(frame, f"Capturing: {image_count}/{num_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        current_time = time.time()
        if image_count < num_images and current_time >= next_capture_time:
            filename = os.path.join(folder_name, f"image_{image_count+1:02d}.jpg")
            cv2.imwrite(filename, frame)
            image_count += 1
            next_capture_time = current_time + capture_interval
        if image_count >= num_images:
            cv2.putText(frame, "Capture Complete", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Optionally, you could reset capturing after some time or leave it as is

    cv2.imshow('Camera', frame)

    # Quit on pressing 'q'
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()