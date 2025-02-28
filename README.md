# HandSpeak Project

This project captures hand gesture images, extracts hand landmarks using MediaPipe, trains a Random Forest classifier to recognize the gestures, and performs real-time inference using your webcam.

Please run the files in this order:
    -collect_imgs.py => This should make a folder with several subfolders inside
    -create_dataset.py => This should make a data.pickle file
    -train_classifier.py => This trains a classifier based on extracted features, and makes model.p
    -inference_classifier => Final file to run, which is where you test the model. It should identify different letters.

## Prerequisites

Ensure you have Python 3 installed. Then install the following packages:

```bash
pip install opencv-python mediapipe scikit-learn numpy

Please see the project document for full details (.docx)