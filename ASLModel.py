import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# Constants
DATA_DIR = r"C:\Users\roanm\Desktop\Cmpe 246 project\ASL_Dataset"
IMG_SIZE = (224, 224)  # Image resize target
BATCH_SIZE = 32
EPOCHS = 1000

# Function to extract red parts of an image
def extract_red_hand(image):
    """Extracts only the red regions (hand) of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Lower and upper red color boundaries
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask = mask1 + mask2
    return cv2.bitwise_and(image, image, mask=mask)

# Function to preprocess images
def preprocess_images(data_dir):
    """Preprocess images: extract red hand and label them."""
    X, y = [], []
    label_map = {chr(i): i - 65 for i in range(65, 91)}  # A=0, B=1, ..., Z=25

    # Load and preprocess images
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            for letter_folder in os.listdir(folder_path):
                letter_folder_path = os.path.join(folder_path, letter_folder)
                if os.path.isdir(letter_folder_path):
                    for image_name in os.listdir(letter_folder_path):
                        image_path = os.path.join(letter_folder_path, image_name)
                        if image_name.endswith(".jpg"):
                            image = cv2.imread(image_path)
                            if image is not None:
                                # Extract red hand
                                image = extract_red_hand(image)
                                image = cv2.resize(image, IMG_SIZE) / 255.0  # Normalize and resize
                                X.append(image)
                                y.append(label_map[letter_folder.upper()])  # Use folder name as label
                            else:
                                print(f"Warning: Unable to read image {image_name}")

    return np.array(X), np.array(y)

# Preprocess data
X, y = preprocess_images(DATA_DIR)
print(f"Loaded {len(X)} images and {len(y)} labels.")

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data size: {len(X_train)} images")
print(f"Validation data size: {len(X_val)} images")

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Try to load the model if it exists, otherwise create a new model
try:
    model = load_model('asl_model.h5')
    print("Model loaded successfully.")
except:
    print("No saved model found. Creating a new model.")
    
    # Load pre-trained MobileNetV2 as base model (without top layers)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    # Build the custom top layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')  # 26 classes for A-Z
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# TensorBoard callback
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Live graph callback for real-time updates during training
class LiveGraphCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots(2, 1, figsize=(8, 6))
        self.ax[0].set_title("Model Accuracy")
        self.ax[0].set_xlabel("Epochs")
        self.ax[0].set_ylabel("Accuracy")
        self.ax[1].set_title("Model Loss")
        self.ax[1].set_xlabel("Epochs")
        self.ax[1].set_ylabel("Loss")
        self.accuracy = []
        self.val_accuracy = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.accuracy.append(logs.get('accuracy', 0))
        self.val_accuracy.append(logs.get('val_accuracy', 0))
        self.loss.append(logs.get('loss', 0))
        self.val_loss.append(logs.get('val_loss', 0))

        self.ax[0].cla()
        self.ax[0].plot(self.accuracy, label='Training Accuracy', color='blue')
        self.ax[0].plot(self.val_accuracy, label='Validation Accuracy', color='orange')
        self.ax[0].legend()

        self.ax[1].cla()
        self.ax[1].plot(self.loss, label='Training Loss', color='blue')
        self.ax[1].plot(self.val_loss, label='Validation Loss', color='orange')
        self.ax[1].legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)  # Allow GUI refresh

live_graph_callback = LiveGraphCallback()

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val),
                    callbacks=[tensorboard_callback, live_graph_callback],
                    class_weight=class_weight_dict)

# Save the trained model
model.save('model.p')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Predictions and evaluation metrics
y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)

# Classification report and confusion matrix
print("Classification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
