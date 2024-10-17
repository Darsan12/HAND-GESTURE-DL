import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Step 1: Set up webcam input
def capture_data():
    data_dir = 'data/'  # Directory to save the images
    gestures = ['thumbs_up', 'thumbs_down', 'fist', 'palm']  # List of gestures
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    for gesture in gestures:
        if not os.path.exists(os.path.join(data_dir, gesture)):
            os.makedirs(os.path.join(data_dir, gesture))

        print(f"Collecting data for gesture: {gesture}")
        for img_id in range(100):  # Capture 100 images per gesture
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Flip horizontally
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Save the image to the respective folder
            img_path = os.path.join(data_dir, gesture, f'{gesture}_{img_id}.jpg')
            cv2.imwrite(img_path, frame)
            cv2.imshow('Collecting Data', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Step 2: Preprocess the data
def preprocess_data(data_dir, img_size=(128, 128)):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=32, class_mode='categorical', subset='training')
    val_gen = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=32, class_mode='categorical', subset='validation')
    return train_gen, val_gen

# Step 3: Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')  # Assuming 4 gesture classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train the model
def train_model(model, train_gen, val_gen, epochs=10):
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save('gesture_recognition_model.h5')

# Step 5: Predict gestures in real-time
def predict_gesture(frame, model):
    img = cv2.resize(frame, (128, 128))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    return class_idx

def run_real_time_prediction():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    model = load_model('gesture_recognition_model.h5')
    gesture_labels = ['thumbs_up', 'thumbs_down', 'fist', 'palm']  # Labels for the gestures

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Optionally crop the hand region using the bounding box (not shown here)
                # Predict the gesture
                gesture = predict_gesture(frame, model)
                cv2.putText(frame, f"Gesture: {gesture_labels[gesture]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Step 6: Execute the process
if __name__ == '__main__':
    # Step 1: Collect and label data
    capture_data()

    # Step 2: Preprocess the data
    data_dir = 'data/'
    train_gen, val_gen = preprocess_data(data_dir)

    # Step 3: Build the CNN model
    model = build_model()

    # Step 4: Train the model
    train_model(model, train_gen, val_gen, epochs=10)

    # Step 5: Use the model for real-time prediction
    run_real_time_prediction()
