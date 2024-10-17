# *Hand Gesture Recognition using CNN and MediaPipe 🎯*
## ***This project focuses on real-time hand gesture recognition by leveraging CNN  and MediaPipe for hand tracking.***

## ***The gestures identified include thumbs up, thumbs down, fist, and palm, which are commonly used in interactive systems.***

### *🚀 Problem Statement*
#### Create a hand gesture recognition system using a webcam that captures various gestures, processes them using MediaPipe, and classifies them using a CNN model. The system should:

1. Collect and store gesture images for model training.
2. Train a CNN to classify gestures based on captured images.
3. Perform real-time predictions to display the identified gesture on screen.
4. 🛠️ Setup and Installation
5. Follow the instructions below to get the project running:

### *Step 1: Prerequisites*
- Python 3.x
- pip for installing dependencies
- Install Dependencies
- Run the following command to install the required packages:
- pip install opencv-python-headless mediapipe tensorflow
- 📄 How to Run the Code
- Step 1: Collect and Label Data
- Run the following command to capture gesture images from your webcam:
- python main.py  # Ensure **`capture_data()`** is uncommented inside **`__main__`**
- Use the 'q' key to stop data collection at any time. Each gesture will save 100 images inside the data/ folder.

### *Step 2: Train the Model*
- Once the data is collected, you can train the CNN model:
- python main.py  # Ensure `train_model()` is enabled inside `__main__`
- This will create a gesture_recognition_model.h5 file containing the trained model.

### *Step 3: Perform Real-Time Gesture Recognition*
- After training, run the model for real-time predictions:
- python main.py  # Ensure `run_real_time_prediction()` is active in `__main__`
- 🧠 Brief Description of the Approach
- Data Capture: Using OpenCV and MediaPipe, the code captures images of hand gestures.
- Preprocessing: The collected images are resized and normalized for model training using ImageDataGenerator.
- Model Training: A CNN model is built with TensorFlow, consisting of convolutional layers, max-pooling, and dense layers. It’s trained using the captured gesture images.
- Real-Time Prediction: The trained model is used to predict gestures in real-time, displaying the gesture name on the webcam feed.
 
### *🔥 Future Improvements*
- Add More Gestures: Expand the gesture library with additional classes.
- Use Bounding Boxes: Improve prediction by cropping the hand region using bounding boxes.
- Deploy as Web App: Convert the model into a web-based application using Flask or Django.

### *🤝 Contributing*
- Feel free to contribute by submitting issues or pull requests. Make sure to follow the repository structure and keep your code clean and documented.

### *📝 License*
- This project is open-source. Use it as you like, but please give credit where due.
