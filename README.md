# Hand Gesture Recognition

Real-time hand gesture recognition (fist, palm, peace) using
MediaPipe and a Random Forest classifier trained on my own dataset.

## How it works
1. MediaPipe detects 21 hand landmarks from the webcam.
2. Landmarks are normalized (relative to the wrist, scaled by hand size).
3. A Random Forest classifier predicts the gesture.
4. A 10-frame vote smooths the result, and low-confidence
   predictions show as "unsure".

## Dataset
Collected by me using collect.py: about ___ samples per gesture,
both hands, many angles.

## Results
- Test accuracy: ___%
- Problem found: palm facing down was predicted as peace,
  because the training data had only one hand orientation.
- Fix: added more varied data and normalized landmarks by hand size.

## Run it
pip install opencv-python mediapipe==0.10.14 numpy pandas scikit-learn
python collect.py
python train.py
python live.py

## What I learned
- A high test accuracy can be misleading if the data is not varied.
- Data variety and normalization matter more than the model choice.
- ML improves through a loop: test, find the weak spot, add data, retrain.
