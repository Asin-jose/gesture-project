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
Collected by me using `collect.py`: about 1,870 samples in total
(roughly 500+ per gesture), recorded with both hands, at different
angles, distances and orientations.

## Results
- Test accuracy: **95%** on 374 held-out samples (80/20 split)
- F1-score: 0.95 for palm and peace, similar for fist

| Gesture | Correct | Total |
|---------|---------|-------|
| Fist    | 100     | 104   |
| Palm    | 131     | 136   |
| Peace   | 126     | 134   |

- Most common mistake: peace predicted as palm (6 times).

## Problem found and fix
- **Problem:** the first model showed 98% accuracy, but live it called
  palm facing down "peace". The training data had only one hand orientation.
- **Fix:** recorded more varied data (both hands, different angles) and
  normalized landmarks by hand size. Live results improved.

## Limitations
- The test samples come from the same recording sessions as the
  training samples, so real-world accuracy may be lower.
- Works best with good lighting and one hand in view.

## Run it
```
pip install opencv-python mediapipe==0.10.14 numpy pandas scikit-learn
python collect.py
python train.py
python live.py
```

## What I learned
- A high test accuracy can be misleading if the data is not varied.
- Data variety and normalization matter more than the model choice.
- ML improves through a loop: test, find the weak spot, add data, retrain.

## Next steps
- Add more gestures (thumbs up, pointing)
- Use gestures to control volume
- Face emotion recognition
