import cv2, mediapipe as mp, pickle, numpy as np
from collections import deque, Counter

model = pickle.load(open("model.pkl", "rb"))
hands = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
history = deque(maxlen=10)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
        wrist = lm.landmark[0]
        mid = lm.landmark[9]
        size = ((mid.x - wrist.x) ** 2 + (mid.y - wrist.y) ** 2) ** 0.5 or 1
        row = []
        for p in lm.landmark:
            row += [(p.x - wrist.x) / size, (p.y - wrist.y) / size, (p.z - wrist.z) / size]

        probs = model.predict_proba(np.array(row).reshape(1, -1))[0]
        best = probs.argmax()
        conf = probs[best]
        history.append(model.classes_[best] if conf >= 0.7 else "unsure")

        shown = Counter(history).most_common(1)[0][0]
        text = f"{shown.upper()} {conf * 100:.0f}%"
        cv2.putText(frame, text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    else:
        history.clear()

    cv2.imshow("Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()