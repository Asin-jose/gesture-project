import cv2, mediapipe as mp

hands = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
    cv2.imshow("Hand", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()