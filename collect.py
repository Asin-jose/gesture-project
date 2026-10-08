import cv2, mediapipe as mp, csv, os

hands = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

labels = {ord("1"): "fist", ord("2"): "palm", ord("3"): "peace"}
counts = {"fist": 0, "palm": 0, "peace": 0}
mode = None

new_file = not os.path.exists("data.csv")
f = open("data.csv", "a", newline="")
writer = csv.writer(f)
if new_file:
    header = [f"{a}{i}" for i in range(21) for a in "xyz"] + ["label"]
    writer.writerow(header)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    key = cv2.waitKey(30) & 0xFF

    if key in labels:
        mode = labels[key]
    elif key == ord("0"):
        mode = None

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
        if mode:
            wrist = lm.landmark[0]
            mid = lm.landmark[9]
            size = ((mid.x - wrist.x) ** 2 + (mid.y - wrist.y) ** 2) ** 0.5 or 1
            row = []
            for p in lm.landmark:
                row += [(p.x - wrist.x) / size, (p.y - wrist.y) / size, (p.z - wrist.z) / size]
            row.append(mode)
            writer.writerow(row)
            counts[mode] += 1
    else:
        cv2.putText(frame, "NO HAND SEEN", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    status = f"RECORDING: {mode}" if mode else "NOT recording (press 1, 2 or 3)"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    text = f"fist:{counts['fist']}  palm:{counts['palm']}  peace:{counts['peace']}"
    cv2.putText(frame, text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Collect", frame)
    if key == ord("q"):
        break

f.close()
cap.release()
cv2.destroyAllWindows()