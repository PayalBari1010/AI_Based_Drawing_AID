import numpy as np
import mediapipe as mp
import cv2
from collections import deque

blackpts = [deque(maxlen=256)]
bluepts = [deque(maxlen=256)]
redpts = [deque(maxlen=256)]

black_index = 0
blue_index = 0
red_index = 0

kernel = np.ones((5,5),np.uint8)

colour_list = [(0, 0, 0), (255, 0, 0), (0, 0, 255)]
index = 0
#canvas
canvas = np.zeros((471,636,3)) + 255
canvas = cv2.rectangle(canvas, (40,1), (140,65), (255,255,255), -1) #clear
canvas = cv2.rectangle(canvas, (160, 1), (255, 65), (0, 0, 0), -1)     #black
canvas = cv2.rectangle(canvas, (275,1), (370,65), (255,0,0), -1)  #blue
canvas = cv2.rectangle(canvas, (390,1), (485,65), (0,0,255), -1)  #red

cv2.putText(canvas, "CLEAR", (65, 33), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Canvas', cv2.WINDOW_AUTOSIZE)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    ret, frame = cap.read()
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#output frame
    frame = cv2.rectangle(frame, (40,1), (140,65), (255,255,255), -1)
    frame = cv2.rectangle(frame, (160,1), (255,65), (0,0,0), -1)
    frame = cv2.rectangle(frame, (275,1), (370,65), (255,0,0), -1)
    frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), -1)
    frame = cv2.rectangle(frame, (515,1), (755,65),(255,255,255),-1)

    cv2.putText(frame, "CLEAR", (65, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "AI Based Writing Aid", (560, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        center = (landmarks[8][0],landmarks[8][1])
        thumb = (landmarks[4][0],landmarks[4][1])
        middle = (landmarks[12][0], landmarks[12][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        print(center[1]-thumb[1])
        if (middle[1]-center[1]<20):
            blackpts.append(deque(maxlen=256))
            black_index += 1
            bluepts.append(deque(maxlen=256))
            blue_index += 1
            redpts.append(deque(maxlen=256))
            red_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear Button
                blackpts = [deque(maxlen=256)]
                bluepts = [deque(maxlen=256)]
                redpts = [deque(maxlen=256)]

                black_index = 0
                blue_index = 0
                red_index = 0

                canvas[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    index = 0 # Black
            elif 275 <= center[0] <= 370:
                    index = 0 # Blue
            elif 390 <= center[0] <= 485:
                    index = 0 # Red
        else :
            if index == 0:
                blackpts[black_index].appendleft(center)
            elif index == 1:
                bluepts[blue_index].appendleft(center)
            elif index == 2:
                redpts[red_index].appendleft(center)

    else:
        blackpts.append(deque(maxlen=256))
        black_index += 1
        bluepts.append(deque(maxlen=256))
        blue_index += 1
        redpts.append(deque(maxlen=256))
        red_index += 1

    points = [blackpts, bluepts, redpts]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colour_list[i], 2)
                cv2.line(canvas, points[i][j][k - 1], points[i][j][k], colour_list[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Canvas", canvas)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

       
                

    








