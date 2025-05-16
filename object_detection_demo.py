import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture(0)
prev_pos = None
smoothed_pos = None
alpha = 0.9
dist = lambda x1, y1, x2, y2: ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17, 17), 0)

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 100,
                               param1=100, param2=30, minRadius=5, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = circles[0, 0]  

        
        if smoothed_pos is None:
            smoothed_pos = chosen[:2].astype(float)
        else:
            smoothed_pos = alpha * chosen[:2] + (1 - alpha) * smoothed_pos

        
        cv.circle(frame, (int(smoothed_pos[0]), int(smoothed_pos[1])), 1, (0, 100, 100), 3)
        cv.circle(frame, (int(smoothed_pos[0]), int(smoothed_pos[1])), chosen[2], (255, 0, 255), 3)

        
        

    cv.imshow('Circles', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
