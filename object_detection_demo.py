import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)  
prevCircles = None
dist = lambda x1, y1, x2, y2: ((x1 - x2) ** 2 + (y1 - y2) ** 2)

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
        chosen = None
        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            if prevCircles is not None:
                if dist(chosen[0], chosen[1], i[0], i[1]) > dist(prevCircles[0], prevCircles[1], i[0], i[1]):
                    chosen = i
            cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
            cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255, 0, 255),3)
            prevCircles = chosen
    cv.imshow('Circles', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()  
cv.destroyAllWindows()