import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QThread

ORANGE_MIN = np.array([0, 92, 160], np.uint8)
ORANGE_MAX = np.array([20, 202, 255], np.uint8)
alpha = 0.9

class VideoThread(QThread):

    def run(self):
        cap = cv.VideoCapture(0)
        smoothed_pos = None
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
            result = cv.bitwise_and(frame, frame, mask=mask)
            grayResult = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            blurFrame = cv.GaussianBlur(grayResult, (17, 17), 0)

            circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 100,
                                      param1=100, param2=30, minRadius=1, maxRadius=30)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                chosen = circles[0, 0]
                if smoothed_pos is None:
                    smoothed_pos = chosen[:2].astype(float)
                else:
                    smoothed_pos = alpha * chosen[:2] + (1 - alpha) * smoothed_pos

                cv.circle(frame, (int(smoothed_pos[0]), int(smoothed_pos[1])), 1, (0, 100, 100), 3)
                cv.circle(frame, (int(smoothed_pos[0]), int(smoothed_pos[1])), chosen[2], (255, 0, 255), 3)

            cv.imshow("frame", frame)
            cv.imshow("mask", mask)

            if cv.waitKey(1) == 27 :
                break

        cap.release()
        cv.destroyAllWindows()

    def start_tracing(self):
        self.running = True
        if not self.isRunning():
            self.start()

    def stop_tracing(self):
        self.running = False

class GUI_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection Demo")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()

        self.start_button = QPushButton("Start tracing")
        self.start_button.clicked.connect(self.start_tracing)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop tracing")
        self.stop_button.clicked.connect(self.stop_tracing)
        self.layout.addWidget(self.stop_button)

        self.setLayout(self.layout)
        self.video_thread = VideoThread()
        

    def start_tracing(self):
        self.video_thread.start_tracing()

    def stop_tracing(self):
        self.video_thread.stop_tracing()

if __name__ == "__main__":
    app = QApplication([])
    window = GUI_Window()
    window.show()
    app.exec_()
