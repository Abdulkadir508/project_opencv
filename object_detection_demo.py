import cv2 as cv
import numpy as np
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# BALKLEUR GRENZEN IN HSV
ORANGE_MIN = np.array([0, 92, 160], np.uint8)
ORANGE_MAX = np.array([20, 202, 255], np.uint8)
alpha = 0.6  # smoothing

cap = cv.VideoCapture("ping_pong_ball.mov")

print(cv.CAP_PROP_FPS, cap.get(cv.CAP_PROP_FPS))
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    position_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        cap = cv.VideoCapture("ping_pong_ball.mov")
        smoothed_pos = None

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            if not ret:
                break

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
            result = cv.bitwise_and(frame, frame, mask=mask)
            gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (9, 9), 0)

            circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.2, 100,
                                      param1=100, param2=30, minRadius=1, maxRadius=30)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                avg_circle = np.mean(circles[0, :, :2], axis=0)

                if smoothed_pos is None:
                    smoothed_pos = avg_circle
                else:
                    smoothed_pos = alpha * avg_circle + (1 - alpha) * smoothed_pos

                x, y = int(smoothed_pos[0]), int(smoothed_pos[1])
                radius = int(np.mean(circles[0, :, 2]))

                cv.circle(frame, (x, y), 1, (0, 100, 100), 3)
                cv.circle(frame, (x, y), radius, (255, 0, 255), 3)

                self.position_signal.emit(x)

            self.change_pixmap_signal.emit(frame)
            cv.waitKey(30)

        cap.release()

    def start_tracing(self):
        self.running = True
        if not self.isRunning():
            self.start()

    def stop_tracing(self):
        self.running = False
        self.wait()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.x_data = []
        self.y_data = []

    def update_plot(self, new_value):
        self.x_data.append(len(self.x_data))
        self.y_data.append(new_value)
        self.axes.clear()
        self.axes.plot(self.x_data, self.y_data, 'r-')
        self.axes.set_title("Balpositie door de tijd")
        self.draw()


class GUI_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Videoanalyse van Pingpongbal")
        self.setGeometry(100, 100, 800, 700)

        self.layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.canvas = PlotCanvas(self, width=5, height=2)
        self.layout.addWidget(self.canvas)

        self.start_button = QPushButton("Start videoanalyse")
        self.start_button.clicked.connect(self.start_tracing)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_tracing)
        self.layout.addWidget(self.stop_button)

        self.setLayout(self.layout)

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.position_signal.connect(self.update_plot)

    def start_tracing(self):
        self.canvas.x_data.clear()
        self.canvas.y_data.clear()
        self.video_thread.start_tracing()

    def stop_tracing(self):
        self.video_thread.stop_tracing()

    def closeEvent(self, event):
        self.stop_tracing()
        event.accept()

    def update_image(self, cv_img):
        qt_img = self.convert_cv_naar_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_naar_qt(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_format.scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def update_plot(self, x_value):
        self.canvas.update_plot(x_value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GUI_Window()
    window.show()
    sys.exit(app.exec_())
