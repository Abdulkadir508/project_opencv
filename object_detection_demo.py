import sys
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# HSV-kleur voor het detecteren van de oranje pingpongbal
ORANGE_MIN = np.array([0, 92, 160], np.uint8)
ORANGE_MAX = np.array([20, 202, 255], np.uint8)
SMOOTHING_ALPHA = 0.6


class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    x_position_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.running = False
        self.video_path = None

    def set_video_path(self, path):
        self.video_path = path

    def run(self):
        if not self.video_path:
            return

        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Kan video niet openen.")
            return

        smoothed_position = None

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
            result = cv.bitwise_and(frame, frame, mask=mask)

            gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (9, 9), 0)

            circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.2, 100,
                                      param1=100, param2=30, minRadius=1, maxRadius=30)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                avg_pos = np.mean(circles[0, :, :2], axis=0)

                smoothed_position = avg_pos if smoothed_position is None else \
                    SMOOTHING_ALPHA * avg_pos + (1 - SMOOTHING_ALPHA) * smoothed_position

                x, y = map(int, smoothed_position)
                radius = int(np.mean(circles[0, :, 2]))

                cv.circle(frame, (x, y), 1, (0, 100, 100), 3)
                cv.circle(frame, (x, y), radius, (255, 0, 255), 3)

                self.x_position_signal.emit(x)

            self.frame_signal.emit(frame)
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

    def update_plot(self, value):
        self.x_data.append(len(self.x_data))
        self.y_data.append(value)
        self.axes.clear()
        self.axes.plot(self.x_data, self.y_data, 'r-')
        self.axes.set_title("Balpositie door de tijd")
        self.draw()


class VideoAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pingpongbal Videoanalyse")
        self.setGeometry(100, 100, 800, 700)

        self.video_thread = VideoThread()
        self.video_thread.frame_signal.connect(self.update_image)
        self.video_thread.x_position_signal.connect(self.update_plot)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.canvas = PlotCanvas(self, width=5, height=2)
        layout.addWidget(self.canvas)

        self.select_button = QPushButton("Selecteer video")
        self.select_button.clicked.connect(self.select_video)
        layout.addWidget(self.select_button)

        self.start_button = QPushButton("Start Analyse")
        self.start_button.clicked.connect(self.start_tracing)
        self.start_button.setEnabled(False)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_tracing)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Kies een videobestand", "", "Video's (*.mp4 *.mov *.avi *.mkv)"
        )
        if file_path:
            self.video_thread.set_video_path(file_path)
            self.start_button.setEnabled(True)
            QMessageBox.information(self, "Video geselecteerd", f"Video: {file_path}")

    def start_tracing(self):
        if not self.video_thread.video_path:
            QMessageBox.warning(self, "Geen video", "Selecteer eerst een video.")
            return

        self.canvas.x_data.clear()
        self.canvas.y_data.clear()
        self.video_thread.start_tracing()

    def stop_tracing(self):
        self.video_thread.stop_tracing()

    def closeEvent(self, event):
        self.stop_tracing()
        event.accept()

    def update_image(self, frame):
        qt_img = self.convert_frame_to_qpixmap(frame)
        self.image_label.setPixmap(qt_img)

    def update_plot(self, x_value):
        self.canvas.update_plot(x_value)

    def convert_frame_to_qpixmap(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img.scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnalyzer()
    window.show()
    sys.exit(app.exec_())
