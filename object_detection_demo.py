import sys
import cv2 as cv
import numpy as np
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Zorg dat OpenCV en numpy geïmporteerd zijn
# Definieer HSV-waarden voor oranje kleur (voor kleurdetectie)
ORANGE_MIN = np.array([0, 92, 160], np.uint8)
ORANGE_MAX = np.array([20, 202, 255], np.uint8)

# Factor voor smoothing van de positie (low-pass filter)
SMOOTHING_ALPHA = 0.6


# Thread die video verwerkt zonder de GUI te blokkeren
class VideoThread(QThread):
    # Signal om een frame te verzenden naar de GUI (numpy array)
    frame_signal = pyqtSignal(np.ndarray)
    # Signal om positie (x, y) te verzenden
    position_signal = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.video_path = None      # Pad naar video
        self.detection_mode = "Cirkel"  # Detectiemodus (standaard cirkel)
        self.cap = None             # Video capture object
        self.frame_idx = 0          # Huidige frame index
        self.total_frames = 0       # Totaal aantal frames in video
        self.smoothed_position = None  # Voor positie smoothing
        self.running = False        # Flag om thread te stoppen

    def set_video_path(self, path):
        """Stel het videopad in en initialiseert capture."""
        self.video_path = path
        if self.cap:
            self.cap.release()  # Release vorige capture indien aanwezig
        self.cap = cv.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("Kan video niet openen.")
            self.cap = None
            return
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))  # Lees totaal frames
        self.frame_idx = 0
        self.smoothed_position = None  # Reset smoothing bij nieuwe video

    def set_detection_mode(self, mode):
        """Wijzig de detectiemodus (cirkel of vierkant)."""
        self.detection_mode = mode
        self.smoothed_position = None  # Reset smoothing bij moduswijziging

    def run(self):
        """Hoofdthread-loop, leest continu frames totdat gestopt."""
        self.running = True
        while self.running and self.cap and self.frame_idx < self.total_frames:
            self.get_next_frame()
            self.msleep(30)  # Pauze (30ms) om CPU niet te overbelasten

    def stop(self):
        """Stop de thread."""
        self.running = False

    def get_next_frame(self):
        """Lees het volgende frame, voer detectie uit en stuur resultaten."""
        if self.cap is None or self.frame_idx >= self.total_frames:
            return

        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.frame_idx)  # Ga naar huidig frame
        ret, frame = self.cap.read()  # Lees frame
        if not ret:
            print("Kan frame niet lezen.")
            return

        self.frame_idx += 1

        # Draai frame 90 graden met de klok mee (voor juiste oriëntatie)
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

        # Converteer frame naar HSV en filter op oranje kleur
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
        result = cv.bitwise_and(frame, frame, mask=mask)

        # Maak afbeelding grijswaarden en blur voor ruisvermindering
        gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (9, 9), 0)

        x = y = None  # Initialiseer positie variabelen

        if self.detection_mode == "Cirkel":
            # Detecteer cirkels met HoughCircles
            circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.2, 100,
                                      param1=100, param2=30, minRadius=1, maxRadius=30)
            if circles is not None:
                # Rond de gevonden cirkels af en converteer naar integers
                circles = np.uint16(np.around(circles))
                # Bereken gemiddelde positie van alle gedetecteerde cirkels
                avg_pos = np.mean(circles[0, :, :2], axis=0)
                # Pas smoothing toe op positie
                self.smoothed_position = avg_pos if self.smoothed_position is None else \
                    SMOOTHING_ALPHA * avg_pos + (1 - SMOOTHING_ALPHA) * self.smoothed_position
                x, y = map(int, self.smoothed_position)
                radius = int(np.mean(circles[0, :, 2]))
                # Teken gedetecteerde cirkel op het frame
                cv.circle(frame, (x, y), radius, (255, 0, 255), 3)
        else:
            # Detecteer contouren als alternatief (vierkant/detectie via contour)
            contours, _ = cv.findContours(blur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                # Vind grootste contour op basis van oppervlakte
                largest = max(contours, key=cv.contourArea)
                x_, y_, w, h = cv.boundingRect(largest)
                center_x = x_ + w // 2
                center_y = y_ + h // 2
                pos = np.array([center_x, center_y])
                # Pas smoothing toe
                self.smoothed_position = pos if self.smoothed_position is None else \
                    SMOOTHING_ALPHA * pos + (1 - SMOOTHING_ALPHA) * self.smoothed_position
                x, y = map(int, self.smoothed_position)
                # Teken rechthoek op het frame
                cv.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)

        # Zend het bewerkte frame naar de GUI
        self.frame_signal.emit(frame)

        # Zend ook positie als die beschikbaar is
        if x is not None and y is not None:
            self.position_signal.emit(x, y)

class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data
    def data(self, index, role):
         if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]
    def rowCount(self, index):
        return len(self._data)
    def columnCount(self, index):
        return len(self._data[0]) if self._data else 0


# Canvas voor live plotten van X en Y posities over tijd
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        # Maak matplotlib figure met 2 subplots (voor X en Y)
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = fig.add_subplot(211)  # Bovenste subplot (X-positie)
        self.ax2 = fig.add_subplot(212)  # Onderste subplot (Y-positie)
        super().__init__(fig)
        self.x_data = []  # Frame nummers
        self.x_pos = []   # X posities
        self.y_pos = []   # Y posities
        self.skip_rate = 1  # Frames overslaan bij plotten (voor performance)
        self.df = pd.DataFrame({
            'Frame': self.x_data,
            'x_pos': self.x_pos,
            'y_pos': self.y_pos
        })
        self.df.to_csv('axis_data.csv', index=False)  # Opslaan als CSV bestand
        print('Data is opgeslagen in axis_data.csv')

    def update_plot(self, x, y):
        """Voeg nieuwe positie toe en update grafiek."""
        # Update alleen bij skip_rate (voor minder frequent plotten)
        if len(self.x_data) == 0 or (len(self.x_data) + 1) % self.skip_rate == 0:
            self.x_data.append(len(self.x_data))
            self.x_pos.append(x)
            self.y_pos.append(y)

            # Wis en herplot X-positie
            self.ax1.clear()
            self.ax1.plot(self.x_data, self.x_pos, 'r-', linewidth=2)
            self.ax1.set_title("X-positie over tijd")
            self.ax1.set_xlabel("Frame")
            self.ax1.set_ylabel("X")

            # Wis en herplot Y-positie
            self.ax2.clear()
            self.ax2.plot(self.x_data, self.y_pos, 'b-', linewidth=2)
            self.ax2.set_title("Y-positie over tijd")
            self.ax2.set_xlabel("Frame")
            self.ax2.set_ylabel("Y")

            self.tight_layout()  # Zorg dat layout netjes is
            self.draw()          # Teken grafiek opnieuw

    def tight_layout(self):
        """Pas layout aan zodat labels niet overlappen."""
        self.figure.tight_layout()

    def set_skip_rate(self, rate):
        """Stel het aantal frames tussen grafiekupdates in."""
        self.skip_rate = rate

    def export_to_csv(self, filename):
        filename = "positiegegevens.csv" if filename is None else filename
        """Exporteer alle positiegegevens naar CSV bestand."""
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame', 'X', 'Y'])
            for i in range(len(self.x_data)):
                writer.writerow([self.x_data[i], self.x_pos[i], self.y_pos[i]])
        print(f"Gegevens geëxporteerd naar {filename}")

# Hoofd GUI klasse
class VideoAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pingpongbal Videoanalyse")  # Venstertitel
        self.setGeometry(100, 100, 1000, 900)             # Venstergrootte
        self.video_thread = VideoThread()                  # Maak video thread aan
        # Connecteer signals van thread met GUI methoden
        self.video_thread.frame_signal.connect(self.update_image)
        self.video_thread.position_signal.connect(self.update_plot)

        self.init_ui()  # Setup interface

    def init_ui(self):
        """Maak en organiseer GUI elementen."""
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        """Layout voor bovenste sectie met tabel bron: https://www.pythonguis.com/tutorials/qtableview-modelviews-numpy-pandas/."""
        self.table = QtWidgets.QTableView()
        data = [["Frame", "X-positie", "Y-positie"],
                ["-", "-", "-"],
                ["-", "-", "-"],
                ["-", "-", "-"],
                ["-", "-", "-"]]
                
        self.table_model = TableModel(data)
        self.table.setModel(self.table_model)
        self.table.setFixedHeight(200)
        top_layout.addWidget(self.table)
        # Label voor het tonen van video frames
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(self.image_label)

        # Layout voor knoppen en instellingen
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)

        # Knop: video selecteren
        self.select_video_btn = QPushButton("Selecteer Video")
        self.select_video_btn.clicked.connect(self.select_video)
        self.select_video_btn.setFixedHeight(40)
        button_layout.addWidget(self.select_video_btn)

        # Knop: start analyse
        self.start_btn = QPushButton("Start Analyse")
        self.start_btn.setFixedHeight(40)
        self.start_btn.setEnabled(False)  # Disabled totdat video geselecteerd is
        self.start_btn.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.start_btn)

        # Knop: stop analyse
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.setEnabled(False)  # Aanvankelijk uit
        self.stop_btn.clicked.connect(self.stop_analysis)
        button_layout.addWidget(self.stop_btn)

        # Knop: volgend frame tonen
        self.next_frame_btn = QPushButton("Volgend frame")
        self.next_frame_btn.setFixedHeight(40)
        self.next_frame_btn.setEnabled(False)
        self.next_frame_btn.clicked.connect(self.next_frame)
        button_layout.addWidget(self.next_frame_btn)

        # Knop: reset analyse
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFixedHeight(40)
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_analysis)
        button_layout.addWidget(self.reset_btn)

        # Dropdown menu voor detectiemodus
        self.detection_mode_dropdown = QComboBox()
        self.detection_mode_dropdown.addItems(["Cirkel", "Vierkant"])
        # Verander modus in thread bij selectie
        self.detection_mode_dropdown.currentTextChanged.connect(
            self.video_thread.set_detection_mode
        )
        button_layout.addWidget(QLabel("Detectiemodus:"))
        button_layout.addWidget(self.detection_mode_dropdown)

        # Spinbox voor frames skippen bij plotten
        self.skip_box = QSpinBox()
        self.skip_box.setRange(1, 20)
        self.skip_box.setValue(1)
        self.skip_box.valueChanged.connect(self.update_skip_rate)
        button_layout.addWidget(QLabel("Aantal frames tussen grafiekpunten:"))
        button_layout.addWidget(self.skip_box)

        button_layout.addStretch()  # Vul resterende ruimte

        top_layout.addLayout(button_layout)
        main_layout.addLayout(top_layout)
        main_layout.addSpacing(20)

        # Voeg plotcanvas toe voor live grafiek
        self.canvas = PlotCanvas(self, width=5, height=6)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

    def update_skip_rate(self):
        """Update de skip rate voor grafiekpunten uit de spinbox."""
        self.canvas.set_skip_rate(self.skip_box.value())

    def select_video(self):
        """Open dialoog om video te selecteren en initialiseert analyse."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Kies videobestand", "", "Video's (*.mp4 *.mov *.avi *.mkv)"
        )
        if file_path:
            self.video_thread.set_video_path(file_path)
            # Activeer knoppen
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.next_frame_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)

            # Wis oude data uit de grafiek
            self.canvas.x_data.clear()
            self.canvas.x_pos.clear()
            self.canvas.y_pos.clear()

            QMessageBox.information(self, "Video geselecteerd", f"Geselecteerd: {file_path}")
      
    def start_analysis(self):
        """Start de video analyse thread."""
        if not self.video_thread.isRunning():
            self.video_thread.start()
        # Knoppen aan-/uitzetten voor juiste status
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.next_frame_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

    def stop_analysis(self):
        """Stop de video analyse thread."""
        self.video_thread.stop()
        self.video_thread.wait()  # Wacht totdat thread klaar is
        # Knoppen resetten
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.next_frame_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

    def reset_analysis(self):
        """Reset alles naar beginstatus."""
        self.video_thread.stop()
        self.video_thread.wait()
        self.video_thread.frame_idx = 0
        self.video_thread.smoothed_position = None
        if self.video_thread.cap:
            self.video_thread.cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        # Wis alle data en grafiek
        self.canvas.x_data.clear()
        self.canvas.x_pos.clear()
        self.canvas.y_pos.clear()
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.draw()

        # Wis beeld
        self.image_label.clear()

        # Knoppen status resetten
        self.start_btn.setEnabled(True)
        self.next_frame_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

    def next_frame(self):
        """Vraag het volgende frame aan van de thread (handmatige frame stap)."""
        self.video_thread.get_next_frame()

    def update_image(self, frame):
        """Update het QLabel met het nieuw ontvangen frame."""
        qt_img = self.convert_frame_to_qpixmap(frame)
        self.image_label.setPixmap(qt_img)

    def update_plot(self, x_value, y_value):
        """Update de plot met de nieuwe x,y positie."""
        self.canvas.update_plot(x_value, y_value)

    def convert_frame_to_qpixmap(self, frame):
        """Converteer een OpenCV BGR frame naar QPixmap voor weergave."""
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # OpenCV gebruikt BGR, Qt RGB
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnalyzer()
    window.show()
    sys.exit(app.exec_())
