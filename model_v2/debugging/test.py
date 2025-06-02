import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor

class FloatingOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(200, 200)
        self.move(1000, 900)  # adjust to position over your player bar

        self.show_dot = False

        # Timer to toggle dot for testing
        self.timer = QTimer()
        self.timer.timeout.connect(self.toggle_dot)
        self.timer.start(1000)  # every second

    def toggle_dot(self):
        self.show_dot = not self.show_dot
        self.update()

    def paintEvent(self, event):
        if self.show_dot:
            painter = QPainter(self)
            painter.setBrush(QColor(255, 0, 0, 180))  # semi-transparent red
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(90, 90, 20, 20)  # draw a small circle at center

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = FloatingOverlay()
    overlay.show()
    sys.exit(app.exec_())
