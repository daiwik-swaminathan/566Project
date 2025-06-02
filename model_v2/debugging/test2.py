import sys
import os
import threading
import time
import numpy as np
import torch
import librosa
import joblib
from PyQt5 import QtWidgets, QtGui, QtCore
import pygetwindow as gw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from model_v2.model3 import ConvDAE, live_process_dae, LatentClassifier
from utils.ffmpeg import SystemAudioCapture

# === Config ===
SAMPLE_RATE = 44100
BUFFER_SECONDS = 10
NUM_CHANNELS = 2
THRESHOLD = 0.5
LATENT_DIM = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('vae_model_state.pth', map_location=device)
n_mfcc = checkpoint['n_mfcc']
time_frames = checkpoint['time_frames']

model = ConvDAE(n_mfcc=n_mfcc, time=time_frames, latent_dim=LATENT_DIM).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

latent_clf = LatentClassifier(latent_dim=LATENT_DIM).to(device)
latent_clf.load_state_dict(checkpoint['latent_clf_state_dict'])

data = np.load('./model_v2/training_latents.npz')
training_latents = data['latents']
data.close()

pca = joblib.load('./model_v2/trained_pca.pkl')
recent_latents = []
session_start_time = time.time()


class RedDotOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setFixedSize(30, 30)
        self.hide()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 180)))  # semi-transparent red
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(0, 0, 30, 30)

    def move_to(self, x, y):
        self.move(x, y)
        self.show()


class HighlightDetectorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎥 Highlight Detector (PyQt5)")

        self.capture = SystemAudioCapture(samplerate=SAMPLE_RATE, channels=NUM_CHANNELS, buffer_seconds=BUFFER_SECONDS)
        self.capture.start()

        self.status_label = QtWidgets.QLabel("Ready", self)
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.status_label)

        self.overlay = RedDotOverlay()
        self.start_prediction_loop()

    def start_prediction_loop(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.run_prediction)
        self.timer.start(1000)  # every second

    def run_prediction(self):
        with self.capture.lock:
            buffer_copy = np.copy(self.capture.buffer)

        self.capture.save_clip("live_file/live_capture.wav")
        y, sr = librosa.load("live_file/live_capture.wav", sr=SAMPLE_RATE, mono=True, duration=BUFFER_SECONDS)

        prediction, new_latent = live_process_dae(y, model, latent_clf, device, n_mfcc=n_mfcc, time_frames=time_frames, threshold=THRESHOLD)
        prob = prediction[0][0]
        pred = prediction[0][1]

        if pred == 1:
            timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time() - session_start_time))
            self.status_label.setText(f"🔥 Highlight detected at {timestamp} (prob={prob:.2f})")
            self.show_overlay()
        else:
            self.status_label.setText(f"No highlight (prob={prob:.2f})")
            self.overlay.hide()

    def show_overlay(self):
        # Find window coordinates (here using a fixed window like Chrome)
        try:
            win = next(w for w in gw.getWindowsWithTitle("Google Chrome") if w.title)
            x, y = win.left, win.top
            w, h = win.width, win.height
            # Example: position near the bottom middle (adjust offsets)
            offset_x = w // 2
            offset_y = int(h * 0.9)
            self.overlay.move_to(x + offset_x, y + offset_y)
        except StopIteration:
            print("⚠ Could not find target window")
            self.overlay.hide()

    def closeEvent(self, event):
        print("🛑 Stopping capture...")
        self.capture.stop()
        self.overlay.close()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = HighlightDetectorApp()
    window.resize(400, 200)
    window.show()
    sys.exit(app.exec_())
