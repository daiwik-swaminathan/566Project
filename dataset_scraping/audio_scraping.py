import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
import csv
import threading
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.ffmpeg import SystemAudioCapture

import uuid

# === Config ===
SAMPLE_RATE = 44100
BUFFER_SECONDS = 10
NUM_CHANNELS = 2  # stereo

session_start_time = time.time()
clip_count = 0

os.makedirs("highlights", exist_ok=True)
os.makedirs("non_highlights", exist_ok=True)

csv_file = open('clips_metadata.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['filename', 'timestamp', 'label'])

# === Audio capture setup ===
capture = SystemAudioCapture(samplerate=SAMPLE_RATE, channels=NUM_CHANNELS, buffer_seconds=BUFFER_SECONDS)
capture.start()

print("🎧 Capturing system audio... Use the GUI to save clips.")

# === Save function ===
def save_clip(label, status_var):
    global clip_count
    # Generate a uuid
    folder_path = "dataset_scraping"
    name = "JP"
    sport = "TrackAndField"
    random_uuid = uuid.uuid4()
    folder = "highlights" if label == 'highlight' else "non_highlights"
    filename = f"{folder_path}/{name}/{sport}/{folder}/{clip_count}_{random_uuid}.wav"

    capture.save_clip(filename)

    timestamp = time.time() - session_start_time
    csv_writer.writerow([filename, f"{timestamp:.2f}", label])
    csv_file.flush()
    print(f"✅ Saved {label} clip to {filename} (time: {timestamp:.2f}s)")
    clip_count += 1
    status_var.set(f"Saved {label} #{clip_count} at {timestamp:.2f}s")

# === GUI setup ===
def start_gui():
    window = tk.Tk()
    window.title("🎥 Highlight Collector")

    status_var = tk.StringVar()
    status_var.set("Ready")

    # === Matplotlib figure ===
    fig, ax = plt.subplots(figsize=(8, 3))
    line, = ax.plot(np.arange(capture.buffer.size), np.zeros(capture.buffer.size))
    ax.set_ylim(-32768, 32767)
    ax.set_xlim(0, capture.buffer.size)
    ax.set_title('Live Audio Waveform (Stereo Interleaved)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # === Buttons ===
    button_frame = tk.Frame(window)
    button_frame.pack(pady=10)

    highlight_btn = tk.Button(button_frame, text="🎯 Save Highlight (Last 10s)",
                              command=lambda: save_clip('highlight', status_var),
                              width=30, height=2, bg='lightgreen')
    highlight_btn.grid(row=0, column=0, padx=5)

    nonhighlight_btn = tk.Button(button_frame, text="🌿 Save Non-Highlight (Last 10s)",
                                 command=lambda: save_clip('non_highlight', status_var),
                                 width=30, height=2, bg='lightblue')
    nonhighlight_btn.grid(row=0, column=1, padx=5)

    status_label = tk.Label(window, textvariable=status_var, font=("Arial", 12))
    status_label.pack(pady=5)

    # === Update loop ===
    def update_plot():
        with capture.lock:
            buffer_copy = np.copy(capture.buffer)
        line.set_ydata(buffer_copy)
        canvas.draw()
        window.after(20, update_plot)  # ~50 FPS

    window.protocol("WM_DELETE_WINDOW", on_close)
    update_plot()
    window.mainloop()

# === Cleanup ===
def on_close():
    print("\n🛑 Stopping capture...")
    capture.stop()
    csv_file.close()
    plt.close('all')
    os._exit(0)

# === Start GUI (main thread) ===
start_gui()
