import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
import threading
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.ffmpeg import SystemAudioCapture
import torch
from model_v2.model import AudioCNN, preprocess_audio_for_model
from model_v2.model2 import ConvVAE, live_process_vae

# === Config ===
SAMPLE_RATE = 44100
BUFFER_SECONDS = 10
NUM_CHANNELS = 2  # stereo
PREDICT_INTERVAL = 1  # seconds between predictions
THRESHOLD = 0.5  # prediction threshold


# === Load your trained PyTorch model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## Load model 1 - CNN based
# n_mels = 128
# time_frames = 862  # or whatever your preprocessed spectrogram width is
# model = AudioCNN(n_mels, time_frames)
# model.load_state_dict(torch.load('model_v2/model_state.pth', map_location=device))
# model.eval()

## Load model 2 - VAE based
checkpoint = torch.load('vae_model_state.pth', map_location=device)
n_mfcc = checkpoint['n_mfcc']
time_frames = checkpoint['time_frames']
model = ConvVAE(n_mfcc=n_mfcc, time=time_frames, latent_dim=32).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

session_start_time = time.time()

# === Audio capture setup ===
capture = SystemAudioCapture(samplerate=SAMPLE_RATE, channels=NUM_CHANNELS, buffer_seconds=BUFFER_SECONDS)
capture.start()
print("🎧 Capturing system audio... Running live prediction...")

# === GUI setup ===
def start_gui():
    window = tk.Tk()
    window.title("🎥 Highlight Detector (PyTorch)")

    status_var = tk.StringVar()
    status_var.set("Ready")

    # === Matplotlib figure ===
    fig, ax = plt.subplots(figsize=(8, 3))
    line, = ax.plot(np.arange(capture.buffer.size), np.zeros(capture.buffer.size))
    ax.set_ylim(-32768, 32767)
    ax.set_xlim(0, capture.buffer.size)
    ax.set_title('Live Audio Waveform')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # === Status label ===
    status_label = tk.Label(window, textvariable=status_var, font=("Arial", 14), fg='red')
    status_label.pack(pady=10)

    # === Update loop ===
    def update_loop():
        # Update waveform
        with capture.lock:
            buffer_copy = np.copy(capture.buffer)
        line.set_ydata(buffer_copy)
        canvas.draw()

        # Run prediction every second
        predict_on_buffer(buffer_copy, status_var, model, device, session_start_time)

        window.after(PREDICT_INTERVAL * 1000, update_loop)  # milliseconds

    window.protocol("WM_DELETE_WINDOW", on_close)
    update_loop()
    window.mainloop()

# === Prediction logic ===
def predict_on_buffer(buffer_copy, status_var, model, device, session_start_time, threshold=0.5):
    # Collapse stereo to mono
    mono_audio = np.mean(buffer_copy.reshape(-1, NUM_CHANNELS), axis=1)
    mono_audio = mono_audio / (np.max(np.abs(mono_audio)) + 1e-8)

    # Preprocess directly from raw audio
    # Using model 1
    # input_array = preprocess_audio_for_model(mono_audio, sample_rate=SAMPLE_RATE, n_mels=n_mels, duration=BUFFER_SECONDS, from_file=False, expected_time_frames=time_frames)
    # input_tensor = torch.tensor(input_array[np.newaxis,np.newaxis, :, :], dtype=torch.float32).to(device)

    # with torch.no_grad():
    #     pred = model(input_tensor)
    #     if isinstance(pred, tuple):
    #         pred = pred[0]
    #     prob = pred.item()

    # if prob > threshold:
    #     timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time() - session_start_time))
    #     print(f"🔥 Highlight detected at {timestamp} (prob={prob:.2f})")
    #     status_var.set(f"🔥 Highlight detected at {timestamp}")
    # else:
    #     status_var.set(f"No highlight (prob={prob:.2f})")
    
 
    # Using model 2
    prediction = live_process_vae(mono_audio, model, device, n_mfcc=n_mfcc, time_frames=time_frames, threshold=threshold)
    print(prediction)
    prob = prediction[0][0]
    pred = prediction[0][1]
    if pred == 1:
        timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time() - session_start_time))
        print(f"🔥 Highlight detected at {timestamp} (prob={prob:.2f})")
        status_var.set(f"🔥 Highlight detected at {timestamp}")
    else:
        status_var.set(f"No highlight (prob={prob:.2f})")




# === Cleanup ===
def on_close():
    print("\n🛑 Stopping capture...")
    capture.stop()
    plt.close('all')
    os._exit(0)

def evaluate_highlight_directory(folder_path):
    total_files = 0
    correctly_predicted = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            total_files += 1
            filepath = os.path.join(folder_path, filename)

            input_array = preprocess_audio_for_model(filepath)
            input_tensor = torch.tensor(input_array[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)

            with torch.no_grad():
                pred = model(input_tensor)
                prob = pred.item()

            if prob > 0.5:
                correctly_predicted += 1
                print(f"{filename}: ✅ Highlight (prob={prob:.2f})")
            else:
                print(f"{filename}: ❌ Missed (prob={prob:.2f})")

    print(f"\nSummary: {correctly_predicted}/{total_files} correctly classified as highlights ({(correctly_predicted / total_files) * 100:.2f}%)")

# Example usage
# evaluate_highlight_directory('./NBAHighlightsWAV')
# === Start GUI (main thread) ===
start_gui()
