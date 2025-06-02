import joblib
import librosa
import numpy as np
import matplotlib

from model_v2.model3 import ConvDAE, live_process_dae
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
from model_v2.model2 import ConvVAE, LatentClassifier, extract_latents, live_process_vae, preprocess_audio_buffer

# === Config ===
SAMPLE_RATE = 44100
BUFFER_SECONDS = 10
NUM_CHANNELS = 2  # stereo
PREDICT_INTERVAL = 1  # seconds between predictions
THRESHOLD = 0.5  # prediction threshold
LATENT_DIM = 32


# === Load your trained PyTorch model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## Load model 1 - CNN based
# n_mels = 128
# time_frames = 862  # or whatever your preprocessed spectrogram width is
# model = AudioCNN(n_mels, time_frames)
# model.load_state_dict(torch.load('model_v2/model_state.pth', map_location=device))
# model.eval()

## Load model 2 - VAE based
# checkpoint = torch.load('vae_model_state.pth', map_location=device)
# n_mfcc = checkpoint['n_mfcc']
# time_frames = checkpoint['time_frames']
# model = ConvVAE(n_mfcc=n_mfcc, time=time_frames, latent_dim=LATENT_DIM).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

## Load model 3
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
training_labels = data['labels']  
data.close()

global pca
pca = joblib.load('./model_v2/trained_pca.pkl')

global recent_latents
recent_latents = []

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

    # === Waveform figure ===
    fig_wave, ax_wave = plt.subplots(figsize=(8, 3))
    line_wave, = ax_wave.plot(np.arange(capture.buffer.size), np.zeros(capture.buffer.size))
    ax_wave.set_ylim(-32768, 32767)
    ax_wave.set_xlim(0, capture.buffer.size)
    ax_wave.set_title('Live Audio Waveform')
    ax_wave.set_xlabel('Sample')
    ax_wave.set_ylabel('Amplitude')

    canvas_wave = FigureCanvasTkAgg(fig_wave, master=window)
    canvas_wave.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # === Latent space figure ===
    fig_latent, ax_latent = plt.subplots(figsize=(5, 5))
    canvas_latent = FigureCanvasTkAgg(fig_latent, master=window)
    canvas_latent.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # === Status label ===
    status_label = tk.Label(window, textvariable=status_var, font=("Arial", 14), fg='red')
    status_label.pack(pady=10)

    # === Update loop ===
    def update_loop():
        with capture.lock:
            buffer_copy = np.copy(capture.buffer)
            
        capture.save_clip("live_file/live_capture.wav") 
        # Update waveform plot
        line_wave.set_ydata(buffer_copy)
        canvas_wave.draw()

        # Run prediction and update latent plot
        predict_on_buffer(buffer_copy, status_var, model, latent_clf, device, session_start_time, ax_latent, canvas_latent)

        window.after(PREDICT_INTERVAL * 1000, update_loop)

    window.protocol("WM_DELETE_WINDOW", on_close)
    update_loop()
    window.mainloop()

# === Prediction logic ===
def predict_on_buffer(buffer_copy, status_var, model, latent_clf, device, session_start_time, ax_latent, canvas_latent, threshold=0.6):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # save_video_latents(buffer_copy)

    # Collapse stereo to mono
    # mono_audio = np.mean(buffer_copy.reshape(-1, NUM_CHANNELS), axis=1)

    # OPTIONAL: Placeholder for denoising (apply here if desired)
    # mono_audio = denoise_audio(mono_audio)

    live_file = "live_file/live_capture.wav"
    y, sr = librosa.load(live_file, sr=SAMPLE_RATE, mono=True, duration=BUFFER_SECONDS)
    # mfcc = preprocess_audio_buffer(y, sample_rate=SAMPLE_RATE, n_mfcc=26, time_frames=862)
    # print(mfcc.shape)
    # print(mfcc)

    # Run prediction (model 2)
    # prediction, new_latent = live_process_vae(y, model, latent_clf, device, n_mfcc=n_mfcc, time_frames=time_frames, threshold=threshold)
    # prob = prediction[0][0]
    # pred = prediction[0][1]

    # Run prediction (model 3)
    prediction, new_latent = live_process_dae(y, model, latent_clf, device, n_mfcc=n_mfcc, time_frames=time_frames, threshold=threshold)
    prob = prediction[0][0]
    pred = prediction[0][1]
 
    if pred == 1:
        timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time() - session_start_time))
        print(f"🔥 Highlight detected at {timestamp} (prob={prob:.2f})")
        status_var.set(f"🔥 Highlight detected at {timestamp}")
    else:
        status_var.set(f"No highlight (prob={prob:.2f})")

    # Apply PCA if needed
    global pca  # reuse fitted PCA
     # Ensure shape (1, latent_dim)
    if new_latent.ndim == 1:
        new_latent = new_latent.reshape(1, -1)

    # Transform to 2D
    new_latent_2d = pca.transform(new_latent)
    training_latents_2d = pca.transform(training_latents)

    global recent_latents
    # Update recent latents list
    recent_latents.append(new_latent_2d[0])  # store as flat (2,)
    if len(recent_latents) > 10:
        recent_latents.pop(0)  # keep only last 10

    # Convert to array for plotting
    recent_latents_array = np.array(recent_latents)

    # Update latent plot
    ax_latent.clear()

    # Keep axis limits fixed
    # ax_latent.set_xlim(-2.5, 2.5)
    # ax_latent.set_ylim(-2, 2)


    # Plot recent points
    # Plot training latent background
    ax_latent.scatter(training_latents_2d[:, 0], training_latents_2d[:, 1],
                      c='lightgray', alpha=0.3, s=30, label='Training Latents')


    ax_latent.scatter(recent_latents_array[:, 0], recent_latents_array[:, 1], c='blue', alpha=0.6, s=50, label='Recent Points')

    # Highlight the most recent point
    ax_latent.scatter(new_latent_2d[:, 0], new_latent_2d[:, 1], c='yellow', edgecolor='black', s=150, marker='*', label='Live Segment')

    ax_latent.set_title('Latent Space (Live)')
    ax_latent.legend()
    canvas_latent.draw()


def save_video_latents(buffer_copy):
    mono_audio = np.mean(buffer_copy.reshape(-1, NUM_CHANNELS), axis=1)
    # Save the audio segment as a .wav file
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time() - session_start_time))
    filename = f"video_segment_{timestamp}.wav"
    output_path = os.path.join("video_segments", filename)
    os.makedirs("video_segments", exist_ok=True)
    capture.save_clip(output_path)

    # Save the latent representation
    prediction, new_latent = live_process_vae(mono_audio, model, latent_clf, device, n_mfcc=n_mfcc, time_frames=time_frames, threshold=10)
    new_latent_2d = pca.transform(new_latent.reshape(1, -1))
    latent_filename = f"latent_segment_{timestamp}.npy"
    latent_output_path = os.path.join("video_segments", latent_filename)
    np.save(latent_output_path, new_latent_2d)

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
