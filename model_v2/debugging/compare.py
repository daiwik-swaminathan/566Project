import librosa
import numpy as np
import matplotlib.pyplot as plt

from model_v2.model2 import preprocess_audio_buffer

# Use this to compare two audio files (USED FOR DEBUGGING PURPOSES)


# Paths to your two files
file1 = r'dataset_scraping\JP\TrackAndField\highlights\14_a097e4c9-0cb8-4e30-803d-5d2a45e1087e.wav'  
file2 = 'live_file/live_capture.wav'

# Load both files
y1, sr1 = librosa.load(file1, sr=44100, mono=True)
y2, sr2 = librosa.load(file2, sr=44100, mono=True)

# # Print basic stats
# print("=== File 1 ===")
# print(f"Shape: {y1.shape}")
# print(f"Min: {np.min(y1)}, Max: {np.max(y1)}")
# print(f"Mean: {np.mean(y1)}, Std: {np.std(y1)}")
# print(f"Energy: {np.sum(y1 ** 2)}")

# print("\n=== File 2 ===")
# print(f"Shape: {y2.shape}")
# print(f"Min: {np.min(y2)}, Max: {np.max(y2)}")
# print(f"Mean: {np.mean(y2)}, Std: {np.std(y2)}")
# print(f"Energy: {np.sum(y2 ** 2)}")

# # Plot waveforms
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(y1)
# plt.title('Training Clip Waveform')
# plt.subplot(2, 1, 2)
# plt.plot(y2)
# plt.title('Live Capture Clip Waveform')
# plt.tight_layout()
# plt.show()

# # Plot spectrograms
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
# librosa.display.specshow(D1, sr=sr1, y_axis='log', x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Training Clip Spectrogram')

# plt.subplot(2, 1, 2)
# D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
# librosa.display.specshow(D2, sr=sr2, y_axis='log', x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Live Capture Clip Spectrogram')

# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Load both .npy files
log_mel_spec1 = np.load('log_mel_spec_1.npy')
log_mel_spec2 = np.load('log_mel_spec_2.npy')

print(f"Spec 1 shape: {log_mel_spec1.shape}")
print(f"Spec 2 shape: {log_mel_spec2.shape}")

# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot first spectrogram
im1 = axes[0].imshow(log_mel_spec1, 
                     aspect='auto', 
                     origin='lower', 
                     cmap='magma')
axes[0].set_title('Log-Mel Spectrogram 1')
axes[0].set_xlabel('Time Frames')
axes[0].set_ylabel('Mel Bands')
fig.colorbar(im1, ax=axes[0], format='%+2.0f dB')

# Plot second spectrogram
im2 = axes[1].imshow(log_mel_spec2, 
                     aspect='auto', 
                     origin='lower', 
                     cmap='magma')
axes[1].set_title('Log-Mel Spectrogram 2')
axes[1].set_xlabel('Time Frames')
axes[1].set_ylabel('Mel Bands')
fig.colorbar(im2, ax=axes[1], format='%+2.0f dB')

plt.tight_layout()
plt.show()



SAMPLE_RATE = 44100
BUFFER_SECONDS = 10
NUM_CHANNELS = 2  # stereo
PREDICT_INTERVAL = 1  # seconds between predictions
THRESHOLD = 0.5  # prediction threshold
LATENT_DIM = 32
n_mfcc = 13  # Number of MFCCs to extract
time_frames = 862  # or whatever your preprocessed spectrogram width is


mfcc1 = preprocess_audio_buffer(y1, sample_rate=SAMPLE_RATE,n_mfcc=n_mfcc, time_frames=time_frames)
mfcc2 = preprocess_audio_buffer(y2, sample_rate=SAMPLE_RATE,n_mfcc=n_mfcc, time_frames=time_frames)

diff = np.abs(mfcc1 - mfcc2)
mean_diff = np.mean(diff)
max_diff = np.max(diff)
print(f"Mean MFCC difference: {mean_diff:.4f}, Max difference: {max_diff:.4f}")
