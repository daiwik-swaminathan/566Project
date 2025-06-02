
# === Config ===
import time
import joblib
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


from model_v2.model2 import ConvVAE, LatentClassifier, extract_latents, live_process_vae, preprocess_audio_buffer


# Testing with a single dowloaded audio file (TESTING FOR DEBGGING PURPOSES)


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
checkpoint = torch.load('vae_model_state.pth', map_location=device)
n_mfcc = checkpoint['n_mfcc']
time_frames = checkpoint['time_frames']
model = ConvVAE(n_mfcc=n_mfcc, time=time_frames, latent_dim=LATENT_DIM).to(device)
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

# path, label, debug = r'dataset_scraping\JP\TrackAndField\highlights\14_a097e4c9-0cb8-4e30-803d-5d2a45e1087e.wav', 1 , 1
path, label, debug  = r'live_file\live_capture.wav', 1, 2
y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=BUFFER_SECONDS, mono=True)
y = y / np.max(np.abs(y) + 1e-8)
y = librosa.util.fix_length(y, size=BUFFER_SECONDS * SAMPLE_RATE)

# prediction, new_latent = live_process_vae(y, model, latent_clf, device, n_mfcc=n_mfcc, time_frames=time_frames, threshold=0.5)

# print(f"Prediction: {prediction}, New Latent: {new_latent}")

mfcc = preprocess_audio_buffer(y, sample_rate=SAMPLE_RATE,n_mfcc=n_mfcc, time_frames=time_frames, debug=debug)
input_tensor = torch.tensor(mfcc[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
dummy_label_tensor  = torch.tensor([-1], dtype=torch.float32)  # Dummy label for TensorDataset
dataset = TensorDataset(input_tensor, dummy_label_tensor)
dataloader = DataLoader(dataset, batch_size=1)
extracted_latents = extract_latents(model, dataloader, device)
new_latent = extracted_latents[0]

print (f"mfcc: {mfcc}")

print(f" New Latent: {new_latent}")


with torch.no_grad():
    for inputs, label in dataloader:
        mu, _ = model.encode(inputs.to(device))
        print(mu)


import matplotlib.pyplot as plt

# Ensure latent shape
if new_latent.ndim == 1:
    new_latent = new_latent.reshape(1, -1)

# Transform to 2D
new_latent_2d = pca.transform(new_latent)
training_latents_2d = pca.transform(training_latents)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(training_latents_2d[:, 0], training_latents_2d[:, 1],
            c=training_labels, cmap='coolwarm', alpha=0.5, label='Training Latents')
plt.scatter(new_latent_2d[:, 0], new_latent_2d[:, 1],
            c='yellow', edgecolor='black', s=150, marker='*', label='Live Segment')

plt.title('Latent Space (PCA Projection)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

