import os
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# === Model: Denoising Autoencoder ===
class ConvDAE(nn.Module):
    def __init__(self, n_mfcc, time, latent_dim=32):
        super(ConvDAE, self).__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_mfcc, time)
            dummy_encoded = self.encoder_cnn(dummy_input)
            self.pre_flatten_shape = dummy_encoded.shape[1:]
            self.flattened_size = dummy_encoded.view(1, -1).size(1)
        self.fc_latent = nn.Linear(self.flattened_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder_cnn = nn.Sequential(
            nn.Unflatten(1, self.pre_flatten_shape),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.clf = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.flatten(self.encoder_cnn(x))
        latent = self.fc_latent(h)
        return latent

    def decode(self, z):
        h = self.fc_decode(z)
        x_recon = self.decoder_cnn(h)
        return x_recon

    def forward(self, x):
        latent = self.encode(x)
        recon_x = self.decode(latent)
        class_output = self.clf(latent)
        return recon_x, latent, class_output

# === Latent Classifier ===
class LatentClassifier(nn.Module):
    def __init__(self, latent_dim):
        super(LatentClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(x)

# === Dataset Class ===
# class MFCCDataset(Dataset):
#     def __init__(self, file_label_list, sr=44100, duration=10, n_mfcc=26, time_frames=862):
#         self.file_label_list = file_label_list
#         self.sr = sr
#         self.samples = sr * duration
#         self.n_mfcc = n_mfcc
#         self.time_frames = time_frames
#         self.duration = duration
#     def __len__(self):
#         return len(self.file_label_list)
#     def __getitem__(self, idx):
#         path, label = self.file_label_list[idx]
#         y, sr = librosa.load(path, sr=self.sr, duration=self.duration, mono=True)
#         y = y / np.max(np.abs(y) + 1e-8)
#         y = librosa.util.fix_length(y, size=self.samples)
#         mfcc = preprocess_audio_buffer(y, sample_rate=self.sr, n_mfcc=self.n_mfcc, time_frames=self.time_frames)
#         return torch.tensor(mfcc[np.newaxis, :, :], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


import numpy as np
import librosa
import torch

def extract_top_energy_segments(y, sr, window_size=2.0, top_k=3):
    hop_length = int(window_size * sr)
    frame_count = (len(y) - hop_length) // hop_length

    energies = []
    segments = []

    for i in range(frame_count):
        start = i * hop_length
        end = start + hop_length
        segment = y[start:end]
        energy = np.sum(segment ** 2)
        energies.append(energy)
        segments.append(segment)

    top_indices = np.argsort(energies)[-top_k:]
    top_segments = [segments[i] for i in top_indices]

    return top_segments

class MFCCDataset(Dataset):
    def __init__(self, file_label_list, sr=44100, duration=10, n_mfcc=26, time_frames=862, top_k=1):
        self.file_label_list = file_label_list
        self.sr = sr
        self.samples = sr * duration
        self.n_mfcc = n_mfcc
        self.time_frames = time_frames
        self.duration = duration
        self.top_k = top_k
        self.augmented_data = []
        self.prepare_data()

    def prepare_data(self):
        for path, label in self.file_label_list:
            y, sr = librosa.load(path, sr=self.sr)
            y = librosa.util.fix_length(y, size=self.samples)

            # Original full clip
            self.augmented_data.append((y, label))

            # High-energy segments
            top_segments = extract_top_energy_segments(y, sr, window_size=2.0, top_k=self.top_k)
            for seg in top_segments:
                # Pad to 10s
                if len(seg) < self.samples:
                    seg = np.pad(seg, (0, self.samples - len(seg)), mode='constant')
                else:
                    seg = seg[:self.samples]
                self.augmented_data.append((seg, label))

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        y, label = self.augmented_data[idx]
        mfcc = preprocess_audio_buffer(y, sample_rate=self.sr, n_mfcc=self.n_mfcc, time_frames=self.time_frames)
        return torch.tensor(mfcc[np.newaxis, :, :], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# === Preprocessing ===
def normalize_audio_to_std(y, target_std=0.02):
    current_std = np.std(y) + 1e-8
    scaled_y = y * (target_std / current_std)
    return np.clip(scaled_y, -1.0, 1.0)

def preprocess_audio_buffer(buffer_copy, sample_rate=44100, n_mfcc=26, time_frames=862, energy_threshold=1e-4):
    if np.issubdtype(buffer_copy.dtype, np.integer):
        buffer_copy = buffer_copy.astype(np.float32) / np.iinfo(buffer_copy.dtype).max
    if buffer_copy.ndim == 2:
        mono_audio = np.mean(buffer_copy, axis=1)
    else:
        mono_audio = buffer_copy

    rms_energy = np.sqrt(np.mean(mono_audio ** 2))
    if rms_energy < energy_threshold:
        mono_audio = np.zeros_like(mono_audio)
    else:
        mono_audio = mono_audio / (np.max(np.abs(mono_audio)) + 1e-8)

    mono_audio = normalize_audio_to_std(mono_audio, target_std=0.02)
    mel_spec = librosa.feature.melspectrogram(y=mono_audio, sr=sample_rate, n_mels=40)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel_spec, sr=sample_rate, n_mfcc=n_mfcc)

    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc, axis=1, keepdims=True)
    mfcc_std_clamped = np.maximum(mfcc_std, 1e-3)
    mfcc_norm = (mfcc - mfcc_mean) / mfcc_std_clamped

    if mfcc_norm.shape[1] < time_frames:
        pad_width = time_frames - mfcc_norm.shape[1]
        mfcc_norm = np.pad(mfcc_norm, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc_norm.shape[1] > time_frames:
        mfcc_norm = mfcc_norm[:, :time_frames]

    return mfcc_norm

# === Training DAE ===
def train_dae(model, dataloader, device, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    clf_loss_fn = nn.BCELoss()
    recon_loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for clean_batch, labels in dataloader:
            clean_batch = clean_batch.to(device)
            labels = labels.float().to(device)

            noise = torch.randn_like(clean_batch) * 0.1
            noisy_batch = clean_batch + noise

            optimizer.zero_grad()
            recon, latent, class_output = model(noisy_batch)
            # Ensure dimensions match
            min_freq = min(recon.shape[2], clean_batch.shape[2])
            min_time = min(recon.shape[3], clean_batch.shape[3])
            recon_cropped = recon[:, :, :min_freq, :min_time]
            clean_cropped = clean_batch[:, :, :min_freq, :min_time]

            recon_loss = recon_loss_fn(recon_cropped, clean_cropped)
            clf_loss_val = clf_loss_fn(class_output.squeeze(1), labels)

            total_batch_loss = 0.5 * recon_loss + 0.5 * clf_loss_val
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# === Utility Functions ===
def extract_latents(model, dataloader, device):
    latents = []
    labels = []
    with torch.no_grad():
        for inputs, label in dataloader:
            latent = model.encode(inputs.to(device))
            latents.append(latent.cpu().numpy())
            labels.append(label.numpy())
    X = np.vstack(latents)
    y = np.concatenate(labels)
    return X, y

def run_predict(model, latent_clf, dataloader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            latent = model.encode(inputs)
            probs = latent_clf(latent).squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            preds = (probs > threshold).float()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            all_probs.append(probs)
            all_preds.append(preds)
    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    res = list(zip(all_probs.numpy(), all_preds.numpy()))
    cleaned_results = [(round(float(prob), 3), int(pred)) for prob, pred in res]
    return cleaned_results

def live_process_dae(buffer_copy, model, latent_clf, device, sample_rate=44100, n_mfcc=26, time_frames=862, threshold=0.5):
    mfcc = preprocess_audio_buffer(buffer_copy, sample_rate=sample_rate, n_mfcc=n_mfcc, time_frames=time_frames)
    input_tensor = torch.tensor(mfcc[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
    dummy_label_tensor = torch.tensor([-1], dtype=torch.float32)
    dataset = TensorDataset(input_tensor, dummy_label_tensor)
    dataloader = DataLoader(dataset, batch_size=1)
    model.eval()
    y_pred = run_predict(model, latent_clf, dataloader, device, threshold=threshold)
    latents = extract_latents(model, dataloader, device)
    return y_pred, latents[0]

def train_vae_classifier(model, latent_clf, dataloader, device, num_epochs=10):
    # Extract latent features
    X_latent, y_labels = extract_latents(model, dataloader, device)
    X_latent = torch.tensor(X_latent, dtype=torch.float32).to(device)
    y_labels = torch.tensor(y_labels, dtype=torch.float32).to(device)

    # Create classifier
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(latent_clf.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = latent_clf(X_latent).squeeze()
        loss = criterion(outputs, y_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# === Visualization ===
def visualize_latent_space(model, dataloader, device, method='tsne', return_reducer=False):
    model.eval()
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            latent = model.encode(inputs.to(device))
            all_latents.append(latent.cpu())
            all_labels.append(labels.cpu())
    latents = torch.cat(all_latents).numpy()
    labels = torch.cat(all_labels).numpy()
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")
    reduced = reducer.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', edgecolor='k', alpha=0.7)
    plt.title(f"Latent Space Visualization ({method.upper()})")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Label')
    plt.show()
    if return_reducer:
        return reducer



def evaluate_dae_highlight_folder(model, latent_clf, device, highlights_dir, non_highlights_dir, n_mfcc=26, time_frames=862, threshold=0.5):
    y_true = []
    y_pred = []

    # Process highlight files
    for f in os.listdir(highlights_dir):
        if f.endswith('.wav'):
            filepath = os.path.join(highlights_dir, f)
            y, sr = librosa.load(filepath, sr=44100, mono=True)
            prediction, _ = live_process_dae(y, model, latent_clf, device, n_mfcc=n_mfcc, time_frames=time_frames, threshold=threshold)
            prob, pred = prediction[0]
            y_true.append(1)
            y_pred.append(pred)
            # print(f"[Highlight] {f} → Pred: {pred}, Prob: {prob:.3f}")

    # Process non-highlight files
    for f in os.listdir(non_highlights_dir):
        if f.endswith('.wav'):
            filepath = os.path.join(non_highlights_dir, f)
            y, sr = librosa.load(filepath, sr=44100, mono=True)
            prediction, _ = live_process_dae(y, model, latent_clf, device, n_mfcc=n_mfcc, time_frames=time_frames, threshold=threshold)
            prob, pred = prediction[0]
            y_true.append(0)
            y_pred.append(pred)
            # print(f"[Non-highlight] {f} → Pred: {pred}, Prob: {prob:.3f}")

    # Compute metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== DAE Evaluation Metrics ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    return precision, recall, accuracy, f1

# === Main ===
if __name__ == "__main__":
    SAMPLE_RATE = 44100
    N_MFCC = 26
    TIME_FRAMES = 862
    LATENT_DIM = 32
    NUM_EPOCHS = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # List of multiple highlight folders
    highlight_folders = [
        './dataset_scraping/JP/TrackAndField/highlights',
        './dataset_scraping/JP/TrackAndField/highlights',
        './NBAHighlightsWAV'
    ]

    non_highlight_folders = [
        './dataset_scraping/JP/TrackAndField/non_highlights',
        './dataset_scraping/JP/TrackAndField/non_highlights',
        './NBANonHighlightsWAV'
    ]

    # Collect all highlight files across folders
    highlight_files = []
    for folder in highlight_folders:
        highlight_files.extend([
            (os.path.join(folder, f), 1)
            for f in os.listdir(folder)
            if f.endswith('.wav')
        ])

    # Collect all non-highlight files across folders
    non_highlight_files = []
    for folder in non_highlight_folders:
        non_highlight_files.extend([
            (os.path.join(folder, f), 0)
            for f in os.listdir(folder)
            if f.endswith('.wav')
        ])

    # Combine into one training list
    label_list = highlight_files + non_highlight_files

    # Shuffle the list for randomness
    random.shuffle(label_list)

    # Define split ratio (e.g., 80% train, 20% validation)
    split_ratio = 0.8
    split_index = int(len(label_list) * split_ratio)

    # Split into train and validation
    train_file_label_list = label_list[:split_index]
    val_file_label_list = label_list[split_index:]

    os.makedirs('validation_set', exist_ok=True)
    with open('validation_set/val_file_label_list.txt', 'w') as f:
        for path, label in val_file_label_list:
            f.write(f"{path}\t{label}\n")

    dataset = MFCCDataset(train_file_label_list, sr=SAMPLE_RATE, n_mfcc=N_MFCC, time_frames=TIME_FRAMES)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ConvDAE(n_mfcc=N_MFCC, time=TIME_FRAMES, latent_dim=LATENT_DIM).to(device)
    train_dae(model, dataloader, device, num_epochs=NUM_EPOCHS)

    latent_clf = LatentClassifier(latent_dim=LATENT_DIM).to(device)
    train_vae_classifier(model, latent_clf, dataloader, device, num_epochs=NUM_EPOCHS)

    # probs_preds = run_predict(model, latent_clf, dataloader, device)
    # for (prob, pred), (filename, true_label) in zip(probs_preds, dataloader.dataset.file_label_list):
    #     print(f"File: {filename}")
    #     print(f"True Label: {true_label}, Predicted: {pred}, Probability: {prob:.3f}")
    #     print("-" * 30)



    training_latents, training_labels = extract_latents(model, dataloader, device)
    os.makedirs('model_v2', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_clf_state_dict': latent_clf.state_dict(),
        'n_mfcc': N_MFCC,
        'time_frames': TIME_FRAMES,
    }, 'vae_model_state.pth')
    np.savez('./model_v2/training_latents.npz', latents=training_latents, labels=training_labels)

    import joblib
    reducer = visualize_latent_space(model, dataloader, device, method='pca', return_reducer=True)
    joblib.dump(reducer, './model_v2/trained_pca.pkl')

    print("✅ Denoising Autoencoder training complete!")
