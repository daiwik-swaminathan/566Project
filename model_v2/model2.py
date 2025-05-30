import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# === ConvVAE Model ===
class ConvVAE(nn.Module):
    def __init__(self, n_mfcc, time, latent_dim=32):
        super(ConvVAE, self).__init__()
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
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder_cnn = nn.Sequential(
            nn.Unflatten(1, self.pre_flatten_shape),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.clf = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.flatten(self.encoder_cnn(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h = self.fc_decode(z)
        x_recon = self.decoder_cnn(h)
        return x_recon
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        class_output = self.clf(mu)
        return recon_x, mu, logvar, class_output

# === VAE Loss ===
def vae_loss(recon_x, x, mu, logvar):
    # Ensure matching size
    min_freq = min(recon_x.shape[2], x.shape[2])
    min_time = min(recon_x.shape[3], x.shape[3])

    recon_x_cropped = recon_x[:, :, :min_freq, :min_time]
    x_cropped = x[:, :, :min_freq, :min_time]

    recon_loss = nn.functional.mse_loss(recon_x_cropped, x_cropped)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# === Dataset Class with Labels ===
class MFCCDataset(Dataset):
    def __init__(self, file_label_list, sr=22050, duration=5, n_mfcc=26, time_frames=862):
        self.file_label_list = file_label_list
        self.sr = sr
        self.samples = sr * duration
        self.n_mfcc = n_mfcc
        self.time_frames = time_frames
    def __len__(self):
        return len(self.file_label_list)
    def __getitem__(self, idx):
        path, label = self.file_label_list[idx]
        y, sr = librosa.load(path, sr=self.sr, duration=self.samples / self.sr)
        y = librosa.util.fix_length(y, size=self.samples)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
        if mfcc.shape[1] < self.time_frames:
            pad_width = self.time_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > self.time_frames:
            mfcc = mfcc[:, :self.time_frames]
        return torch.tensor(mfcc[np.newaxis, :, :], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# === Training Loop ===
def train_vae(model, dataloader, device, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    clf_loss = nn.BCELoss()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch, labels in dataloader:
            batch = batch.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            recon, mu, logvar, class_output = model(batch)
            clf_loss_val = clf_loss(class_output.squeeze(1), labels)
            vae_loss_val = vae_loss(recon, batch, mu, logvar)
            
            total_loss = 0.5 * vae_loss_val + 0.5 * clf_loss_val

            total_loss.backward()
            optimizer.step()
            total_loss += total_loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def run_predict(model, dataloader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            # Extract latent features
            mu, _ = model.encode(inputs)

            # Run only classifier
            class_output = model.clf(mu)
            probs = class_output.squeeze(1).cpu()
            preds = (probs > threshold).float()

            all_probs.append(probs)
            all_preds.append(preds)

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    res = list(zip(all_probs.numpy(), all_preds.numpy()))
    cleaned_results = [(round(float(prob), 3), int(pred)) for prob, pred in res]

    return cleaned_results
 
def extract_latents(model, dataloader, device):
    latents = []
    labels = []
    with torch.no_grad():
        for inputs, label in dataloader:
            mu, _ = model.encode(inputs.to(device))
            latents.append(mu.cpu().numpy())
            labels.append(label.numpy())

    X = np.vstack(latents)
    y = np.concatenate(labels)
    return X,y

# === Live Processing ===
def preprocess_audio_buffer(buffer_copy, sample_rate=44100, n_mfcc=26, time_frames=862):
    mono_audio = np.mean(buffer_copy.reshape(-1, 2), axis=1)
    mono_audio = mono_audio / (np.max(np.abs(mono_audio)) + 1e-8)
    mfcc = librosa.feature.mfcc(y=mono_audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
    if mfcc.shape[1] < time_frames:
        pad_width = time_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > time_frames:
        mfcc = mfcc[:, :time_frames]
    return mfcc

def live_process_vae(buffer_copy, model, device, sample_rate=44100, n_mfcc=26, time_frames=862, threshold=0.5):
    mfcc = preprocess_audio_buffer(buffer_copy, sample_rate=sample_rate,n_mfcc=n_mfcc, time_frames=time_frames)
    input_tensor = torch.tensor(mfcc[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
    dummy_label_tensor  = torch.tensor([-1], dtype=torch.float32)  # Dummy label for TensorDataset
    dataset = TensorDataset(input_tensor, dummy_label_tensor)
    dataloader = DataLoader(dataset, batch_size=1)
    model.eval()
    y_pred = run_predict(model, dataloader, device, threshold=threshold)
    return y_pred


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_latent_space(model, dataloader, device, method='tsne'):
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            mu, _ = model.encode(inputs)
            all_latents.append(mu.cpu())
            all_labels.append(labels.cpu())

    latents = torch.cat(all_latents).numpy()
    labels = torch.cat(all_labels).numpy()

    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    reduced = reducer.fit_transform(latents)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.title(f"Latent Space Visualization ({method.upper()})")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter, label='Label')
    plt.show()


# === Main ===
if __name__ == "__main__":
    SAMPLE_RATE = 44100
    N_MFCC = 26
    TIME_FRAMES = 862
    LATENT_DIM = 32
    THRESHOLD = 0.1
    NUM_EPOCHS = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Step 1: Prepare dataset ===
    highlights_folder = './NBAHighlightsWAV'
    non_highlights_folder = './NBANonHighlightsWAV'
    highlight_files = [(os.path.join(highlights_folder, f), 1) for f in os.listdir(highlights_folder) if f.endswith('.wav')]
    non_highlight_files = [(os.path.join(non_highlights_folder, f), 0) for f in os.listdir(non_highlights_folder) if f.endswith('.wav')]
    train_file_label_list = highlight_files + non_highlight_files

    dataset = MFCCDataset(train_file_label_list, sr=SAMPLE_RATE, n_mfcc=N_MFCC, time_frames=TIME_FRAMES)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # === Step 2: Train model ===
    model = ConvVAE(n_mfcc=N_MFCC, time=TIME_FRAMES, latent_dim=LATENT_DIM).to(device)
    train_vae(model, dataloader, device, num_epochs=NUM_EPOCHS)

    # === Step 3: Evaluate ===
    probs_preds = run_predict(model, dataloader, device)
    file_label_list = dataloader.dataset.file_label_list
    for (prob, pred), (filename, true_label) in zip(probs_preds, file_label_list):
        print(f"File: {filename}")
        print(f"True Label: {true_label}, Predicted: {pred}, Probability: {prob:.3f}")
        print("-" * 30)


    os.makedirs('model_v2', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_mfcc': N_MFCC,
        'time_frames': TIME_FRAMES,
    }, 'vae_model_state.pth')
    print("✅ Model saved.")

    visualize_latent_space(model, dataloader, device, method='tsne')



