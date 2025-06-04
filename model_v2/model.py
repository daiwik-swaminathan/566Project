import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# === Parameters ===
highlight_dir = './NBAHighlightsWAV'
nonhighlight_dir = './NBANonHighlightsWAV'
sample_rate = 22050
n_mels = 128
duration = 10.0

# === Model ===
class AudioCNN(nn.Module):
    def __init__(self, n_mels, time_frames):
        super(AudioCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * (n_mels // 4) * (time_frames // 4), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# === Dataset ===
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Utilities ===
def load_mel_spectrogram(path):
    y, sr = librosa.load(path, sr=sample_rate, duration=duration)
    if len(y) < int(duration * sr):
        y = np.pad(y, (0, int(duration * sr) - len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

# === Preprocess Function for Inference ===
def preprocess_audio_for_model(input_data, sample_rate=44100, n_mels=128, duration=10.0, from_file=True, expected_time_frames=862):
    if from_file:
        y, sr = librosa.load(input_data, sr=sample_rate, duration=duration)
    else:
        y = input_data
        sr = sample_rate

    if len(y) < int(duration * sr):
        y = np.pad(y, (0, int(duration * sr) - len(y)))

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    print(S_db.shape[1], expected_time_frames)

    # Adjust time_frames
    if S_db.shape[1] < expected_time_frames:
        pad_width = expected_time_frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
    elif S_db.shape[1] > expected_time_frames:
        S_db = S_db[:, :expected_time_frames]

    return S_db

# === Load Dataset ===
def load_dataset(folder, label):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            filepath = os.path.join(folder, filename)
            mel = preprocess_audio_for_model(filepath)
            data.append((mel[np.newaxis, :, :], label))  # (1, n_mels, time_frames)
    return data

# === Main Training Loop ===
def train():
    # Load data
    highlight_data = load_dataset(highlight_dir, 1)
    nonhighlight_data = load_dataset(nonhighlight_dir, 0)
    all_data = highlight_data + nonhighlight_data
    np.random.shuffle(all_data)

    X = np.array([mel for mel, _ in all_data])
    y = np.array([label for _, label in all_data])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Setup datasets
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Model
    time_frames = X_train.shape[-1]
    print(time_frames, n_mels)
    model = AudioCNN(n_mels, time_frames)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/30 - Loss: {running_loss:.6f}")

    os.makedirs('model_v2', exist_ok=True)
    torch.save(model.state_dict(), 'model_v2/model_state.pth')

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).cpu().numpy()
            y_true.extend(targets.numpy())
            y_pred.extend(preds)

    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    train()
