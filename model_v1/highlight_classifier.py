import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# === Paths and Parameters ===
highlight_dir = os.path.expanduser('./NBAHighlightsWAV')
nonhighlight_dir = os.path.expanduser('./NBANonHighlightsWAV')
sample_rate = 22050
n_mels = 128

# === Load and Convert to Mel Spectrogram ===
def load_mel_spectrogram(path, duration=5.0):
    y, sr = librosa.load(path, sr=sample_rate, duration=duration)
    if len(y) < int(duration * sr):
        y = np.pad(y, (0, int(duration * sr) - len(y)))  # pad short clips
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def load_dataset(folder, label):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            filepath = os.path.join(folder, filename)
            mel = load_mel_spectrogram(filepath)
            data.append((mel, label))
    return data

# === Load and Prepare Data ===
highlight_data = load_dataset(highlight_dir, 1)
nonhighlight_data = load_dataset(nonhighlight_dir, 0)
all_data = highlight_data + nonhighlight_data
np.random.shuffle(all_data)

X = [mel.flatten() for mel, _ in all_data]
y = [label for _, label in all_data]
X = np.array(X)
y = np.array(y)

# === PCA Reduction ===
pca = PCA(n_components=36)
X_pca = pca.fit_transform(X)
X_pca = X_pca.reshape(-1, 1, 6, 6).astype(np.float32)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# === PyTorch Dataset & DataLoader ===
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# === Define CNN Model ===
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # (1, 6, 6) → (32, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (32, 2, 2)
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model = CNNClassifier()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# === Training Loop ===
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
    
    print(f"Epoch {epoch+1}/30 - Loss: {running_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'model_v1/model_state.pth')

# === Evaluation ===
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
