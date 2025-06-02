import os
import librosa
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import torch

from model_v2.model2 import LatentClassifier
from model_v2.model3 import ConvDAE, evaluate_dae_highlight_folder

def compute_rms_energy(y):
    return np.sqrt(np.mean(y ** 2))

def compute_average_frequency(y, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(spectral_centroid)   

def predict_highlight(y, sr, mode='energy', energy_threshold=0.05, freq_threshold=3000):
    if mode == 'energy':
        rms = compute_rms_energy(y)
        return int(rms > energy_threshold)
    elif mode == 'frequency':
        avg_freq = compute_average_frequency(y, sr)
        return int(avg_freq > freq_threshold)
    else:
        raise ValueError("Mode must be 'energy' or 'frequency'")

def evaluate_highlight_folder(highlights_dir, non_highlights_dir, mode='energy', energy_threshold=0.05, freq_threshold=3000, verbose=True):
    y_true = []
    y_pred = []

    # Process highlight files
    for f in os.listdir(highlights_dir):
        if f.endswith('.wav'):
            y, sr = librosa.load(os.path.join(highlights_dir, f), sr=None)
            y_true.append(1)
            pred = predict_highlight(y, sr, mode, energy_threshold, freq_threshold)
            y_pred.append(pred)

    # Process non-highlight files
    for f in os.listdir(non_highlights_dir):
        if f.endswith('.wav'):
            y, sr = librosa.load(os.path.join(non_highlights_dir, f), sr=None)
            y_true.append(0)
            pred = predict_highlight(y, sr, mode, energy_threshold, freq_threshold)
            y_pred.append(pred)

    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if verbose:
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")
    return precision, recall, accuracy, f1

import numpy as np

def evaluate_energy_sweep(highlights_folder, non_highlights_folder, thresholds):
    best_score = 0
    best_thresh = None

    for thresh in thresholds:
        # print(f"Testing energy threshold: {thresh:.4f}")
        precision, recall, accuracy , f1 = evaluate_highlight_folder(highlights_folder, non_highlights_folder, mode='energy', energy_threshold=thresh, verbose=False)
        score = accuracy  # simple sum score; you can replace with F1 or other

        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"\n🔥 Best energy threshold: {best_thresh:.4f} with combined score {best_score:.4f}")
    evaluate_highlight_folder(highlights_folder, non_highlights_folder, mode='energy', energy_threshold=best_thresh)

def evaluate_frequency_sweep(highlights_folder, non_highlights_folder, thresholds):
    best_score = 0
    best_thresh = None

    for thresh in thresholds:
        # print(f"Testing frequency threshold: {thresh:.2f} Hz")
        precision, recall, accuracy, f1  = evaluate_highlight_folder(highlights_folder, non_highlights_folder, mode='frequency', freq_threshold=thresh, verbose=False)
        score = accuracy

        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"\n🔥 Best frequency threshold: {best_thresh:.2f} Hz with combined score {best_score:.4f}")
    evaluate_highlight_folder(highlights_folder, non_highlights_folder, mode='frequency', freq_threshold=best_thresh)

import os
import shutil

def organize_validation_files(filelist_path, dest_root='validation_set'):
    highlight_dir = os.path.join(dest_root, 'highlights')
    non_highlight_dir = os.path.join(dest_root, 'non_highlights')
    os.makedirs(highlight_dir, exist_ok=True)
    os.makedirs(non_highlight_dir, exist_ok=True)

    with open(filelist_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                src_path, label = parts
                label = int(label)
                filename = os.path.basename(src_path)
                dest_dir = highlight_dir if label == 1 else non_highlight_dir
                dest_path = os.path.join(dest_dir, filename)

                try:
                    shutil.copy(src_path, dest_path)
                    print(f"✅ Copied {src_path} → {dest_path}")
                except Exception as e:
                    print(f"⚠ Failed to copy {src_path}: {e}")

# Example usage:
organize_validation_files('validation_set/val_file_label_list.txt')

# Example usage
highlights_folder = r'./validation_set/highlights'
non_highlights_folder = r'./validation_set/non_highlights'
# highlights_folder = r'./NBAHighlightsWAV'
# non_highlights_folder = r'./NBANonHighlightsWAV'


# print("Evaluating energy-based model...\n")
# Evaluate energy-based model
# evaluate_highlight_folder(highlights_folder, non_highlights_folder, mode='energy', energy_threshold=0.02)

# print("\nEvaluating frequency-based model...\n")
# Evaluate frequency-based model
# evaluate_highlight_folder(highlights_folder, non_highlights_folder, mode='frequency', freq_threshold=3000)

# Example usage:
energy_thresholds = np.linspace(0.005, 0.05, 10)  # sweep 0.005 → 0.05
frequency_thresholds = np.linspace(1000, 5000, 10)  # sweep 1 kHz → 5 kHz

print("\n=== Energy Sweep ===\n")
evaluate_energy_sweep(highlights_folder, non_highlights_folder, energy_thresholds)

print("\n=== Frequency Sweep ===\n")
evaluate_frequency_sweep(highlights_folder, non_highlights_folder, frequency_thresholds)


SAMPLE_RATE = 44100
N_MFCC = 26
TIME_FRAMES = 862
LATENT_DIM = 32
NUM_EPOCHS = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('vae_model_state.pth', map_location=device)
n_mfcc = checkpoint['n_mfcc']
time_frames = checkpoint['time_frames']
model = ConvDAE(n_mfcc=n_mfcc, time=time_frames, latent_dim=LATENT_DIM).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

latent_clf = LatentClassifier(latent_dim=LATENT_DIM).to(device)
latent_clf.load_state_dict(checkpoint['latent_clf_state_dict'])

print("\nEvaluating DAE model on highlights folder...")
evaluate_dae_highlight_folder(
    model=model,
    latent_clf=latent_clf,
    device=device,
    highlights_dir=highlights_folder,
    non_highlights_dir=non_highlights_folder,
    n_mfcc=n_mfcc,
    time_frames=time_frames,
    threshold=0.5
)