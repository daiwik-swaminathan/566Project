# Sports Highlight Audio Classifier

This project classifies short sport game audio clips as **highlight** or **non-highlight** moments using Mel spectrograms and a Convolutional Neural Network (CNN). Dimensionality reduction is applied using PCA as required by the course.

## 1. 📁 Directory Structure

```
.
├── highlight_classifier.py       # Main script for training and evaluating the model
├── README.md
├── NBAHighlightsWAV/            # Folder containing 27 highlight `.wav` clips
└── NBANonHighlightsWAV/         # Folder containing 27 non-highlight `.wav` clips
```

## 2. 🔍 Project Overview

- **Goal**: Classify audio clips from NBA games into highlights (e.g., game-winners, buzzer-beaters) vs. non-highlights (e.g., free throws, regular plays).
- **Input**: Short `.wav` files of NBA commentary.
- **Output**: Binary classification label (`1 = highlight`, `0 = non-highlight`).

## 3. 🧠 Methodology

1. **Audio Preprocessing**
   - Each `.wav` file is loaded using `librosa`
   - Audio is padded or trimmed to 5 seconds for uniform length
   - A **Mel spectrogram** is computed for each clip
   - Spectrograms are converted to log scale (dB)

2. **Dimensionality Reduction**
   - Spectrograms are flattened into 1D vectors
   - **PCA** is applied to reduce dimensionality to **36 components**
   - The reduced vectors are reshaped into `6 × 6` grayscale image-like tensors for CNN input

3. **Model Architecture**
   - A simple **CNN** with:
     - One 2D convolutional layer with a 3×3 kernel
     - Max pooling layer (2×2)
     - Flatten + dense layer
     - Binary output with sigmoid activation
   - Architecture chosen to avoid over-shrinking the 6×6 input size

4. **Training Setup**
   - 80/20 train-test split (43 training samples, 11 test samples)
   - Trained for 30 epochs with a batch size of 8
   - Loss optimized using binary cross-entropy and Adam optimizer
   - Final model evaluated on test set using accuracy, precision, recall, F1-score, and confusion matrix

## 4. ✅ Requirements

We recommend using a Python virtual environment to isolate dependencies.

### 1. Create and activate a virtual environment

These steps assume you have Apple Silicon (PyTorch for Apple Silicon)

```bash
python3 -m venv highlight_env
source highlight_env/bin/activate
```

```bash
pip install -r requirements.txt
```

OR manually:

```bash
pip install torch torchvision torchaudio          
pip install librosa numpy matplotlib scikit-learn
```

## 5. 🚀 Running the Project

1. Ensure the following folders are populated:
   - `NBAHighlightsWAV` — 27 `.wav` highlight clips
   - `NBANonHighlightsWAV` — 27 `.wav` non-highlight clips

2. Run the classifier:

```bash
python highlight_classifier.py
```

## 6. 📊 Output

- **Console metrics**:
  - Training/validation accuracy
  - Final test accuracy
  - Confusion matrix and classification report

## 7. 🔍 Example Spectrogram

(Sample spectrogram can be generated in the script. Replace this with an actual image if desired.)

## 8. 📌 Notes

- **PCA** is applied before CNN to satisfy the project’s dimensionality reduction requirement.
- If you want to switch to a traditional ML classifier (e.g., SVM), simply remove the CNN section and use the PCA-reduced vectors directly.
