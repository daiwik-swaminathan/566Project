# 🏀 NBA Highlight Audio Classifier

This project classifies short NBA game audio clips as **highlight** or **non-highlight** moments using Mel spectrograms and a Convolutional Neural Network (CNN). Dimensionality reduction is applied using PCA as required by the course.

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
   - Load each audio clip using `librosa`
   - Compute a **Mel spectrogram**
   - Convert spectrogram to log scale (dB)

2. **Dimensionality Reduction**
   - Flatten spectrograms
   - Apply **PCA** to reduce to 100 dimensions
   - Reshape into `10 × 10` grayscale image-like input for CNN

3. **Model Architecture**
   - A simple **CNN** with:
     - Two convolutional layers
     - Max pooling
     - Dense layers
   - Binary output using sigmoid activation

4. **Training Setup**
   - 80/20 train-test split
   - Trained for 30 epochs with a batch size of 8

## 4. ✅ Requirements

Install dependencies with:

```bash
pip install librosa numpy matplotlib scikit-learn tensorflow
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