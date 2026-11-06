# Infant Cry Detection using AI

A machine learning system that listens to audio in real time, detects whether a baby is crying, and pushes the result to **Firebase** so that a connected app or device can alert parents right away.

---

## ✨ Features

- 🎙️ **Real-time monitoring**: records audio from the microphone in 5-second clips, in a continuous loop
- 🔊 **Audio amplification**: boosts the volume of each recording (via `pydub`) before analysis, which helps with quiet input
- 🎼 **Audio feature extraction** with `librosa` (26 features per clip):
  - Chroma STFT
  - RMS energy
  - Spectral centroid
  - Spectral bandwidth
  - Spectral roll-off
  - Zero-crossing rate
  - 20 MFCCs (Mel-Frequency Cepstral Coefficients)
- 🧠 **Classification**: a **Logistic Regression** model (scikit-learn) labels each clip as `crying` or `not crying`
- ☁️ **Firebase integration**: sends the live status (`1` = crying, `0` = normal) to `/BabyCondition/BabyCrying` in the Firebase Realtime Database
- 🧹 **Auto cleanup**: deletes temporary audio and CSV files after each prediction

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | **92.29%** |
| ROC-AUC | **0.933** |
| F1 score (macro) | **0.814** |
| Confusion matrix (TN, FP, FN, TP) | 416, 12, 26, 39 |

*Train/test split: 85% / 15% (`random_state=4`)*

---

## 🛠️ Tech Stack

`Python` · `librosa` · `scikit-learn` · `pandas` · `NumPy` · `pydub` · `ffmpeg` · `matplotlib` · `scikit-plot` · `Firebase` · `Google Colab`

---

## ⚙️ How It Works

```
🎤 Record 5s audio → 🔊 Amplify → 🎼 Extract 26 features → 🧠 Logistic Regression → ☁️ Firebase → 📱 Alert
```

1. **Dataset prep**: audio files are sorted into `crying/` and `notcrying/` folders, and their features are written to `dataset.csv`.
2. **Training**: the Logistic Regression model is trained on the extracted features.
3. **Evaluation**: accuracy, confusion matrix, ROC-AUC, F1 score and the ROC curve are computed.
4. **Live detection**: the browser microphone (through Colab JavaScript) records audio, and each clip is classified and sent to Firebase.

---

## 🚀 Getting Started

1. Open the notebook in **Google Colab** and mount Google Drive.
2. Install dependencies:
   ```bash
   pip install librosa scikit-learn pydub ffmpeg-python scikit-plot firebase-admin
   pip install git+https://github.com/ozgur/python-firebase
   ```
3. Update the placeholders in the notebook:
   - `<your-dataset-location>`: Drive folder containing `crying/` and `notcrying/`
   - `<your-features-csv-location>`: a pre-extracted features CSV (optional)
   - `<your-firebase-link-to-send-data-to>`: your Firebase Realtime Database URL
4. Run all cells and allow microphone access. Live predictions start automatically.

---

## 📁 Project Structure

```
├── FinalFeaturesLogisticRegression.ipynb   # Main notebook
├── dataset.csv                             # Generated features dataset
└── README.md
```

---

## 🎓 About

Developed as a **Final Year Project Module**: An AI-powered smart baby monitor that combines audio signal processing, machine learning and IoT/cloud connectivity.
