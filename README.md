# Real-Time Hand Gesture Recognition using SIFT

## 📌 Overview
This project implements real-time hand gesture recognition using:

- SIFT (Scale-Invariant Feature Transform)
- Bag of Visual Words (BoVW)
- KMeans clustering
- Support Vector Machine (SVM)
- OpenCV for real-time webcam detection

The system extracts SIFT keypoints from hand gestures, converts them into visual word histograms, and classifies gestures using SVM.

---

## 📂 Project Structure
dataset/
build_histogram.py
build_vocab.py
realtime_predict.py
sift_features.py
train_model.py
---

## ⚙️ Installation

```bash
pip install opencv-contrib-python numpy scikit-learn joblib▶️ How to Run
1️⃣ Build Vocabulary
python build_vocab.py
2️⃣ Train Model
python train_model.py
3️⃣ Run Real-Time Prediction
python realtime_predict.py

Press q to exit.
