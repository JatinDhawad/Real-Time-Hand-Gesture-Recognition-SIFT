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
pip install opencv-contrib-python numpy scikit-learn joblib
▶️ How to Run

1️⃣ Build Vocabulary
python build_vocab.py

2️⃣ Train Model
python train_model.py

3️⃣ Run Real-Time Prediction
python realtime_predict.py

Press q to exit.
```
## 🧠 How It Works
Step 1 — Feature Extraction (SIFT)

Each gesture image is converted to grayscale.

SIFT detects scale and rotation invariant keypoints.

Local descriptors are extracted around each keypoint.

Step 2 — Bag of Visual Words (BoVW)

All descriptors from the dataset are clustered using KMeans.

Each cluster center represents a “visual word”.

Every image is converted into a histogram of visual word frequencies.

Step 3 — Classification (SVM)

Histograms are used as feature vectors.

A Support Vector Machine (SVM) classifier is trained.

During real-time detection, the histogram of the live frame is classified.

## 📊 Model Details

Feature Extractor: SIFT

Clustering Algorithm: KMeans

Number of Clusters: (e.g., 200 or 300)

Classifier: Linear SVM

Input: Live webcam frame (ROI-based detection)

## 🎯 Key Advantages

Robust to scale and rotation

Works with limited dataset compared to deep learning

Lightweight classical computer vision approach

Real-time performance on CPU

## ⚠ Limitations

Performance depends on lighting conditions

Background clutter may reduce accuracy

Requires sufficient dataset variation

Slower than ORB but more accurate

## 📸 Sample Output

<img width="940" height="485" alt="image" src="https://github.com/user-attachments/assets/133dfcba-0e7d-434c-b6c5-0f8a0adfded9" />
<img width="940" height="496" alt="image" src="https://github.com/user-attachments/assets/98eed341-9a67-44b3-b0e1-fdd7c6222921" />
<img width="940" height="492" alt="image" src="https://github.com/user-attachments/assets/751e07e7-c3bc-4ddc-964f-cfa2b70a7ed1" />
<img width="940" height="500" alt="image" src="https://github.com/user-attachments/assets/2dd1a6f9-493f-403f-8cbc-c10877e774d7" />

## 🚀 Future Improvements

Add skin color segmentation for better ROI extraction

Implement majority voting across frames

Increase dataset size for higher accuracy

Replace SIFT with ORB for faster performance

Convert to deep learning (CNN-based gesture recognition)

## 📈 Performance

Works best under good lighting

Accuracy improves with more training images

Best results achieved with:

100+ images per gesture

200–300 KMeans clusters

Clean background

## 👨‍💻 Author

Jatin Dhawad
B.Tech Computer Engineering
Computer Vision Project
