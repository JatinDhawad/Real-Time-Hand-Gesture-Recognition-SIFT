import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sift_features import extract_sift
from build_histogram import build_histogram

kmeans = joblib.load("vocabulary.pkl")

X = []
y = []

dataset_path = "dataset"

for label in os.listdir(dataset_path):
    for img_name in os.listdir(f"{dataset_path}/{label}"):
        img = cv2.imread(f"{dataset_path}/{label}/{img_name}")
        desc = extract_sift(img)
        hist = build_histogram(desc, kmeans)

        X.append(hist)
        y.append(label)

model = SVC(kernel="linear")
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained and saved.")
