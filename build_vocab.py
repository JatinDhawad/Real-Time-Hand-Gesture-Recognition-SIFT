import os
import cv2
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sift_features import extract_sift

dataset_path = "dataset"
all_descriptors = []

for label in os.listdir(dataset_path):
    for img_name in os.listdir(f"{dataset_path}/{label}"):
        img = cv2.imread(f"{dataset_path}/{label}/{img_name}")
        desc = extract_sift(img)

        if desc is not None:
            all_descriptors.extend(desc)

all_descriptors = np.array(all_descriptors)

kmeans = KMeans(n_clusters=100)
kmeans.fit(all_descriptors)

joblib.dump(kmeans, "vocabulary.pkl")
print("Vocabulary saved.")
