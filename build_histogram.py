import numpy as np

def build_histogram(descriptors, kmeans):
    histogram = np.zeros(len(kmeans.cluster_centers_))

    if descriptors is not None:
        clusters = kmeans.predict(descriptors)

        for c in clusters:
            histogram[c] += 1

    return histogram
