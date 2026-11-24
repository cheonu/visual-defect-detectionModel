"""Unit 2a: distance and nearest-neighbour search.

The detector's whole idea: a defective part is FAR from every good part. So the
core operation is measuring how far a query vector is from the nearest good one.
"""

import numpy as np

STORE_PATH = "data/embeddings.npz"


def load_store():
    """Read back the (names, vectors) saved by embed_train.py."""
    data = np.load(STORE_PATH)
    return data["names"], data["vectors"]


def euclidean_distance(a, b):
    """Straight-line distance between two equal-length vectors."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def nearest_distance(query, vectors):
    """Distance from query to its nearest vector in the store (linear scan)."""
    best = float("inf")
    for vector in vectors:
        distance = euclidean_distance(vector, query)
        if distance < best:
            best = distance
    return best


def nearest_other_distance(index, vectors):
    """Distance from vectors[index] to its nearest DIFFERENT vector (leave-one-out)."""
    best = float("inf")
    for j in range(len(vectors)):
        if j == index: 
            continue
        distance = euclidean_distance(vectors[index], vectors[j])
        if distance < best:
            best = distance
    return best
