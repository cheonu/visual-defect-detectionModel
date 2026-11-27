"""The hybrid detector: CLS verdict + patch heatmap.

The verdict uses the whole-image (CLS) embedding — reliable (95% catch on test).
The heatmap uses the patch map — the model's best guess at where the anomaly is.

This is the function the website's /inspect endpoint calls.
"""

import json
import numpy as np

import nearest
from embedder import Embedder
import config
from patch_scoring import patch_distances


class Detector:
    """Loads everything once, then inspects images."""

    def __init__(self):
        self.embedder = Embedder()
        self.vectors = nearest.load_store()[1]                 # CLS memory bank
        self.threshold = json.load(open("data/threshold.json"))["threshold"]
        self.patch_bank = np.load("data/patch_bank.npy")       # patch memory bank

    def inspect(self, path):
        """One image -> {verdict, score, threshold, patch_scores}."""
        distance = nearest.nearest_distance(self.embedder.embed(path), self.vectors)
        verdict = "defective" if distance > self.threshold else "good"
        patch_scores = patch_distances(self.embedder.embed_patches(path), self.patch_bank)
        return {
            "verdict": verdict,
            "score": round(distance, 3),
            "threshold": round(self.threshold, 3),
            "patch_scores": patch_scores,   # 256 values -> the heatmap
        }
