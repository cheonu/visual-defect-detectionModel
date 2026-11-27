"""Compute the patch-level threshold from leave-one-out scores on the training bank.

Run:  .venv/bin/python patch_threshold.py
"""

import json
import datetime
import numpy as np

import config
from patch_scoring import image_score

BANK = "data/patch_bank.npy"
PATCHES_PER_IMAGE = 256


def main():
    bank = np.load(BANK)
    num_images = bank.shape[0] // PATCHES_PER_IMAGE

    scores = []
    for i in range(num_images):
        start = i * PATCHES_PER_IMAGE
        end = start + PATCHES_PER_IMAGE
        score = image_score(bank[start:end], bank,
                            exclude_start=start, exclude_end=end)
        scores.append(score)

    scores = np.array(scores)
    threshold = float(scores.max())

    record = {
        "threshold": threshold,
        "date": datetime.date.today().isoformat(),
        "data": f"leave-one-out patch scores of {num_images} train/good images ({config.DATASET})",
        "rule": "max of good-good patch scores (leave-one-out)",
        "mean": float(scores.mean()),
        "max": float(scores.max()),
    }
    with open("data/patch_threshold.json", "w") as f:
        json.dump(record, f, indent=2)

    print(f"patch threshold  min {scores.min():.3f}  mean {scores.mean():.3f}  "
          f"max {threshold:.3f}")
    print("saved data/patch_threshold.json")


if __name__ == "__main__":
    main()
