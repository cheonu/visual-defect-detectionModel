"""Compute the CLS threshold from leave-one-out scores on the training bank.

Leave-one-out at the IMAGE level: exclude all augmented versions of the same
photo, not just the identical vector. (Augmented versions of one photo are near
twins; letting them match each other would collapse the threshold to ~0.)

Run:  .venv/bin/python pick_threshold.py
"""

import json
import datetime
import numpy as np

import nearest
import config


def main():
    names, vectors = nearest.load_store()
    stems = np.array([n.rsplit("_", 1)[0] for n in names])   # "IMG_0090_rot+2" -> "IMG_0090"
    images = sorted(set(stems.tolist()))

    scores = []
    for image in images:
        own = np.where(stems == image)[0]
        other = np.where(stems != image)[0]
        sims = vectors[own] @ vectors[other].T               # cosine (unit length)
        max_sim = sims.max(axis=1)                           # nearest other-image version
        dists = np.sqrt(np.maximum(2 - 2 * max_sim, 0.0))
        scores.append(float(dists.max()))                    # this image's LOO score

    scores = np.array(scores)
    threshold = float(scores.max())

    record = {
        "threshold": threshold,
        "date": datetime.date.today().isoformat(),
        "data": f"leave-one-out distances of {len(images)} train/good images "
                f"({config.DATASET}; {len(names)} bank vectors incl. augmentation)",
        "rule": "max of good-good distances, leave-one-out at the image level",
        "mean": float(scores.mean()),
        "max": float(scores.max()),
    }
    with open("data/threshold.json", "w") as f:
        json.dump(record, f, indent=2)

    print(f"threshold  min {scores.min():.3f}  mean {scores.mean():.3f}  "
          f"max {threshold:.3f}   ({len(images)} images)")
    print("saved data/threshold.json")


if __name__ == "__main__":
    main()
