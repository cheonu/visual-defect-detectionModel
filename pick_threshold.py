"""Unit 2b: pick the defective threshold from the training distances.

Run:  .venv/bin/python pick_threshold.py
"""

import json
import datetime
import numpy as np

import nearest
import config


def main():
    names, vectors = nearest.load_store()

    # leave-one-out: how far is each good image from its nearest OTHER good image
    distances = [nearest.nearest_other_distance(i, vectors)
                 for i in range(len(vectors))]
    distances = np.array(distances)

    print(f"leave-one-out distances of the {len(vectors)} good training images:")
    print(f"  min {distances.min():.3f}   "
          f"mean {distances.mean():.3f}   "
          f"max {distances.max():.3f}   "
          f"std {distances.std():.3f}")

    threshold = float(distances.max())   # no good image was ever farther than this

    record = {
        "threshold": threshold,
        "date": datetime.date.today().isoformat(),
        "data": f"leave-one-out nearest distances of {len(vectors)} train/good images ({config.DATASET})",
        "rule": "max of good-good nearest distances",
        "mean": float(distances.mean()),
        "std": float(distances.std()),
    }
    with open("data/threshold.json", "w") as f:
        json.dump(record, f, indent=2)

    print(f"threshold = {threshold:.3f}   (saved to data/threshold.json)")


if __name__ == "__main__":
    main()
