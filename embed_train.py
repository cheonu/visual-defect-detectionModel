"""Unit 1c: embed every training image once and store (filename, vector) pairs.

Run:  .venv/bin/python embed_train.py
"""

import numpy as np
from pathlib import Path

from embedder import Embedder
import config


def main():
    embedder = Embedder()                            # load the model once
    paths = sorted(Path(config.TRAIN_GOOD).glob("*.png"))

    names = []
    vectors = []
    for path in paths:                               # one linear pass
        names.append(path.name)
        vectors.append(embedder.embed(path))

    vectors = np.stack(vectors)                      # list of 512-arrays -> (209, 512)
    names = np.array(names)

    np.savez("data/embeddings.npz", names=names, vectors=vectors)

    print(f"stored {len(names)} vectors  shape {vectors.shape}")
    print("saved to data/embeddings.npz")


if __name__ == "__main__":
    main()
