"""Unit 1c: embed every training image (with augmentation) into the memory bank.

Each good image becomes several augmented versions, so the bank tolerates the
small placement/lighting variation the rig produces.

Run:  .venv/bin/python embed_train.py
"""

import numpy as np
from PIL import Image

from embedder import Embedder
import config
import augment


def main():
    embedder = Embedder()
    paths = config.list_images(config.TRAIN_GOOD)

    names = []
    vectors = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        for aug_name, augmented in augment.apply_all(image):
            names.append(f"{path.stem}_{aug_name}")
            vectors.append(embedder.embed_image(augmented))

    vectors = np.stack(vectors)
    names = np.array(names)

    np.savez("data/embeddings.npz", names=names, vectors=vectors)
    print(f"stored {len(names)} vectors  shape {vectors.shape}  "
          f"({len(paths)} images x {len(augment.AUGMENTATIONS)} augmentations)")
    print("saved to data/embeddings.npz")


if __name__ == "__main__":
    main()
