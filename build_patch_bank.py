"""Build the patch memory bank from the training images.

Each good training image -> its 256 patch tokens; stack them all into one bank
of good patches.

Run:  .venv/bin/python build_patch_bank.py
"""

import numpy as np

from embedder import Embedder
import config


def main():
    embedder = Embedder()
    paths = config.list_images(config.TRAIN_GOOD)

    patches = []
    for path in paths:
        patches.append(embedder.embed_patches(path))

    bank = np.concatenate(patches, axis=0)   # (num_images * 256, dim)
    np.save("data/patch_bank.npy", bank)

    print(f"patch bank  shape {bank.shape}  "
          f"({len(paths)} images x {len(patches[0])} patches)")
    print("saved to data/patch_bank.npy")


if __name__ == "__main__":
    main()
