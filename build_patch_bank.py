"""Build the patch memory bank from the training images (with augmentation).

Each good image -> its 256 patch tokens, for every augmented version.

Run:  .venv/bin/python build_patch_bank.py
"""

import numpy as np
from PIL import Image

from embedder import Embedder
import config
import augment


def main():
    embedder = Embedder()
    paths = config.list_images(config.TRAIN_GOOD)

    patches = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        for aug_name, augmented in augment.apply_all(image):
            patches.append(embedder.embed_patches_image(augmented))

    bank = np.concatenate(patches, axis=0)
    np.save("data/patch_bank.npy", bank)
    print(f"patch bank  shape {bank.shape}  "
          f"({len(paths)} images x {len(augment.AUGMENTATIONS)} augmentations x 256 patches)")
    print("saved to data/patch_bank.npy")


if __name__ == "__main__":
    main()
