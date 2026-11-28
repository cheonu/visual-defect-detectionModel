"""Score good vs defective images with patch scoring (max patch distance).

Prints the scores for validation and test, so you can see whether the two
classes separate. No threshold applied — just the raw score distributions.

Run:  .venv/bin/python score_patches.py
"""

import numpy as np

from embedder import Embedder
import config
from patch_scoring import image_score


def main():
    bank = np.load("data/patch_bank.npy")
    embedder = Embedder()

    for set_name, good, bad in [
            ("validation", config.VALIDATION_GOOD, config.VALIDATION_DEFECTIVE),
            ("test", config.TEST_GOOD, config.TEST_DEFECTIVE)]:
        g = [round(image_score(embedder.embed_patches(p), bank), 3)
             for p in config.list_images(good)]
        b = [round(image_score(embedder.embed_patches(p), bank), 3)
             for p in config.list_images(bad)]
        print(f"{set_name} good:      " + "  ".join(f"{s:.3f}" for s in g))
        print(f"{set_name} defective: " + "  ".join(f"{s:.3f}" for s in b))


if __name__ == "__main__":
    main()
