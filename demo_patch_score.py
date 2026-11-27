"""Score one image's patches against the good bank, and save a heatmap.

Run:  .venv/bin/python demo_patch_score.py [image_path]
"""

import sys
import numpy as np
from PIL import Image

from embedder import Embedder
import config

GRID = 16   # 224 / 14 patch size


def patch_scores(query_patches, bank):
    """Per-patch min distance to the bank. query: (256, dim), bank: (N, dim)."""
    sims = query_patches @ bank.T            # cosine similarity (both unit length)
    max_sim = sims.max(axis=1)               # nearest good patch per query patch
    return 2 - 2 * max_sim                   # distance = 2 - 2 * cosine


def save_heatmap(scores, image_path, out_path):
    """Overlay the 16x16 anomaly scores on the image (red = anomalous)."""
    grid = scores.reshape(GRID, GRID)
    lo, hi = grid.min(), grid.max()
    norm = (grid - lo) / (hi - lo) if hi > lo else np.zeros_like(grid)

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    heat = Image.fromarray((norm * 255).astype(np.uint8), "L")
    heat = heat.resize((w, h), Image.Resampling.BILINEAR)
    red = Image.new("RGB", (w, h), (255, 0, 0))
    overlay = Image.composite(red, image, heat)
    Image.blend(image, overlay, 0.5).save(out_path)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else config.list_images(config.TEST_DEFECTIVE)[0]

    bank = np.load("data/patch_bank.npy")
    embedder = Embedder()
    scores = patch_scores(embedder.embed_patches(path), bank)

    idx = scores.argmax()
    print(f"image         {path}")
    print(f"patch scores  min {scores.min():.3f}  mean {scores.mean():.3f}  "
          f"max {scores.max():.3f}")
    print(f"hottest patch  row {idx // GRID}  col {idx % GRID}  "
          f"score {scores[idx]:.3f}")

    save_heatmap(scores, path, "reports/patch_heatmap.png")
    print("saved reports/patch_heatmap.png")


if __name__ == "__main__":
    main()
