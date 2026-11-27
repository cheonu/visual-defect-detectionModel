"""Patch-level scoring: per-patch distances, the image-level score, and the heatmap."""

import numpy as np
from PIL import Image

GRID = 16   # 224 / 14 patch size


def patch_distances(patches, bank):
    """Per-patch min distance to the bank. patches: (P, dim), bank: (N, dim)."""
    sims = patches @ bank.T              # cosine (unit length)
    max_sim = sims.max(axis=1)           # nearest good patch per query patch
    return np.sqrt(np.maximum(2 - 2 * max_sim, 0.0))


def image_score(patches, bank, exclude_start=None, exclude_end=None):
    """Max patch distance — the detector's score for one image.

    exclude_start/exclude_end: bank column range to skip (leave-one-out).
    """
    sims = patches @ bank.T
    if exclude_start is not None:
        sims[:, exclude_start:exclude_end] = -1.0   # self patches -> worst cosine
    max_sim = sims.max(axis=1)
    dists = np.sqrt(np.maximum(2 - 2 * max_sim, 0.0))
    return float(dists.max())


def render_heatmap(scores, image_path, out_path):
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
