"""Demonstrate the hybrid detector: CLS verdict + patch heatmap.

Run:  .venv/bin/python demo_detector.py [image_path]
"""

import sys

from detector import Detector
from patch_scoring import render_heatmap, GRID
import config


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else config.list_images(config.TEST_DEFECTIVE)[0]

    detector = Detector()
    result = detector.inspect(path)

    print(f"image      {path}")
    print(f"verdict    {result['verdict'].upper()}   "
          f"(score {result['score']} vs threshold {result['threshold']})")

    scores = result["patch_scores"]
    hot = scores.argmax()
    print(f"hot patch  row {hot // GRID} col {hot % GRID}  score {scores[hot]:.3f}")

    out = "reports/hybrid_heatmap.png"
    render_heatmap(scores, path, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
