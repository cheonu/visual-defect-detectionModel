"""Import the bottle-label data: convert HEIC -> PNG and split into train/val/test.

Run:  .venv/bin/python import_bottle.py
"""

import subprocess
from pathlib import Path

SRC = Path.home() / "dev/bottle-rig/processed/label"
DEST = Path("data/bottle_label")

TRAIN_N = 25
VAL_GOOD_N = 8
VAL_BAD_N = 12
TEST_GOOD_N = 8
TEST_BAD_N = 12


def convert(src_heic, dst_png):
    subprocess.run(["sips", "-s", "format", "png", str(src_heic), "--out", str(dst_png)],
                   check=True, capture_output=True)


def main():
    good = sorted((SRC / "good").glob("*.HEIC"))
    defect = sorted((SRC / "defect").glob("*.HEIC"))

    split = {
        "train/good": good[:TRAIN_N],
        "validation/good": good[TRAIN_N:TRAIN_N + VAL_GOOD_N],
        "test/good": good[TRAIN_N + VAL_GOOD_N:TRAIN_N + VAL_GOOD_N + TEST_GOOD_N],
        "validation/defective": defect[:VAL_BAD_N],
        "test/defective": defect[VAL_BAD_N:VAL_BAD_N + TEST_BAD_N],
    }

    for folder, files in split.items():
        dest = DEST / folder
        dest.mkdir(parents=True, exist_ok=True)
        for src in files:
            convert(src, dest / (src.stem + ".png"))
        print(f"{folder:<24} {len(files)}")

    print(f"total: {len(good)} good, {len(defect)} defective")


if __name__ == "__main__":
    main()
