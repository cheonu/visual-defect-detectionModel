"""Re-split the 24 defects into validation/test, stratified by severity.

The naive filename split put 9/10 obvious in validation and 10/12 borderline in
test. This splits each severity tier evenly so the evaluation is fair.

Run:  .venv/bin/python stratify_defects.py
"""

import subprocess
from pathlib import Path

SRC = Path.home() / "dev/bottle-rig/processed/label/defect"
DEST = Path("data/bottle_label")

# severity tiers, keyed by image number (IMG_0144 -> "0144")
OBVIOUS = ["0144", "0142", "0155", "0148", "0147", "0141",
           "0146", "0150", "0152", "0151"]
BORDERLINE = ["0156", "0163", "0162", "0154", "0160", "0153",
              "0161", "0157", "0164", "0158"]
BONUS = ["0137", "0143", "0149", "0159"]


def number(fname):
    return fname.stem.split("_")[1]   # "IMG_0144" -> "0144"


def convert(src_heic, dst_png):
    subprocess.run(["sips", "-s", "format", "png", str(src_heic), "--out", str(dst_png)],
                   check=True, capture_output=True)


def main():
    for folder in ["validation/defective", "test/defective"]:
        d = DEST / folder
        for p in d.glob("*.png"):
            p.unlink()

    files = {number(p): p for p in SRC.glob("*.HEIC")}

    for tier in [OBVIOUS, BORDERLINE, BONUS]:
        tier = sorted(tier)
        half = len(tier) // 2
        for i, num in enumerate(tier):
            folder = "validation/defective" if i < half else "test/defective"
            src = files[num]
            convert(src, DEST / folder / (src.stem + ".png"))
        print(f"tier {len(tier)}: {half} validation, {len(tier) - half} test")

    for folder in ["validation/defective", "test/defective"]:
        n = len(list((DEST / folder).glob("*.png")))
        print(f"{folder}: {n}")


if __name__ == "__main__":
    main()
