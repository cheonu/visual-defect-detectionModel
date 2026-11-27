"""Build the VisA pcb1 dataset into the pipeline layout.

Splits the raw Normal/Anomaly images into train / validation / test.
Deterministic: filenames are sorted, then the first N go to each bucket.

Run:  .venv/bin/python build_visa.py
"""

import shutil
from pathlib import Path

SRC = Path("data/visa_pcb1/pcb1/Data/Images")
DEST = Path("data/visa_pcb1")

TRAIN_N = 300
VAL_GOOD_N = 40
VAL_BAD_N = 20
TEST_GOOD_N = 60
TEST_BAD_N = 80


def copy_group(files, dest_folder):
    dest = DEST / dest_folder
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, dest / f.name)
    return len(files)


def main():
    normal = sorted((SRC / "Normal").glob("*.JPG"))
    anomaly = sorted((SRC / "Anomaly").glob("*.JPG"))

    train = normal[:TRAIN_N]
    val_good = normal[TRAIN_N:TRAIN_N + VAL_GOOD_N]
    test_good = normal[TRAIN_N + VAL_GOOD_N:TRAIN_N + VAL_GOOD_N + TEST_GOOD_N]
    val_bad = anomaly[:VAL_BAD_N]
    test_bad = anomaly[VAL_BAD_N:VAL_BAD_N + TEST_BAD_N]

    print(f"train/good            {copy_group(train, 'train/good')}")
    print(f"validation/good       {copy_group(val_good, 'validation/good')}")
    print(f"validation/defective  {copy_group(val_bad, 'validation/defective')}")
    print(f"test/good             {copy_group(test_good, 'test/good')}")
    print(f"test/defective        {copy_group(test_bad, 'test/defective')}")


if __name__ == "__main__":
    main()
