"""The knobs.

Change these numbers here, in one place, instead of hunting through the code.
"""

from pathlib import Path

IMAGE_SIZE = 224                    # every image is resized to this square before embedding
MODEL_NAME = "dinov2_vitl14"        # which DINOv2 variant: vits14 / vitb14 / vitl14
DATASET = "bottle_label"            # which dataset directory under data/

DATA_DIR = f"data/{DATASET}"

TRAIN_GOOD = f"{DATA_DIR}/train/good"
VALIDATION_GOOD = f"{DATA_DIR}/validation/good"
VALIDATION_DEFECTIVE = f"{DATA_DIR}/validation/defective"
TEST_GOOD = f"{DATA_DIR}/test/good"
TEST_DEFECTIVE = f"{DATA_DIR}/test/defective"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    """Sorted image files in a folder (.jpg/.jpeg/.png, case-insensitive)."""
    return sorted(p for p in Path(folder).glob("*")
                  if p.suffix.lower() in IMAGE_SUFFIXES)
