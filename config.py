"""The knobs.

Change these numbers here, in one place, instead of hunting through the code.
"""

IMAGE_SIZE = 224                    # every image is resized to this square before embedding
DATA_DIR = "data/mvtec_bottle"      # where the downloaded images live

TRAIN_GOOD = f"{DATA_DIR}/train/good"
TEST_GOOD = f"{DATA_DIR}/test/good"
TEST_DEFECTIVE = f"{DATA_DIR}/test/defective"
HELDOUT_GOOD = f"{DATA_DIR}/test/heldout/good"
HELDOUT_DEFECTIVE = f"{DATA_DIR}/test/heldout/defective"
