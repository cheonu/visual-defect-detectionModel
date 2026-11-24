"""Unit 0b, last step: carve a held-out split out of the test set.

The held-out set is the one drawer we promise not to open until the final
evaluation (Unit 3). It proves the reported numbers weren't tuned on the
images we report on.

Rule: move a fixed 20% slice of test/good and of each defect type into
test/heldout/. The slice is deterministic — for each group we move the LAST
files in sorted-name order — so the split is reproducible and traceable.

Run:  .venv/bin/python split_heldout.py
"""

from pathlib import Path

DEST = Path("data/mvtec_bottle")
HOLDOUT = 4          # 20% of test/good (20) and of each defect type (~20-22)


def main():
    (DEST / "test/heldout/good").mkdir(parents=True, exist_ok=True)
    (DEST / "test/heldout/defective").mkdir(parents=True, exist_ok=True)

    # guard: this split should only ever run once
    if any((DEST / "test/heldout").rglob("*.png")):
        print("heldout already has images; nothing moved")
        return

    moved = 0

    good = sorted((DEST / "test/good").glob("*.png"))
    for path in good[-HOLDOUT:]:
        path.rename(DEST / "test/heldout/good" / path.name)
        moved += 1

    for defect_type in ("broken_large", "broken_small", "contamination"):
        group = sorted((DEST / "test/defective").glob(f"{defect_type}_*.png"))
        for path in group[-HOLDOUT:]:
            path.rename(DEST / "test/heldout/defective" / path.name)
            moved += 1

    print(f"moved {moved} images into test/heldout/")

    for folder in ("test/good", "test/defective",
                   "test/heldout/good", "test/heldout/defective"):
        n = len(list((DEST / folder).glob("*.png")))
        print(f"  {folder:>24}  {n}")


if __name__ == "__main__":
    main()
