# Visual Defect Detection — Bench Demo

Overhead camera on parts, one defect class only, a detector that flags the bad
ones, and a shadow-mode report in the same format a paid pilot delivers.

**Status: Unit 0 (bench rig + dataset layout). No model yet, on purpose.**

## The rig check

Before any model: 20 photos of the *same good part*, fixed camera height, one
lamp in one position, plain background. If those 20 photos disagree, the bench
is the problem, not the model.

    .venv/bin/python rig_check.py --selftest      # proves the maths, no photos
    .venv/bin/python rig_check.py photos/rig_01   # your 20 real photos

Verdict `rig is stable` -> shoot the dataset. `rig drifted` -> fix height and
lamp, reshoot. Do not collect data on a drifting rig.

## Dataset layout (Unit 0)

    data/mvtec_bottle/
      train/good/              209 images
      test/good/                20 images
      test/defective/           63 images  (broken_large, broken_small, contamination)
      test/heldout/good/        small split copied out of test/good
      test/heldout/defective/   small split copied out of test/defective

Every folder gets a fixed share of the test set. **No training or threshold
step ever reads `test/heldout/`.** A number measured on data the model already
saw means nothing.

Verify the counts before trusting any download:

    209 train/good + 20 test/good + 63 test/defective

### MVTec licence — read this

MVTec AD is **CC BY-NC-SA 4.0, non-commercial.** Fine for this bench demo and
for learning. It must **not** be shipped inside a paid pilot or a client system,
and MVTec images must **not** appear in a client report as if they were the
client's product. The paid pilot always trains on the client's own products and
defects.

The official download page is https://www.mvtec.com/company/research/datasets/mvtec-ad
(URL has been unreliable since late 2025). HuggingFace mirrors are in
circulation as a fallback — verify the counts above before trusting one.

## Build rules

- One unit at a time, finish line before the next unit. See `PLAN.md`.
- Every result traces back to an image filename. No result without a filename.
- No new model architecture until the nearest-neighbour baseline is measured.
- The demo proves the pipeline and the reporting discipline, never a client result.
