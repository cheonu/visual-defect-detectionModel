"""Rig check for the bench demo (Unit 0.2).

Answers one question: are 20 photos of the same good part similar enough that a
difference between two photos can mean "different part" instead of "different
moment"?

No model, no dataset. Every photo is shrunk to 1024 grey levels, then compared
level by level. Run it with:

    .venv/bin/python rig_check.py photos/rig_01    # real photos
    .venv/bin/python rig_check.py --selftest       # prove the maths, no photos
"""

import sys
from pathlib import Path

from PIL import Image

SIZE = 32               # fingerprint is 32 x 32 grey levels per photo
UNSTABLE_LEVEL = 40     # a pixel shifting this many grey levels is suspicious
UNSTABLE_SHARE = 0.02   # this share of unstable pixels means "reshoot"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def fingerprint(path):
    """One image file -> list of 1024 grey levels (0 black .. 255 white), row by row."""
    with Image.open(path) as image:
        small = image.convert("L").resize((SIZE, SIZE))
    return list(small.tobytes())


def stack_ranges(photos):
    """photos: dict of filename -> list of 1024 grey levels.

    Return a list of 1024 numbers. Number i is the largest value pixel i reached
    in any photo, minus the smallest value it reached in any photo.
    """
    photo_lists = list(photos.values())
    ranges = []
    for i in range(len(photo_lists[0])):  
        highest = -1
        lowest = 256
        for levels in photo_lists:
            value = levels[i]
            if value > highest:
                highest = value
            if value < lowest:
                lowest = value

        ranges.append(highest-lowest)
    return ranges


def mean_level(levels):
    return sum(levels) / len(levels)


def save_drift_map(ranges):
    """Write rig_drift.png: bright areas are the pixels that moved between shots.

    Saved to reports/ (NOT the input folder), so the next run doesn't read the
    drift map back as if it were another photo of the part.
    """
    output = Path("reports") / "rig_drift.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    brightened = [min(255, level * 4) for level in ranges]
    map_image = Image.new("L", (SIZE, SIZE))
    map_image.putdata(brightened)
    map_image.resize((320, 320), Image.Resampling.NEAREST).save(output)
    return output


def report(photos, folder):
    ranges = stack_ranges(photos)
    means = [mean_level(levels) for levels in photos.values()]

    print(f"photos            {len(photos)}  ({folder})")
    print(f"brightness        min {min(means):.1f}   max {max(means):.1f}   spread {max(means) - min(means):.1f}")

    worst = max(ranges)
    worst_index = ranges.index(worst)
    print(f"worst pixel       position {worst_index} (row {worst_index // SIZE}, col {worst_index % SIZE}) moved {worst}")

    unstable = sum(1 for level in ranges if level >= UNSTABLE_LEVEL)
    share = unstable / len(ranges)
    print(f"unstable pixels   {unstable} of {len(ranges)} ({share * 100:.1f}%)")

    if share <= UNSTABLE_SHARE:
        print("verdict           rig is stable, shoot the dataset")
    else:
        print("verdict           rig drifted, fix height and lamp, reshoot")

    drift_map = save_drift_map(ranges)
    print(f"drift map         {drift_map}")


def selftest():
    """Three fake 2x2 photos. Only pixel 0 changes, from 10 to 30."""
    photos = {
        "a": [10, 50, 50, 50],
        "b": [20, 50, 50, 50],
        "c": [30, 50, 50, 50],
    }
    got = stack_ranges(photos)
    expected = [20, 0, 0, 0]
    assert got == expected, f"expected {expected}, got {got}"
    print("selftest ok:", got)


def main():
    if "--selftest" in sys.argv:
        selftest()
        return

    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("photos/rig_01")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(f"no jpg or png files in {folder}")
        return

    photos = {path.name: fingerprint(path) for path in paths}
    report(photos, folder)


if __name__ == "__main__":
    main()
