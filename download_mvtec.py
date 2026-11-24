"""Unit 0b: download the MVTec AD bottle images and lay them out.

Source: the foersben/mvtec-ad mirror on HuggingFace, which keeps the original
MVTec AD folder layout. Verified Nov 20 2026: bottle = 209 train/good,
20 test/good, 63 defective (broken_large 20 + broken_small 22 + contamination 21).

Licence: CC BY-NC-SA 4.0, non-commercial. See README.md.

Run:  .venv/bin/python download_mvtec.py
"""

import json
import urllib.request
from pathlib import Path

REPO = "foersben/mvtec-ad"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/main/bottle?recursive=true"
RESOLVE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/bottle/"

DEST = Path("data/mvtec_bottle")

EXPECTED = {
    "train/good": 209,
    "test/good": 20,
    "test/defective": 63,
}


def fetch_paths():
    """Ask HuggingFace for every file path in the bottle/ folder."""
    with urllib.request.urlopen(TREE_URL) as response:
        entries = json.load(response)
    return [entry["path"] for entry in entries]


def local_path(source_path):
    """Map one HuggingFace path to its place in data/mvtec_bottle."""
    relative = source_path.split("bottle/", 1)[1]   # e.g. "test/broken_large/000.png"
    parts = relative.split("/")                       # ["test", "broken_large", "000.png"]
    filename = parts[-1]
    folder = "/".join(parts[:-1])                     # "test/broken_large"

    if folder == "train/good":
        return DEST / "train/good" / filename
    if folder == "test/good":
        return DEST / "test/good" / filename
    # any other test/... folder is one of the three defect types
    return DEST / "test/defective" / f"{parts[1]}_{filename}"


def download(source_path):
    """Fetch one image and write it to its local place. Skip if already there."""
    destination = local_path(source_path)
    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = RESOLVE_URL + source_path.split("bottle/", 1)[1]
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())
    return True


def main():
    all_paths = fetch_paths()
    images = [p for p in all_paths if p.endswith(".png") and "/ground_truth/" not in p]
    print(f"images to download: {len(images)}")

    downloaded = 0
    for path in images:
        if download(path):
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"  ...{downloaded}")
    print(f"downloaded: {downloaded}")

    for folder, want in EXPECTED.items():
        have = len(list((DEST / folder).glob("*.png")))
        mark = "ok" if have == want else "MISMATCH"
        print(f"  {folder:>16}  {have:>3}  want {want}  {mark}")


if __name__ == "__main__":
    main()
