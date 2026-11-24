"""Unit 2a: show the nearest-good distance for one good and one defective part.

    .venv/bin/python demo_nearest.py
"""

from pathlib import Path

import nearest
from embedder import Embedder
import config


def main():
    embedder = Embedder()
    names, vectors = nearest.load_store()

    for label, folder in [("good", config.TEST_GOOD),
                          ("defective", config.TEST_DEFECTIVE)]:
        path = sorted(Path(folder).glob("*.png"))[0]
        query = embedder.embed(path)
        distance = nearest.nearest_distance(query, vectors)
        print(f"{label:>9}   {path.name:>22}   nearest-good distance {distance:.3f}")


if __name__ == "__main__":
    main()
