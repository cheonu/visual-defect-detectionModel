"""Unit 1b: prove the Embedder class works on one image.

    .venv/bin/python embed_one.py
"""

from embedder import Embedder

IMAGE = "data/mvtec_bottle/train/good/000.png"


def main():
    embedder = Embedder()
    vector = embedder.embed(IMAGE)
    print(f"image         {IMAGE}")
    print(f"vector shape  {vector.shape}")
    print(f"first 8 numbers  {[round(float(x), 3) for x in vector[:8]]}")


if __name__ == "__main__":
    main()
