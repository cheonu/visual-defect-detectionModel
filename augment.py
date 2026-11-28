"""Light, deterministic augmentation for the good training set.

Each good image becomes several versions so the memory bank tolerates the small
placement/lighting variation a fixed rig actually produces. Applied ONLY to the
good training set — never to validation or test.

Deterministic (a fixed list, not random) so every run is reproducible.
"""

from PIL import Image, ImageEnhance


def _shift(im, dx, dy):
    return im.transform(im.size, Image.AFFINE, (1, 0, dx, 0, 1, dy),
                        resample=Image.BILINEAR)


# name -> function(image) -> image. Small perturbations only: no flips, no big rotations.
AUGMENTATIONS = [
    ("orig",        lambda im: im),
    ("rot+2",       lambda im: im.rotate(2, resample=Image.BILINEAR)),
    ("rot-2",       lambda im: im.rotate(-2, resample=Image.BILINEAR)),
    ("bright+10",   lambda im: ImageEnhance.Brightness(im).enhance(1.10)),
    ("bright-10",   lambda im: ImageEnhance.Brightness(im).enhance(0.90)),
    ("contrast+10", lambda im: ImageEnhance.Contrast(im).enhance(1.10)),
    ("contrast-10", lambda im: ImageEnhance.Contrast(im).enhance(0.90)),
    ("shift+3",     lambda im: _shift(im, 3, 0)),
    ("shift-3",     lambda im: _shift(im, -3, 0)),
]


def apply_all(image):
    """Yield (name, augmented_image) for every augmentation, including the original."""
    for name, fn in AUGMENTATIONS:
        yield name, fn(image)
