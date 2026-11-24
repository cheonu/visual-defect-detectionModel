"""Unit 1b: the reusable Embedder class.

Bundles the model, the transform, and the size into one object, so later units
just call `Embedder().embed(path)` instead of re-loading the model each time.

The knobs live in config.py, not here.
"""

import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

import config

MEAN = [0.485, 0.456, 0.406]   # ImageNet channel averages (fixed for resnet18)
STD = [0.229, 0.224, 0.225]    # ImageNet channel spread (fixed for resnet18)


class Embedder:
    """Loads a pretrained network once, then turns images into vectors."""

    def __init__(self):
        self.size = config.IMAGE_SIZE
        self.model = self._load_model()
        self.transform = self._build_transform()

    def _load_model(self):
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()   # cut off the classifier, keep the summary
        model.eval()
        return model

    def _build_transform(self):
        return transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def embed(self, path):
        """One image path -> one 512-number vector of unit length."""
        image = Image.open(path).convert("RGB")          # open, force 3 channels
        tensor = self.transform(image).unsqueeze(0)      # preprocess + batch dim
        with torch.no_grad():                            # no gradient bookkeeping
            vector = self.model(tensor)
        vector = vector.squeeze(0).numpy()               # drop batch dim, plain array
        norm = np.linalg.norm(vector)                    # the vector's length
        return vector / norm                             # scale to length 1
