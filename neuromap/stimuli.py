"""Stimulus loading + deterministic preprocessing.

Stimulus identity is guaranteed by the sha256 hashes recorded in
data/manifests/neural_data_manifest.json (computed on the raw uint8 pixel arrays
straight from AllenSDK's stimulus template, before any resizing). All model-facing
transforms below are deterministic (no random crop/flip) so evaluation stimuli are
never accidentally augmented.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Deterministic resize target for all torch model families.
MODEL_INPUT_SIZE = 224

torch_transform = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_stimulus_set(stim_dir: Path) -> list[Image.Image]:
    """Loads scene_000.png ... scene_117.png in index order. Order == frame id == column order
    used everywhere else in the pipeline (neural response tables index by the same frame id)."""
    paths = sorted(Path(stim_dir).glob("scene_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"No stimulus images found in {stim_dir}; run scripts/01_fetch_neural_data.py first")
    return [Image.open(p).convert("L") for p in paths]


def stimuli_to_tensor_batch(images: list[Image.Image]) -> torch.Tensor:
    return torch.stack([torch_transform(im) for im in images], dim=0)


def stimuli_to_pixel_baseline(images: list[Image.Image], size: int = 32) -> np.ndarray:
    """Raw downsampled grayscale pixels, flattened. A deliberately dumb baseline."""
    out = []
    resize = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)
    for im in images:
        arr = np.asarray(resize(im), dtype=np.float32) / 255.0
        out.append(arr.flatten())
    return np.stack(out, axis=0)
