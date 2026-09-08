"""Gabor filter-bank baseline: a classical V1-simple-cell-inspired feature space,
used as a non-learned baseline between raw pixels and pretrained deep nets."""
from PIL import Image

import numpy as np
from skimage.filters import gabor
from skimage.measure import block_reduce
from skimage.transform import resize as sk_resize

GABOR_SIZE = 128
ORIENTATIONS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
FREQUENCIES = [0.05, 0.15, 0.25]
POOL_GRID = 8  # each filter's energy map is average-pooled to POOL_GRID x POOL_GRID


def gabor_features(images: list[Image.Image]) -> np.ndarray:
    feats = []
    for im in images:
        arr = np.asarray(im.resize((GABOR_SIZE, GABOR_SIZE), Image.BILINEAR), dtype=np.float64) / 255.0
        filt_outs = []
        for theta in ORIENTATIONS:
            for freq in FREQUENCIES:
                real, imag = gabor(arr, frequency=freq, theta=theta)
                energy = np.sqrt(real**2 + imag**2)
                block = GABOR_SIZE // POOL_GRID
                pooled = block_reduce(energy, (block, block), np.mean)
                filt_outs.append(pooled.flatten())
        feats.append(np.concatenate(filt_outs))
    return np.stack(feats, axis=0)
