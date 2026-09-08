#!/usr/bin/env python3
"""Extract artificial-model activations for the 118 natural_scenes stimuli.

Caches each (model, layer) feature matrix to features/<model>__<layer>.npy plus a
manifest recording model/weights/layer/preprocessing/stimulus-hash/git-SHA — a job is
skipped (not recomputed) if its cache file already exists (resumable, avoids
recomputation per the project's compute-management requirement).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuromap.stimuli import load_stimulus_set, stimuli_to_tensor_batch, stimuli_to_pixel_baseline
from neuromap.gabor import gabor_features
from neuromap.models import REGISTRY, extract_layer_activations
from neuromap.provenance import git_sha, hash_array

STIM_DIR = ROOT / "data" / "processed" / "stimuli" / "natural_scenes"
FEATURES_DIR = ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)


def save(name: str, layer: str, arr: np.ndarray, extra: dict):
    path = FEATURES_DIR / f"{name}__{layer}.npy"
    np.save(path, arr)
    manifest = {
        "model": name, "layer": layer, "shape": list(arr.shape),
        "feature_hash": hash_array(arr), "git_sha": git_sha(), **extra,
    }
    with open(FEATURES_DIR / f"{name}__{layer}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  saved {name}/{layer}: shape={arr.shape}")


def main():
    images = load_stimulus_set(STIM_DIR)
    stim_hash = hash_array(np.stack([np.asarray(im) for im in images]))
    print(f"Loaded {len(images)} stimuli, combined hash={stim_hash}")

    # --- classical baselines ---
    if not (FEATURES_DIR / "pixels__raw.npy").exists():
        print("[extract] pixels (32x32 downsample)")
        pix = stimuli_to_pixel_baseline(images, size=32)
        save("pixels", "raw", pix, {"family": "classical", "weights_id": "none",
                                     "task_objective": "none", "stimulus_hash": stim_hash})
    else:
        print("[skip, cached] pixels/raw")

    if not (FEATURES_DIR / "gabor__raw.npy").exists():
        print("[extract] gabor filter bank")
        gab = gabor_features(images)
        save("gabor", "raw", gab, {"family": "classical", "weights_id": "none",
                                    "task_objective": "none", "stimulus_hash": stim_hash})
    else:
        print("[skip, cached] gabor/raw")

    # --- pretrained deep models ---
    batch = stimuli_to_tensor_batch(images)
    for model_name, spec in REGISTRY.items():
        needed = [l for l in spec.layer_names if not (FEATURES_DIR / f"{model_name}__{l}.npy").exists()]
        if not needed:
            print(f"[skip, cached] {model_name} (all layers present)")
            continue
        print(f"[extract] {model_name}: layers {spec.layer_names}")
        acts = extract_layer_activations(spec, batch)
        for layer, arr in acts.items():
            save(model_name, layer, arr, {"family": spec.family, "weights_id": spec.weights_id,
                                           "task_objective": spec.task_objective, "stimulus_hash": stim_hash})

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
