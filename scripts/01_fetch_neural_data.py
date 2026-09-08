#!/usr/bin/env python3
"""Download (via AllenSDK cache) and process Allen Brain Observatory natural_scenes sessions.

For each session listed in configs/dataset.yaml:
  - downloads/caches the NWB file (AllenSDK BrainObservatoryCache)
  - extracts trial-level and image-averaged dF/F responses
  - computes per-cell split-half reliability
  - saves to data/processed/neural/<area>/<experiment_id>_{trials,images,reliability}.parquet

Also extracts and hashes the 118 natural_scenes stimulus images once (shared across all sessions
using this stimulus set) to data/processed/stimuli/natural_scenes/.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuromap.neural import extract_trial_responses, average_by_image, split_half_reliability, NeuralPreprocessConfig
from neuromap.provenance import git_sha, hash_array, write_manifest


def extract_stimulus_images(dataset, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    template = dataset.get_stimulus_template("natural_scenes")  # (118, H, W) float32, values shared across sessions
    hashes = {}
    for i, img in enumerate(template):
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        path = out_dir / f"scene_{i:03d}.png"
        if not path.exists():
            Image.fromarray(img_u8).save(path)
        hashes[str(i)] = hash_array(img_u8)
    return hashes


def main():
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache

    cfg_path = ROOT / "configs" / "dataset.yaml"
    with open(cfg_path) as f:
        dataset_cfg = yaml.safe_load(f)

    boc_dir = ROOT / "data" / "raw" / "allen_boc"
    boc = BrainObservatoryCache(manifest_file=str(boc_dir / "manifest.json"))

    preprocess_cfg = NeuralPreprocessConfig()
    stim_out = ROOT / "data" / "processed" / "stimuli" / "natural_scenes"

    session_records = []
    stim_hashes = None

    for area, sessions in dataset_cfg["areas"].items():
        area_dir = ROOT / "data" / "processed" / "neural" / area
        area_dir.mkdir(parents=True, exist_ok=True)

        for sess in sessions:
            exp_id = sess["experiment_id"]
            trials_path = area_dir / f"{exp_id}_trials.parquet"
            images_path = area_dir / f"{exp_id}_images.parquet"
            reliab_path = area_dir / f"{exp_id}_reliability.parquet"

            if trials_path.exists() and images_path.exists() and reliab_path.exists():
                print(f"[skip, cached] {area}/{exp_id}")
                trial_df = pd.read_parquet(trials_path)
                n_cells = len([c for c in trial_df.columns if c.startswith("cell_")])
                session_records.append({"area": area, "experiment_id": exp_id, "n_cells": n_cells,
                                         "n_trials": len(trial_df)})
                continue

            print(f"[fetch] {area}/{exp_id}")
            ds = None
            for attempt in range(5):
                try:
                    ds = boc.get_ophys_experiment_data(exp_id)
                    break
                except Exception as e:
                    print(f"    retry {attempt+1}/5 after error: {e}")
            if ds is None:
                print(f"    giving up on {exp_id} this run, will retry on next invocation")
                continue

            if stim_hashes is None:
                stim_hashes = extract_stimulus_images(ds, stim_out)

            trial_df = extract_trial_responses(ds, preprocess_cfg)
            image_df = average_by_image(trial_df, preprocess_cfg)
            reliab = split_half_reliability(trial_df, preprocess_cfg)

            trial_df.to_parquet(trials_path)
            image_df.to_parquet(images_path)
            reliab.to_frame().to_parquet(reliab_path)

            session_records.append({
                "area": area, "experiment_id": exp_id,
                "n_cells": len([c for c in trial_df.columns if c.startswith("cell_")]),
                "n_trials": len(trial_df),
                "n_images": len(image_df),
                "mean_reliability": float(np.nanmean(reliab.values)),
            })

    if stim_hashes is None:
        # all sessions were cached from a prior run; still need stimulus hashes for the manifest
        existing = list(stim_out.glob("scene_*.png"))
        if existing:
            stim_hashes = {p.stem.split("_")[1]: hash_array(np.array(Image.open(p))) for p in existing}

    write_manifest(ROOT / "data" / "manifests" / "neural_data_manifest.json", {
        "dataset": dataset_cfg["dataset"],
        "source": dataset_cfg["source"],
        "stimulus": dataset_cfg["stimulus"],
        "preprocessing_config": preprocess_cfg.as_dict(),
        "stimulus_image_hashes": stim_hashes,
        "sessions": session_records,
        "n_sessions": len(session_records),
    })
    print(f"Done. {len(session_records)} sessions processed. Manifest written.")


if __name__ == "__main__":
    main()
