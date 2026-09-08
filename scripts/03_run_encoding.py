#!/usr/bin/env python3
"""Run ridge-regression encoding models for every (session, model-layer) pair, plus the
mean-response and pixel/Gabor baselines. Canonical result: results/encoding/encoding_results.parquet,
one row per (area, experiment_id, model, layer) with per-neuron correlations and provenance.

Resumable: rows already present (matched on area/experiment_id/model/layer) are skipped.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuromap.encoding import run_encoding, mean_response_baseline
from neuromap.provenance import git_sha, hash_config

FEATURES_DIR = ROOT / "features"
NEURAL_DIR = ROOT / "data" / "processed" / "neural"
OUT_PATH = ROOT / "results" / "encoding" / "encoding_results.parquet"
N_OUTER_FOLDS = 5
N_PCA_COMPONENTS = 50
SEED = 0


def load_feature_sets():
    sets = {}
    for path in sorted(FEATURES_DIR.glob("*.npy")):
        model, layer = path.stem.split("__")
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        sets[(model, layer)] = (np.load(path), meta)
    return sets


def main():
    with open(ROOT / "configs" / "dataset.yaml") as f:
        dataset_cfg = yaml.safe_load(f)

    feature_sets = load_feature_sets()
    print(f"Loaded {len(feature_sets)} feature sets: {list(feature_sets.keys())}")

    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    done_keys = set(zip(existing.get("area", []), existing.get("experiment_id", []),
                         existing.get("model", []), existing.get("layer", [])))

    rows = list(existing.to_dict("records"))
    run_cfg = {"n_outer_folds": N_OUTER_FOLDS, "n_pca_components": N_PCA_COMPONENTS, "seed": SEED}
    cfg_hash = hash_config(run_cfg)

    for area, sessions in dataset_cfg["areas"].items():
        for sess in sessions:
            exp_id = sess["experiment_id"]
            images_path = NEURAL_DIR / area / f"{exp_id}_images.parquet"
            reliab_path = NEURAL_DIR / area / f"{exp_id}_reliability.parquet"
            if not images_path.exists():
                print(f"[missing neural data, skip] {area}/{exp_id}")
                continue

            image_df = pd.read_parquet(images_path)  # index = frame id (0..117), columns = cell_*
            reliab = pd.read_parquet(reliab_path)["split_half_reliability"]
            cell_cols = [c for c in image_df.columns if c.startswith("cell_")]

            frame_ids = image_df.index.to_numpy()
            Y = image_df[cell_cols].to_numpy()

            if (area, exp_id, "mean_baseline", "none") not in done_keys:
                res = mean_response_baseline(Y, n_outer_folds=N_OUTER_FOLDS, seed=SEED)
                rows.append({
                    "area": area, "experiment_id": exp_id, "model": "mean_baseline", "layer": "none",
                    "family": "baseline", "task_objective": "none", "weights_id": "none",
                    "mean_r": res.mean_r, "per_neuron_r": res.per_neuron_r.tolist(),
                    "reliability": reliab.reindex(cell_cols).tolist(),
                    "n_neurons": res.n_neurons, "n_images": res.n_images, "n_features": 0,
                    "run_config_hash": cfg_hash, "git_sha": git_sha(),
                })
                print(f"  {area}/{exp_id} mean_baseline: r={res.mean_r:.4f}")

            for (model, layer), (feat, meta) in feature_sets.items():
                key = (area, exp_id, model, layer)
                if key in done_keys:
                    continue
                X = feat[frame_ids]  # align model features to this session's image order
                res = run_encoding(X, Y, n_outer_folds=N_OUTER_FOLDS, n_pca_components=N_PCA_COMPONENTS, seed=SEED)
                rows.append({
                    "area": area, "experiment_id": exp_id, "model": model, "layer": layer,
                    "family": meta["family"], "task_objective": meta["task_objective"],
                    "weights_id": meta["weights_id"],
                    "mean_r": res.mean_r, "per_neuron_r": res.per_neuron_r.tolist(),
                    "reliability": reliab.reindex(cell_cols).tolist(),
                    "n_neurons": res.n_neurons, "n_images": res.n_images, "n_features": res.n_features,
                    "run_config_hash": cfg_hash, "git_sha": git_sha(),
                })
                print(f"  {area}/{exp_id} {model}/{layer}: r={res.mean_r:.4f}")

            # checkpoint after every session so a crash doesn't lose completed work
            pd.DataFrame(rows).to_parquet(OUT_PATH)

    pd.DataFrame(rows).to_parquet(OUT_PATH)
    print(f"Done. {len(rows)} total (session, model, layer) rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
