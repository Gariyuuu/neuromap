#!/usr/bin/env python3
"""RSA (bootstrap CI) and linear CKA between each session's neural population RDM and
every model-layer's RDM. Canonical result: results/rsa/rsa_results.parquet."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuromap.rsa import rsa_bootstrap_ci, linear_cka
from neuromap.provenance import git_sha

FEATURES_DIR = ROOT / "features"
NEURAL_DIR = ROOT / "data" / "processed" / "neural"
OUT_PATH = ROOT / "results" / "rsa" / "rsa_results.parquet"
N_BOOT = 1000
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
    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    done_keys = set(zip(existing.get("area", []), existing.get("experiment_id", []),
                         existing.get("model", []), existing.get("layer", [])))
    rows = list(existing.to_dict("records"))

    for area, sessions in dataset_cfg["areas"].items():
        for sess in sessions:
            exp_id = sess["experiment_id"]
            images_path = NEURAL_DIR / area / f"{exp_id}_images.parquet"
            if not images_path.exists():
                print(f"[missing neural data, skip] {area}/{exp_id}")
                continue
            image_df = pd.read_parquet(images_path)
            cell_cols = [c for c in image_df.columns if c.startswith("cell_")]
            frame_ids = image_df.index.to_numpy()
            Y = image_df[cell_cols].to_numpy()

            for (model, layer), (feat, meta) in feature_sets.items():
                key = (area, exp_id, model, layer)
                if key in done_keys:
                    continue
                X = feat[frame_ids]
                rsa = rsa_bootstrap_ci(Y, X, n_boot=N_BOOT, seed=SEED)
                cka = linear_cka(Y, X)
                rows.append({
                    "area": area, "experiment_id": exp_id, "model": model, "layer": layer,
                    "family": meta["family"], "task_objective": meta["task_objective"],
                    "rsa_rho": rsa["rho"], "rsa_ci_lo": rsa["ci_lo"], "rsa_ci_hi": rsa["ci_hi"],
                    "rsa_n_boot": rsa["n_boot"], "cka": cka,
                    "n_neurons": Y.shape[1], "n_images": Y.shape[0], "git_sha": git_sha(),
                })
                print(f"  {area}/{exp_id} {model}/{layer}: RSA rho={rsa['rho']:.3f} [{rsa['ci_lo']:.3f},{rsa['ci_hi']:.3f}], CKA={cka:.3f}")

            pd.DataFrame(rows).to_parquet(OUT_PATH)

    pd.DataFrame(rows).to_parquet(OUT_PATH)
    print(f"Done. {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
