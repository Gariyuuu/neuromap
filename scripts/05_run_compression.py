#!/usr/bin/env python3
"""Compression experiment (RQ6): progressively PCA-reduce each DEEP model-layer's
activations and re-measure neural predictivity + RSA. Restricted to the learned model
families (RQ6 asks about "compressed model representations", not classical baselines)
and to one representative session per area, to keep the (dims x feature-sets x sessions)
grid tractable on a single machine — this is a scope-management choice, documented here
rather than left implicit, not a silent shortcut. The full 18-session grid can be run by
setting SESSIONS_PER_AREA = None below.

Canonical result: results/compression/compression_results.parquet.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuromap.encoding import run_encoding
from neuromap.rsa import rsa_compare, rdm_correlation_distance
from neuromap.provenance import git_sha

FEATURES_DIR = ROOT / "features"
NEURAL_DIR = ROOT / "data" / "processed" / "neural"
OUT_PATH = ROOT / "results" / "compression" / "compression_results.parquet"
DIMS = [4, 16, 64]
MIN_NATIVE_DIM = 32
DEEP_MODEL_FAMILIES = {"cnn_supervised", "vit_supervised", "self_supervised"}
SESSIONS_PER_AREA = 1  # None = use all sessions configured in configs/dataset.yaml
SEED = 0


def load_feature_sets():
    sets = {}
    for path in sorted(FEATURES_DIR.glob("*.npy")):
        model, layer = path.stem.split("__")
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        if meta["family"] not in DEEP_MODEL_FAMILIES:
            continue
        arr = np.load(path)
        if arr.shape[1] >= MIN_NATIVE_DIM:
            sets[(model, layer)] = (arr, meta)
    return sets


def main():
    with open(ROOT / "configs" / "dataset.yaml") as f:
        dataset_cfg = yaml.safe_load(f)

    feature_sets = load_feature_sets()
    print(f"Compressing {len(feature_sets)} eligible feature sets across dims {DIMS}")

    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    done_keys = set(zip(existing.get("area", []), existing.get("experiment_id", []),
                         existing.get("model", []), existing.get("layer", []), existing.get("n_dims", [])))
    rows = list(existing.to_dict("records"))

    for area, sessions in dataset_cfg["areas"].items():
        sessions_to_run = sessions[:SESSIONS_PER_AREA] if SESSIONS_PER_AREA else sessions
        for sess in sessions_to_run:
            exp_id = sess["experiment_id"]
            images_path = NEURAL_DIR / area / f"{exp_id}_images.parquet"
            if not images_path.exists():
                continue
            image_df = pd.read_parquet(images_path)
            cell_cols = [c for c in image_df.columns if c.startswith("cell_")]
            frame_ids = image_df.index.to_numpy()
            Y = image_df[cell_cols].to_numpy()

            for (model, layer), (feat, meta) in feature_sets.items():
                X_full = feat[frame_ids]
                # full-dimensionality reference point (PCA to 50, matching the primary encoding run)
                dims_to_run = [d for d in DIMS if (area, exp_id, model, layer, d) not in done_keys]
                if not dims_to_run:
                    continue

                # variance retained is computed once on the full (train-set) PCA spectrum for reference
                pca_full = PCA(n_components=min(X_full.shape[0] - 1, X_full.shape[1]), random_state=SEED).fit(X_full)
                cum_var = np.cumsum(pca_full.explained_variance_ratio_)

                for d in dims_to_run:
                    t0 = time.time()
                    enc = run_encoding(X_full, Y, n_outer_folds=5, n_pca_components=d, seed=SEED)
                    runtime_s = time.time() - t0

                    pca_d = PCA(n_components=min(d, X_full.shape[0] - 1, X_full.shape[1]), random_state=SEED).fit(X_full)
                    X_compressed = pca_d.transform(X_full)
                    rsa_rho = rsa_compare(rdm_correlation_distance(Y), rdm_correlation_distance(X_compressed))
                    variance_retained = float(cum_var[min(d, len(cum_var)) - 1]) if len(cum_var) > 0 else float("nan")

                    rows.append({
                        "area": area, "experiment_id": exp_id, "model": model, "layer": layer,
                        "family": meta["family"], "n_dims": d, "native_dim": X_full.shape[1],
                        "variance_retained": variance_retained, "mean_r": enc.mean_r,
                        "rsa_rho": rsa_rho, "runtime_s": runtime_s,
                        "storage_bytes_per_image": d * 4, "git_sha": git_sha(),
                    })
                    print(f"  {area}/{exp_id} {model}/{layer} d={d}: r={enc.mean_r:.4f} var={variance_retained:.3f} rsa={rsa_rho:.3f}")

            pd.DataFrame(rows).to_parquet(OUT_PATH)

    pd.DataFrame(rows).to_parquet(OUT_PATH)
    print(f"Done. {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
