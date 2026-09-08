#!/usr/bin/env python3
"""Export compact JSON artifacts for the Next.js site from the canonical result parquets/JSON.
All heavy computation already happened offline; this script only reshapes+trims. Output goes
to site/public/data/ so it's served as static JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuromap.provenance import git_sha

OUT_DIR = ROOT / "site" / "public" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def jd(obj, path):
    with open(OUT_DIR / path, "w") as f:
        json.dump(obj, f, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print(f"wrote {path}")


def main():
    with open(ROOT / "configs" / "dataset.yaml") as f:
        dataset_cfg = yaml.safe_load(f)
    with open(ROOT / "data" / "manifests" / "neural_data_manifest.json") as f:
        neural_manifest = json.load(f)

    enc = pd.read_parquet(ROOT / "results" / "encoding" / "encoding_results.parquet")
    enc_models = enc[enc["model"] != "mean_baseline"].copy()

    rsa_path = ROOT / "results" / "rsa" / "rsa_results.parquet"
    rsa = pd.read_parquet(rsa_path) if rsa_path.exists() else pd.DataFrame()

    comp_path = ROOT / "results" / "compression" / "compression_results.parquet"
    comp = pd.read_parquet(comp_path) if comp_path.exists() else pd.DataFrame()

    ranking_path = ROOT / "results" / "encoding" / "ranking_stability.json"
    ranking = json.load(open(ranking_path)) if ranking_path.exists() else {}

    failure_path = ROOT / "results" / "failures" / "failure_analysis.json"
    failures = json.load(open(failure_path)) if failure_path.exists() else {}

    # --- overview ---
    jd({
        "dataset": neural_manifest["dataset"], "source": neural_manifest["source"],
        "stimulus": neural_manifest["stimulus"], "n_sessions": neural_manifest["n_sessions"],
        "n_areas": len(dataset_cfg["areas"]), "areas": list(dataset_cfg["areas"].keys()),
        "n_models": enc_models["model"].nunique(), "models": sorted(enc_models["model"].unique().tolist()),
        "n_neurons_total": int(enc.groupby(["area", "experiment_id"])["n_neurons"].first().sum()),
        "git_sha": git_sha(),
    }, "overview.json")

    # --- datasets (data card) ---
    jd({
        "preprocessing_config": neural_manifest["preprocessing_config"],
        "sessions": neural_manifest["sessions"],
        "dataset_config": dataset_cfg,
    }, "datasets.json")

    # --- brain area x model heatmap ---
    best_layer = enc_models.loc[enc_models.groupby(["area", "experiment_id", "model"])["mean_r"].idxmax()]
    area_model = best_layer.groupby(["area", "model"]).agg(
        mean=("mean_r", "mean"), std=("mean_r", "std"), count=("mean_r", "count"),
        family=("family", "first"),
    ).reset_index()
    jd(area_model.to_dict(orient="records"), "brain_area_model_heatmap.json")

    # --- per-session detail (for the neural explorer) ---
    session_detail = best_layer[["area", "experiment_id", "model", "layer", "mean_r", "n_neurons", "n_images"]]
    jd(session_detail.to_dict(orient="records"), "session_detail.json")

    # --- layer profiles (RQ2/RQ3: layer depth vs area) ---
    layer_profile = enc_models.groupby(["area", "model", "layer"])["mean_r"].mean().reset_index()
    jd(layer_profile.to_dict(orient="records"), "layer_profiles.json")

    # --- RSA matrices ---
    if not rsa.empty:
        rsa_best = rsa.loc[rsa.groupby(["area", "experiment_id", "model"])["rsa_rho"].idxmax()]
        jd(rsa_best[["area", "experiment_id", "model", "layer", "rsa_rho", "rsa_ci_lo", "rsa_ci_hi", "cka"]]
           .to_dict(orient="records"), "rsa_results.json")

    # --- compression curves ---
    if not comp.empty:
        jd(comp.to_dict(orient="records"), "compression_results.json")

    # --- ranking stability ---
    jd(ranking, "ranking_stability.json")

    # --- failure analysis ---
    jd(failures, "failure_analysis.json")

    print("Site data export complete.")


if __name__ == "__main__":
    main()
