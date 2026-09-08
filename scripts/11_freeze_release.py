#!/usr/bin/env python3
"""Freeze a research release: git SHA, data manifest, model manifest, config hashes,
metrics summary, figure list, and paper version, all in one JSON."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuromap.provenance import git_sha, hash_file


def main():
    release = {"git_sha": git_sha(), "git_sha_short": git_sha(short=True)}

    neural_manifest_path = ROOT / "data" / "manifests" / "neural_data_manifest.json"
    if neural_manifest_path.exists():
        release["neural_data_manifest"] = json.load(open(neural_manifest_path))

    feature_manifests = []
    for p in sorted((ROOT / "features").glob("*.json")):
        feature_manifests.append(json.load(open(p)))
    release["model_manifests"] = feature_manifests

    config_files = ["configs/dataset.yaml", "configs/hypotheses.yaml"]
    release["config_hashes"] = {c: hash_file(ROOT / c) for c in config_files if (ROOT / c).exists()}

    metrics = {}
    enc_path = ROOT / "results" / "encoding" / "encoding_results.parquet"
    if enc_path.exists():
        df = pd.read_parquet(enc_path)
        df = df[df["model"] != "mean_baseline"]
        metrics["encoding_mean_r_by_model"] = df.groupby("model")["mean_r"].mean().to_dict()
        metrics["n_encoding_rows"] = len(df)

    rsa_path = ROOT / "results" / "rsa" / "rsa_results.parquet"
    if rsa_path.exists():
        rsa_df = pd.read_parquet(rsa_path)
        metrics["rsa_mean_rho_by_model"] = rsa_df.groupby("model")["rsa_rho"].mean().to_dict()
        metrics["n_rsa_rows"] = len(rsa_df)

    comp_path = ROOT / "results" / "compression" / "compression_results.parquet"
    if comp_path.exists():
        metrics["n_compression_rows"] = len(pd.read_parquet(comp_path))

    release["metrics_summary"] = metrics
    release["figures"] = [p.name for p in sorted((ROOT / "figures").glob("*.png"))]
    release["paper_exists"] = (ROOT / "paper" / "paper.md").exists()
    release["paper_hash"] = hash_file(ROOT / "paper" / "paper.md") if release["paper_exists"] else None

    out_path = ROOT / "results" / "RELEASE.json"
    with open(out_path, "w") as f:
        json.dump(release, f, indent=2, default=str)
    print(f"Frozen release written to {out_path}")
    print(json.dumps({k: v for k, v in release.items() if k not in ("neural_data_manifest", "model_manifests")}, indent=2, default=str))


if __name__ == "__main__":
    main()
