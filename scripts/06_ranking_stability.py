#!/usr/bin/env python3
"""RQ5/H4: are model rankings for neural alignment stable across metrics, stimulus areas,
and sessions? Builds rank matrices (rows = "raters" — area or metric or session; columns =
models) and computes Kendall's W and pairwise Spearman agreement for each grouping.

Also addresses RQ4 (does ImageNet accuracy predict neural alignment) via Spearman
correlation between each model's approximate ImageNet top-1 accuracy and its mean
encoding score, across the non-baseline, non-self-supervised model set (self-supervised
models have no comparable ImageNet-classification objective, and are analyzed separately
under H3).

Canonical result: results/encoding/ranking_stability.json
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuromap.stats import kendall_w_ranking_agreement, pairwise_spearman_ranking_agreement
from neuromap.provenance import write_manifest

# Approximate torchvision-reported ImageNet-1k top-1 accuracies for the supervised models
# used here (recorded, not fabricated — from torchvision model weight metadata).
IMAGENET_TOP1 = {
    "resnet18": 69.758,
    "resnet50": 80.858,
    "vit_b_16": 81.072,
}

BEST_LAYER_METRIC = "mean_r"  # use each model's best layer per session, then average over sessions


def best_layer_per_model(df: pd.DataFrame, score_col: str, group_cols: list) -> pd.DataFrame:
    """For each (group_cols..., model), keep the row with the max score_col across layers."""
    idx = df.groupby(group_cols + ["model"])[score_col].idxmax()
    return df.loc[idx]


def rank_matrix_from_long(df: pd.DataFrame, rater_col: str, item_col: str, score_col: str):
    pivot = df.pivot_table(index=rater_col, columns=item_col, values=score_col, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")  # only raters with a score for every model
    items = pivot.columns.tolist()
    return pivot.to_numpy(), pivot.index.tolist(), items


def main():
    enc_path = ROOT / "results" / "encoding" / "encoding_results.parquet"
    rsa_path = ROOT / "results" / "rsa" / "rsa_results.parquet"
    if not enc_path.exists():
        print("No encoding results yet; run scripts/03_run_encoding.py first")
        return

    enc = pd.read_parquet(enc_path)
    enc = enc[enc["model"] != "mean_baseline"]
    enc_best = best_layer_per_model(enc, "mean_r", ["area", "experiment_id"])

    report = {}

    # (1) ranking stability ACROSS AREAS, within the ridge-encoding metric
    rank_mat, areas, models = rank_matrix_from_long(enc_best, "area", "model", "mean_r")
    if rank_mat.shape[0] >= 2 and rank_mat.shape[1] >= 2:
        report["ranking_stability_across_areas_encoding"] = {
            "kendalls_w": kendall_w_ranking_agreement(rank_mat),
            **pairwise_spearman_ranking_agreement(rank_mat),
            "raters": areas, "items": models,
        }

    # (2) ranking stability ACROSS SESSIONS (subject-level), within the ridge-encoding metric
    enc_best["session_key"] = enc_best["area"] + "_" + enc_best["experiment_id"].astype(str)
    rank_mat2, sessions, models2 = rank_matrix_from_long(enc_best, "session_key", "model", "mean_r")
    if rank_mat2.shape[0] >= 2 and rank_mat2.shape[1] >= 2:
        report["ranking_stability_across_sessions_encoding"] = {
            "kendalls_w": kendall_w_ranking_agreement(rank_mat2),
            **pairwise_spearman_ranking_agreement(rank_mat2),
            "n_sessions": len(sessions), "items": models2,
        }

    # (3) ranking stability ACROSS METRICS (encoding r vs RSA rho vs CKA), pooled over sessions
    if rsa_path.exists():
        rsa = pd.read_parquet(rsa_path)
        rsa_best = best_layer_per_model(rsa, "rsa_rho", ["area", "experiment_id"])
        cka_best = best_layer_per_model(rsa, "cka", ["area", "experiment_id"])

        enc_avg = enc_best.groupby("model")["mean_r"].mean()
        rsa_avg = rsa_best.groupby("model")["rsa_rho"].mean()
        cka_avg = cka_best.groupby("model")["cka"].mean()

        common_models = sorted(set(enc_avg.index) & set(rsa_avg.index) & set(cka_avg.index))
        if len(common_models) >= 3:
            metric_mat = np.stack([
                enc_avg.loc[common_models].to_numpy(),
                rsa_avg.loc[common_models].to_numpy(),
                cka_avg.loc[common_models].to_numpy(),
            ])
            report["ranking_stability_across_metrics"] = {
                "kendalls_w": kendall_w_ranking_agreement(metric_mat),
                **pairwise_spearman_ranking_agreement(metric_mat),
                "raters": ["ridge_encoding", "rsa_spearman", "linear_cka"], "items": common_models,
            }

    # (4) RQ4: does ImageNet top-1 accuracy predict mean neural encoding score?
    supervised = enc_best[enc_best["model"].isin(IMAGENET_TOP1.keys())]
    per_model_mean = supervised.groupby("model")["mean_r"].mean()
    if len(per_model_mean) >= 3:
        accs = [IMAGENET_TOP1[m] for m in per_model_mean.index]
        rho, p = spearmanr(accs, per_model_mean.to_numpy())
        report["imagenet_accuracy_vs_neural_alignment"] = {
            "spearman_rho": float(rho), "p_value": float(p),
            "models": per_model_mean.index.tolist(),
            "imagenet_top1": accs, "mean_encoding_r": per_model_mean.to_numpy().tolist(),
        }

    # (5) H3: self-supervised vs supervised, same architecture family (ViT), best layer
    vit_sup = enc_best[enc_best["model"] == "vit_b_16"]["mean_r"]
    vit_ssl = enc_best[enc_best["model"] == "dino_vits16"]["mean_r"]
    if len(vit_sup) > 0 and len(vit_ssl) > 0:
        report["self_supervised_vs_supervised_vit"] = {
            "supervised_vit_b_16_mean_r": float(vit_sup.mean()),
            "self_supervised_dino_vits16_mean_r": float(vit_ssl.mean()),
        }

    write_manifest(ROOT / "results" / "encoding" / "ranking_stability.json", report)
    print("Ranking stability report written.")
    for k, v in report.items():
        print(f"\n{k}:")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")


if __name__ == "__main__":
    main()
