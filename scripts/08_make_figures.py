#!/usr/bin/env python3
"""Generate static figures for the paper from canonical result files.
Each figure is skipped (with a message) if its required result file doesn't exist yet."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0b0d10", "axes.facecolor": "#12161b",
    "savefig.facecolor": "#0b0d10", "text.color": "#e6ebf0",
    "axes.edgecolor": "#232a32", "axes.labelcolor": "#e6ebf0",
    "xtick.color": "#8b96a3", "ytick.color": "#8b96a3",
    "grid.color": "#232a32", "font.size": 10,
})

MODEL_ORDER = ["pixels", "gabor", "resnet18", "resnet50", "vit_b_16", "dino_vits16"]
MODEL_LABELS = {
    "pixels": "Pixels", "gabor": "Gabor", "resnet18": "ResNet-18", "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16", "dino_vits16": "DINO ViT-S/16",
}


def fig_area_model_heatmap():
    path = ROOT / "results" / "encoding" / "encoding_results.parquet"
    if not path.exists():
        print("[skip] area_model_heatmap: no encoding results"); return
    df = pd.read_parquet(path)
    df = df[df["model"] != "mean_baseline"]
    best = df.loc[df.groupby(["area", "experiment_id", "model"])["mean_r"].idxmax()]
    pivot = best.groupby(["area", "model"])["mean_r"].mean().unstack()
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([MODEL_LABELS[m] for m in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, label="Mean encoding r")
    ax.set_title("Brain area × model — mean held-out encoding score")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "area_model_heatmap.png", dpi=150)
    plt.close(fig)
    print("[ok] area_model_heatmap.png")


def fig_layer_depth_curves():
    path = ROOT / "results" / "encoding" / "encoding_results.parquet"
    if not path.exists():
        print("[skip] layer_depth_curves: no encoding results"); return
    df = pd.read_parquet(path)
    df = df[~df["model"].isin(["mean_baseline", "pixels", "gabor"])]
    layer_order = {
        "resnet18": ["layer1", "layer2", "layer3", "layer4", "avgpool"],
        "resnet50": ["layer1", "layer2", "layer3", "layer4", "avgpool"],
        "vit_b_16": ["block2", "block5", "block8", "block11"],
        "dino_vits16": ["block2", "block5", "block8", "block11"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    for ax, (model, layers) in zip(axes.flat, layer_order.items()):
        sub = df[df["model"] == model]
        for area, g in sub.groupby("area"):
            means = [g[g["layer"] == l]["mean_r"].mean() for l in layers]
            ax.plot(range(len(layers)), means, marker="o", label=area, linewidth=1.5)
        ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, rotation=30, ha="right", fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_ylabel("Mean encoding r")
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Layer depth vs. mean encoding score, by visual area")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "layer_depth_curves.png", dpi=150)
    plt.close(fig)
    print("[ok] layer_depth_curves.png")


def fig_noise_ceiling_comparison():
    manifest_path = ROOT / "data" / "manifests" / "neural_data_manifest.json"
    if not manifest_path.exists():
        print("[skip] noise_ceiling_comparison: no neural manifest"); return
    manifest = json.load(open(manifest_path))
    sessions = pd.DataFrame(manifest["sessions"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(sessions)), sessions["mean_reliability"], color="#3ecf8e")
    ax.set_xticks(range(len(sessions))); ax.set_xticklabels(sessions["area"] + "/" + sessions["experiment_id"].astype(str), rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Mean split-half reliability (Spearman-Brown)")
    ax.set_title("Per-session neural reliability (noise-ceiling proxy)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_ceiling_comparison.png", dpi=150)
    plt.close(fig)
    print("[ok] noise_ceiling_comparison.png")


def fig_rsa_matrix():
    path = ROOT / "results" / "rsa" / "rsa_results.parquet"
    if not path.exists():
        print("[skip] rsa_matrix: no RSA results yet"); return
    df = pd.read_parquet(path)
    best = df.loc[df.groupby(["area", "experiment_id", "model"])["rsa_rho"].idxmax()]
    pivot = best.groupby(["area", "model"])["rsa_rho"].mean().unstack()
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([MODEL_LABELS[m] for m in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    fig.colorbar(im, label="RSA ρ")
    ax.set_title("Brain area × model — RSA (Spearman ρ on RDMs)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rsa_matrix.png", dpi=150)
    plt.close(fig)
    print("[ok] rsa_matrix.png")


def fig_ranking_uncertainty():
    path = ROOT / "results" / "encoding" / "ranking_stability.json"
    if not path.exists():
        print("[skip] ranking_uncertainty: no ranking_stability.json yet"); return
    report = json.load(open(path))
    groups = {k: v["kendalls_w"] for k, v in report.items() if isinstance(v, dict) and "kendalls_w" in v}
    if not groups:
        print("[skip] ranking_uncertainty: no kendalls_w entries"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(groups)), list(groups.values()), color="#6ea8fe")
    ax.axhline(1.0, color="#3ecf8e", linestyle="--", linewidth=1, label="perfect agreement")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(list(groups.keys()), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Kendall's W")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model-ranking agreement across areas / sessions / metrics")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ranking_uncertainty.png", dpi=150)
    plt.close(fig)
    print("[ok] ranking_uncertainty.png")


def fig_compression_curves():
    path = ROOT / "results" / "compression" / "compression_results.parquet"
    if not path.exists():
        print("[skip] compression_curves: no compression results yet"); return
    df = pd.read_parquet(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in df.groupby("model"):
        curve = g.groupby("n_dims")["mean_r"].mean().sort_index()
        ax.plot(curve.index, curve.values, marker="o", label=MODEL_LABELS.get(model, model))
    ax.set_xscale("log")
    ax.set_xlabel("PCA components"); ax.set_ylabel("Mean encoding r")
    ax.set_title("Encoding score vs. activation compression")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "compression_curves.png", dpi=150)
    plt.close(fig)
    print("[ok] compression_curves.png")


def fig_stimulus_residuals():
    path = ROOT / "results" / "failures" / "stimulus_residuals.parquet"
    if not path.exists():
        print("[skip] stimulus_residuals: no failure analysis yet"); return
    df = pd.read_parquet(path)
    per_image = df.groupby("frame_id")["population_pattern_r"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(per_image)), per_image.values, color="#e0a458")
    ax.set_xlabel("Stimulus image (sorted, hardest first)")
    ax.set_ylabel("Mean out-of-fold population-pattern r")
    ax.set_title("Per-stimulus difficulty (averaged across sessions, best model per session)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stimulus_residuals.png", dpi=150)
    plt.close(fig)
    print("[ok] stimulus_residuals.png")


def fig_brain_region_comparison():
    path = ROOT / "results" / "encoding" / "encoding_results.parquet"
    if not path.exists():
        print("[skip] brain_region_comparison: no encoding results"); return
    df = pd.read_parquet(path)
    df = df[df["model"] != "mean_baseline"]
    best = df.loc[df.groupby(["area", "experiment_id", "model"])["mean_r"].idxmax()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    areas = sorted(best["area"].unique())
    data = [best[best["area"] == a]["mean_r"].values for a in areas]
    bp = ax.boxplot(data, tick_labels=areas, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3ecf8e"); patch.set_alpha(0.6)
    ax.set_ylabel("Mean encoding r (all models pooled)")
    ax.set_title("Distribution of encoding scores by brain area")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "brain_region_comparison.png", dpi=150)
    plt.close(fig)
    print("[ok] brain_region_comparison.png")


def main():
    fig_area_model_heatmap()
    fig_layer_depth_curves()
    fig_noise_ceiling_comparison()
    fig_rsa_matrix()
    fig_ranking_uncertainty()
    fig_compression_curves()
    fig_stimulus_residuals()
    fig_brain_region_comparison()


if __name__ == "__main__":
    main()
