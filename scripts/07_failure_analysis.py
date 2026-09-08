#!/usr/bin/env python3
"""Failure analysis: neurons no model predicts well, stimuli with large population-level
residuals under the session's best model, and a pointer back to the ranking-disagreement
and accuracy-vs-alignment results computed in scripts/06_ranking_stability.py.

Canonical result: results/failures/failure_analysis.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuromap.encoding import run_encoding
from neuromap.provenance import write_manifest

FEATURES_DIR = ROOT / "features"
NEURAL_DIR = ROOT / "data" / "processed" / "neural"
POOR_NEURON_R_THRESHOLD = 0.1


def load_feature(model, layer):
    return np.load(FEATURES_DIR / f"{model}__{layer}.npy")


def main():
    enc_path = ROOT / "results" / "encoding" / "encoding_results.parquet"
    if not enc_path.exists():
        print("No encoding results yet; run scripts/03_run_encoding.py first")
        return
    enc = pd.read_parquet(enc_path)
    enc_models = enc[enc["model"] != "mean_baseline"]

    with open(ROOT / "configs" / "dataset.yaml") as f:
        dataset_cfg = yaml.safe_load(f)

    poor_neurons = []
    stimulus_residuals = []

    # neurons poorly predicted by ANY model in ANY session (max r across all model/layer rows for that neuron's session)
    for (area, exp_id), group in enc_models.groupby(["area", "experiment_id"]):
        per_neuron_max = np.nanmax(np.stack(group["per_neuron_r"].apply(np.array).to_numpy()), axis=0)
        n_poor = int(np.sum(per_neuron_max < POOR_NEURON_R_THRESHOLD))
        poor_neurons.append({
            "area": area, "experiment_id": int(exp_id), "n_neurons": len(per_neuron_max),
            "n_poorly_predicted_by_all_models": n_poor,
            "fraction_poorly_predicted": float(n_poor / len(per_neuron_max)),
        })

        # best (model, layer) for this session -> recompute OOF predictions for stimulus-level residuals
        best_row = group.loc[group["mean_r"].idxmax()]
        images_path = NEURAL_DIR / area / f"{exp_id}_images.parquet"
        image_df = pd.read_parquet(images_path)
        cell_cols = [c for c in image_df.columns if c.startswith("cell_")]
        frame_ids = image_df.index.to_numpy()
        Y = image_df[cell_cols].to_numpy()
        X = load_feature(best_row["model"], best_row["layer"])[frame_ids]

        res = run_encoding(X, Y, n_outer_folds=5, n_pca_components=50, seed=0)
        # population-pattern correlation per image: how well does the predicted response
        # PATTERN ACROSS NEURONS match the actual pattern, for that one held-out image
        for i, frame_id in enumerate(frame_ids):
            actual, pred = res.oof_actual[i], res.oof_predictions[i]
            if np.std(actual) > 0 and np.std(pred) > 0:
                pattern_r = float(np.corrcoef(actual, pred)[0, 1])
            else:
                pattern_r = float("nan")
            stimulus_residuals.append({
                "area": area, "experiment_id": int(exp_id), "frame_id": int(frame_id),
                "best_model": best_row["model"], "best_layer": best_row["layer"],
                "population_pattern_r": pattern_r,
            })

    poor_df = pd.DataFrame(poor_neurons)
    resid_df = pd.DataFrame(stimulus_residuals)

    # aggregate hardest stimuli: mean population-pattern-r per image, averaged across sessions/areas
    hardest_stimuli = (resid_df.groupby("frame_id")["population_pattern_r"]
                       .mean().sort_values().head(10))

    report = {
        "poor_neuron_threshold_r": POOR_NEURON_R_THRESHOLD,
        "poor_neurons_by_session": poor_neurons,
        "overall_fraction_poorly_predicted": float(poor_df["n_poorly_predicted_by_all_models"].sum() /
                                                     poor_df["n_neurons"].sum()),
        "hardest_stimuli_frame_ids": hardest_stimuli.index.tolist(),
        "hardest_stimuli_mean_pattern_r": hardest_stimuli.to_numpy().tolist(),
        "note": "Area-ranking-disagreement and ImageNet-accuracy-vs-alignment results are in "
                "results/encoding/ranking_stability.json (scripts/06_ranking_stability.py) — "
                "this file focuses on per-neuron and per-stimulus failure modes.",
    }

    resid_df.to_parquet(ROOT / "results" / "failures" / "stimulus_residuals.parquet")
    write_manifest(ROOT / "results" / "failures" / "failure_analysis.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "poor_neurons_by_session"}, indent=2))


if __name__ == "__main__":
    main()
