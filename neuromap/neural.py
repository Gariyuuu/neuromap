"""Neural response processing for Allen Brain Observatory natural_scenes sessions.

Preprocessing decisions (persisted, not implicit):
  - response window: mean dF/F over frames [start + RESPONSE_OFFSET, start + RESPONSE_OFFSET + RESPONSE_LEN)
    RESPONSE_OFFSET accounts for GCaMP6f onset latency; RESPONSE_LEN matches stimulus presentation length.
  - blank/gray-screen trials (frame == -1) are excluded from image-level analysis.
  - trial-averaged response per image = mean over all repeats of that image within a session.
  - reliability = split-half Spearman-Brown corrected correlation, computed over odd/even trial splits.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

RESPONSE_OFFSET_FRAMES = 2
RESPONSE_LEN_FRAMES = 7
BLANK_FRAME = -1


@dataclass
class NeuralPreprocessConfig:
    response_offset_frames: int = RESPONSE_OFFSET_FRAMES
    response_len_frames: int = RESPONSE_LEN_FRAMES
    blank_frame: int = BLANK_FRAME
    trial_aggregation: str = "mean_dff_in_window"
    image_aggregation: str = "mean_over_repeats"
    reliability_method: str = "split_half_spearman_brown"

    def as_dict(self):
        return asdict(self)


def extract_trial_responses(dataset, cfg: NeuralPreprocessConfig = NeuralPreprocessConfig()) -> pd.DataFrame:
    """One row per stimulus presentation, one column per cell, plus 'frame' (image id)."""
    stim_table = dataset.get_stimulus_table("natural_scenes")
    _, dff = dataset.get_dff_traces()  # (n_cells, n_timepoints)
    cell_ids = dataset.get_cell_specimen_ids()
    n_timepoints = dff.shape[1]

    rows = []
    for _, trial in stim_table.iterrows():
        w0 = trial["start"] + cfg.response_offset_frames
        w1 = min(w0 + cfg.response_len_frames, n_timepoints)
        if w0 >= n_timepoints or w1 <= w0:
            continue
        resp = dff[:, w0:w1].mean(axis=1)
        rows.append(resp)

    resp_matrix = np.stack(rows, axis=0)
    df = pd.DataFrame(resp_matrix, columns=[f"cell_{c}" for c in cell_ids])
    df["frame"] = stim_table["frame"].values[: len(df)]
    return df


def average_by_image(trial_df: pd.DataFrame, cfg: NeuralPreprocessConfig = NeuralPreprocessConfig()) -> pd.DataFrame:
    """Mean response per image (excludes blank trials). Index = frame (image id), sorted."""
    df = trial_df[trial_df["frame"] != cfg.blank_frame].copy()
    grouped = df.groupby("frame").mean()
    return grouped.sort_index()


def split_half_reliability(trial_df: pd.DataFrame, cfg: NeuralPreprocessConfig = NeuralPreprocessConfig(),
                            seed: int = 0) -> pd.Series:
    """Per-cell split-half reliability (Spearman-Brown corrected) across repeats of each image."""
    rng = np.random.default_rng(seed)
    df = trial_df[trial_df["frame"] != cfg.blank_frame].copy()
    cell_cols = [c for c in df.columns if c.startswith("cell_")]

    half_a_means = {c: [] for c in cell_cols}
    half_b_means = {c: [] for c in cell_cols}
    for frame_id, group in df.groupby("frame"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        mid = len(idx) // 2
        if mid < 1:
            continue
        a, b = idx[:mid], idx[mid : 2 * mid]
        for c in cell_cols:
            half_a_means[c].append(group.loc[a, c].mean())
            half_b_means[c].append(group.loc[b, c].mean())

    reliab = {}
    for c in cell_cols:
        a = np.array(half_a_means[c])
        b = np.array(half_b_means[c])
        if np.std(a) == 0 or np.std(b) == 0 or len(a) < 3:
            r = np.nan
        else:
            r = np.corrcoef(a, b)[0, 1]
        r_sb = (2 * r) / (1 + r) if np.isfinite(r) and (1 + r) != 0 else np.nan
        reliab[c] = r_sb
    return pd.Series(reliab, name="split_half_reliability")
