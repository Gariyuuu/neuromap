"""Tests neural response processing against a tiny synthetic fixture that mimics the
AllenSDK BrainObservatoryNwbDataSet interface (get_stimulus_table / get_dff_traces /
get_cell_specimen_ids) — no real neuroscience data is downloaded in CI, per project spec."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromap.neural import extract_trial_responses, average_by_image, split_half_reliability, NeuralPreprocessConfig


class FakeDataset:
    """3 images x 4 repeats each + 2 blank trials, 2 cells, deterministic per-image responses
    plus small noise, so reliability should come out high but not exactly 1."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        frames = [0, 1, 2] * 4 + [-1, -1]
        rng.shuffle(frames)
        self.frames = frames
        n_trials = len(frames)
        frame_len = 7
        gap = 10
        starts = [i * (frame_len + gap) for i in range(n_trials)]
        self._stim_table = pd.DataFrame({
            "frame": frames, "start": starts, "end": [s + frame_len for s in starts],
        })
        n_timepoints = starts[-1] + frame_len + 20
        true_response = {0: [1.0, -1.0], 1: [0.0, 2.0], 2: [-1.0, 0.5], -1: [0.0, 0.0]}
        dff = np.zeros((2, n_timepoints))
        for f, s in zip(frames, starts):
            base = true_response[f]
            window = slice(s + 2, s + 2 + frame_len)
            for cell in range(2):
                dff[cell, window] = base[cell] + rng.normal(0, 0.05, frame_len)
        self._dff = dff
        self._cell_ids = [101, 102]

    def get_stimulus_table(self, name):
        assert name == "natural_scenes"
        return self._stim_table

    def get_dff_traces(self):
        return np.arange(self._dff.shape[1]), self._dff

    def get_cell_specimen_ids(self):
        return self._cell_ids


def test_trial_extraction_shape_and_columns():
    ds = FakeDataset()
    trial_df = extract_trial_responses(ds)
    assert list(trial_df.columns) == ["cell_101", "cell_102", "frame"]
    assert len(trial_df) == len(ds.frames)


def test_image_average_excludes_blanks_and_recovers_true_response():
    ds = FakeDataset()
    trial_df = extract_trial_responses(ds)
    image_df = average_by_image(trial_df)
    assert list(image_df.index) == [0, 1, 2]  # blanks (-1) excluded, sorted
    assert np.isclose(image_df.loc[0, "cell_101"], 1.0, atol=0.1)
    assert np.isclose(image_df.loc[1, "cell_102"], 2.0, atol=0.1)
    assert np.isclose(image_df.loc[2, "cell_101"], -1.0, atol=0.1)


def test_split_half_reliability_is_high_for_low_noise_signal():
    ds = FakeDataset()
    trial_df = extract_trial_responses(ds)
    reliab = split_half_reliability(trial_df, seed=0)
    assert reliab["cell_101"] > 0.8
    assert reliab["cell_102"] > 0.8


def test_split_half_reliability_near_zero_for_pure_noise_cell():
    cfg = NeuralPreprocessConfig()
    rng = np.random.default_rng(0)
    n_trials = 60
    frames = list(rng.integers(0, 3, n_trials))
    df = pd.DataFrame({"frame": frames, "cell_x": rng.standard_normal(n_trials)})
    reliab = split_half_reliability(df, cfg, seed=0)
    assert abs(reliab["cell_x"]) < 0.6  # noisy but not a strict bound; mainly checks it runs and isn't ~1
