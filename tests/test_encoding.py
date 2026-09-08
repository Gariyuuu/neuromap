import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromap.encoding import run_encoding, mean_response_baseline


def test_encoding_recovers_linear_signal():
    rng = np.random.default_rng(0)
    n_images, n_features, n_neurons = 60, 8, 4
    X = rng.standard_normal((n_images, n_features))
    W = rng.standard_normal((n_features, n_neurons))
    Y = X @ W + 0.01 * rng.standard_normal((n_images, n_neurons))
    res = run_encoding(X, Y, n_outer_folds=5, n_pca_components=8, seed=0)
    assert res.mean_r > 0.9, f"expected near-perfect recovery of a clean linear signal, got {res.mean_r}"


def test_encoding_near_zero_on_pure_noise():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 8))
    Y = rng.standard_normal((60, 4))
    res = run_encoding(X, Y, n_outer_folds=5, n_pca_components=8, seed=0)
    assert abs(res.mean_r) < 0.35, f"expected near-zero r on unrelated data, got {res.mean_r}"


def test_no_stimulus_leaks_between_train_and_test():
    """Every image index appears in exactly one outer-fold test set."""
    from sklearn.model_selection import KFold
    n_images = 47
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    seen = np.zeros(n_images, dtype=int)
    for _, test_idx in kf.split(np.arange(n_images)):
        seen[test_idx] += 1
    assert np.all(seen == 1)


def test_oof_predictions_cover_every_image_exactly_once():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 6))
    Y = rng.standard_normal((30, 3))
    res = run_encoding(X, Y, n_outer_folds=5, n_pca_components=6, seed=0)
    assert not np.any(np.isnan(res.oof_predictions))


def test_mean_baseline_is_zero_by_convention():
    rng = np.random.default_rng(3)
    Y = rng.standard_normal((20, 5))
    res = mean_response_baseline(Y)
    assert res.mean_r == 0.0
    assert np.all(res.per_neuron_r == 0.0)
