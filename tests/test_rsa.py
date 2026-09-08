import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromap.rsa import rdm_correlation_distance, rsa_compare, linear_cka, rsa_bootstrap_ci


def test_identical_representations_give_perfect_rsa():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 10))
    rho = rsa_compare(rdm_correlation_distance(X), rdm_correlation_distance(X))
    assert rho > 0.999


def test_identical_representations_give_cka_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 10))
    cka = linear_cka(X, X)
    assert abs(cka - 1.0) < 1e-6


def test_cka_invariant_to_orthogonal_rotation():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 10))
    Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    X_rot = X @ Q
    cka = linear_cka(X, X_rot)
    assert abs(cka - 1.0) < 1e-6, "linear CKA must be invariant to orthogonal transforms"


def test_unrelated_representations_give_low_rsa():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 10))
    Y = rng.standard_normal((40, 10))
    rho = rsa_compare(rdm_correlation_distance(X), rdm_correlation_distance(Y))
    assert abs(rho) < 0.5


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((25, 8))
    result = rsa_bootstrap_ci(X, X, n_boot=200, seed=0)
    assert result["ci_lo"] <= result["rho"] <= result["ci_hi"] + 1e-9
