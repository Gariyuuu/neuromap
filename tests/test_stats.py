import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromap.stats import (bootstrap_ci, paired_permutation_test, correct_multiple_comparisons,
                             kendall_w_ranking_agreement, pairwise_spearman_ranking_agreement)


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=200)
    result = bootstrap_ci(values, n_boot=1000, seed=0)
    assert result["ci_lo"] < 0.5 < result["ci_hi"]


def test_permutation_test_detects_real_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.05, 30)
    b = rng.normal(0.3, 0.05, 30)
    result = paired_permutation_test(a, b, n_perm=2000, seed=0)
    assert result["p_value"] < 0.01


def test_permutation_test_null_case():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 30)
    b = a + rng.normal(0, 0.001, 30)  # nearly identical
    result = paired_permutation_test(a, b, n_perm=2000, seed=0)
    assert result["p_value"] > 0.05


def test_fdr_correction_reduces_false_positives():
    rng = np.random.default_rng(0)
    p_values = rng.uniform(0, 1, 100).tolist()  # all null
    result = correct_multiple_comparisons(p_values, method="fdr_bh", alpha=0.05)
    assert sum(result["reject"]) < 20  # far fewer than the ~5 raw-alpha false positives would balloon to


def test_kendalls_w_perfect_agreement():
    ranks = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    w = kendall_w_ranking_agreement(ranks)
    assert abs(w - 1.0) < 1e-6


def test_kendalls_w_no_agreement():
    ranks = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    w = kendall_w_ranking_agreement(ranks)
    assert w < 0.1


def test_pairwise_spearman_agreement_perfect():
    ranks = np.array([[1, 2, 3], [1, 2, 3]])
    result = pairwise_spearman_ranking_agreement(ranks)
    assert result["mean_pairwise_spearman"] > 0.99
