"""Statistical machinery: bootstrap CIs, permutation tests, multiple-comparison
correction, and ranking-stability analysis.

Neurons within one session are NOT independent samples in the same sense as separate
human subjects (project spec: "Avoid treating thousands of neurons as fully
independent"). We therefore report two levels throughout: per-neuron distributions
(descriptive) and session/subject-level aggregates (one number per session) used for
any confirmatory claim or CI that is meant to generalize.
"""
from itertools import combinations

import numpy as np
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, ci: float = 0.95,
                  statistic=np.mean) -> dict:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    boots = [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return {"point": float(statistic(values)), "ci_lo": float(lo), "ci_hi": float(hi), "n": len(values)}


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0) -> dict:
    """Two-sided permutation test on paired differences (e.g. per-session score of model A
    vs model B on the same session/stimuli), by randomly flipping the sign of each pair."""
    a, b = np.asarray(a), np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    diff = a - b
    obs = diff.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(diff)))
    perm_means = (signs * diff[None, :]).mean(axis=1)
    p = float(np.mean(np.abs(perm_means) >= np.abs(obs)))
    return {"observed_diff": float(obs), "p_value": p, "n_perm": n_perm, "n_pairs": len(diff)}


def correct_multiple_comparisons(p_values: list, method: str = "fdr_bh", alpha: float = 0.05) -> dict:
    p_values = np.asarray(p_values)
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return {"reject": reject.tolist(), "p_adjusted": p_adj.tolist(), "method": method, "alpha": alpha}


def kendall_w_ranking_agreement(rank_matrix: np.ndarray) -> float:
    """Kendall's W (coefficient of concordance) across raters (rows) ranking the same
    items (columns) — used for agreement across metrics/datasets/subjects on model ranking.
    W=1 -> perfect agreement, W=0 -> no agreement (chance)."""
    m, n = rank_matrix.shape  # m raters, n items
    ranks = np.apply_along_axis(rankdata, 1, rank_matrix)
    rank_sums = ranks.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)
    w = 12 * ss / (m**2 * (n**3 - n)) if n > 1 else float("nan")
    return float(w)


def pairwise_spearman_ranking_agreement(rank_matrix: np.ndarray) -> dict:
    """All pairwise Spearman correlations between raters' rankings of the same items."""
    from scipy.stats import spearmanr
    m = rank_matrix.shape[0]
    rhos = []
    for i, j in combinations(range(m), 2):
        rho, _ = spearmanr(rank_matrix[i], rank_matrix[j])
        rhos.append(rho)
    rhos = np.array(rhos)
    return {"mean_pairwise_spearman": float(np.nanmean(rhos)), "min": float(np.nanmin(rhos)),
            "max": float(np.nanmax(rhos)), "n_pairs": len(rhos)}
