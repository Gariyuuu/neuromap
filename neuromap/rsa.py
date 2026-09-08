"""Representational Similarity Analysis (RSA) and linear CKA.

RSA and CKA are NOT interchangeable (per project spec): RSA compares the rank structure
of pairwise (dis)similarities (robust to arbitrary invertible linear transforms of the
representation, sensitive only to distances), while CKA compares representations more
directly via a kernel alignment statistic sensitive to the actual geometry of the
feature space. Both are reported separately, never averaged together.
"""
import numpy as np
from scipy.stats import spearmanr


def rdm_correlation_distance(X: np.ndarray) -> np.ndarray:
    """X: (n_images, n_features). Returns condition x condition RDM using 1 - Pearson r."""
    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    Xn = Xc / norms
    corr = Xn @ Xn.T
    return 1.0 - corr


def upper_tri(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]


def rsa_compare(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Spearman rank correlation between the upper triangles of two RDMs."""
    a, b = upper_tri(rdm_a), upper_tri(rdm_b)
    rho, _ = spearmanr(a, b)
    return float(rho)


def rsa_bootstrap_ci(X_neural: np.ndarray, X_model: np.ndarray, n_boot: int = 1000, seed: int = 0,
                      ci: float = 0.95) -> dict:
    """Bootstrap over stimuli (images), recomputing both RDMs each resample."""
    rng = np.random.default_rng(seed)
    n = X_neural.shape[0]
    point_estimate = rsa_compare(rdm_correlation_distance(X_neural), rdm_correlation_distance(X_model))

    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(idx)) < 3:
            continue
        rdm_n = rdm_correlation_distance(X_neural[idx])
        rdm_m = rdm_correlation_distance(X_model[idx])
        v = rsa_compare(rdm_n, rdm_m)
        if np.isfinite(v):
            boot_vals.append(v)

    boot_vals = np.array(boot_vals)
    lo = np.percentile(boot_vals, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_vals, (1 + ci) / 2 * 100)
    return {"rho": point_estimate, "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": len(boot_vals)}


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two (n_images, n_features) representations."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    hsic_xx = np.linalg.norm(Xc.T @ Xc, ord="fro")
    hsic_yy = np.linalg.norm(Yc.T @ Yc, ord="fro")
    denom = hsic_xx * hsic_yy
    if denom == 0:
        return float("nan")
    return float(hsic_xy / denom)
