"""Ridge-regression encoding models: predict per-neuron responses from model features.

Design choices (H: overly flexible mappings can obscure representation quality, so we
start with a plain linear mapping):
  - Ridge regression, one model per neuron, features shared across neurons in a session.
  - Split by STIMULUS (image), never by trial — repeats of the same image never appear
    in both train and test.
  - Nested CV: outer k-fold for the reported score, inner k-fold (on the outer-train
    fold only) to pick the ridge alpha. Alpha is never selected using test data.
  - PCA (fit on the outer-train fold only) reduces feature dimensionality before ridge,
    preventing leakage from high-dimensional model layers into few dozen training images.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 6, 17)


@dataclass
class EncodingResult:
    per_neuron_r: np.ndarray  # (n_neurons,) Pearson r on held-out images, mean across outer folds
    per_neuron_r_folds: np.ndarray  # (n_folds, n_neurons)
    mean_r: float
    n_images: int
    n_neurons: int
    n_features: int
    n_pca_components: int
    oof_predictions: np.ndarray = None  # (n_images, n_neurons) out-of-fold predictions, image order preserved
    oof_actual: np.ndarray = None  # (n_images, n_neurons) actual responses, same order as oof_predictions


def _fit_predict_fold(X_train, X_test, Y_train, Y_test, n_pca_components):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    n_comp = min(n_pca_components, X_train.shape[0] - 1, X_train.shape[1])
    if n_comp > 0 and X_train.shape[1] > n_comp:
        pca = PCA(n_components=n_comp, random_state=0).fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    model = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)

    r = np.full(Y_test.shape[1], np.nan)
    for j in range(Y_test.shape[1]):
        if np.std(Y_test[:, j]) > 0 and np.std(preds[:, j]) > 0:
            r[j] = np.corrcoef(Y_test[:, j], preds[:, j])[0, 1]
    return r, preds


def run_encoding(X: np.ndarray, Y: np.ndarray, n_outer_folds: int = 5, n_pca_components: int = 50,
                  seed: int = 0) -> EncodingResult:
    """X: (n_images, n_features) model features. Y: (n_images, n_neurons) neural responses.
    Both must be aligned/sorted by the same image index before calling this."""
    assert X.shape[0] == Y.shape[0], "stimulus-image count mismatch between features and neural data"
    kf = KFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)

    fold_scores = []
    oof_preds = np.full_like(Y, np.nan, dtype=float)
    for train_idx, test_idx in kf.split(X):
        r, preds = _fit_predict_fold(X[train_idx], X[test_idx], Y[train_idx], Y[test_idx], n_pca_components)
        fold_scores.append(r)
        oof_preds[test_idx] = preds
    fold_scores = np.stack(fold_scores, axis=0)

    return EncodingResult(
        per_neuron_r=np.nanmean(fold_scores, axis=0),
        per_neuron_r_folds=fold_scores,
        mean_r=float(np.nanmean(fold_scores)),
        n_images=X.shape[0], n_neurons=Y.shape[1],
        n_features=X.shape[1], n_pca_components=min(n_pca_components, X.shape[1]),
        oof_predictions=oof_preds, oof_actual=Y,
    )


def mean_response_baseline(Y: np.ndarray, n_outer_folds: int = 5, seed: int = 0) -> EncodingResult:
    """The training-set mean response, predicted as a constant for every held-out image.
    Pearson correlation with a constant vector is mathematically undefined (zero variance
    in the prediction), so by convention this baseline is scored as r=0 for every neuron
    on every fold — it explains no image-to-image variance, which is exactly the point of
    including it as the floor every real model must beat."""
    n_folds = n_outer_folds if Y.shape[0] >= n_outer_folds else max(2, Y.shape[0])
    fold_scores = np.zeros((n_folds, Y.shape[1]))
    return EncodingResult(
        per_neuron_r=np.zeros(Y.shape[1]), per_neuron_r_folds=fold_scores,
        mean_r=0.0, n_images=Y.shape[0], n_neurons=Y.shape[1],
        n_features=0, n_pca_components=0,
    )
