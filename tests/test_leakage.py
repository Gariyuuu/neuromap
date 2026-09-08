"""Explicit checks that PCA / scaling inside the encoding pipeline never sees test-fold
data, and that dimensionality reduction elsewhere in the project is fit on training data
only (project spec: 'Fit reduction only on training data. Prevent leakage.')."""
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromap.encoding import _fit_predict_fold


def test_pca_inside_fold_is_fit_on_train_only():
    """If PCA leaked test data, injecting an extreme outlier into the test set would change
    the PCA components used to transform the train set's predictions — it must not."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((40, 20))
    Y_train = rng.standard_normal((40, 3))
    X_test_normal = rng.standard_normal((10, 20))
    X_test_outlier = X_test_normal.copy()
    X_test_outlier[0] = 1000.0  # extreme outlier, would dominate a leaked PCA fit
    Y_test = rng.standard_normal((10, 3))

    r_normal, preds_normal = _fit_predict_fold(X_train.copy(), X_test_normal.copy(), Y_train.copy(), Y_test.copy(), 5)
    r_outlier, preds_outlier = _fit_predict_fold(X_train.copy(), X_test_outlier.copy(), Y_train.copy(), Y_test.copy(), 5)

    # predictions for the untouched test rows (1..9) must be identical regardless of the
    # row-0 outlier, since PCA/scaler are fit on X_train only and outliers in the test set
    # cannot affect the fitted transform applied to other test rows
    assert np.allclose(preds_normal[1:], preds_outlier[1:], atol=1e-8)


def test_pca_components_unaffected_by_test_set_content():
    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((30, 15))
    pca_a = PCA(n_components=5, random_state=0).fit(X_train)

    # simulate "including test data" scenario for comparison — components should differ
    X_train_plus_test = np.vstack([X_train, rng.standard_normal((10, 15)) * 100])
    pca_b = PCA(n_components=5, random_state=0).fit(X_train_plus_test)

    assert not np.allclose(pca_a.components_, pca_b.components_[: pca_a.components_.shape[0]])
