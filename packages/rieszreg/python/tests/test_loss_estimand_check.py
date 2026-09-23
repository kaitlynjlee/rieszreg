"""The loss/estimand check raises only when the estimand's α is known to
leave the loss's range; when that depends on the data, it warns."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rieszreg import ATE, FitResult, OutcomeRegNormSq, RieszEstimator
from rieszreg.losses import KLLoss


class _Const:
    def predict_eta(self, X):
        return np.zeros(len(X))

    def predict_alpha(self, X):
        return np.ones(len(X))


class _AugBackend:
    def fit_augmented(self, aug_train, aug_valid, loss, **kwargs):
        return FitResult(predictor=_Const())


def _fit(estimand, X, y=None):
    return RieszEstimator(estimand=estimand, backend=_AugBackend(), loss=KLLoss()).fit(X, y)


def test_outcome_regression_with_some_negative_y_warns_but_fits():
    """E[Y | X] = 1.5 + 0.5 x > 0, but about 1 in 10 draws of y are negative."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(300, 1))
    y = 1.5 + 0.5 * x[:, 0] + rng.normal(size=300)
    assert (y < 0).any()
    with pytest.warns(UserWarning, match="may be negative somewhere"):
        _fit(OutcomeRegNormSq(), x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _fit(OutcomeRegNormSq(), x, np.abs(y))


def test_estimand_whose_alpha_is_negative_still_raises():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.binomial(1, 0.5, 200), rng.normal(size=200)])
    with pytest.raises(ValueError, match="takes negative values"):
        _fit(ATE(), X)
