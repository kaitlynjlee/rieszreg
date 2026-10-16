"""The loss/estimand check warns, and still fits, when the mean of α,
m̄ = E[m(Z, 1)], is outside the loss's range, so α can't stay inside it.
The comment in `Loss.check_estimand` explains why it warns rather than raises
and why it checks nothing beyond the mean."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rieszreg import ATE, ATT, AdditiveShift, FitResult, LocalShift, OutcomeRegNormSq, RieszEstimator
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


def test_outcome_regression_with_some_negative_y_fits_without_warning():
    """E[Y | X] = 1.5 + 0.5 x > 0, so α = E[Y | X] fits under KLLoss, though
    about 1 in 10 draws of y are negative."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(300, 1))
    y = 1.5 + 0.5 * x[:, 0] + rng.normal(size=300)
    assert (y < 0).any()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _fit(OutcomeRegNormSq(), x, y)


def test_estimand_whose_alpha_is_negative_warns_and_fits():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.binomial(1, 0.5, 200), rng.normal(size=200)])
    with pytest.warns(UserWarning, match="takes negative values"):
        est = _fit(ATE(), X)
    assert hasattr(est, "predictor_")


@pytest.mark.parametrize("make", [
    lambda: ATE(),
    lambda: ATT(),
    lambda: AdditiveShift(delta=0.37),
    lambda: LocalShift(delta=0.37, threshold=0.1),
])
def test_subtracting_estimands_have_mean_zero_and_warn(make):
    """Their functional subtracts, so m̄ = E[m(Z, 1)] cancels to exactly 0 for
    every individual, which the mean check flags under a loss that keeps
    α above 0."""
    est = make()
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    a = (rng.normal(size=500) * 3.7 + x
         if isinstance(est, (AdditiveShift, LocalShift)) else rng.binomial(1, 0.5, 500))
    X = np.column_stack([a, x])
    assert est.bind(["a", "x"]).augment(X).m_bar == 0.0
    with pytest.warns(UserWarning, match="mean of .* is 0, .*takes negative values"):
        _fit(est, X)
