"""Smoke test on each built-in estimand.

Difference-style estimands (ATE, ATT, AdditiveShift, LocalShift) require a
sieve — locally constant gives an identically zero moment. Single-point
estimands (TSM) work either way; we use locally constant for them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from forestriesz import ATE, ATT, AdditiveShift, ForestRieszRegressor, LocalShift, TSM


@pytest.fixture
def df_binary():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    pi = 1.0 / (1.0 + np.exp(-0.5 * x))
    a = (rng.uniform(size=n) < pi).astype(float)
    return pd.DataFrame({"a": a, "x": x})


@pytest.fixture
def df_continuous():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    a = rng.normal(0.5 * x, 1.0)  # continuous treatment
    return pd.DataFrame({"a": a, "x": x})


def _make(estimand, force_constant=False):
    """Build a regressor; default uses the auto sieve. Pass force_constant=True
    to opt into the locally constant fit (which is degenerate for built-ins
    and triggers the row-constant check)."""
    return ForestRieszRegressor(
        estimand=estimand,
        riesz_feature_fns=None if force_constant else "auto",
        n_estimators=30,
        min_samples_leaf=15,
        random_state=0,
    )


def test_ate_auto(df_binary):
    est = _make(ATE())     # auto picks the [1{T=0}, 1{T=1}] sieve
    est.fit(df_binary)
    pred = est.predict(df_binary)
    assert pred.shape == (len(df_binary),)
    assert np.all(np.isfinite(pred))


def test_att_auto(df_binary):
    est = _make(ATT())
    est.fit(df_binary)
    pred = est.predict(df_binary)
    assert pred.shape == (len(df_binary),)
    assert np.all(np.isfinite(pred))


def test_tsm_auto(df_binary):
    est = _make(TSM(level=1))
    est.fit(df_binary)
    pred = est.predict(df_binary)
    assert pred.shape == (len(df_binary),)
    assert np.all(np.isfinite(pred))


def test_additive_shift_constant_raises(df_continuous):
    # Custom estimand with no default sieve → auto falls back to constant →
    # row-constant check fires.
    est = _make(AdditiveShift(delta=0.5))
    with pytest.raises(ValueError, match="AugForestRieszRegressor"):
        est.fit(df_continuous)


def test_local_shift_constant_raises(df_continuous):
    est = _make(LocalShift(delta=0.5, threshold=0.0))
    with pytest.raises(ValueError, match="AugForestRieszRegressor"):
        est.fit(df_continuous)


# The forest never splits on treatment, so a basis that ignores it gives the
# forest no information about the intervention: α̂ estimates E[α₀ | X], which
# is the sum of the estimand's trace coefficients (0 for ATE / ATT, 1 for TSM).
_NO_TREATMENT_BASIS = [lambda f: np.ones(len(f)), lambda f: f[:, 1]]


@pytest.mark.parametrize(
    "estimand, collapsed_to",
    [(ATE(), 0.0), (ATT(), 0.0), (TSM(level=1), 1.0)],
    ids=["ATE", "ATT", "TSM"],
)
def test_treatment_free_basis_warns(df_binary, estimand, collapsed_to):
    est = ForestRieszRegressor(
        estimand=estimand, riesz_feature_fns=_NO_TREATMENT_BASIS,
        n_estimators=8, random_state=0,
    )
    with pytest.warns(UserWarning, match="uninformative"):
        est.fit(df_binary)
    np.testing.assert_allclose(est.predict(df_binary), collapsed_to, atol=1e-8)


def test_zero_moment_basis_warns(df_binary):
    # 1{A=0} varies with treatment but vanishes at TSM(level=1)'s evaluation
    # point (A=1), so every row's moment is 0.
    est = ForestRieszRegressor(
        estimand=TSM(level=1), riesz_feature_fns=[lambda f: 1.0 - f[:, 0]],
        n_estimators=8, random_state=0,
    )
    with pytest.warns(UserWarning, match="zero moment"):
        est.fit(df_binary)
    np.testing.assert_allclose(est.predict(df_binary), 0.0, atol=1e-8)


def test_basis_with_one_treatment_free_function_fits(df_binary):
    # Covariate-only functions next to treatment indicators: a usable fit.
    basis = [lambda f: 1.0 - f[:, 0], lambda f: f[:, 0], lambda f: f[:, 1]]
    est = ForestRieszRegressor(
        estimand=ATE(), riesz_feature_fns=basis, n_estimators=8, random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred = est.fit(df_binary).predict(df_binary)
    assert not [w for w in caught if "riesz_feature_fns" in str(w.message)
                or "zero moment" in str(w.message)]
    assert np.all(np.isfinite(pred)) and np.any(pred != 0)
