"""Synthetic DGPs with 40 covariates for comparing rieszboost backends.

One class per estimand, in the style of `boosting_for_rr/rrboost/dgp_many_cov.py`.
Each class exposes the same interface:

  - `expected_trt(x)`            : true propensity score P(A=1|X)
  - `expected_outcome(a, x)`     : true outcome regression mu(a, X)
  - `gen_data(n, rng)`           : draw n iid rows, returns (Y, A, X)
  - `riesz_rep_nuisance(a, x, expected_trt)` : Riesz representer as a
                                   function of a (possibly estimated)
                                   propensity score
  - `riesz_rep(a, x)`            : true Riesz representer alpha_0(A, X)
  - `true_psi(rng, n_mc)`        : Monte Carlo ground truth for the estimand
  - `to_frame(a, x)`             : predictor frame Z = (A, X) for RieszBooster

`ATE` and `ATT` share the data-generating process in `ManyCovariateDGP` and
differ only in the representer and the estimand.

Covariates split into four blocks:
  - confounders (x1-x20): all 20 enter both the propensity score and the
    outcome regression, including the heterogeneous-treatment-effect term.
  - outcome-only (x21-x25): predict Y but do not confound treatment.
  - treatment-only (x26-x30): predict A but do not affect Y.
  - noise (x31-x40): no effect on A or Y.

Both nuisance functions are nonlinear (a per-covariate mix of linear,
centered-quadratic, and sinusoidal terms, plus adjacent-pair interactions
within the confounder block) and covariates are drawn correlated (AR(1),
rho=0.3), so backends have to separate signal from noise under realistic
collinearity and a genuinely high-dimensional confounding structure.

Per-covariate coefficients are generated once at import time from a fixed
RNG seed (alternating sign, magnitude decaying with index) so the DGP is a
fixed, reproducible nonlinear function -- not re-randomized per gen_data()
call.

Ground truth for psi has no closed form given the nonlinearity; `true_psi`
gets it by Monte Carlo integration over the covariate distribution using the
exactly-known outcome regression -- no model fitting involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def _ar1_cholesky(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    return np.linalg.cholesky(cov)


def _decaying_coefs(k: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Alternating-sign coefficients with magnitude decaying in index, so
    covariates within a block matter at different strengths (as in real
    high-dimensional confounding) while all remain nonzero."""
    sign = np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
    decay = 1.0 / np.sqrt(np.arange(1, k + 1))
    jitter = rng.uniform(0.7, 1.3, k)
    return scale * sign * decay * jitter


def _nonlinear_transform(X_block: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Weighted sum of columns, each passed through a nonlinearity that
    cycles by column index: linear, centered quadratic, sinusoidal."""
    n, k = X_block.shape
    out = np.zeros(n)
    for j in range(k):
        col = X_block[:, j]
        r = j % 3
        if r == 0:
            term = col
        elif r == 1:
            term = col**2 - 1.0  # centered: E[X^2] = 1 for standard normal
        else:
            term = np.sin(np.pi * col)
        out += coefs[j] * term
    return out


def _adjacent_interactions(X_block: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """sum_j coefs[j] * X[:, j] * X[:, j+1] over adjacent column pairs."""
    return np.sum(coefs[None, :] * X_block[:, :-1] * X_block[:, 1:], axis=1)


# Fixed-seed coefficient generation: the DGP is a deterministic nonlinear
# function, not re-randomized per gen_data() call.
_COEF_RNG = np.random.default_rng(20250923)


class ManyCovariateDGP:
    """Binary-treatment DGP shared by `ATE` and `ATT`."""

    n_confounders = 20
    n_outcome_only = 5
    n_treatment_only = 5
    n_noise = 10
    n_covariates = n_confounders + n_outcome_only + n_treatment_only + n_noise  # 40

    confounders = slice(0, 20)            # x1-x20:  affect A and Y
    outcome_only = slice(20, 25)          # x21-x25: affect Y only
    treatment_only = slice(25, 30)        # x26-x30: affect A only
    # x31-x40 (indices 30-39): pure noise, no effect on A or Y

    feature_names = [f"x{i + 1}" for i in range(n_covariates)]
    ar1_rho = 0.3
    sigma_y = 2.5
    pi_clip = (0.025, 0.975)

    _chol = _ar1_cholesky(n_covariates, ar1_rho)

    _pi_conf_coefs = _decaying_coefs(n_confounders, 0.35, _COEF_RNG)
    _pi_interaction_coefs = _decaying_coefs(n_confounders - 1, 0.15, _COEF_RNG)
    _pi_treatment_only_coefs = _decaying_coefs(n_treatment_only, 0.4, _COEF_RNG)

    _mu_conf_coefs = _decaying_coefs(n_confounders, 0.7, _COEF_RNG)
    _mu_interaction_coefs = _decaying_coefs(n_confounders - 1, 0.3, _COEF_RNG)
    _mu_outcome_only_coefs = _decaying_coefs(n_outcome_only, 1.0, _COEF_RNG)
    _mu_te_coefs = _decaying_coefs(n_confounders, 0.4, _COEF_RNG)
    _te_intercept = 3.0

    @classmethod
    def draw_covariates(cls, n: int, rng: np.random.Generator) -> np.ndarray:
        """n x 40 matrix, marginally N(0,1), AR(1)-correlated (rho=0.3)."""
        return rng.standard_normal((n, cls.n_covariates)) @ cls._chol.T

    @classmethod
    def expected_trt(cls, x: np.ndarray) -> np.ndarray:
        """Nonlinear P(A=1|X): all 20 confounders (nonlinear terms + adjacent
        interactions) plus the treatment-only block."""
        conf = x[:, cls.confounders]
        logit_pi = (
            0.1
            + _nonlinear_transform(conf, cls._pi_conf_coefs)
            + _adjacent_interactions(conf, cls._pi_interaction_coefs)
            + x[:, cls.treatment_only] @ cls._pi_treatment_only_coefs
        )
        return np.clip(expit(logit_pi), *cls.pi_clip)

    @classmethod
    def expected_outcome(cls, a, x: np.ndarray) -> np.ndarray:
        """Nonlinear mu(a, X): all 20 confounders (nonlinear terms + adjacent
        interactions) plus the outcome-only block, with a heterogeneous
        treatment effect that is itself a nonlinear function of all 20
        confounders. `a` may be an array or a scalar."""
        conf = x[:, cls.confounders]
        baseline = (
            _nonlinear_transform(conf, cls._mu_conf_coefs)
            + _adjacent_interactions(conf, cls._mu_interaction_coefs)
            + x[:, cls.outcome_only] @ cls._mu_outcome_only_coefs
        )
        return baseline + a * (cls._te_intercept + _nonlinear_transform(conf, cls._mu_te_coefs))

    @classmethod
    def gen_data(cls, n: int, rng: np.random.Generator):
        """Draw n iid rows. Returns (Y, A, X)."""
        X = cls.draw_covariates(n, rng)
        A = rng.binomial(1, cls.expected_trt(X)).astype(np.float64)
        Y = cls.expected_outcome(A, X) + rng.normal(0.0, cls.sigma_y, n)
        return Y, A, X

    @classmethod
    def riesz_rep(cls, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """True Riesz representer alpha_0(A, X)."""
        return cls.riesz_rep_nuisance(a, x, cls.expected_trt)

    @classmethod
    def to_frame(cls, a: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(x, columns=cls.feature_names)
        df.insert(0, "a", a.astype(float))
        return df


class ATE(ManyCovariateDGP):

    @staticmethod
    def riesz_rep_nuisance(a, x, expected_trt):
        """A/pi(X) - (1-A)/(1-pi(X))."""
        p = expected_trt(x)
        return a / p - (1 - a) / (1 - p)

    @classmethod
    def true_psi(cls, rng: np.random.Generator, n_mc: int = 2_000_000) -> float:
        """E[mu(1, X) - mu(0, X)] by Monte Carlo over X."""
        X = cls.draw_covariates(n_mc, rng)
        return float(np.mean(cls.expected_outcome(1.0, X) - cls.expected_outcome(0.0, X)))


class ATT(ManyCovariateDGP):

    @staticmethod
    def riesz_rep_nuisance(a, x, expected_trt):
        """*Partial-parameter* representer A - (1-A) pi(X)/(1-pi(X))."""
        p = expected_trt(x)
        return a - (1 - a) * p / (1 - p)

    @classmethod
    def true_psi(cls, rng: np.random.Generator, n_mc: int = 2_000_000) -> float:
        """E[mu(1, X) - mu(0, X) | A=1] by Monte Carlo over X. Weights by
        pi(X) = E[A|X] directly (Rao-Blackwellized) rather than drawing A,
        for a lower-variance estimate."""
        X = cls.draw_covariates(n_mc, rng)
        pi = cls.expected_trt(X)
        tau = cls.expected_outcome(1.0, X) - cls.expected_outcome(0.0, X)
        return float(np.sum(pi * tau) / np.sum(pi))
