"""Data-generating processes for the backend comparison simulation.

Both DGPs share one structure. With covariates X in R^p,

    X       ~ N(0, Sigma),   Sigma_jk = 0.3^|j - k|        (AR(1) correlation)
    A | X   ~ Bernoulli(pi(X)),   pi(X) clipped to [0.025, 0.975]
    Y | A,X ~ N(mu(A, X), 2.5^2)

and the covariates split into four blocks:

    confounders     affect A and Y
    outcome-only    affect Y only
    treatment-only  affect A only
    noise           affect neither

The two DGPs differ in p and in how nonlinear pi and mu are.

    EasyDGP  p = 10, pi and mu linear plus one quadratic term each.
    HardDGP  p = 40, pi and mu built from linear, quadratic, and sine terms
             of 20 confounders, plus products of adjacent confounders.

Each DGP exposes the true propensity score `propensity(X)` and the true
outcome regression `outcome_regression(a, X)`. The true Riesz representer
and the true estimand value are properties of the estimand, so they live in
`estimands.py`.
"""

import numpy as np
import pandas as pd
from scipy.special import expit


class DGP:
    """Shared structure: correlated Gaussian covariates, a Bernoulli
    treatment, and a Gaussian outcome. Subclasses define the covariate
    blocks and the two nuisance functions pi and mu."""

    name: str
    n_covariates: int
    confounders: slice
    outcome_only: slice
    treatment_only: slice      # the remaining covariates are noise

    rho = 0.3                  # AR(1) correlation between covariates
    sigma_y = 2.5              # outcome noise sd
    pi_clip = (0.025, 0.975)   # bounds on the propensity score

    def __init__(self):
        idx = np.arange(self.n_covariates)
        self._chol = np.linalg.cholesky(self.rho ** np.abs(idx[:, None] - idx[None, :]))
        self.covariates = [f"x{j + 1}" for j in range(self.n_covariates)]

    def propensity(self, X):
        """pi(X) = P(A = 1 | X)."""
        return np.clip(expit(self.logit_propensity(X)), *self.pi_clip)

    def logit_propensity(self, X):
        raise NotImplementedError

    def outcome_regression(self, a, X):
        """mu(a, X) = E[Y | A = a, X]. `a` may be an array or a scalar."""
        raise NotImplementedError

    def draw_covariates(self, n, rng):
        return rng.standard_normal((n, self.n_covariates)) @ self._chol.T

    def sample(self, n, rng):
        """n iid draws of (Y, A, X)."""
        X = self.draw_covariates(n, rng)
        A = rng.binomial(1, self.propensity(X)).astype(float)
        Y = self.outcome_regression(A, X) + rng.normal(0.0, self.sigma_y, n)
        return Y, A, X

    def to_frame(self, A, X):
        """Predictor frame Z = (A, X) with columns a, x1, ..., xp."""
        return pd.DataFrame(X, columns=self.covariates).assign(a=A)[["a", *self.covariates]]


class EasyDGP(DGP):
    """p = 10: confounders x1-x4, outcome-only x5-x6, treatment-only x7-x8,
    noise x9-x10.

        logit pi(X) = 0.1 + 0.5 x1 - 0.4 x2 + 0.3 x3 - 0.2 x4 + 0.25 (x1^2 - 1)
                      + 0.3 x7 - 0.3 x8
        mu(a, X)    = x1 - 0.8 x2 + 0.6 x3 - 0.4 x4 + 0.5 (x2^2 - 1)
                      + 0.8 x5 - 0.5 x6 + a (3 + 0.5 x1)
    """

    name = "easy"
    n_covariates = 10
    confounders = slice(0, 4)
    outcome_only = slice(4, 6)
    treatment_only = slice(6, 8)

    def logit_propensity(self, X):
        x = X.T
        return (0.1 + 0.5 * x[0] - 0.4 * x[1] + 0.3 * x[2] - 0.2 * x[3] + 0.25 * (x[0] ** 2 - 1)
                + 0.3 * x[6] - 0.3 * x[7])

    def outcome_regression(self, a, X):
        x = X.T
        baseline = (x[0] - 0.8 * x[1] + 0.6 * x[2] - 0.4 * x[3] + 0.5 * (x[1] ** 2 - 1)
                    + 0.8 * x[4] - 0.5 * x[5])
        return baseline + a * (3.0 + 0.5 * x[0])


class HardDGP(DGP):
    """p = 40: confounders x1-x20, outcome-only x21-x25, treatment-only
    x26-x30, noise x31-x40. With C the confounder block,

        logit pi(X) = 0.1 + f(C; b1) + g(C; b2) + x26:x30 . b3
        mu(a, X)    = f(C; c1) + g(C; c2) + x21:x25 . c3 + a (3 + f(C; c4))

    where f(C; b) = sum_j b_j h_j(C_j) with h_j cycling through the linear,
    centered quadratic (x^2 - 1), and sin(pi x) terms, and
    g(C; b) = sum_j b_j C_j C_{j+1} sums products of adjacent confounders.
    The treatment effect 3 + f(C; c4) varies with all 20 confounders.

    Each coefficient vector alternates in sign and decays like 1/sqrt(j),
    times a Uniform(0.7, 1.3) jitter. The coefficients are drawn once from a
    fixed seed, so the DGP is the same function in every replicate.
    """

    name = "hard"
    n_covariates = 40
    confounders = slice(0, 20)
    outcome_only = slice(20, 25)
    treatment_only = slice(25, 30)

    def __init__(self, coef_seed=20250923):
        super().__init__()
        rng = np.random.default_rng(coef_seed)
        k = self.confounders.stop

        def coefs(length, scale):
            sign = np.where(np.arange(length) % 2 == 0, 1.0, -1.0)
            return scale * sign / np.sqrt(np.arange(1, length + 1)) * rng.uniform(0.7, 1.3, length)

        # Drawn in this order; changing the order changes the DGP.
        self.pi_main = coefs(k, 0.35)
        self.pi_pairs = coefs(k - 1, 0.15)
        self.pi_treatment_only = coefs(5, 0.4)
        self.mu_main = coefs(k, 0.7)
        self.mu_pairs = coefs(k - 1, 0.3)
        self.mu_outcome_only = coefs(5, 1.0)
        self.mu_effect = coefs(k, 0.4)

    @staticmethod
    def _main_effects(C, b):
        """f(C; b): column j enters as C_j, C_j^2 - 1, or sin(pi C_j) for j mod 3 = 0, 1, 2."""
        terms = [C[:, j] if j % 3 == 0 else C[:, j] ** 2 - 1 if j % 3 == 1 else np.sin(np.pi * C[:, j])
                 for j in range(C.shape[1])]
        return np.column_stack(terms) @ b

    @staticmethod
    def _adjacent_pairs(C, b):
        """g(C; b) = sum_j b_j C_j C_{j+1}."""
        return (C[:, :-1] * C[:, 1:]) @ b

    def logit_propensity(self, X):
        C = X[:, self.confounders]
        return (0.1 + self._main_effects(C, self.pi_main) + self._adjacent_pairs(C, self.pi_pairs)
                + X[:, self.treatment_only] @ self.pi_treatment_only)

    def outcome_regression(self, a, X):
        C = X[:, self.confounders]
        baseline = (self._main_effects(C, self.mu_main) + self._adjacent_pairs(C, self.mu_pairs)
                    + X[:, self.outcome_only] @ self.mu_outcome_only)
        return baseline + a * (3.0 + self._main_effects(C, self.mu_effect))


DGPS = {"easy": EasyDGP(), "hard": HardDGP()}
