"""Candidate DGPs for the claims in AIMS.md, their true representers, and the
diagnostics that check each DGP does what AIMS.md says it should.

Every binary DGP draws X ~ Unif(-1, 1)^p (p = 5 unless named otherwise) and
A ~ Bernoulli(expit(rho(X))). Only rho changes between them; every DGP,
binary or continuous, shares the outcome mu (see _mu), which is hard to fit
everywhere so that the one-step estimator depends on the quality of alpha-hat.
The first outcome, _mu_easy, was mostly linear, and with it every method's
one-step estimate looked alike.

    base(X) = 0.9 X1 - 0.9 X2 + 1.2 (X3^2 - 1/3) + X4 X5      (five_confounders)

Claim 1, Riesz regression vs the propensity plug-in:
    overlap_g{05,1,2,3}  rho = gamma base(X), gamma in {0.5, 1, 2, 3}
    tail_feature         rho = 3 X1 + 1.5 sin(4 pi X2) 1{X1 < -0.6}
    bulk_feature         rho = 3 X1 + 1.5 sin(4 pi X2) 1{|X1| < 0.2}
    logistic             rho = 0.8 X1 - 0.8 X2 + 0.6 X3 + 0.5 X4, linear
The tail and bulk features have the same amplitude and cover the same 20% of
the X1 range. The tail one sits where pi is small (pi from 0.05 to 0.14 on
the smooth part), so where alpha is large; the bulk one sits where pi is
near 0.5.

Claim 2, learners:
    smooth       rho = 1.3 sin(pi X1) + cos(pi X2) + 0.8 X3
    step         rho = 1.6 1{X1 > 0} - 1.6 1{X2 > 0.4} + 1.2 1{X3 > -0.3} - 0.6
    sparse50     p = 50, rho = 1.2 X1 - 1.2 X2 + 1.5 (X3^2 - 1/3)
    additive     rho = sin(pi X1) + 1.5 (X2^2 - 1/3) - |X3| + 1/2 + 0.8 X4 - 0.6 X5
    interaction  rho = 2.5 X1 X2 X3 + 1.5 X4 X5
The weak-overlap row of claim 2 is overlap_g2 and overlap_g3.

Hard representers, for the one-step estimator (see _hard_shared_1):
    hard_shared_{1,2}    rho = c f(X), f the outcome's baseline
    hard_unshared_{1,2}  rho = c f(X3, X1, X4, X5, X2), the same law, not mu's features
with c = 0.78 or 1.56, matching sd(rho) of overlap_g1 and overlap_g2.
    mix{33,67}_{1,2}     rho = c [w f(X) + sqrt(1 - w^2) f(X perm)] / k_w, w = 1/3, 2/3,
                         the rungs between unshared (w = 0) and shared (w = 1)

Claim 3, losses (TSM at A = 1):
    additive, additive_g2 (rho doubled), interaction.

Claim 1, continuous exposure, additive shift by delta = 0.5:
    A = m(X) + s(X) eps,  m(X) = 0.8 X1 - 0.6 X2 + 0.5 (X3^2 - 1/3)
    shift_gauss    s = 1,               eps ~ N(0, 1)
    shift_hetero   s = exp(0.6 X1),     eps ~ N(0, 1)
    shift_skew     s = 1,               eps ~ (Gamma(2, 1) - 2) / sqrt(2)
    shift_bimodal  s = 1,               eps ~ 0.9 S + 0.45 N(0, 1), S = +-1 w.p. 1/2
Each eps has mean 0 and variance 1 (the bimodal one 1.01), so only the
shape of the conditional density changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

DELTA = 0.5   # additive shift


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _unif(p):
    def sample(n, rng):
        return rng.uniform(-1, 1, size=(n, p))
    return sample


def _mu_easy(a, X):
    """The first outcome (five_confounders): about 87% of Var(mu) is linear in
    (A, X), so an outcome learner fits it about equally well everywhere and
    the one-step estimator barely depends on alpha-hat. Kept for comparison."""
    x1, x2, x3, x4, x5 = X[:, :5].T
    tau = 1.0 + 0.6 * x1 + 0.5 * np.cos(np.pi * x5)
    return (1.5 * np.sin(np.pi * x1 / 2) - x2**3 + 0.8 * x3 + 0.8 * x4
            + 0.6 * np.cos(np.pi * x5) + a * tau)


def _tau(X):
    x1, x2, x5 = X[:, 0], X[:, 1], X[:, 4]
    return 1.0 + 0.8 * np.sin(2 * np.pi * x1) + x1 * x2 + 0.6 * np.cos(np.pi * (x2 + x5))


def _f(X):
    """The baseline of the outcome, f(X) below. The hard_shared propensities
    reuse it, and the hard_unshared ones reuse it on permuted covariates."""
    x1, x2, x3, x4, x5 = X[:, :5].T
    return np.sin(2 * np.pi * x1) + 1.2 * x2 * x3 + np.cos(np.pi * (x3 + x4)) + 0.8 * (x4 > 0) * x5


def _mu(a, X):
    """The outcome, hard to fit everywhere, for every DGP:

        mu(a, X) = f(X) + a tau(X) + 0.6 (sin(2a) - a sin 2)(1 + X1)
        f(X)     = sin(2 pi X1) + 1.2 X2 X3 + cos(pi (X3 + X4)) + 0.8 1{X4 > 0} X5
        tau(X)   = 1 + 0.8 sin(2 pi X1) + X1 X2 + 0.6 cos(pi (X2 + X5))

    Two periods of a sine, products, a cosine of a sum (an interaction no
    additive model holds) and a step, in both f and the effect tau, all in the
    covariates that drive the propensity. The last term is nonlinear in a and
    zero at a = 0 and a = 1, so it changes only the continuous-exposure DGPs."""
    return _f(X) + a * _tau(X) + 0.6 * (np.sin(2 * a) - a * np.sin(2.0)) * (1 + X[:, 0])


def m_mu0(estimand, a, X):
    """The estimand's functional applied to the true outcome regression, row by row."""
    if estimand == "ATE":
        return _mu(np.ones(len(X)), X) - _mu(np.zeros(len(X)), X)
    if estimand == "TSM":
        return _mu(np.ones(len(X)), X)
    return _mu(a + DELTA, X) - _mu(a, X)


@lru_cache(maxsize=None)
def true_psi(dgp_name: str, estimand: str) -> float:
    """The true estimand. For a binary treatment it is exact: every term of
    f and of tau - 1 integrates to 0 over X ~ Unif(-1, 1) (sin(2 pi X1),
    X1 X2, X2 X3, 1{X4 > 0} X5, and the cosine of a sum, since E cos(pi U) =
    E sin(pi U) = 0), so the ATE = E tau = 1 and the TSM at A = 1 = E f + E tau
    = 1. For the shift the term nonlinear in a depends on the law of A, so
    psi0 is integrated by Monte Carlo over 4M draws (MC SE about 2e-4)."""
    if estimand in ("ATE", "TSM"):
        return 1.0
    g = SHIFT[dgp_name]
    rng = np.random.default_rng(20261020)
    vals = []
    for _ in range(4):
        d = g.draw(1_000_000, rng)
        vals.append(m_mu0("SHIFT", d["A"].to_numpy(), d[g.covariates].to_numpy()))
    return float(np.concatenate(vals).mean())


def _outcome_diagnostics(d, covariates) -> dict:
    """How hard mu is: the share of Var(mu) and of Var(tau) that a linear fit
    in (A, X) explains, and the share of Var(Y) that is signal."""
    X, a = d[covariates].to_numpy()[:, :5], d["A"].to_numpy()
    mu, tau = _mu(a, X), _tau(X)
    design = np.column_stack([np.ones(len(a)), a, X])

    def r2(y, D):
        return 1 - np.var(y - D @ np.linalg.lstsq(D, y, rcond=None)[0]) / np.var(y)
    return {"linear R2 of mu": r2(mu, design), "linear R2 of tau": r2(tau, design[:, [0, 2, 3, 4, 5, 6]]),
            "Var(mu)/Var(Y)": np.var(mu) / np.var(d["Y"])}


# ----------------------------------------------------------- binary DGPs ---

@dataclass(frozen=True)
class BinaryDGP:
    name: str
    p: int
    logit: Callable[[np.ndarray], np.ndarray]
    sigma: float = 1.0

    @property
    def covariates(self) -> list[str]:
        return [f"X{j}" for j in range(1, self.p + 1)]

    def propensity(self, X):
        return expit(self.logit(X))

    def draw(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        X = _unif(self.p)(n, rng)
        a = rng.binomial(1, self.propensity(X)).astype(float)
        df = pd.DataFrame(X, columns=self.covariates)
        df.insert(0, "A", a)
        df["Y"] = _mu(a, X) + rng.normal(0, self.sigma, n)
        return df

    def true_alpha(self, estimand: str, a, X):
        pi = self.propensity(X)
        if estimand == "ATE":
            return a / pi - (1 - a) / (1 - pi)
        if estimand == "TSM":
            return a / pi
        raise ValueError(estimand)

    def diagnose(self, n: int = 1_000_000, seed: int = 1) -> dict:
        d = self.draw(n, np.random.default_rng(seed))
        X, a = d[self.covariates].to_numpy(), d["A"].to_numpy()
        pi = self.propensity(X)
        ate, tsm = self.true_alpha("ATE", a, X), self.true_alpha("TSM", a, X)
        lo = np.minimum(pi, 1 - pi)
        rho = self.logit(X)
        design = np.column_stack([np.ones(len(X)), X])
        rho_lin = design @ np.linalg.lstsq(design, rho, rcond=None)[0]
        mu0 = _mu(a, X)
        return {
            "sd(logit)": rho.std(),
            "linear R2 of logit": 1 - np.var(rho - rho_lin) / np.var(rho),
            "corr(logit, f)": np.corrcoef(rho, _f(X))[0, 1],
            "corr(logit, mu)": np.corrcoef(rho, mu0)[0, 1],
            "P(A=1)": a.mean(),
            "pi 1%": np.quantile(pi, 0.01),
            "pi 99%": np.quantile(pi, 0.99),
            "share pi<0.05": np.mean(pi < 0.05),
            "share min(pi,1-pi)<0.05": np.mean(lo < 0.05),
            "share min(pi,1-pi)<0.01": np.mean(lo < 0.01),
            "ATE sd(alpha0)": ate.std(),
            "ATE max|alpha0|": np.abs(ate).max(),
            "ATE 99.9% |alpha0|": np.quantile(np.abs(ate), 0.999),
            "TSM max alpha0": tsm.max(),
            "TSM sd(alpha0)": tsm.std(),
            **_outcome_diagnostics(d, self.covariates),
        }


def _base(X):
    x1, x2, x3, x4, x5 = X[:, :5].T
    return 0.9 * x1 - 0.9 * x2 + 1.2 * (x3**2 - 1 / 3) + x4 * x5


# Module-level functions, so a DGP pickles into worker processes.
def _g05(X): return 0.5 * _base(X)
def _g1(X): return _base(X)
def _g2(X): return 2.0 * _base(X)
def _g3(X): return 3.0 * _base(X)


def _tail(X):
    x1, x2 = X[:, 0], X[:, 1]
    return 3 * x1 + 1.5 * np.sin(4 * np.pi * x2) * (x1 < -0.6)


def _bulk(X):
    x1, x2 = X[:, 0], X[:, 1]
    return 3 * x1 + 1.5 * np.sin(4 * np.pi * x2) * (np.abs(x1) < 0.2)


def _logistic(X):
    x1, x2, x3, x4 = X[:, :4].T
    return 0.8 * x1 - 0.8 * x2 + 0.6 * x3 + 0.5 * x4


def _smooth(X):
    x1, x2, x3 = X[:, :3].T
    return 1.3 * np.sin(np.pi * x1) + np.cos(np.pi * x2) + 0.8 * x3


def _step(X):
    x1, x2, x3 = X[:, :3].T
    return 1.6 * (x1 > 0) - 1.6 * (x2 > 0.4) + 1.2 * (x3 > -0.3) - 0.6


def _sparse(X):
    x1, x2, x3 = X[:, :3].T
    return 1.2 * x1 - 1.2 * x2 + 1.5 * (x3**2 - 1 / 3)


def _additive(X):
    x1, x2, x3, x4, x5 = X[:, :5].T
    return np.sin(np.pi * x1) + 1.5 * (x2**2 - 1 / 3) - np.abs(x3) + 0.5 + 0.8 * x4 - 0.6 * x5


def _additive_g2(X): return 2.0 * _additive(X)


# Hard representers. hard_shared's logit is the outcome's own baseline f, so
# a learner that smooths over a feature of f misses it in alpha-hat and in
# mu-hat alike, and the two errors line up. hard_unshared's logit is f on the
# covariates permuted as (X3, X1, X4, X5, X2):
#     sin(2 pi X3) + 1.2 X1 X4 + cos(pi (X4 + X5)) + 0.8 1{X5 > 0} X2,
# none of whose terms appears in mu or tau, or correlates with one. The
# covariates are iid, so the two logits have exactly the same distribution:
# the same overlap and the same difficulty, and they differ only in whether
# alpha0's hard features are mu's. The scales 0.78 and 1.56 give the logit
# the standard deviation of overlap_g1 and overlap_g2.
_PERM = [2, 0, 3, 4, 1]
def _hard_shared_1(X): return 0.78 * _f(X)
def _hard_shared_2(X): return 1.56 * _f(X)
def _hard_unshared_1(X): return 0.78 * _f(X[:, _PERM])
def _hard_unshared_2(X): return 1.56 * _f(X[:, _PERM])


# The mixing ladder between them: rho = c [w f(X) + sqrt(1 - w^2) f(X perm)] / k_w.
# w = 1 is hard_shared and w = 0 is hard_unshared. corr(f(X), f(X perm)) is
# -0.0625 (4M draws), so k_w = sqrt(1 + 2 w sqrt(1 - w^2) (-0.0625)) keeps
# sd(rho) equal to sd(c f) at every w, and the overlap matched.
_RHO_F = -0.0625


def _mix(w, c):
    k = np.sqrt(1 + 2 * w * np.sqrt(1 - w**2) * _RHO_F)

    def logit(X):
        return c * (w * _f(X) + np.sqrt(1 - w**2) * _f(X[:, _PERM])) / k
    return logit


def _mix33_1(X): return _mix(1 / 3, 0.78)(X)
def _mix67_1(X): return _mix(2 / 3, 0.78)(X)
def _mix33_2(X): return _mix(1 / 3, 1.56)(X)
def _mix67_2(X): return _mix(2 / 3, 1.56)(X)


def _interaction(X):
    x1, x2, x3, x4, x5 = X[:, :5].T
    return 2.5 * x1 * x2 * x3 + 1.5 * x4 * x5


BINARY = {d.name: d for d in [
    BinaryDGP("overlap_g05", 5, _g05),
    BinaryDGP("overlap_g1", 5, _g1),
    BinaryDGP("overlap_g2", 5, _g2),
    BinaryDGP("overlap_g3", 5, _g3),
    BinaryDGP("tail_feature", 5, _tail),
    BinaryDGP("bulk_feature", 5, _bulk),
    BinaryDGP("logistic", 5, _logistic),
    BinaryDGP("smooth", 5, _smooth),
    BinaryDGP("step", 5, _step),
    BinaryDGP("sparse50", 50, _sparse),
    BinaryDGP("additive", 5, _additive),
    BinaryDGP("additive_g2", 5, _additive_g2),
    BinaryDGP("interaction", 5, _interaction),
    BinaryDGP("hard_shared_1", 5, _hard_shared_1),
    BinaryDGP("hard_shared_2", 5, _hard_shared_2),
    BinaryDGP("hard_unshared_1", 5, _hard_unshared_1),
    BinaryDGP("hard_unshared_2", 5, _hard_unshared_2),
    BinaryDGP("mix33_1", 5, _mix33_1),
    BinaryDGP("mix67_1", 5, _mix67_1),
    BinaryDGP("mix33_2", 5, _mix33_2),
    BinaryDGP("mix67_2", 5, _mix67_2),
]}


# ------------------------------------------------- continuous exposure ---

def _shift_mean(X):
    x1, x2, x3 = X[:, :3].T
    return 0.8 * x1 - 0.6 * x2 + 0.5 * (x3**2 - 1 / 3)


def _scale_one(X): return np.ones(len(X))
def _scale_hetero(X): return np.exp(0.6 * X[:, 0])


_SQ2 = np.sqrt(2.0)


def _eps_gauss(n, rng): return rng.normal(size=n)
def _pdf_gauss(e): return stats.norm.pdf(e)
def _eps_skew(n, rng): return (rng.gamma(2.0, 1.0, size=n) - 2.0) / _SQ2
def _pdf_skew(e): return _SQ2 * stats.gamma.pdf(_SQ2 * e + 2.0, 2.0)
def _eps_bimodal(n, rng): return 0.9 * rng.choice([-1.0, 1.0], size=n) + 0.45 * rng.normal(size=n)
def _pdf_bimodal(e): return 0.5 * (stats.norm.pdf(e, -0.9, 0.45) + stats.norm.pdf(e, 0.9, 0.45))


@dataclass(frozen=True)
class ShiftDGP:
    """A = m(X) + s(X) eps. The estimand is the additive shift effect
    E[mu(A + delta, X) - mu(A, X)], whose representer is
    alpha0(a, x) = p(a - delta | x) / p(a | x) - 1."""
    name: str
    scale: Callable[[np.ndarray], np.ndarray]
    eps: Callable[[int, np.random.Generator], np.ndarray]
    eps_pdf: Callable[[np.ndarray], np.ndarray]
    p: int = 5
    sigma: float = 1.0

    @property
    def covariates(self) -> list[str]:
        return [f"X{j}" for j in range(1, self.p + 1)]

    def draw(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        X = _unif(self.p)(n, rng)
        a = _shift_mean(X) + self.scale(X) * self.eps(n, rng)
        df = pd.DataFrame(X, columns=self.covariates)
        df.insert(0, "A", a)
        df["Y"] = _mu(a, X) + rng.normal(0, self.sigma, n)
        return df

    def density(self, a, X):
        s = self.scale(X)
        return self.eps_pdf((a - _shift_mean(X)) / s) / s

    def true_alpha(self, estimand: str, a, X):
        assert estimand == "SHIFT"
        return self.density(a - DELTA, X) / self.density(a, X) - 1.0

    def diagnose(self, n: int = 1_000_000, seed: int = 1) -> dict:
        d = self.draw(n, np.random.default_rng(seed))
        X, a = d[self.covariates].to_numpy(), d["A"].to_numpy()
        alpha = self.true_alpha("SHIFT", a, X)
        return {
            "sd(A)": a.std(),
            "mean alpha0": alpha.mean(),
            "sd(alpha0)": alpha.std(),
            "max|alpha0|": np.abs(alpha).max(),
            "99.9% |alpha0|": np.quantile(np.abs(alpha), 0.999),
            "share alpha0 > 3": np.mean(alpha > 3),
            "psi0": true_psi(self.name, "SHIFT"),
            **_outcome_diagnostics(d, self.covariates),
        }


SHIFT = {d.name: d for d in [
    ShiftDGP("shift_gauss", _scale_one, _eps_gauss, _pdf_gauss),
    ShiftDGP("shift_hetero", _scale_hetero, _eps_gauss, _pdf_gauss),
    ShiftDGP("shift_skew", _scale_one, _eps_skew, _pdf_skew),
    ShiftDGP("shift_bimodal", _scale_one, _eps_bimodal, _pdf_bimodal),
]}

DGPS = {**BINARY, **SHIFT}


if __name__ == "__main__":
    import sys
    rows = {name: DGPS[name].diagnose() for name in (sys.argv[1:] or DGPS)}
    pd.set_option("display.width", 200)
    for kind in (BinaryDGP, ShiftDGP):
        sub = {k: v for k, v in rows.items() if isinstance(DGPS[k], kind)}
        if sub:
            print(pd.DataFrame(sub).T.to_string(float_format=lambda v: f"{v:.4g}"), "\n")
