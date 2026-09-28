"""The data-generating processes (DGPs), their true representers, and their
true estimands.

five_confounders (the main DGP). X1..X5 ~ Unif(-1, 1). Every covariate enters
both the treatment log-odds rho and the outcome regression mu, so all five
confound, but each enters the two through a different shape:

    rho(X)   = 0.9 X1 - 0.9 X2 + 1.2 (X3^2 - 1/3) + 1.0 X4 X5
    mu(a, X) = 1.5 sin(pi X1 / 2) - X2^3 + 0.8 X3 + 0.8 X4 + 0.6 cos(pi X5) + a tau(X)
    tau(X)   = 1 + 0.6 X1 + 0.5 cos(pi X5)
    Y        = mu(A, X) + N(0, 1)

X1 is linear in rho and S-shaped in mu, X2 linear and cubic, X3 quadratic and
linear, and X4, X5 a product in rho and additive in mu. The quadratic and the
product make rho far from logistic in the main terms. X1 and X2 move rho and
mu in the same direction, so a difference in means is confounded upward. The
effect tau rises with X1, as does the propensity, so the ATT exceeds the ATE.
The true ATE is 1 exactly, since E[X1] = 0 and E[cos(pi X5)] = 0.

shared_terms (the first main DGP, kept for comparison). The same covariates,
but the same nonlinear terms drive both rho and mu, which makes the errors of
two tree learners line up:

    rho(X)   = 0.8 X1 - 0.8 X2 + 1.5 (X3^2 - 1/3) + 1.2 X4 X5
    mu(a, X) = X1 - 0.8 X2 + sin(pi X3) + 0.8 X4 X5 + a (1 + 0.8 X1 + (X3^2 - 1/3))

lee_schuler (the binary-treatment DGP of Lee & Schuler 2025, Section 3.1).
One covariate X ~ Unif(0, 1).

    logit pi(X) = -0.02 X - X^2 + 4 log(X + 0.3) + 1.5
    mu(a, X)    = 5 X + 9 X a + 5 sin(pi X) + 25 (a - 2)
    Y           = mu(A, X) + N(0, 1)

The true ATE is 25 + 9 E[X] = 29.5 exactly (the paper reports 29.502).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class DGP:
    name: str
    covariates: tuple[str, ...]
    sample_x: Callable[[int, np.random.Generator], np.ndarray]   # n x p covariates
    propensity: Callable[[np.ndarray], np.ndarray]
    mu: Callable[[np.ndarray, np.ndarray], np.ndarray]
    sigma: float
    true_ate: float | None = None     # when known exactly

    def tau(self, X):
        return self.mu(np.ones(len(X)), X) - self.mu(np.zeros(len(X)), X)

    def draw(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        X = self.sample_x(n, rng)
        a = rng.binomial(1, self.propensity(X)).astype(float)
        y = self.mu(a, X) + rng.normal(0, self.sigma, n)
        df = pd.DataFrame(X, columns=list(self.covariates))
        df.insert(0, "A", a)
        df["Y"] = y
        return df

    def true_alpha(self, estimand: str, a: np.ndarray, X: np.ndarray) -> np.ndarray:
        """True Riesz representer at (a, X). For the ATT this is the
        representer of the numerator E[A tau(X)], which is what the learners fit."""
        pi = self.propensity(X)
        if estimand == "ATE":
            return a / pi - (1 - a) / (1 - pi)
        if estimand == "ATT":
            return a - (1 - a) * pi / (1 - pi)
        raise ValueError(estimand)

    def true_psi(self, estimand: str, n: int = 4_000_000, seed: int = 0) -> tuple[float, float]:
        """True estimand and the Monte Carlo SE of computing it. The ATT is
        E[pi tau] / E[pi], integrated over X with pi as a weight rather than by
        drawing A, which has the same limit and less Monte Carlo error."""
        if estimand == "ATE" and self.true_ate is not None:
            return self.true_ate, 0.0
        X = self.sample_x(n, np.random.default_rng(seed))
        t = self.tau(X)
        if estimand == "ATE":
            return float(t.mean()), float(t.std(ddof=1) / np.sqrt(n))
        pi = self.propensity(X)
        psi = (pi * t).mean() / pi.mean()
        se = np.std(pi * (t - psi), ddof=1) / pi.mean() / np.sqrt(n)   # delta method
        return float(psi), float(se)

    def diagnose(self, n: int = 1_000_000, seed: int = 1) -> dict:
        """Overlap, variance explained, and nonlinearity, on one large draw."""
        from sklearn.linear_model import LogisticRegression
        d = self.draw(n, np.random.default_rng(seed))
        X, a = d[list(self.covariates)].to_numpy(), d["A"].to_numpy()
        pi, m = self.propensity(X), self.mu(a, X)
        design = np.column_stack([np.ones(n), a, X])
        m_lin = design @ np.linalg.lstsq(design, m, rcond=None)[0]
        sub = slice(0, 200_000)
        rho_lin = LogisticRegression(C=np.inf, max_iter=1000).fit(X[sub], a[sub]).decision_function(X[sub])
        r = np.log(pi[sub] / (1 - pi[sub]))
        out = {
            "P(A=1)": a.mean(),
            "pi 1st pct": np.quantile(pi, 0.01),
            "pi 99th pct": np.quantile(pi, 0.99),
            "pi min": pi.min(),
            "pi max": pi.max(),
            "Var(mu)/Var(Y)": m.var() / d["Y"].var(),
            "linear share of Var(mu)": m_lin.var() / m.var(),
            "linear share of Var(rho)": 1 - np.mean((r - rho_lin) ** 2) / r.var(),
            "Var(tau)": self.tau(X).var(),
        }
        for est in ("ATE", "ATT"):
            alpha = self.true_alpha(est, a, X)
            psi, se = self.true_psi(est)
            out[f"{est} psi0"] = psi
            out[f"{est} psi0 MC SE"] = se
            out[f"{est} sd(alpha0)"] = alpha.std()
            out[f"{est} max|alpha0|"] = np.abs(alpha).max()
        return out


# ------------------------------------------------------- five_confounders ---
# Module-level named functions, so a DGP pickles into worker processes.

def _unif5(n, rng):
    return rng.uniform(-1, 1, size=(n, 5))


def _fc_propensity(X):
    x1, x2, x3, x4, x5 = X.T
    return expit(0.9 * x1 - 0.9 * x2 + 1.2 * (x3**2 - 1 / 3) + 1.0 * x4 * x5)


def _fc_mu(a, X):
    x1, x2, x3, x4, x5 = X.T
    tau = 1.0 + 0.6 * x1 + 0.5 * np.cos(np.pi * x5)
    return (1.5 * np.sin(np.pi * x1 / 2) - x2**3 + 0.8 * x3 + 0.8 * x4
            + 0.6 * np.cos(np.pi * x5) + a * tau)


# ----------------------------------------------------------- shared_terms ---

def _st_propensity(X):
    x1, x2, x3, x4, x5 = X.T
    return expit(0.8 * x1 - 0.8 * x2 + 1.5 * (x3**2 - 1 / 3) + 1.2 * x4 * x5)


def _st_mu(a, X):
    x1, x2, x3, x4, x5 = X.T
    tau = 1.0 + 0.8 * x1 + (x3**2 - 1 / 3)
    return x1 - 0.8 * x2 + np.sin(np.pi * x3) + 0.8 * x4 * x5 + a * tau


# ------------------------------------------------------------ lee_schuler ---

def _ls_x(n, rng):
    return rng.uniform(0, 1, size=(n, 1))


def _ls_propensity(X):
    x = X[:, 0]
    return expit(-0.02 * x - x**2 + 4.0 * np.log(x + 0.3) + 1.5)


def _ls_mu(a, X):
    x = X[:, 0]
    return 5 * x + 9 * x * a + 5 * np.sin(np.pi * x) + 25 * (a - 2)


FIVE = tuple(f"X{j}" for j in range(1, 6))
DGPS = {
    "five_confounders": DGP("five_confounders", FIVE, _unif5, _fc_propensity, _fc_mu,
                            sigma=1.0, true_ate=1.0),
    "shared_terms": DGP("shared_terms", FIVE, _unif5, _st_propensity, _st_mu,
                        sigma=1.0, true_ate=1.0),
    "lee_schuler": DGP("lee_schuler", ("X",), _ls_x, _ls_propensity, _ls_mu,
                       sigma=1.0, true_ate=29.5),
}
DEFAULT = "five_confounders"


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or list(DGPS):
        print(f"== {name} ==")
        for k, v in DGPS[name].diagnose().items():
            print(f"{k:28s} {v:10.4f}")
