"""The binary-treatment DGP of the RieszBoost ICML manuscript (Lee & Schuler),
Section 3.3 and Appendix C.2.1: ten confounders, estimands ATE and ATT.

    X1, ..., X10 ~ Uniform(0, 1), independently
    A | X ~ Bernoulli(pi(X)),   pi(X) = expit(rho(X))
    rho(X) = -0.7 X1 - 0.8 (X2 + X6)^2 - 0.8 X3 X4 + 0.6 sin(4 pi X5)
             + 0.6 X7 - 0.7 log(X8 + X9 + 2 X10) + 1.9
    Y | A, X ~ N(mu(A, X), 1)
    mu(a, X) = 5a - 4 X1 + 3 (X2 + X3)(2a - 1) + 8 X5 a + 0.5 (2 X4 - X6)^2
               + 8 expit(0.8 (2a - 1) + 0.5 X7 - X8) + 4 cos(pi a (X9 + X10))

The manuscript prints the sine term as "0.6 sin(45)"; the authors' code
(icml_rieszboost/rrboost/dgp_many_cov.py) has 0.6 sin(4 pi X5), used here.

The ATT's Riesz representer is that of the partial parameter
E[A (mu(1, X) - mu(0, X))], as in the manuscript's Appendix B.1:
    alpha_ATE(a, x) = a / pi(x) - (1 - a) / (1 - pi(x))
    alpha_ATT(a, x) = a - (1 - a) pi(x) / (1 - pi(x))

  python dgp.py          # true values and DGP diagnostics on one large draw
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COV = [f"X{j}" for j in range(1, 11)]

# True values from `truth()`: 10 draws of 2e6 covariate vectors (MCSE 0.001).
# The manuscript reports 12.315 and 11.843; its ATT comes from one unseeded
# draw of 1e6, whose Monte Carlo error is about the size of the gap.
PSI = {"ATE": 12.3146, "ATT": 11.8353}


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def propensity(X):
    x = [X[:, j] for j in range(10)]
    return expit(-0.7 * x[0] - 0.8 * (x[1] + x[5]) ** 2 - 0.8 * x[2] * x[3] + 0.6 * np.sin(4 * np.pi * x[4])
                 + 0.6 * x[6] - 0.7 * np.log(x[7] + x[8] + 2 * x[9]) + 1.9)


def outcome(a, X):
    x = [X[:, j] for j in range(10)]
    return (5 * a - 4 * x[0] + 3 * (x[1] + x[2]) * (2 * a - 1) + 8 * x[4] * a + 0.5 * (2 * x[3] - x[5]) ** 2
            + 8 * expit(0.8 * (2 * a - 1) + 0.5 * x[6] - x[7]) + 4 * np.cos(a * np.pi * (x[8] + x[9])))


def true_alpha(estimand, a, X):
    pi = propensity(X)
    return a / pi - (1 - a) / (1 - pi) if estimand == "ATE" else a - (1 - a) * pi / (1 - pi)


def draw(n, rng):
    """One dataset of n rows: columns X1..X10, A, Y."""
    X = rng.uniform(0, 1, (n, 10))
    A = rng.binomial(1, propensity(X)).astype(float)
    Y = outcome(A, X) + rng.normal(0, 1, n)
    return pd.DataFrame(X, columns=COV).assign(A=A, Y=Y)


def truth(n_draws=10, n=2_000_000, seed=1):
    """True ATE and ATT, with their Monte Carlo standard errors."""
    rng = np.random.default_rng(seed)
    est = {"ATE": [], "ATT": []}
    for _ in range(n_draws):
        X = rng.uniform(0, 1, (n, 10))
        pi, tau = propensity(X), outcome(1, X) - outcome(0, X)
        est["ATE"].append(tau.mean())
        est["ATT"].append((pi * tau).mean() / pi.mean())
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(n_draws))) for k, v in est.items()}


def diagnostics(n=1_000_000, seed=2):
    """Overlap, the size of alpha_0, and how much of Var(Y) mu explains."""
    d = draw(n, np.random.default_rng(seed))
    X, A, Y = d[COV].to_numpy(), d["A"].to_numpy(), d["Y"].to_numpy()
    pi = propensity(X)
    mu = outcome(A, X)
    out = {"P(A=1)": A.mean()}
    out.update({f"pi q{int(100 * q)}": np.quantile(pi, q) for q in (0.01, 0.05, 0.5, 0.95, 0.99)})
    for e in ("ATE", "ATT"):
        a0 = true_alpha(e, A, X)
        out[f"sd alpha_{e}"] = a0.std()
        out[f"max |alpha_{e}|"] = np.abs(a0).max()
    out["Var(mu)/Var(Y)"] = mu.var() / Y.var()
    return out


if __name__ == "__main__":
    for k, (v, se) in truth().items():
        print(f"psi_{k} = {v:.4f} (MCSE {se:.4f})")
    for k, v in diagnostics().items():
        print(f"{k:>16}: {v:.3f}")
