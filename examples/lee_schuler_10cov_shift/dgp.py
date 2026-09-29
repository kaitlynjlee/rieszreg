"""The continuous-exposure DGP of the RieszBoost ICML manuscript (Lee &
Schuler), Section 3.4 and Appendix C.2.2: ten confounders, estimands ASE and
LASE of the shift A -> A + delta, delta = 1, with the LASE's shift applied
only where A < t, t = 0.

    X1, ..., X10 ~ Uniform(0, 1), independently
    A | X ~ N(m(X), 3^2)
    m(X) = -1.1 X1 - 0.8 (X2 + X6)^2 - 0.8 X3 X4 + 0.6 sin(4 pi X5)
           + 0.6 X7 - 0.7 log(X8 + X9 + 2 X10) + 3
    Y | A, X ~ N(mu(A, X), 1)
    mu(a, X) = 3a - 10 X1 + 2 (X2 + X3) sqrt(|a|) + 1.5 X5 a + 0.5 (2 X4 - X6)^2
               + 10 expit(0.5 a + 0.5 X7 - 1.5 X8) + cos(pi a (X9 + X10))

    psi_ASE  = E[mu(A + delta, X) - mu(A, X)]
    psi_LASE = E[mu(A + delta, X) - mu(A, X) | A < t]

The manuscript prints the exposure variance as 3; the authors' code
(icml_rieszboost/rrboost/dgp_many_cov.py) uses standard deviation 3, and only
that reproduces the manuscript's psi_ASE = 4.786 and psi_LASE = 4.058
(variance 3 gives 5.098 and 4.289). Standard deviation 3 is used here.

With r(a, x) = p(a - delta | x) / p(a | x) = exp((delta (a - m(x)) - delta^2 / 2) / sigma^2),
the representers are (the LASE's is that of the partial parameter
E[1(A < t) (mu(A + delta, X) - mu(A, X))], as in the manuscript's Appendix B.3):
    alpha_ASE(a, x)  = r(a, x) - 1
    alpha_LASE(a, x) = 1(a < t + delta) r(a, x) - 1(a < t)

  python dgp.py          # true values and DGP diagnostics on one large draw
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COV = [f"X{j}" for j in range(1, 11)]
DELTA = 1.0       # the shift
THRESHOLD = 0.0   # the LASE shifts only units with A < THRESHOLD
SIGMA_A = 3.0     # sd of A given X

# True values from `truth()`: 10 draws of 2e6 (X, A) pairs (MCSE 0.0002 and
# 0.0004). The manuscript reports 4.786 and 4.058.
PSI = {"ASE": 4.7857, "LASE": 4.0584}


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def exposure_mean(X):
    x = [X[:, j] for j in range(10)]
    return (-1.1 * x[0] - 0.8 * (x[1] + x[5]) ** 2 - 0.8 * x[2] * x[3] + 0.6 * np.sin(4 * np.pi * x[4])
            + 0.6 * x[6] - 0.7 * np.log(x[7] + x[8] + 2 * x[9]) + 3)


def outcome(a, X):
    x = [X[:, j] for j in range(10)]
    return (3 * a - 10 * x[0] + 2 * (x[1] + x[2]) * np.sqrt(np.abs(a)) + 1.5 * x[4] * a
            + 0.5 * (2 * x[3] - x[5]) ** 2 + 10 * expit(0.5 * a + 0.5 * x[6] - 1.5 * x[7])
            + np.cos(a * np.pi * (x[8] + x[9])))


def density_ratio(a, X):
    """r(a, x) = p(a - delta | x) / p(a | x)."""
    return np.exp((DELTA * (a - exposure_mean(X)) - DELTA ** 2 / 2) / SIGMA_A ** 2)


def true_alpha(estimand, a, X):
    r = density_ratio(a, X)
    if estimand == "ASE":
        return r - 1
    return (a < THRESHOLD + DELTA) * r - (a < THRESHOLD)


def draw(n, rng):
    """One dataset of n rows: columns X1..X10, A, Y."""
    X = rng.uniform(0, 1, (n, 10))
    A = rng.normal(exposure_mean(X), SIGMA_A)
    Y = outcome(A, X) + rng.normal(0, 1, n)
    return pd.DataFrame(X, columns=COV).assign(A=A, Y=Y)


def truth(n_draws=10, n=2_000_000, seed=1):
    """True ASE and LASE, with their Monte Carlo standard errors."""
    rng = np.random.default_rng(seed)
    est = {"ASE": [], "LASE": []}
    for _ in range(n_draws):
        X = rng.uniform(0, 1, (n, 10))
        A = rng.normal(exposure_mean(X), SIGMA_A)
        tau = outcome(A + DELTA, X) - outcome(A, X)
        est["ASE"].append(tau.mean())
        est["LASE"].append(tau[A < THRESHOLD].mean())
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(n_draws))) for k, v in est.items()}


def diagnostics(n=1_000_000, seed=2):
    """Share shifted by the LASE, the spread of the density ratio (overlap
    for the shift), the size of alpha_0, and how much of Var(Y) mu explains."""
    d = draw(n, np.random.default_rng(seed))
    X, A, Y = d[COV].to_numpy(), d["A"].to_numpy(), d["Y"].to_numpy()
    r = density_ratio(A, X)
    mu = outcome(A, X)
    out = {"P(A < t)": np.mean(A < THRESHOLD), "P(t <= A < t + delta)": np.mean((A >= THRESHOLD) & (A < THRESHOLD + DELTA))}
    out.update({f"A q{int(100 * q)}": np.quantile(A, q) for q in (0.01, 0.5, 0.99)})
    out.update({f"r q{int(100 * q)}": np.quantile(r, q) for q in (0.01, 0.05, 0.5, 0.95, 0.99)})
    out["max r"] = r.max()
    for e in ("ASE", "LASE"):
        a0 = true_alpha(e, A, X)
        out[f"sd alpha_{e}"] = a0.std()
        out[f"max |alpha_{e}|"] = np.abs(a0).max()
    out["Var(mu)/Var(Y)"] = mu.var() / Y.var()
    return out


if __name__ == "__main__":
    for k, (v, se) in truth().items():
        print(f"psi_{k} = {v:.4f} (MCSE {se:.4f})")
    for k, v in diagnostics().items():
        print(f"{k:>22}: {v:.3f}")
