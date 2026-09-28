"""Why is the one-step ATE biased with a GBT propensity plug-in on the main DGP?

Crosses the outcome regression used in the one-step estimator,
  mu-hat in {xgb (the pipeline's), ols (correctly specified), true},
with the source of alpha-hat,
  alpha in {oracle, xgb (inverted XGBoost pi-hat), logit (inverted, correctly specified logistic pi-hat)},
all cross-fitted on the pipeline's 5 folds, at several n. If the XGBoost
plug-in is unbiased once mu-hat is good, the bias is the second-order
remainder E[(alpha-hat - alpha0)(mu0 - mu-hat)] from errors that line up, not
a problem with pi-hat alone. If it is still biased with the true mu, there is
a bug. Also records how well pi-hat is calibrated.

  ../../.venv/bin/python diagnose_nuisance.py --reps 40 --n 1000 2000 8000
"""

from __future__ import annotations

import _env  # noqa: F401

import argparse

import numpy as np
import pandas as pd

import dgp
import run

# Correctly specified bases for mu(a, X) and for the log-odds rho(X), per DGP.

def _mu_basis_shared(a, X):
    x1, x2, x3, x4, x5 = X.T
    return np.column_stack([np.ones(len(a)), x1, x2, np.sin(np.pi * x3), x4 * x5,
                            a, a * x1, a * (x3**2 - 1 / 3)])


def _rho_basis_shared(X):
    x1, x2, x3, x4, x5 = X.T
    return np.column_stack([x1, x2, x3**2, x4 * x5])


def _mu_basis_five(a, X):
    x1, x2, x3, x4, x5 = X.T
    return np.column_stack([np.ones(len(a)), np.sin(np.pi * x1 / 2), x2**3, x3, x4,
                            np.cos(np.pi * x5), a, a * x1, a * np.cos(np.pi * x5)])


BASES = {"shared_terms": (_mu_basis_shared, _rho_basis_shared),
         "five_confounders": (_mu_basis_five, _rho_basis_shared)}
DGP_NAME = "shared_terms"
G = dgp.DGPS[DGP_NAME]


def _mu_basis(a, X):
    return BASES[DGP_NAME][0](a, X)


def _rho_basis(X):
    return BASES[DGP_NAME][1](X)


def parametric_oof(d, folds):
    """Cross-fitted OLS mu-hat and logistic pi-hat on the true bases."""
    from sklearn.linear_model import LogisticRegression
    X, a, y = d[list(G.covariates)].to_numpy(), d["A"].to_numpy(), d["Y"].to_numpy()
    mu, mu1, mu0, pi = (np.empty(len(d)) for _ in range(4))
    for k in range(run.K):
        tr, te = folds != k, folds == k
        beta = np.linalg.lstsq(_mu_basis(a[tr], X[tr]), y[tr], rcond=None)[0]
        mu[te] = _mu_basis(a[te], X[te]) @ beta
        mu1[te] = _mu_basis(np.ones(te.sum()), X[te]) @ beta
        mu0[te] = _mu_basis(np.zeros(te.sum()), X[te]) @ beta
        pi[te] = LogisticRegression(C=np.inf, max_iter=2000).fit(
            _rho_basis(X[tr]), a[tr]).predict_proba(_rho_basis(X[te]))[:, 1]
    return pd.DataFrame({"mu": mu, "mu1": mu1, "mu0": mu0}), pi


def diagnose(n: int, rep: int, dgp_name: str = DGP_NAME) -> tuple[list[dict], pd.DataFrame]:
    global DGP_NAME, G
    DGP_NAME, G = dgp_name, dgp.DGPS[dgp_name]
    d, folds = run.cell_data(DGP_NAME, n, rep)
    X, a = d[list(G.covariates)].to_numpy(), d["A"].to_numpy()
    mu_xgb, _ = run.select(d, folds, "outcome", DGP_NAME, n, rep)
    pi_xgb = run.select(d, folds, "propensity", DGP_NAME, n, rep)[0]["pi"].to_numpy()
    mu_ols, pi_logit = parametric_oof(d, folds)
    mu_true = pd.DataFrame({"mu": G.mu(a, X), "mu1": G.mu(np.ones(n), X), "mu0": G.mu(np.zeros(n), X)})
    pi0 = G.propensity(X)
    alpha0 = G.true_alpha("ATE", a, X)
    alphas = {"oracle": alpha0,
              "xgb": run.propensity_alpha("ATE", a, pi_xgb)["alpha"].to_numpy(),
              "logit": run.propensity_alpha("ATE", a, pi_logit)["alpha"].to_numpy()}
    rows = []
    for mu_name, mu in (("xgb", mu_xgb), ("ols", mu_ols), ("true", mu_true)):
        mu_err = mu["mu"].to_numpy() - G.mu(a, X)
        plugin = float((mu["mu1"] - mu["mu0"]).mean())
        for al_name, al in alphas.items():
            est, se = run.one_step("ATE", d, mu, al)
            rows.append(dict(n=n, rep=rep, mu_hat=mu_name, alpha=al_name, est=est, se=se,
                             plugin=plugin, remainder=float(np.mean((al - alpha0) * -mu_err)),
                             mu_rmse=float(np.sqrt(np.mean(mu_err**2))),
                             alpha_rmse=float(np.sqrt(np.mean((al - alpha0) ** 2)))))
    calib = pd.DataFrame({"n": n, "rep": rep, "pi_hat": pi_xgb, "pi_logit": pi_logit,
                          "pi0": pi0, "A": a})
    return rows, calib


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n", type=int, nargs="+", default=[1000, 2000, 8000])
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--dgp", default="shared_terms", choices=list(BASES))
    args = ap.parse_args()
    DGP_NAME, G = args.dgp, dgp.DGPS[args.dgp]
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs, batch_size=1)(
        delayed(diagnose)(n, r, DGP_NAME) for n in sorted(args.n, reverse=True) for r in range(args.reps))
    res = pd.DataFrame([row for rows, _ in out for row in rows])
    calib = pd.concat([c for _, c in out], ignore_index=True)
    res.to_csv(run.HERE / "results" / f"diagnose_nuisance_{DGP_NAME}.csv", index=False)

    pd.set_option("display.width", 250)
    res["err"] = res["est"] - G.true_psi("ATE")[0]
    res["cover"] = res["err"].abs() <= 1.96 * res["se"]
    g = res.groupby(["n", "mu_hat", "alpha"])
    R = g.size()
    tab = pd.DataFrame({"bias": g.err.mean(), "mcse": g.err.std(ddof=1) / np.sqrt(R),
                        "mean_se": g.se.mean(), "coverage": g.cover.mean(),
                        "remainder": g.remainder.mean(), "plugin_bias": g.plugin.mean() - G.true_psi("ATE")[0],
                        "mu_rmse": g.mu_rmse.mean(), "alpha_rmse": g.alpha_rmse.mean()})
    print(tab.round(3).to_string())

    print("\nCalibration of the XGBoost pi-hat, pooled over reps (bins of pi-hat):")
    calib["bin"] = pd.cut(calib["pi_hat"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    print(calib.groupby(["n", "bin"], observed=True).agg(
        mean_pi_hat=("pi_hat", "mean"), mean_A=("A", "mean"), mean_pi0=("pi0", "mean"),
        rows=("A", "size")).round(3).to_string())
    print("\nShrinkage: slope of pi-hat on pi0 (1 = no shrinkage)")
    for n, c in calib.groupby("n"):
        print(n, "xgb", round(np.polyfit(c.pi0, c.pi_hat, 1)[0], 3), " logit", round(np.polyfit(c.pi0, c.pi_logit, 1)[0], 3))
