"""Simulation comparing Riesz representer learners for the ATE and ATT on an
easier DGP than the one in `dgp.py`: 10 covariates instead of 40, 4
confounders instead of 20, and nuisance functions that are mostly linear.

The learners, tuning grids, outcome regression, and one-step estimators are
imported from `run_backend_comparison.py`, so the two simulations differ only
in the DGP.

DGP (covariates marginally N(0,1), AR(1)-correlated with rho=0.3, noise
sd 2.5 as in `dgp.py`):
  - confounders (x1-x4), outcome-only (x5-x6), treatment-only (x7-x8),
    noise (x9-x10)
  - logit P(A=1|X) is linear in x1-x4 and x7-x8, plus a centered quadratic
    term in x1
  - mu(a, X) is linear in x1-x6, plus a centered quadratic term in x2, with
    treatment effect 3 + 0.5 x1

Run:
    python run_easy_dgp.py --n_reps 20 --n_jobs 8
"""

import os

# One thread per process: torch (riesznet) and xgboost (rieszboost) can crash
# on macOS when both use multithreaded OpenMP. Set before either is imported.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.model_selection import GridSearchCV, KFold

import dgp
from rieszboost import ATE, ATT
from run_backend_comparison import (
    BASE_SEED, CV_FOLDS, LEARNERS, N_DATA, N_FOLDS, N_MC_TRUTH, N_OUTCOME_TREES,
    OUTCOME_GRID, XGBRegressorES, ate_estimator, att_estimator, summarize,
)


""" Easier DGP, with the same interface as the classes in dgp.py """
class FewCovariateDGP(dgp.ManyCovariateDGP):
    n_confounders = 4
    n_outcome_only = 2
    n_treatment_only = 2
    n_noise = 2
    n_covariates = n_confounders + n_outcome_only + n_treatment_only + n_noise  # 10

    confounders = slice(0, 4)             # x1-x4:  affect A and Y
    outcome_only = slice(4, 6)            # x5-x6:  affect Y only
    treatment_only = slice(6, 8)          # x7-x8:  affect A only
    # x9-x10 (indices 8-9): pure noise, no effect on A or Y

    feature_names = [f"x{i + 1}" for i in range(n_covariates)]
    _chol = dgp._ar1_cholesky(n_covariates, dgp.ManyCovariateDGP.ar1_rho)

    _pi_conf_coefs = np.array([0.5, -0.4, 0.3, -0.2])
    _pi_treatment_only_coefs = np.array([0.3, -0.3])
    _mu_conf_coefs = np.array([1.0, -0.8, 0.6, -0.4])
    _mu_outcome_only_coefs = np.array([0.8, -0.5])
    _te_intercept = 3.0

    @classmethod
    def expected_trt(cls, x):
        logit_pi = (
            0.1
            + x[:, cls.confounders] @ cls._pi_conf_coefs
            + 0.25 * (x[:, 0] ** 2 - 1.0)
            + x[:, cls.treatment_only] @ cls._pi_treatment_only_coefs
        )
        return np.clip(expit(logit_pi), *cls.pi_clip)

    @classmethod
    def expected_outcome(cls, a, x):
        baseline = (
            x[:, cls.confounders] @ cls._mu_conf_coefs
            + 0.5 * (x[:, 1] ** 2 - 1.0)
            + x[:, cls.outcome_only] @ cls._mu_outcome_only_coefs
        )
        return baseline + a * (cls._te_intercept + 0.5 * x[:, 0])


# The representer and the Monte Carlo truth come from dgp.ATE / dgp.ATT
class EasyATE(FewCovariateDGP, dgp.ATE):
    pass


class EasyATT(FewCovariateDGP, dgp.ATT):
    pass


# estimand -> (DGP, rieszboost estimand, one-step estimator)
covariates = tuple(FewCovariateDGP.feature_names)
ESTIMANDS = {
    "ATE": (EasyATE, ATE(treatment="a", covariates=covariates), ate_estimator),
    "ATT": (EasyATT, ATT(treatment="a", covariates=covariates), att_estimator),
}


""" Functions for running simulation reps """
def run_rep(rep, n=N_DATA, cv_folds=CV_FOLDS):
    seed = BASE_SEED + rep
    rows = []
    for name, (dgp_cls, estimand, estimator) in ESTIMANDS.items():
        # Same seed for both estimands, so ATE and ATT see the same data
        Y, A, X = dgp_cls.gen_data(n, np.random.default_rng(seed))
        Z = dgp_cls.to_frame(A, X)
        true_mu = dgp_cls.expected_outcome(A, X)
        true_alpha = dgp_cls.riesz_rep(A, X)
        folds = list(KFold(N_FOLDS, shuffle=True, random_state=seed).split(Z))

        # Outcome regression: out-of-fold mu(A, X), mu(1, X), mu(0, X)
        mu, mu1, mu0 = np.empty(n), np.empty(n), np.empty(n)
        for train, test in folds:
            outcome_model = GridSearchCV(
                XGBRegressorES(n_estimators=N_OUTCOME_TREES, early_stopping_rounds=20, n_jobs=1, random_state=seed),
                OUTCOME_GRID, cv=cv_folds,
            ).fit(Z.iloc[train], Y[train])
            mu[test] = outcome_model.predict(Z.iloc[test])
            mu1[test] = outcome_model.predict(Z.iloc[test].assign(a=1.0))
            mu0[test] = outcome_model.predict(Z.iloc[test].assign(a=0.0))

        # Riesz representer: out-of-fold alpha(A, X) for each learner
        for learner, (make_model, grid) in LEARNERS.items():
            start = time.time()
            alpha = np.empty(n)
            for train, test in folds:
                riesz_model = GridSearchCV(
                    make_model(estimand, seed), grid, cv=cv_folds,
                ).fit(Z.iloc[train])
                alpha[test] = riesz_model.predict(Z.iloc[test])

            est, se = estimator(A, Y, mu, mu1, mu0, alpha)
            rows.append({
                "rep": rep,
                "estimand": name,
                "learner": learner,
                "est": est,
                "se": se,
                "mu_rmse": np.sqrt(np.mean((mu - true_mu) ** 2)),
                "alpha_rmse": np.sqrt(np.mean((alpha - true_alpha) ** 2)),
                "alpha_mae": np.mean(np.abs(alpha - true_alpha)),
                "fit_seconds": time.time() - start,
            })
    return rows


""" Main execution """
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=1, help="replicates to run in parallel")
    parser.add_argument("--n_data", type=int, default=N_DATA)
    parser.add_argument("--cv_folds", type=int, default=CV_FOLDS)
    parser.add_argument("--out_csv", type=str, default="results_easy.csv")
    args = parser.parse_args()

    true_psi = {
        name: dgp_cls.true_psi(np.random.default_rng(12345), N_MC_TRUTH)
        for name, (dgp_cls, _, _) in ESTIMANDS.items()
    }
    print(f"True psi: {true_psi}", flush=True)

    # Hand the workers run_rep from the importable module, not from __main__:
    # classes defined in __main__ (the DGP above) fail to unpickle in joblib
    # worker processes
    from run_easy_dgp import run_rep
    rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_rep)(rep, n=args.n_data, cv_folds=args.cv_folds) for rep in range(args.n_reps)
    )
    results = pd.DataFrame([row for rep_rows in rows for row in rep_rows])
    results["truth"] = results.estimand.map(true_psi)
    results.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    print(summarize(results).round(3).to_string())
