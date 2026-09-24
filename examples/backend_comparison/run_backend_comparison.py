"""Simulation comparing Riesz representer learners for the ATE and ATT.

Data come from the 40-covariate DGP in `dgp.py`. Each replicate:
  1. draws N_DATA observations,
  2. cross-fits the outcome regression mu and the Riesz representer alpha
     over N_FOLDS folds, tuning the hyperparameters of both by GridSearchCV
     within each training fold and refitting on the whole training fold with
     the selected values (every learner also early-stops on a held-out 20% of
     its training rows),
  3. plugs mu_hat and alpha_hat into the one-step estimator of psi,
  4. records the estimate, its standard error, and the errors of mu_hat and
     alpha_hat against the true outcome regression and representer.

Riesz representer learners:
  xgboost      : rieszboost, boosting with XGBoost trees: learning rate and
                 tree depth tuned by 5-fold CV,
                 early stopping with patience 20 up to 20000 trees, and
                 stochastic boosting on a random 80% of individuals each round
  riesznet     : riesznet, a neural network set up as in Lee & Schuler: 3 hidden
                 layers of width 200, ELU, AdamW with weight decay 1e-3, early
                 stopping with patience 10, the learning rate tuned by 5-fold
                 CV, and the best of 2 random restarts

Run:
    python run_backend_comparison.py --n_reps 20 --n_jobs 8
"""

import argparse
import time
from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from xgboost import XGBRegressor

import dgp
from rieszboost import ATE, ATT, RieszBooster
from riesznet import RieszNet

""" Parameters for simulation """
N_DATA = 2000
N_FOLDS = 2          # cross-fitting folds
CV_FOLDS = 5         # GridSearchCV folds for tuning mu and alpha
BASE_SEED = 510
N_MC_TRUTH = 2_000_000
# Every learner fits up to a maximum number of trees / epochs; early stopping picks how many to keep
N_OUTCOME_TREES = 10_000
N_RIESZBOOST_TREES = 20_000
N_EPOCHS = 1_000     # riesznet
N_RESTARTS = 2       # riesznet random restarts
RIESZBOOST_GRID = {"max_depth": [3, 5, 7], "learning_rate": [0.0001, 0.001, 0.01]}
EARLY_STOP = dict(early_stopping_rounds=20, validation_fraction=0.2)   # rieszboost
RIESZBOOST_SUBSAMPLE = 0.8   # fraction of individuals drawn each boosting round
NET_LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.01]   # grid for riesznet


""" One-step estimators """
def ate_estimator(A, Y, mu, mu1, mu0, alpha):
    ic = mu1 - mu0 + alpha * (Y - mu)
    return ic.mean(), ic.std(ddof=1) / np.sqrt(len(Y))


def att_estimator(A, Y, mu, mu1, mu0, alpha):
    # alpha is the partial-parameter representer, so divide by P(A=1)
    p = A.mean()
    est = np.mean(A * (mu1 - mu0) + alpha * (Y - mu)) / p
    ic = (A * (mu1 - mu0 - est) + alpha * (Y - mu)) / p
    return est, ic.std(ddof=1) / np.sqrt(len(Y))


# estimand -> (DGP, rieszboost estimand, one-step estimator)
covariates = tuple(dgp.ManyCovariateDGP.feature_names)
ESTIMANDS = {
    "ATE": (dgp.ATE, ATE(treatment="a", covariates=covariates), ate_estimator),
    "ATT": (dgp.ATT, ATT(treatment="a", covariates=covariates), att_estimator),
}


""" Outcome regression model: XGBoost with early stopping, tuned over this grid """
class XGBRegressorES(XGBRegressor):
    """XGBRegressor that holds out `validation_fraction` of the training rows
    and stops adding trees once the held-out error stops improving."""

    def fit(self, X, y, validation_fraction=0.2):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=validation_fraction, random_state=self.random_state)
        return super().fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)


OUTCOME_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.001],
    "reg_lambda": [0.1, 1.0],
}


""" Riesz representer learners """
class RieszNetAdamW(RieszNet):
    """RieszNet trained with AdamW (decoupled weight decay) instead of Adam."""

    def _resolved_backend(self):
        return replace(super()._resolved_backend(), optimizer_factory=partial(
            torch.optim.AdamW, lr=self.learning_rate, weight_decay=self.weight_decay))


class BestOfRestarts(BaseEstimator):
    """Fits `estimator` from `n_restarts` random initializations and keeps the
    fit with the lowest validation loss."""

    def __init__(self, estimator, n_restarts=N_RESTARTS):
        self.estimator = estimator
        self.n_restarts = n_restarts

    def fit(self, Z, y=None):
        seed = self.estimator.random_state
        fits = [clone(self.estimator).set_params(random_state=seed + k).fit(Z)
                for k in range(self.n_restarts)]
        self.best_ = min(fits, key=lambda fit: fit.best_score_)
        return self

    def predict(self, Z):
        return self.best_.predict(Z)

    def score(self, Z, y=None):
        return self.best_.score(Z)


# name -> (unfitted model for a given estimand and seed, GridSearchCV grid)
LEARNERS = {
    "xgboost": (
        lambda estimand, seed: RieszBooster(
            estimand=estimand, random_state=seed, n_estimators=N_RIESZBOOST_TREES,
            subsample=RIESZBOOST_SUBSAMPLE, **EARLY_STOP),
        RIESZBOOST_GRID,
    ),
    "riesznet": (
        lambda estimand, seed: BestOfRestarts(RieszNetAdamW(
            estimand=estimand, random_state=seed,
            hidden_sizes=(200, 200, 200), activation="elu", weight_decay=1e-3,
            epochs=N_EPOCHS, early_stopping_rounds=10, validation_fraction=0.2)),
        {"estimator__learning_rate": NET_LEARNING_RATES},
    ),
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


def summarize(results):
    results["error"] = results.est - results.truth
    results["covered"] = np.abs(results.error) <= 1.96 * results.se
    return results.groupby(["estimand", "learner"], sort=False).agg(
        bias=("error", "mean"),
        emp_sd=("est", "std"),
        avg_se=("se", "mean"),
        rmse=("error", lambda e: np.sqrt(np.mean(e**2))),
        coverage=("covered", "mean"),
        mu_rmse=("mu_rmse", "mean"),
        alpha_rmse=("alpha_rmse", "mean"),
        alpha_mae=("alpha_mae", "mean"),
        seconds=("fit_seconds", "mean"),
    )


""" Main execution """
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=1, help="replicates to run in parallel")
    parser.add_argument("--out_csv", type=str, default="results.csv")
    args = parser.parse_args()

    true_psi = {
        name: dgp_cls.true_psi(np.random.default_rng(12345), N_MC_TRUTH)
        for name, (dgp_cls, _, _) in ESTIMANDS.items()
    }
    print(f"True psi: {true_psi}", flush=True)

    rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_rep)(rep) for rep in range(args.n_reps)
    )
    results = pd.DataFrame([row for rep_rows in rows for row in rep_rows])
    results["truth"] = results.estimand.map(true_psi)
    results.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    print(summarize(results).round(3).to_string())
