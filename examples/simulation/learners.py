"""Learners, their fixed settings, and their tuning grids.

Every learner type here has a grid of settings. Each setting is fit on the
same K cross-fitting folds, its out-of-fold predictions give its
cross-validated risk, and the setting with the smallest risk is selected
(a discrete super learner within each learner type). The selected setting's
out-of-fold predictions are then the cross-fitted nuisance estimate.

A learner function takes (train, test, **setting) and returns
(predictions on test as a DataFrame, iterations kept by early stopping).
Everything that determines a fit is in its source and its setting, which is
what the cache key hashes.
"""

from __future__ import annotations

import itertools

import _env  # noqa: F401  (threads and import paths; must precede numpy)
import numpy as np
import pandas as pd

SEED = 0            # every learner with internal randomness gets this seed
VALID_FRAC = 0.2    # held-out share of each training fold used for early stopping
PATIENCE = 20       # early-stopping patience, in trees or epochs
L2_PENALTY = 0.0    # L2 penalty on leaf values, for every boosted learner


def covariates(df: pd.DataFrame) -> list[str]:
    """Every column except the treatment A and the outcome Y."""
    return [c for c in df.columns if c not in ("A", "Y")]


def _grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]


# ---------------------------------------------------------------- xgboost ---
# Outcome regression and propensity score, fit with ordinary xgboost.

def _xgb_split(train: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    is_valid = rng.random(len(train)) < VALID_FRAC
    return train[~is_valid], train[is_valid]


def outcome_xgb(train, test, max_iter, max_depth, learning_rate):
    import xgboost as xgb
    fit_part, valid = _xgb_split(train)
    cols = ["A"] + covariates(train)
    model = xgb.XGBRegressor(
        n_estimators=max_iter, learning_rate=learning_rate, max_depth=max_depth,
        reg_lambda=L2_PENALTY, early_stopping_rounds=PATIENCE, n_jobs=1, random_state=SEED,
    ).fit(fit_part[cols], fit_part["Y"], eval_set=[(valid[cols], valid["Y"])], verbose=False)
    Z = test[cols]
    preds = pd.DataFrame({
        "mu": model.predict(Z),
        "mu1": model.predict(Z.assign(A=1.0)),
        "mu0": model.predict(Z.assign(A=0.0)),
    })
    return preds, model.best_iteration + 1


def propensity_xgb(train, test, max_iter, max_depth, learning_rate):
    import xgboost as xgb
    fit_part, valid = _xgb_split(train)
    X = covariates(train)
    model = xgb.XGBClassifier(
        n_estimators=max_iter, learning_rate=learning_rate, max_depth=max_depth,
        reg_lambda=L2_PENALTY, early_stopping_rounds=PATIENCE, n_jobs=1, random_state=SEED,
    ).fit(fit_part[X], fit_part["A"], eval_set=[(valid[X], valid["A"])], verbose=False)
    return pd.DataFrame({"pi": model.predict_proba(test[X])[:, 1]}), model.best_iteration + 1


# ------------------------------------------------------ Riesz regression ---

def _estimand(name, covs):
    from rieszreg import ATE, ATT
    return {"ATE": ATE, "ATT": ATT}[name](treatment="A", covariates=tuple(covs))


def _riesz_predict(est, test):
    Z = test[["A"] + covariates(test)]
    return pd.DataFrame({
        "alpha": est.predict(Z),
        "alpha1": est.predict(Z.assign(A=1.0)),
        "alpha0": est.predict(Z.assign(A=0.0)),
    })


def rieszboost(train, test, estimand, max_iter, max_depth, learning_rate):
    from rieszboost import RieszBooster
    est = RieszBooster(
        estimand=_estimand(estimand, covariates(train)), n_estimators=max_iter, learning_rate=learning_rate,
        max_depth=max_depth, reg_lambda=L2_PENALTY, early_stopping_rounds=PATIENCE,
        validation_fraction=VALID_FRAC, random_state=SEED,
    ).fit(train[["A"] + covariates(train)])
    return _riesz_predict(est, test), est.best_iteration_ + 1


def riesznet(train, test, estimand, max_iter, learning_rate, weight_decay):
    from riesznet import RieszNet
    est = RieszNet(
        estimand=_estimand(estimand, covariates(train)), hidden_sizes=(64, 64), activation="elu",
        learning_rate=learning_rate, weight_decay=weight_decay, epochs=max_iter,
        batch_size=64, early_stopping_rounds=PATIENCE, validation_fraction=VALID_FRAC,
        snapshot_epochs=[],
        random_state=SEED,
    ).fit(train[["A"] + covariates(train)])
    return _riesz_predict(est, test), est.best_iteration_ + 1


# ------------------------------------------------------------- registry ---
# name -> (function, fixed keyword arguments, tuning grid). The fixed
# arguments include max_iter, the cap on trees or epochs.
#
# The grids were moved after a 20-replicate check of the first grids
# (depth {2, 3, 4} x learning rate {0.01, 0.03, 0.1} for every tree learner;
# learning rate {1e-4, 1e-3, 1e-2} x weight decay {0, 1e-3, 1e-2} for
# riesznet), by the edge rule on median cross-validated regret:
#   * outcome and propensity chose depth 2 and learning rate 0.1, both edges.
#   * rieszboost for the ATT chose learning rate 0.01, an edge, at every n.
#     The smaller rate needs more trees, so its cap rises to 5000.
#   * riesznet at 1e-4 ran into its epoch cap; 3e-4 covers the same path.

#
# Depth was then set to {3, 5, 7} for every tree learner, and rieszboost's
# learning rates to {0.001, 0.003, 0.01, 0.03} with a 20000-tree cap, by the
# author's choice, while debugging the tree learners' performance.
# The L2 leaf penalty of every boosted learner was then set to 0 (from 1).
#
# On the redesigned five_confounders DGP (20 replicates per n), every tree
# learner chose the smallest depth (3) and smallest learning rate at every n,
# so depth became {1, 2, 3, 5} for all three, with learning rate
# {0.01, 0.03, 0.1} for outcome and propensity and {0.0003, 0.001, 0.003, 0.01}
# for rieszboost.

XGB_GRID = _grid(max_depth=[1, 2, 3, 5], learning_rate=[0.01, 0.03, 0.1])
RIESZBOOST_GRID = _grid(max_depth=[1, 2, 3, 5], learning_rate=[0.0003, 0.001, 0.003, 0.01])
RIESZNET_GRID = _grid(learning_rate=[3e-4, 1e-3, 3e-3, 1e-2], weight_decay=[0.0, 1e-3, 1e-2])

LEARNERS = {
    "outcome": (outcome_xgb, {"max_iter": 2000}, XGB_GRID),
    "propensity": (propensity_xgb, {"max_iter": 2000}, XGB_GRID),
    "rieszboost_ATE": (rieszboost, {"estimand": "ATE", "max_iter": 20000}, RIESZBOOST_GRID),
    "rieszboost_ATT": (rieszboost, {"estimand": "ATT", "max_iter": 20000}, RIESZBOOST_GRID),
    "riesznet_ATE": (riesznet, {"estimand": "ATE", "max_iter": 500}, RIESZNET_GRID),
    "riesznet_ATT": (riesznet, {"estimand": "ATT", "max_iter": 500}, RIESZNET_GRID),
}


# ------------------------------------------------ held-out risk per row ---

def riesz_loss(estimand, a, alpha, alpha1, alpha0):
    """Per-row squared Riesz loss alpha(Z)^2 - 2 m(alpha)(Z). Its mean equals
    E[(alpha - alpha0)^2] up to a constant that does not depend on alpha."""
    m = alpha1 - alpha0
    if estimand == "ATT":
        m = a * m
    return alpha**2 - 2 * m


def row_loss(learner, data: pd.DataFrame, preds: pd.DataFrame) -> np.ndarray:
    """Per-row held-out loss that cross-validation minimizes for each learner."""
    if learner == "outcome":
        return (data["Y"].to_numpy() - preds["mu"].to_numpy()) ** 2
    if learner == "propensity":
        p = np.clip(preds["pi"].to_numpy(), 1e-12, 1 - 1e-12)
        a = data["A"].to_numpy()
        return -(a * np.log(p) + (1 - a) * np.log(1 - p))
    estimand = learner.split("_")[1]
    return riesz_loss(estimand, data["A"].to_numpy(), *(preds[c].to_numpy()
                      for c in ("alpha", "alpha1", "alpha0")))
