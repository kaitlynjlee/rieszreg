"""Learners for the two nuisance functions, and their tuning grids.

Each learner is a `Learner`: a name, a function returning the unfitted
estimator, and the grid GridSearchCV tunes over. Everything else is fixed.

Outcome regression mu(A, X), shared by every Riesz learner:
    OUTCOME_LEARNER  XGBoost regression.
        fixed  up to 10,000 trees; early stopping with patience 20 on a
               held-out 20% of the training rows
        tuned  max_depth, learning_rate, reg_lambda (OUTCOME_LEARNER.grid)

Riesz representer alpha(A, X), the learners being compared:
    xgboost   rieszboost with XGBoost trees.
        fixed  up to 20,000 trees; early stopping with patience 20 on a
               held-out 20% of individuals; each round fits on a random 80%
               of individuals (subsample)
        tuned  max_depth, learning_rate
    riesznet  riesznet, set up as in Lee & Schuler.
        fixed  3 hidden layers of width 200; ELU activations; AdamW with
               weight decay 1e-3; up to 1,000 epochs with early stopping
               (patience 10) on a held-out 20%; keeps the better of 2 random
               initializations (lower validation loss)
        tuned  learning_rate

Every learner uses one thread and a fixed seed, so a fit depends only on its
data. The replicates already run in parallel.
"""

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable

import torch
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from rieszboost import RieszBooster
from riesznet import RieszNet


@dataclass
class Learner:
    """`make(estimand, seed)` returns an unfitted sklearn estimator, which
    GridSearchCV tunes over `grid`. The outcome learner ignores `estimand`."""

    name: str
    make: Callable
    grid: dict


# ---------------------------------------------------------------- outcome regression

class XGBRegressorES(XGBRegressor):
    """XGBRegressor that holds out `validation_fraction` of its training rows
    and stops adding trees once the held-out error stops improving."""

    def fit(self, Z, y, validation_fraction=0.2):
        Z_train, Z_valid, y_train, y_valid = train_test_split(
            Z, y, test_size=validation_fraction, random_state=self.random_state)
        return super().fit(Z_train, y_train, eval_set=[(Z_valid, y_valid)], verbose=False)


OUTCOME_LEARNER = Learner(
    name="xgboost_regression",
    make=lambda estimand, seed: XGBRegressorES(
        n_estimators=10_000, early_stopping_rounds=20, n_jobs=1, random_state=seed),
    grid={"max_depth": [2, 3, 4], "learning_rate": [0.01, 0.001], "reg_lambda": [0.1, 1.0]},
)


# ---------------------------------------------------------------- Riesz representer

class RieszNetAdamW(RieszNet):
    """RieszNet trained with AdamW (decoupled weight decay) instead of Adam."""

    def _resolved_backend(self):
        return replace(super()._resolved_backend(), optimizer_factory=partial(
            torch.optim.AdamW, lr=self.learning_rate, weight_decay=self.weight_decay))


class BestOfRestarts(BaseEstimator):
    """Fits `estimator` from `n_restarts` random initializations and keeps the
    fit with the lowest validation loss."""

    def __init__(self, estimator, n_restarts=2):
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


RIESZ_LEARNERS = {learner.name: learner for learner in [
    Learner(
        name="xgboost",
        make=lambda estimand, seed: RieszBooster(
            estimand=estimand, random_state=seed, n_estimators=20_000, subsample=0.8,
            early_stopping_rounds=20, validation_fraction=0.2),
        grid={"max_depth": [3, 5, 7], "learning_rate": [0.0001, 0.001, 0.01]},
    ),
    Learner(
        name="riesznet",
        make=lambda estimand, seed: BestOfRestarts(RieszNetAdamW(
            estimand=estimand, random_state=seed, hidden_sizes=(200, 200, 200), activation="elu",
            weight_decay=1e-3, epochs=1_000, early_stopping_rounds=10, validation_fraction=0.2)),
        grid={"estimator__learning_rate": [0.00001, 0.0001, 0.001, 0.01]},
    ),
]}


# ---------------------------------------------------------------- early stopping

def stopping_iteration(model):
    """(iteration early stopping kept, most iterations allowed) for a fitted
    model: trees for the XGBoost learners, epochs for riesznet."""
    model = getattr(model, "best_", model)   # BestOfRestarts: the restart it kept
    if isinstance(model, XGBRegressor):
        return model.best_iteration, model.n_estimators
    return model.best_iteration_, model.epochs if isinstance(model, RieszNet) else model.n_estimators
