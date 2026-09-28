"""Are the boosted learners underfitting? Learning curves against the truth.

For one training fold (80% of an n = 2000 dataset, as in cross-fitting), fit
each boosted learner with no early stopping, then read off, at every tree
count, (a) the loss on the 20% validation split that early stopping watches
and (b) the error against the true function on a large independent test draw.
Early stopping with patience 20 is replayed on (a). Comparing the error at
the replayed stopping point with the smallest error along the path shows
whether early stopping stops too soon, and comparing across depths shows
whether shallow trees are chosen because deeper ones are worse or because
they stop before they get good.

  ../../.venv/bin/python diagnose_boosting.py --reps 3
"""

from __future__ import annotations

import _env  # noqa: F401

import argparse

import numpy as np
import pandas as pd

import dgp
import learners as L
import run

G = dgp.DGPS["shared_terms"]
STEP = 5                          # path resolution, in trees
PATIENCE_STEPS = L.PATIENCE // STEP


def _replay_early_stopping(valid_loss: np.ndarray) -> int:
    """Index along the path where patience-20 early stopping would keep."""
    best, best_i, waited = np.inf, 0, 0
    for i, v in enumerate(valid_loss):
        if v < best - 1e-12:
            best, best_i, waited = v, i, 0
        else:
            waited += 1
            if waited >= PATIENCE_STEPS:
                break
    return best_i


def _curves(label, grid, valid_loss, test_err):
    es = _replay_early_stopping(valid_loss)
    best = int(np.argmin(test_err))
    return dict(learner=label, trees_at_es=grid[es], err_at_es=test_err[es],
                trees_at_best=grid[best], err_best=test_err[best],
                err_at_cap=test_err[-1], ran_to_cap=es == len(grid) - 1)


def riesz_curves(fit, valid, test, estimand, depth, lr, n_trees):
    from rieszboost import RieszBooster
    cols = ["A"] + list(G.covariates)
    est = RieszBooster(estimand=L._estimand(estimand, G.covariates), n_estimators=n_trees, learning_rate=lr,
                       max_depth=depth, random_state=L.SEED).fit(fit[cols])
    grid = list(range(STEP, n_trees + 1, STEP))

    def path(df):
        Z = df[cols]
        return (est.predict_path(Z, grid), est.predict_path(Z.assign(A=1.0), grid),
                est.predict_path(Z.assign(A=0.0), grid))

    al, al1, al0 = path(valid)
    a_v = valid["A"].to_numpy()[:, None]
    m = al1 - al0 if estimand == "ATE" else a_v * (al1 - al0)
    valid_loss = (al**2 - 2 * m).mean(0)
    at = est.predict_path(test[cols], grid)
    a0 = G.true_alpha(estimand, test["A"].to_numpy(), test[list(G.covariates)].to_numpy())
    test_err = np.sqrt(((at - a0[:, None]) ** 2).mean(0))
    return grid, valid_loss, test_err


def xgb_curves(fit, valid, test, which, depth, lr, n_trees):
    import xgboost as xgb
    if which == "outcome":
        cols, target = ["A"] + list(G.covariates), "Y"
        model = xgb.XGBRegressor(n_estimators=n_trees, learning_rate=lr, max_depth=depth,
                                 n_jobs=1, random_state=L.SEED).fit(fit[cols], fit[target])
        pred = lambda df, k: model.predict(df[cols], iteration_range=(0, k))
        truth = G.mu(test["A"].to_numpy(), test[list(G.covariates)].to_numpy())
        loss = lambda y, p: (y - p) ** 2
    else:
        cols, target = list(G.covariates), "A"
        model = xgb.XGBClassifier(n_estimators=n_trees, learning_rate=lr, max_depth=depth,
                                  n_jobs=1, random_state=L.SEED).fit(fit[cols], fit[target])
        pred = lambda df, k: model.predict_proba(df[cols], iteration_range=(0, k))[:, 1]
        truth = G.propensity(test[list(G.covariates)].to_numpy())
        loss = lambda y, p: -(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
    grid = list(range(STEP, n_trees + 1, STEP))
    valid_loss = np.array([loss(valid[target].to_numpy(), pred(valid, k)).mean() for k in grid])
    test_err = np.array([np.sqrt(((pred(test, k) - truth) ** 2).mean()) for k in grid])
    return grid, valid_loss, test_err


def diagnose(rep: int, n: int = 2000, n_test: int = 20_000) -> pd.DataFrame:
    d, folds = run.cell_data("shared_terms", n, rep)
    train = d[folds != 0].reset_index(drop=True)
    rng = np.random.default_rng(L.SEED)
    is_valid = rng.random(len(train)) < L.VALID_FRAC
    fit, valid = train[~is_valid], train[is_valid]
    test = G.draw(n_test, np.random.default_rng([99, rep]))
    rows = []
    for depth in (1, 2, 3, 4, 6):
        for lr in (0.03, 0.1):
            for estimand in ("ATE", "ATT"):
                n_trees = 3000 if lr == 0.03 else 1500
                g, v, e = riesz_curves(fit, valid, test, estimand, depth, lr, n_trees)
                rows.append(dict(_curves(f"rieszboost_{estimand}", g, v, e), depth=depth, lr=lr))
            for which in ("outcome", "propensity"):
                g, v, e = xgb_curves(fit, valid, test, which, depth, lr, 1500)
                rows.append(dict(_curves(which, g, v, e), depth=depth, lr=lr))
    return pd.DataFrame(rows).assign(rep=rep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    from joblib import Parallel, delayed
    out = pd.concat(Parallel(n_jobs=min(args.reps, 9))(delayed(diagnose)(r) for r in range(args.reps)))
    out.to_csv(run.HERE / "results" / "diagnose_boosting.csv", index=False)
    pd.set_option("display.width", 250)
    summ = out.groupby(["learner", "depth", "lr"]).agg(
        err_at_es=("err_at_es", "mean"), err_best=("err_best", "mean"), err_at_cap=("err_at_cap", "mean"),
        trees_at_es=("trees_at_es", "median"), trees_at_best=("trees_at_best", "median"),
        ran_to_cap=("ran_to_cap", "mean"))
    print(summ.round(3).to_string())
