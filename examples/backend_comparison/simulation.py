"""Runs the simulation, one cell at a time. A cell is one simulated dataset,
named by its DGP, sample size n, and replicate r.

For each cell:

  1. Draw n observations O = (Y, A, X) with seed 510 + r. The seed depends
     only on the cell's name, so a cell run alone gets the same data as it
     does in the full grid.
  2. Split the rows into 2 cross-fitting folds, with the same seed.
  3. Cross-fit the outcome regression. For each fold, tune and fit mu on the
     other fold, then predict mu(A, X), mu(1, X), and mu(0, X) on this fold.
     Both estimands share these predictions.
  4. Cross-fit the Riesz representer for each estimand and Riesz learner,
     over the same folds.
  5. Plug the predictions into each estimand's one-step estimator. Record the
     estimate, its standard error, and the errors of mu_hat and alpha_hat
     against the true mu and alpha_0.

Tuning happens inside each training fold. GridSearchCV with `cv_folds` folds
picks the hyperparameters, then refits on the whole training fold.

Caching. Steps 3 and 4 take nearly all the compute. Each cross-fit is saved
in cache/ (ignored by git) as its out-of-fold predictions plus the CV score
of every grid point. The cache key covers everything that determines the fit:
the data, the folds, cv_folds, the learner's full hyperparameters and grid,
the source of its classes, and the source of the rieszreg packages. A rerun,
a resume after a crash, or a new learner or estimator therefore refits only
what changed. Deleting cache/ changes the runtime and nothing else.

run_simulation() returns two tidy tables:
    results  one row per (dgp, n, rep, estimand, learner)
    tuning   one row per (dgp, n, rep, nuisance, estimand, learner, fold,
             grid point): the grid point, its CV score, and whether
             GridSearchCV selected it. Selected rows also carry the early-
             stopping iteration of the refit and the time the fold took.

Command line:
    python simulation.py --dgp easy --n 2000 --reps 20 --n_jobs 8
"""

import os

# One thread per process: the cells run in parallel, and torch (riesznet)
# and xgboost can crash on macOS when both use multithreaded OpenMP. Must be
# set before numpy, torch, or xgboost is imported.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
import hashlib
import inspect
import time
from functools import cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold

import rieszboost
import rieszreg
import riesznet
from dgps import DGPS
from estimands import ESTIMANDS
from learners import OUTCOME_LEARNER, RIESZ_LEARNERS, stopping_iteration

BASE_SEED = 510
N_CROSSFIT_FOLDS = 2
TRUTH_SEED = 12345
N_MC_TRUTH = 2_000_000
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"


@cache
def true_values(dgp_name):
    """{estimand: (psi_0, Monte Carlo SE of psi_0)} under one DGP."""
    return {name: e.true_value(DGPS[dgp_name], np.random.default_rng(TRUTH_SEED), N_MC_TRUTH)
            for name, e in ESTIMANDS.items()}


def draw_cell(dgp, n, rep):
    """Seed, data, and cross-fitting folds of one cell."""
    seed = BASE_SEED + rep
    Y, A, X = dgp.sample(n, np.random.default_rng(seed))
    folds = list(KFold(N_CROSSFIT_FOLDS, shuffle=True, random_state=seed).split(X))
    return seed, Y, A, X, folds


def crossfit(learner, estimand, seed, Z, y, folds, cv_folds, arms=False):
    """Out-of-fold predictions of one learner, tuned by GridSearchCV inside
    each training fold. Returns (predictions, tuning).

    predictions has a column `pred`, plus `pred1` and `pred0` (the prediction
    at a = 1 and a = 0) when arms=True. tuning has one row per fold and grid
    point. `y` is None for the Riesz learners, which fit on Z alone.
    """
    preds = {col: np.full(len(Z), np.nan) for col in (["pred", "pred1", "pred0"] if arms else ["pred"])}
    tuning = []
    for fold, (train, test) in enumerate(folds):
        start = time.time()
        search = GridSearchCV(learner.make(estimand, seed), learner.grid, cv=cv_folds)
        search.fit(Z.iloc[train], None if y is None else y[train])
        Z_test = Z.iloc[test]
        preds["pred"][test] = search.predict(Z_test)
        if arms:
            preds["pred1"][test] = search.predict(Z_test.assign(a=1.0))
            preds["pred0"][test] = search.predict(Z_test.assign(a=0.0))

        cv = search.cv_results_
        grid = pd.DataFrame([{k.removeprefix("estimator__"): v for k, v in p.items()} for p in cv["params"]])
        selected = np.arange(len(grid)) == search.best_index_
        iteration, cap = stopping_iteration(search.best_estimator_)
        tuning.append(grid.assign(
            fold=fold, cv_score=cv["mean_test_score"], selected=selected,
            best_iteration=np.where(selected, iteration, np.nan),
            iteration_cap=np.where(selected, cap, np.nan),
            seconds=np.where(selected, time.time() - start, np.nan),
        ))
    return pd.DataFrame(preds), pd.concat(tuning, ignore_index=True)


# ---------------------------------------------------------------- cache

def _digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(part.tobytes() if isinstance(part, np.ndarray) else repr(part).encode())
    return h.hexdigest()[:16]


def _package_source():
    """Source of the workspace packages plus versions of the third-party
    learners, so upgrading or editing either invalidates the cache."""
    files = sorted(f for pkg in (rieszreg, rieszboost, riesznet) for f in Path(pkg.__file__).parent.rglob("*.py"))
    return _digest(*(f.read_text() for f in files), *(version(p) for p in ("xgboost", "torch", "scikit-learn")))


PACKAGE_SOURCE = _package_source()


def learner_key(learner, estimand, seed):
    """Everything about a learner that determines its fit: all hyperparameters
    (including defaults and the seed), the grid, and the source of the
    wrapper classes in learners.py."""
    model = learner.make(estimand, seed)
    # Estimands enter through their factory spec (their repr holds a memory
    # address); nested estimators through their own estimator__* params.
    params = {k: getattr(v, "factory_spec", v) for k, v in model.get_params(deep=True).items()
              if not isinstance(v, BaseEstimator)}
    classes = {type(model), type(getattr(model, "estimator", model))}
    sources = sorted(inspect.getsource(c) for c in classes if c.__module__ == "learners")
    return repr(params), learner.grid, sources, inspect.getsource(crossfit)


def cached(label, key, compute):
    """Loads cache/<label>_<hash of key>.pkl, or computes and saves it. Writes
    to a temporary name first, so a killed run leaves no partial file."""
    path = CACHE_DIR / f"{label}_{_digest(*key)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    pd.to_pickle(value, tmp)
    os.replace(tmp, path)
    return value


# ---------------------------------------------------------------- run

def run_cell(dgp_name, n, rep, estimands, learners, cv_folds):
    """One cell. Returns (results rows, tuning table)."""
    dgp = DGPS[dgp_name]
    seed, Y, A, X, folds = draw_cell(dgp, n, rep)
    Z = dgp.to_frame(A, X)
    ids = dict(dgp=dgp_name, n=n, rep=rep)
    data_key = (Z.to_numpy(), Y, *(test for _, test in folds), cv_folds, PACKAGE_SOURCE)

    def fit(label, learner, estimand, y, arms=False):
        return cached(f"{dgp_name}_n{n}/rep{rep:03d}/{label}",
                      (*data_key, *learner_key(learner, estimand, seed)),
                      lambda: crossfit(learner, estimand, seed, Z, y, folds, cv_folds, arms))

    # Outcome regression, shared by every estimand and Riesz learner
    mu_preds, mu_tuning = fit("mu", OUTCOME_LEARNER, None, Y, arms=True)
    mu, mu1, mu0 = mu_preds.pred.to_numpy(), mu_preds.pred1.to_numpy(), mu_preds.pred0.to_numpy()
    mu_rmse = np.sqrt(np.mean((mu - dgp.outcome_regression(A, X)) ** 2))
    tuning = [mu_tuning.assign(**ids, nuisance="mu", estimand="shared", learner=OUTCOME_LEARNER.name)]

    # Riesz representer, per estimand and learner, then the one-step estimate
    results = []
    for estimand in (ESTIMANDS[name] for name in estimands):
        true_alpha = estimand.true_representer(dgp, A, X)
        for learner in (RIESZ_LEARNERS[name] for name in learners):
            row = dict(ids, estimand=estimand.name, learner=learner.name,
                       truth=true_values(dgp_name)[estimand.name][0], mu_rmse=mu_rmse)
            try:
                alpha_preds, alpha_tuning = fit(f"alpha_{estimand.name}_{learner.name}", learner,
                                                estimand.riesz_estimand(dgp.covariates), None)
            except Exception as err:   # record the failure and keep going; failures are counted
                print(f"{ids} {estimand.name} {learner.name} failed: {err!r}", flush=True)
                results.append(row)
                continue
            alpha = alpha_preds.pred.to_numpy()
            est, se = estimand.one_step(A, Y, mu, mu1, mu0, alpha)
            results.append(dict(
                row, est=est, se=se,
                alpha_rmse=np.sqrt(np.mean((alpha - true_alpha) ** 2)),
                alpha_mae=np.mean(np.abs(alpha - true_alpha)),
                fit_seconds=alpha_tuning.seconds.sum(),
            ))
            tuning.append(alpha_tuning.assign(**ids, nuisance="alpha", estimand=estimand.name,
                                              learner=learner.name))
    return results, pd.concat(tuning, ignore_index=True)


def run_simulation(dgps=("easy",), ns=(2000,), reps=range(20), estimands=tuple(ESTIMANDS),
                   learners=tuple(RIESZ_LEARNERS), cv_folds=5, n_jobs=1):
    """Runs every cell of dgps x ns x reps, for the requested estimands and
    Riesz learners, `n_jobs` cells at a time. Any subset returns exactly the
    rows the full grid would. Returns {"results": ..., "tuning": ...}."""
    cells = [(d, n, rep) for d in dgps for n in ns for rep in reps]
    jobs = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(run_cell)(*cell, estimands, learners, cv_folds) for cell in cells)
    results, tuning, start = [], [], time.time()
    for k, (cell_results, cell_tuning) in enumerate(jobs, 1):
        results += cell_results
        tuning.append(cell_tuning)
        print(f"{k}/{len(cells)} cells done ({(time.time() - start) / 60:.1f} min)", flush=True)
    order = ["dgp", "n", "rep", "estimand", "learner"]
    return {
        "results": pd.DataFrame(results).sort_values(order, ignore_index=True),
        "tuning": pd.concat(tuning, ignore_index=True).sort_values(order + ["fold"], ignore_index=True),
    }


def save(tables, out_dir):
    """Writes each table to out_dir/<name>.csv, the record every display is
    computed from."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dgp", choices=DGPS, default="easy")
    parser.add_argument("--n", type=int, default=2000, help="sample size of each dataset")
    parser.add_argument("--reps", type=int, default=20, help="replicates 0, ..., reps - 1")
    parser.add_argument("--cv_folds", type=int, default=5, help="GridSearchCV folds for tuning")
    parser.add_argument("--n_jobs", type=int, default=1, help="cells to run in parallel")
    args = parser.parse_args()

    from simulation import run_simulation, save   # workers must unpickle these from a module, not __main__
    tables = run_simulation([args.dgp], [args.n], range(args.reps), cv_folds=args.cv_folds, n_jobs=args.n_jobs)
    save(tables, RESULTS_DIR / f"{args.dgp}_n{args.n}")
    print(f"Wrote {RESULTS_DIR / f'{args.dgp}_n{args.n}'}")
