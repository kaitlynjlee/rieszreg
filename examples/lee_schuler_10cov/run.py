"""rieszboost, riesznet and the non-augmented ForestRiesz on the Lee & Schuler
10-confounder binary DGP (dgp.py), for the ATE and the ATT.

Protocol, as in the RieszBoost ICML manuscript (Section 3.3, Appendix C.2.3):
each replicate draws an estimation set of 500 and then an independent
training set of 500. Every learner is tuned by 5-fold cross-validation on the
training set, over the grid below, then refit on the whole training set at
the selected setting and evaluated on the estimation set. The outcome
regression is one tuned XGBoost model per replicate, shared by every
representer, and each representer goes into the one-step (DML) estimator.
The oracle plugs in the true representer, so its gap to a learner is the
cost of estimating alpha alone.

Each (replicate, learner) pair is one task and caches its estimation-set
predictions and the CV risk of every setting, so `summarize` assembles the
tables and adding a learner never refits the others.

  python run.py run --reps 0:2 --jobs 8      # a smoke test
  python run.py summarize                    # tables, tuning checks
"""

from __future__ import annotations

import _env  # noqa: F401  (threads and import paths; must precede numpy)

import argparse
import hashlib
import importlib.metadata as md
import inspect
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dgp

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("RIESZ_SIM_OUT", HERE))
CACHE = OUT_ROOT / "cache" / "lee_schuler_10cov"
RESULTS = OUT_ROOT / "results" / "lee_schuler_10cov"

# Same seed and draw order as examples/simulation/lee_schuler_icml.py, so the
# two studies see identical datasets and pair replicate by replicate.
BASE_SEED = 20260928
N, K = 500, 5
ESTIMANDS = ("ATE", "ATT")
SEED = 0              # every learner with internal randomness
VALID_FRAC = 0.2      # share of each training fold held out for early stopping


def _grid(**axes):
    return [dict(zip(axes, v)) for v in itertools.product(*axes.values())]


# ------------------------------------------------------------ settings ---
# The manuscript's settings (Appendix C.2.3). Boosted learners: patience 200,
# cap 20000 trees, subsample 0.9. RieszNet: 3 hidden layers of 200, ELU, Adam
# with weight decay 1e-3, patience 10, cap 1000 epochs. ForestRiesz is not in
# the manuscript: the published ForestRiesz settings (the package defaults),
# with the leaf size tuned.

BOOST = dict(max_iter=20000, patience=200, subsample=0.9)
NET = dict(max_iter=1000, patience=10, hidden_sizes=(200, 200, 200), weight_decay=1e-3, batch_size=64)

XGB_GRID = _grid(learning_rate=[1e-5, 1e-4, 1e-3, 1e-2], max_depth=[3, 5, 7])
RIESZNET_GRID = _grid(learning_rate=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
FOREST_GRID = _grid(min_samples_leaf=[2, 5, 10, 20, 50])


# ------------------------------------------------------------ learners ---
# Each takes (train, test, **setting) and returns (predictions on test as a
# DataFrame, iterations kept by early stopping, or None).

def _estimand(name):
    from rieszreg import ATE, ATT
    return {"ATE": ATE, "ATT": ATT}[name](treatment="A", covariates=tuple(dgp.COV))


def _alpha_preds(est, test):
    Z = test[["A"] + dgp.COV]
    return pd.DataFrame({"alpha": est.predict(Z), "alpha1": est.predict(Z.assign(A=1.0)),
                         "alpha0": est.predict(Z.assign(A=0.0))})


def outcome_xgb(train, test, learning_rate, max_depth):
    import xgboost as xgb
    valid = np.random.default_rng(SEED).random(len(train)) < VALID_FRAC
    cols = ["A"] + dgp.COV
    fit_part, val = train[~valid], train[valid]
    m = xgb.XGBRegressor(n_estimators=BOOST["max_iter"], learning_rate=learning_rate, max_depth=max_depth,
                         subsample=BOOST["subsample"], early_stopping_rounds=BOOST["patience"],
                         n_jobs=1, random_state=SEED)
    m.fit(fit_part[cols], fit_part["Y"], eval_set=[(val[cols], val["Y"])], verbose=False)
    Z = test[cols]
    preds = pd.DataFrame({"mu": m.predict(Z), "mu1": m.predict(Z.assign(A=1.0)), "mu0": m.predict(Z.assign(A=0.0))})
    return preds, m.best_iteration + 1


def rieszboost(estimand):
    def fit(train, test, learning_rate, max_depth):
        from rieszboost import RieszBooster
        b = RieszBooster(estimand=_estimand(estimand), n_estimators=BOOST["max_iter"], learning_rate=learning_rate,
                         max_depth=max_depth, reg_lambda=0.0, subsample=BOOST["subsample"],
                         early_stopping_rounds=BOOST["patience"], validation_fraction=VALID_FRAC,
                         random_state=SEED).fit(train[["A"] + dgp.COV])
        return _alpha_preds(b, test), b.best_iteration_ + 1
    return fit


def riesznet(estimand):
    def fit(train, test, learning_rate):
        from riesznet import RieszNet
        n = RieszNet(estimand=_estimand(estimand), hidden_sizes=NET["hidden_sizes"], activation="elu",
                     learning_rate=learning_rate, weight_decay=NET["weight_decay"], epochs=NET["max_iter"],
                     batch_size=NET["batch_size"], early_stopping_rounds=NET["patience"],
                     validation_fraction=VALID_FRAC, snapshot_epochs=[], random_state=SEED).fit(train[["A"] + dgp.COV])
        return _alpha_preds(n, test), n.best_iteration_ + 1
    return fit


def forestriesz(estimand):
    """Non-augmented ForestRiesz (Chernozhukov et al. 2022) on EconML's GRF."""
    def fit(train, test, min_samples_leaf):
        from forestriesz import ForestRieszRegressor
        f = ForestRieszRegressor(estimand=_estimand(estimand), min_samples_leaf=min_samples_leaf,
                                 n_jobs=1, random_state=SEED).fit(train[["A"] + dgp.COV])
        return _alpha_preds(f, test), None
    return fit


def row_loss(kind, data, preds):
    """Per-row held-out loss that CV minimizes: squared error for the outcome
    regression, and the squared Riesz loss alpha^2 - 2 m(alpha) otherwise."""
    if kind == "outcome":
        return (data["Y"].to_numpy() - preds["mu"].to_numpy()) ** 2
    m = preds["alpha1"].to_numpy() - preds["alpha0"].to_numpy()
    if kind == "ATT":
        m = data["A"].to_numpy() * m
    return preds["alpha"].to_numpy() ** 2 - 2 * m


# name -> (fit function, grid, loss kind, iteration cap). Slowest first, so a
# parallel run starts the long tasks early.
COMPONENTS = {
    **{f"rieszboost_{e}": (rieszboost(e), XGB_GRID, e, BOOST["max_iter"]) for e in ESTIMANDS},
    "outcome": (outcome_xgb, XGB_GRID, "outcome", BOOST["max_iter"]),
    **{f"riesznet_{e}": (riesznet(e), RIESZNET_GRID, e, NET["max_iter"]) for e in ESTIMANDS},
    **{f"forestriesz_{e}": (forestriesz(e), FOREST_GRID, e, None) for e in ESTIMANDS},
}
LEARNERS = ("rieszboost", "riesznet", "forestriesz")


# ----------------------------------------------------------- one task ---

def data(rep):
    """Estimation set, then training set."""
    rng = np.random.default_rng([BASE_SEED, rep])
    return dgp.draw(N, rng), dgp.draw(N, rng)


def cv_select(fit, grid, kind, train, test):
    """K-fold CV over `grid` by mean held-out loss; refit the best setting on
    all of `train` and predict `test`."""
    folds = np.arange(len(train)) * K // len(train)     # rows are iid, so contiguous folds
    risk, iters = {}, {}
    for s in grid:
        losses, its = [], []
        for k in range(K):
            val = train[folds == k].reset_index(drop=True)
            p, it = fit(train[folds != k], val, **s)
            losses.append(row_loss(kind, val, p))
            its.append(it)
        risk[str(s)], iters[str(s)] = float(np.concatenate(losses).mean()), its
    best = grid[[str(s) for s in grid].index(min(risk, key=risk.get))]
    preds, it = fit(train, test, **best)
    return preds, {"best": best, "cv_risk": risk, "cv_iters": iters, "iters": it}


def _versions():
    out = {}
    for p in ("numpy", "pandas", "scikit-learn", "xgboost", "torch", "econml"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def cache_key(comp):
    """Everything that determines a component's fit except the replicate: its
    learner source and settings, the data and CV code, the package versions,
    and the source of the rieszreg packages it uses."""
    fit, grid, kind, cap = COMPONENTS[comp]
    stem = comp.split("_")[0]
    parts = [inspect.getsource(globals()[stem if stem != "outcome" else "outcome_xgb"]),
             repr((grid, kind, cap, BOOST, NET, BASE_SEED, N, K, SEED, VALID_FRAC)),
             inspect.getsource(cv_select), inspect.getsource(row_loss), inspect.getsource(data),
             inspect.getsource(dgp), repr(sorted(_versions().items()))]
    pkgs = {"rieszboost": ("rieszreg", "rieszboost"), "riesznet": ("rieszreg", "riesznet"),
            "forestriesz": ("rieszreg", "forestriesz")}.get(stem, ())
    for pkg in pkgs:
        parts += [p.read_text() for p in sorted((_env.REPO / "packages" / pkg / "python" / pkg).rglob("*.py"))]
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]


def cache_path(comp, rep, key):
    return CACHE / comp / f"rep{rep}_{key}.pkl"


def run_task(rep, comp, key):
    path = cache_path(comp, rep, key)
    if path.exists():
        return
    fit, grid, kind, _ = COMPONENTS[comp]
    est, train = data(rep)
    t0 = time.perf_counter()
    try:
        preds, record = cv_select(fit, grid, kind, train, est)
        out = {"preds": {c: preds[c].to_numpy() for c in preds}, "record": record}
    except Exception as err:          # a failed fit is recorded as missing, and the run continues
        out = {"preds": None, "record": {"error": repr(err)}}
    out["seconds"] = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.to_pickle(out, tmp)
    tmp.replace(path)
    print(f"done rep={rep} {comp} {out['seconds']:.0f}s{' FAILED' if out['preds'] is None else ''}", flush=True)


def run(reps, comps, jobs):
    keys = {c: cache_key(c) for c in comps}
    tasks = [(r, c, keys[c]) for c in comps for r in reps if not cache_path(c, r, keys[c]).exists()]
    print(f"{len(tasks)} tasks to run", flush=True)
    if jobs == 1:
        for t in tasks:
            run_task(*t)
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=jobs, batch_size=1)(delayed(run_task)(*t) for t in tasks)


# ------------------------------------------------------------- summary ---

# The manuscript's Tables 1 and 2 (1000 replicates), for reference. Its
# RieszNet ATE row has MAE above RMSE, which averages of per-replicate values
# cannot do, so one of those two numbers is a typo.
PAPER = {
    ("ATE", "rieszboost"): dict(alpha_rmse=0.902, alpha_mae=0.534, pct_bias=0.421, rmse=0.238, coverage=0.941),
    ("ATE", "riesznet"): dict(alpha_rmse=0.989, alpha_mae=2.489, pct_bias=-0.854, rmse=0.253, coverage=0.905),
    ("ATT", "rieszboost"): dict(alpha_rmse=0.675, alpha_mae=0.303, pct_bias=0.582, rmse=0.259, coverage=0.936),
    ("ATT", "riesznet"): dict(alpha_rmse=3.194, alpha_mae=1.610, pct_bias=0.253, rmse=0.283, coverage=0.952),
}


def load(comp):
    key = cache_key(comp)
    return {int(f.name[3:].split("_")[0]): pd.read_pickle(f) for f in (CACHE / comp).glob(f"rep*_{key}.pkl")}


def one_step(estimand, d, mu, alpha):
    a, y = d["A"].to_numpy(), d["Y"].to_numpy()
    tau, resid = mu["mu1"] - mu["mu0"], y - mu["mu"]
    if estimand == "ATE":
        phi = tau + alpha * resid
        est = phi.mean()
    else:
        p = a.mean()
        est = np.mean(a * tau + alpha * resid) / p
        phi = (a * (tau - est) + alpha * resid) / p
    return float(est), float(phi.std(ddof=1) / np.sqrt(len(y)))


def estimates():
    """One row per (replicate, estimand, method)."""
    mu = load("outcome")
    fits = {c: load(c) for c in COMPONENTS if c != "outcome"}
    rows = []
    for rep in sorted(mu):
        if mu[rep]["preds"] is None:
            continue
        d, _ = data(rep)
        X, A = d[dgp.COV].to_numpy(), d["A"].to_numpy()
        for e in ESTIMANDS:
            a0 = dgp.true_alpha(e, A, X)
            for method in ("oracle",) + LEARNERS:
                if method == "oracle":
                    alpha = a0
                else:
                    got = fits[f"{method}_{e}"].get(rep)
                    if got is None:
                        continue
                    if got["preds"] is None:
                        rows.append(dict(rep=rep, estimand=e, method=method, failed=True))
                        continue
                    alpha = got["preds"]["alpha"]
                est, se = one_step(e, d, mu[rep]["preds"], alpha)
                rows.append(dict(rep=rep, estimand=e, method=method, failed=False, est=est, se=se,
                                 alpha_rmse=float(np.sqrt(np.mean((alpha - a0) ** 2))),
                                 alpha_mae=float(np.mean(np.abs(alpha - a0)))))
    return pd.DataFrame(rows)


def table(df):
    """Performance per (estimand, method), each measure with its Monte Carlo SE."""
    rows = []
    for (e, m), g in df.groupby(["estimand", "method"], sort=False):
        ok = g[~g.failed]
        R = len(ok)
        err = ok.est.to_numpy() - dgp.PSI[e]
        rmse = np.sqrt(np.mean(err ** 2))
        cov = np.mean(np.abs(err) <= 1.96 * ok.se.to_numpy())
        rows.append(dict(
            estimand=e, method=m, reps=R, failed=int(g.failed.sum()),
            alpha_rmse=ok.alpha_rmse.mean(), alpha_rmse_mcse=ok.alpha_rmse.std(ddof=1) / np.sqrt(R),
            alpha_mae=ok.alpha_mae.mean(), alpha_mae_mcse=ok.alpha_mae.std(ddof=1) / np.sqrt(R),
            pct_bias=100 * err.mean() / dgp.PSI[e], pct_bias_mcse=100 * err.std(ddof=1) / np.sqrt(R) / dgp.PSI[e],
            rmse=rmse, rmse_mcse=np.std(err ** 2, ddof=1) / np.sqrt(R) / (2 * rmse),
            emp_se=err.std(ddof=1), mean_se=ok.se.mean(),
            coverage=cov, coverage_mcse=np.sqrt(cov * (1 - cov) / R)))
    return pd.DataFrame(rows)


def tuning():
    """Per component, the settings CV selected, the share of selections on the
    low or high edge of each tuned axis, and the share of refits that reached
    the iteration cap. A consistent edge means the grid should move."""
    lines = []
    for comp, (_, grid, _, cap) in COMPONENTS.items():
        recs = [g["record"] for g in load(comp).values() if g["preds"] is not None]
        if not recs:
            continue
        lines.append(f"\n{comp}: {len(recs)} replicates")
        lines.append(pd.Series([str(r["best"]) for r in recs]).value_counts(normalize=True).round(2).to_string())
        for axis in grid[0]:
            vals = sorted({s[axis] for s in grid})
            if len(vals) > 1:
                chosen = np.array([r["best"][axis] for r in recs])
                lines.append(f"  {axis}: at lowest {np.mean(chosen == vals[0]):.2f}, at highest {np.mean(chosen == vals[-1]):.2f}")
        if cap:
            its = np.array([r["iters"] for r in recs])
            lines.append(f"  refit iterations: median {int(np.median(its))}, share at cap {np.mean(its >= cap):.2f}")
    return "\n".join(lines)


def summarize():
    df = estimates()
    if df.empty:
        print("no results yet")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "estimates.csv", index=False)
    t = table(df)
    t.to_csv(RESULTS / "table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("Representer accuracy (manuscript Table 1) and one-step estimator (Table 2), with MCSEs:")
    print(t.round(3).to_string(index=False))
    paper = pd.DataFrame([dict(estimand=e, method=m, **v) for (e, m), v in PAPER.items()])
    print("\nThe manuscript, for reference:")
    print(paper.to_string(index=False))
    print(tuning())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--reps", default="0:2", help="half-open range: 0:100 runs replicates 0 to 99")
    r.add_argument("--components", nargs="*", default=None, help=f"default: all of {list(COMPONENTS)}")
    r.add_argument("--jobs", type=int, default=1)
    sub.add_parser("summarize")
    a = ap.parse_args()
    if a.cmd == "run":
        lo, hi = (int(v) for v in a.reps.split(":"))
        run(range(lo, hi), a.components or list(COMPONENTS), a.jobs)
    else:
        summarize()
