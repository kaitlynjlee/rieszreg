"""Replication of the RieszBoost ICML manuscript, Section 3.3: binary
treatment, 10 confounders, the ATE and the ATT (Tables 1 and 2).

DGP (Appendix C.2.1, as coded in icml_rieszboost/rrboost/dgp_many_cov.py):
    X1..X10 ~ Uniform(0, 1)
    A | X ~ Bernoulli(expit(-0.7 X1 - 0.8 (X2 + X6)^2 - 0.8 X3 X4
                             + 0.6 sin(4 pi X5) + 0.6 X7 - 0.7 log(X8 + X9 + 2 X10) + 1.9))
    Y | A, X ~ N(5A - 4X1 + 3(X2 + X3)(2A - 1) + 8 X5 A + 0.5 (2X4 - X6)^2
                 + 8 expit(0.8(2A - 1) + 0.5 X7 - X8) + 4 cos(pi A (X9 + X10)), 1)
    (The manuscript prints "0.6 sin(45)"; the code has 0.6 sin(4 pi X5).)

Protocol, as in the manuscript: each replicate draws an estimation set of 500
and then an independent training set of 500. Every learner is tuned by 5-fold
CV on the training set and refit on all of it; the DML estimator is evaluated
on the estimation set. ATE and ATT share each replicate's data and nuisances.

Two arms, on the same data:
  * as coded: the authors' code (icml_rieszboost), unmodified, for the
    outcome regression, the propensity score (tuned on accuracy, GridSearchCV's
    default), rieszboost_ref and riesznet_ref.
  * our packages: rieszboost and riesznet under the protocol the manuscript
    describes, rieszboost_ascoded (ours with the settings the authors' code
    actually ran; see below), the non-augmented ForestRiesz, and a propensity
    score tuned on log loss.

What the authors' code runs, where it differs from the manuscript:
  * ATE RieszBoost is always depth 3: the tree learner is built in __init__
    and sklearn's set_params never rebuilds it, so the depth grid is inert.
  * ATT RieszBoost runs with patience 10 and subsample 0.5, not 200 and 0.9:
    its get_params drops those arguments, so GridSearchCV's clones reset them.
  * The propensity score is tuned on accuracy; the outcome regression on R^2.
  * RieszNet: optax "adam", so the stated weight decay is not applied; early
    stopping keeps the last epoch's weights; the learning rate is the one
    with the single best fold score, not the best mean over folds.
  * The global numpy RNG is never seeded, so their tables cannot be matched
    number for number. Here each (replicate, component) seeds it.

Every component is one task per replicate. It caches its predictions on the
estimation set (and its CV risks), so `summarize` assembles the estimates
and adding a method never refits the others.

  uv run --with equinox==0.13.8 --with optax==0.2.6 --with polars --with rich \\
         --with chex --with jaxtyping --with tqdm \\
      python lee_schuler_icml.py run --reps 0:2 --jobs 8
  python lee_schuler_icml.py summarize
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("DISABLE_TQDM", "1")

import _env  # noqa: F401,E402  (threads and import paths; must precede numpy)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import learners as L  # noqa: E402

REF = Path(os.environ.get("ICML_RIESZBOOST",
                          Path.home() / "Documents/research/RieszBoost/icml_rieszboost")).expanduser()
sys.path.insert(0, str(REF))
from rrboost.dgp_many_cov import ATE as _G  # noqa: E402
from rrboost import estimators as _E  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("RIESZ_SIM_OUT", HERE))
CACHE = OUT_ROOT / "cache" / "lee_schuler_icml"
RESULTS = OUT_ROOT / "results" / "lee_schuler_icml"

BASE_SEED = 20260928
N_DATA, K = 500, 5
COV = [f"X{j}" for j in range(1, 11)]
ESTIMANDS = ("ATE", "ATT")

# True values: 10 draws of 2e6 covariates each (MCSE 0.001), from `truth`.
PSI = {"ATE": 12.3146, "ATT": 11.8353}
PSI_PAPER = {"ATE": 12.315, "ATT": 11.843}

# The authors' settings (sim_files/ate_many_cov.py and att_many_cov.py).
VAL_FRAC, N_EST, ESR, SAMPLE_PROP = 0.2, 20000, 200, 0.9
XGB_PARAMS = {"learning_rate": [1e-5, 1e-4, 1e-3, 1e-2], "max_depth": [3, 5, 7]}
NN_LR = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
NUM_EPOCHS, NN_PATIENCE, NUM_RESTARTS = 1000, 10, 2

# Our ForestRiesz: the published ForestRiesz settings (the package defaults),
# with the leaf size tuned by CV Riesz loss.
FOREST_GRID = L._grid(min_samples_leaf=[2, 5, 10, 20, 50])


# ------------------------------------------------------------------ data ---

def draw(n, rng):
    X = rng.uniform(0, 1, (n, 10))
    A = rng.binomial(1, _G.expected_trt(X)).astype(float)
    Y = _G.expected_outcome(A, X) + rng.normal(0, 1, n)
    return Y, A, X


def data(rep):
    """Estimation set, then training set, as in the authors' run_rep."""
    rng = np.random.default_rng([BASE_SEED, rep])
    return draw(N_DATA, rng), draw(N_DATA, rng)


def frame(Y, A, X):
    return pd.DataFrame(X, columns=COV).assign(A=A, Y=Y)


def pi0(X):
    return _G.expected_trt(X)


def alpha_from_pi(estimand, a, p):
    return a / p - (1 - a) / (1 - p) if estimand == "ATE" else a - (1 - a) * p / (1 - p)


def truth(n_draws=10, n=2_000_000, seed=1):
    rng = np.random.default_rng(seed)
    out = {"ATE": [], "ATT": []}
    for _ in range(n_draws):
        X = rng.uniform(0, 1, (n, 10))
        p, tau = pi0(X), _G.expected_outcome(1, X) - _G.expected_outcome(0, X)
        out["ATE"].append(tau.mean())
        out["ATT"].append((p * tau).mean() / p.mean())
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(n_draws))) for k, v in out.items()}


# ------------------------------------------------ as coded (their code) ---
# Each returns (predictions on the estimation set, tuning record).

def _Z(A, X):
    from rrboost.utils import Z_concat
    return Z_concat(A, X)


def _gscv_record(gs, extra=None):
    rec = {"best": gs.best_params_,
           "cv": {str(p): float(s) for p, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"])}}
    rec.update(extra or {})
    return rec


def c_outcome(est, tr):
    from sklearn.model_selection import GridSearchCV
    from rrboost.xgboost_cv import XGBRegressor_ES
    (Y, A, X), (Yt, At, Xt) = est, tr
    gs = GridSearchCV(XGBRegressor_ES(nthread=1, validation_fraction=VAL_FRAC, n_estimators=N_EST,
                                      early_stopping_rounds=ESR, subsample=SAMPLE_PROP),
                      XGB_PARAMS, n_jobs=1).fit(_Z(At, Xt), Yt)
    m = gs.best_estimator_
    preds = {"mu": m.predict(_Z(A, X)), "mu1": m.predict(_Z(1, X)), "mu0": m.predict(_Z(0, X))}
    return preds, _gscv_record(gs, {"iters": int(m.best_iteration) + 1})


def _propensity(est, tr, scoring):
    from sklearn.model_selection import GridSearchCV
    from rrboost.xgboost_cv import XGBClassifier_ES
    (Y, A, X), (Yt, At, Xt) = est, tr
    gs = GridSearchCV(XGBClassifier_ES(nthread=1, validation_fraction=VAL_FRAC, n_estimators=N_EST,
                                       early_stopping_rounds=ESR, subsample=SAMPLE_PROP),
                      XGB_PARAMS, n_jobs=1, scoring=scoring).fit(Xt, At)
    m = gs.best_estimator_
    return {"pi": m.predict_proba(X)[:, 1]}, _gscv_record(gs, {"iters": int(m.best_iteration) + 1})


def c_propensity(est, tr):
    """As coded: GridSearchCV's default score for a classifier, accuracy."""
    return _propensity(est, tr, None)


def c_propensity_ll(est, tr):
    """The same learner and grid, tuned on log loss."""
    return _propensity(est, tr, "neg_log_loss")


def c_rieszboost_ref(estimand, est, tr):
    from sklearn.model_selection import GridSearchCV
    from rrboost import boosters
    (Y, A, X), (Yt, At, Xt) = est, tr
    B = getattr(boosters, f"{estimand}_ES_stochastic")
    gs = GridSearchCV(B(validation_fraction=VAL_FRAC, n_estimators=N_EST, early_stopping_rounds=ESR,
                        early_stopping=True, sample_prop=SAMPLE_PROP),
                      XGB_PARAMS, n_jobs=1).fit(_Z(At, Xt), Yt)
    b = gs.best_estimator_
    return {"alpha": b.predict(_Z(A, X))}, _gscv_record(gs, {
        "iters": len(b.learners), "tree_depth_used": b.learner.max_depth,
        "patience_used": b.early_stopping_rounds, "sample_prop_used": b.sample_prop})


def c_riesznet_ref(estimand, est, tr, rep):
    import jax
    from jax import random as jr
    from nn_estimators import estimands as nne
    from nn_estimators.estimators import RieszNet
    from nn_estimators.model_selection import GridSearchCV_lr
    (Y, A, X), (Yt, At, Xt) = est, tr
    import contextlib
    import io
    key = jr.PRNGKey(510 + 1120 * rep)       # the authors' key schedule
    E = nne.AverageTreatmentEffect if estimand == "ATE" else nne.AverageTreatmentAmongTreated
    printed = io.StringIO()                  # GridSearchCV_lr only prints the rate it picks
    with contextlib.redirect_stdout(printed):
        net = GridSearchCV_lr(estimator=RieszNet(estimand=E, covariate_dim=10, treatment_dim=1,
                                                 activation=jax.nn.elu, key=key),
                              x=_Z(At, Xt), y=Yt, key=key, num_epochs=NUM_EPOCHS, patience=NN_PATIENCE,
                              num_restarts=NUM_RESTARTS, n_jobs=1, learning_rate=np.array(NN_LR),
                              show_log=False)
    lr = [ln.split(":")[1].strip() for ln in printed.getvalue().splitlines() if ln.startswith("Best learning rate")]
    return {"alpha": np.asarray(net.predict(_Z(A, X))).ravel()}, {"best": {"learning_rate": float(lr[-1])} if lr else None}


# ------------------------------------------------------ our packages ---

def _cv_select(fit_fn, grid, estimand, train, test):
    """5-fold CV (KFold, unshuffled, as GridSearchCV's default) over `grid`
    by mean held-out Riesz loss; refit the best setting on all of train and
    predict alpha on test. fit_fn(train, test, **setting) -> (preds, iters)."""
    folds = np.arange(len(train)) * K // len(train)
    cv, iters = {}, {}
    for s in grid:
        losses, its = [], []
        for k in range(K):
            p, it = fit_fn(train[folds != k], train[folds == k].reset_index(drop=True), **s)
            v = train[folds == k]
            losses.append(L.riesz_loss(estimand, v["A"].to_numpy(),
                                       *(p[c].to_numpy() for c in ("alpha", "alpha1", "alpha0"))).mean())
            its.append(it)
        cv[str(s)], iters[str(s)] = float(np.mean(losses)), its
    best = min(cv, key=cv.get)
    s = grid[list(map(str, grid)).index(best)]
    p, it = fit_fn(train, test, **s)
    return {"alpha": p["alpha"].to_numpy()}, {"best": s, "cv": cv, "cv_iters": iters, "iters": it}


def _rieszboost(estimand, patience, subsample):
    def fit(train, test, learning_rate, max_depth):
        from rieszboost import RieszBooster
        b = RieszBooster(estimand=L._estimand(estimand, COV), n_estimators=N_EST, learning_rate=learning_rate,
                         max_depth=max_depth, reg_lambda=0.0, subsample=subsample,
                         early_stopping_rounds=patience, validation_fraction=VAL_FRAC,
                         random_state=L.SEED).fit(train[["A"] + COV])
        return L._riesz_predict(b, test), b.best_iteration_ + 1
    return fit


def c_rieszboost(estimand, est, tr):
    """Ours, as the manuscript describes: lr x depth grid, patience 200,
    subsample 0.9, cap 20000 trees, no L2 on leaves (as in sklearn trees)."""
    return _cv_select(_rieszboost(estimand, ESR, SAMPLE_PROP), L._grid(**XGB_PARAMS), estimand,
                      frame(*tr), frame(*est))


def c_rieszboost_ascoded(estimand, est, tr):
    """Ours, with the settings the authors' code actually ran."""
    if estimand == "ATE":
        grid, fit = L._grid(learning_rate=XGB_PARAMS["learning_rate"], max_depth=[3]), _rieszboost("ATE", ESR, SAMPLE_PROP)
    else:
        grid, fit = L._grid(**XGB_PARAMS), _rieszboost("ATT", 10, 0.5)
    return _cv_select(fit, grid, estimand, frame(*tr), frame(*est))


def c_riesznet(estimand, est, tr):
    """Ours (PyTorch), with the manuscript's architecture: 3 hidden layers of
    200, ELU, Adam with weight decay 1e-3, patience 10, cap 1000 epochs."""
    def fit(train, test, learning_rate):
        from riesznet import RieszNet
        n = RieszNet(estimand=L._estimand(estimand, COV), hidden_sizes=(200, 200, 200), activation="elu",
                     learning_rate=learning_rate, weight_decay=1e-3, epochs=NUM_EPOCHS, batch_size=64,
                     early_stopping_rounds=NN_PATIENCE, validation_fraction=VAL_FRAC, snapshot_epochs=[],
                     random_state=L.SEED).fit(train[["A"] + COV])
        return L._riesz_predict(n, test), n.best_iteration_ + 1
    return _cv_select(fit, L._grid(learning_rate=NN_LR), estimand, frame(*tr), frame(*est))


def c_forestriesz(estimand, est, tr):
    """The non-augmented ForestRiesz (Chernozhukov et al. 2022) on EconML's GRF."""
    def fit(train, test, min_samples_leaf):
        from forestriesz import ForestRieszRegressor
        f = ForestRieszRegressor(estimand=L._estimand(estimand, COV), min_samples_leaf=min_samples_leaf,
                                 n_jobs=1, random_state=L.SEED).fit(train[["A"] + COV])
        return L._riesz_predict(f, test), 0
    return _cv_select(fit, FOREST_GRID, estimand, frame(*tr), frame(*est))


# ------------------------------------------------------------ registry ---
# component -> (function, needs estimand, needs rep). Ordered slowest first,
# so a parallel run starts the long tasks early.

COMPONENTS = {
    **{f"rieszboost_ref_{e}": (c_rieszboost_ref, e, False) for e in ESTIMANDS},
    **{f"riesznet_ref_{e}": (c_riesznet_ref, e, True) for e in ESTIMANDS},
    **{f"rieszboost_{e}": (c_rieszboost, e, False) for e in ESTIMANDS},
    **{f"rieszboost_ascoded_{e}": (c_rieszboost_ascoded, e, False) for e in ESTIMANDS},
    **{f"riesznet_{e}": (c_riesznet, e, False) for e in ESTIMANDS},
    "outcome": (c_outcome, None, False),
    "propensity": (c_propensity, None, False),
    "propensity_ll": (c_propensity_ll, None, False),
    **{f"forestriesz_{e}": (c_forestriesz, e, False) for e in ESTIMANDS},
}

# method -> alpha source per estimand. "oracle" is the true alpha.
METHODS = {
    "oracle": None,
    "indirect": "propensity",
    "indirect_ll": "propensity_ll",
    "rieszboost_ref": "rieszboost_ref_{e}",
    "rieszboost_ascoded": "rieszboost_ascoded_{e}",
    "rieszboost": "rieszboost_{e}",
    "riesznet_ref": "riesznet_ref_{e}",
    "riesznet": "riesznet_{e}",
    "forestriesz": "forestriesz_{e}",
}


def _versions():
    import importlib.metadata as md
    return {p: md.version(p) for p in ("numpy", "scikit-learn", "xgboost", "torch", "jax", "econml")
            if _has(p, md)}


def _has(p, md):
    try:
        md.version(p)
        return True
    except md.PackageNotFoundError:
        return False


def _key(comp):
    """Everything that determines a component's fit except the replicate:
    its source, the shared settings, the package versions and, for our
    learners, the package source."""
    fn = COMPONENTS[comp][0]
    parts = [inspect.getsource(fn), repr((BASE_SEED, N_DATA, K, VAL_FRAC, N_EST, ESR, SAMPLE_PROP, XGB_PARAMS,
                                          NN_LR, NUM_EPOCHS, NN_PATIENCE, NUM_RESTARTS, FOREST_GRID)),
             repr(sorted(_versions().items()))]
    for f in (_cv_select, _rieszboost, _propensity, _gscv_record, draw, data, frame):
        parts.append(inspect.getsource(f))
    ref_src = {"rieszboost_ref": ["rrboost/boosters.py"], "riesznet_ref": ["nn_estimators"],
               "outcome": ["rrboost/xgboost_cv.py"], "propensity": ["rrboost/xgboost_cv.py"]}
    ours = {"rieszboost": ["rieszreg", "rieszboost"], "riesznet": ["rieszreg", "riesznet"],
            "forestriesz": ["rieszreg", "forestriesz"]}
    stem = comp.rsplit("_", 1)[0] if comp.endswith(ESTIMANDS) else comp
    stem = stem.replace("_ascoded", "").replace("_ll", "")
    for rel in ref_src.get(stem, []):
        parts += [p.read_text() for p in sorted((REF / rel).rglob("*.py") if (REF / rel).is_dir() else [REF / rel])]
    for pkg in ours.get(stem, []):
        parts += [p.read_text() for p in sorted((_env.REPO / "packages" / pkg / "python" / pkg).rglob("*.py"))]
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]


def cache_path(comp, rep, key=None):
    return CACHE / comp / f"rep{rep}_{key or _key(comp)}.pkl"


def run_task(rep, comp, key):
    path = cache_path(comp, rep, key)
    if path.exists():
        return
    fn, estimand, needs_rep = COMPONENTS[comp]
    np.random.seed(zlib.crc32(f"{BASE_SEED}/{rep}/{comp}".encode()))   # their code draws from the global RNG
    est, tr = data(rep)
    args = ((estimand,) if estimand else ()) + (est, tr) + ((rep,) if needs_rep else ())
    t0 = time.perf_counter()
    try:
        preds, record = fn(*args)
        out = {"preds": {k: np.asarray(v, dtype=float) for k, v in preds.items()}, "record": record}
    except Exception as err:          # a failed fit is recorded, and the run continues
        out = {"preds": None, "record": {"error": repr(err)}}
    out["seconds"] = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.to_pickle(out, tmp)
    tmp.replace(path)
    print(f"done rep={rep} {comp} {out['seconds']:.0f}s{' FAILED' if out['preds'] is None else ''}", flush=True)


def run(reps, comps, jobs):
    keys = {c: _key(c) for c in comps}
    tasks = [(r, c, keys[c]) for c in comps for r in reps if not cache_path(c, r, keys[c]).exists()]
    print(f"{len(tasks)} tasks to run", flush=True)
    if jobs == 1:
        for t in tasks:
            run_task(*t)
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=jobs, batch_size=1)(delayed(run_task)(*t) for t in tasks)


# ------------------------------------------------------------- summary ---

PAPER = {  # manuscript Tables 1 and 2 (1000 replicates)
    ("ATE", "rieszboost_ref"): dict(alpha_rmse=0.902, alpha_mae=0.534, pct_bias=0.421, rmse=0.238, coverage=0.941),
    ("ATE", "indirect"): dict(alpha_rmse=0.895, alpha_mae=0.532, pct_bias=0.237, rmse=0.237, coverage=0.946),
    ("ATE", "riesznet_ref"): dict(alpha_rmse=0.989, alpha_mae=2.489, pct_bias=-0.854, rmse=0.253, coverage=0.905),
    ("ATT", "rieszboost_ref"): dict(alpha_rmse=0.675, alpha_mae=0.303, pct_bias=0.582, rmse=0.259, coverage=0.936),
    ("ATT", "indirect"): dict(alpha_rmse=0.709, alpha_mae=0.300, pct_bias=0.405, rmse=0.273, coverage=0.943),
    ("ATT", "riesznet_ref"): dict(alpha_rmse=3.194, alpha_mae=1.610, pct_bias=0.253, rmse=0.283, coverage=0.952),
}


def load(comp, reps=None):
    key, d = _key(comp), CACHE / comp
    out = {}
    for f in d.glob(f"rep*_{key}.pkl"):
        r = int(f.name[3:].split("_")[0])
        if reps is None or r in reps:
            out[r] = pd.read_pickle(f)
    return out


def estimates():
    """One row per (rep, estimand, method): the DML estimate, its SE, and the
    alpha-hat error on the estimation set. Only replicates with an outcome fit."""
    mu = load("outcome")
    comp_cache = {}
    rows = []
    for rep in sorted(mu):
        if mu[rep]["preds"] is None:
            continue
        (Y, A, X), _ = data(rep)
        m = mu[rep]["preds"]
        outcome = lambda a, X_, m=m: m["mu1"] if np.all(a == 1) else m["mu0"]  # noqa: E731
        for e in ESTIMANDS:
            a0 = alpha_from_pi(e, A, pi0(X))
            for method, src in METHODS.items():
                if src is None:
                    alpha = a0
                else:
                    c = src.format(e=e)
                    if c not in comp_cache:
                        comp_cache[c] = load(c)
                    got = comp_cache[c].get(rep)
                    if got is None:
                        continue
                    if got["preds"] is None:
                        rows.append(dict(rep=rep, estimand=e, method=method, failed=True))
                        continue
                    p = got["preds"]
                    alpha = alpha_from_pi(e, A, p["pi"]) if "pi" in p else p["alpha"]
                est, se = getattr(_E, e)(X, A, Y, lambda A_, X_, al=alpha: al, outcome)
                rows.append(dict(rep=rep, estimand=e, method=method, failed=False, est=est, se=se,
                                 alpha_rmse=float(np.sqrt(np.mean((alpha - a0) ** 2))),
                                 alpha_mae=float(np.mean(np.abs(alpha - a0)))))
    return pd.DataFrame(rows)


def table(df):
    """Performance per (estimand, method), each with its Monte Carlo SE."""
    ok = df[~df.failed].copy()
    ok["err"] = ok.est - ok.estimand.map(PSI)
    ok["cover"] = ok.err.abs() <= 1.96 * ok.se
    out = []
    for (e, m), g in ok.groupby(["estimand", "method"], sort=False):
        R, err = len(g), g.err.to_numpy()
        mse = np.mean(err ** 2)
        rmse = np.sqrt(mse)
        out.append(dict(
            estimand=e, method=m, reps=R, failed=int(df[(df.estimand == e) & (df.method == m)].failed.sum()),
            alpha_rmse=g.alpha_rmse.mean(), alpha_rmse_mcse=g.alpha_rmse.std(ddof=1) / np.sqrt(R),
            alpha_mae=g.alpha_mae.mean(), alpha_mae_mcse=g.alpha_mae.std(ddof=1) / np.sqrt(R),
            pct_bias=100 * err.mean() / PSI[e], pct_bias_mcse=100 * err.std(ddof=1) / np.sqrt(R) / PSI[e],
            rmse=rmse, rmse_mcse=np.std(err ** 2, ddof=1) / np.sqrt(R) / (2 * rmse),
            emp_se=err.std(ddof=1), mean_se=g.se.mean(),
            coverage=g.cover.mean(), coverage_mcse=np.sqrt(g.cover.mean() * (1 - g.cover.mean()) / R)))
    return pd.DataFrame(out)


def compare_to_paper(t):
    """z = (ours - paper) / sqrt(MCSE_ours^2 + MCSE_paper^2). The paper's MCSE
    is ours rescaled from our replicate count to its 1000."""
    rows = []
    for (e, m), p in PAPER.items():
        r = t[(t.estimand == e) & (t.method == m)]
        if r.empty:
            continue
        r = r.iloc[0]
        for k, v in p.items():
            ours, se = r[k], r[f"{k}_mcse"]
            se_paper = se * np.sqrt(r.reps / 1000)
            rows.append(dict(estimand=e, method=m, measure=k, paper=v, ours=ours, mcse=se,
                             z=(ours - v) / np.hypot(se, se_paper)))
    return pd.DataFrame(rows)


def tuning(comps=None):
    """Per component: how often each setting was selected, the share of
    selections on a grid edge, and the share of refits at the iteration cap."""
    lines = []
    for comp in comps or COMPONENTS:
        got = load(comp)
        recs = [g["record"] for g in got.values() if g["preds"] is not None and "best" in g["record"]]
        if not recs:
            continue
        best = pd.Series([str(r["best"]) for r in recs]).value_counts(normalize=True)
        lines.append(f"\n{comp} ({len(recs)} reps, {np.mean([g['seconds'] for g in got.values()]):.0f}s mean)")
        lines.append(best.round(2).to_string())
        its = [r.get("iters") for r in recs if isinstance(r.get("iters"), (int, np.integer))]
        if its:
            cap = NUM_EPOCHS if "riesznet" in comp else N_EST
            lines.append(f"  refit iterations: median {int(np.median(its))}, share at cap {np.mean(np.array(its) >= cap):.2f}")
        for k in ("tree_depth_used", "patience_used", "sample_prop_used"):
            if k in recs[0]:
                lines.append(f"  {k}: {sorted(set(r[k] for r in recs))}")
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
    cols = ["estimand", "method", "reps", "failed", "alpha_rmse", "alpha_mae", "pct_bias", "rmse",
            "rmse_mcse", "emp_se", "mean_se", "coverage", "coverage_mcse"]
    print(t[cols].round(3).to_string(index=False))
    c = compare_to_paper(t)
    if not c.empty:
        c.to_csv(RESULTS / "compare_to_paper.csv", index=False)
        print("\nAgainst the manuscript (z uses both MCSEs):")
        print(c.round(3).to_string(index=False))
    print(tuning())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--reps", default="0:2", help="half-open range, e.g. 0:100")
    r.add_argument("--components", nargs="*", default=None, help="default: all")
    r.add_argument("--jobs", type=int, default=1)
    sub.add_parser("summarize")
    sub.add_parser("truth")
    sub.add_parser("keys")
    a = ap.parse_args()
    if a.cmd == "run":
        lo, hi = (int(v) for v in a.reps.split(":"))
        run(range(lo, hi), a.components or list(COMPONENTS), a.jobs)
    elif a.cmd == "summarize":
        summarize()
    elif a.cmd == "truth":
        print(truth())
    elif a.cmd == "keys":
        for c in COMPONENTS:
            print(c, _key(c))
