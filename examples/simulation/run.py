"""One pipeline from a simulated dataset to every stored result.

A cell is one simulated dataset, named by its sample size n and repetition
rep. For each cell:

  1. Draw n observations and K = 5 cross-fitting folds from a random stream
     named by (n, rep), so a cell draws the same data alone as in the full grid.
  2. Fit every setting of every learner type on each fold's training part and
     predict the held-out part (cached). Pooling the held-out predictions gives
     each setting's cross-validated risk; the setting with the smallest risk is
     selected, and its out-of-fold predictions are the cross-fitted estimate.
  3. For each estimand (ATE, ATT) and each source of alpha-hat (oracle,
     propensity plug-in, rieszboost, riesznet), compute the one-step estimate
     with the same cross-fitted outcome regression mu-hat.

Usage:
  python run.py --n 500 1000 2000 --reps 0:100 --jobs 9
  python run.py --dgp lee_schuler --n 1000 --reps 0:100 --jobs 9
"""

from __future__ import annotations

import _env  # noqa: F401  (threads and import paths; must precede numpy)

import argparse
import hashlib
import inspect
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import dgp
import learners as L

HERE = Path(__file__).resolve().parent
# Where fits and results go. Defaults to this folder; set RIESZ_SIM_OUT to put
# them elsewhere, e.g. on cluster scratch storage.
OUT_ROOT = Path(os.environ.get("RIESZ_SIM_OUT", HERE))
CACHE_DIR = OUT_ROOT / "cache"             # git-ignored; deleting it changes only runtime
CELL_DIR = OUT_ROOT / "results" / "cells"  # one file of raw results per cell
BASE_SEED = 20260924
K = 5
N_OBS = (500, 1000, 2000)
ESTIMANDS = ("ATE", "ATT")
METHODS = ("oracle", "propensity", "rieszboost", "riesznet")
PI_CLIP = 0.01   # propensity plug-in clips pi-hat to [0.01, 0.99]


# ------------------------------------------------------------ data ---

def cell_data(dgp_name: str, n: int, rep: int) -> tuple[pd.DataFrame, np.ndarray]:
    # shared_terms, the first main DGP, keeps the seed it had before other DGPs
    # existed, so its replicates stay paired with the archived runs. PCG64.
    seed = ([BASE_SEED, n, rep] if dgp_name == "shared_terms"
            else [BASE_SEED, zlib.crc32(dgp_name.encode()), n, rep])
    rng = np.random.default_rng(seed)
    d = dgp.DGPS[dgp_name].draw(n, rng)
    folds = rng.permutation(np.arange(n) % K)
    return d, folds


# ----------------------------------------------------------- cache ---

def _package_digest() -> str:
    """Hash of the learner packages' source and the library versions, so a
    change to rieszboost or riesznet invalidates every fit that used it."""
    import sklearn
    import torch
    import xgboost
    h = hashlib.sha256()
    for pkg in ("rieszreg", "rieszboost", "riesznet"):
        root = _env.REPO / "packages" / pkg / "python" / pkg
        for f in sorted(root.rglob("*.py")):
            h.update(f.read_bytes())
    h.update(f"{xgboost.__version__} {torch.__version__} {sklearn.__version__} {np.__version__}".encode())
    return h.hexdigest()[:16]


PACKAGE_DIGEST = _package_digest()


def _digest(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, pd.DataFrame):
            h.update(pd.util.hash_pandas_object(p, index=False).to_numpy().tobytes())
        elif isinstance(p, np.ndarray):
            h.update(p.tobytes())
        elif callable(p):
            h.update(inspect.getsource(p).encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:20]


def learner_fingerprint() -> str:
    """Hash of everything in learners.py and the packages that shapes a
    result: every learner's source, fixed arguments and grid, the shared
    constants, and the package source and versions. Stored in each cell's
    results, so a cell made under other grids is never mistaken for a
    finished one."""
    parts = [PACKAGE_DIGEST, L.SEED, L.VALID_FRAC, L.PATIENCE, L.L2_PENALTY, L.covariates,
             L._estimand, L._riesz_predict, L._xgb_split, L.row_loss, L.riesz_loss, PI_CLIP, K]
    for name, (func, fixed, grid) in L.LEARNERS.items():
        parts += [name, func, fixed, grid]
    return _digest(*parts)


def _setting_label(setting: dict) -> str:
    return "_".join(f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in setting.items())


def fit_oof(d: pd.DataFrame, folds: np.ndarray, learner: str, setting: dict,
            dgp_name: str, n: int, rep: int) -> dict:
    """Out-of-fold predictions of one setting of one learner type, and the
    iterations early stopping kept in each fold. Cached, keyed on the data, the
    folds, the learner's source and constants, the setting and the packages."""
    func, fixed, _ = L.LEARNERS[learner]
    constants = (L.SEED, L.VALID_FRAC, L.PATIENCE)
    if func is not L.riesznet:          # the boosted learners read L2_PENALTY
        constants += (L.L2_PENALTY,)
    key = _digest(d, folds, func, L._estimand, L._riesz_predict, L._xgb_split, L.covariates,
                  constants, fixed, setting, PACKAGE_DIGEST)
    path = CACHE_DIR / dgp_name / learner / f"n{n}_rep{rep}_{_setting_label(setting)}_{key}.pkl"
    if path.exists():
        return pd.read_pickle(path)

    parts, iters, failed = [], [], 0
    for k in range(K):
        train = d[folds != k].reset_index(drop=True)
        test = d[folds == k]
        try:
            preds, it = func(train, test.reset_index(drop=True), **fixed, **setting)
        except Exception as e:  # count failures, do not crash the run
            print(f"FAILED {learner} {setting} n={n} rep={rep} fold={k}: {e!r}", flush=True)
            failed += 1
            preds, it = None, np.nan
        if preds is not None:
            preds.index = test.index
            parts.append(preds)
        iters.append(it)
    out = {"preds": pd.concat(parts).sort_index() if parts else None,
           "iters": iters, "failed": failed}
    if failed:          # never cache a partial fit
        return out
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    pd.to_pickle(out, tmp)
    os.replace(tmp, path)
    return out


def select(d, folds, learner, dgp_name, n, rep) -> tuple[pd.DataFrame, list[dict]]:
    """Fit every setting in the learner's grid; return the selected setting's
    out-of-fold predictions and one tuning row per setting."""
    _, fixed, grid = L.LEARNERS[learner]
    cap = fixed["max_iter"]
    rows, fits = [], []
    for setting in grid:
        f = fit_oof(d, folds, learner, setting, dgp_name, n, rep)
        ok = f["preds"] is not None and f["failed"] == 0
        risk = float(np.mean(L.row_loss(learner, d, f["preds"]))) if ok else np.nan
        rows.append(dict(learner=learner, setting=json.dumps(setting), cv_risk=risk,
                         iters_max=np.nanmax(f["iters"]) if ok else np.nan,
                         iters_mean=np.nanmean(f["iters"]) if ok else np.nan,
                         cap=cap, failed=f["failed"]))
        fits.append(f)
    risks = np.array([r["cv_risk"] for r in rows])
    if np.all(np.isnan(risks)):
        return None, rows
    best = int(np.nanargmin(risks))
    for i, r in enumerate(rows):
        r["selected"] = i == best
    return fits[best]["preds"], rows


# ------------------------------------------------------ estimators ---

def one_step(estimand: str, d: pd.DataFrame, mu: pd.DataFrame, alpha: np.ndarray):
    a, y = d["A"].to_numpy(), d["Y"].to_numpy()
    tau_hat = (mu["mu1"] - mu["mu0"]).to_numpy()
    resid = y - mu["mu"].to_numpy()
    n = len(y)
    if estimand == "ATE":
        phi = tau_hat + alpha * resid
        est = phi.mean()
    else:
        p = a.mean()
        est = np.mean(a * tau_hat + alpha * resid) / p
        phi = (a * (tau_hat - est) + alpha * resid) / p
    return float(est), float(phi.std(ddof=1) / np.sqrt(n))


def propensity_alpha(estimand, a, pi):
    pi = np.clip(pi, PI_CLIP, 1 - PI_CLIP)
    alpha = lambda aa: aa / pi - (1 - aa) / (1 - pi) if estimand == "ATE" else aa - (1 - aa) * pi / (1 - pi)
    return pd.DataFrame({"alpha": alpha(a), "alpha1": alpha(np.ones_like(a)),
                         "alpha0": alpha(np.zeros_like(a))})


def oracle_alpha(G, estimand, a, X):
    return pd.DataFrame({"alpha": G.true_alpha(estimand, a, X),
                         "alpha1": G.true_alpha(estimand, np.ones_like(a), X),
                         "alpha0": G.true_alpha(estimand, np.zeros_like(a), X)})


# ------------------------------------------------------------- run ---

def run_cell(dgp_name: str, n: int, rep: int, overwrite: bool = False) -> None:
    out_path = CELL_DIR / dgp_name / f"n{n}_rep{rep}.pkl"
    if not overwrite and is_done(dgp_name, n, rep):
        return
    t0 = time.perf_counter()
    G = dgp.DGPS[dgp_name]
    d, folds = cell_data(dgp_name, n, rep)
    X, a = d[list(G.covariates)].to_numpy(), d["A"].to_numpy()
    ids = dict(dgp=dgp_name, n=n, rep=rep)

    tuning, selected = [], {}
    for learner in L.LEARNERS:
        preds, rows = select(d, folds, learner, dgp_name, n, rep)
        selected[learner] = preds
        tuning += [dict(ids, **r) for r in rows]

    mu = selected["outcome"]
    nuisance = []
    if mu is not None:
        nuisance.append(dict(ids, nuisance="mu", rmse=float(np.sqrt(np.mean(
            (mu["mu"].to_numpy() - G.mu(a, X)) ** 2)))))
    if selected["propensity"] is not None:
        nuisance.append(dict(ids, nuisance="pi", rmse=float(np.sqrt(np.mean(
            (selected["propensity"]["pi"].to_numpy() - G.propensity(X)) ** 2)))))

    estimates = []
    for estimand in ESTIMANDS:
        alpha0 = G.true_alpha(estimand, a, X)
        sources = {
            "oracle": oracle_alpha(G, estimand, a, X),
            "propensity": (None if selected["propensity"] is None else
                           propensity_alpha(estimand, a, selected["propensity"]["pi"].to_numpy())),
            "rieszboost": selected[f"rieszboost_{estimand}"],
            "riesznet": selected[f"riesznet_{estimand}"],
        }
        for method, al in sources.items():
            row = dict(ids, estimand=estimand, method=method, est=np.nan, se=np.nan,
                       alpha_rmse=np.nan, riesz_loss=np.nan)
            if al is not None and mu is not None:
                alpha = al["alpha"].to_numpy()
                row["est"], row["se"] = one_step(estimand, d, mu, alpha)
                row["alpha_rmse"] = float(np.sqrt(np.mean((alpha - alpha0) ** 2)))
                row["riesz_loss"] = float(np.mean(L.riesz_loss(
                    estimand, a, alpha, al["alpha1"].to_numpy(), al["alpha0"].to_numpy())))
            estimates.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.tmp{os.getpid()}")
    pd.to_pickle({"estimates": pd.DataFrame(estimates), "tuning": pd.DataFrame(tuning),
                  "nuisance": pd.DataFrame(nuisance), "fingerprint": learner_fingerprint(),
                  "seconds": time.perf_counter() - t0}, tmp)
    os.replace(tmp, out_path)
    print(f"done {dgp_name} n={n} rep={rep} in {time.perf_counter() - t0:.0f}s", flush=True)


def run_sim(dgp_names=(dgp.DEFAULT,), n_obs=N_OBS, reps=range(2), jobs: int = 1,
            overwrite: bool = False) -> None:
    # largest n first, so the slowest cells do not trail at the end
    cells = [(g, n, r) for g in dgp_names for n in sorted(n_obs, reverse=True) for r in reps]
    if jobs == 1:
        for c in cells:
            run_cell(*c, overwrite=overwrite)
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=jobs, batch_size=1)(delayed(run_cell)(*c, overwrite=overwrite) for c in cells)


def is_done(dgp_name: str, n: int, rep: int) -> bool:
    """A cell is done when its results exist and were made by the current learners."""
    path = CELL_DIR / dgp_name / f"n{n}_rep{rep}.pkl"
    return path.exists() and pd.read_pickle(path).get("fingerprint") == learner_fingerprint()


def load_results(dgp_name: str = dgp.DEFAULT, n_obs=None, reps=None,
                 any_fingerprint: bool = False) -> dict[str, pd.DataFrame]:
    """Concatenate the per-cell results of one DGP. Every table and figure
    reads this. Cells made by other learners or grids are skipped, with a
    count, unless any_fingerprint is set."""
    parts: dict[str, list] = {"estimates": [], "tuning": [], "nuisance": []}
    current, skipped = learner_fingerprint(), 0
    for f in sorted((CELL_DIR / dgp_name).glob("n*_rep*.pkl")):
        n, rep = (int(s) for s in f.stem[1:].split("_rep"))
        if (n_obs is not None and n not in n_obs) or (reps is not None and rep not in reps):
            continue
        cell = pd.read_pickle(f)
        if not any_fingerprint and cell.get("fingerprint") != current:
            skipped += 1
            continue
        for k in parts:
            parts[k].append(cell[k])
    if skipped:
        print(f"load_results: skipped {skipped} cells made by other learners or grids "
              f"(pass any_fingerprint=True to include them)")
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in parts.items()}


def _parse_reps(s: str) -> range:
    lo, hi = s.split(":")
    return range(int(lo), int(hi))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", nargs="+", default=[dgp.DEFAULT], choices=list(dgp.DGPS))
    ap.add_argument("--n", type=int, nargs="+", default=list(N_OBS))
    ap.add_argument("--reps", type=_parse_reps, default=range(2), help="half-open, e.g. 0:100")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute cell results (fits still come from the cache)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    run_sim(args.dgp, args.n, args.reps, args.jobs, args.overwrite)
    print(f"total {time.perf_counter() - t0:.0f}s")
