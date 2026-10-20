"""
Pilot check: can the design resolve the contrast the mockup promises?

Python counterpart of pilot_check.R.

Every contrast a mockup displays reduces to a per-repetition difference d_i.
The displayed quantity is mean(d); its Monte Carlo SE at n_sim repetitions is
sd(d)/sqrt(n_sim). So to show a gap at k Monte Carlo SEs you need

    n_sim >= (k * sd(d) / mean(d))**2

This is a power calculation for the simulation itself. Run it on ~100-200
pilot repetitions, BEFORE the full run, for every cell comparison the mockup
promises. See references/piloting.md.

Pairing matters: both methods see the same datasets, so d_i is formed within a
repetition and sd(d) is usually far below either method's own spread.

Input `results`: tidy pilot output with columns rep, DGP, est, se and a column
naming the method (`method`, default "estimator"). Optional: lo, hi, p.
Contrasts pair repetitions within each group of the `by` columns. When the
design has factors beyond the DGP, such as sample size or learner library,
name them in `by`, so that each group holds one row per repetition per method.
To compare libraries for one estimator, set method="library" and put
"estimator" in `by`.

Run:  uv run --with numpy,pandas,scipy python pilot_check.py
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm

MEASURES = ("bias", "mse", "coverage", "rejection")


def pilot_contrast(results: pd.DataFrame, measure: str, a: str, b: str | None = None,
                   true_values: dict | None = None, k: float = 5.0,
                   n_sim_planned: int | None = None,
                   level: float = 0.95, alpha: float = 0.05,
                   by: str | list[str] = "DGP", method: str = "estimator") -> pd.DataFrame:
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    if measure in ("bias", "mse", "coverage") and not true_values:
        raise ValueError(f"`true_values` is required for measure '{measure}': "
                         "a dict of the true estimand per DGP")

    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, g in results.groupby(by, sort=False):
        theta = (true_values or {}).get(g.DGP.iloc[0])
        d = _pilot_d(g, measure, a, b, theta, level, alpha, method)
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)),
                     **_pilot_verdict(d, measure, a, b, k, n_sim_planned)})
    return pd.DataFrame(rows)


def _pilot_d(g, measure, a, b, theta, level, alpha, method) -> np.ndarray:
    def pick(m):
        r = g[g[method] == m].sort_values("rep")
        if r.empty:
            raise ValueError(f"{method} '{m}' not found in DGP '{g.DGP.iloc[0]}'")
        # A repeated rep means another design factor varies inside the group,
        # and pairing on rep would silently match rows from different cells.
        if r.rep.duplicated().any():
            raise ValueError(
                f"`rep` repeats for {method} '{m}' within one group. Name the other "
                "design factors in `by`, e.g. by=['DGP', 'n_obs', 'library'].")
        return r

    A = pick(a)
    B = pick(b) if b else None
    if B is not None:                       # pair strictly on rep
        keep = np.intersect1d(A.rep.values, B.rep.values)
        A, B = A[A.rep.isin(keep)], B[B.rep.isin(keep)]

    def covers(r):
        if {"lo", "hi"}.issubset(r.columns):
            return (r.lo.values <= theta) & (theta <= r.hi.values)
        z = norm.ppf(1 - (1 - level) / 2)
        return (r.est.values - z * r.se.values <= theta) & (
            theta <= r.est.values + z * r.se.values)

    def rejects(r):
        if "p" in r.columns:
            return r.p.values <= alpha
        return np.abs(r.est.values) >= norm.ppf(1 - alpha / 2) * r.se.values

    stat = {
        "bias": lambda r: r.est.values - theta,
        "mse": lambda r: (r.est.values - theta) ** 2,
        "coverage": lambda r: covers(r).astype(float),
        "rejection": lambda r: rejects(r).astype(float),
    }[measure]

    dA = stat(A)
    if B is not None:
        return dA - stat(B)
    if measure == "coverage":
        return dA - level                   # against nominal
    return dA                               # against zero


def _pilot_verdict(d, measure, a, b, k, n_planned) -> dict:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n_pilot = len(d)
    gap, sd_d = float(d.mean()), float(d.std(ddof=1))

    # The pilot's own gap estimate is noisy, and sizing the full run on the
    # point estimate inherits that noise: a gap that happened to come out large
    # gives an n_sim that is too small, so the full run lands short of k MCSEs.
    # Two guards. Report how well the PILOT itself resolves the gap, and size
    # the run on a conservative lower bound for |gap|. Use n_sim_safe.
    #
    # A d that never varies (every pilot rep covered, say) has an unknown SD,
    # not a zero one, and sizing on sd = 0 would advise n_sim >= 0.
    no_spread = not np.isfinite(sd_d) or sd_d == 0
    se_gap = sd_d / np.sqrt(n_pilot) if n_pilot else np.inf
    pilot_z = np.nan if no_spread else abs(gap) / se_gap
    gap_lo = np.nan if no_spread else max(0.0, abs(gap) - norm.ppf(0.975) * se_gap)

    if no_spread:
        need = safe = np.nan
    else:
        need = np.inf if gap == 0 else int(np.ceil((k * sd_d / abs(gap)) ** 2))
        safe = np.inf if gap_lo <= 0 else int(np.ceil((k * sd_d / gap_lo) ** 2))
    planned = bool(n_planned) and not no_spread
    mcse_planned = sd_d / np.sqrt(n_planned) if planned else np.nan
    z_planned = abs(gap) / mcse_planned if planned else np.nan

    if no_spread:
        verdict = (f"NO SPREAD: d took the same value in all {n_pilot} pilot reps, so its "
                   "SD is unknown rather than zero (common for coverage and rejection "
                   "contrasts). Enlarge the pilot.")
    elif pilot_z < 2:
        enlarge = (f" to >= {int(np.ceil((2 * sd_d / abs(gap)) ** 2)):,} reps"
                   if gap else "")
        verdict = (f"PILOT TOO SMALL: the gap is not distinguishable from zero in the "
                   f"pilot itself (|z| = {pilot_z:.2f} on {n_pilot} reps), so any n_sim "
                   f"estimate here is noise. Enlarge the pilot{enlarge}, "
                   f"or treat the contrast as unresolvable.")
    elif not np.isfinite(safe):
        verdict = "DEAD: no gap the pilot can detect. Redesign or drop the cell."
    elif n_planned and safe <= n_planned:
        verdict = (f"OK at planned n_sim = {n_planned} (gap is {z_planned:.1f} MCSEs; "
                   f"safe requirement {safe})")
    elif safe <= 50_000:
        verdict = (f"RAISE n_sim to >= {safe} (optimistic {need}, "
                   f"planned {n_planned or 'unset'})")
    else:
        verdict = (f"REDESIGN: needs n_sim >= {safe:,}. Amplify the signal in the DGP, "
                   f"change what the table displays, or change n_obs.")

    return dict(measure=measure, contrast=a if b is None else f"{a} - {b}",
                n_pilot=n_pilot, gap=gap, sd_d=sd_d, pilot_z=pilot_z,
                mcse_at_planned=mcse_planned, mcse_multiples=z_planned,
                n_sim_required=need, n_sim_safe=safe, verdict=verdict)


def pilot_sweep(results, measure, true_values=None, k=5.0, n_sim_planned=None,
                method="estimator", **kw):
    """Every pair of methods, for one measure."""
    ms = list(dict.fromkeys(results[method]))
    return pd.concat(
        [pilot_contrast(results, measure, x, y, true_values, k, n_sim_planned,
                        method=method, **kw)
         for x, y in itertools.combinations(ms, 2)],
        ignore_index=True)


def pilot_equivalence(results, measure, a, b, true_values, margin, k=2.0, **kw):
    """For claims that two methods perform the SAME.

    A small gap with a large MCSE is an unresolvable design, not evidence of
    equivalence. State the largest difference `margin` that still counts as
    equivalent, then size the run so the interval for the gap fits inside it.
    """
    r = pilot_contrast(results, measure, a, b, true_values, k=k, **kw).copy()
    r["margin"] = margin
    spread = np.isfinite(r.sd_d) & (r.sd_d > 0)
    r["n_sim_required"] = np.where(spread, np.ceil((2 * k * r.sd_d / margin) ** 2), np.nan)
    r["n_sim_safe"] = r["n_sim_required"]
    r["verdict"] = [f"equivalence within {margin:.4g} needs n_sim >= {int(n):,}" if ok else v
                    for n, ok, v in zip(r.n_sim_required, spread, r.verdict)]  # keeps NO SPREAD
    return r


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    theta = 1.0

    def mk(dgp, reps, bias_a, bias_b, s=0.10, rho=0.9):
        shared = rng.normal(0, s * np.sqrt(rho), reps)
        ea = theta + bias_a + shared + rng.normal(0, s * np.sqrt(1 - rho), reps)
        eb = theta + bias_b + shared + rng.normal(0, s * np.sqrt(1 - rho), reps)
        return pd.concat([
            pd.DataFrame(dict(rep=range(reps), DGP=dgp, estimator="plugin", est=ea, se=s)),
            pd.DataFrame(dict(rep=range(reps), DGP=dgp, estimator="dr", est=eb, se=s))])

    # "cancel": misspecification bias nearly cancels in the contrast.
    # "sharp":  it does not.
    pilot = pd.concat([mk("cancel", 150, 0.0224, 0.0200),
                       mk("sharp", 150, 0.0500, 0.0000)], ignore_index=True)

    res = pilot_contrast(pilot, "bias", "plugin", "dr",
                         {"cancel": theta, "sharp": theta}, k=5, n_sim_planned=1000)
    print(res[["DGP", "gap", "sd_d", "pilot_z", "n_sim_required", "n_sim_safe"]]
          .to_string(index=False))
    print()
    for _, r in res.iterrows():
        print(f"  - {r.DGP} :: {r.verdict}")
