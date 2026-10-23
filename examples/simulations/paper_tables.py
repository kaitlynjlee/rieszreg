"""Build the paper's tables (LaTeX bodies) and the overlap-ladder figure from a
summarized run (its rows.csv and psi.csv, written by `pilot.py summarize`).

    python paper_tables.py OUT_DIR FIGURE_PDF [RUN]

RUN is a folder under results/ (default final_split, the final run; pilot_cf5
for the 50-replicate pilot). The main-text tables use the eight core ATE and
ATT DGPs (CORE); tab_appendix.tex and tab_appendix_alpha.tex hold the five
others (APPENDIX), which have 200 replicates in the final run. Run from examples/simulations,
with a Python that has matplotlib (the project venv does not; Anaconda's does).
The prose numbers in the paper are not regenerated: check them by hand."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import dgp as D  # noqa: E402

OUT = Path(sys.argv[1])
FIG = Path(sys.argv[2])
RUN = Path("results") / (sys.argv[3] if len(sys.argv) > 3 else "final_split")
p = pd.read_csv(RUN / "psi.csv")
r = pd.read_csv(RUN / "rows.csv")

CORE = ["overlap_g05", "overlap_g1", "overlap_g2", "overlap_g3", "logistic", "sparse50", "interaction", "steps10"]
APPENDIX = ["tail_feature", "bulk_feature", "smooth", "step", "additive"]
ATE = CORE + APPENDIX
SHIFT = ["shift_gauss", "shift_hetero", "shift_skew", "shift_bimodal"]
NOT_WEAK = [g for g in CORE if g != "overlap_g3"]
REF = {"ate": "plugin_xgb_clip01", "att": "plugin_xgb_clip05", "shift": "plugin_classif", "localshift": "plugin_classif"}
RN = {"ate": "riesznet_clip01", "att": "riesznet_clip05", "shift": "riesznet_clip01", "localshift": "riesznet_clip01",
      "tsm": "riesznet_clip01"}
LEARNERS = ["rieszboost", "riesznet", "krrr", "riesztree", "augforestriesz", "forestriesz"]
NAME = {"rieszboost": "rieszboost", "riesznet": "riesznet (capped)", "krrr": "krrr", "riesztree": "riesztree",
        "augforestriesz": "augforestriesz", "forestriesz": "forestriesz", "oracle": "oracle",
        "ref": "reference plug-in", "plugin_logit": "logistic plug-in", "rieszboost_kl": "rieszboost, KL loss"}


def variant(blk, m):
    return {"ref": REF[blk], "riesznet": RN[blk]}.get(m, m)


def zpair(x, a, b):
    d = (x[a] - x[b]).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() / se if se > 0 else np.nan


def rng(v, fmt="{:.2f}"):
    v = [x for x in v if np.isfinite(x)]
    lo, hi = fmt.format(min(v)), fmt.format(max(v))
    return lo if lo == hi else f"{lo}--{hi}"


# ---- Table: claim 1 and 4, all estimands -------------------------------
def works_cells(blk, dgps, m):
    s = p[(p.block == blk) & p.dgp.isin(dgps)]
    v = variant(blk, m)
    if v not in set(s.variant):
        return ["--", "--", "--"]
    w = s.pivot_table(index=["dgp", "rep"], columns="variant", values="sqerr")
    cov = s[s.variant == v].groupby("dgp").cover.mean()
    ratio = [np.sqrt(w.loc[g][v].mean() / w.loc[g][REF[blk]].mean()) for g in dgps]
    worse = sum(zpair(w.loc[g], v, REF[blk]) >= 3 for g in dgps) if v != REF[blk] else None
    return [rng(cov.values), "1" if v == REF[blk] else rng(ratio), "--" if worse is None else f"{worse}/{len(dgps)}"]


blocks = [("ate", NOT_WEAK), ("att", NOT_WEAK), ("shift", SHIFT), ("localshift", SHIFT)]
lines = []
for m in ["oracle", "ref", "plugin_logit"] + LEARNERS:
    cells = sum((works_cells(b, d, m) for b, d in blocks), [])
    lines.append(NAME[m] + " & " + " & ".join(cells) + r" \\")
    if m == "plugin_logit":
        lines.append(r"\addlinespace")
(OUT / "tab_works.tex").write_text("\n".join(lines) + "\n")

# ---- Table: TSM rate --------------------------------------------------
lines = []
s = p[p.block == "tsm"]
for g in ["additive", "additive_g2", "interaction"]:
    for n in [250, 1000, 4000]:
        c = s[(s.dgp == g) & (s.n == n)]
        cells = []
        for v in ["oracle", "plugin_xgb_clip01", "rieszboost", "rieszboost_kl", RN["tsm"], "krrr", "forestriesz"]:
            x = c[c.variant == v]
            cells.append(f"{np.sqrt(x.sqerr.mean()) * np.sqrt(n):.2f} ({x.cover.mean():.2f})")
        lines.append((rf"\texttt{{{g.replace('_', chr(92) + '_')}}}" if n == 250 else "") + f" & {n} & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")
(OUT / "tab_tsm.tex").write_text("\n".join(lines[:-1]) + "\n")

# ---- Table: claim 2, which learner has the best alpha-hat ---------------
def learner_rows(blk, dgps, ms):
    w = r[(r.block == blk) & r.dgp.isin(dgps)].pivot_table(index=["dgp", "rep"], columns="variant", values="rmse")
    out = []
    for g in dgps:
        x = w.loc[g]
        vs = [variant(blk, m) for m in ms]
        mean = {v: x[v].mean() for v in vs}
        best = min(vs, key=mean.get)
        second = min([v for v in vs if v != best], key=mean.get)
        resolved = zpair(x, best, second) <= -3
        cells = []
        for m in LEARNERS:
            v = variant(blk, m)
            if v not in vs:
                cells.append("--")
                continue
            cell = f"{mean[v] / mean[best]:.2f}"
            cells.append(rf"\textbf{{{cell}}}" if v == best and resolved else cell)
        out.append(rf"\texttt{{{g.replace('_', chr(92) + '_')}}} & " + " & ".join(cells) + r" \\")
    return out


lines = [r"\multicolumn{7}{l}{\emph{ATE}} \\"] + learner_rows("ate", CORE, LEARNERS)
lines += [r"\addlinespace", r"\multicolumn{7}{l}{\emph{ATT}} \\"] + learner_rows("att", CORE, LEARNERS)
lines += [r"\addlinespace", r"\multicolumn{7}{l}{\emph{Additive shift}} \\"] + learner_rows("shift", SHIFT, ["rieszboost", "riesznet", "krrr"])
lines += [r"\addlinespace", r"\multicolumn{7}{l}{\emph{LASE}} \\"] + learner_rows("localshift", SHIFT, ["rieszboost", "riesznet", "krrr"])
(OUT / "tab_learners.tex").write_text("\n".join(lines) + "\n")

# ---- Table: claim 2, why differences shrink in psi-hat ----------------
def decomp(blk, dgps, ms):
    rows = []
    for g in dgps:
        c = p[(p.block == blk) & (p.dgp == g)]
        e = c.pivot_table(index="rep", columns="variant", values="err")
        sq = c.pivot_table(index="rep", columns="variant", values="sqerr")
        ar = r[(r.block == blk) & (r.dgp == g)].groupby("variant").rmse.mean()
        vs = [variant(blk, m) for m in ms]
        delta = e[vs].sub(e["oracle"], axis=0)
        ratio = np.sqrt((delta ** 2).mean()).mean() / e["oracle"].std()
        rmse = np.sqrt(sq[vs].mean())
        rows.append(dict(shared=e["oracle"].std(), alpha_part=np.sqrt((delta ** 2).mean()).mean(), ratio=ratio,
                         a_spread=ar[vs].max() / ar[vs].min(), p_spread=rmse.max() / rmse.min()))
    return pd.DataFrame(rows)


lines = []
for label, blk, dgps, ms in [(f"ATE ({len(NOT_WEAK)} DGPs)", "ate", NOT_WEAK, LEARNERS),
                             (f"ATT ({len(NOT_WEAK)} DGPs)", "att", NOT_WEAK, LEARNERS),
                             ("Additive shift (4)", "shift", SHIFT, ["rieszboost", "riesznet", "krrr"]),
                             ("LASE (4)", "localshift", SHIFT, ["rieszboost", "riesznet", "krrr"])]:
    t = decomp(blk, dgps, ms)
    lines.append(f"{label} & {rng(t.shared)} & {rng(t.alpha_part)} & {rng(t.ratio)} & "
                 f"{rng(t.a_spread)} & {rng(t.p_spread)} \\\\")
(OUT / "tab_decomp.tex").write_text("\n".join(lines) + "\n")

# ---- Appendix: the five other ATE and ATT DGPs, 200 replicates ------------
# Per method and DGP: coverage, and RMSE as a ratio to the reference plug-in's,
# bold where the method is resolved worse than the reference (z >= 3).
def appendix_cells(blk, m):
    s = p[(p.block == blk) & p.dgp.isin(APPENDIX)]
    v = variant(blk, m)
    w = s.pivot_table(index=["dgp", "rep"], columns="variant", values="sqerr")
    cells = []
    for g in APPENDIX:
        cov = s[(s.dgp == g) & (s.variant == v)].cover.mean()
        if v == REF[blk]:
            cells += [f"{cov:.2f}", "1"]
            continue
        ratio = f"{np.sqrt(w.loc[g][v].mean() / w.loc[g][REF[blk]].mean()):.2f}"
        cells += [f"{cov:.2f}", rf"\textbf{{{ratio}}}" if zpair(w.loc[g], v, REF[blk]) >= 3 else ratio]
    return cells


lines = []
for blk, label in [("ate", "ATE"), ("att", "ATT")]:
    lines.append(rf"\multicolumn{{11}}{{l}}{{\emph{{{label}}}}} \\")
    for m in ["oracle", "ref", "plugin_logit"] + LEARNERS:
        lines.append(NAME[m] + " & " + " & ".join(appendix_cells(blk, m)) + r" \\")
    lines.append(r"\addlinespace")
(OUT / "tab_appendix.tex").write_text("\n".join(lines[:-1]) + "\n")
lines = [r"\multicolumn{7}{l}{\emph{ATE}} \\"] + learner_rows("ate", APPENDIX, LEARNERS)
lines += [r"\addlinespace", r"\multicolumn{7}{l}{\emph{ATT}} \\"] + learner_rows("att", APPENDIX, LEARNERS)
(OUT / "tab_appendix_alpha.tex").write_text("\n".join(lines) + "\n")

# ---- Table: claim 3, KL vs squared ------------------------------------
lines = []
for g in ["additive", "additive_g2", "interaction"]:
    for n in [250, 1000, 4000]:
        a = r[(r.block == "tsm") & (r.dgp == g) & (r.n == n)].pivot_table(index="rep", columns="variant", values="rmse")
        q = p[(p.block == "tsm") & (p.dgp == g) & (p.n == n)].pivot_table(index="rep", columns="variant", values="sqerr")
        lines.append((rf"\texttt{{{g.replace('_', chr(92) + '_')}}}" if n == 250 else "") +
                     f" & {n} & {a.rieszboost.mean():.3f} & {a.rieszboost_kl.mean():.3f} & {zpair(a, 'rieszboost_kl', 'rieszboost'):+.1f}"
                     f" & {np.sqrt(q.rieszboost.mean()):.3f} & {np.sqrt(q.rieszboost_kl.mean()):.3f} & {zpair(q, 'rieszboost_kl', 'rieszboost'):+.1f} \\\\")
    lines.append(r"\addlinespace")
(OUT / "tab_kl.tex").write_text("\n".join(lines[:-1]) + "\n")

# ---- Figure: claim 1, the ATE overlap ladder ----------------------------
gam = {"overlap_g05": 0.5, "overlap_g1": 1, "overlap_g2": 2, "overlap_g3": 3}
s = p[(p.block == "ate") & p.dgp.isin(gam)]
cov = s.groupby(["dgp", "variant"]).cover.mean().unstack()
w = s.pivot_table(index=["dgp", "rep"], columns="variant", values="sqerr")
x = [gam[g] for g in gam]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
series = [("ref", "reference plug-in")] + [(m, NAME[m]) for m in LEARNERS]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
mcse = np.sqrt(0.95 * 0.05 / s.groupby("dgp").rep.nunique().min())
ax[0].axhspan(0.95 - 2 * mcse, 0.95 + 2 * mcse, color="#d9d9d4", alpha=0.6, lw=0)
ax[0].axhline(0.95, color="#8a8a85", lw=1, ls="--")
ax[0].plot(x, [cov.loc[g, "oracle"] for g in gam], color="#5c5c58", lw=1.5, ls=":", marker="o", ms=4, label="oracle")
ax[1].axhline(1.0, color="#8a8a85", lw=1, ls="--")
for (m, label), col, mk in zip(series, COLORS, MARKERS):
    v = variant("ate", m)
    ax[0].plot(x, [cov.loc[g, v] for g in gam], color=col, lw=1.5, marker=mk, ms=5, label=label)
    ax[1].plot(x, [np.sqrt(w.loc[g][v].mean() / w.loc[g]["plugin_xgb_clip01"].mean()) for g in gam],
               color=col, lw=1.5, marker=mk, ms=5)
for a in ax:
    a.set_xticks(x)
    a.set_xlabel(r"logit scale $\gamma$ (larger: weaker overlap)")
    a.grid(axis="y", color="#e6e6e1", lw=0.6)
ax[0].set_ylabel("coverage of the 95% interval")
ax[0].set_ylim(0.7, 1.0)
ax[1].set_ylabel("RMSE / reference plug-in's RMSE")
fig.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(rect=(0, 0.08, 1, 1))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, bbox_inches="tight")

# ---- numbers quoted in the text -----------------------------------------
print("ATT psi0:", {g: round(D.true_psi(g, "ATT"), 3) for g in ATE})
print("shift psi0:", {g: round(D.true_psi(g, "SHIFT"), 3) for g in SHIFT})
print("LASE psi0:", {g: round(D.true_psi(g, "LOCALSHIFT"), 3) for g in SHIFT})
print("g3 coverage:", cov.loc["overlap_g3"].round(2).to_dict())
print("g3 ratio:", {v: round(np.sqrt(w.loc["overlap_g3"][v].mean() / w.loc["overlap_g3"]["plugin_xgb_clip01"].mean()), 2) for v in ["oracle"] + [variant("ate", m) for m in LEARNERS]})
for blk in ["att"]:
    c = p[(p.block == blk) & (p.dgp == "overlap_g3")].groupby("variant").cover.mean()
    print("ATT g3 coverage:", c[[REF[blk], RN[blk]] + LEARNERS[2:] + ["rieszboost"]].round(2).to_dict())
