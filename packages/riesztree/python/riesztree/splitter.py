"""Per-leaf Bregman optimum and split-gain computation.

Given the per-augmented-row coefficients ``(D_r, C_r)`` from
``Estimand.augment`` (where ``D_r`` is ``is_original`` and ``C_r``
is ``potential_deriv_coef``), the constant-per-leaf augmented Bregman loss
in leaf ``ℓ`` is

    L(α_ℓ) = D_ℓ · h̃(α_ℓ) + C_ℓ · h'(α_ℓ),

with ``D_ℓ = Σ_{r∈ℓ} D_r`` and ``C_ℓ = Σ_{r∈ℓ} C_r``. Setting the
derivative to zero and using the identity ``h̃'(t) = t · h''(t)``:

    h''(α_ℓ) · (D_ℓ · α_ℓ + C_ℓ) = 0   ⇒   α_ℓ* = -C_ℓ / D_ℓ,

with the value projected to the loss's α-domain when it falls outside.
The splitter sweeps each candidate split position and computes

    gain = L(α_p*) - L(α_ℓ*) - L(α_r*),

picking the largest. ``min_orig_leaf`` constrains the count of original
(``D > 0``) rows in each child.

The per-loss closed forms for ``L(α*)`` and ``α*`` are written once, in
:mod:`riesztree.fast._loss_kernels`; :func:`make_leaf_solvers` hands the
same compiled kernels to the Python paths (leaf payloads, pruning, the
categorical sweep), so split search and leaf values can't drift apart.
A leaf loss is only defined up to a term linear in ``C_ℓ`` (the KL kernel
uses ``h(t) = t log t - t`` where ``rieszreg.KLLoss`` uses ``t log t``);
such a term cancels in every gain and in the pruning criterion, since
``C_parent = C_left + C_right``.
"""

from __future__ import annotations

import numpy as np

from rieszreg import Loss

from .fast._loss_kernels import py_dispatch_alpha_at_opt, py_dispatch_leaf_loss
from .fast._splitter import _lookup_user_kernel, loss_kind_for


# ---------------------------------------------------------------------------
# Leaf-loss-at-optimum and α* per Loss.

def make_leaf_solvers(loss: Loss):
    """Return ``(loss_at_optimum, alpha_at_optimum)`` callables for ``loss``.

    Both signatures: ``f(D: float, C: float) -> float``.

    Built-in losses (``SquaredLoss``, ``KLLoss``, ``BernoulliLoss``,
    ``BoundedSquaredLoss``) get the compiled kernels the split sweeps use.
    A loss registered via :func:`riesztree.fast.register_fast_leaf_solver`
    gets its registered pair; the registry is checked first, as in
    :func:`riesztree.fast._splitter.loss_kind_for`, so a registered
    subclass of a built-in uses the registered solver for both split search
    and leaf values. Any other loss raises ``NotImplementedError``.
    """
    user_entry = _lookup_user_kernel(loss)
    if user_entry is not None:
        return user_entry
    kind, lo, hi, _ = loss_kind_for(loss)

    def leaf_loss(D: float, C: float) -> float:
        return py_dispatch_leaf_loss(kind, D, C, lo, hi)

    def alpha_at_opt(D: float, C: float) -> float:
        return py_dispatch_alpha_at_opt(kind, D, C, lo, hi)

    return leaf_loss, alpha_at_opt


# ---------------------------------------------------------------------------
# Per-feature split sweep.

def best_split_continuous(
    feature_col: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    leaf_loss,
    *,
    min_orig_leaf: int,
):
    """Best gain split on a continuous feature for the rows in ``idx``.

    Pure-Python reference for the Cython sweep in
    :mod:`riesztree.fast._splitter_c` (the parity tests compare the two).
    Returns ``(gain, threshold, left_idx, right_idx)`` or ``None``.
    """
    vals = feature_col[idx]
    order = np.argsort(vals, kind="mergesort")
    sidx = idx[order]
    svals = feature_col[sidx]
    sD = D[sidx]
    sC = C[sidx]
    sorig = (sD > 0).astype(np.int64)

    cum_D = np.cumsum(sD)
    cum_C = np.cumsum(sC)
    cum_orig = np.cumsum(sorig)
    total_D = cum_D[-1]
    total_C = cum_C[-1]
    total_orig = cum_orig[-1]
    parent_loss = leaf_loss(float(total_D), float(total_C))

    best = None
    n = len(svals)
    for k in range(n - 1):
        if svals[k] == svals[k + 1]:
            continue
        D_l = float(cum_D[k])
        C_l = float(cum_C[k])
        n_l = int(cum_orig[k])
        D_r = float(total_D - D_l)
        C_r = float(total_C - C_l)
        n_r = int(total_orig - n_l)
        if n_l < min_orig_leaf or n_r < min_orig_leaf:
            continue
        L_l = leaf_loss(D_l, C_l)
        L_r = leaf_loss(D_r, C_r)
        if not np.isfinite(L_l) or not np.isfinite(L_r):
            continue
        gain = parent_loss - L_l - L_r
        if best is None or gain > best[0]:
            thresh = 0.5 * (svals[k] + svals[k + 1])
            if thresh == svals[k + 1] or not np.isfinite(thresh):
                thresh = svals[k]  # midpoint rounded onto the right value
            best = (gain, float(thresh), sidx[: k + 1].copy(), sidx[k + 1:].copy())
    return best


def best_split_categorical(
    feature_col: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    leaf_loss,
    alpha_at_opt,
    *,
    min_orig_leaf: int,
):
    """Best gain split on a categorical feature for rows in ``idx``.

    Implements the standard Breiman et al. trick: order the levels by the
    within-level optimal α (``α*_level = -C_level / D_level`` projected to
    the loss's α-domain), then sweep contiguous splits in that order. The
    ordering theorem holds for any convex per-level objective, including
    the augmented Bregman loss.
    """
    vals = feature_col[idx]
    levels, inverse = np.unique(vals, return_inverse=True)
    if len(levels) < 2:
        return None

    # Per-level (D, C, n_orig) sums.
    n_lev = len(levels)
    D_idx = D[idx]
    D_lev = np.bincount(inverse, weights=D_idx, minlength=n_lev)
    C_lev = np.bincount(inverse, weights=C[idx], minlength=n_lev)
    n_orig_lev = np.bincount(inverse, weights=(D_idx > 0), minlength=n_lev).astype(np.int64)

    # Sort levels by α*_level.
    a_star_lev = np.array(
        [alpha_at_opt(float(d), float(c)) for d, c in zip(D_lev, C_lev)]
    )
    order = np.argsort(a_star_lev, kind="mergesort")

    cum_D = np.cumsum(D_lev[order])
    cum_C = np.cumsum(C_lev[order])
    cum_orig = np.cumsum(n_orig_lev[order])
    total_D = cum_D[-1]
    total_C = cum_C[-1]
    total_orig = cum_orig[-1]
    parent_loss = leaf_loss(float(total_D), float(total_C))

    # Rank of each level in the α*-sorted order; a split at k sends the
    # levels with rank ≤ k left.
    rank = np.empty(n_lev, dtype=np.int64)
    rank[order] = np.arange(n_lev)
    row_rank = rank[inverse]

    best = None
    for k in range(len(order) - 1):
        D_l = float(cum_D[k])
        C_l = float(cum_C[k])
        n_l = int(cum_orig[k])
        D_r = float(total_D - D_l)
        C_r = float(total_C - C_l)
        n_r = int(total_orig - n_l)
        if n_l < min_orig_leaf or n_r < min_orig_leaf:
            continue
        L_l = leaf_loss(D_l, C_l)
        L_r = leaf_loss(D_r, C_r)
        if not np.isfinite(L_l) or not np.isfinite(L_r):
            continue
        gain = parent_loss - L_l - L_r
        if best is None or gain > best[0]:
            best = (gain, k)
    if best is None:
        return None
    gain, k = best
    goes_left = row_rank <= k
    return (
        gain,
        tuple(int(levels[i]) for i in order[: k + 1]),
        idx[goes_left],
        idx[~goes_left],
    )
