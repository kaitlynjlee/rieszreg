# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Iterative depthwise grow loop for the exact splitter, on presorted columns.

Mirrors :func:`riesztree.fast._grow_c.grow_depthwise_hist_c` (the
histogram-path Cython driver) but operates on raw continuous features
instead of pre-binned features.

Each feature is argsorted once, at the root. Every node owns the same
slice ``[start, end)`` of all ``p`` sorted index arrays, and each slice
stays sorted by its feature: a split stable-partitions every feature's
slice (left rows first, sorted order kept within each side). The split
search for a node then reads its rows in sorted order directly, so a level
of the tree costs O(n_aug · p) rather than the O(n_aug · p · log n_node)
of sorting each (node × feature) slice. The price is one int32 index per
(row, feature), plus a column-major copy of the features.

Same scope guards as :func:`grow_depthwise_hist_c`: ``splitter='exact'``,
``growth_policy='depthwise'``, no categoricals, no max_features
subsampling, no early stopping, built-in loss. Other configurations keep
the Python grower in :mod:`riesztree.grow`.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport INFINITY, isfinite

from ._loss_kernels cimport dispatch_leaf_loss, dispatch_alpha_at_opt

ctypedef cnp.float64_t f64
ctypedef cnp.int64_t i64
ctypedef cnp.int32_t i32
ctypedef cnp.uint8_t u8


# ---------------------------------------------------------------------------
# Best-split sweep across all features for a single node.
#
# Returns (best_feat, best_threshold, best_gain) via out parameters.
# Sets best_feat = -1 if no valid split exists.

cdef void _best_split_at_node(
    f64[:, ::1] Xt,        # (p, n_aug) column-major features
    i32[:, ::1] order,     # (p, n_aug) per-feature sorted row indices
    f64[::1] D,
    f64[::1] C,
    Py_ssize_t start,
    Py_ssize_t end,
    int n_features,
    int loss_kind,
    double bounded_lo,
    double bounded_hi,
    int min_orig_leaf,
    f64[::1] svals,    # scratch, size >= n_node
    f64[::1] cum_D,    # scratch
    f64[::1] cum_C,    # scratch
    i64[::1] cum_orig, # scratch
    int* best_feat_out,
    Py_ssize_t* best_k_out,
    double* best_threshold_out,
    double* best_gain_out,
) noexcept nogil:
    """Best split across all features for the node's rows.

    Per-feature ``total_D`` / ``total_C`` / ``parent_loss`` are recomputed
    from the sort-order cumulative sums (``cum_*[n_node - 1]``), as in the
    per-node Cython splitter, so gains compare the same way on both paths.
    ``best_k_out`` is the last left-child position in the winning
    feature's sorted slice.
    """
    cdef Py_ssize_t n_node = end - start
    cdef Py_ssize_t k
    cdef i32 row
    cdef int j
    cdef double total_D, total_C, parent_loss
    cdef i64 total_orig
    cdef double D_l, C_l, D_r, C_r, L_l, L_r, gain
    cdef i64 n_l, n_r
    cdef Py_ssize_t feat_best_k
    cdef double feat_best_gain, feat_best_thr
    cdef double best_gain_overall = -INFINITY
    cdef int best_feat = -1
    cdef Py_ssize_t best_k = -1
    cdef double best_threshold = 0.0

    for j in range(n_features):
        # Gather the node's rows in this feature's sorted order, with
        # cumulative sums.
        D_l = 0.0
        C_l = 0.0
        n_l = 0
        for k in range(n_node):
            row = order[j, start + k]
            svals[k] = Xt[j, row]
            D_l += D[row]
            C_l += C[row]
            if D[row] > 0.0:
                n_l += 1
            cum_D[k] = D_l
            cum_C[k] = C_l
            cum_orig[k] = n_l

        total_D = cum_D[n_node - 1]
        total_C = cum_C[n_node - 1]
        total_orig = cum_orig[n_node - 1]
        parent_loss = dispatch_leaf_loss(
            loss_kind, total_D, total_C, bounded_lo, bounded_hi
        )

        # Sweep candidate split positions.
        feat_best_gain = -INFINITY
        feat_best_k = -1
        feat_best_thr = 0.0
        for k in range(n_node - 1):
            if svals[k] == svals[k + 1]:
                continue
            D_l = cum_D[k]
            C_l = cum_C[k]
            n_l = cum_orig[k]
            n_r = total_orig - n_l
            if n_l < min_orig_leaf or n_r < min_orig_leaf:
                continue
            D_r = total_D - D_l
            C_r = total_C - C_l
            L_l = dispatch_leaf_loss(loss_kind, D_l, C_l, bounded_lo, bounded_hi)
            L_r = dispatch_leaf_loss(loss_kind, D_r, C_r, bounded_lo, bounded_hi)
            if not isfinite(L_l) or not isfinite(L_r):
                continue
            gain = parent_loss - L_l - L_r
            if gain > feat_best_gain:
                feat_best_gain = gain
                feat_best_k = k
                feat_best_thr = 0.5 * (svals[k] + svals[k + 1])
                # Rounding can land the midpoint on the right value (or
                # overflow); fall back to the left value, as sklearn does.
                if feat_best_thr == svals[k + 1] or not isfinite(feat_best_thr):
                    feat_best_thr = svals[k]

        if feat_best_k >= 0 and feat_best_gain > best_gain_overall:
            best_gain_overall = feat_best_gain
            best_feat = j
            best_k = feat_best_k
            best_threshold = feat_best_thr

    best_feat_out[0] = best_feat
    best_k_out[0] = best_k
    best_threshold_out[0] = best_threshold
    best_gain_out[0] = best_gain_overall


# ---------------------------------------------------------------------------
# Split a node's slice in every feature's sorted order.

cdef void _partition_all_features(
    i32[:, ::1] order,
    Py_ssize_t start,
    Py_ssize_t mid,
    Py_ssize_t end,
    int n_features,
    int best_feat,
    u8[::1] goes_left,   # scratch, size n_aug
    i32[::1] right_buf,  # scratch, size >= n_node
) noexcept nogil:
    """Stable-partition ``order[j, start:end]`` for every feature ``j`` so the
    left child's rows come first, each side keeping its sorted order. The
    winning feature's slice is already split at ``mid`` (its first
    ``mid - start`` rows are ``<= threshold``)."""
    cdef Py_ssize_t i, k_l, k_r
    cdef int j
    cdef i32 row
    for i in range(start, end):
        goes_left[order[best_feat, i]] = 1 if i < mid else 0
    for j in range(n_features):
        if j == best_feat:
            continue
        k_l = start
        k_r = 0
        for i in range(start, end):
            row = order[j, i]
            if goes_left[row]:
                order[j, k_l] = row
                k_l += 1
            else:
                right_buf[k_r] = row
                k_r += 1
        for i in range(k_r):
            order[j, k_l + i] = right_buf[i]


# ---------------------------------------------------------------------------
# Driver entry point.

def grow_depthwise_exact_c(
    cnp.ndarray[f64, ndim=2, mode="c"] features,
    cnp.ndarray[f64, ndim=1] D,
    cnp.ndarray[f64, ndim=1] C,
    int max_depth,
    int min_samples_split,
    int min_orig_leaf,
    double min_impurity_decrease,
    int loss_kind,
    double bounded_lo,
    double bounded_hi,
):
    """Iterative depthwise grow on the exact splitter.

    Returns a :class:`riesztree.fast._grow_c.GrowableFlatTree` with one
    slot per node. The Python ``Node`` tree is rebuilt once at fit-end
    via :func:`riesztree.fast._tree.node_from_growable_flat_tree`.
    """
    from ._grow_c import GrowableFlatTree

    cdef Py_ssize_t n_aug = D.shape[0]
    cdef int n_features = features.shape[1]
    if n_aug >= 2**31 - 1:
        raise ValueError(
            f"splitter='exact' indexes rows with int32; got {n_aug} augmented "
            "rows. Use splitter='hist'."
        )

    # Node cap: a depth-d tree has at most 2^(d+1) - 1 nodes, and every
    # leaf is non-empty, so there are at most 2 * n_aug - 1 nodes.
    cdef Py_ssize_t max_nodes_cap = 2 * n_aug + 1
    # Shift in Py_ssize_t: a C int overflows at max_depth = 30.
    if max_depth < 31 and ((<Py_ssize_t>1) << (max_depth + 1)) + 1 < max_nodes_cap:
        max_nodes_cap = ((<Py_ssize_t>1) << (max_depth + 1)) + 1

    cdef object tree = GrowableFlatTree(max_nodes_cap)
    cdef i32[::1] tree_feature = tree.feature
    cdef f64[::1] tree_threshold = tree.threshold
    cdef i32[::1] tree_left = tree.left
    cdef i32[::1] tree_right = tree.right
    cdef f64[::1] tree_value = tree.value
    cdef f64[::1] tree_D_sum = tree.D_sum
    cdef f64[::1] tree_C_sum = tree.C_sum
    cdef i64[::1] tree_n_orig = tree.n_orig
    cdef i64[::1] tree_n_aug = tree.n_aug
    cdef f64[::1] tree_gain_v = tree.gain
    cdef i32[::1] tree_depth = tree.depth

    # Column-major features and per-feature sorted row orders (stable, so
    # ties keep row order). Sorted one feature at a time so the int64
    # argsort output is never held for all features at once.
    cdef cnp.ndarray[f64, ndim=2] Xt_arr = np.ascontiguousarray(features.T)
    cdef cnp.ndarray[i32, ndim=2] order_arr = np.empty((n_features, n_aug), dtype=np.int32)
    cdef int jj
    for jj in range(n_features):
        order_arr[jj] = np.argsort(Xt_arr[jj], kind="stable")
    cdef f64[:, ::1] Xt_v = Xt_arr
    cdef i32[:, ::1] order_v = order_arr
    cdef f64[::1] D_v = D
    cdef f64[::1] C_v = C

    # Scratch buffers, allocated once and reused across nodes. Sized to
    # the largest node (the root).
    cdef f64[::1] svals_v = np.empty(n_aug, dtype=np.float64)
    cdef f64[::1] cum_D_v = np.empty(n_aug, dtype=np.float64)
    cdef f64[::1] cum_C_v = np.empty(n_aug, dtype=np.float64)
    cdef i64[::1] cum_orig_v = np.empty(n_aug, dtype=np.int64)
    cdef u8[::1] goes_left_v = np.empty(n_aug, dtype=np.uint8)
    cdef i32[::1] right_buf_v = np.empty(n_aug, dtype=np.int32)

    # Per-loop scalars.
    cdef Py_ssize_t i
    cdef i32 row
    cdef double root_D = 0.0
    cdef double root_C = 0.0
    cdef i64 root_orig = 0
    cdef double root_alpha
    cdef Py_ssize_t root_idx
    cdef Py_ssize_t node_idx, start, end
    cdef int depth_v, best_feat
    cdef Py_ssize_t best_k
    cdef double best_threshold, best_gain
    cdef Py_ssize_t mid, left_idx, right_idx, n_left, n_right
    cdef double left_D, right_D, left_C, right_C, left_alpha, right_alpha
    cdef i64 left_orig, right_orig

    # Bootstrap the root.
    for i in range(n_aug):
        root_D += D_v[i]
        root_C += C_v[i]
        if D_v[i] > 0.0:
            root_orig += 1
    root_alpha = dispatch_alpha_at_opt(loss_kind, root_D, root_C, bounded_lo, bounded_hi)

    root_idx = tree.n_nodes_used
    if root_idx >= tree.max_nodes:
        raise RuntimeError(
            f"GrowableFlatTree exhausted at {tree.max_nodes} nodes."
        )
    tree_feature[root_idx] = -1
    tree_value[root_idx] = root_alpha
    tree_D_sum[root_idx] = root_D
    tree_C_sum[root_idx] = root_C
    tree_n_orig[root_idx] = root_orig
    tree_n_aug[root_idx] = n_aug
    tree_depth[root_idx] = 0
    tree.n_nodes_used += 1

    # Worklist entries: (node_idx, start, end, depth).
    cdef list worklist = [(root_idx, 0, n_aug, 0)]
    cdef object item

    while worklist:
        item = worklist.pop()
        node_idx = item[0]
        start = item[1]
        end = item[2]
        depth_v = item[3]

        # Stopping conditions (mirror Python ``_recurse``).
        if depth_v >= max_depth:
            continue
        if tree_n_orig[node_idx] < min_samples_split:
            continue
        if (end - start) < 2:
            continue

        _best_split_at_node(
            Xt_v, order_v, D_v, C_v, start, end, n_features,
            loss_kind, bounded_lo, bounded_hi, min_orig_leaf,
            svals_v, cum_D_v, cum_C_v, cum_orig_v,
            &best_feat, &best_k, &best_threshold, &best_gain,
        )
        if best_feat < 0:
            continue
        if best_gain <= min_impurity_decrease:
            continue

        # The winning feature's sorted slice splits after position best_k:
        # its first rows are exactly those with x <= threshold.
        mid = start + best_k + 1
        n_left = mid - start
        n_right = end - mid
        _partition_all_features(
            order_v, start, mid, end, n_features, best_feat,
            goes_left_v, right_buf_v,
        )

        # Child payloads, summed directly over each child's rows.
        left_D = 0.0
        left_C = 0.0
        left_orig = 0
        for i in range(start, mid):
            row = order_v[0, i]
            left_D += D_v[row]
            left_C += C_v[row]
            if D_v[row] > 0.0:
                left_orig += 1
        right_D = 0.0
        right_C = 0.0
        right_orig = 0
        for i in range(mid, end):
            row = order_v[0, i]
            right_D += D_v[row]
            right_C += C_v[row]
            if D_v[row] > 0.0:
                right_orig += 1

        left_alpha = dispatch_alpha_at_opt(
            loss_kind, left_D, left_C, bounded_lo, bounded_hi
        )
        right_alpha = dispatch_alpha_at_opt(
            loss_kind, right_D, right_C, bounded_lo, bounded_hi
        )

        # Append left and right child leaves to the tree.
        left_idx = tree.n_nodes_used
        if left_idx + 1 >= tree.max_nodes:
            raise RuntimeError(
                f"GrowableFlatTree exhausted at {tree.max_nodes} nodes; "
                "increase the per-fit cap."
            )
        right_idx = left_idx + 1
        tree_feature[left_idx] = -1
        tree_value[left_idx] = left_alpha
        tree_D_sum[left_idx] = left_D
        tree_C_sum[left_idx] = left_C
        tree_n_orig[left_idx] = left_orig
        tree_n_aug[left_idx] = n_left
        tree_depth[left_idx] = depth_v + 1
        tree_feature[right_idx] = -1
        tree_value[right_idx] = right_alpha
        tree_D_sum[right_idx] = right_D
        tree_C_sum[right_idx] = right_C
        tree_n_orig[right_idx] = right_orig
        tree_n_aug[right_idx] = n_right
        tree_depth[right_idx] = depth_v + 1
        tree.n_nodes_used += 2

        # Convert the parent slot from leaf to internal.
        tree_feature[node_idx] = best_feat
        tree_threshold[node_idx] = best_threshold
        tree_left[node_idx] = <i32>left_idx
        tree_right[node_idx] = <i32>right_idx
        tree_gain_v[node_idx] = best_gain

        # Push children. Append right first so .pop() processes left first
        # (matches the Python grower's depth-first, left-first order).
        worklist.append((right_idx, mid, end, depth_v + 1))
        worklist.append((left_idx, start, mid, depth_v + 1))

    return tree
