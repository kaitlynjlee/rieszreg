"""Sklearn-compatible orchestrator for Riesz representer estimation.

`RieszEstimator` takes (estimand, loss, backend) at construction and implements
the standard sklearn `fit / predict / score` API. Learner-specific
hyperparameters live on subclasses (e.g. `RieszBooster` in `rieszboost` adds
`max_depth`, `reg_lambda`, `subsample`).

Composes with `sklearn.model_selection.GridSearchCV`, `cross_val_predict`,
`clone`, `Pipeline`, etc.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from ._omp import warn_if_multi_backend_omp
from .backends import Backend, load_predictor
from .estimands.base import Estimand, FiniteEvalEstimand, estimand_from_spec
from .estimands.tracer import trace
from .losses import Loss, SquaredLoss, loss_from_spec


def _is_dataframe(Z) -> bool:
    return hasattr(Z, "columns") and hasattr(Z, "iloc")


def _missing_columns_message(estimand: Estimand, missing, columns) -> str:
    return (
        f"The data is missing columns {missing} needed by {estimand.name}; "
        f"it has columns {list(columns)}. Tell the estimand your column "
        "names, e.g. ATE(treatment=\"treated\", covariates=[\"age\", \"income\"])."
    )


def _n_columns(Z) -> int:
    if _is_dataframe(Z):
        return Z.shape[1]
    arr = np.asarray(Z)
    return 1 if arr.ndim == 1 else arr.shape[1]


def _rows_from_Z(Z, estimand: Estimand) -> list[dict]:
    """Convert ndarray or DataFrame Z into a list of row-dicts keyed by
    `estimand.feature_keys`. Ndarray input is interpreted column-by-column in
    `feature_keys` order; DataFrame columns are matched by name.

    Hot path during fit (the augmentation engine consumes row-dicts).
    Predict-only paths should use :func:`_features_from_Z` instead, which
    skips the row-dict pivot entirely.
    """
    if _is_dataframe(Z):
        cols_needed = list(estimand.feature_keys)
        missing = [c for c in cols_needed if c not in Z.columns]
        if missing:
            raise ValueError(_missing_columns_message(estimand, missing, Z.columns))
        # Vectorise the per-column extraction: one .to_numpy() per column
        # rather than O(n*p) .iloc lookups. The downstream consumers see
        # the same list-of-dicts shape.
        col_arrs = {k: Z[k].to_numpy() for k in cols_needed}
        n = len(Z)
        return [
            {k: col_arrs[k][i] for k in cols_needed}
            for i in range(n)
        ]

    arr = np.asarray(Z)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != len(estimand.feature_keys):
        raise ValueError(
            f"Estimand {estimand.name!r} expects {len(estimand.feature_keys)} "
            f"feature columns ({estimand.feature_keys}), got Z.shape[1]="
            f"{arr.shape[1]}."
        )
    return [
        {k: arr[i, j] for j, k in enumerate(estimand.feature_keys)}
        for i in range(arr.shape[0])
    ]


def _features_from_Z(Z, estimand: Estimand) -> np.ndarray:
    """Fast DataFrame/ndarray → ``feature_keys``-ordered float ndarray.

    Skips the ``list[dict]`` pivot used by the fit path. Used by the
    predict-only entry points (where the augmentation engine isn't
    needed) so prediction stays at numpy / Cython speed end-to-end.
    """
    if _is_dataframe(Z):
        cols_needed = list(estimand.feature_keys)
        missing = [c for c in cols_needed if c not in Z.columns]
        if missing:
            raise ValueError(_missing_columns_message(estimand, missing, Z.columns))
        return Z[cols_needed].to_numpy(dtype=float)

    arr = np.asarray(Z, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != len(estimand.feature_keys):
        raise ValueError(
            f"Estimand {estimand.name!r} expects {len(estimand.feature_keys)} "
            f"feature columns ({estimand.feature_keys}), got Z.shape[1]="
            f"{arr.shape[1]}."
        )
    return arr


def _ys_from_y(y, n: int) -> list | None:
    """Coerce `y` (a sklearn-style outcome vector) into a list of per-row
    scalars aligned with the rows. Returns None when `y is None`. Raises if
    the length doesn't match `n`."""
    if y is None:
        return None
    if hasattr(y, "to_numpy"):
        y_arr = y.to_numpy()
    else:
        y_arr = np.asarray(y)
    if y_arr.ndim > 1:
        y_arr = y_arr.reshape(-1)
    if len(y_arr) != n:
        raise ValueError(
            f"len(y)={len(y_arr)} does not match number of rows in Z ({n})."
        )
    return list(y_arr)


def _features_from_rows(rows: Sequence[dict], estimand: Estimand) -> np.ndarray:
    return np.asarray(
        [[row[k] for k in estimand.feature_keys] for row in rows], dtype=float
    )


def _split_Z(Z, y, validation_fraction: float, random_state: int):
    """Split (Z, y) into train/valid by `validation_fraction`. `y=None` is
    threaded through unchanged. Returns `(Z_train, Z_valid, y_train, y_valid)`
    where the validation halves are `None` when `validation_fraction <= 0`."""
    n = len(Z) if _is_dataframe(Z) else len(np.asarray(Z))
    if validation_fraction <= 0:
        return Z, None, y, None
    idx = np.arange(n)
    tr_idx, va_idx = train_test_split(
        idx, test_size=validation_fraction, random_state=random_state
    )
    if _is_dataframe(Z):
        Z_train, Z_valid = Z.iloc[tr_idx], Z.iloc[va_idx]
    else:
        arr = np.asarray(Z)
        Z_train, Z_valid = arr[tr_idx], arr[va_idx]
    if y is None:
        return Z_train, Z_valid, None, None
    if hasattr(y, "iloc"):
        y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]
    else:
        y_arr = np.asarray(y)
        y_train, y_valid = y_arr[tr_idx], y_arr[va_idx]
    return Z_train, Z_valid, y_train, y_valid


class RieszEstimator(BaseEstimator):
    """Estimate the Riesz representer α₀ of a causal estimand.

    Most users start from a learner package's subclass, which picks the
    backend for you: `RieszBooster` (rieszboost), `KernelRieszRegressor`
    (krrr), `ForestRieszRegressor` / `AugForestRieszRegressor` (forestriesz),
    `RieszNet` (riesznet), `RieszTreeRegressor` (riesztree). Use
    `RieszEstimator` directly to pair an estimand with a backend object.

    Parameters
    ----------
    estimand : Estimand
        What you want to estimate, e.g. ``ATE(treatment="treated")``. The
        estimand also names the treatment and covariate columns of ``X``.
    backend : Backend
        The learner that fits α̂ (e.g. ``XGBoostBackend()``). Required here;
        the learner-package subclasses supply one for you.
    loss : Loss, default=None
        The Bregman-Riesz loss to minimize. Defaults to `SquaredLoss()`.
    init : float or None
        α-space initialization. ``None`` (default) sets α to the constant
        that minimizes the empirical Riesz loss — namely ``m̄ = E[m(Z, 1)]``
        on the training rows, projected into the loss's α-domain. Pass an
        explicit float to override (e.g. ``init=0`` for hard-zero start).
    random_state : int, default=0
        Seed for every source of randomness in the fit (validation split,
        subsampling, weight initialization).

    Attributes
    ----------
    estimand_ : Estimand
        The estimand with its columns resolved against the training data.
    n_features_in_ : int
        Number of columns α̂ is a function of (treatment + covariates).
    feature_names_in_ : ndarray of str
        Those column names, in the order α̂ uses them.
    loss_ : Loss
        The loss used for fitting.
    """

    def __init__(
        self,
        estimand: Estimand,
        backend: Backend | None = None,
        loss: Loss | None = None,
        init: float | None = None,
        random_state: int = 0,
    ):
        self.estimand = estimand
        self.backend = backend
        self.loss = loss
        self.init = init
        self.random_state = random_state

    # ---- internal accessors that resolve defaults / hyperparams ----

    def _resolved_backend(self) -> Backend:
        if self.backend is None:
            raise ValueError(
                "RieszEstimator requires a `backend=`. Either pass one explicitly "
                "or use a subclass (e.g. RieszBooster) that bakes in a default."
            )
        return self.backend

    def _resolved_loss(self) -> Loss:
        return self.loss if self.loss is not None else SquaredLoss()

    def _backend_hyperparams(self) -> dict:
        """Backend-specific hyperparameters routed via `hyperparams=`. Subclasses
        override to surface their own knobs (max_depth, reg_lambda, etc.)."""
        return {}

    # ---- sklearn API ----

    def fit(self, Z, y=None, eval_set=None, eval_y=None) -> "RieszEstimator":
        """Fit the Riesz representer.

        Parameters
        ----------
        Z : DataFrame or ndarray of shape (n, p)
            The treatment column plus covariates (sklearn's ``X``). With a
            DataFrame, columns are matched by the names the estimand was
            given; with an ndarray, the treatment is column 0.
        y : array-like of shape (n,), optional
            The outcome. The built-in treatment estimands (``ATE``, ``ATT``,
            ``TSM``, ``AdditiveShift``, ``LocalShift``) do not use it — the
            Riesz representer depends only on treatment and covariates — so
            passing it is harmless. Estimands whose functional reads the
            outcome (``OutcomeRegNormSq``, custom Y-dependent ones) need it.
        eval_set : DataFrame or ndarray, optional
            Held-out rows for early stopping / λ selection, when the learner
            uses one. Overrides the learner's internal ``validation_fraction``.
        eval_y : array-like, optional
            Outcome for the ``eval_set`` rows.
        """
        warn_if_multi_backend_omp()
        loss = self._resolved_loss()
        backend = self._resolved_backend()

        if not isinstance(self.estimand, FiniteEvalEstimand):
            raise TypeError(
                f"estimand must be a built-in estimand (ATE, ATT, TSM, "
                f"AdditiveShift, LocalShift, OutcomeRegNormSq) or a "
                f"FiniteEvalEstimand(feature_keys=..., m=...); got "
                f"{type(self.estimand).__name__}."
            )
        estimand = self.estimand.bind(
            list(Z.columns) if _is_dataframe(Z) else _n_columns(Z)
        )

        # Resolve validation slice. Backends that use a held-out slice for
        # fit-time logic (early stopping, λ selection) expose
        # `validation_fraction` as a constructor attribute; the orchestrator
        # reads it via getattr and performs the split before augmentation.
        val_frac = float(getattr(backend, "validation_fraction", 0.0) or 0.0)
        if eval_set is not None:
            Z_train, Z_valid = Z, eval_set
            y_train, y_valid = y, eval_y
        elif val_frac > 0:
            Z_train, Z_valid, y_train, y_valid = _split_Z(
                Z, y, val_frac, self.random_state
            )
        else:
            Z_train, Z_valid = Z, None
            y_train, y_valid = y, None

        # Augmentation: dispatch via `estimand.augment(features, ys)`.
        # Built-in subclasses override with vectorised numpy; custom estimands
        # use the inherited Tracer-based default.
        feats_train = _features_from_Z(Z_train, estimand)
        n_train = feats_train.shape[0]
        ys_train = _ys_from_y(y_train, n_train)
        aug_train = estimand.augment(feats_train, ys=ys_train)

        if Z_valid is not None and len(Z_valid) > 0:
            feats_valid = _features_from_Z(Z_valid, estimand)
            ys_valid = _ys_from_y(y_valid, feats_valid.shape[0])
            aug_valid = estimand.augment(feats_valid, ys=ys_valid)
        else:
            feats_valid = None
            ys_valid = None
            aug_valid = None

        # Resolve init in α-space, then convert to η.
        # Default (init=None): use the constant that minimizes the empirical
        # Riesz loss. For any Bregman loss with strictly convex φ this is
        # m̄ = E[m(Z, 1)] (FOC ψ'(a) = φ''(a)·m̄ collapses to a = m̄ via
        # ψ'(t) = t·φ''(t)). Each loss projects m̄ into its α-domain.
        # Built-in subclasses carry the closed-form m_bar as a class attribute;
        # custom estimands fall back to a per-row trace.
        init_arg = self.init
        if init_arg is None:
            mbar_builtin = estimand.m_bar
            if mbar_builtin is not None and ys_train is None:
                m_bar = float(mbar_builtin)
            else:
                rows_train = _rows_from_Z(Z_train, estimand)
                if ys_train is None:
                    m_bar = float(np.mean(
                        [sum(c for c, _ in trace(estimand, z)) for z in rows_train]
                    ))
                else:
                    m_bar = float(np.mean(
                        [
                            sum(c for c, _ in trace(estimand, z, y_i))
                            for z, y_i in zip(rows_train, ys_train)
                        ]
                    ))
            init_alpha = loss.best_constant_init(m_bar)
        elif isinstance(init_arg, (int, float)):
            init_alpha = float(init_arg)
        else:
            raise ValueError(f"init must be float or None; got {init_arg!r}")
        base_score = float(loss.alpha_to_eta(init_alpha))

        common_kwargs = dict(
            base_score=base_score,
            random_state=self.random_state,
            hyperparams=self._backend_hyperparams(),
        )

        # Dispatch: moment-style backends consume rows + estimand directly.
        # Augmentation-style backends receive a precomputed AugmentedDataset.
        # Backends implementing both default to fit_augmented for back-compat.
        uses_moment_path = hasattr(backend, "fit_rows") and not hasattr(backend, "fit_augmented")
        if uses_moment_path:
            # Moment backends consume row-dicts; materialise from the feature
            # ndarrays we already built.
            rows_train = _rows_from_Z(Z_train, estimand)
            rows_valid = (
                _rows_from_Z(Z_valid, estimand)
                if Z_valid is not None and len(Z_valid) > 0
                else None
            )
            result = backend.fit_rows(
                rows_train,
                rows_valid,
                estimand,
                loss,
                ys_train=ys_train,
                ys_valid=ys_valid,
                **common_kwargs,
            )
        else:
            result = backend.fit_augmented(aug_train, aug_valid, loss, **common_kwargs)

        self.predictor_ = result.predictor
        self.best_iteration_ = result.best_iteration
        self.best_score_ = result.best_score
        self.base_score_ = base_score
        self.loss_ = loss
        self.estimand_ = estimand
        self.n_features_in_ = len(estimand.feature_keys)
        self.feature_names_in_ = np.asarray(estimand.feature_keys, dtype=object)
        return self

    def _features(self, Z) -> np.ndarray:
        """Check the estimator is fitted and pull α̂'s input columns from Z."""
        check_is_fitted(self, "predictor_")
        return _features_from_Z(Z, self.estimand_)

    def predict(self, Z) -> np.ndarray:
        """Return α̂ evaluated at each row of Z, shape ``(n,)``."""
        feats = self._features(Z)
        return self.predictor_.predict_alpha(feats)

    def _loss_on(self, Z, y, loss: Loss) -> float:
        """Mean per-row Riesz loss of the fitted α̂ on (Z, y) under ``loss``."""
        feats = self._features(Z)
        aug = self.estimand_.augment(feats, ys=_ys_from_y(y, feats.shape[0]))
        alpha = self.loss_.link_to_alpha(self.predictor_.predict_eta(aug.features))
        return float(
            np.sum(loss.aug_loss_alpha(aug.is_original, aug.potential_deriv_coef, alpha))
            / aug.n_rows
        )

    def riesz_loss(self, Z, y=None) -> float:
        """Mean per-row Riesz loss on (Z, y) under the loss the estimator was
        trained with. Lower is better. Pass ``y`` only when the estimand's
        functional reads the outcome."""
        return self._loss_on(Z, y, self.loss_)

    def score(self, Z, y=None) -> float:
        """Return negative held-out canonical Riesz loss (squared loss).

        Following sklearn convention (R² for regressors, accuracy for
        classifiers), `score()` evaluates a fixed yardstick — squared Riesz
        loss — independent of the loss the estimator was trained with. This
        makes `cross_val_score` and `GridSearchCV` results comparable across
        estimators fit with different losses (e.g. KL vs squared).

        Pass `scoring=riesz_scorer(loss=...)` to sklearn CV utilities to use a
        different yardstick. `riesz_loss(Z)` is the own-loss diagnostic.
        `y` is plumbed into `m(alpha)(z, y)` for Y-dependent estimands.
        """
        return -self._loss_on(Z, y, SquaredLoss())

    def diagnose(self, Z, **kwargs):
        """Health checks on α̂ over Z: magnitude, extreme values (a sign of
        poor overlap / near-positivity violations), and held-out Riesz loss.
        Returns a `Diagnostics` object; call ``.summary()`` for a report.
        Keyword arguments (``extreme_threshold``, ``extreme_fraction_warn``)
        are forwarded to `rieszreg.diagnose`."""
        from .diagnostics import diagnose
        return diagnose(estimator=self, Z=Z, **kwargs)

    # ---- serialization ----

    def save(self, path) -> None:
        """Save a fitted estimator to a directory.

        Writes:
          - the predictor's binary payload (whatever format the backend uses —
            native model files, joblib pickles, torch state_dicts, etc.) via
            `predictor.save(dir_path)`
          - `metadata.json` with the loss spec, estimand factory_spec (if
            built-in), feature_keys, base_score, best_iteration_, and the
            estimator's constructor hyperparameters.

        Custom (non-built-in) estimands cannot be auto-reconstructed; the file
        will save fine, but `.load(path)` will require the user to pass
        `estimand=...` explicitly.
        """
        import json
        from pathlib import Path

        check_is_fitted(self, "predictor_")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.predictor_.save(path)

        metadata = {
            "rieszreg_format_version": 1,
            "predictor_kind": self.predictor_.kind,
            "loss": self.loss_.to_spec(),
            "estimand_factory_spec": self.estimand_.factory_spec,  # None if custom
            "feature_keys": list(self.estimand_.feature_keys),
            "base_score": self.base_score_,
            "best_iteration": self.best_iteration_,
            "best_score": self.best_score_,
            "estimator_class": type(self).__name__,
            "hyperparameters": self._save_hyperparameters(),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_hyperparameters(self) -> dict:
        """Snapshot of constructor args for round-trip. Subclasses extend."""
        return {
            "init": self.init,
            "random_state": self.random_state,
        }

    @classmethod
    def load(cls, path, *, estimand: Estimand | None = None) -> "RieszEstimator":
        """Load an estimator from a directory written by `save(...)`.

        For custom (non-built-in) estimands, pass `estimand=` to inject the
        original Estimand instance. For built-ins, reconstruction is automatic.

        The predictor binary is loaded via the registered predictor-loader for
        its `kind` (see `rieszreg.backends.register_predictor_loader`).
        Implementation packages register loaders at import time, so importing
        the relevant package (e.g. `import rieszboost`) is enough.
        """
        import json
        from pathlib import Path

        path = Path(path)
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        loss = loss_from_spec(metadata["loss"])

        if estimand is None:
            spec = metadata.get("estimand_factory_spec")
            if spec is None:
                raise ValueError(
                    f"Saved estimator at {path} has a custom (non-built-in) "
                    "estimand. Pass `estimand=...` explicitly to "
                    f"{cls.__name__}.load(path, estimand=my_estimand)."
                )
            estimand = estimand_from_spec(spec)
        # A passed-in built-in like ATE() has covariates=None; resolve it
        # against the columns the model was trained on.
        estimand = estimand.bind(metadata["feature_keys"])
        if tuple(estimand.feature_keys) != tuple(metadata["feature_keys"]):
            raise ValueError(
                f"The estimand passed to load() uses columns "
                f"{list(estimand.feature_keys)}, but the model at {path} was "
                f"fit on {metadata['feature_keys']}."
            )

        predictor = load_predictor(
            metadata["predictor_kind"],
            path,
            base_score=metadata["base_score"],
            loss=loss,
            best_iteration=metadata.get("best_iteration"),
        )

        hp = metadata.get("hyperparameters", {})
        instance = cls._construct_for_load(estimand=estimand, loss=loss, hyperparameters=hp)
        instance.predictor_ = predictor
        instance.best_iteration_ = metadata.get("best_iteration")
        instance.best_score_ = metadata.get("best_score")
        instance.base_score_ = metadata["base_score"]
        instance.loss_ = loss
        instance.estimand_ = estimand
        instance.n_features_in_ = len(metadata["feature_keys"])
        instance.feature_names_in_ = np.asarray(metadata["feature_keys"], dtype=object)
        return instance

    @classmethod
    def _construct_for_load(cls, *, estimand, loss, hyperparameters: dict):
        """Build an unfit instance from saved hyperparameters. Subclasses
        override to consume their additional knobs."""
        return cls(
            estimand=estimand,
            loss=loss,
            init=hyperparameters.get("init"),
            random_state=hyperparameters.get("random_state", 0),
        )
