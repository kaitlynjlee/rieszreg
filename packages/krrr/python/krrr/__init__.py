"""krrr: kernel ridge Riesz regression.

A learner package in the RieszReg family. Estimates the Riesz representer α
of a linear estimand ψ = E[m(μ)(Z)] using kernel ridge regression.

Reuses ``rieszreg``'s ``Estimand``, ``Tracer``, ``AugmentedDataset``,
``Loss``, ``Diagnostics``, and sklearn glue. The kernel solve plugs in
through the ``Backend`` Protocol from ``rieszreg``.

    from krrr import KernelRieszRegressor, ATE

    krr = KernelRieszRegressor(estimand=ATE(treatment="a"))
    krr.fit(Z)                 # Z: treatment column + covariates
    alpha_hat = krr.predict(Z)
    print(krr.diagnose(Z).summary())

Every estimand, loss and diagnostic from ``rieszreg`` is re-exported here.
"""

from rieszreg.user_api import *  # noqa: F401,F403
from rieszreg.user_api import __all__ as _shared

from .backend import KernelRidgeBackend
from .diagnostics import KernelDiagnostics
from .estimator import KernelRieszRegressor
from .kernels import (
    Gaussian,
    Kernel,
    Linear,
    Matern,
    Polynomial,
    Product,
    Scaled,
    Sum,
    Tensor,
)

__all__ = [
    *_shared,
    "KernelRieszRegressor",
    "KernelRidgeBackend",
    "KernelDiagnostics",
    "Kernel",
    "Gaussian",
    "Matern",
    "Linear",
    "Polynomial",
    "Tensor",
    "Sum",
    "Product",
    "Scaled",
]
