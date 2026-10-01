"""riesznet: neural-network Riesz regression (PyTorch).

`RieszNet` trains a small neural network (an MLP) to estimate the Riesz
representer α̂ of a causal estimand:

    from riesznet import RieszNet, ATE
    net = RieszNet(estimand=ATE(treatment="treated"), epochs=200)
    net.fit(Z)                   # Z: treatment column + covariates
    alpha_hat = net.predict(Z)

Custom architectures: build a ``TorchBackend`` and pass it to
``RieszEstimator(estimand, backend=...)``.

Every estimand, loss and diagnostic from ``rieszreg`` is re-exported here.
torch loads lazily, on first use of ``RieszNet`` / ``TorchBackend``.
"""

from __future__ import annotations

from rieszreg.user_api import *  # noqa: F401,F403
from rieszreg.user_api import __all__ as _shared
from rieszreg.backends import register_predictor_loader as _register

# So `RieszEstimator.load` works after `import riesznet` alone, without
# importing torch until a saved model is actually loaded.
_register("riesznet", "riesznet.backend:TorchPredictor.load")

__all__ = [*_shared, "RieszNet", "TorchBackend", "build_mlp", "build_adam"]


# Defer torch import until a torch-using symbol is accessed, so importing
# riesznet next to rieszboost doesn't map two libomp copies into one process.
_LAZY = {
    "RieszNet": ("estimator", "RieszNet"),
    "TorchBackend": ("backend", "TorchBackend"),
    "TorchPredictor": ("backend", "TorchPredictor"),
    "build_mlp": ("modules", "build_mlp"),
    "build_adam": ("modules", "build_adam"),
}


def __getattr__(name):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        from importlib import import_module
        return getattr(import_module(f"{__name__}.{mod_name}"), attr)
    raise AttributeError(f"module 'riesznet' has no attribute {name!r}")
