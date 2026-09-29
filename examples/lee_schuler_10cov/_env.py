"""Process setup, imported before numpy, xgboost or torch.

One thread per fit, because parallelism runs across (replicate, learner)
tasks. It also avoids the libomp deadlock between the macOS wheels of
xgboost and torch.

The package source directories go on sys.path and PYTHONPATH directly, so
worker processes can import the workspace packages even when macOS marks the
venv's editable-install .pth files hidden and Python skips them.
"""

import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"

REPO = Path(__file__).resolve().parents[2]
PACKAGE_DIRS = [str(REPO / "packages" / p / "python")
                for p in ("rieszreg", "rieszboost", "riesznet", "riesztree", "forestriesz")]
for p in PACKAGE_DIRS:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    PACKAGE_DIRS + [str(Path(__file__).resolve().parent)] + [os.environ.get("PYTHONPATH", "")])
