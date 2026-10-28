r"""
src/mscad/experiments_step12_ae30/baseline6_ae30.py -- Baseline 6 (AE -> PCA -> CNN-LSTM) at AE latent 30.

NEW FILE.  No training code: it calls experiments_baseline6.run_baseline6, the same function the AE=15
run uses, inside `common_ae30.ae30()` so the bottleneck is
config.AE_LATENT_DIM_AE30 and every output lands under
config.OUTPUTS_ROOT_AE30.

PCA keeps the fixed config.B6_PCA_COMPONENTS = 3, because
`models.build_cnn_lstm` refuses a sequence shorter than CNN_KERNEL.
So B6 is 67 -> 30 -> 3 here against 67 -> 15 -> 3 at AE=15.
run_baseline6 assembles its own inputs from Step 8, so the shared
Step 7/8/9 chain is prepared first.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from . import common_ae30 as c30  # noqa: E402


def _ae15():
    """The AE=15 wrapper for this configuration, or None if it is not there."""
    try:
        from ..experiments_step12 import baseline6 as sibling
    except ImportError:                                     # pragma: no cover
        return None
    return sibling


_SIBLING = _ae15()

KEY = config.BASELINE6_NAME
# the AE=15 wrapper's own declaration of what this configuration is, with the
# one thing this run changes overridden
COMPONENTS = ({} if _SIBLING is None else
              dict(_SIBLING.COMPONENTS,
                   **{"autoencoder": "yes, trained here at latent %d"
                                     % c30.latent_dim()}))
NOTE = (
    "PCA keeps its fixed 3 components (build_cnn_lstm refuses a sequence "
    "shorter than CNN_KERNEL), so this row is 67 -> 30 -> 3 against "
    "67 -> 15 -> 3 at AE=15."
)


def train(verbose: bool = True) -> dict[str, Any]:
    """`experiments_baseline6.run_baseline6`, unchanged.

    It reads the Step 8 features itself rather than taking them as arguments,
    so the shared Step 7/8/9 chain is run first -- inside the same redirect,
    and only when its outputs are missing.
    """
    from .. import experiments_baseline6 as b6

    c30.prepare_latent_features(verbose=verbose)
    return b6.run_baseline6(overwrite=True, verbose=verbose)

def run(verbose: bool = True) -> dict[str, Any]:
    return c30.run_one_ae30(KEY, train, components=COMPONENTS, note=NOTE,
                            verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
