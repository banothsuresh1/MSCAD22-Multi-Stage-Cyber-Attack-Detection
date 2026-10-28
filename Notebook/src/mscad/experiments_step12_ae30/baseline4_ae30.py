r"""
src/mscad/experiments_step12_ae30/baseline4_ae30.py -- Baseline 4 (AE -> PCA -> LSTM) at AE latent 30.

NEW FILE.  No training code: it calls experiments_pca.run_baseline4, the same function the AE=15
run uses, inside `common_ae30.ae30()` so the bottleneck is
config.AE_LATENT_DIM_AE30 and every output lands under
config.OUTPUTS_ROOT_AE30.

PCA keeps config.PCA_N_COMPONENTS = 0.95, a VARIANCE TARGET.  The
number of components it keeps out of 30 is whatever reaches 95% of the
variance -- it is read from the run, never assumed.
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
        from ..experiments_step12 import baseline4 as sibling
    except ImportError:                                     # pragma: no cover
        return None
    return sibling


_SIBLING = _ae15()

KEY = "baseline4_ae_pca_lstm"
# the AE=15 wrapper's own declaration of what this configuration is, with the
# one thing this run changes overridden
COMPONENTS = ({} if _SIBLING is None else
              dict(_SIBLING.COMPONENTS,
                   **{"autoencoder": "yes, trained here at latent %d"
                                     % c30.latent_dim()}))
NOTE = (
    "PCA keeps the 0.95 variance target, so the component count out of 30 is "
    "whatever reaches 95% of the variance -- read it from the row, it is not "
    "a setting."
)


def train(verbose: bool = True) -> dict[str, Any]:
    """`experiments_pca.run_baseline4`, unchanged."""
    from .. import experiments_pca as xpca

    inputs = c30.inputs()
    return xpca.run_baseline4(inputs["y_train"], inputs["y_test"],
                              inputs["meta_test"], inputs["label_mapping"],
                              duplicate_flags=inputs["duplicate_flags"],
                              overwrite=True, verbose=verbose)

def run(verbose: bool = True) -> dict[str, Any]:
    return c30.run_one_ae30(KEY, train, components=COMPONENTS, note=NOTE,
                            verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
