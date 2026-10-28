r"""
src/mscad/experiments_step12_ae30/approach1_ae30.py -- Approach 1 (FS -> AE -> CNN-LSTM) at AE latent 30.

NEW FILE.  No training code: it calls autoencoder.run_step7/run_step8, imbalance.run_step9,
models.run_step10 and evaluation.run_step11, the same function the AE=15
run uses, inside `common_ae30.ae30()` so the bottleneck is
config.AE_LATENT_DIM_AE30 and every output lands under
config.OUTPUTS_ROOT_AE30.

CAVEAT: Approach 1 starts from the 30 SELECTED features, so a latent
width of 30 is NOT a bottleneck -- its autoencoder maps 30 -> 30 and
compresses nothing.

Approach 1 is not one trainer but the Step 7 -> 11 chain: the
autoencoder, the encoded features, the class weights, the classifier
and its evaluation.  At AE=30 every one of those has to be redone,
so this calls the five existing runners in order, inside the
redirect.  It cannot reuse the saved Step 7 encoder: that one is 15
wide.
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
        from ..experiments_step12 import approach1 as sibling
    except ImportError:                                     # pragma: no cover
        return None
    return sibling


_SIBLING = _ae15()

KEY = "proposed"
# the AE=15 wrapper's own declaration of what this configuration is, with the
# one thing this run changes overridden
COMPONENTS = ({} if _SIBLING is None else
              dict(_SIBLING.COMPONENTS,
                   **{"autoencoder": "yes, trained here at latent %d"
                                     % c30.latent_dim()}))
NOTE = (
    "Approach 1 starts from 30 selected features, so AE=30 maps 30 -> 30 and "
    "compresses nothing: this row is an autoencoder with no bottleneck, not a "
    "wider bottleneck."
)


def train(verbose: bool = True) -> dict[str, Any]:
    """The Step 7 -> 11 chain, each runner unchanged and in order."""
    from .. import evaluation as ev

    c30.prepare_latent_features(verbose=verbose)
    c30.train_classifier(verbose=verbose)
    return ev.run_step11(verbose=verbose)

def run(verbose: bool = True) -> dict[str, Any]:
    return c30.run_one_ae30(KEY, train, components=COMPONENTS, note=NOTE,
                            verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
