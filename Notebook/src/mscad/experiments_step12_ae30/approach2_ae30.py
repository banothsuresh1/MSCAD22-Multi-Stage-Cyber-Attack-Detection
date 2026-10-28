r"""
src/mscad/experiments_step12_ae30/approach2_ae30.py -- Approach 2 (FS -> AE -> CNN-Attention-LSTM) at AE latent 30.

NEW FILE.  No training code: it calls models_attention.run_step10b and run_step11b, the same function the AE=15
run uses, inside `common_ae30.ae30()` so the bottleneck is
config.AE_LATENT_DIM_AE30 and every output lands under
config.OUTPUTS_ROOT_AE30.

CAVEAT: Approach 2 also starts from the 30 selected features, so
AE=30 maps 30 -> 30 and compresses nothing.

It trains on the Step 8 features, so the shared Step 7/8/9 chain is
prepared first -- the same encoder Approach 1 uses, trained once.
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
        from ..experiments_step12 import approach2 as sibling
    except ImportError:                                     # pragma: no cover
        return None
    return sibling


_SIBLING = _ae15()

KEY = config.APPROACH2_KEY
# the AE=15 wrapper's own declaration of what this configuration is, with the
# one thing this run changes overridden
COMPONENTS = ({} if _SIBLING is None else
              dict(_SIBLING.COMPONENTS,
                   **{"autoencoder": "yes, trained here at latent %d"
                                     % c30.latent_dim()}))
NOTE = (
    "Approach 2 starts from 30 selected features, so AE=30 maps 30 -> 30 and "
    "compresses nothing: this row is an autoencoder with no bottleneck, not a "
    "wider bottleneck."
)


def train(verbose: bool = True) -> dict[str, Any]:
    """`models_attention.run_step10b` then `run_step11b`, both unchanged."""
    from .. import models_attention as ma

    c30.prepare_latent_features(verbose=verbose)
    ma.run_step10b(overwrite=True, verbose=verbose)
    return ma.run_step11b(overwrite=True, verbose=verbose)

def run(verbose: bool = True) -> dict[str, Any]:
    return c30.run_one_ae30(KEY, train, components=COMPONENTS, note=NOTE,
                            verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
