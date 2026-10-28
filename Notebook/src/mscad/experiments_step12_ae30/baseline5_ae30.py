r"""
src/mscad/experiments_step12_ae30/baseline5_ae30.py -- Baseline 5 (MI-only top 30 -> AE -> CNN-LSTM) at AE latent 30.

NEW FILE.  No training code: it calls experiments_baseline5.run_baseline5, the same function the AE=15
run uses, inside `common_ae30.ae30()` so the bottleneck is
config.AE_LATENT_DIM_AE30 and every output lands under
config.OUTPUTS_ROOT_AE30.

CAVEAT: Baseline 5 starts from 30 MI-selected features, so a latent
width of 30 is NOT a bottleneck -- its autoencoder maps 30 -> 30 and
compresses nothing.  The row is still meaningful, but it answers
"what does an autoencoder that cannot compress do?" rather than
"how wide should the bottleneck be?".
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
        from ..experiments_step12 import baseline5 as sibling
    except ImportError:                                     # pragma: no cover
        return None
    return sibling


_SIBLING = _ae15()

KEY = "baseline5"
# the AE=15 wrapper's own declaration of what this configuration is, with the
# one thing this run changes overridden
COMPONENTS = ({} if _SIBLING is None else
              dict(_SIBLING.COMPONENTS,
                   **{"autoencoder": "yes, trained here at latent %d"
                                     % c30.latent_dim()}))
NOTE = (
    "Baseline 5 starts from 30 selected features, so AE=30 maps 30 -> 30 and "
    "compresses nothing: this row is an autoencoder with no bottleneck, not a "
    "wider bottleneck."
)


def train(verbose: bool = True) -> dict[str, Any]:
    """`experiments_baseline5.run_baseline5`, unchanged."""
    from .. import experiments_baseline5 as b5

    inputs = c30.inputs()
    return b5.run_baseline5(inputs["y_train"], inputs["y_test"],
                            inputs["meta_test"], inputs["label_mapping"],
                            duplicate_flags=inputs["duplicate_flags"],
                            overwrite=True, verbose=verbose)

def run(verbose: bool = True) -> dict[str, Any]:
    return c30.run_one_ae30(KEY, train, components=COMPONENTS, note=NOTE,
                            verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
