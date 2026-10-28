r"""
src/mscad/experiments_step12_ae30/baseline1_ae30.py -- Baseline 1 (AE -> CNN-LSTM, no selection) at AE latent 30.

NEW FILE.  No training code: it calls experiments.run_experiment, the same function the AE=15
run uses, inside `common_ae30.ae30()` so the bottleneck is
config.AE_LATENT_DIM_AE30 and every output lands under
config.OUTPUTS_ROOT_AE30.

The autoencoder is 67 -> 30 here rather than 67 -> 15, so this row
is a genuine bottleneck comparison.
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
        from ..experiments_step12 import baseline1 as sibling
    except ImportError:                                     # pragma: no cover
        return None
    return sibling


_SIBLING = _ae15()

KEY = "baseline1_ae_cnn_lstm"
# the AE=15 wrapper's own declaration of what this configuration is, with the
# one thing this run changes overridden
COMPONENTS = ({} if _SIBLING is None else
              dict(_SIBLING.COMPONENTS,
                   **{"autoencoder": "yes, trained here at latent %d"
                                     % c30.latent_dim()}))
NOTE = None


def train(verbose: bool = True) -> dict[str, Any]:
    """`experiments.run_experiment` with this configuration's existing spec."""
    from .. import experiments as xp

    spec = next(s for s in config.EXPERIMENTS if s["name"] == KEY)
    inputs = c30.inputs()
    return xp.run_experiment(spec, inputs["y_train"], inputs["y_test"],
                             meta_test=inputs["meta_test"],
                             label_mapping=inputs["label_mapping"],
                             duplicate_flags=inputs["duplicate_flags"],
                             overwrite=True, verbose=verbose)

def run(verbose: bool = True) -> dict[str, Any]:
    return c30.run_one_ae30(KEY, train, components=COMPONENTS, note=NOTE,
                            verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
