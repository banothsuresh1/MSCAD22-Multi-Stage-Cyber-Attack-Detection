r"""
src/mscad/experiments_step12_ae30/baseline3_ae30.py -- Baseline 3 (CNN-LSTM, no AE) at AE latent 30.

NEW FILE.  No training code: it calls nothing -- see below, the same function the AE=15
run uses, inside `common_ae30.ae30()` so the bottleneck is
config.AE_LATENT_DIM_AE30 and every output lands under
config.OUTPUTS_ROOT_AE30.

Baseline 3 has NO autoencoder, so its result cannot depend on the
latent width: there is nothing to retrain, and retraining it would
only add run-to-run noise to a row that is the same experiment.

It is kept for symmetry with the other seven rows.  The AE=15 result
is COPIED into the AE=30 folder, and the copy records where it came
from -- `copied_from`, `latent_dim_insensitive` and `note` -- so the
file says what it is and the table reads a real file rather than a
blank.
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
        from ..experiments_step12 import baseline3 as sibling
    except ImportError:                                     # pragma: no cover
        return None
    return sibling


_SIBLING = _ae15()

KEY = "baseline3_cnn_lstm"
# the AE=15 wrapper's own declaration of what this configuration is, with the
# one thing this run changes overridden
COMPONENTS = ({} if _SIBLING is None else
              dict(_SIBLING.COMPONENTS,
                   **{"autoencoder": "yes, trained here at latent %d"
                                     % c30.latent_dim()}))
NOTE = (
    "Baseline 3 has no autoencoder, so it is insensitive to the latent width: "
    "this row is the AE=15 result, copied with its provenance recorded in the "
    "file, not a second training run."
)


def train(verbose: bool = True) -> dict[str, Any]:
    """Copy the AE=15 result, recording in the file that it is a copy."""
    import json

    from .. import comparison as cmp

    source = cmp.result_path(KEY)
    if source is None or not Path(source).exists():
        raise c30.Ae30Error(
            "Baseline 3 has no AE=15 result to carry over, and it has no "
            "autoencoder to retrain:\n    missing: %s\n"
            "Run Baseline 3 in the AE=15 table first." % source)
    metrics = json.loads(Path(source).read_text(encoding="utf-8"))
    metrics = {**metrics,
               "copied_from": str(source),
               "latent_dim_insensitive": True,
               "note": "no autoencoder, so the latent width cannot change this "
                       "result; this is the AE=15 run, not a second training"}
    return {"metrics": metrics, "status": "copied_from_ae15"}

def run(verbose: bool = True) -> dict[str, Any]:
    return c30.run_one_ae30(KEY, train, components=COMPONENTS, note=NOTE,
                            verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
