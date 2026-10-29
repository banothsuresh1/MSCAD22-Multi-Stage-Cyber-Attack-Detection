r"""B3 at AE=64: no autoencoder, so the latent width cannot change it.  Its
AE=64 row is the AE=15 result, copied with copied_from / latent_dim_insensitive
recorded in the file -- not retrained.  The AE=15 source path is resolved
BEFORE the ae64() redirect (resolving it inside would read the empty AE=64
root -- the bug fixed in baseline3_ae30.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from . import common_ae64 as c64  # noqa: E402

NAME = "baseline3_cnn_lstm"
LABEL = "B3: CNN-LSTM (no AE)  [AE=64: copied from AE=15]"


def _make_train(source: Path | None):
    def train(verbose: bool = True) -> dict[str, Any]:
        if source is None or not Path(source).exists():
            raise c64.Ae64Error(
                "B3 has no AE=15 result to carry over, and no autoencoder to "
                "retrain:\n    missing: %s\nRun B3 in the AE=15 table first."
                % source)
        metrics = json.loads(Path(source).read_text(encoding="utf-8"))
        metrics = {**metrics, "config": NAME, "label": LABEL,
                   "ae64_sensitivity": True, "copied_from": str(source),
                   "latent_dim_insensitive": True,
                   "note": "no autoencoder, so the latent width cannot change "
                           "this result; this is the AE=15 run, not a retrain"}
        place = c64.folder(NAME)
        Path(place).mkdir(parents=True, exist_ok=True)
        (Path(place) / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        return {"metrics": metrics, "status": "copied_from_ae15"}
    return train


def run(verbose: bool = True) -> dict[str, Any]:
    from .. import comparison as cmp
    source = cmp.result_path(NAME)          # AE=15 official root, before redirect
    return c64.run_one_ae64(NAME, _make_train(source), verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
