r"""
Approach 1 -- FS -> AE -> CNN-LSTM.  The proposed pipeline.

LOAD ONLY.  Approach 1 was trained by Step 10 and scored by Step 11; this file
reads those saved files and never builds a model.  It calls
`evaluation.load_step11()` and `comparison.load_result()`, both unchanged.

    python -m src.mscad.experiments_step12.approach1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import config

from .. import comparison as cmp
from .. import step12_checks as s12
from . import common

KEY = "proposed"
DISPLAY_LABEL = cmp.display_label(KEY)
COMPONENTS = {
    "feature selection": "Step 6 fused MI + Pearson + PI, top %d"
                         % config.N_SELECTED_FEATURES,
    "autoencoder": "yes, Step 7 encoder (%d -> %d)"
                   % (config.N_SELECTED_FEATURES, config.AE_LATENT_DIM),
    "PCA": "no",
    "CNN": "yes",
    "attention": "no",
    "LSTM": "yes",
    "input features": config.N_SELECTED_FEATURES,
    "what it isolates": config.STEP12_ISOLATES.get(KEY, ""),
}


def _finished() -> bool:
    """Approach 1 counts as finished when EITHER saved place holds it.

    Its published evaluation is Step 11's; Step 12 also keeps a `proposed` row
    so the comparison can read all eight the same way.  Either one means there
    is nothing to train.
    """
    from .. import evaluation as ev

    return bool(ev.step11_outputs_exist() or common.is_finished(KEY))


def _files() -> list[tuple[str, Path]]:
    """The saved files only Approach 1 has: Step 10's model, Step 11's metrics."""
    out: list[tuple[str, Path]] = []
    step10 = config.usable_output_dir(config.STEP10_OUTPUT_DIR)
    if step10 is not None:
        out.append(("step10 model", Path(step10) / config.STEP10_FILES["model"]))
        card = Path(step10) / config.STEP10_FILES.get("model_card",
                                                      "model_card.json")
        out.append(("step10 model card", card))
    step11 = config.usable_output_dir(config.STEP11_OUTPUT_DIR)
    if step11 is not None:
        out.append(("step11 metrics",
                    Path(step11) / config.STEP11_FILES["metrics"]))
        out.append(("step11 predictions",
                    Path(step11) / config.STEP11_FILES["predictions"]))
    return out


def load(verbose: bool = True) -> dict[str, Any]:
    """Read Step 11's saved evaluation and Step 12's `proposed` row."""
    from .. import evaluation as ev

    step11 = (ev.load_step11(verbose=verbose) if ev.step11_outputs_exist()
              else None)
    row = cmp.load_result(KEY)
    metrics = row if row is not None else (step11 or {}).get("metrics")
    return {"metrics": metrics, "step11": step11, "step12_row": row,
            "sources": {"step11": str(config.STEP11_OUTPUT_DIR),
                        "step12 row": str(common.metrics_path(KEY))}}


def train(verbose: bool = True) -> dict[str, Any]:
    """Only reachable with LOCK_TRAINED_MODELS = False and no saved result.

    Even then nothing new is written here: Step 11's own runner is called, and
    it is the one that decides what to evaluate.
    """
    from .. import evaluation as ev

    return ev.run_step11(verbose=verbose)


def consistency(verbose: bool = True):
    """Does Step 11's Approach 1 agree with Step 12's row, and with any backup?

    `step12_checks.check_approach1_consistency`, unchanged: it reports the
    sources side by side and chooses none of them.
    """
    checks = s12.Checks()
    table = s12.check_approach1_consistency(checks, verbose=verbose)
    return {"table": table, "checks": checks.frame(),
            "failures": checks.failures}


def run(verbose: bool = True) -> dict[str, Any]:
    return common.run_one(KEY, load, train, components=COMPONENTS,
                          extra_files=_files, finished=_finished,
                          verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    import sys

    raise SystemExit(common.cli(sys.modules[__name__]))
