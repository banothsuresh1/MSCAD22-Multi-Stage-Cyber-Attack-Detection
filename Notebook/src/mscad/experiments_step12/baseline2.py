r"""
Baseline 2 -- AE -> LSTM, with NO convolutional front end.

LOAD ONLY.  Baseline 2 was trained and scored by an earlier Step 12 run; this file
reads its saved metrics.json and never builds a model.  The only training code
it names is the EXISTING per-configuration runner
`experiments.run_experiment`, which is reached solely when
LOCK_TRAINED_MODELS is False and no saved result exists.

    python -m src.mscad.experiments_step12.baseline2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import config

from .. import comparison as cmp
from . import common

KEY = 'baseline2_ae_lstm'
DISPLAY_LABEL = cmp.display_label(KEY)
SPEC = next(s for s in config.EXPERIMENTS if s["name"] == KEY)
COMPONENTS = {
    "feature selection": "none (all Step 5 features)",
    "autoencoder": "yes, a new AE fitted on all features (-> %d)"
                   % config.AE_LATENT_DIM,
    "PCA": "no",
    "CNN": "no",
    "attention": "no",
    "LSTM": "yes",
    "input features": "all Step 5 features",
    "what it isolates": config.STEP12_ISOLATES.get(KEY, ""),
}


def _files() -> list[tuple[str, Path]]:
    """The extra saved files this configuration's folder holds."""
    return common.saved_files(KEY, {})


def load(verbose: bool = True) -> dict[str, Any]:
    """Read the saved metrics.json, through comparison.load_result()."""
    metrics = cmp.load_result(KEY)
    folder = common.metrics_path(KEY)
    if verbose and metrics is not None:
        from .. import data_loading as dl

        dl.show_table(dl.kv_table([
            ("status", "loaded from disk; nothing was retrained"),
            ("folder", str(None if folder is None else folder.parent)),
            ("macro-F1", "%.4f" % metrics["full"]["f1_macro"]),
            ("parameters", metrics.get("parameters")),
        ]), note="Baseline 2 isolates the CNN front end: it is Baseline 1 with the Conv1D/MaxPool stack removed.")
    return {"metrics": metrics,
            "folder": str(None if folder is None else folder.parent)}


def train(verbose: bool = True) -> dict[str, Any]:
    """Only reachable with LOCK_TRAINED_MODELS = False and no saved result.

    Calls the existing runner with the existing inputs; no training code is
    written in this folder.
    """
    from .. import experiments as xp

    inputs = common.evaluation_inputs()
    return xp.run_experiment(SPEC, inputs["y_train"], inputs["y_test"],
                             inputs["meta_test"], inputs["label_mapping"],
                             inputs["duplicate_flags"], verbose=verbose)


def run(verbose: bool = True) -> dict[str, Any]:
    return common.run_one(KEY, load, train, components=COMPONENTS,
                          extra_files=_files, verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    import sys

    raise SystemExit(common.cli(sys.modules[__name__]))
