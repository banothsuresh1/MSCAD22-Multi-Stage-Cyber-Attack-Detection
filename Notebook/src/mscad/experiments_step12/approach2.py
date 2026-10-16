r"""
Approach 2 -- FS -> AE -> CNN-Attention-LSTM.

LOAD ONLY.  Approach 2 was trained by notebook section 10.2 and scored by
11.x; this file reads those saved files and never builds a model.  It calls
`models_attention.load_step10b()` and `comparison.load_result()`, unchanged.

    python -m src.mscad.experiments_step12.approach2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import config

from .. import comparison as cmp
from . import common

KEY = getattr(config, "APPROACH2_KEY", "approach2_attention")
DISPLAY_LABEL = cmp.display_label(KEY)
COMPONENTS = {
    "feature selection": "Step 6 fused MI + Pearson + PI, top %d"
                         % config.N_SELECTED_FEATURES,
    "autoencoder": "yes, Step 7 encoder (%d -> %d)"
                   % (config.N_SELECTED_FEATURES, config.AE_LATENT_DIM),
    "PCA": "no",
    "CNN": "yes",
    "attention": "yes, additive over the LSTM sequence",
    "LSTM": "yes (returns the sequence)",
    "input features": config.N_SELECTED_FEATURES,
    "what it isolates": config.STEP12_ISOLATES.get(KEY, ""),
}


def _finished() -> bool:
    from .. import models_attention as ma

    return bool(ma.step11b_outputs_exist() or common.is_finished(KEY))


def _files() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    step10b = config.usable_output_dir(
        getattr(config, "STEP10B_OUTPUT_DIR", config.STEP10_OUTPUT_DIR))
    if step10b is not None:
        names = getattr(config, "STEP10B_FILES", {})
        for kind in ("model", "model_card", "history"):
            if names.get(kind):
                out.append(("step10b %s" % kind, Path(step10b) / names[kind]))
    return out


def load(verbose: bool = True) -> dict[str, Any]:
    """Read the saved Approach 2 model card and its Step 11b metrics.

    The model itself is NOT loaded: nothing here needs the weights, and
    loading a Keras model to print a number it already saved is the kind of
    accident that leaves a model file newer than its metrics.
    """
    from .. import models_attention as ma

    folder = config.usable_output_dir(
        getattr(config, "STEP10B_OUTPUT_DIR", config.STEP10_OUTPUT_DIR))
    card = {}
    if folder is not None:
        import json

        path = Path(folder) / getattr(config, "STEP10B_FILES", {}).get(
            "model_card", "model_card.json")
        if path.exists():
            card = json.loads(path.read_text(encoding="utf-8"))
    metrics = cmp.load_result(KEY)
    if verbose and card:
        from .. import data_loading as dl

        dl.show_table(dl.kv_table([
            ("status", "loaded from disk; the model was not reopened"),
            ("parameters", card.get("parameters")),
            ("attention", "yes"),
            ("selected by 10.2", card.get("selected_as_primary",
                                          "(not recorded)")),
        ]), note="Approach 2's saved model card. "
                 "models_attention.load_step10b() reopens the weights when "
                 "they are actually needed (14.x).")
    return {"metrics": metrics, "model_card": card,
            "loader": "models_attention.load_step10b (not called: weights "
                      "are not needed to read a saved metric)",
            "step11b": str(getattr(config, "STEP11B_OUTPUT_DIR", ""))}


def train(verbose: bool = True) -> dict[str, Any]:
    """Only reachable with LOCK_TRAINED_MODELS = False and no saved result."""
    from .. import models_attention as ma

    return ma.run_step11b(verbose=verbose)


def run(verbose: bool = True) -> dict[str, Any]:
    return common.run_one(KEY, load, train, components=COMPONENTS,
                          extra_files=_files, finished=_finished,
                          verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    import sys

    raise SystemExit(common.cli(sys.modules[__name__]))
