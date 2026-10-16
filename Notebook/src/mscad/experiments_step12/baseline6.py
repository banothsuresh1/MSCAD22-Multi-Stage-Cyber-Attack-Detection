r"""
Baseline 6 -- AE -> PCA -> CNN-LSTM.  The only configuration that trains.

Baseline 6 is new, so this is the one file that may build a model.  Even here
nothing is implemented twice: the training, the PCA reuse, the length check
and the saving all live in the existing `experiments_baseline6` module, which
goes through the same CNN-LSTM builder (`models.build_cnn_lstm`), the same
training loop (`experiments.train_classifier`) and the same evaluator
(`evaluation.evaluate`) as Approach 1.  This file wires it to the Step 12
entry point and reports it like the other seven.

B6 differs from B4 only by the classifier (CNN-LSTM instead of LSTM) and from
B1 only by the PCA stage, so the feature space is Baseline 4's SAVED encoder
and SAVED PCA, loaded and never refitted -- unless B6_PCA_COMPONENTS is an
integer, which fits a NEW PCA on the TRAINING latents only, inside B6's own
folder, and labels the row PCA(k).

    python -m src.mscad.experiments_step12.baseline6
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import config

from .. import comparison as cmp
from .. import experiments_baseline6 as b6
from . import common

KEY = getattr(config, "BASELINE6_NAME", "baseline6_ae_pca_cnn_lstm")
DISPLAY_LABEL = cmp.display_label(KEY)
SPEC = dict(config.BASELINE6_SPEC)


def reuses_baseline4_pca() -> bool:
    """True while B6_PCA_COMPONENTS is "same_as_B4" rather than an integer."""
    setting = getattr(config, "B6_PCA_COMPONENTS", "same_as_B4")
    return isinstance(setting, str)


def _pca_component() -> str:
    if reuses_baseline4_pca():
        return "yes, Baseline 4's SAVED PCA reused, never refitted"
    return ("yes, a NEW PCA(%s) fitted on the TRAINING latents only; "
            "Baseline 4's own pca.joblib is left untouched"
            % config.B6_PCA_COMPONENTS)


COMPONENTS = {
    "feature selection": "none (all Step 5 features)",
    "autoencoder": "yes, Baseline 4's SAVED encoder reused (-> %d), never "
                   "refitted" % config.AE_LATENT_DIM,
    "PCA": _pca_component(),
    "CNN": "yes (the shared Conv1D/MaxPool stack, unchanged)",
    "attention": "no",
    "LSTM": "yes",
    "input features": "all Step 5 features",
    "what it isolates": config.STEP12_ISOLATES.get(KEY, ""),
}


def _files() -> list[tuple[str, Path]]:
    """B6's own folder, plus the two Baseline 4 files it reuses."""
    files = common.saved_files(KEY, config.BASELINE6_FILES)
    folder4 = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder4 is not None:
        source = Path(folder4) / config.BASELINE4["name"]
        files.append(("reused: B4 encoder",
                      source / config.BASELINE4_FILES["encoder"]))
        files.append(("reused: B4 pca",
                      source / config.BASELINE4_FILES["pca"]))
    return files


def pca_length_check(n_components: int | None = None,
                     verbose: bool = True) -> dict[str, Any]:
    """What the shared Conv1D/MaxPool stack has to slide over.

    Reported, never hidden, and never worked around by changing the shared
    architecture: `experiments_baseline6.pca_length_table` mirrors
    `models.build_cnn_lstm`'s own two rules, so it says whether B6 can be
    built at all before anything is trained.
    """
    if n_components is None:
        metrics = cmp.load_result(KEY) or {}
        n_components = metrics.get("pca_components")
    if n_components is None:
        setting = getattr(config, "B6_PCA_COMPONENTS", "same_as_B4")
        if not isinstance(setting, str):
            n_components = int(setting)
    if n_components is None:
        return {"table": pd.DataFrame(), "warning": None,
                "detail": "the component count is only known once the feature "
                          "space is loaded (B6_PCA_COMPONENTS = %r)"
                          % getattr(config, "B6_PCA_COMPONENTS", None)}
    table = b6.pca_length_table(int(n_components))
    warning = b6.pca_length_warning(table)
    if verbose:
        from .. import data_loading as dl

        dl.show_table(table, all_columns=True,
                      note="Copied from models.build_cnn_lstm's own rules: it "
                           "refuses an input shorter than the kernel, and "
                           "pools only when there are pool*2 steps to halve. "
                           "The shared architecture is NOT changed for B6.")
        if warning:
            print("  [WARN]  %s" % warning)
    return {"table": table, "warning": warning,
            "n_components": int(n_components)}


def load(verbose: bool = True) -> dict[str, Any]:
    """Read a finished Baseline 6 back, through its own loader."""
    out = b6.load_baseline6(verbose=verbose)
    out["length"] = out.get("pca_length")
    return out


def train(verbose: bool = True) -> dict[str, Any]:
    """Train Baseline 6 -- the one configuration in Step 12 that is new.

    `experiments_baseline6.run_baseline6` does the work: it loads Baseline 4's
    saved encoder and PCA, asserts the protocol against Approach 1's, checks
    the sequence length before the builder is reached, trains through the
    shared trainer and scores with the unchanged Step 11 evaluate().
    """
    return b6.run_baseline6(overwrite=getattr(config, "OVERWRITE_B6",
                                              config.OVERWRITE_BASELINE6),
                            verbose=verbose)


def _checks(out: dict[str, Any]) -> pd.DataFrame:
    """B6's own protocol table when it trained; the shared one when it loaded.

    When B6 trains here, `experiments_baseline6.protocol_checks` has the real
    `info` from the run and checks every equality against Approach 1's
    protocol.  When it loads, that table was saved beside the model and is
    read back rather than re-asserted against a run that is not happening.
    """
    table = out.get("protocol")
    if table is not None and len(table):
        return table
    return common.protocol_checks(KEY, out.get("info"))


def run(verbose: bool = True) -> dict[str, Any]:
    result = common.run_one(KEY, load, train, components=COMPONENTS,
                            extra_files=_files, checks=_checks,
                            verbose=verbose)
    metrics = result.get("metrics") or {}
    length = pca_length_check(metrics.get("pca_components"), verbose=verbose)
    result["length"] = length["table"]
    result["length_warning"] = length["warning"]
    return result


if __name__ == "__main__":                                  # pragma: no cover
    import sys

    raise SystemExit(common.cli(sys.modules[__name__]))
