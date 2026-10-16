r"""
src/mscad/fusion_all.py -- STEP 13 ADDENDUM: adaptive evidence fusion across
every approach and baseline.

NEW FILE.  Nothing existing is modified: fusion.py, explainability.py,
experiments*.py, comparison.py, models*.py, config.py's Step 13 block and every
file already in C:\MSCAD2022\outputs\step13\ are left exactly as they are.
`FUSION_MODELS` still names the three models Step 13 fuses, and Step 14 still
explains that fusion.

What this adds
--------------
Step 13 answered "can three imperfect models be combined so that each is
listened to where it is strong?".  Two questions it did not answer:

  * what happens when EVERY configuration that exists on disk is offered to the
    same fusion (FALL), and
  * what happens when the fusion is given a SUBSET chosen on the validation
    split (FSEL)?

Both are answered with the Step 13 arithmetic, called rather than copied: the
reliability, confidence, weighting, fusion, threshold and scoring functions all
come from `fusion`.  This module adds only what Step 13 has no reason to own --
loaders for the three models it cannot load itself (Approach 2, Baseline 4,
Baseline 5), a diversity check, the greedy subset search, and the tables.

Nothing is retrained
--------------------
Every model is read from the folder its own step wrote:

    proposed, baseline1-3   step12\<name>\model.keras  (or step10\ for the
                            proposed model, as Step 13 already does)
    approach2_attention     step10_attention\cnn_attention_lstm.keras
    baseline4_ae_pca_lstm   step12\baseline4_ae_pca_lstm\  + pca.joblib
    baseline5               step12\baseline5\             + its MI feature list

A model that is not there is SKIPPED by name, with the cells that produce it
named, and the fusion continues with the rest.  No branch of this module trains
anything.

No leakage
----------
The reliabilities, every threshold AND the choice of which models to include
come from the validation part of the Step 9 split.  The test set is scored once,
at the end, with all of those held fixed.  `greedy_forward_selection` is not
given the test probabilities at all -- it cannot see them.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from . import comparison as cmp  # noqa: E402
from . import data_loading as dl  # noqa: E402
from . import evaluation as ev  # noqa: E402
from . import fusion as fu  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = fu.TARGET_COLUMN

# the names the comparison table uses for the three fusion variants
F3_NAME = "F3: Step 13 fusion (Approach 1 + B2 + B3)"
FALL_NAME = "FALL: adaptive fusion, all available models"
FSEL_NAME = "FSEL: adaptive fusion, validation-chosen subset"


class FusionAllError(Exception):
    """Raised when the addendum cannot continue without guessing."""


# ===========================================================================
# 13.8.0  WHICH MODELS, AND WHAT THEY ARE CALLED
# ===========================================================================
def display_label(key: str) -> str:
    """The Step 12 display label for a configuration key.

    Delegates to `comparison.display_label`, so every table in Step 12, 13 and
    this addendum names a model the same way.
    """
    return cmp.display_label(key)


def model_keys(keys: Sequence[str] | None = None,
               verbose: bool = True) -> list[str]:
    """`FUSION_ALL_MODELS`, verified against `STEP12_DISPLAY_ORDER`.

    The internal keys are what every saved folder and every other table is
    addressed by, so a typo here would silently fuse a different set.  Checked
    rather than trusted.
    """
    keys = list(config.FUSION_ALL_MODELS if keys is None else keys)
    known = [k for k, _label in config.STEP12_DISPLAY_ORDER]
    unknown = [k for k in keys if k not in known]
    if unknown:
        raise FusionAllError(
            "FUSION_ALL_MODELS names %s, which are not keys of "
            "STEP12_DISPLAY_ORDER (%s). Use the exact internal keys."
            % (unknown, known))
    duplicates = [k for k in set(keys) if keys.count(k) > 1]
    if duplicates:
        raise FusionAllError(
            "FUSION_ALL_MODELS lists %s more than once" % duplicates)
    if verbose:
        show_note("Offering %d configurations to the fusion: %s. Every key is a "
                  "`STEP12_DISPLAY_ORDER` key, so the labels and the saved "
                  "folders agree."
                  % (len(keys), ", ".join(display_label(k) for k in keys)))
    return keys


def ordered(keys: Sequence[str]) -> list[str]:
    """`keys` in STEP12_DISPLAY_ORDER order, whatever order they arrive in."""
    order = [k for k, _label in config.STEP12_DISPLAY_ORDER]
    return sorted(keys, key=lambda k: order.index(k) if k in order else len(order))


def predictions_path(key: str) -> Path | None:
    """Where this configuration's saved `predictions.csv` lives, or None.

    `comparison.result_path` already knows that Approach 2 is written by Step
    11.3 and everything else by Step 12, so the two places are not spelled out
    again here.
    """
    path = cmp.result_path(key)
    return None if path is None else path.parent / "predictions.csv"


def which_step_produces(key: str) -> str:
    """The cells a reader has to run to make a missing model appear."""
    return cmp.which_step_produces(key)


# ===========================================================================
# 13.8.1  LOADING EVERY MODEL -- read only, never retrained
# ===========================================================================
def _keras():
    from . import models as md
    return md._keras()


def _encoder_from(folder: Path, what: str):
    """(encode, feature order) from an `encoder.keras` + `ae_config.json` pair.

    This is the shape Step 12's baselines and the Baseline 5 addendum both save,
    and `autoencoder.make_encoder_function` is the same function they encoded
    with, so the latent space is reproduced rather than approximated.
    """
    from . import autoencoder as ae

    encoder_path = folder / "encoder.keras"
    settings_path = folder / "ae_config.json"
    for path in (encoder_path, settings_path):
        if not path.exists():
            raise FusionAllError("%s is missing:\n    %s" % (what, path))
    encoder = _keras().models.load_model(encoder_path)
    order = list(json.loads(settings_path.read_text(encoding="utf-8"))["features"])
    return ae.make_encoder_function(encoder, order), order


def _bundle_step12(key: str, verbose: bool = False) -> dict[str, Any]:
    """proposed, baseline1, baseline2, baseline3 -- straight through Step 13."""
    bundle = fu.load_model_bundle(key, verbose=verbose)
    bundle["source"] = "fusion.load_model_bundle"
    return bundle


def _bundle_approach2(key: str, verbose: bool = False) -> dict[str, Any]:
    """Approach 2: the proposed feature pipeline, the Step 10.2 model.

    Approach 2 is the Step 10 stack with attention added, so it takes exactly
    the proposed configuration's inputs -- the Step 6 top 30, encoded by the
    Step 7 encoder.  Those two are loaded with the same calls Step 13 makes for
    the proposed model; only the classifier differs.
    """
    from . import autoencoder as ae
    from . import experiments as xp
    from . import models_attention as ma

    loaded = ma.load_step10b(verbose=False)          # raises if 10.2 never ran
    source = xp.load_feature_source("selected30", verbose=False)
    encode, _encoder, order = ae.load_encoder(verbose=False)
    X_train = encode(source["X_train"][order], what="training features")
    X_test = encode(source["X_test"][order], what="test features")
    if verbose:
        print("  loaded %-24s model=%s  features=%d"
              % (key, Path(loaded["folder"]).name, X_train.shape[1]))
    return {"name": key, "spec": {"name": key, "label": display_label(key)},
            "model": loaded["model"], "X_train": X_train, "X_test": X_test,
            "model_path": str(Path(loaded["folder"])
                              / config.STEP10B_FILES["model"]),
            "source": "models_attention.load_step10b"}


def _bundle_baseline4(key: str, verbose: bool = False) -> dict[str, Any]:
    """Baseline 4: encoder -> PCA -> LSTM, all three read from its own folder.

    Baseline 4 saved its own copy of the encoder it borrowed and the fitted
    `pca.joblib`, so the latent-then-PCA space is reproduced exactly, and
    `experiments_pca.transform_with_pca` is the same `.transform()`-only call
    Baseline 4 used.
    """
    from . import experiments as xp
    from . import experiments_pca as xpca

    folder12 = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder12 is None:
        raise FusionAllError("the Step 12 output folder is not usable here -- %s"
                             % config.why_unusable(config.STEP12_OUTPUT_DIR))
    folder = folder12 / key
    model_path = folder / config.BASELINE4_FILES["model"]
    pca_path = folder / config.BASELINE4_FILES["pca"]
    for path, what in ((model_path, "the Baseline 4 classifier"),
                       (pca_path, "the fitted Baseline 4 PCA")):
        if not path.exists():
            raise FusionAllError("%s is missing:\n    %s" % (what, path))

    from joblib import load as joblib_load
    model = _keras().models.load_model(model_path)
    pca = joblib_load(pca_path)
    encode, order = _encoder_from(folder, "the Baseline 4 encoder")

    source = xp.load_feature_source(config.BASELINE4["feature_source"],
                                    verbose=False)
    Z_train = encode(source["X_train"][order], what="training features")
    Z_test = encode(source["X_test"][order], what="test features")
    X_train = xpca.transform_with_pca(pca, Z_train, "the training latent features")
    X_test = xpca.transform_with_pca(pca, Z_test, "the test latent features")
    if verbose:
        print("  loaded %-24s model=%s  features=%d (PCA)"
              % (key, model_path.name, X_train.shape[1]))
    return {"name": key, "spec": dict(config.BASELINE4), "model": model,
            "X_train": X_train, "X_test": X_test,
            "model_path": str(model_path), "pca": pca,
            "source": "baseline4 folder (encoder + pca.joblib + model)"}


def _bundle_baseline5(key: str, verbose: bool = False) -> dict[str, Any]:
    """Baseline 5: its MI top-30 list, its own encoder, its CNN-LSTM."""
    from . import experiments as xp

    folder12 = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder12 is None:
        raise FusionAllError("the Step 12 output folder is not usable here -- %s"
                             % config.why_unusable(config.STEP12_OUTPUT_DIR))
    folder = folder12 / config.BASELINE5_NAME
    model_path = folder / config.BASELINE5_FILES["model"]
    selected_path = folder / config.BASELINE5_FILES["selected"]
    for path, what in ((model_path, "the Baseline 5 classifier"),
                       (selected_path, "the Baseline 5 MI feature list")):
        if not path.exists():
            raise FusionAllError("%s is missing:\n    %s" % (what, path))

    model = _keras().models.load_model(model_path)
    features = list(pd.read_csv(selected_path)["feature"])
    encode, order = _encoder_from(folder, "the Baseline 5 encoder")
    if sorted(order) != sorted(features):
        raise FusionAllError(
            "the Baseline 5 encoder was fitted on %d features but its saved MI "
            "selection has %d; re-run the Baseline 5 cells (12.4) with "
            "OVERWRITE_BASELINE5 = True" % (len(order), len(features)))

    source = xp.load_feature_source("all", verbose=False)
    X_train = encode(source["X_train"][order], what="training features")
    X_test = encode(source["X_test"][order], what="test features")
    if verbose:
        print("  loaded %-24s model=%s  features=%d"
              % (key, model_path.name, X_train.shape[1]))
    return {"name": key, "spec": dict(config.BASELINE5_SPEC), "model": model,
            "X_train": X_train, "X_test": X_test,
            "model_path": str(model_path),
            "source": "baseline5 folder (MI list + encoder + model)"}


# Approach 2 is dispatched by `config.APPROACH2_KEY` below rather than listed
# here, so a rename of that key cannot leave a stale string behind.
_LOADERS = {
    "baseline4_ae_pca_lstm": _bundle_baseline4,
    "baseline5": _bundle_baseline5,
}


def load_bundle(key: str, verbose: bool = False) -> dict[str, Any]:
    """The saved classifier and feature pipeline for one configuration key."""
    approach2 = getattr(config, "APPROACH2_KEY", "approach2_attention")
    if key == approach2:
        return _bundle_approach2(key, verbose=verbose)
    loader = _LOADERS.get(key)
    if loader is not None:
        return loader(key, verbose=verbose)
    return _bundle_step12(key, verbose=verbose)


def available_bundles(keys: Sequence[str] | None = None, verbose: bool = True
                      ) -> tuple[dict[str, dict], pd.DataFrame]:
    """Load what is on disk; skip what is not, by name, and carry on.

    Returns (bundles, availability).  A model that cannot be loaded gets a row
    saying so and the cells that produce it -- never a trained replacement.
    """
    keys = model_keys(keys, verbose=False)
    bundles: dict[str, dict] = {}
    rows = []
    for key in keys:
        try:
            bundles[key] = load_bundle(key, verbose=verbose)
            rows.append({"model": display_label(key), "key": key,
                         "available": True,
                         "features into the classifier":
                             int(bundles[key]["X_test"].shape[1]),
                         "loaded from": bundles[key].get("source", ""),
                         "note": ""})
        except Exception as problem:                 # a missing file, not a bug
            first = str(problem).strip().split("\n")[0]
            message = ("%s not available -- %s. Run %s."
                       % (display_label(key), first, which_step_produces(key)))
            if verbose:
                print("  [SKIP]  %s" % message)
            rows.append({"model": display_label(key), "key": key,
                         "available": False,
                         "features into the classifier": 0,
                         "loaded from": "", "note": message})
    availability = pd.DataFrame(rows)
    if not bundles:
        raise FusionAllError(
            "none of the %d configurations in FUSION_ALL_MODELS could be loaded, "
            "so there is nothing to fuse. The notes say which cells produce "
            "each one." % len(keys))
    if len(bundles) < 2:
        raise FusionAllError(
            "only %s could be loaded; fusing needs at least two models. Run the "
            "cells named in the availability table."
            % display_label(next(iter(bundles))))
    return bundles, availability


# ===========================================================================
# 13.8.1b  PROBABILITIES, ALIGNED
# ===========================================================================
def test_probabilities(bundles: dict[str, dict], y_test: pd.DataFrame,
                       meta_test: pd.DataFrame | None, verbose: bool = True
                       ) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Each model's test probabilities, cross-checked against its saved file.

    `fusion.test_probabilities` looks only in `step12\\<name>\\`, which is where
    four of these seven live; Approach 2's predictions are written by Step 11.3
    instead.  The path comes from `comparison.result_path` so neither place is
    hard-coded twice, and the 1e-6 comparison is the same one Step 13 makes: a
    mismatch means the saved model is not the one that produced those
    predictions.
    """
    probs: dict[str, np.ndarray] = {}
    checks = []
    for key, bundle in bundles.items():
        values, _idx = fu.model_probabilities(bundle, bundle["X_test"], y_test,
                                              meta_test)
        probs[key] = values
        saved_path = predictions_path(key)
        if saved_path is not None and saved_path.exists():
            saved = pd.read_csv(saved_path, low_memory=False)
            columns = [c for c in saved.columns if c.startswith("p_")]
            if len(columns) == values.shape[1] and len(saved) == len(values):
                difference = float(np.abs(saved[columns].to_numpy() - values).max())
                checks.append({"model": display_label(key),
                               "source": str(saved_path.parent.name) + "/"
                                         + saved_path.name,
                               "max abs difference": difference,
                               "result": "[OK]" if difference < 1e-6 else "[FAIL]"})
            else:
                checks.append({"model": display_label(key),
                               "source": saved_path.name,
                               "max abs difference": float("nan"),
                               "result": "[FAIL]"})
        else:
            checks.append({"model": display_label(key), "source": "(not saved)",
                           "max abs difference": float("nan"),
                           "result": "[SKIP]"})
    table = pd.DataFrame(checks)
    if verbose:
        show_table(table, note="Probabilities recomputed from the saved model "
                               "against the saved `predictions.csv`, to 1e-6. A "
                               "`[FAIL]` would mean the file and the model "
                               "disagree.")
    failed = table[table["result"] == "[FAIL]"]
    if len(failed):
        raise FusionAllError(
            "the recomputed probabilities do not match the saved predictions for "
            "%s (tolerance 1e-6). Re-run the cells that produced them."
            % ", ".join(failed["model"]))
    return probs, table


def alignment(test_probs: dict[str, np.ndarray], y_test: pd.DataFrame,
              meta_test: pd.DataFrame | None, class_order: Sequence[str],
              verbose: bool = True):
    """[OK]/[FAIL] that every model describes the same rows in the same order.

    Built as the per-model frames Step 11/12 would have written and handed to
    `fusion.check_alignment`, which raises rather than warning -- fusing frames
    that disagree on row order gives numbers that look fine and mean nothing.
    """
    orig = (meta_test["Orig_Row"].to_numpy() if meta_test is not None
            and "Orig_Row" in meta_test.columns else np.arange(len(y_test)))
    frames = {display_label(key): pd.DataFrame(
        {"Orig_Row": orig, "y_true": y_test[TARGET_COLUMN].to_numpy(),
         **{"p_%s" % class_order[c]: probs[:, c]
            for c in range(len(class_order))}})
        for key, probs in test_probs.items()}
    return fu.check_alignment(frames, class_order, verbose=verbose)


# ===========================================================================
# 13.8.2  DIVERSITY -- validation rows only
# ===========================================================================
def disagreement_table(val_probs: dict[str, np.ndarray],
                       keys: Sequence[str] | None = None) -> pd.DataFrame:
    """Pairwise share of validation flows where the predicted class differs.

    0 means two models are interchangeable on these rows; fusing them cannot
    add information, whatever their weights.
    """
    keys = list(val_probs) if keys is None else list(keys)
    predicted = {k: np.argmax(np.asarray(val_probs[k], dtype="float64"), axis=1)
                 for k in keys}
    labels = [display_label(k) for k in keys]
    grid = pd.DataFrame(0.0, index=labels, columns=labels)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            share = float(np.mean(predicted[a] != predicted[b]))
            grid.iloc[i, j] = round(share, 6)
    grid.index.name = "model"
    return grid


def risk_correlation_table(val_probs: dict[str, np.ndarray],
                           benign_codes: Sequence[int],
                           keys: Sequence[str] | None = None) -> pd.DataFrame:
    """Pairwise Pearson correlation of the attack risk r_m on validation rows.

    `fusion.attack_risk` is the same r_m the fusion itself combines.  A constant
    r_m has no correlation to report, so that pair reads NaN rather than 0.
    """
    keys = list(val_probs) if keys is None else list(keys)
    risk = {k: fu.attack_risk(val_probs[k], benign_codes) for k in keys}
    labels = [display_label(k) for k in keys]
    grid = pd.DataFrame(np.nan, index=labels, columns=labels)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            left, right = risk[a], risk[b]
            if left.std(ddof=0) < config.EPSILON or right.std(ddof=0) < config.EPSILON:
                value = float("nan")
            else:
                value = float(np.corrcoef(left, right)[0, 1])
            grid.iloc[i, j] = value if np.isnan(value) else round(value, 6)
    grid.index.name = "model"
    return grid


# ===========================================================================
# 13.8.3  THE FUSION VARIANTS -- Step 13's arithmetic, called not copied
# ===========================================================================
def validation_scores(y_true: np.ndarray, probs: np.ndarray,
                      n_classes: int) -> dict[str, float]:
    """Macro-F1 and G-mean of one set of VALIDATION probabilities.

    Through `evaluation.headline_metrics`, the same function Steps 11-12 score
    with, so a validation figure here is comparable with a test figure there.
    """
    probs = np.asarray(probs, dtype="float64")
    y_pred = np.argmax(probs, axis=1)
    headline = ev.headline_metrics(np.asarray(y_true), y_pred, probs, n_classes)
    return {"macro_f1": float(headline.get("f1_macro") or 0.0),
            "g_mean": float(headline.get("g_mean") or 0.0)}


def subset_metric(scores: dict[str, float], metric: str | None = None) -> float:
    """The one validation number the subset search maximises."""
    metric = config.SUBSET_METRIC if metric is None else metric
    if metric not in scores:
        raise FusionAllError(
            "SUBSET_METRIC is %r; it must be one of %s"
            % (metric, ", ".join(sorted(scores))))
    return float(scores[metric])


def _contexts_for(keys: Sequence[str], val_probs: dict[str, np.ndarray],
                  test_probs: dict[str, np.ndarray] | None, inputs: dict):
    """(validation contexts, test contexts, context names) for one subset.

    `fusion.context_sets` needs a list per side; when `test_probs` is None --
    which is the case during the subset search -- the validation list is passed
    for both and the test contexts are thrown away, so no test probability can
    reach a selection decision.
    """
    val_list = [val_probs[k] for k in keys]
    test_list = (val_list if test_probs is None
                 else [test_probs[k] for k in keys])
    val_ctx, test_ctx, names = fu.context_sets(
        config.CONTEXT_MODE, val_list, test_list, inputs["y_train"],
        inputs["meta_test"], inputs["label_names"], inputs["n_classes"])
    if val_ctx is None:
        raise FusionAllError(
            "CONTEXT_MODE = \"protocol\" needs the Protocol column in the Step 5 "
            "metadata. Set CONTEXT_MODE = \"consensus_class\" or run Step 5.")
    return val_ctx, (None if test_probs is None else test_ctx), names


def fit_on_validation(keys: Sequence[str], val_probs: dict[str, np.ndarray],
                      inputs: dict, verbose: bool = False) -> dict[str, Any]:
    """Everything the fusion LEARNS, from the validation rows of one subset.

    The reliabilities (`fusion.build_reliability`), the fused validation
    distribution (`fusion.fuse_once`) and the threshold
    (`fusion.choose_threshold`) -- all Step 13 functions, all on validation
    only.  Returns the validation score too, which is what the subset search
    compares.
    """
    keys = list(keys)
    y_val_true = np.asarray(inputs["y_val"][TARGET_COLUMN])
    val_ctx, _test_ctx, ctx_names = _contexts_for(keys, val_probs, None, inputs)

    reliability, info = fu.build_reliability(
        {k: val_probs[k] for k in keys}, y_val_true, val_ctx, ctx_names,
        inputs["n_classes"], verbose=False)
    val_list = [val_probs[k] for k in keys]
    val_fused, val_weights, _rel, _conf = fu.fuse_once(
        val_list, keys, info["lookup"], info["macro"], val_ctx)
    val_risk = fu.attack_risk(val_fused, inputs["benign_codes"])
    threshold = fu.choose_threshold(
        val_risk, np.asarray(inputs["y_val"]["Label_binary"]).astype("int64"))
    scores = validation_scores(y_val_true, val_fused, inputs["n_classes"])
    if verbose:
        print("    validation macro-F1 %.4f, G-mean %.4f, tau %.6f"
              % (scores["macro_f1"], scores["g_mean"], threshold["tau"]))
    return {"keys": keys, "reliability": reliability, "info": info,
            "threshold": threshold, "validation_scores": scores,
            "validation_fused": val_fused, "validation_risk": val_risk,
            "validation_weights": val_weights, "validation_contexts": val_ctx,
            "context_names": ctx_names}


def score_on_test(fitted: dict[str, Any], test_probs: dict[str, np.ndarray],
                  inputs: dict, name: str) -> dict[str, Any]:
    """Apply a validation-fitted fusion to the test set, once.

    Nothing is re-fitted here: the reliabilities, the weights' settings and tau
    all come from `fitted`, which never saw a test row.
    """
    keys = list(fitted["keys"])
    test_list = [test_probs[k] for k in keys]

    # The context of a test flow is label-free -- in "consensus_class" mode it
    # is the models' own mean prediction, in "protocol" mode it is metadata --
    # so it can be read at test time without using the ground truth.
    if config.CONTEXT_MODE == "consensus_class":
        test_ctx = fu.consensus_context(test_list)
    else:
        test_ctx = fu.protocol_contexts(inputs["y_train"],
                                        inputs["meta_test"])[1]
    if test_ctx is None:
        raise FusionAllError(
            "CONTEXT_MODE = \"protocol\" needs the Protocol column in the Step 5 "
            "test metadata. Set CONTEXT_MODE = \"consensus_class\" or run Step 5.")

    started = time.perf_counter()
    fused, weights, rel_matrix, conf_matrix = fu.fuse_once(
        test_list, keys, fitted["info"]["lookup"], fitted["info"]["macro"],
        test_ctx)
    risk = fu.attack_risk(fused, inputs["benign_codes"])
    if config.TEMPORAL_SMOOTHING and inputs["meta_test"] is not None:
        risk = fu.ema_within_capture(risk, inputs["meta_test"]["Source_Capture"],
                                     inputs["meta_test"].get("Timestamp"))
    overhead = (time.perf_counter() - started) / max(len(fused), 1) * 1e6

    tau = float(fitted["threshold"]["tau"])
    result = fu.score_probabilities(
        name, fused, inputs["y_test"], inputs["benign_codes"], tau,
        inputs["duplicate_flags"], inputs["label_names"], inputs["n_classes"],
        overhead_ms=overhead)
    return {"name": name, "keys": keys, "result": result, "fused": fused,
            "weights": weights, "risk": risk, "contexts": test_ctx,
            "rel_matrix": rel_matrix, "conf_matrix": conf_matrix,
            "threshold": fitted["threshold"],
            "validation_scores": fitted["validation_scores"],
            "reliability": fitted["reliability"], "info": fitted["info"],
            "overhead_ms_per_1000": overhead}


def adaptive_variant(keys: Sequence[str], val_probs: dict[str, np.ndarray],
                     test_probs: dict[str, np.ndarray], inputs: dict,
                     name: str, verbose: bool = False) -> dict[str, Any]:
    """One adaptive fusion end to end: fit on validation, score the test once."""
    fitted = fit_on_validation(keys, val_probs, inputs, verbose=verbose)
    return score_on_test(fitted, test_probs, inputs, name)


# --- the fixed-weight baselines, each with its own validation threshold -----
def fixed_variants(keys: Sequence[str], val_probs: dict[str, np.ndarray],
                   test_probs: dict[str, np.ndarray], inputs: dict
                   ) -> list[dict[str, Any]]:
    """Simple average, majority vote and static macro-F1 weights, over `keys`.

    Each gets a threshold chosen on its OWN validation risk: comparing a fixed
    baseline against the adaptive fusion at the adaptive fusion's tau would
    hand the comparison to whichever variant tau happened to suit.
    """
    keys = list(keys)
    val_list = [val_probs[k] for k in keys]
    test_list = [test_probs[k] for k in keys]
    y_val_binary = np.asarray(inputs["y_val"]["Label_binary"]).astype("int64")
    y_val_true = np.asarray(inputs["y_val"][TARGET_COLUMN])

    def tau_for(val_fused: np.ndarray) -> dict[str, Any]:
        return fu.choose_threshold(
            fu.attack_risk(val_fused, inputs["benign_codes"]), y_val_binary)

    out = []

    # 1. simple average
    val_average = fu.simple_average(val_list)
    threshold = tau_for(val_average)
    started = time.perf_counter()
    average = fu.simple_average(test_list)
    overhead = (time.perf_counter() - started) / max(len(average), 1) * 1e6
    out.append({"name": "fixed: simple average (all models)",
                "keys": keys, "threshold": threshold,
                "validation_scores": validation_scores(
                    y_val_true, val_average, inputs["n_classes"]),
                "result": fu.score_probabilities(
                    "fixed: simple average (all models)", average,
                    inputs["y_test"], inputs["benign_codes"],
                    float(threshold["tau"]), inputs["duplicate_flags"],
                    inputs["label_names"], inputs["n_classes"],
                    overhead_ms=overhead)})

    # 2. majority vote -- a class, not a distribution, so the risk (and tau)
    #    come from the mean probabilities while the prediction comes from the vote
    out.append({"name": "fixed: majority vote (all models)",
                "keys": keys, "threshold": threshold,
                "validation_scores": {
                    "macro_f1": fu.macro_f1(y_val_true,
                                            fu.majority_vote(val_list)),
                    "g_mean": float("nan")},
                "result": fu.score_probabilities(
                    "fixed: majority vote (all models)", average,
                    inputs["y_test"], inputs["benign_codes"],
                    float(threshold["tau"]), inputs["duplicate_flags"],
                    inputs["label_names"], inputs["n_classes"],
                    predicted=fu.majority_vote(test_list))})

    # 3. static weights from the VALIDATION macro-F1 of each model
    macro = {k: fu.macro_f1(y_val_true,
                            np.argmax(np.asarray(val_probs[k]), axis=1))
             for k in keys}
    static = fu.static_weights([macro[k] for k in keys])
    val_static = fu.fuse_probabilities(val_list,
                                       np.tile(static, (len(val_list[0]), 1)))
    threshold_static = tau_for(val_static)
    started = time.perf_counter()
    static_fused = fu.fuse_probabilities(
        test_list, np.tile(static, (len(test_list[0]), 1)))
    overhead = (time.perf_counter() - started) / max(len(static_fused), 1) * 1e6
    out.append({"name": "fixed: static macro-F1 weights (all models)",
                "keys": keys, "threshold": threshold_static,
                "static_weights": {display_label(k): round(float(w), 6)
                                   for k, w in zip(keys, static)},
                "validation_scores": validation_scores(
                    y_val_true, val_static, inputs["n_classes"]),
                "result": fu.score_probabilities(
                    "fixed: static macro-F1 weights (all models)", static_fused,
                    inputs["y_test"], inputs["benign_codes"],
                    float(threshold_static["tau"]), inputs["duplicate_flags"],
                    inputs["label_names"], inputs["n_classes"],
                    overhead_ms=overhead)})
    return out


# --- single models, scored at their own validation threshold ----------------
def single_variants(keys: Sequence[str], val_probs: dict[str, np.ndarray],
                    test_probs: dict[str, np.ndarray], inputs: dict
                    ) -> list[dict[str, Any]]:
    """Every available model on its own, in the display order."""
    y_val_binary = np.asarray(inputs["y_val"]["Label_binary"]).astype("int64")
    y_val_true = np.asarray(inputs["y_val"][TARGET_COLUMN])
    out = []
    for key in ordered(keys):
        threshold = fu.choose_threshold(
            fu.attack_risk(val_probs[key], inputs["benign_codes"]), y_val_binary)
        out.append({"name": display_label(key), "keys": [key],
                    "threshold": threshold,
                    "validation_scores": validation_scores(
                        y_val_true, val_probs[key], inputs["n_classes"]),
                    "result": fu.score_probabilities(
                        display_label(key), test_probs[key], inputs["y_test"],
                        inputs["benign_codes"], float(threshold["tau"]),
                        inputs["duplicate_flags"], inputs["label_names"],
                        inputs["n_classes"])})
    return out


# ===========================================================================
# 13.8.3b  THE SUBSET -- greedy forward selection on VALIDATION only
# ===========================================================================
def greedy_forward_selection(keys: Sequence[str],
                             val_probs: dict[str, np.ndarray], inputs: dict,
                             max_models: int | None = None,
                             metric: str | None = None,
                             verbose: bool = True
                             ) -> tuple[list[str], pd.DataFrame]:
    """Choose a subset by adding the model that most improves the VALIDATION
    score of the adaptive fusion.

    The test probabilities are not an argument: this function cannot see the
    test set, which is what makes the choice of models a validation-only
    decision rather than a claim about the test rows.

    Deterministic by construction: candidates are tried in
    `STEP12_DISPLAY_ORDER`, a candidate must beat the current best by more than
    `EPSILON` to be added, and the first candidate to do so wins a tie.  Stops
    at `SUBSET_MAX_MODELS` or when nothing improves.
    """
    keys = ordered(keys)
    max_models = config.SUBSET_MAX_MODELS if max_models is None else max_models
    metric = config.SUBSET_METRIC if metric is None else metric
    y_val_true = np.asarray(inputs["y_val"][TARGET_COLUMN])
    if max_models < 1:
        raise FusionAllError("SUBSET_MAX_MODELS must be at least 1; got %r"
                             % max_models)

    rows: list[dict[str, Any]] = []

    # step 1: the best single model on validation, in display order on a tie
    singles = {k: subset_metric(validation_scores(
        y_val_true, val_probs[k], inputs["n_classes"]), metric) for k in keys}
    best_single = max(keys, key=lambda k: (singles[k], -keys.index(k)))
    chosen = [best_single]
    score = singles[best_single]
    rows.append({"step": 1, "model added": display_label(best_single),
                 "key added": best_single, "models in subset": 1,
                 "validation %s" % metric: round(score, 6),
                 "improvement": round(score, 6),
                 "decision": "best single model on validation"})
    if verbose:
        print("  [1]  %-40s validation %s %.4f"
              % (display_label(best_single), metric, score))

    # steps 2..max: add whichever remaining model improves the fusion most
    while len(chosen) < min(max_models, len(keys)):
        candidates = [k for k in keys if k not in chosen]
        best_key, best_score = None, score
        for key in candidates:
            fitted = fit_on_validation(chosen + [key], val_probs, inputs)
            value = subset_metric(fitted["validation_scores"], metric)
            rows.append({"step": len(chosen) + 1,
                         "model added": display_label(key), "key added": key,
                         "models in subset": len(chosen) + 1,
                         "validation %s" % metric: round(value, 6),
                         "improvement": round(value - score, 6),
                         "decision": "tried"})
            if value > best_score + config.EPSILON:
                best_key, best_score = key, value
        if best_key is None:
            rows.append({"step": len(chosen) + 1, "model added": "(none)",
                         "key added": "", "models in subset": len(chosen),
                         "validation %s" % metric: round(score, 6),
                         "improvement": 0.0,
                         "decision": "stopped: no candidate improved the "
                                     "validation score"})
            if verbose:
                print("  [-]  no candidate improved the validation %s; stopping "
                      "at %d models" % (metric, len(chosen)))
            break
        chosen.append(best_key)
        score = best_score
        for row in rows:                      # mark the winner of this step
            if (row["step"] == len(chosen) and row["key added"] == best_key
                    and row["decision"] == "tried"):
                row["decision"] = "ADDED"
        if verbose:
            print("  [%d]  + %-38s validation %s %.4f"
                  % (len(chosen), display_label(best_key), metric, score))
    else:
        if len(chosen) >= max_models:
            rows.append({"step": len(chosen), "model added": "(none)",
                         "key added": "", "models in subset": len(chosen),
                         "validation %s" % metric: round(score, 6),
                         "improvement": 0.0,
                         "decision": "stopped: SUBSET_MAX_MODELS = %d reached"
                                     % max_models})
            if verbose:
                print("  [-]  SUBSET_MAX_MODELS = %d reached" % max_models)

    path = pd.DataFrame(rows)
    return ordered(chosen), path


# ===========================================================================
# 13.8.4  THE COMPARISON TABLE
# ===========================================================================
LOWER_IS_BETTER = {"binary FPR", "fusion overhead (ms/1k)"}


def row_from_saved_metrics(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """A comparison row for the existing Step 13 fusion, from its metrics.json.

    F3 is LOADED, never recomputed: the Step 13 numbers in the write-up have to
    be the Step 13 numbers here.  `fusion.comparison_row` turns the saved blocks
    into the same columns every other row uses.
    """
    fusion_block = metrics.get("fusion") or {}
    if not fusion_block:
        raise FusionAllError(
            "the saved Step 13 metrics.json has no \"fusion\" block, so F3 "
            "cannot be read from it")
    return fu.comparison_row({
        "name": name,
        "full": fusion_block.get("full") or {},
        "duplicate_free": fusion_block.get("duplicate_free") or {},
        "binary": fusion_block.get("binary") or {},
        "overhead_ms_per_1000": metrics.get("fusion_overhead_ms_per_1000_flows"),
    })


def comparison_table(variants: Sequence[dict[str, Any]],
                     extra_rows: Sequence[dict[str, Any]] = ()) -> pd.DataFrame:
    """One row per variant, in the order given, with the Step 13 columns."""
    rows = [fu.comparison_row(v["result"]) for v in variants]
    rows.extend(dict(r) for r in extra_rows)
    return pd.DataFrame(rows)


def mark_best(table: pd.DataFrame, tolerance: float = 1e-9) -> pd.DataFrame:
    """A `*` beside the best value in every numeric column, on every tied row.

    Lower is better for the FPR and the overhead.  A tie is marked on each tied
    row: naming one arbitrary winner would be a false statement.
    """
    if table.empty:
        return table.copy()
    marked = table.copy()
    for column in table.columns:
        if column == "configuration":
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        best = values.min() if column in LOWER_IS_BETTER else values.max()
        formatted = []
        for value in values:
            if pd.isna(value):
                formatted.append("")
                continue
            star = "*" if abs(float(value) - float(best)) <= tolerance else " "
            formatted.append("%.4f%s" % (value, star))
        marked[column] = formatted
    return marked


def per_class_tables(variants: Sequence[dict[str, Any]], inputs: dict
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(full, duplicate-free) label x variant F1 tables.

    The full table comes from `fusion.per_class_f1_frame`; the duplicate-free
    one is built the same way on the rows whose exact feature vector does NOT
    also occur in training, which is the conservative view Steps 11-12 report.
    """
    results = [v["result"] for v in variants]
    full = fu.per_class_f1_frame(results)

    clean = ~np.asarray(inputs["duplicate_flags"], dtype=bool)
    if not clean.any():
        return full, pd.DataFrame()
    y_true = np.asarray(inputs["y_test"][TARGET_COLUMN])[clean]
    columns, support = {}, None
    for variant in variants:
        result = variant["result"]
        y_pred = np.asarray(result["y_pred"])[clean]
        table = ev.per_class_table(y_true, y_pred, inputs["label_names"],
                                   inputs["n_classes"])
        columns[result["name"]] = table.set_index("Label")["f1"]
        if support is None:
            support = table.set_index("Label")["support"]
    free = pd.DataFrame(columns)
    free.insert(0, "test rows", support)
    free.index.name = "Label"
    return full, free.reset_index()


def weights_by_class(variant: dict[str, Any], inputs: dict) -> pd.DataFrame:
    """Mean weight per model per TRUE class, for one adaptive variant.

    `fusion.weights_summary_table` groups by context and by true class; the true
    class grouping is the one that says whether the adaptation was right, so
    that is what is kept here, with the display labels as the model names.
    """
    labels = [display_label(k) for k in variant["keys"]]
    summary = fu.weights_summary_table(
        variant["weights"], labels, variant["contexts"],
        np.asarray(inputs["y_test"][TARGET_COLUMN]), inputs["label_names"])
    by_class = summary[summary["grouping"] == "true class"].copy()
    return by_class.rename(columns={"group": "true class"}).drop(
        columns=["grouping"])


def short_label(key: str) -> str:
    """"Approach 1", "B4", ... -- the display label's own short form.

    Taken from `STEP12_DISPLAY_ORDER` rather than invented: the labels there
    read "B4: AE -> PCA -> LSTM", and the part before the colon is what a
    column header can carry without becoming unreadable.
    """
    return display_label(key).split(":")[0].strip()


# ===========================================================================
# 13.8.4b  CHARTS
# ===========================================================================
# The project's five series hues (fusion._SERIES) plus two more, because this
# addendum stacks up to seven models in one bar.  Assigned per MODEL in
# STEP12_DISPLAY_ORDER order and never cycled, so a model keeps its colour when
# the subset changes.  The seven were checked for colour-blind separation and
# the chroma floor against this surface (worst adjacent pair dE 9.1 protan);
# three sit below 3:1 contrast, which is why every segment is directly labelled
# and the same numbers are saved as a CSV beside the chart.
_SERIES7 = list(fu._SERIES) + ["#8a63d2", "#00a9b5"]


def model_colour(key: str) -> str:
    order = [k for k, _label in config.STEP12_DISPLAY_ORDER]
    index = order.index(key) if key in order else len(order)
    return _SERIES7[index % len(_SERIES7)]


def plot_diversity_heatmap(grid: pd.DataFrame, path, title: str,
                           fmt: str = "%.3f", vmin: float = 0.0,
                           vmax: float = 1.0, verbose: bool = True):
    """A square model x model matrix as one sequential hue, light to dark.

    One measure, one hue: the same blue ramp Step 13's reliability heatmap uses.
    Every cell is annotated, so the colour is a second encoding rather than the
    only one.
    """
    plt = fu._pyplot(verbose)
    if plt is None or grid.empty:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    values = grid.to_numpy(dtype="float64")
    cmap = LinearSegmentedColormap.from_list("mscad_blue", fu._SEQ_RAMP)
    size = max(5.5, 1.15 * grid.shape[1] + 3.0)
    fig, ax = plt.subplots(figsize=(size, max(4.2, 0.95 * grid.shape[0] + 2.4)),
                           dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                      interpolation="nearest")
    # the axes carry the short labels ("Approach 1", "B4"); the saved CSV keeps
    # the full ones, so nothing is lost and seven of them still fit
    def short(text) -> str:
        return str(text).split(":")[0].strip()

    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([short(c) for c in grid.columns], rotation=30,
                       ha="right", fontsize=9, color=dl._INK_SOFT)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([short(i) for i in grid.index], fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_title(title, color=dl._INK, fontsize=12, pad=14)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    span = (vmax - vmin) or 1.0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(values[i, j]):
                ax.annotate("--", (j, i), ha="center", va="center", fontsize=8,
                            color=dl._INK_SOFT)
                continue
            dark = (values[i, j] - vmin) / span > 0.55
            ax.annotate(fmt % values[i, j], (j, i), ha="center", va="center",
                        fontsize=8, color="#ffffff" if dark else dl._INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.ax.tick_params(colors=dl._INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)
    return fu._finish(fig, path, verbose)


def plot_weight_stack(by_class: pd.DataFrame, keys: Sequence[str], path,
                      title: str, verbose: bool = True):
    """Mean weight per model within each true class, stacked to 1.

    Step 13's `plot_mean_weights` cycles five hues, which repeats a colour once
    a sixth model joins; this keeps one fixed colour per model for up to seven
    and leaves a 2px surface gap between the segments.
    """
    plt = fu._pyplot(verbose)
    if plt is None or by_class.empty:
        return None
    grid = by_class.pivot(index="true class", columns="model",
                          values="mean weight")
    names = [display_label(k) for k in keys if display_label(k) in grid.columns]
    grid = grid[names]
    rows = [str(v) for v in grid.index]
    positions = np.arange(len(rows), dtype="float64")

    fig, ax = plt.subplots(figsize=(max(8.5, 1.6 * len(rows) + 3.5), 5.4),
                           dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    fu._style(ax)

    bottom = np.zeros(len(rows))
    gap = 0.004                          # the 2px surface gap between segments
    for i, name in enumerate(grid.columns):
        key = [k for k in keys if display_label(k) == name][0]
        values = grid[name].to_numpy(dtype="float64")
        ax.bar(positions, values, 0.6, bottom=bottom + i * gap,
               label=short_label(key), color=model_colour(key))
        for x, (value, base) in enumerate(zip(values, bottom)):
            if value > 0.07:
                ax.annotate("%.2f" % value, (x, base + value / 2), ha="center",
                            va="center", fontsize=8, color="#ffffff")
        bottom = bottom + values
    ax.set_xticks(positions)
    ax.set_xticklabels(rows, rotation=20, ha="right", fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_ylabel("mean weight (sums to 1)", color=dl._INK_SOFT, fontsize=10)
    ax.set_title(title, color=dl._INK, fontsize=12, pad=14)
    ax.set_ylim(0, 1.16)
    legend = ax.legend(frameon=False, loc="upper right", fontsize=8.5,
                       ncol=min(len(grid.columns), 4))
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return fu._finish(fig, path, verbose)


# ===========================================================================
# 13.8.4c  THE EXISTING STEP 13 OUTPUTS -- read, and proved untouched
# ===========================================================================
def step13_fingerprint() -> dict[str, str]:
    """SHA-256 of every file in the Step 13 folder, as it stands now."""
    folder = config.usable_output_dir(config.STEP13_OUTPUT_DIR)
    if folder is None or not Path(folder).exists():
        return {}
    out = {}
    for path in sorted(Path(folder).glob("*")):
        if path.is_file():
            out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def assert_step13_unchanged(before: dict[str, str], verbose: bool = True
                            ) -> tuple[pd.DataFrame, bool]:
    """[OK]/[FAIL] that nothing in step13\\ moved while this addendum ran."""
    after = step13_fingerprint()
    rows = []
    names = sorted(set(before) | set(after))
    for name in names:
        old, new = before.get(name), after.get(name)
        rows.append({"file": name,
                     "sha256 before": (old or "(absent)")[:16],
                     "sha256 after": (new or "(absent)")[:16],
                     "result": "[OK]" if old == new and old is not None
                               else "[FAIL]"})
    if not names:
        rows.append({"file": "(the Step 13 folder is not readable here)",
                     "sha256 before": "", "sha256 after": "",
                     "result": "[SKIP]"})
    table = pd.DataFrame(rows)
    ok = bool((table["result"] != "[FAIL]").all())
    if verbose:
        show_table(table, note="Every file in `step13\\` is SHA-256 "
                               "fingerprinted before this addendum runs and "
                               "again afterwards. This is the add-only "
                               "guarantee, checked rather than asserted.")
        print("[%s]  existing Step 13 outputs unchanged"
              % ("OK" if ok else "FAIL"))
    return table, ok


def load_step13_metrics() -> dict[str, Any] | None:
    """The saved Step 13 metrics.json, or None when Step 13 has not run."""
    folder = config.usable_output_dir(config.STEP13_OUTPUT_DIR)
    if folder is None:
        return None
    path = Path(folder) / config.STEP13_FILES["metrics"]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ===========================================================================
# 13.8.4d  THE FILE THAT CARRIES THE EVIDENCE
# ===========================================================================
def fused_predictions_frame(variants: dict[str, dict[str, Any]], inputs: dict,
                            keys: Sequence[str]) -> pd.DataFrame:
    """One row per test flow: every model's r_m, and each variant's w, P, R.

    Rounded to 8 places rather than 6 for the same reason Step 13 does it:
    seven weights rounded to 6 places can miss 1.0 by enough to fail the 1e-6
    sum check this file is read back and tested against.
    """
    keys = list(keys)
    y_true = np.asarray(inputs["y_test"][TARGET_COLUMN])
    label_names = inputs["label_names"]
    meta = inputs["meta_test"]
    orig = (meta["Orig_Row"].to_numpy()
            if meta is not None and "Orig_Row" in meta.columns
            else np.arange(len(y_true)))

    frame = pd.DataFrame({
        "Orig_Row": orig,
        "y_true": y_true,
        "y_true_label": [label_names.get(int(c), str(c)) for c in y_true],
    })
    if meta is not None and "Source_Capture" in meta.columns:
        frame["Source_Capture"] = meta["Source_Capture"].to_numpy()

    for key in keys:
        frame["r_%s" % short_label(key)] = np.round(
            fu.attack_risk(inputs["test_probs"][key], inputs["benign_codes"]), 8)

    for tag, variant in variants.items():
        for j, key in enumerate(variant["keys"]):
            frame["w_%s_%s" % (tag, short_label(key))] = np.round(
                np.asarray(variant["weights"], dtype="float64")[:, j], 8)
        fused = np.asarray(variant["fused"], dtype="float64")
        for c in range(inputs["n_classes"]):
            frame["P_%s_%s" % (tag, label_names.get(c, str(c)))] = np.round(
                fused[:, c], 8)
        risk = np.asarray(variant["risk"], dtype="float64")
        tau = float(variant["threshold"]["tau"])
        predicted = np.argmax(fused, axis=1)
        frame["R_%s" % tag] = np.round(risk, 8)
        frame["class_%s" % tag] = [label_names.get(int(c), str(c))
                                   for c in predicted]
        frame["decision_%s" % tag] = (risk >= tau).astype("int64")
    return frame


# ===========================================================================
# 13.8.5  SUMMARY, WRITTEN FROM THE COMPUTED NUMBERS
# ===========================================================================
def _value(table: pd.DataFrame, name: str, column: str) -> float:
    rows = table[table["configuration"] == name]
    if rows.empty or column not in table.columns:
        return float("nan")
    return float(pd.to_numeric(rows.iloc[0][column], errors="coerce"))


def _best_single(table: pd.DataFrame, singles: Sequence[str],
                 column: str) -> tuple[str, float]:
    """(name, value) of the best single model in one column."""
    pairs = [(n, _value(table, n, column)) for n in singles]
    pairs = [(n, v) for n, v in pairs if not np.isnan(v)]
    if not pairs:
        return "(none)", float("nan")
    return max(pairs, key=lambda pair: pair[1])


def _fmt(value: float, places: int = 4) -> str:
    return "n/a" if value is None or np.isnan(value) else ("%%.%df" % places) % value


def build_summary(out: dict[str, Any]) -> str:
    """The 13.8.5 discussion, every claim traced to a computed number."""
    table = out["comparison"]
    singles = [display_label(k) for k in out["fused_keys"]]
    lines: list[str] = []
    lines.append("### 13.8.5 Fusion across all approaches and baselines -- "
                 "what the numbers say")
    lines.append("")

    available = out["availability"]
    missing = available[~available["available"]]
    lines.append("**Models.** %d of %d configurations were available and fused: "
                 "%s." % (len(out["fused_keys"]), len(available),
                          ", ".join(singles)))
    if len(missing):
        lines.append("Skipped, not trained on this machine: %s. Nothing was "
                     "trained to fill the gap."
                     % ", ".join(missing["model"].tolist()))
    lines.append("")

    # --- F3 vs FALL vs FSEL vs the best single model --------------------
    lines.append("**Does fusing everything help?**")
    for column, label in (("F1 (macro)", "macro-F1"),
                          ("G-mean", "G-mean"),
                          ("F1 macro (dup-free)", "macro-F1 (duplicate-free)"),
                          ("G-mean (dup-free)", "G-mean (duplicate-free)"),
                          ("binary FPR", "binary FPR (lower is better)")):
        best_name, best_value = _best_single(table, singles, column)
        f3 = _value(table, F3_NAME, column)
        fall = _value(table, FALL_NAME, column)
        fsel = _value(table, FSEL_NAME, column)
        pieces = ["best single %s = %s (%s)" % (label, _fmt(best_value),
                                                best_name),
                  "F3 = %s" % _fmt(f3), "FALL = %s" % _fmt(fall)]
        if not np.isnan(fsel):
            pieces.append("FSEL = %s" % _fmt(fsel))
        lines.append("- %s." % "; ".join(pieces))
        lower = column in LOWER_IS_BETTER
        contenders = {"FALL": fall, "FSEL": fsel, "F3": f3,
                      "the best single model": best_value}
        contenders = {k: v for k, v in contenders.items() if not np.isnan(v)}
        if contenders:
            winner = (min if lower else max)(contenders, key=contenders.get)
            tied = [k for k, v in contenders.items()
                    if abs(v - contenders[winner]) <= 1e-9]
            if len(tied) == len(contenders):
                lines.append("  - nothing separates on this metric: every "
                             "contender scores %s." % _fmt(contenders[winner]))
            elif len(tied) > 1:
                lines.append("  - best on this metric: %s (a tie)."
                             % " and ".join(sorted(tied)))
            else:
                lines.append("  - best on this metric: %s." % winner)
    lines.append("")

    # --- the subset, against the diversity check ------------------------
    if out.get("selected_keys"):
        lines.append("**Which models the greedy search kept.** %s, chosen on "
                     "the validation split alone (metric: %s, "
                     "SUBSET_MAX_MODELS = %d)."
                     % (", ".join(display_label(k)
                                  for k in out["selected_keys"]),
                        config.SUBSET_METRIC, config.SUBSET_MAX_MODELS))
        path = out.get("selection_path")
        if path is not None and len(path):
            added = path[path["decision"].isin(["ADDED",
                                                "best single model on "
                                                "validation"])]
            steps = ["%s (validation %s %s)"
                     % (row["model added"], config.SUBSET_METRIC,
                        _fmt(float(row["validation %s" % config.SUBSET_METRIC])))
                     for _i, row in added.iterrows()]
            lines.append("- Selection path: %s." % " -> ".join(steps))
            stopped = path[path["decision"].str.startswith("stopped")]
            if len(stopped):
                lines.append("- Stopped because %s."
                             % stopped.iloc[-1]["decision"]
                             .replace("stopped: ", ""))
    else:
        lines.append("**Subset selection** was not run "
                     "(SUBSET_SELECTION = %r)." % config.SUBSET_SELECTION)
    lines.append("")

    # --- diversity ------------------------------------------------------
    disagreement = out.get("disagreement")
    if disagreement is not None and len(disagreement) > 1:
        values = disagreement.to_numpy(dtype="float64").copy()
        np.fill_diagonal(values, np.nan)
        flat = [(disagreement.index[i], disagreement.columns[j], values[i, j])
                for i in range(values.shape[0])
                for j in range(values.shape[1]) if i < j]
        flat = [f for f in flat if not np.isnan(f[2])]
        if flat:
            lowest = min(flat, key=lambda f: f[2])
            highest = max(flat, key=lambda f: f[2])
            lines.append("**Diversity (validation rows).** The most "
                         "interchangeable pair is %s and %s, disagreeing on "
                         "%.2f%% of flows; the most diverse is %s and %s at "
                         "%.2f%%. Models that almost always agree can only add "
                         "weight to an opinion the fusion already has."
                         % (lowest[0], lowest[1], 100 * lowest[2],
                            highest[0], highest[1], 100 * highest[2]))
            if out.get("selected_keys"):
                chosen = {display_label(k) for k in out["selected_keys"]}
                within = [f for f in flat
                          if f[0] in chosen and f[1] in chosen]
                if within:
                    mean_within = float(np.mean([f[2] for f in within]))
                    mean_all = float(np.mean([f[2] for f in flat]))
                    lines.append("- Mean pairwise disagreement inside the "
                                 "chosen subset %.4f against %.4f across all "
                                 "available pairs, so the search %s a more "
                                 "diverse-than-average set."
                                 % (mean_within, mean_all,
                                    "did keep" if mean_within > mean_all
                                    else "did not keep"))
    lines.append("")

    # --- caveats ---------------------------------------------------------
    lines.append("**Caveats.**")
    lines.append("- The reliabilities and every threshold come from the "
                 "validation part of the Step 9 split, which Step 10 also used "
                 "for early stopping, so they are slightly optimistic. The "
                 "subset choice has the same caveat: it is a validation "
                 "decision, not a test-set one.")
    rare = out.get("rare_classes") or []
    if rare:
        lines.append("- Rare classes: %s. A per-class F1 computed on that many "
                     "rows moves in large steps and should not be read as a "
                     "precise figure."
                     % ", ".join("%s (%d test rows)" % (name, count)
                                 for name, count in rare))
    lines.append("- One seed (RANDOM_STATE = %d). The gaps above are "
                 "differences between single runs, not estimates with a spread."
                 % config.RANDOM_STATE)
    lines.append("- F3 is **read from `step13\\metrics.json`**, not recomputed, "
                 "so the Step 13 row here is the Step 13 row in the write-up.")
    lines.append("")
    lines.append("**Explainability.** Step 14 explains the original three-model "
                 "fusion (F3), unchanged. Explaining FALL or FSEL -- per-model "
                 "contributions over up to %d models -- would be a separate "
                 "addendum; nothing in Step 14 is altered by this section."
                 % len(out["fused_keys"]))
    return "\n".join(lines)


# ===========================================================================
# 13.8.6  SAVE, THEN READ BACK AND CHECK
# ===========================================================================
def build_metrics(out: dict[str, Any], saved_files: dict[str, str],
                  summary: str | None = None) -> dict[str, Any]:
    """Everything needed to reproduce these numbers, as one JSON object."""
    table = out["comparison"]
    variants = {}
    for tag, variant in out["variants"].items():
        result = variant["result"]
        variants[tag] = {
            "models": [display_label(k) for k in variant["keys"]],
            "keys": list(variant["keys"]),
            "threshold": variant["threshold"],
            "validation_scores": variant.get("validation_scores"),
            "full": result["full"], "duplicate_free": result["duplicate_free"],
            "binary": result["binary"],
            "binary_duplicate_free": result["binary_duplicate_free"],
            "overhead_ms_per_1000_flows": result.get("overhead_ms_per_1000"),
            "mean_weight": {
                display_label(k): round(float(np.asarray(
                    variant["weights"], dtype="float64")[:, j].mean()), 6)
                for j, k in enumerate(variant["keys"])},
        }
    return {
        "step": "13.8",
        "name": "Adaptive Evidence Fusion across all approaches and baselines",
        "offered": [display_label(k) for k in out["offered_keys"]],
        "fused": [display_label(k) for k in out["fused_keys"]],
        "skipped": out["availability"].loc[
            ~out["availability"]["available"], "model"].tolist(),
        "test_rows": int(len(out["y_test"])),
        "duplicate_free_rows": int((~np.asarray(out["duplicate_flags"],
                                                dtype=bool)).sum()),
        "validation_rows": int(len(out["y_val"])),
        "classes": int(out["n_classes"]),
        "settings": {
            "FUSION_ALL_MODELS": list(config.FUSION_ALL_MODELS),
            "SUBSET_SELECTION": config.SUBSET_SELECTION,
            "SUBSET_MAX_MODELS": config.SUBSET_MAX_MODELS,
            "SUBSET_METRIC": config.SUBSET_METRIC,
            # reused from the Step 13 block, unchanged
            "RELIABILITY_METRIC": config.RELIABILITY_METRIC,
            "SHRINKAGE_K": config.SHRINKAGE_K,
            "CONFIDENCE_TERM": config.CONFIDENCE_TERM,
            "WEIGHT_MODE": config.WEIGHT_MODE,
            "WEIGHT_TEMPERATURE": config.WEIGHT_TEMPERATURE,
            "CONTEXT_MODE": config.CONTEXT_MODE,
            "THRESHOLD_RULE": config.THRESHOLD_RULE,
            "TARGET_FPR": config.TARGET_FPR,
            "EPSILON": config.EPSILON,
        },
        "fusion_models_step13": list(config.FUSION_MODELS),
        "selected_keys": list(out.get("selected_keys") or []),
        "selected": [display_label(k) for k in (out.get("selected_keys") or [])],
        "validation_macro_f1_per_model": {
            display_label(k): round(float(v), 6)
            for k, v in (out.get("validation_macro_f1") or {}).items()},
        "variants": variants,
        "best_per_metric": {
            column: {"configuration":
                     (table.loc[pd.to_numeric(table[column], errors="coerce")
                                .idxmin() if column in LOWER_IS_BETTER
                                else pd.to_numeric(table[column],
                                                   errors="coerce").idxmax(),
                                "configuration"]
                      if pd.to_numeric(table[column],
                                       errors="coerce").notna().any() else None)}
            for column in ("F1 (macro)", "G-mean", "F1 macro (dup-free)",
                           "G-mean (dup-free)", "binary FPR")
            if column in table.columns},
        "step13_unchanged": bool(out.get("step13_unchanged")),
        # the fingerprint travels with the metrics so the resumable path can
        # re-check it: a reader who opens this notebook a week later still
        # learns whether anything in step13\ has moved since
        "step13_fingerprint": dict(out.get("step13_before") or {}),
        "alignment_ok": bool(out.get("alignment_ok")),
        "leakage_note": ("the reliabilities, every threshold and the choice of "
                         "which models to include come from the validation part "
                         "of the Step 9 split; the test set is scored once with "
                         "all of them held fixed"),
        "files": saved_files,
        "summary": summary,
    }


def verify_saved(folder: Path, out: dict[str, Any], verbose: bool = True
                 ) -> tuple[pd.DataFrame, bool]:
    """Read every saved file back from disk and check it.

    Nothing is trusted from memory: the weights, the fused distributions and the
    risk are re-tested from the CSV as a reader would find it.
    """
    names = config.STEP13ALL_FILES
    checks: list[dict[str, Any]] = []
    expected_rows = int(len(out["y_test"]))

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    for key in ("availability", "disagreement", "risk_corr", "reliability",
                "fused_predictions", "comparison", "per_class", "metrics",
                "thresholds", "summary"):
        path = folder / names[key]
        add("%s was written" % names[key], path.exists(),
            "%s bytes" % f"{path.stat().st_size:,}" if path.exists()
            else "missing")

    path = folder / names["fused_predictions"]
    if path.exists():
        frame = pd.read_csv(path, low_memory=False)
        add("fused_predictions_all.csv has one row per test flow",
            len(frame) == expected_rows,
            "%s rows, expected %s" % (f"{len(frame):,}",
                                      f"{expected_rows:,}"))
        for tag in out["variants"]:
            weight_columns = [c for c in frame.columns
                              if c.startswith("w_%s_" % tag)]
            if weight_columns:
                totals = frame[weight_columns].sum(axis=1)
                worst = float(np.abs(totals - 1.0).max())
                add("%s weights sum to 1 per flow" % tag, worst < 1e-6,
                    "worst deviation %.2e over %d models" % (worst,
                                                             len(weight_columns)))
            p_columns = [c for c in frame.columns
                         if c.startswith("P_%s_" % tag)]
            if p_columns:
                totals = frame[p_columns].sum(axis=1)
                worst = float(np.abs(totals - 1.0).max())
                add("%s fused rows sum to 1" % tag, worst < 1e-6,
                    "worst deviation %.2e over %d classes" % (worst,
                                                              len(p_columns)))
            risk_column = "R_%s" % tag
            if risk_column in frame.columns:
                values = frame[risk_column].to_numpy(dtype="float64")
                add("%s risk is inside [0, 1]" % tag,
                    bool(values.min() >= -1e-9 and values.max() <= 1 + 1e-9),
                    "min %.6f, max %.6f" % (values.min(), values.max()))

    path = folder / names["comparison"]
    if path.exists():
        table = pd.read_csv(path)
        add("the comparison table carries every variant",
            len(table) == len(out["comparison"]),
            "%d rows" % len(table))
        add("the comparison table uses the display labels",
            not any(str(v).startswith("single: ") for v in table["configuration"]),
            ", ".join(str(v) for v in table["configuration"].head(3)))

    path = folder / names["metrics"]
    if path.exists():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        add("metrics_all.json records the fused models",
            len(metrics.get("fused", [])) == len(out["fused_keys"]),
            ", ".join(metrics.get("fused", [])))

    add("the existing Step 13 outputs are byte-identical",
        bool(out.get("step13_unchanged")),
        "SHA-256 before and after, %d files"
        % len(out.get("step13_before") or {}))

    table = pd.DataFrame(checks)
    ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table, all_columns=True,
                   note="Read back from disk after writing, not asserted from "
                        "memory.")
    return table, ok


def save_step13all(out: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Write every 13.8 output into step13_all\\, then verify it from disk."""
    names = config.STEP13ALL_FILES
    result: dict[str, Any] = {}
    folder = prepare_output_dir(config.STEP13ALL_OUTPUT_DIR,
                                "the Step 13.8 outputs", verbose=verbose)
    if folder is None:
        result.update({"status": "skipped_not_windows", "ok": False})
        return result
    folder = Path(folder)

    out["availability"].to_csv(folder / names["availability"], index=False)
    out["disagreement"].to_csv(folder / names["disagreement"])
    out["risk_corr"].to_csv(folder / names["risk_corr"])
    out["reliability"].to_csv(folder / names["reliability"], index=False)
    out["comparison"].to_csv(folder / names["comparison"], index=False)
    out["comparison_marked"].to_csv(folder / names["comparison_marked"],
                                    index=False)
    out["fused_predictions"].to_csv(folder / names["fused_predictions"],
                                    index=False)
    if len(out["per_class"]):
        out["per_class"].to_csv(folder / names["per_class"], index=False)
    if out.get("per_class_free") is not None and len(out["per_class_free"]):
        out["per_class_free"].to_csv(folder / names["per_class_free"],
                                     index=False)
    if out.get("selection_path") is not None and len(out["selection_path"]):
        out["selection_path"].to_csv(folder / names["selection_path"],
                                     index=False)
    for tag, key in (("FALL", "weights_all"), ("FSEL", "weights_subset")):
        frame = (out.get("weights_by_class") or {}).get(tag)
        if frame is not None and len(frame):
            frame.to_csv(folder / names[key], index=False)
    (folder / names["thresholds"]).write_text(json.dumps(
        {tag: variant["threshold"] for tag, variant in out["variants"].items()},
        indent=2, default=str), encoding="utf-8")

    # ---- charts ----------------------------------------------------------
    charts: dict[str, str] = {}
    drawn = fu.plot_comparison_bars(
        out["comparison"], ["F1 (macro)", "G-mean", "binary FPR"],
        "Full test set -- F1 and G-mean higher is better, FPR lower",
        folder / names["chart_metrics"], verbose=verbose)
    if drawn:
        charts["metrics"] = str(drawn)
    drawn = fu.plot_comparison_bars(
        out["comparison"], ["F1 macro (dup-free)", "G-mean (dup-free)"],
        "Duplicate-free test subset -- the conservative comparison",
        folder / names["chart_metrics_free"], verbose=verbose)
    if drawn:
        charts["metrics_duplicate_free"] = str(drawn)
    for tag, key, title in (
            ("FALL", "chart_weights_all",
             "Mean fusion weight per model, by true class -- FALL"),
            ("FSEL", "chart_weights_subset",
             "Mean fusion weight per model, by true class -- FSEL")):
        frame = (out.get("weights_by_class") or {}).get(tag)
        variant = out["variants"].get(tag)
        if frame is not None and len(frame) and variant is not None:
            drawn = plot_weight_stack(frame, variant["keys"],
                                      folder / names[key], title,
                                      verbose=verbose)
            if drawn:
                charts["weights_%s" % tag] = str(drawn)
    drawn = plot_diversity_heatmap(
        out["disagreement"], folder / names["chart_diversity"],
        "Pairwise disagreement on the validation rows (share of flows)",
        verbose=verbose)
    if drawn:
        charts["diversity"] = str(drawn)
    result["charts"] = charts

    summary = build_summary(out)
    (folder / names["summary"]).write_text(summary, encoding="utf-8")
    result["summary"] = summary

    saved_files = {key: names[key] for key in names
                   if (folder / names[key]).exists()}
    metrics = build_metrics(out, saved_files, summary=summary)
    (folder / names["metrics"]).write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    result["metrics"] = metrics

    report, ok = verify_saved(folder, out, verbose=verbose)
    report.to_csv(folder / names["report"], index=False)
    result.update({"report": report, "ok": ok, "status": "written",
                   "folder": str(folder)})
    if verbose:
        print("  [%s]  Step 13.8 outputs written to %s"
              % ("OK" if ok else "FAIL", folder))
    return result


# ===========================================================================
# 13.8.7  RUN EVERYTHING
# ===========================================================================
def step13all_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP13ALL_OUTPUT_DIR)
    if folder is None:
        return False
    return all((Path(folder) / config.STEP13ALL_FILES[key]).exists()
               for key in ("metrics", "comparison", "fused_predictions",
                           "availability"))


def load_step13all(verbose: bool = True) -> dict[str, Any]:
    """Read a finished 13.8 back from disk.  Nothing is recomputed."""
    folder = config.usable_output_dir(config.STEP13ALL_OUTPUT_DIR)
    if folder is None:
        raise FusionAllError("the Step 13.8 folder is not usable here -- %s"
                             % config.why_unusable(config.STEP13ALL_OUTPUT_DIR))
    folder = Path(folder)
    names = config.STEP13ALL_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    for key in ("availability", "reliability", "comparison",
                "comparison_marked", "fused_predictions", "per_class",
                "per_class_free", "selection_path", "report"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    for key in ("disagreement", "risk_corr"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, index_col=0)
    weights = {}
    for tag, key in (("FALL", "weights_all"), ("FSEL", "weights_subset")):
        path = folder / names[key]
        if path.exists():
            weights[tag] = pd.read_csv(path)
    out["weights_by_class"] = weights
    for key in ("thresholds", "metrics"):
        path = folder / names[key]
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    path = folder / names["summary"]
    if path.exists():
        out["summary"] = path.read_text(encoding="utf-8")
    else:
        out["summary"] = (out.get("metrics") or {}).get("summary")

    # the add-only guarantee, re-checked on the resumable path: the fingerprint
    # saved when these outputs were written is compared against step13\ as it
    # stands now, so the notebook's [OK] line means something on this run too
    recorded = (out.get("metrics") or {}).get("step13_fingerprint") or {}
    if recorded:
        table, ok = assert_step13_unchanged(recorded, verbose=False)
        out["step13_unchanged_table"] = table
        out["step13_unchanged"] = ok
    else:
        out["step13_unchanged"] = bool(
            (out.get("metrics") or {}).get("step13_unchanged"))
    if verbose:
        metrics = out.get("metrics") or {}
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP13ALL = False)"),
            ("folder", str(folder)),
            ("models fused", ", ".join(metrics.get("fused", []))),
            ("subset chosen", ", ".join(metrics.get("selected", [])) or "(none)"),
            ("variants compared", len(out.get("comparison", []))),
        ]), note="Set `OVERWRITE_STEP13ALL = True` in config.py to recompute.")
    return out


def run_step13all(overwrite: bool | None = None, keys: Sequence[str] | None = None,
                  verbose: bool = True) -> dict[str, Any]:
    """STEP 13.8 end to end: load every model, check diversity, fuse, compare.

    Order of operations, and why it is this order:

      1. fingerprint `step13\\` so the add-only guarantee can be proved;
      2. load every available model, skipping what is not on disk;
      3. validation and test probabilities, cross-checked and aligned;
      4. diversity, on validation rows;
      5. the three adaptive variants -- F3 read from disk, FALL over everything
         available, FSEL over a validation-chosen subset -- plus the fixed
         baselines, each with its own validation threshold;
      6. the test set, scored once;
      7. save, read back, verify, and re-fingerprint `step13\\`.
    """
    overwrite = (config.OVERWRITE_STEP13ALL if overwrite is None else overwrite)
    if not overwrite and step13all_outputs_exist():
        return load_step13all(verbose=verbose)

    step13_before = step13_fingerprint()
    offered = model_keys(keys, verbose=verbose)

    # ---- inputs, exactly as Step 13 reads them -------------------------
    loaded = ev.load_evaluation_inputs(verbose=False)
    y_test, meta_test = loaded["y_test"], loaded.get("meta_test")
    label_names = ev.label_names_from_mapping(loaded["label_mapping"])
    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder8 is None:
        raise FusionAllError("the Step 8 outputs are not readable here -- %s"
                             % config.why_unusable(config.STEP8_OUTPUT_DIR))
    y_train = pd.read_csv(folder8 / config.STEP8_FILES["y_train"],
                          low_memory=False)
    n_classes = int(max(int(y_train[TARGET_COLUMN].max()),
                        int(y_test[TARGET_COLUMN].max())) + 1)
    benign_codes = ev.benign_codes_from(y_test, label_names)
    class_order = [label_names.get(c, str(c)) for c in range(n_classes)]
    duplicate_flags = np.zeros(len(y_test), dtype=bool)
    if loaded["X_train"] is not None and loaded["X_test"] is not None:
        duplicate_flags = ev.duplicate_mask(loaded["X_train"], loaded["X_test"])

    # ---- 13.8.1 the models ---------------------------------------------
    if verbose:
        show_title("13.8.1 -- loading every available model", level=4)
    bundles, availability = available_bundles(offered, verbose=verbose)
    fused_keys = ordered(list(bundles))

    val_probs, y_val, _positions = fu.validation_probabilities(
        bundles, y_train, verbose=verbose)
    test_probs, match_table = test_probabilities(bundles, y_test, meta_test,
                                                 verbose=verbose)
    alignment_table, alignment_ok = alignment(test_probs, y_test, meta_test,
                                              class_order, verbose=verbose)

    inputs = {"y_test": y_test, "y_val": y_val, "y_train": y_train,
              "meta_test": meta_test, "label_names": label_names,
              "n_classes": n_classes, "benign_codes": benign_codes,
              "duplicate_flags": duplicate_flags, "test_probs": test_probs}
    y_val_true = np.asarray(y_val[TARGET_COLUMN])
    validation_macro = {k: fu.macro_f1(
        y_val_true, np.argmax(np.asarray(val_probs[k]), axis=1))
        for k in fused_keys}
    availability["validation macro-F1"] = [
        round(validation_macro.get(k, float("nan")), 6)
        for k in availability["key"]]
    availability["validation G-mean"] = [
        round(validation_scores(y_val_true, val_probs[k], n_classes)["g_mean"], 6)
        if k in val_probs else float("nan") for k in availability["key"]]

    # ---- 13.8.2 diversity, validation only -----------------------------
    if verbose:
        show_title("13.8.2 -- diversity on the validation rows", level=4)
    disagreement = disagreement_table(val_probs, fused_keys)
    risk_corr = risk_correlation_table(val_probs, benign_codes, fused_keys)
    if verbose:
        show_table(disagreement.reset_index(), all_columns=True,
                   note="Share of validation flows where the two models predict "
                        "a different class. A pair at 0.00 is interchangeable "
                        "here, and fusing it cannot add information.")
        show_table(risk_corr.reset_index(), all_columns=True,
                   note="Pearson correlation of the attack risk r_m on the same "
                        "rows. Highly correlated risks move together, so a "
                        "weight shift between them changes little.")

    # ---- 13.8.3 the variants -------------------------------------------
    if verbose:
        show_title("13.8.3 -- fusion variants", level=4)
    variants: dict[str, dict[str, Any]] = {}
    fall = adaptive_variant(fused_keys, val_probs, test_probs, inputs,
                            FALL_NAME, verbose=verbose)
    variants["FALL"] = fall

    selected_keys: list[str] = []
    selection_path = pd.DataFrame()
    if str(config.SUBSET_SELECTION) == "greedy_forward":
        if verbose:
            print("  greedy forward selection on the validation split "
                  "(metric: %s, at most %d models):"
                  % (config.SUBSET_METRIC, config.SUBSET_MAX_MODELS))
        selected_keys, selection_path = greedy_forward_selection(
            fused_keys, val_probs, inputs, verbose=verbose)
        if len(selected_keys) >= 2:
            variants["FSEL"] = adaptive_variant(selected_keys, val_probs,
                                                test_probs, inputs, FSEL_NAME,
                                                verbose=verbose)
        elif verbose:
            show_note("The greedy search kept a single model (%s), so there is "
                      "no FSEL fusion to report -- a one-model \"fusion\" is "
                      "that model."
                      % display_label(selected_keys[0] if selected_keys else "?"))
    elif verbose:
        show_note("SUBSET_SELECTION = %r, so no subset variant is computed."
                  % config.SUBSET_SELECTION)

    singles = single_variants(fused_keys, val_probs, test_probs, inputs)
    fixed = fixed_variants(fused_keys, val_probs, test_probs, inputs)

    # ---- 13.8.4 the comparison, on the untouched test set --------------
    if verbose:
        show_title("13.8.4 -- the untouched test set, scored once", level=4)
    extra_rows = []
    step13_metrics = load_step13_metrics()
    if step13_metrics is not None:
        extra_rows.append(row_from_saved_metrics(F3_NAME, step13_metrics))
    elif verbose:
        show_note("`step13\\metrics.json` is not on this machine, so the F3 row "
                  "is omitted rather than recomputed. Run Step 13 first to "
                  "include it.")

    ordered_variants = singles + fixed + [variants["FALL"]]
    if "FSEL" in variants:
        ordered_variants.append(variants["FSEL"])
    comparison = comparison_table(ordered_variants, extra_rows)
    # F3 is a fusion, so it belongs beside the other two rather than last
    if extra_rows:
        order = [r for r in comparison["configuration"]
                 if r not in (F3_NAME, FALL_NAME, FSEL_NAME)]
        order += [n for n in (F3_NAME, FALL_NAME, FSEL_NAME)
                  if n in set(comparison["configuration"])]
        comparison = (comparison.set_index("configuration").loc[order]
                      .reset_index())
    marked = mark_best(comparison)

    per_class, per_class_free = per_class_tables(ordered_variants, inputs)
    weights_frames = {tag: weights_by_class(variant, inputs)
                      for tag, variant in variants.items()}
    predictions = fused_predictions_frame(variants, inputs, fused_keys)

    support = ev.per_class_table(np.asarray(y_test[TARGET_COLUMN]),
                                 np.asarray(y_test[TARGET_COLUMN]),
                                 label_names, n_classes)
    rare = [(row["Label"], int(row["support"])) for _i, row in support.iterrows()
            if int(row["support"]) <= 20]

    out: dict[str, Any] = {
        "offered_keys": offered, "fused_keys": fused_keys,
        "selected_keys": selected_keys, "selection_path": selection_path,
        "availability": availability, "disagreement": disagreement,
        "risk_corr": risk_corr, "variants": variants,
        "singles": singles, "fixed": fixed, "results": ordered_variants,
        "comparison": comparison, "comparison_marked": marked,
        "per_class": per_class, "per_class_free": per_class_free,
        "weights_by_class": weights_frames, "fused_predictions": predictions,
        "reliability": variants["FALL"]["reliability"],
        "probability_match": match_table, "alignment": alignment_table,
        "alignment_ok": alignment_ok,
        "validation_macro_f1": validation_macro,
        "step13_metrics": step13_metrics, "step13_before": step13_before,
        "rare_classes": rare,
        "y_test": y_test, "y_val": y_val, "meta_test": meta_test,
        "label_names": label_names, "n_classes": n_classes,
        "benign_codes": benign_codes, "duplicate_flags": duplicate_flags,
        "test_probs": test_probs, "val_probs": val_probs,
    }

    # ---- the add-only guarantee, before anything is written ------------
    unchanged_table, unchanged_ok = assert_step13_unchanged(step13_before,
                                                            verbose=verbose)
    out["step13_unchanged_table"] = unchanged_table
    out["step13_unchanged"] = unchanged_ok

    saved = save_step13all(out, verbose=verbose)
    out.update({"status": saved.get("status", "written"), "saved": saved,
                "ok": bool(saved.get("ok") and unchanged_ok and alignment_ok)})
    for key in ("report", "metrics", "summary", "charts", "folder"):
        if key in saved:
            out[key] = saved[key]

    # and again afterwards: writing the addendum must not have touched step13\
    after_table, after_ok = assert_step13_unchanged(step13_before, verbose=False)
    out["step13_unchanged_after_write"] = after_ok
    out["step13_unchanged_table"] = after_table
    if not after_ok:
        raise FusionAllError(
            "writing the Step 13.8 outputs changed a file in the Step 13 "
            "folder. That must never happen; the addendum writes only to %s."
            % config.STEP13ALL_OUTPUT_DIR)

    if verbose:
        show_title("Step 13.8 -- comparison", level=3)
        show_table(marked, all_columns=True,
                   note="Every available single model, the fixed-weight fusions "
                        "over all of them, the Step 13 three-model fusion (F3, "
                        "read from disk) and the two adaptive variants. `*` "
                        "marks the best value in each column, on every tied "
                        "row; lower is better for the FPR and the overhead.")
        print("[%s]  existing Step 13 outputs unchanged"
              % ("OK" if after_ok else "FAIL"))
    return out


# ===========================================================================
# Step 12 completion -- the eighth configuration's hue  (APPENDED, add-only)
# ===========================================================================
# Baseline 6 joins STEP12_DISPLAY_ORDER and FUSION_ALL_MODELS, so the series
# above would wrap at index 7 and hand B6 Approach 1's blue.  Appending the
# eighth hue from config.MODEL_COLOURS keeps one colour per model.  No line
# above is changed: model_colour() reads this list at call time.
_eighth_hue = (getattr(config, "MODEL_COLOURS", {}) or {}).get(
    getattr(config, "BASELINE6_NAME", ""), "")
if _eighth_hue and _eighth_hue not in _SERIES7:
    _SERIES7 = _SERIES7 + [_eighth_hue]
del _eighth_hue
