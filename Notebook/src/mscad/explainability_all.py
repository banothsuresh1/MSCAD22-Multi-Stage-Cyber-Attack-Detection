r"""
src/mscad/explainability_all.py -- STEP 14 ADDENDUM: explainability across
every approach and baseline, and across every fusion variant.

NEW FILE.  Nothing existing is modified: explainability.py, fusion.py,
fusion_all.py, comparison.py, experiments*.py, models*.py, config.py's Step 14
block and every file in C:\MSCAD2022\outputs\step14\ are left exactly as they
are.  Step 14 still explains the three-model fusion; this sits beside it.

What this adds
--------------
Step 14 attributes the decision of the three fused models.  Three questions it
does not answer:

  * what do the OTHER configurations -- Approach 2, B1, B4, B5 -- base their
    decisions on, on their own named features?
  * where do the models agree and disagree in their reasoning, and whose
    explanations are the most faithful?
  * how does each FUSION VARIANT (F3, FALL, FSEL) split its risk across models?

The Step 14 arithmetic is CALLED, not copied: `integrated_gradients`,
`completeness_table`, `global_importance`, `perturbation_drops`,
`deletion_check`, `attribution_agreement`, `stability`, `fusion_breakdown`,
`fusion_consistency`, `build_alert_report` and the vocabulary helpers all come
from `explainability`, which is not edited.  This module adds only what Step 14
has no reason to own: composed models for the three configurations it cannot
compose itself, the cross-model comparison, the per-variant contributions, and
the presentation.

Integrated Gradients runs on `tf.GradientTape` through `explainability`, so
nothing here reaches for the Keras 3-only `ops` namespace or any other API that
Keras 2 does not have; it runs on the same Keras the models were saved with.

Nothing is retrained
--------------------
Every model, encoder, PCA, fusion weight and threshold is loaded from the step
that wrote it.  A configuration that is not on disk is skipped by name, with
the cells that produce it, and the rest are still explained.

No leakage
----------
The IG baseline x' is the median of the TRAINING benign rows in each model's own
input space; tau comes from Step 13 unchanged, tau_s from the Step 9 validation
split, and the alert variant is chosen on VALIDATION macro-F1.  No test label
chooses any setting.
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
from . import explainability as xai  # noqa: E402
from . import fusion as fz  # noqa: E402
from . import fusion_all as fa  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = xai.TARGET_COLUMN
TARGET_CLASS = xai.TARGET_CLASS
TARGET_RISK = xai.TARGET_RISK
BAND_ATTACK = xai.BAND_ATTACK
BAND_SUSPICIOUS = xai.BAND_SUSPICIOUS
NA_TEXT = "--"                      # how a feature outside a model's input prints
NA_FOOTNOTE = "-- = not an input of this model"

# How far a recomputed 13.8 weight may sit from the saved one before the
# reliability behind it is treated as unrecoverable -- see recover_reliability.
WEIGHT_RECOVERY_TOL = 5e-3

VARIANTS = ("F3", "FALL", "FSEL")
VARIANT_TITLES = {
    "F3": "F3 -- Step 13 fusion (Approach 1 + B2 + B3)",
    "FALL": "FALL -- 13.8 fusion over all available models",
    "FSEL": "FSEL -- 13.8 fusion over the validation-chosen subset",
}


class XaiAllError(Exception):
    """Raised when the addendum cannot continue without guessing."""


# ===========================================================================
# 14.9.0  NAMES, COLOURS AND THE MODELS THEMSELVES
# ===========================================================================
def display_label(key: str) -> str:
    """The Step 12 display label -- the only name that reaches a table."""
    return cmp.display_label(key)


def short_label(key: str) -> str:
    """"Approach 1", "B4": the display label's own short form."""
    return fa.short_label(key)


def model_colour(key: str) -> str:
    """One fixed colour per model, in display order, for every chart."""
    colours = getattr(config, "MODEL_COLOURS", {}) or {}
    if key in colours:
        return colours[key]
    return fa.model_colour(key)


def model_keys(keys: Sequence[str] | None = None,
               verbose: bool = False) -> list[str]:
    """`XAI_ALL_MODELS`, verified against `STEP12_DISPLAY_ORDER`."""
    keys = list(getattr(config, "XAI_ALL_MODELS", []) if keys is None else keys)
    known = [k for k, _label in config.STEP12_DISPLAY_ORDER]
    unknown = [k for k in keys if k not in known]
    if unknown:
        raise XaiAllError(
            "XAI_ALL_MODELS names %s, which are not keys of "
            "STEP12_DISPLAY_ORDER (%s)." % (unknown, known))
    if verbose:
        show_note("Explaining %d configurations: %s."
                  % (len(keys), ", ".join(display_label(k) for k in keys)))
    return keys


def ordered(keys: Sequence[str]) -> list[str]:
    return fa.ordered(keys)


def which_step_produces(key: str) -> str:
    return cmp.which_step_produces(key)


# --- the composed models ---------------------------------------------------
def pca_as_linear_layer(encoder, pca, name: str = "encoder_then_pca"):
    """encoder -> PCA, as one frozen differentiable Keras model.

    A PCA transform is an affine map, so it goes inside the differentiated
    function as a Dense layer rather than being applied beforehand:

        z_pca = (z - mean) W^T      ->   kernel = W^T,  bias = -mean W^T

    with W = `pca.components_`.  Attributions then land on Baseline 4's
    ORIGINAL named features instead of on PCA components, which is the whole
    point of composing.  Nothing is fitted here: both pieces are loaded.
    """
    keras = xai._keras()
    components = np.asarray(pca.components_, dtype="float32")      # (k, latent)
    mean = np.asarray(pca.mean_, dtype="float32")                  # (latent,)
    kernel = components.T                                          # (latent, k)
    bias = -mean @ kernel                                          # (k,)

    latent = int(kernel.shape[0])
    inputs = keras.Input(shape=(int(encoder.input_shape[-1]),),
                         name="encoder_input")
    hidden = encoder(inputs)
    if int(hidden.shape[-1]) != latent:
        raise XaiAllError(
            "the Baseline 4 encoder returns %d numbers but its PCA was fitted "
            "on %d" % (int(hidden.shape[-1]), latent))
    projection = keras.layers.Dense(int(kernel.shape[1]), use_bias=True,
                                    activation=None, name="pca")
    outputs = projection(hidden)
    block = keras.Model(inputs, outputs, name=name)
    projection.set_weights([kernel, bias])
    block.trainable = False
    return block


def _composed_from(classifier, encoder, X_train: pd.DataFrame,
                   X_test: pd.DataFrame, key: str, saved_probabilities,
                   model_path: str) -> dict[str, Any]:
    """The bundle shape `explainability.load_composed` returns."""
    features = list(X_test.columns)
    composed = xai.build_composed_model(classifier, encoder=encoder,
                                        n_features=len(features),
                                        name="composed_%s" % key)
    return {"name": key, "composed": composed, "classifier": classifier,
            "encoder": encoder, "X_train": X_train, "X_test": X_test,
            "features": features, "saved_probabilities": saved_probabilities,
            "model_path": str(model_path)}


def _saved_probabilities(key: str) -> np.ndarray | None:
    """The probabilities this configuration's own step wrote, or None."""
    path = fa.predictions_path(key)
    if path is None or not path.exists():
        return None
    saved = pd.read_csv(path, low_memory=False)
    columns = [c for c in saved.columns if c.startswith("p_")]
    return saved[columns].to_numpy(dtype="float64") if columns else None


def _load_approach2(key: str) -> dict[str, Any]:
    """Approach 2: the proposed inputs and encoder, the Step 10.2 classifier."""
    from . import autoencoder as ae
    from . import experiments as xp
    from . import models_attention as ma

    loaded = ma.load_step10b(verbose=False)            # raises if 10.2 never ran
    source = xp.load_feature_source("selected30", verbose=False)
    _encode, encoder, order = ae.load_encoder(verbose=False)
    return _composed_from(
        loaded["model"], encoder, source["X_train"][order],
        source["X_test"][order], key, _saved_probabilities(key),
        Path(loaded["folder"]) / config.STEP10B_FILES["model"])


def _load_baseline4(key: str) -> dict[str, Any]:
    """Baseline 4: encoder -> PCA (as a fixed linear map) -> LSTM."""
    from . import experiments as xp

    folder12 = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder12 is None:
        raise XaiAllError("the Step 12 output folder is not usable here -- %s"
                          % config.why_unusable(config.STEP12_OUTPUT_DIR))
    folder = Path(folder12) / key
    model_path = folder / config.BASELINE4_FILES["model"]
    pca_path = folder / config.BASELINE4_FILES["pca"]
    encoder_path = folder / config.BASELINE4_FILES["encoder"]
    settings_path = folder / config.BASELINE4_FILES["ae_config"]
    for path, what in ((model_path, "the Baseline 4 classifier"),
                       (pca_path, "the fitted Baseline 4 PCA"),
                       (encoder_path, "the Baseline 4 encoder"),
                       (settings_path, "the Baseline 4 encoder settings")):
        if not path.exists():
            raise XaiAllError("%s is missing:\n    %s" % (what, path))

    from joblib import load as joblib_load
    keras = xai._keras()
    classifier = keras.models.load_model(model_path)
    encoder = keras.models.load_model(encoder_path)
    pca = joblib_load(pca_path)
    order = list(json.loads(settings_path.read_text(encoding="utf-8"))
                 ["features"])
    block = pca_as_linear_layer(encoder, pca, name="baseline4_encoder_pca")

    source = xp.load_feature_source(config.BASELINE4["feature_source"],
                                    verbose=False)
    bundle = _composed_from(classifier, block, source["X_train"][order],
                            source["X_test"][order], key,
                            _saved_probabilities(key), model_path)
    bundle["pca"] = pca
    return bundle


def _load_baseline5(key: str) -> dict[str, Any]:
    """Baseline 5: its MI top-30 list, its own encoder, its CNN-LSTM."""
    from . import experiments as xp

    folder12 = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder12 is None:
        raise XaiAllError("the Step 12 output folder is not usable here -- %s"
                          % config.why_unusable(config.STEP12_OUTPUT_DIR))
    folder = Path(folder12) / config.BASELINE5_NAME
    model_path = folder / config.BASELINE5_FILES["model"]
    encoder_path = folder / config.BASELINE5_FILES["encoder"]
    settings_path = folder / config.BASELINE5_FILES["ae_config"]
    for path, what in ((model_path, "the Baseline 5 classifier"),
                       (encoder_path, "the Baseline 5 encoder"),
                       (settings_path, "the Baseline 5 encoder settings")):
        if not path.exists():
            raise XaiAllError("%s is missing:\n    %s" % (what, path))

    keras = xai._keras()
    classifier = keras.models.load_model(model_path)
    encoder = keras.models.load_model(encoder_path)
    order = list(json.loads(settings_path.read_text(encoding="utf-8"))
                 ["features"])
    source = xp.load_feature_source("all", verbose=False)
    return _composed_from(classifier, encoder, source["X_train"][order],
                          source["X_test"][order], key,
                          _saved_probabilities(key), model_path)


def _load_baseline6(key: str) -> dict[str, Any]:
    """Baseline 6: features -> Baseline 4's encoder -> PCA -> CNN-LSTM.

    The same composition Baseline 4 gets, with Baseline 6's classifier: the
    encoder and the PCA are Baseline 4's saved ones (or, when
    B6_PCA_COMPONENTS is an integer, the PCA Baseline 6 fitted on the training
    latents and saved in its own folder), and the PCA goes INSIDE the
    differentiated function as a frozen affine layer so the attributions land
    on the original named features.
    """
    from . import experiments as xp
    from . import experiments_baseline6 as b6

    place = b6.folder()
    if place is None:
        raise XaiAllError("the Baseline 6 output folder is not usable here -- %s"
                          % config.why_unusable(config.STEP12_B6_OUTPUT_DIR))
    names = config.BASELINE6_FILES
    model_path = place / names["model"]
    if not model_path.exists():
        raise XaiAllError("the Baseline 6 classifier is missing:\n    %s"
                          % model_path)

    pipeline = b6.load_baseline4_pipeline(verbose=False)
    pca = pipeline["pca"]
    own_pca = place / names["pca"]
    if own_pca.exists():                        # B6 fitted its own, so use it
        from joblib import load as joblib_load
        pca = joblib_load(own_pca)

    classifier = xai._keras().models.load_model(model_path)
    block = pca_as_linear_layer(pipeline["encoder"], pca,
                                name="baseline6_encoder_pca")
    order = pipeline["order"]
    source = xp.load_feature_source(config.BASELINE6_SPEC["feature_source"],
                                    verbose=False)
    bundle = _composed_from(classifier, block, source["X_train"][order],
                            source["X_test"][order], key,
                            _saved_probabilities(key), model_path)
    bundle["pca"] = pca
    return bundle


def load_composed_any(key: str, verbose: bool = False) -> dict[str, Any]:
    """The composed model for one configuration, on its ORIGINAL features."""
    approach2 = getattr(config, "APPROACH2_KEY", "approach2_attention")
    baseline6 = getattr(config, "BASELINE6_NAME", "baseline6_ae_pca_cnn_lstm")
    if key == approach2:
        bundle = _load_approach2(key)
    elif key == baseline6:
        bundle = _load_baseline6(key)
    elif key == "baseline4_ae_pca_lstm":
        bundle = _load_baseline4(key)
    elif key == config.BASELINE5_NAME:
        bundle = _load_baseline5(key)
    else:
        bundle = xai.load_composed(key, verbose=False)
    if verbose:
        print("  loaded %-26s %d features -> %s"
              % (short_label(key), len(bundle["features"]),
                 "encoder -> classifier" if bundle.get("encoder") is not None
                 else "classifier"))
    return bundle


def available_models(keys: Sequence[str] | None = None, rows=None,
                     verbose: bool = True
                     ) -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame]:
    """Load and CHECK every composed model; skip what is not on disk.

    Returns (bundles, availability, composed_checks).  A model is kept only if
    its composed form reproduces the probabilities its own step saved to 1e-5:
    attributions from a model that does not is an explanation of something
    else.
    """
    keys = model_keys(keys)
    bundles: dict[str, dict] = {}
    rows_out, checks = [], []
    for key in keys:
        label = display_label(key)
        try:
            bundle = load_composed_any(key, verbose=verbose)
        except Exception as problem:               # a missing file, not a bug
            first = str(problem).strip().split("\n")[0]
            message = ("%s not available -- %s. Run %s."
                       % (label, first, which_step_produces(key)))
            if verbose:
                print("  [SKIP]  %s" % message)
            rows_out.append({"model": label, "key": key, "available": False,
                             "input features": 0, "composed": NA_TEXT,
                             "note": message})
            continue

        reference = bundle.get("saved_probabilities")
        if reference is None:
            checks.append({"model": label, "max abs difference": float("nan"),
                           "tolerance": 1e-5, "result": "[SKIP]",
                           "detail": "this configuration saved no "
                                     "predictions.csv to compare against"})
            bundles[key] = bundle
            rows_out.append({"model": label, "key": key, "available": True,
                             "input features": len(bundle["features"]),
                             "composed": "[SKIP]", "note": "no saved "
                             "predictions.csv to check against"})
            continue
        try:
            row = xai.check_composed_model(bundle["composed"],
                                           bundle["X_test"], reference,
                                           label, tolerance=1e-5,
                                           verbose=False)
            checks.append(row | {"detail": "composed model reproduces the "
                                           "saved probabilities"})
            bundles[key] = bundle
            rows_out.append({"model": label, "key": key, "available": True,
                             "input features": len(bundle["features"]),
                             "composed": "[OK]", "note": ""})
        except Exception as problem:
            first = str(problem).strip().split("\n")[0]
            checks.append({"model": label, "max abs difference": float("nan"),
                           "tolerance": 1e-5, "result": "[FAIL]",
                           "detail": first})
            rows_out.append({"model": label, "key": key, "available": False,
                             "input features": len(bundle["features"]),
                             "composed": "[FAIL]",
                             "note": "%s is not explained: %s" % (label, first)})
            if verbose:
                print("  [SKIP]  %s -- the composed model does not reproduce "
                      "its saved probabilities" % label)

    availability = pd.DataFrame(rows_out)
    composed_checks = pd.DataFrame(checks)
    if not bundles:
        raise XaiAllError(
            "none of the %d configurations could be composed and checked, so "
            "there is nothing to explain. The availability table says which "
            "cells produce each one." % len(keys))
    return bundles, availability, composed_checks


# ===========================================================================
# 14.9.0b  PRESENTATION -- one way of showing a table, used by every cell
# ===========================================================================
def _fmt_number(value: Any, decimals: int = 4) -> str:
    """A number as the notebook should read it, or `--` when it is NA."""
    if value is None:
        return NA_TEXT
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return NA_TEXT
    if float(number).is_integer() and abs(number) >= 1000:
        return "{:,.0f}".format(number)
    return ("%%.%df" % decimals) % number


def _format_frame(frame: pd.DataFrame, decimals: int,
                  percent_columns: Sequence[str],
                  count_columns: Sequence[str]) -> pd.DataFrame:
    """Every cell as text, so NA reads `--` and counts carry separators."""
    out = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        values = frame[column]
        if column in percent_columns:
            out[column] = [NA_TEXT if pd.isna(v) else "%.1f" % float(v)
                           for v in values]
        elif column in count_columns:
            out[column] = [NA_TEXT if pd.isna(v) else "{:,.0f}".format(float(v))
                           for v in values]
        elif pd.api.types.is_bool_dtype(values):
            out[column] = ["yes" if bool(v) else "no" for v in values]
        elif pd.api.types.is_numeric_dtype(values):
            out[column] = [_fmt_number(v, decimals) for v in values]
        else:
            out[column] = [NA_TEXT if (v is None or (isinstance(v, float)
                                                     and np.isnan(v)))
                           else str(v) for v in values]
    return out


def present(frame: pd.DataFrame, caption: str, reading: str | None = None,
            best: Sequence[str] = (), lower_is_better: Sequence[str] = (),
            decimals: int = 4, percent_columns: Sequence[str] = (),
            count_columns: Sequence[str] = (), top: int | None = None,
            full_file: str | None = None, na_footnote: bool = False) -> None:
    """One table, rendered the same way everywhere.

    A pandas Styler in Jupyter (index hidden, caption under the table, the best
    value in each named column in bold) and plain text outside it, so the same
    cell runs in a terminal.  `top` shows the first rows and names the saved
    CSV that holds the rest -- a 400-row table in a notebook is not a table a
    reader can use.
    """
    if frame is None or not len(frame):
        show_note("_%s: nothing to show._" % caption)
        return
    shown = frame if top is None else frame.head(top)
    numbers = {column: pd.to_numeric(shown[column], errors="coerce")
               for column in shown.columns}
    bold: dict[str, set[int]] = {}
    for column in best:
        if column not in shown.columns or numbers[column].notna().sum() == 0:
            continue
        target = (numbers[column].min() if column in lower_is_better
                  else numbers[column].max())
        bold[column] = {position for position, value
                        in enumerate(numbers[column])
                        if pd.notna(value) and abs(value - target) <= 1e-12}

    text = _format_frame(shown, decimals, percent_columns, count_columns)
    note = caption
    if top is not None and len(frame) > top:
        note += "  Showing the top %d of %s rows%s." % (
            top, "{:,}".format(len(frame)),
            " (full table: `%s`)" % full_file if full_file else "")
    elif full_file:
        note += "  (saved as `%s`)" % full_file
    if na_footnote:
        note += "  %s." % NA_FOOTNOTE
    if lower_is_better:
        note += "  Lower is better for: %s." % ", ".join(lower_is_better)

    if not dl.in_notebook():
        marked = text.copy()
        for column, positions in bold.items():
            marked[column] = [("*%s*" % v) if i in positions else v
                              for i, v in enumerate(marked[column])]
        with pd.option_context("display.max_columns", None, "display.width", None):
            print(marked.to_string(index=False))
        show_note(note)
        return

    from IPython.display import display

    def emphasise(column: pd.Series) -> list[str]:
        positions = bold.get(column.name, set())
        return ["font-weight: 700" if i in positions else ""
                for i in range(len(column))]

    styler = (text.style
              .set_caption(note)
              .apply(emphasise, axis=0)
              .set_table_styles([
                  {"selector": "caption",
                   "props": [("caption-side", "bottom"), ("text-align", "left"),
                             ("padding-top", "8px"), ("font-size", "90%"),
                             ("color", dl._INK_SOFT)]},
                  {"selector": "th", "props": [("text-align", "left")]},
                  {"selector": "td", "props": [("text-align", "right")]}]))
    try:
        styler = styler.hide(axis="index")
    except AttributeError:                       # pragma: no cover - old pandas
        styler = styler.hide_index()
    display(styler)
    if reading:
        show_reading(reading)


def show_reading(sentence: str) -> None:
    """The one-sentence finding that goes under an output."""
    if dl.in_notebook():
        from IPython.display import Markdown, display
        display(Markdown("**Reading:** %s" % sentence))
    else:
        print("Reading: %s" % sentence)


def show_chart(path, verbose: bool = True) -> None:
    """Show a saved chart once, or say which file is missing."""
    path = Path(path)
    if path.exists() and dl.in_notebook():
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    elif not path.exists() and verbose:
        print("missing:", path)


def status_line(label: str, flows: int, seconds: float, extra: str = "") -> None:
    """The only progress output: one line per model."""
    minutes, remainder = divmod(int(round(seconds)), 60)
    print("  %-28s explained %s flows in %dm%02ds%s"
          % (label, "{:,}".format(int(flows)), minutes, remainder,
             (" | %s" % extra) if extra else ""))


# ===========================================================================
# 14.9.2  INTEGRATED GRADIENTS, PER MODEL -- Step 14's functions, called
# ===========================================================================
def explain_model(key: str, bundle: dict[str, Any], rows: np.ndarray,
                  orig_row: Sequence[Any], probabilities: np.ndarray,
                  inputs: dict[str, Any], verbose: bool = True
                  ) -> dict[str, Any]:
    """Everything 14.9.2 asks of one model, through `explainability`.

    Both targets (the predicted-class probability and the attack risk),
    completeness, the perturbation agreement, the deletion check, stability
    under noise and the IG latency -- every one of them is the Step 14
    function, called with this model's own composed form, features and
    baseline.
    """
    benign = inputs["benign_codes"]
    y_train = inputs["y_train"]
    label_names = inputs["label_names"]
    label = display_label(key)

    X_selected = bundle["X_test"].iloc[rows].reset_index(drop=True)
    features = list(bundle["features"])
    baseline = xai.baseline_vector(bundle["X_train"], y_train, benign)
    predicted = np.argmax(np.asarray(probabilities)[rows], axis=1)

    started = time.perf_counter()
    by_class = xai.integrated_gradients(bundle["composed"], X_selected,
                                        baseline, target=TARGET_CLASS,
                                        class_index=predicted)
    by_risk = xai.integrated_gradients(bundle["composed"], X_selected, baseline,
                                       target=TARGET_RISK, benign_codes=benign)
    ig_seconds = time.perf_counter() - started

    completeness = pd.concat(
        [xai.completeness_table(by_class, label, TARGET_CLASS),
         xai.completeness_table(by_risk, label, TARGET_RISK)],
        ignore_index=True)
    importance = pd.concat(
        [xai.global_importance(by_class["attributions"], features, label,
                               TARGET_CLASS, predicted=predicted,
                               label_names=label_names),
         xai.global_importance(by_risk["attributions"], features, label,
                               TARGET_RISK, predicted=predicted,
                               label_names=label_names)], ignore_index=True)

    started = time.perf_counter()
    perturbed = xai.perturbation_drops(bundle["composed"], X_selected, baseline,
                                       target=TARGET_RISK, benign_codes=benign)
    perturb_seconds = time.perf_counter() - started
    agreement = xai.attribution_agreement(by_risk["attributions"],
                                          perturbed["drops"])
    agreement.insert(0, "model", label)
    agreement["Orig_Row"] = list(orig_row)

    deletion = xai.deletion_check(bundle["composed"], X_selected, baseline,
                                  by_risk["attributions"], target=TARGET_RISK,
                                  benign_codes=benign)
    capped = int(min(xai.STABILITY_MAX_FLOWS, len(rows)))
    stable = xai.stability(bundle["composed"], X_selected.iloc[:capped],
                           baseline, by_risk["attributions"][:capped],
                           target=TARGET_RISK, benign_codes=benign)
    stable = dict(stable) | {"flows": capped}

    flows = max(len(rows), 1)
    baseline_risk = float(xai._scores(
        bundle["composed"],
        np.broadcast_to(baseline, (1,) + baseline.shape).astype("float32"),
        TARGET_RISK, None, benign)[0])
    latency = {
        "ig_ms_per_flow": round(ig_seconds / flows * 1000 / 2, 4),
        "ig_seconds_total": round(ig_seconds, 3),
        "perturbation_ms_per_flow": round(perturb_seconds / flows * 1000, 4),
        "baseline_risk": round(baseline_risk, 6),
    }
    share = float(completeness[completeness["target"] == TARGET_RISK]
                  ["within_tolerance"].mean())
    if verbose:
        status_line(label, len(rows), ig_seconds,
                    "completeness %.1f%%, F(x') risk %.3f"
                    % (100 * share, baseline_risk))
    return {
        "key": key, "label": label, "features": features,
        "baseline": baseline, "baseline_risk": baseline_risk,
        "attributions": {TARGET_CLASS: by_class["attributions"],
                         TARGET_RISK: by_risk["attributions"]},
        "predicted": predicted, "completeness": completeness,
        "importance": importance, "drops": perturbed["drops"],
        "agreement": agreement, "deletion": deletion, "stability": stable,
        "latency": latency, "completeness_share": share,
    }


def attention_supplement(bundle: dict[str, Any], rows: np.ndarray,
                         predicted: np.ndarray, inputs: dict[str, Any],
                         verbose: bool = True) -> pd.DataFrame:
    """Approach 2's attention weights per position, as a SUPPLEMENTARY view.

    Attention is over the CNN feature-map POSITIONS, not over input features,
    so it cannot say which flow feature mattered and is never used as the
    explanation here.  It is reported because it is the one thing Approach 2
    has that the others do not.
    """
    from . import models_attention as ma

    try:
        X = bundle["X_test"].iloc[rows]
        encoder = bundle.get("encoder")
        latent = (encoder.predict(X.to_numpy(dtype="float32"), verbose=0)
                  if encoder is not None else X.to_numpy(dtype="float32"))
        sequence = np.asarray(latent, dtype="float32")[..., None]
        weights = ma.attention_weights(bundle["classifier"], sequence)
    except Exception as problem:                  # pragma: no cover - optional
        if verbose:
            print("  [SKIP]  the Approach 2 attention view is unavailable: %s"
                  % str(problem).split("\n")[0])
        return pd.DataFrame()

    label_names = inputs["label_names"]
    records = []
    for code in sorted({int(c) for c in np.asarray(predicted).tolist()}):
        selected = np.asarray(predicted) == code
        if not selected.any():
            continue
        mean = weights[selected].mean(axis=0)
        for position, value in enumerate(mean):
            records.append({
                "predicted class": label_names.get(code, str(code)),
                "position": position + 1,
                "mean attention": round(float(value), 6),
                "flows": int(selected.sum())})
    return pd.DataFrame(records)


# ===========================================================================
# 14.9.3  CROSS-MODEL COMPARISON OF THE EXPLANATIONS
# ===========================================================================
def importance_matrix(explained: dict[str, dict[str, Any]],
                      target: str = TARGET_RISK,
                      scope: str = "all explained flows") -> pd.DataFrame:
    """feature x model of mean |IG|, with NaN where a model never sees it.

    NaN rather than 0 on purpose: 0 would say "this model looked at the feature
    and found it irrelevant", which is a different statement from "this feature
    is not one of its inputs".  Every table built from this prints those cells
    as `--`.
    """
    columns: dict[str, pd.Series] = {}
    for key in ordered(list(explained)):
        frame = explained[key]["importance"]
        part = frame[(frame["target"] == target) & (frame["scope"] == scope)]
        columns[display_label(key)] = part.set_index("feature")["mean_abs_ig"]
    matrix = pd.DataFrame(columns)
    matrix.index.name = "feature"
    return matrix


def top_features(matrix: pd.DataFrame, key_label: str,
                 k: int | None = None) -> list[str]:
    """One model's top-k features by mean |IG|, strongest first."""
    k = getattr(config, "XAI_TOP_FEATURES", 10) if k is None else int(k)
    column = matrix[key_label].dropna().sort_values(ascending=False)
    return list(column.head(k).index)


def top_feature_table(matrix: pd.DataFrame, k: int | None = None
                      ) -> pd.DataFrame:
    """rank x model: each model's top-k features, side by side."""
    k = getattr(config, "XAI_TOP_FEATURES", 10) if k is None else int(k)
    data = {}
    for label in matrix.columns:
        names = top_features(matrix, label, k)
        data[label] = names + [NA_TEXT] * (k - len(names))
    return pd.DataFrame(data, index=["#%d" % (i + 1) for i in range(k)]
                        ).rename_axis("rank").reset_index()


def explanation_agreement(matrix: pd.DataFrame, k: int | None = None
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Spearman, top-k overlap) between models, on their SHARED features.

    Two models with different feature sets can only be compared where both
    look: the correlation is computed on the intersection, and the number of
    shared features travels with it so a high correlation over four features
    is not read as agreement over thirty.
    """
    k = getattr(config, "XAI_TOP_FEATURES", 10) if k is None else int(k)
    labels = list(matrix.columns)
    spearman_grid = pd.DataFrame(np.nan, index=labels, columns=labels)
    overlap_grid = pd.DataFrame(np.nan, index=labels, columns=labels)
    for left in labels:
        for right in labels:
            a, b = matrix[left], matrix[right]
            shared = a.notna() & b.notna()
            if shared.sum() < 2:
                continue
            spearman_grid.loc[left, right] = round(
                xai.spearman(a[shared].to_numpy(), b[shared].to_numpy()), 6)
            overlap_grid.loc[left, right] = round(
                xai.top_k_overlap(a[shared].to_numpy(), b[shared].to_numpy(),
                                  min(k, int(shared.sum()))), 6)
    spearman_grid.index.name = "model"
    overlap_grid.index.name = "model"
    return spearman_grid, overlap_grid


def shared_feature_counts(matrix: pd.DataFrame) -> pd.DataFrame:
    """How many features each pair of models actually has in common."""
    labels = list(matrix.columns)
    grid = pd.DataFrame(0, index=labels, columns=labels, dtype="int64")
    for left in labels:
        for right in labels:
            grid.loc[left, right] = int(
                (matrix[left].notna() & matrix[right].notna()).sum())
    grid.index.name = "model"
    return grid


def consensus_features(matrix: pd.DataFrame, k: int | None = None
                       ) -> pd.DataFrame:
    """Which features are in many models' top-k, and which in only one.

    `models that can see it` is the denominator that matters: a feature only
    two models take as input cannot be in seven models' top-k.
    """
    k = getattr(config, "XAI_TOP_FEATURES", 10) if k is None else int(k)
    tops = {label: set(top_features(matrix, label, k))
            for label in matrix.columns}
    records = []
    for feature in matrix.index:
        can_see = [label for label in matrix.columns
                   if not pd.isna(matrix.loc[feature, label])]
        in_top = [label for label in can_see if feature in tops[label]]
        if not in_top:
            continue
        records.append({
            "feature": feature,
            "in top-%d of" % k: len(in_top),
            "models that can see it": len(can_see),
            "share of those models": round(len(in_top) / max(len(can_see), 1), 4),
            "models": ", ".join(short_label(key) for key in
                                [k2 for k2, _l in config.STEP12_DISPLAY_ORDER
                                 if display_label(k2) in in_top]),
            "only this model": in_top[0] if len(in_top) == 1 else "",
        })
    frame = pd.DataFrame(records)
    if len(frame):
        frame = frame.sort_values(
            ["in top-%d of" % k, "share of those models", "feature"],
            ascending=[False, False, True]).reset_index(drop=True)
    return frame


def quality_table(explained: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per model: is its explanation faithful, stable and affordable?"""
    records = []
    for key in ordered(list(explained)):
        result = explained[key]
        completeness = result["completeness"]
        risk_part = completeness[completeness["target"] == TARGET_RISK]
        agreement = result["agreement"]
        deletion = result["deletion"]
        stable = result["stability"]
        records.append({
            "model": display_label(key),
            "completeness within tolerance (%)":
                round(100 * float(risk_part["within_tolerance"].mean()), 1),
            "IG vs perturbation (Spearman)":
                round(float(agreement["spearman"].mean()), 4),
            "top-%d overlap with perturbation" % config.TOP_K_FEATURES:
                round(float(agreement["top_k_overlap"].mean()), 4),
            "deletion gap (top-%d - random)" % deletion["k"]:
                round(float(deletion["margin"]), 4),
            "stability (Spearman under noise)":
                round(float(stable.get("mean_spearman", float("nan"))), 4),
            "IG ms per flow": round(float(result["latency"]["ig_ms_per_flow"]), 2),
            "F(x') attack risk": round(float(result["baseline_risk"]), 4),
            "explained flows": int(len(result["predicted"])),
        })
    return pd.DataFrame(records)


# ===========================================================================
# 14.9.4  THE FUSION VARIANTS, EXPLAINED EXACTLY
# ===========================================================================
def _variant_keys(tag: str, step13all: dict[str, Any] | None) -> list[str]:
    """Which models a variant fuses, from the saved metrics rather than guessed."""
    if tag == "F3":
        return list(config.FUSION_MODELS)
    metrics = (step13all or {}).get("metrics") or {}
    block = (metrics.get("variants") or {}).get(tag) or {}
    return list(block.get("keys") or [])


def variant_frame(tag: str, step13: dict[str, Any],
                  step13all: dict[str, Any] | None) -> pd.DataFrame | None:
    """The saved per-flow file a variant's numbers are read from."""
    if tag == "F3":
        return step13.get("fused_predictions")
    return (step13all or {}).get("fused_predictions")


def recover_reliability(keys: Sequence[str],
                        val_probs: dict[str, np.ndarray],
                        test_probs: dict[str, np.ndarray],
                        fusion_inputs: dict[str, Any]
                        ) -> tuple[np.ndarray, np.ndarray, float]:
    """Rel_m(k(x)) and Conf_m(x) for a 13.8 variant, and how far they reproduce
    the weights that file saved.

    13.8 saves w_m per flow but not the two quantities behind it, and a report
    that prints `nan` for a model's reliability is a report an analyst cannot
    use.  Both are recomputed with 13.8's own `fit_on_validation` -- the same
    validation rows, the same shrinkage -- and then CHECKED against the weights
    the file holds.

    The tolerance is WEIGHT_RECOVERY_TOL, not 1e-6, and the reason is worth
    stating: 13.8 computed its weights from the classifiers' own probabilities,
    while everything here goes through the COMPOSED models, which reproduce
    those probabilities to 1e-5.  A weight is a ratio of products of such
    quantities, so a few times 1e-3 is the expected disagreement, not a fault.
    Beyond that the caller falls back to NaN rather than printing a reliability
    that is not the one the fusion used; the w_m a report prints always comes
    from the saved file either way.
    """
    keys = list(keys)
    fitted = fa.fit_on_validation(keys, val_probs, fusion_inputs)
    test_list = [test_probs[k] for k in keys]
    if config.CONTEXT_MODE == "consensus_class":
        contexts = fz.consensus_context(test_list)
    else:
        contexts = fz.protocol_contexts(fusion_inputs["y_train"],
                                        fusion_inputs["meta_test"])[1]
    reliability = fz.reliability_matrix(fitted["info"]["lookup"],
                                        fitted["info"]["macro"], keys, contexts)
    confidence = np.column_stack([
        fz.normalised_entropy_confidence(p) for p in test_list])
    implied = fz.dynamic_weights(reliability, confidence)
    return reliability, confidence, implied, contexts


def variant_matrices(tag: str, keys: Sequence[str], frame: pd.DataFrame,
                     class_order: Sequence[str],
                     test_probs: dict[str, np.ndarray] | None = None,
                     val_probs: dict[str, np.ndarray] | None = None,
                     fusion_inputs: dict[str, Any] | None = None
                     ) -> dict[str, np.ndarray]:
    """w_m, r_m, R and the fused class for one variant, READ from its file.

    F3's file is Step 13's `fused_predictions.csv`, whose columns are named by
    the internal model key; FALL and FSEL come from 13.8's
    `fused_predictions_all.csv`, whose columns carry the variant tag and the
    short label.  Both are read rather than recomputed, so `R = sum C_m` below
    is a check on the file that produced the alert.

    Conf_m is recomputed from the model's own probabilities -- the same
    `1 - normalised entropy` the fusion used -- and Rel_m is left NaN for the
    13.8 variants, which do not save it per flow.
    """
    keys = list(keys)
    if tag == "F3":
        matrices = xai.step13_matrices(frame, keys, class_order)
        return matrices

    def column(prefix: str, names: Sequence[str]) -> np.ndarray:
        return np.column_stack([frame["%s%s" % (prefix, name)].to_numpy(
            dtype="float64") for name in names])

    shorts = [short_label(k) for k in keys]
    weights = column("w_%s_" % tag, shorts)
    model_risk = column("r_", shorts)
    confidence = (np.column_stack([
        fz.normalised_entropy_confidence(test_probs[k]) for k in keys])
        if test_probs is not None
        else np.full(weights.shape, np.nan))

    reliability = np.full(weights.shape, np.nan)
    context_column = None
    weights_gap = float("nan")
    if val_probs is not None and fusion_inputs is not None and test_probs:
        try:
            recovered, confidence_again, implied, contexts = \
                recover_reliability(keys, val_probs, test_probs, fusion_inputs)
            weights_gap = float(np.abs(implied - weights).max())
            if weights_gap <= WEIGHT_RECOVERY_TOL:   # it is the same fusion
                reliability, confidence = recovered, confidence_again
                # the context a flow fell in is what picked its reliabilities,
                # so the report can name it instead of printing "n/a"
                names = fusion_inputs.get("label_names") or {}
                context_column = np.array(
                    [names.get(int(key), str(key))
                     if str(key).lstrip("-").isdigit() else str(key)
                     for key in np.asarray(contexts)], dtype=object)
        except Exception:                        # pragma: no cover - optional
            weights_gap = float("nan")
    fused_probabilities = np.column_stack(
        [frame["P_%s_%s" % (tag, name)].to_numpy(dtype="float64")
         for name in class_order])
    return {
        "weights": weights,
        "reliability": reliability,
        "confidence": confidence,
        "weights_recovered_gap": weights_gap,
        "model_risk": model_risk,
        "risk": frame["R_%s" % tag].to_numpy(dtype="float64"),
        "fused_class": np.argmax(fused_probabilities, axis=1).astype("int64"),
        "fused_probabilities": fused_probabilities,
        "context": (context_column if context_column is not None
                    else frame["context"].to_numpy(dtype=object)
                    if "context" in frame.columns
                    else np.full(len(frame), "n/a", dtype=object)),
    }


def variant_contributions(tag: str, keys: Sequence[str],
                          matrices: dict[str, np.ndarray], rows: np.ndarray,
                          probabilities: dict[str, np.ndarray],
                          inputs: dict[str, Any]
                          ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """C_m = w_m r_m per flow and model, and R = sum C_m checked against R.

    Straight through `explainability.fusion_breakdown` and
    `fusion_consistency`: the identity is Step 14's, and the only thing this
    adds is doing it per variant.  Models are addressed by their SHORT display
    label, so nothing in a report or a table carries an internal key.
    """
    labels = [short_label(k) for k in keys]
    by_label = {short_label(k): probabilities[k] for k in keys}
    orig_row = inputs["orig_row"]
    breakdown = xai.fusion_breakdown(
        rows, labels, by_label, matrices["weights"], matrices["reliability"],
        matrices["confidence"], inputs["benign_codes"], inputs["label_names"],
        orig_row=orig_row)
    breakdown.insert(0, "variant", tag)
    consistency = xai.fusion_consistency(breakdown, matrices["risk"], rows)
    return breakdown, consistency


def dominant_model_share(breakdown: pd.DataFrame, inputs: dict[str, Any],
                         fused_class: Sequence[int], rows: np.ndarray
                         ) -> pd.DataFrame:
    """Share of explained alerts where each model has the largest C_m.

    Reported overall and per fused class: a fusion that leans on one model
    everywhere is a different object from one that changes its mind by class.
    """
    label_names = inputs["label_names"]
    fused_class = np.asarray(fused_class)
    rows = np.asarray(rows, dtype="int64")
    winners = breakdown.loc[breakdown.groupby("flow")["C_m"].idxmax(),
                            ["flow", "model"]]
    winners["class"] = [label_names.get(int(fused_class[rows[int(f)]]),
                                        str(fused_class[rows[int(f)]]))
                        for f in winners["flow"]]
    records = []
    for scope, part in [("all explained alerts", winners)] + [
            (name, group) for name, group in winners.groupby("class")]:
        total = max(len(part), 1)
        counts = part["model"].value_counts()
        for model in breakdown["model"].unique():
            records.append({
                "scope": scope, "model": model,
                "alerts where it dominates": int(counts.get(model, 0)),
                "share (%)": round(100 * int(counts.get(model, 0)) / total, 1),
                "alerts": int(len(part))})
    return pd.DataFrame(records)


def variant_disagreements(variants: dict[str, dict[str, Any]],
                          rows: np.ndarray, inputs: dict[str, Any],
                          limit: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Where the variants decide differently, and which model moved it.

    Returns (summary, examples).  The example rows name the model whose
    contribution differs most between the two variants on that flow, which is
    the nearest thing to "why" the arithmetic can support.
    """
    tags = [tag for tag in VARIANTS if tag in variants]
    rows = np.asarray(rows, dtype="int64")
    decisions = {}
    for tag in tags:
        matrices = variants[tag]["matrices"]
        tau = float(variants[tag]["tau"])
        decisions[tag] = (np.asarray(matrices["risk"])[rows] >= tau).astype(int)

    summary, examples = [], []
    for i, left in enumerate(tags):
        for right in tags[i + 1:]:
            differ = decisions[left] != decisions[right]
            summary.append({
                "pair": "%s vs %s" % (left, right),
                "explained alerts": int(len(rows)),
                "decisions that differ": int(differ.sum()),
                "share (%)": round(100 * float(differ.mean()), 1)})
            for position in np.flatnonzero(differ)[:limit]:
                a = variants[left]["breakdown"]
                b = variants[right]["breakdown"]
                left_part = a[a["flow"] == position].set_index("model")["C_m"]
                right_part = b[b["flow"] == position].set_index("model")["C_m"]
                shared = left_part.index.union(right_part.index)
                gap = (right_part.reindex(shared).fillna(0.0)
                       - left_part.reindex(shared).fillna(0.0))
                mover = gap.abs().idxmax() if len(gap) else "n/a"
                examples.append({
                    "pair": "%s vs %s" % (left, right),
                    "Orig_Row": inputs["orig_row"][position],
                    "%s decision" % left: "ATTACK" if decisions[left][position]
                                          else "below tau",
                    "%s decision" % right: "ATTACK" if decisions[right][position]
                                           else "below tau",
                    "R (%s)" % left: round(float(np.asarray(
                        variants[left]["matrices"]["risk"])[rows[position]]), 6),
                    "R (%s)" % right: round(float(np.asarray(
                        variants[right]["matrices"]["risk"])[rows[position]]), 6),
                    "model that moved it most": mover,
                    "its change in C_m": round(float(gap.get(mover, np.nan)), 6),
                })
    return pd.DataFrame(summary), pd.DataFrame(examples)


# ===========================================================================
# 14.9.5  WHICH VARIANT THE ALERT REPORTS EXPLAIN
# ===========================================================================
def validation_macro_f1(tag: str, keys: Sequence[str],
                        val_probs: dict[str, np.ndarray],
                        fusion_inputs: dict[str, Any],
                        step13all: dict[str, Any] | None) -> float:
    """A variant's VALIDATION macro-F1 -- the only basis for choosing one.

    13.8 saved it for FALL and FSEL; F3's is computed here with 13.8's own
    `fit_on_validation`, on the validation probabilities only.  No test row
    takes part in this decision.
    """
    metrics = (step13all or {}).get("metrics") or {}
    block = (metrics.get("variants") or {}).get(tag) or {}
    saved = (block.get("validation_scores") or {}).get("macro_f1")
    if saved is not None:
        return float(saved)
    if not keys or any(k not in val_probs for k in keys):
        return float("nan")
    fitted = fa.fit_on_validation(list(keys), val_probs, fusion_inputs)
    return float(fitted["validation_scores"]["macro_f1"])


def choose_alert_variant(scores: dict[str, float],
                         setting: str | None = None) -> tuple[str, str]:
    """(tag, why).  "auto" takes the best VALIDATION macro-F1; a named variant
    is used as given, and refused if it is not available."""
    setting = (config.ALERT_FUSION_VARIANT if setting is None else setting)
    usable = {tag: value for tag, value in scores.items()
              if value is not None and not np.isnan(value)}
    if not usable:
        raise XaiAllError(
            "no fusion variant is available to explain: Step 13 and 13.8 both "
            "have to have run.")
    if setting == "auto":
        best = max(usable, key=lambda tag: (usable[tag], -VARIANTS.index(tag)))
        return best, ("ALERT_FUSION_VARIANT = \"auto\", and %s has the best "
                      "VALIDATION macro-F1 (%.4f against %s)"
                      % (best, usable[best],
                         ", ".join("%s %.4f" % (tag, value)
                                   for tag, value in usable.items()
                                   if tag != best)))
    if setting not in usable:
        raise XaiAllError(
            "ALERT_FUSION_VARIANT is %r but that variant is not available "
            "here (available: %s)" % (setting, ", ".join(sorted(usable))))
    return setting, ("ALERT_FUSION_VARIANT = %r, so that variant is explained "
                     "whatever its validation score (%.4f)"
                     % (setting, usable[setting]))


def other_models_row(position: int, key_order: Sequence[str],
                     fused_keys: Sequence[str],
                     probabilities: dict[str, np.ndarray], row: int,
                     label_names: dict[int, str]) -> dict[str, str]:
    """The compact line for models that are NOT in the chosen fusion.

    Predicted class and probability only: they did not contribute to R, so
    giving them the same space as a fused model would overstate their part.
    """
    pieces = []
    for key in key_order:
        if key in fused_keys or key not in probabilities:
            continue
        probability = np.asarray(probabilities[key], dtype="float64")[row]
        predicted = int(np.argmax(probability))
        pieces.append("%s: %s (P = %.2f)"
                      % (short_label(key),
                         label_names.get(predicted, str(predicted)),
                         float(probability.max())))
    return {"Evidence component": "Other models (not fused)",
            "Detail": ("; ".join(pieces) if pieces
                       else "every available model is in this fusion"),
            "Confidence": "Context"}


def build_reports(tag: str, fused_keys: Sequence[str], rows: np.ndarray,
                  breakdown: pd.DataFrame, matrices: dict[str, np.ndarray],
                  explained: dict[str, dict[str, Any]],
                  probabilities: dict[str, np.ndarray], inputs: dict[str, Any],
                  bands: np.ndarray, tau: float, tau_s: float | None,
                  completeness_ok: dict[int, bool], limit: int | None = None
                  ) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """One evidence report per alert, through `explainability.build_alert_report`.

    Step 14's assembler does the work; this supplies the context and appends
    the one row it has no reason to know about -- the models outside this
    fusion.  Models are addressed by their short display label throughout, so
    no internal key reaches the report.
    """
    label_names = inputs["label_names"]
    orig_row = inputs["orig_row"]
    labels = [short_label(k) for k in fused_keys]
    attributions = {short_label(k): explained[k]["attributions"][TARGET_RISK]
                    for k in fused_keys if k in explained}
    feature_names = {short_label(k): explained[k]["features"]
                     for k in fused_keys if k in explained}
    training_counts = inputs["y_train"][TARGET_COLUMN].value_counts().to_dict()

    context = {
        "label_names": label_names, "model_names": labels,
        "breakdown": breakdown, "meta": inputs["meta_test"],
        "tau": tau, "tau_s": tau_s,
        "risk": matrices["risk"], "bands": bands,
        "fused_class": matrices["fused_class"],
        "fused_probabilities": matrices["fused_probabilities"],
        "completeness_ok": completeness_ok,
        "attributions": attributions, "feature_names": feature_names,
        "unscaled_rows": {int(r): inputs["X_test_unscaled"].iloc[int(r)]
                          for r in rows},
        "context_labels": {int(r): matrices["context"][int(r)] for r in rows},
        "training_rows": {int(k): int(v) for k, v in training_counts.items()},
        "y_true": np.asarray(inputs["y_test"][TARGET_COLUMN]),
    }

    reports, flats = [], []
    for position, row in enumerate(rows):
        if bands[row] not in (BAND_ATTACK, BAND_SUSPICIOUS):
            continue
        report = xai.build_alert_report(position, int(row), context)
        extra = other_models_row(position, ordered(list(probabilities)),
                                 fused_keys, probabilities, int(row),
                                 label_names)
        report["table"] = pd.concat(
            [report["table"], pd.DataFrame([extra])], ignore_index=True)
        report["flat"]["row: Other models (not fused)"] = extra["Detail"]
        report["flat"]["variant"] = tag
        report["variant"] = tag
        report["Orig_Row"] = orig_row[position]
        reports.append(report)
        flats.append(report["flat"])
        if limit is not None and len(reports) >= limit:
            break
    return reports, pd.DataFrame(flats)


def examples_per_class(reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The first alert of each fused attack class, in label order."""
    seen, chosen = set(), []
    for report in reports:
        label = report["fused_label"]
        if label in seen:
            continue
        seen.add(label)
        chosen.append(report)
    return sorted(chosen, key=lambda r: str(r["fused_label"]))


def render_report(report: dict[str, Any], dominant: str | None = None) -> None:
    """One alert as a clean three-column card."""
    title = "**ALERT %s -- %s (%s)**" % (report["alert_id"],
                                         report["fused_label"], report["band"])
    if dl.in_notebook():
        from IPython.display import Markdown, display
        display(Markdown("---"))
        display(Markdown(title + ("  \nDominant model: **%s**" % dominant
                                  if dominant else "")))
    else:
        print()
        print("-" * 78)
        print(title.replace("**", ""))
    table = report["table"].copy()
    if dominant:
        table["Evidence component"] = [
            ("%s  <-- dominant" % name)
            if name.endswith(dominant) and name.startswith("Model") else name
            for name in table["Evidence component"]]
    present(table, "Evidence behind %s. The decision row is first; every model "
                   "row carries P, w, its contribution C = w*r and its top-%d "
                   "features with their original unscaled values."
            % (report["alert_id"], config.TOP_K_FEATURES))


# ===========================================================================
# 14.9.x  CHARTS -- one idea each, one colour per model everywhere
# ===========================================================================
def _diverging_cmap():
    """Two hues and a neutral middle, for a measure that has a sign.

    A correlation runs from -1 to 1 through 0, so a single-hue ramp would hide
    the sign.  The two poles are the project's own blue and orange.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "mscad_diverging", ["#eb6834", "#f6c3ac", "#ebebe8", "#9ec5f4",
                            "#2a78d6"])


def plot_global_importance(matrix: pd.DataFrame, keys: Sequence[str], path,
                           top: int = 8, verbose: bool = True):
    """Each model's top features as horizontal bars, one panel per model."""
    plt = fz._pyplot(verbose)
    if plt is None or matrix.empty:
        return None
    labels = [display_label(k) for k in keys if display_label(k) in matrix.columns]
    columns = min(len(labels), 3) or 1
    rows = int(np.ceil(len(labels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10.0, 2.6 * rows + 1.0),
                             dpi=150, squeeze=False)
    fig.patch.set_facecolor(dl._SURFACE)

    for position, label in enumerate(labels):
        ax = axes[position // columns][position % columns]
        fz._style(ax)
        key = [k for k in keys if display_label(k) == label][0]
        series = matrix[label].dropna().sort_values(ascending=False).head(top)
        names = [str(n)[:28] for n in series.index][::-1]
        values = series.to_numpy()[::-1]
        ax.barh(np.arange(len(names)), values, 0.68, color=model_colour(key))
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=7.5, color=dl._INK_SOFT)
        ax.set_title(short_label(key), color=dl._INK, fontsize=10, pad=8)
        ax.set_xlabel("mean |IG| (attack risk)", color=dl._INK_SOFT, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.grid(True, color=dl._GRID, linewidth=0.8)
        ax.yaxis.grid(False)
    for empty in range(len(labels), rows * columns):
        axes[empty // columns][empty % columns].axis("off")
    fig.suptitle("Top features per model, by mean |IG| on the attack risk",
                 color=dl._INK, fontsize=12)
    return fz._finish(fig, path, verbose)


def plot_agreement_heatmap(grid: pd.DataFrame, path, title: str,
                           verbose: bool = True):
    """A signed model x model matrix: two hues through a neutral middle."""
    plt = fz._pyplot(verbose)
    if plt is None or grid.empty:
        return None
    values = grid.to_numpy(dtype="float64")
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * grid.shape[1] + 3.0),
                                    max(4.0, 0.9 * grid.shape[0] + 2.2)),
                           dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)
    image = ax.imshow(values, cmap=_diverging_cmap(), vmin=-1.0, vmax=1.0,
                      aspect="auto", interpolation="nearest")

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
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(values[i, j]):
                ax.annotate(NA_TEXT, (j, i), ha="center", va="center",
                            fontsize=8, color=dl._INK_SOFT)
                continue
            ax.annotate("%.2f" % values[i, j], (j, i), ha="center",
                        va="center", fontsize=8.5,
                        color="#ffffff" if abs(values[i, j]) > 0.75 else dl._INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("Spearman of mean |IG| on shared features",
                  color=dl._INK_SOFT, fontsize=9)
    bar.ax.tick_params(colors=dl._INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)
    return fz._finish(fig, path, verbose)


def plot_quality(table: pd.DataFrame, keys: Sequence[str], path,
                 verbose: bool = True):
    """Three panels, because the three measures have three different units."""
    plt = fz._pyplot(verbose)
    if plt is None or table.empty:
        return None
    panels = [
        ("completeness within tolerance (%)", "share of flows (%)", (0, 105)),
        ([c for c in table.columns if c.startswith("deletion gap")][0],
         "risk drop, top-K minus random", None),
        ("stability (Spearman under noise)", "Spearman", (0, 1.05)),
    ]
    labels = list(table["model"])
    colours = [model_colour([k for k in keys if display_label(k) == label][0])
               for label in labels]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    for ax, (column, ylabel, limits) in zip(axes, panels):
        fz._style(ax)
        values = pd.to_numeric(table[column], errors="coerce").to_numpy()
        bars = ax.bar(np.arange(len(labels)), np.nan_to_num(values), 0.62,
                      color=colours)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.annotate("%.2f" % value,
                        (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7.5, color=dl._INK)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels([str(v).split(":")[0] for v in labels], rotation=35,
                           ha="right", fontsize=8, color=dl._INK_SOFT)
        ax.set_ylabel(ylabel, color=dl._INK_SOFT, fontsize=9)
        ax.set_title(column.split(" (")[0], color=dl._INK, fontsize=10, pad=8)
        if limits:
            ax.set_ylim(*limits)
    fig.suptitle("Explanation quality per model -- higher is better in all "
                 "three", color=dl._INK, fontsize=12)
    return fz._finish(fig, path, verbose)


def plot_deletion(explained: dict[str, dict[str, Any]], keys: Sequence[str],
                  path, verbose: bool = True):
    """Top-K removal against a random removal of the same size, per model."""
    plt = fz._pyplot(verbose)
    if plt is None or not explained:
        return None
    labels = [short_label(k) for k in keys if k in explained]
    top = [explained[k]["deletion"]["mean_drop_top_k"] for k in keys
           if k in explained]
    chance = [explained[k]["deletion"]["mean_drop_random_k"] for k in keys
              if k in explained]
    positions = np.arange(len(labels), dtype="float64")
    fig, ax = plt.subplots(figsize=(10.0, 4.6), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    fz._style(ax)
    ax.bar(positions - 0.19, top, 0.36, label="top-K features removed",
           color="#2a78d6")
    ax.bar(positions + 0.19, chance, 0.36, label="random K removed",
           color="#b9c6d6")
    for x, (a, b) in enumerate(zip(top, chance)):
        ax.annotate("%.3f" % a, (x - 0.19, a), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7.5, color=dl._INK)
        ax.annotate("%.3f" % b, (x + 0.19, b), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7.5, color=dl._INK)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_ylabel("mean drop in attack risk", color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Deletion check: does removing the attributed features cost "
                 "more than chance?", color=dl._INK, fontsize=12, pad=14)
    legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return fz._finish(fig, path, verbose)


def plot_dominant(share: pd.DataFrame, keys: Sequence[str], path,
                  verbose: bool = True):
    """Who dominates each variant's alerts, as one stacked bar per variant."""
    plt = fz._pyplot(verbose)
    if plt is None or share.empty:
        return None
    part = share[share["scope"] == "all explained alerts"]
    variants = list(dict.fromkeys(part["variant"]))
    labels = [short_label(k) for k in keys]
    fig, ax = plt.subplots(figsize=(10.0, 4.6), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    fz._style(ax)
    positions = np.arange(len(variants), dtype="float64")
    bottom = np.zeros(len(variants))
    gap = 0.4
    for index, key in enumerate(keys):
        name = short_label(key)
        values = np.array([
            float(part[(part["variant"] == tag) & (part["model"] == name)]
                  ["share (%)"].sum()) for tag in variants])
        if not values.any():
            continue
        ax.bar(positions, values, 0.55, bottom=bottom + index * gap, label=name,
               color=model_colour(key))
        for x, (value, base) in enumerate(zip(values, bottom)):
            if value > 6:
                ax.annotate("%.0f%%" % value, (x, base + value / 2),
                            ha="center", va="center", fontsize=8,
                            color="#ffffff")
        bottom = bottom + values
    ax.set_xticks(positions)
    ax.set_xticklabels(variants, fontsize=10, color=dl._INK_SOFT)
    ax.set_ylabel("share of explained alerts (%)", color=dl._INK_SOFT,
                  fontsize=10)
    ax.set_title("Which model contributes most to the risk, per fusion variant",
                 color=dl._INK, fontsize=12, pad=14)
    ax.set_ylim(0, 118)
    legend = ax.legend(frameon=False, loc="upper right", fontsize=8.5,
                       ncol=min(len(labels), 4))
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return fz._finish(fig, path, verbose)


# ===========================================================================
# 14.9.x  THE EXISTING STEP 14 OUTPUTS -- read, and proved untouched
# ===========================================================================
PROTECTED_STEP14 = ("metrics", "alerts")        # the two the brief names


def step14_fingerprint() -> dict[str, str]:
    """SHA-256 of every file in the Step 14 folder, as it stands now."""
    folder = config.usable_output_dir(config.STEP14_OUTPUT_DIR)
    if folder is None or not Path(folder).exists():
        return {}
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(folder).glob("*")) if path.is_file()}


def assert_step14_unchanged(before: dict[str, str], verbose: bool = False
                            ) -> tuple[pd.DataFrame, bool]:
    """[OK]/[FAIL] that nothing in step14\\ moved while this addendum ran."""
    after = step14_fingerprint()
    names = sorted(set(before) | set(after))
    protected = {config.STEP14_FILES[key] for key in PROTECTED_STEP14}
    rows = []
    for name in names:
        old, new = before.get(name), after.get(name)
        rows.append({"file": name,
                     "named in the brief": name in protected,
                     "sha256 before": (old or "(absent)")[:16],
                     "sha256 after": (new or "(absent)")[:16],
                     "result": "[OK]" if old == new and old is not None
                               else "[FAIL]"})
    if not names:
        rows.append({"file": "(the Step 14 folder is not readable here)",
                     "named in the brief": False, "sha256 before": "",
                     "sha256 after": "", "result": "[SKIP]"})
    table = pd.DataFrame(rows)
    ok = bool((table["result"] != "[FAIL]").all())
    if verbose:
        print("[%s]  existing Step 14 outputs unchanged"
              % ("OK" if ok else "FAIL"))
    return table, ok


def load_step14_outputs() -> dict[str, Any]:
    """Step 14's saved tables, read only -- its explained flows and its tau."""
    folder = config.usable_output_dir(config.STEP14_OUTPUT_DIR)
    out: dict[str, Any] = {}
    if folder is None:
        return out
    folder = Path(folder)
    names = config.STEP14_FILES
    for key in ("explanations", "alerts", "global_importance"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    for key in ("metrics", "thresholds"):
        path = folder / names[key]
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def step14_explained_rows(step14: dict[str, Any],
                          meta_test: pd.DataFrame | None,
                          n_rows: int) -> np.ndarray:
    """The TEST ROWS Step 14 explained, from the Orig_Rows it saved.

    Explaining the same flows is what makes 14.9.3 a comparison rather than two
    unrelated samples, so they are read back rather than re-sampled.
    """
    frame = step14.get("explanations")
    if frame is None or "Orig_Row" not in frame.columns:
        return np.array([], dtype="int64")
    wanted = pd.unique(frame["Orig_Row"])
    if meta_test is None or "Orig_Row" not in meta_test.columns:
        return np.array(sorted(int(v) for v in wanted if int(v) < n_rows),
                        dtype="int64")
    position = {int(value): index for index, value
                in enumerate(meta_test["Orig_Row"].to_numpy())}
    rows = [position[int(v)] for v in wanted if int(v) in position]
    return np.array(sorted(set(rows)), dtype="int64")


# ===========================================================================
# 14.9.x  VERIFICATION -- one table, not scattered checks
# ===========================================================================
class Checks:
    """Every check in one place, with the section it belongs to."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, section: str, check: str, ok: bool | None, detail: str
            ) -> None:
        self.rows.append({
            "section": section, "check": check,
            "result": "[SKIP]" if ok is None else ("[OK]" if ok else "[FAIL]"),
            "detail": detail})

    def frame(self, section: str | None = None) -> pd.DataFrame:
        table = pd.DataFrame(self.rows)
        if section is not None and len(table):
            table = table[table["section"] == section].reset_index(drop=True)
        return table

    @property
    def ok(self) -> bool:
        table = self.frame()
        return bool(len(table)) and bool((table["result"] != "[FAIL]").all())


def verify_from_disk(folder: Path, out: dict[str, Any], checks: Checks
                     ) -> None:
    """Read the saved files back and check them, as a reader would find them."""
    names = config.STEP14ALL_FILES
    section = "14.9.6 saved files"
    for key in ("composed_checks", "explanations", "global_importance",
                "quality", "agreement", "contributions", "alerts",
                "metrics", "examples", "summary"):
        path = folder / names[key]
        checks.add(section, "%s was written" % names[key], path.exists(),
                   "{:,} bytes".format(path.stat().st_size) if path.exists()
                   else "missing")

    path = folder / names["explanations"]
    if path.exists():
        frame = pd.read_csv(path, low_memory=False)
        expected = sum(len(out["explained"][k]["features"]) * 2
                       * len(out["rows"]) for k in out["explained"])
        checks.add(section, "explanations_all.csv has one row per "
                            "(flow, model, feature, target)",
                   len(frame) == expected,
                   "{:,} rows, expected {:,}".format(len(frame), expected))
        checks.add(section, "no NaN in the saved attributions",
                   bool(frame["ig"].notna().all()),
                   "{:,} values".format(len(frame)))

    path = folder / names["global_importance"]
    if path.exists():
        frame = pd.read_csv(path, low_memory=False)
        models = frame["model"].nunique()
        checks.add(section, "global importance covers every explained model",
                   models == len(out["explained"]),
                   "%d models" % models)

    path = folder / names["contributions"]
    if path.exists():
        frame = pd.read_csv(path, low_memory=False)
        for tag in out["variants"]:
            part = frame[frame["variant"] == tag]
            if not len(part):
                continue
            totals = part.groupby("flow")["C_m"].sum()
            risk = np.asarray(out["variants"][tag]["matrices"]["risk"],
                              dtype="float64")[out["rows"]]
            worst = float(np.abs(totals.to_numpy() - risk).max())
            checks.add(section, "%s: R equals the sum of its contributions"
                                % tag, worst <= 1e-6,
                       "max |R - sum C_m| = %.2e over {:,} alerts".format(
                           len(totals)) % worst)

    path = folder / names["alerts"]
    if path.exists():
        frame = pd.read_csv(path, low_memory=False)
        checks.add(section, "alert_explanations_all.csv has one row per alert",
                   len(frame) == len(out["reports"]),
                   "{:,} rows".format(len(frame)))
        if "dominant_model" in frame.columns:
            keys = {short_label(k) for k in out["fused_keys"]}
            checks.add(section, "no internal key appears in the alert table",
                       bool(set(frame["dominant_model"]) <= keys),
                       ", ".join(sorted(set(frame["dominant_model"]))[:4]))


# ===========================================================================
# 14.9.6  SUMMARY, WRITTEN FROM THE COMPUTED NUMBERS
# ===========================================================================
def _fmt(value: Any, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if np.isnan(number) else ("%%.%df" % places) % number


def build_summary(out: dict[str, Any]) -> str:
    """The 14.9.6 discussion: every claim traced to a computed table."""
    lines: list[str] = ["### 14.9.6 Explainability across all models -- what "
                        "the numbers say", ""]
    matrix = out["importance_matrix"]
    quality = out["quality"]
    k = getattr(config, "XAI_TOP_FEATURES", 10)

    explained = [display_label(key) for key in out["fused_keys_all"]]
    lines.append("**Models explained.** %d of %d configurations: %s."
                 % (len(explained), len(out["offered_keys"]),
                    ", ".join(explained)))
    skipped = out["availability"][~out["availability"]["available"]]
    if len(skipped):
        lines.append("Skipped: %s. Nothing was trained to fill the gap."
                     % "; ".join(skipped["note"]))
    lines.append("")

    # --- which features drive the models -------------------------------
    lines.append("**What drives the attack classes.**")
    if len(matrix.columns):
        counts: dict[str, list[str]] = {}
        for label in matrix.columns:
            for feature in top_features(matrix, label, k):
                counts.setdefault(feature, []).append(label)
        if counts:
            ranked = sorted(counts.items(),
                            key=lambda item: (-len(item[1]), item[0]))
            top_feature, holders = ranked[0]
            lines.append("- `%s` is in the top-%d of %d of the %d explained "
                         "models -- the widest agreement in the table."
                         % (top_feature, k, len(holders), len(matrix.columns)))
            shared = [name for name, models in ranked if len(models) >= 2][:5]
            if shared:
                lines.append("- Features in at least two models' top-%d: %s."
                             % (k, ", ".join("`%s`" % f for f in shared)))
            alone = [name for name, models in ranked if len(models) == 1][:5]
            if alone:
                lines.append("- Features only one model ranks that highly: %s "
                             "-- where the models' reasoning parts company."
                             % ", ".join("`%s` (%s)" % (f, counts[f][0]
                                                        .split(":")[0])
                                         for f in alone))
    agreement = out.get("agreement_spearman")
    if agreement is not None and len(agreement) > 1:
        values = agreement.to_numpy(dtype="float64").copy()
        np.fill_diagonal(values, np.nan)
        flat = [(agreement.index[i], agreement.columns[j], values[i, j])
                for i in range(values.shape[0]) for j in range(values.shape[1])
                if i < j and not np.isnan(values[i, j])]
        if flat:
            closest = max(flat, key=lambda f: f[2])
            furthest = min(flat, key=lambda f: f[2])
            lines.append("- The most similar reasoning is %s and %s "
                         "(Spearman %.3f on their shared features); the least "
                         "similar is %s and %s (%.3f)."
                         % (closest[0].split(":")[0], closest[1].split(":")[0],
                            closest[2], furthest[0].split(":")[0],
                            furthest[1].split(":")[0], furthest[2]))
    lines.append("")

    # --- whose explanations hold up ------------------------------------
    lines.append("**Whose explanations hold up.**")
    if len(quality):
        deletion_column = [c for c in quality.columns
                           if c.startswith("deletion gap")][0]
        for column, direction, what, unit, places in (
                ("completeness within tolerance (%)", "max",
                 "completeness (sum IG = F(x) - F(x'))", "% of flows", 1),
                ("IG vs perturbation (Spearman)", "max",
                 "agreement with the perturbation check", "Spearman", 4),
                (deletion_column, "max", "deletion-check gap",
                 "risk drop above chance", 4),
                ("stability (Spearman under noise)", "max",
                 "stability under small noise", "Spearman", 4),
                ("IG ms per flow", "min", "IG cost per flow",
                 "ms per flow", 2)):
            values = pd.to_numeric(quality[column], errors="coerce")
            if values.notna().sum() == 0:
                continue
            position = values.idxmin() if direction == "min" else values.idxmax()
            lines.append("- %s %s: %s (%s %s)."
                         % ("Lowest" if direction == "min" else "Best", what,
                            quality.loc[position, "model"].split(":")[0],
                            _fmt(values[position], places), unit))
    lines.append("")

    # --- the fusion variants --------------------------------------------
    lines.append("**Does the chosen fusion lean on one model?**")
    lines.append("- The alert reports explain **%s**: %s."
                 % (out["variant"], out["variant_reason"]))
    share = out.get("dominant_share")
    if share is not None and len(share):
        part = share[(share["variant"] == out["variant"])
                     & (share["scope"] == "all explained alerts")]
        if len(part):
            leader = part.loc[part["share (%)"].idxmax()]
            lines.append("- %s has the largest contribution C = w*r in %.1f%% "
                         "of the explained alerts under %s; the rest are split "
                         "across %d other model(s)."
                         % (leader["model"], leader["share (%)"],
                            out["variant"],
                            int((part["share (%)"] > 0).sum() - 1)))
            if float(leader["share (%)"]) >= 80:
                lines.append("  - That is a fusion resting on one model: the "
                             "others change the number, not the verdict.")
    differences = out.get("variant_disagreement")
    if differences is not None and len(differences):
        for _i, row in differences.iterrows():
            lines.append("- %s disagree on the decision for %d of %s explained "
                         "alerts (%.1f%%)."
                         % (row["pair"], row["decisions that differ"],
                            "{:,}".format(row["explained alerts"]),
                            row["share (%)"]))
    lines.append("")

    # --- caveats ---------------------------------------------------------
    lines.append("**Caveats.**")
    lines.append("- IG explains **model behaviour**, not ground-truth causes. "
                 "A feature with a large attribution is one this model leaned "
                 "on, which is not the same as a feature that caused the "
                 "attack.")
    lines.append("- Approach 2's attention is **supplementary**: it weights "
                 "CNN feature-map positions, not input features, so it cannot "
                 "name a flow feature and is never used as the explanation "
                 "here.")
    lines.append("- The models have different inputs. A blank cell is `%s` -- "
                 "%s -- never a zero, which would claim the model looked and "
                 "found nothing." % (NA_TEXT, NA_FOOTNOTE.split("= ")[1]))
    rare = out.get("rare_classes") or []
    if rare:
        lines.append("- Few explained alerts for: %s. A per-class reading on "
                     "that many alerts moves in large steps."
                     % ", ".join("%s (%d)" % (name, count)
                                 for name, count in rare))
    lines.append("- One seed (RANDOM_STATE = %d), one IG baseline "
                 "(IG_BASELINE = %r) and %d IG steps. Every figure above is a "
                 "single run, not an estimate with a spread."
                 % (config.RANDOM_STATE, config.IG_BASELINE, config.IG_STEPS))
    lines.append("- Step 14 is unchanged: it explains F3, and every file in "
                 "`step14\\` is SHA-256 identical before and after this "
                 "section.")
    return "\n".join(lines)


# ===========================================================================
# 14.9.x  SAVE, THEN READ BACK AND CHECK
# ===========================================================================
def build_metrics(out: dict[str, Any], saved_files: dict[str, str],
                  summary: str | None = None) -> dict[str, Any]:
    """Everything needed to reproduce these numbers, as one JSON object."""
    quality = out["quality"]
    return {
        "step": "14.9",
        "name": "Explainability across all approaches, baselines and fusion "
                "variants",
        "offered": [display_label(k) for k in out["offered_keys"]],
        "explained": [display_label(k) for k in out["fused_keys_all"]],
        "skipped": out["availability"].loc[
            ~out["availability"]["available"], "model"].tolist(),
        "explained_flows": int(len(out["rows"])),
        "alerts_reported": int(len(out["reports"])),
        "alert_variant": out["variant"],
        "alert_variant_reason": out["variant_reason"],
        "variant_validation_macro_f1": {
            tag: (None if value is None or np.isnan(value) else round(float(value), 6))
            for tag, value in (out.get("variant_scores") or {}).items()},
        "variants_explained": {
            tag: {"models": [short_label(k) for k in block["keys"]],
                  "tau": round(float(block["tau"]), 6),
                  "max_abs_R_minus_sum_C": round(
                      float(block["consistency"]["max_abs_difference"]), 12),
                  "consistent": bool(block["consistency"]["ok"])}
            for tag, block in out["variants"].items()},
        "settings": {
            "XAI_ALL_MODELS": list(config.XAI_ALL_MODELS),
            "ALERT_FUSION_VARIANT": config.ALERT_FUSION_VARIANT,
            "XAI_TOP_FEATURES": getattr(config, "XAI_TOP_FEATURES", 10),
            # reused from the Step 14 block, unchanged
            "IG_STEPS": config.IG_STEPS, "IG_BASELINE": config.IG_BASELINE,
            "TOP_K_FEATURES": config.TOP_K_FEATURES,
            "COMPLETENESS_TOL": config.COMPLETENESS_TOL,
            "STABILITY_NOISE_STD": config.STABILITY_NOISE_STD,
            "STABILITY_N_REPEATS": config.STABILITY_N_REPEATS,
            "PERTURB_MODE": config.PERTURB_MODE,
            "EXPLAIN_SAMPLE_PER_CLASS": config.EXPLAIN_SAMPLE_PER_CLASS,
        },
        "quality": {row["model"]: {k: (None if pd.isna(v) else v)
                                   for k, v in row.items() if k != "model"}
                    for _i, row in quality.iterrows()},
        "ig_baselines": {display_label(k): {
            "attack_risk_at_baseline": round(float(
                out["explained"][k]["baseline_risk"]), 6),
            "features": len(out["explained"][k]["features"])}
            for k in out["explained"]},
        "step14_unchanged": bool(out.get("step14_unchanged")),
        "step14_fingerprint": dict(out.get("step14_before") or {}),
        "leakage_note": ("the IG baseline is the median of the TRAINING benign "
                         "rows in each model's own input space; tau comes from "
                         "Step 13 unchanged, tau_s from the Step 9 validation "
                         "split, and the alert variant is chosen on validation "
                         "macro-F1"),
        "files": saved_files,
        "summary": summary,
    }


def save_step14all(out: dict[str, Any], checks: Checks, verbose: bool = True
                   ) -> dict[str, Any]:
    """Write every 14.9 output into step14_all\\, then verify it from disk."""
    names = config.STEP14ALL_FILES
    result: dict[str, Any] = {}
    folder = prepare_output_dir(config.STEP14ALL_OUTPUT_DIR,
                                "the Step 14.9 outputs", verbose=verbose)
    if folder is None:
        result.update({"status": "skipped_not_windows", "ok": False})
        return result
    folder = Path(folder)

    out["composed_checks"].to_csv(folder / names["composed_checks"], index=False)
    out["availability"].to_csv(folder / "model_availability_all.csv", index=False)
    out["ig_baselines"].to_csv(folder / names["ig_baselines"], index=False)
    out["explanations"].to_csv(folder / names["explanations"], index=False)
    out["global_importance"].to_csv(folder / names["global_importance"],
                                    index=False)
    out["quality"].to_csv(folder / names["quality"], index=False)
    out["agreement_spearman"].to_csv(folder / names["agreement"])
    out["consensus"].to_csv(folder / names["consensus"], index=False)
    out["contributions"].to_csv(folder / names["contributions"], index=False)
    out["variant_table"].to_csv(folder / names["variants"], index=False)
    if out.get("dominant_share") is not None and len(out["dominant_share"]):
        out["dominant_share"].to_csv(folder / names["dominant"], index=False)
    if (out.get("variant_disagreement") is not None
            and len(out["variant_disagreement"])):
        out["variant_disagreement"].to_csv(folder / names["disagreement"],
                                           index=False)
    # the example reports as rows, so a reader who only loads the saved outputs
    # still gets the cards rather than an empty cell
    example_rows = []
    for report in out.get("examples", []):
        for _i, line in report["table"].iterrows():
            example_rows.append({"alert_id": report["alert_id"],
                                 "fused_label": report["fused_label"],
                                 "band": report["band"],
                                 "dominant_model": report["flat"].get(
                                     "dominant_model", ""),
                                 **dict(line)})
    if example_rows:
        pd.DataFrame(example_rows).to_csv(folder / names["example_rows"],
                                          index=False)
    out["alerts"].to_csv(folder / names["alerts"], index=False)
    if out.get("attention") is not None and len(out["attention"]):
        out["attention"].to_csv(folder / names["attention"], index=False)
    (folder / names["examples"]).write_text(
        xai.examples_markdown(out["examples"]), encoding="utf-8")

    # ---- charts ----------------------------------------------------------
    charts: dict[str, str] = {}
    keys = list(out["explained"])
    drawn = plot_global_importance(out["importance_matrix"], keys,
                                   folder / names["chart_global"],
                                   verbose=verbose)
    if drawn:
        charts["global"] = str(drawn)
    matrix = out["importance_matrix"]
    if len(matrix):
        top_rows = (matrix.max(axis=1).sort_values(ascending=False)
                    .head(15).index)
        drawn = fa.plot_diversity_heatmap(
            matrix.loc[top_rows], folder / names["chart_feature_heatmap"],
            "Mean |IG| on the attack risk -- top features x model",
            fmt="%.3f", vmin=0.0,
            vmax=float(np.nanmax(matrix.loc[top_rows].to_numpy())) or 1.0,
            verbose=verbose)
        if drawn:
            charts["feature_heatmap"] = str(drawn)
    drawn = plot_agreement_heatmap(
        out["agreement_spearman"], folder / names["chart_agreement"],
        "Do two models reason alike? Spearman of mean |IG| on shared features",
        verbose=verbose)
    if drawn:
        charts["agreement"] = str(drawn)
    drawn = plot_quality(out["quality"], keys, folder / names["chart_quality"],
                         verbose=verbose)
    if drawn:
        charts["quality"] = str(drawn)
    drawn = plot_deletion(out["explained"], keys,
                          folder / names["chart_deletion"], verbose=verbose)
    if drawn:
        charts["deletion"] = str(drawn)
    drawn = plot_dominant(out["dominant_share"], keys,
                          folder / names["chart_dominant"], verbose=verbose)
    if drawn:
        charts["dominant"] = str(drawn)
    attention = out.get("attention")
    if attention is not None and len(attention):
        grid = attention.pivot(index="predicted class", columns="position",
                               values="mean attention")
        drawn = fa.plot_diversity_heatmap(
            grid, folder / names["chart_attention"],
            "Approach 2 -- mean attention per feature-map position "
            "(supplementary, not the explanation)",
            fmt="%.2f", vmin=0.0,
            vmax=float(np.nanmax(grid.to_numpy())) or 1.0, verbose=verbose)
        if drawn:
            charts["attention"] = str(drawn)
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

    verify_from_disk(folder, out, checks)
    table, ok = assert_step14_unchanged(out["step14_before"], verbose=False)
    checks.add("14.9.6 saved files",
               "the existing Step 14 outputs are byte-identical", ok,
               "SHA-256 before and after, %d files" % len(table))
    out["step14_unchanged"] = ok
    out["step14_unchanged_table"] = table

    verification = checks.frame()
    verification.to_csv(folder / names["verification"], index=False)
    verification.to_csv(folder / names["report"], index=False)
    result.update({"verification": verification, "ok": bool(checks.ok),
                   "status": "written", "folder": str(folder)})
    if verbose:
        print("  [%s]  Step 14.9 outputs written to %s"
              % ("OK" if checks.ok else "FAIL", folder))
    return result


def step14all_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP14ALL_OUTPUT_DIR)
    if folder is None:
        return False
    return all((Path(folder) / config.STEP14ALL_FILES[key]).exists()
               for key in ("metrics", "quality", "alerts", "explanations"))


def load_step14all(verbose: bool = True) -> dict[str, Any]:
    """Read a finished 14.9 back from disk.  Nothing is recomputed."""
    folder = config.usable_output_dir(config.STEP14ALL_OUTPUT_DIR)
    if folder is None:
        raise XaiAllError("the Step 14.9 folder is not usable here -- %s"
                          % config.why_unusable(config.STEP14ALL_OUTPUT_DIR))
    folder = Path(folder)
    names = config.STEP14ALL_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    for key in ("composed_checks", "ig_baselines", "explanations",
                "global_importance", "quality", "consensus", "contributions",
                "variants", "alerts", "attention", "verification",
                "dominant", "disagreement", "example_rows"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    out["variant_table"] = out.get("variants")
    out["dominant_share"] = out.get("dominant")
    out["variant_disagreement"] = out.get("disagreement")

    # feature x model, rebuilt from the saved long table rather than left out:
    # 14.9.3's main table must look the same on the resumable path
    importance = out.get("global_importance")
    if importance is not None and len(importance):
        part = importance[(importance["target"] == TARGET_RISK)
                          & (importance["scope"] == "all explained flows")]
        out["importance_matrix"] = part.pivot(index="feature", columns="model",
                                              values="mean_abs_ig")
        out["importance_matrix"].index.name = "feature"
    else:
        out["importance_matrix"] = pd.DataFrame()

    # the example cards, rebuilt from their saved rows
    rows = out.get("example_rows")
    examples = []
    if rows is not None and len(rows):
        for alert_id, part in rows.groupby("alert_id", sort=False):
            examples.append({
                "alert_id": alert_id,
                "fused_label": part["fused_label"].iloc[0],
                "band": part["band"].iloc[0],
                "table": part[["Evidence component", "Detail",
                               "Confidence"]].reset_index(drop=True),
                "flat": {"dominant_model": part["dominant_model"].iloc[0]}})
    out["examples"] = examples
    path = folder / "model_availability_all.csv"
    if path.exists():
        out["availability"] = pd.read_csv(path)
    path = folder / names["agreement"]
    if path.exists():
        out["agreement_spearman"] = pd.read_csv(path, index_col=0)
    path = folder / names["metrics"]
    if path.exists():
        out["metrics"] = json.loads(path.read_text(encoding="utf-8"))
    path = folder / names["summary"]
    out["summary"] = (path.read_text(encoding="utf-8") if path.exists()
                      else (out.get("metrics") or {}).get("summary"))
    path = folder / names["examples"]
    out["examples_markdown"] = (path.read_text(encoding="utf-8")
                                if path.exists() else "")

    recorded = (out.get("metrics") or {}).get("step14_fingerprint") or {}
    if recorded:
        table, ok = assert_step14_unchanged(recorded, verbose=False)
        out["step14_unchanged_table"] = table
        out["step14_unchanged"] = ok
    else:
        out["step14_unchanged"] = bool(
            (out.get("metrics") or {}).get("step14_unchanged"))
    metrics = out.get("metrics") or {}
    out["variant"] = metrics.get("alert_variant")
    out["variant_reason"] = metrics.get("alert_variant_reason")
    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP14ALL = False)"),
            ("folder", str(folder)),
            ("models explained", ", ".join(metrics.get("explained", []))),
            ("alert variant", metrics.get("alert_variant")),
            ("alerts reported", "{:,}".format(metrics.get("alerts_reported", 0))),
        ]), note="Set `OVERWRITE_STEP14ALL = True` in config.py to recompute.")
    return out


# ===========================================================================
# 14.9.7  RUN EVERYTHING
# ===========================================================================
def _validation_probabilities(bundles: dict[str, dict], y_train: pd.DataFrame
                              ) -> tuple[dict[str, np.ndarray], pd.DataFrame,
                                         np.ndarray]:
    """Each model's probabilities on the Step 9 VALIDATION rows.

    Through the same composed models the attributions use, and through Step
    13's own `validation_positions`, so these are the rows every earlier step
    validated on.
    """
    from . import imbalance as im

    marker = pd.DataFrame({"row": np.arange(len(y_train))})
    _fit, _yf, val_rows, y_val, _w, _s = im.make_training_sets(marker, y_train,
                                                               verbose=False)
    positions = val_rows["row"].to_numpy()
    probabilities = {}
    for key, bundle in bundles.items():
        X_val = bundle["X_train"].iloc[positions].reset_index(drop=True)
        probabilities[key] = xai.composed_probabilities(bundle["composed"],
                                                        X_val)
    return probabilities, y_val, positions


def run_step14all(overwrite: bool | None = None,
                  keys: Sequence[str] | None = None,
                  verbose: bool = True) -> dict[str, Any]:
    """STEP 14.9 end to end.

    The order, and why it is this order:

      1. fingerprint `step14\\`, so the add-only guarantee can be proved;
      2. load and CHECK one composed model per configuration (14.9.1);
      3. validation probabilities -> each variant's validation macro-F1 ->
         the alert variant, decided before any test row is looked at;
      4. the flows to explain: Step 14's own, plus this variant's alerts;
      5. Integrated Gradients per model, with completeness, perturbation,
         deletion, stability and latency (14.9.2);
      6. the cross-model comparison (14.9.3);
      7. every variant's contributions, checked against its own saved R
         (14.9.4);
      8. the alert reports for the chosen variant (14.9.5);
      9. save, read back, verify, and re-fingerprint `step14\\`.
    """
    overwrite = (config.OVERWRITE_STEP14ALL if overwrite is None else overwrite)
    if not overwrite and step14all_outputs_exist():
        return load_step14all(verbose=verbose)

    checks = Checks()
    step14_before = step14_fingerprint()
    offered = model_keys(keys, verbose=verbose)

    # ---- inputs, exactly as Step 14 reads them --------------------------
    inputs = xai.load_step14_inputs(verbose=False)
    y_test, y_train = inputs["y_test"], inputs["y_train"]
    meta_test = inputs["meta_test"]
    label_names, benign = inputs["label_names"], inputs["benign_codes"]
    class_order, n_classes = inputs["class_order"], inputs["n_classes"]
    tau = float(inputs["tau"])
    step14 = load_step14_outputs()
    step13all = None
    if fa.step13all_outputs_exist():
        step13all = fa.load_step13all(verbose=False)
    elif verbose:
        show_note("13.8 has not been run on this machine, so only **F3** is "
                  "explained. Run the 13.8 cells to add FALL and FSEL.")

    # ---- 14.9.1  the composed models ------------------------------------
    if verbose:
        show_title("14.9.1 -- one composed model per configuration", level=4)
    bundles, availability, composed_checks = available_models(offered,
                                                              verbose=verbose)
    explained_keys = ordered(list(bundles))
    for _i, row in composed_checks.iterrows():
        checks.add("14.9.1 composed models",
                   "%s reproduces its saved probabilities" % row["model"],
                   None if row["result"] == "[SKIP]" else row["result"] == "[OK]",
                   row.get("detail", ""))

    test_probs = {key: xai.composed_probabilities(bundle["composed"],
                                                  bundle["X_test"])
                  for key, bundle in bundles.items()}
    val_probs, y_val, _positions = _validation_probabilities(bundles, y_train)

    # ---- the alert variant, on VALIDATION only --------------------------
    fusion_inputs = {
        "y_test": y_test, "y_val": y_val, "y_train": y_train,
        "meta_test": meta_test, "label_names": label_names,
        "n_classes": n_classes, "benign_codes": benign,
        "duplicate_flags": np.zeros(len(y_test), dtype=bool),
        "test_probs": test_probs,
    }
    variant_keys = {tag: _variant_keys(tag, step13all) for tag in VARIANTS}
    variant_scores = {}
    for tag in VARIANTS:
        wanted = variant_keys[tag]
        if not wanted or any(k not in val_probs for k in wanted):
            variant_scores[tag] = float("nan")
            continue
        variant_scores[tag] = validation_macro_f1(tag, wanted, val_probs,
                                                  fusion_inputs, step13all)
    variant, reason = choose_alert_variant(variant_scores)
    checks.add("14.9.5 alert reports", "the alert variant is a validation "
                                       "decision", True, reason)

    # ---- the flows to explain -------------------------------------------
    step14_rows = step14_explained_rows(step14, meta_test, len(y_test))
    frames = {tag: variant_frame(tag, inputs["step13"], step13all)
              for tag in VARIANTS}
    variants: dict[str, dict[str, Any]] = {}
    for tag in VARIANTS:
        wanted = variant_keys[tag]
        frame = frames[tag]
        if not wanted or frame is None or any(k not in test_probs
                                              for k in wanted):
            continue
        try:
            matrices = variant_matrices(tag, wanted, frame, class_order,
                                        test_probs, val_probs,
                                        fusion_inputs | {"y_train": y_train})
        except KeyError:                          # a column the file lacks
            continue
        variants[tag] = {"keys": wanted, "matrices": matrices,
                         "tau": tau if tag == "F3" else float(
                             ((step13all or {}).get("metrics") or {})
                             .get("variants", {}).get(tag, {})
                             .get("threshold", {}).get("tau", tau))}

    if variant not in variants:
        raise XaiAllError(
            "the chosen alert variant %r has no per-flow file to explain; run "
            "the 13.8 cells first." % variant)

    chosen = variants[variant]
    tau_s = (step14.get("thresholds") or {}).get("tau_s")
    bands = xai.decision_band(chosen["matrices"]["risk"], chosen["tau"],
                              tau_s if config.SUSPICIOUS_BAND else None)
    extra = xai.select_flows(bands, chosen["matrices"]["fused_class"])
    rows = np.array(sorted(set(step14_rows.tolist()) | set(extra.tolist())),
                    dtype="int64")
    orig_row = (meta_test["Orig_Row"].to_numpy()[rows]
                if meta_test is not None and "Orig_Row" in meta_test.columns
                else rows)
    inputs["orig_row"] = orig_row
    inputs["meta_test"] = meta_test
    checks.add("14.9.2 attribution", "the flows Step 14 explained are explained "
                                     "here too",
               bool(set(step14_rows.tolist()) <= set(rows.tolist())),
               "%s from Step 14, %s added from %s alerts, %s explained in all"
               % ("{:,}".format(len(step14_rows)),
                  "{:,}".format(len(set(extra.tolist()) - set(step14_rows.tolist()))),
                  variant, "{:,}".format(len(rows))))

    # ---- 14.9.2  Integrated Gradients, per model -------------------------
    if verbose:
        show_title("14.9.2 -- Integrated Gradients for every model", level=4)
        print("  %s flows x %d models, IG_STEPS = %d"
              % ("{:,}".format(len(rows)), len(bundles), config.IG_STEPS))
    explained: dict[str, dict[str, Any]] = {}
    for key in explained_keys:
        explained[key] = explain_model(key, bundles[key], rows, orig_row,
                                       test_probs[key], inputs | {
                                           "y_train": y_train,
                                           "label_names": label_names,
                                           "benign_codes": benign},
                                       verbose=verbose)
        result = explained[key]
        checks.add("14.9.2 attribution",
                   "%s: sum IG = F(x) - F(x') within %.0f%%"
                   % (display_label(key), 100 * config.COMPLETENESS_TOL),
                   result["completeness_share"] >= 0.5,
                   "%.1f%% of explained flows" % (100 * result["completeness_share"]))
        checks.add("14.9.2 attribution",
                   "%s: removing the top-%d costs more than chance"
                   % (display_label(key), result["deletion"]["k"]),
                   bool(result["deletion"]["top_k_drops_more"]),
                   "top-K %.4f vs random %.4f"
                   % (result["deletion"]["mean_drop_top_k"],
                      result["deletion"]["mean_drop_random_k"]))

    attention = pd.DataFrame()
    approach2 = getattr(config, "APPROACH2_KEY", "approach2_attention")
    if approach2 in explained:
        attention = attention_supplement(bundles[approach2], rows,
                                         explained[approach2]["predicted"],
                                         {"label_names": label_names},
                                         verbose=verbose)

    # ---- 14.9.3  the cross-model comparison ------------------------------
    if verbose:
        show_title("14.9.3 -- comparing the explanations", level=4)
    matrix = importance_matrix(explained)
    spearman_grid, overlap_grid = explanation_agreement(matrix)
    shared = shared_feature_counts(matrix)
    consensus = consensus_features(matrix)
    quality = quality_table(explained)
    checks.add("14.9.3 cross-model",
               "a feature outside a model's inputs is NA, not zero",
               bool(matrix.isna().to_numpy().any()
                    or len({tuple(sorted(explained[k]["features"]))
                            for k in explained}) == 1),
               "%d of %d cells are NA"
               % (int(matrix.isna().to_numpy().sum()), int(matrix.size)))

    global_importance = pd.concat(
        [explained[k]["importance"] for k in explained_keys], ignore_index=True)
    explanations = xai.explanations_frame(
        rows, orig_row,
        {display_label(k): explained[k]["attributions"] for k in explained_keys},
        {display_label(k): explained[k]["features"] for k in explained_keys},
        drops={display_label(k): explained[k]["drops"] for k in explained_keys})
    ig_baselines = pd.DataFrame([{
        "model": display_label(k),
        "input features": len(explained[k]["features"]),
        "baseline": config.IG_BASELINE,
        "attack risk at the baseline F(x')": round(
            float(explained[k]["baseline_risk"]), 6),
        "IG steps": config.IG_STEPS} for k in explained_keys])

    # ---- 14.9.4  every variant's contributions ---------------------------
    if verbose:
        show_title("14.9.4 -- how each fusion variant splits its risk", level=4)
    contributions, variant_rows = [], []
    for tag, block in variants.items():
        breakdown, consistency = variant_contributions(
            tag, block["keys"], block["matrices"], rows, test_probs, inputs)
        block["breakdown"] = breakdown
        block["consistency"] = consistency
        contributions.append(breakdown)
        checks.add("14.9.4 fusion variants",
                   "%s: R equals the sum of its contributions" % tag,
                   bool(consistency["ok"]),
                   "max |R - sum C_m| = %.2e over %s explained flows"
                   % (consistency["max_abs_difference"],
                      "{:,}".format(len(rows))))
        gap = block["matrices"].get("weights_recovered_gap")
        if gap is not None and tag != "F3":
            checks.add("14.9.4 fusion variants",
                       "%s: Rel_m and Conf_m reproduce the saved weights" % tag,
                       None if np.isnan(gap) else gap <= WEIGHT_RECOVERY_TOL,
                       "could not be recomputed, so the reports show no "
                       "reliability for this variant" if np.isnan(gap)
                       else ("max |w recomputed - w saved| = %.2e against a "
                             "%.0e tolerance -- 13.8 weighted the classifiers' "
                             "own probabilities and these go through the "
                             "composed models, which agree to 1e-5"
                             % (gap, WEIGHT_RECOVERY_TOL)))
        variant_rows.append({
            "variant": tag, "models fused": len(block["keys"]),
            "models": ", ".join(short_label(k) for k in block["keys"]),
            "tau": round(float(block["tau"]), 6),
            "validation macro-F1": (None if np.isnan(variant_scores.get(tag, np.nan))
                                    else round(float(variant_scores[tag]), 6)),
            "alerts among the explained flows": int(
                (np.asarray(block["matrices"]["risk"])[rows]
                 >= float(block["tau"])).sum()),
            "max |R - sum C_m|": float(consistency["max_abs_difference"]),
            "explains the alert reports": tag == variant,
        })
    contributions_frame = (pd.concat(contributions, ignore_index=True)
                           if contributions else pd.DataFrame())
    variant_table = pd.DataFrame(variant_rows)

    dominant = []
    for tag, block in variants.items():
        part = dominant_model_share(block["breakdown"], inputs,
                                    block["matrices"]["fused_class"], rows)
        part.insert(0, "variant", tag)
        dominant.append(part)
    dominant_share = (pd.concat(dominant, ignore_index=True) if dominant
                      else pd.DataFrame())
    disagreement, disagreement_examples = variant_disagreements(variants, rows,
                                                                inputs)

    # ---- 14.9.5  the alert reports ---------------------------------------
    if verbose:
        show_title("14.9.5 -- the per-alert evidence reports", level=4)
    risk_completeness = {}
    for position in range(len(rows)):
        risk_completeness[position] = all(
            bool(explained[k]["completeness"].loc[
                (explained[k]["completeness"]["target"] == TARGET_RISK),
                "within_tolerance"].to_numpy()[position])
            for k in explained_keys)
    reports, alerts = build_reports(
        variant, chosen["keys"], rows, chosen["breakdown"], chosen["matrices"],
        explained, test_probs, inputs | {"y_train": y_train,
                                         "label_names": label_names,
                                         "X_test_unscaled":
                                             inputs["X_test_unscaled"]},
        bands, chosen["tau"], tau_s, risk_completeness)
    examples = examples_per_class(reports)
    counts = (alerts["fused_label"].value_counts().to_dict()
              if len(alerts) else {})
    rare = [(name, int(count)) for name, count in sorted(counts.items())
            if count <= 5]

    out: dict[str, Any] = {
        "offered_keys": offered, "fused_keys_all": explained_keys,
        "availability": availability, "composed_checks": composed_checks,
        "explained": explained, "rows": rows, "orig_row": orig_row,
        "bands": bands, "tau": chosen["tau"], "tau_s": tau_s,
        "importance_matrix": matrix, "agreement_spearman": spearman_grid,
        "agreement_overlap": overlap_grid, "shared_features": shared,
        "consensus": consensus, "quality": quality,
        "global_importance": global_importance, "explanations": explanations,
        "ig_baselines": ig_baselines, "attention": attention,
        "variants": variants, "variant_scores": variant_scores,
        "variant": variant, "variant_reason": reason,
        "variant_table": variant_table, "contributions": contributions_frame,
        "dominant_share": dominant_share,
        "variant_disagreement": disagreement,
        "variant_disagreement_examples": disagreement_examples,
        "reports": reports, "alerts": alerts, "examples": examples,
        "fused_keys": chosen["keys"], "rare_classes": rare,
        "test_probs": test_probs, "val_probs": val_probs,
        "label_names": label_names, "y_test": y_test,
        "step14": step14, "step14_before": step14_before,
        "step13all": step13all, "checks": checks,
    }

    saved = save_step14all(out, checks, verbose=verbose)
    out.update({"status": saved.get("status", "written"), "saved": saved,
                "ok": bool(saved.get("ok"))})
    for key in ("verification", "metrics", "summary", "charts", "folder"):
        if key in saved:
            out[key] = saved[key]

    after_table, after_ok = assert_step14_unchanged(step14_before, verbose=False)
    out["step14_unchanged_table"] = after_table
    out["step14_unchanged"] = after_ok
    if not after_ok:
        raise XaiAllError(
            "writing the Step 14.9 outputs changed a file in the Step 14 "
            "folder. That must never happen; the addendum writes only to %s."
            % config.STEP14ALL_OUTPUT_DIR)
    if verbose:
        print("[%s]  existing Step 14 outputs unchanged"
              % ("OK" if after_ok else "FAIL"))
    return out
