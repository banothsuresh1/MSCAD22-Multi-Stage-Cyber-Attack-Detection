"""
src/mscad/fusion.py -- STEP 13: Reliability-Aware Adaptive Evidence Fusion.

NEW FILE.  Steps 1-12 are finished: nothing in converter.py, data_loading.py,
preprocessing.py, splitting.py, feature_selection.py, autoencoder.py,
imbalance.py, models.py, evaluation.py or experiments.py is modified.  This
module calls those functions and adds its own helpers beside them.

What this does, and how it differs from an ordinary ensemble
------------------------------------------------------------
Three heterogeneous models vote on every flow.  A simple average gives each the
same say everywhere; this gives each a say that varies PER FLOW, from two
measured quantities:

    Rel_m(k)   how reliable model m has been in the behavioural CONTEXT k that
               this flow falls in, measured on the validation part
    Conf_m(x)  how confident model m is on THIS flow, as 1 - normalised entropy

A model that is historically weak on port scans and unsure about this particular
flow is listened to less; a model that is strong on that context and sure of
itself is listened to more.  The weights are recomputed for every flow.

No leakage
----------
Every quantity that is LEARNED -- the per-context reliabilities, the weight
settings and the decision threshold -- comes from the validation part of the
Step 9 split.  The test set is used once, at the end, with those values held
fixed.  Nothing here retrains a model.

One honest caveat, repeated in the notebook
-------------------------------------------
That validation part was also used for early stopping in Step 10, so the
reliability estimates are slightly optimistic.  A fully clean design would hold
out a third split for fusion alone; with 23-row classes in this dataset that
would leave too few rows per class to estimate a per-context F1 at all.
"""

from __future__ import annotations

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

from . import data_loading as dl  # noqa: E402
from . import evaluation as ev  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = "Label_encoded"


class FusionError(Exception):
    """Raised when Step 13 cannot continue without guessing."""


# ===========================================================================
# 13.1  THE PIECES -- pure functions, no TensorFlow, no file I/O
# ===========================================================================
def attack_risk(probs: np.ndarray, benign_codes: Sequence[int]) -> np.ndarray:
    """r(x) = 1 - sum of the benign classes' probability.

    Subtracting the benign mass rather than summing the attack classes keeps the
    result exact when the row sums to 1, and cannot silently omit a class that is
    neither benign nor one of the attack classes being summed.
    """
    probs = np.asarray(probs, dtype="float64")
    benign = [int(c) for c in benign_codes if 0 <= int(c) < probs.shape[1]]
    if not benign:
        return np.ones(len(probs), dtype="float64")
    return 1.0 - probs[:, benign].sum(axis=1)


def normalised_entropy_confidence(probs: np.ndarray,
                                  enabled: bool | None = None) -> np.ndarray:
    """Conf(x) = 1 - H(p) / ln(C), with 0 ln 0 treated as 0.

    1.0 for a one-hot distribution and 0.0 for a uniform one, so it is on the
    same [0, 1] scale as the reliability it multiplies.
    """
    enabled = config.CONFIDENCE_TERM if enabled is None else enabled
    probs = np.asarray(probs, dtype="float64")
    if not enabled:
        return np.ones(len(probs), dtype="float64")
    n_classes = probs.shape[1]
    if n_classes < 2:
        return np.ones(len(probs), dtype="float64")
    safe = np.where(probs > 0, probs, 1.0)          # 0 ln 0 := 0
    entropy = -np.sum(np.where(probs > 0, probs * np.log(safe), 0.0), axis=1)
    return 1.0 - entropy / np.log(n_classes)


def consensus_context(probs_list: Sequence[np.ndarray]) -> np.ndarray:
    """k(x) = argmax_c mean_m p_m(c|x).  Label-free, so usable at test time."""
    stacked = np.stack([np.asarray(p, dtype="float64") for p in probs_list])
    return np.argmax(stacked.mean(axis=0), axis=1).astype("int64")


def shrunk_reliability(f1_context: float, f1_macro: float, n_context: int,
                       k: int | None = None) -> float:
    """Rel = (n_k F1(k) + K F1_macro) / (n_k + K).

    A context seen twice gives an F1 of 0 or 1 and nothing in between, so the
    per-context estimate is pulled towards the model's global macro-F1.  K is the
    number of rows at which the two contribute equally; n_k = 0 returns the
    global figure exactly.
    """
    k = config.SHRINKAGE_K if k is None else k
    n_context = int(n_context)
    if n_context <= 0:
        return float(f1_macro)
    if n_context + k == 0:
        return float(f1_context)
    return float((n_context * f1_context + k * f1_macro) / (n_context + k))


def dynamic_weights(reliability: np.ndarray, confidence: np.ndarray,
                    mode: str | None = None, temperature: float | None = None,
                    epsilon: float | None = None) -> np.ndarray:
    """w(x) from s_m(x) = Rel_m(k(x)) * Conf_m(x).  Rows are non-negative, sum 1.

    `reliability` and `confidence` are both (n_flows, n_models).
    """
    mode = config.WEIGHT_MODE if mode is None else mode
    temperature = config.WEIGHT_TEMPERATURE if temperature is None else temperature
    epsilon = config.EPSILON if epsilon is None else epsilon

    reliability = np.asarray(reliability, dtype="float64")
    confidence = np.asarray(confidence, dtype="float64")
    if reliability.shape != confidence.shape:
        raise FusionError(
            "reliability %s and confidence %s must have the same shape"
            % (reliability.shape, confidence.shape))
    scores = np.clip(reliability * confidence, 0.0, None)
    n_models = scores.shape[1]

    if mode == "proportional":
        totals = scores.sum(axis=1, keepdims=True)
        weights = np.divide(scores, totals, out=np.full_like(scores, 1.0 / n_models),
                            where=totals >= epsilon)
    elif mode == "softmax":
        if temperature <= 0:
            raise FusionError("WEIGHT_TEMPERATURE must be positive; got %r"
                              % temperature)
        shifted = scores / temperature
        shifted = shifted - shifted.max(axis=1, keepdims=True)   # stable
        exponent = np.exp(shifted)
        weights = exponent / exponent.sum(axis=1, keepdims=True)
        # a flow whose scores are all ~0 carries no evidence either way
        flat = scores.sum(axis=1) < epsilon
        weights[flat] = 1.0 / n_models
    else:
        raise FusionError(
            "WEIGHT_MODE is %r; it must be \"proportional\" or \"softmax\"" % mode)
    return weights


def fuse_probabilities(probs_list: Sequence[np.ndarray],
                       weights: np.ndarray) -> np.ndarray:
    """P(c|x) = sum_m w_m(x) p_m(c|x).  Rows sum to 1 when the inputs do."""
    stacked = np.stack([np.asarray(p, dtype="float64") for p in probs_list])
    weights = np.asarray(weights, dtype="float64")
    if weights.shape != (stacked.shape[1], stacked.shape[0]):
        raise FusionError(
            "weights must be (n_flows, n_models) = (%d, %d); got %s"
            % (stacked.shape[1], stacked.shape[0], weights.shape))
    return np.einsum("mnc,nm->nc", stacked, weights)


# --- the fixed-weight baselines this is compared against -------------------
def simple_average(probs_list: Sequence[np.ndarray]) -> np.ndarray:
    """Every model weighted equally, on every flow."""
    stacked = np.stack([np.asarray(p, dtype="float64") for p in probs_list])
    return stacked.mean(axis=0)


def majority_vote(probs_list: Sequence[np.ndarray]) -> np.ndarray:
    """The most-voted class per flow; ties broken by the highest mean probability.

    Returns predicted class indices.  A vote discards the models' confidence
    entirely, which is exactly what makes it a useful contrast.
    """
    stacked = np.stack([np.asarray(p, dtype="float64") for p in probs_list])
    votes = np.argmax(stacked, axis=2)                     # (n_models, n_flows)
    n_classes = stacked.shape[2]
    counts = np.zeros((stacked.shape[1], n_classes), dtype="int64")
    for m in range(stacked.shape[0]):
        counts[np.arange(stacked.shape[1]), votes[m]] += 1
    mean_probs = stacked.mean(axis=0)
    best = counts.max(axis=1, keepdims=True)
    tied = counts == best
    # among the tied classes take the one with the highest mean probability
    return np.argmax(np.where(tied, mean_probs, -np.inf), axis=1).astype("int64")


def static_weights(macro_f1: Sequence[float]) -> np.ndarray:
    """One weight vector for every flow, from each model's validation macro-F1."""
    values = np.clip(np.asarray(macro_f1, dtype="float64"), 0.0, None)
    total = values.sum()
    if total < config.EPSILON:
        return np.full(len(values), 1.0 / len(values))
    return values / total


# --- the decision threshold ------------------------------------------------
def choose_threshold(risk: np.ndarray, y_binary: np.ndarray,
                     rule: str | None = None,
                     target_fpr: float | None = None) -> dict[str, Any]:
    """Pick tau on the VALIDATION risk scores.  Never looks at the test set.

    "max_f1"      the threshold maximising binary F1
    "fpr_at_most" the LOWEST threshold whose false-positive rate is within
                  TARGET_FPR -- lowest, because a higher one would discard recall
                  for no gain in the constraint
    """
    rule = config.THRESHOLD_RULE if rule is None else rule
    target_fpr = config.TARGET_FPR if target_fpr is None else target_fpr
    risk = np.asarray(risk, dtype="float64")
    truth = np.asarray(y_binary).astype("int64")

    if len(risk) == 0:
        raise FusionError("cannot choose a threshold from an empty validation set")

    candidates = np.unique(np.concatenate([[0.0], risk, [1.0]]))
    rows = []
    positives = int((truth == 1).sum())
    negatives = int((truth == 0).sum())
    for tau in candidates:
        predicted = (risk >= tau).astype("int64")
        tp = int(((predicted == 1) & (truth == 1)).sum())
        fp = int(((predicted == 1) & (truth == 0)).sum())
        fn = positives - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        fpr = fp / negatives if negatives else 0.0
        rows.append({"tau": float(tau), "precision": precision, "recall": recall,
                     "f1": f1, "fpr": fpr})
    table = pd.DataFrame(rows)

    if rule == "max_f1":
        index = int(table["f1"].idxmax())
    elif rule == "fpr_at_most":
        allowed = table[table["fpr"] <= target_fpr]
        if allowed.empty:
            raise FusionError(
                "no threshold reaches a false-positive rate of %.4f or less on the "
                "validation set (the lowest achievable is %.4f). Raise TARGET_FPR "
                "or use THRESHOLD_RULE = \"max_f1\"."
                % (target_fpr, float(table["fpr"].min())))
        index = int(allowed["tau"].idxmin())
    else:
        raise FusionError(
            "THRESHOLD_RULE is %r; it must be \"max_f1\" or \"fpr_at_most\"" % rule)

    chosen = table.loc[index]
    return {"tau": float(chosen["tau"]), "rule": rule,
            "target_fpr": target_fpr if rule == "fpr_at_most" else None,
            "validation_precision": float(chosen["precision"]),
            "validation_recall": float(chosen["recall"]),
            "validation_f1": float(chosen["f1"]),
            "validation_fpr": float(chosen["fpr"]),
            "candidates": len(candidates),
            "chosen_on": "the validation part of the Step 9 split"}


def ema_within_capture(risk: np.ndarray, capture: Sequence[Any],
                       order: Sequence[Any] | None = None,
                       alpha: float | None = None) -> np.ndarray:
    """Exponential moving average of the risk, never crossing a capture boundary.

    Each capture is smoothed independently and the result is written back to the
    original row positions, so the returned array still lines up with the input.
    """
    alpha = config.EMA_ALPHA if alpha is None else alpha
    if not 0 < alpha <= 1:
        raise FusionError("EMA_ALPHA must be in (0, 1]; got %r" % alpha)
    risk = np.asarray(risk, dtype="float64")
    capture = np.asarray(capture, dtype=object)
    if len(capture) != len(risk):
        raise FusionError("risk and capture must have the same length")

    out = risk.copy()
    frame = pd.DataFrame({"row": np.arange(len(risk)), "capture": capture})
    frame["order"] = (np.arange(len(risk)) if order is None
                      else pd.Series(list(order)).to_numpy())
    for _name, part in frame.groupby("capture", sort=False):
        rows = part.sort_values(["order", "row"], kind="stable")["row"].to_numpy()
        smoothed = risk[rows[0]]
        for position in rows:
            smoothed = alpha * risk[position] + (1 - alpha) * smoothed
            out[position] = smoothed
    return out


# ===========================================================================
# 13.2  CONTEXT RELIABILITY -- measured on the validation part only
# ===========================================================================
def per_class_f1(y_true, y_pred, n_classes: int,
                 metric: str | None = None) -> np.ndarray:
    """F1 (or precision) per class, in class order."""
    from sklearn.metrics import precision_recall_fscore_support

    metric = config.RELIABILITY_METRIC if metric is None else metric
    labels = list(range(n_classes))
    precision, _recall, f1, _support = precision_recall_fscore_support(
        np.asarray(y_true), np.asarray(y_pred), labels=labels, average=None,
        zero_division=0)
    if metric == "precision":
        return np.asarray(precision, dtype="float64")
    if metric == "f1":
        return np.asarray(f1, dtype="float64")
    raise FusionError(
        "RELIABILITY_METRIC is %r; it must be \"f1\" or \"precision\"" % metric)


def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(y_true), np.asarray(y_pred),
                          average="macro", zero_division=0))


def build_reliability(val_probs: dict[str, np.ndarray], y_val_true: np.ndarray,
                      contexts: np.ndarray, context_names: dict[Any, str],
                      n_classes: int, context_mode: str | None = None,
                      shrinkage: int | None = None,
                      verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Rel_m(k) for every model and context, from the VALIDATION rows only.

    In "consensus_class" mode the context is a class, so F1_m(k) is that model's
    F1 for that class.  In "protocol" mode the context is a protocol, so F1_m(k)
    is the model's macro-F1 over the validation rows of that protocol.
    """
    context_mode = config.CONTEXT_MODE if context_mode is None else context_mode
    y_val_true = np.asarray(y_val_true)
    contexts = np.asarray(contexts)

    rows = []
    lookup: dict[tuple[str, Any], float] = {}
    globals_: dict[str, float] = {}
    for name, probs in val_probs.items():
        predicted = np.argmax(np.asarray(probs), axis=1)
        model_macro = macro_f1(y_val_true, predicted)
        globals_[name] = model_macro
        class_f1 = per_class_f1(y_val_true, predicted, n_classes)

        for key in sorted(set(contexts.tolist())):
            selected = contexts == key
            n_context = int(selected.sum())
            if context_mode == "consensus_class":
                # the context IS a class, so use that class's F1
                raw = float(class_f1[int(key)]) if int(key) < n_classes else 0.0
            else:
                raw = (macro_f1(y_val_true[selected], predicted[selected])
                       if n_context else model_macro)
            reliability = shrunk_reliability(raw, model_macro, n_context, shrinkage)
            rows.append({
                "model": name, "context": context_names.get(key, str(key)),
                "context_key": key, "n_validation_rows": n_context,
                "%s_in_context" % config.RELIABILITY_METRIC: round(raw, 6),
                "macro_f1_global": round(model_macro, 6),
                "reliability": round(reliability, 6)})
            lookup[(name, key)] = reliability

    table = pd.DataFrame(rows)
    if verbose:
        show_table(table, note="Measured on the validation part of the Step 9 split "
                               "only. `reliability` is the context figure shrunk "
                               "towards the model's global macro-F1 with "
                               "K = %d, so a context with few rows cannot swing a "
                               "weight on the strength of two examples."
                               % (config.SHRINKAGE_K if shrinkage is None
                                  else shrinkage))
    return table, {"lookup": lookup, "macro": globals_}


def validation_positions(y_train: pd.DataFrame) -> np.ndarray:
    """The row positions of the Step 9 validation part inside the training set.

    The split is a pure function of (y_train, CLF_VAL_SIZE, RANDOM_STATE), so
    asking Step 9 for it again selects exactly the rows Steps 10-12 validated on.
    """
    from . import imbalance as im

    marker = pd.DataFrame({"row": np.arange(len(y_train))})
    _fit, _y_fit, val_rows, _y_val, _w, _s = im.make_training_sets(
        marker, y_train, verbose=False)
    return val_rows["row"].to_numpy()


def protocol_contexts(y_train: pd.DataFrame, meta_test: pd.DataFrame | None):
    """(validation protocols, test protocols), or (None, None) if unavailable.

    The protocol is metadata, never a model feature, so reading it here does not
    put an identifier into the fusion -- it only says which behavioural context a
    flow belongs to.
    """
    folder5 = config.usable_output_dir(config.STEP5_OUTPUT_DIR)
    if folder5 is None or meta_test is None or "Protocol" not in meta_test.columns:
        return None, None
    path = folder5 / config.STEP5_FILES["meta_train"]
    if not path.exists():
        return None, None
    meta_train = pd.read_csv(path, low_memory=False)
    if "Protocol" not in meta_train.columns:
        return None, None
    positions = validation_positions(y_train)
    protocols = meta_train["Protocol"].to_numpy()
    if positions.max(initial=-1) >= len(protocols):
        return None, None
    return (np.asarray(protocols[positions], dtype=object),
            np.asarray(meta_test["Protocol"].to_numpy(), dtype=object))


def context_sets(mode: str, val_list: Sequence[np.ndarray],
                 test_list: Sequence[np.ndarray], y_train: pd.DataFrame,
                 meta_test: pd.DataFrame | None, label_names: dict[int, str],
                 n_classes: int):
    """(validation contexts, test contexts, context names) for one CONTEXT_MODE.

    Returns (None, None, None) when the mode cannot be built from what is on
    disk, which is how the protocol ablation is skipped rather than guessed.
    """
    if mode == "consensus_class":
        return (consensus_context(val_list), consensus_context(test_list),
                {c: label_names.get(c, str(c)) for c in range(n_classes)})
    if mode == "protocol":
        validation, test = protocol_contexts(y_train, meta_test)
        if validation is None:
            return None, None, None
        keys = set(validation.tolist()) | set(test.tolist())
        return validation, test, {k: str(k) for k in keys}
    raise FusionError(
        "CONTEXT_MODE is %r; it must be \"consensus_class\" or \"protocol\"" % mode)


def reliability_matrix(lookup: dict, macro: dict, model_names: Sequence[str],
                       contexts: np.ndarray, use_context: bool = True) -> np.ndarray:
    """(n_flows, n_models) of Rel_m(k(x)).

    `use_context=False` is the "no context" ablation: every flow gets the model's
    global macro-F1 instead.
    """
    contexts = np.asarray(contexts)
    out = np.empty((len(contexts), len(model_names)), dtype="float64")
    for j, name in enumerate(model_names):
        if not use_context:
            out[:, j] = macro[name]
            continue
        out[:, j] = [lookup.get((name, key), macro[name]) for key in contexts]
    return out


# ===========================================================================
# 13.3  ALIGNMENT -- every model must describe the same rows, in the same order
# ===========================================================================
def check_alignment(frames: dict[str, pd.DataFrame], class_order: Sequence[str],
                    verbose: bool = True):
    """[OK]/[FAIL] that the per-model frames line up on Orig_Row and class order.

    Fusing frames that disagree on row order produces numbers that look fine and
    are meaningless, so this raises rather than warning.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    names = list(frames)
    first = names[0]
    reference = frames[first]

    missing_col = [n for n, f in frames.items() if "Orig_Row" not in f.columns]
    add("every model frame carries Orig_Row", not missing_col,
        "all %d frames" % len(frames) if not missing_col
        else "missing in %s" % missing_col)
    if missing_col:
        raise FusionError(
            "%s has no Orig_Row column, so the models cannot be aligned row by row"
            % missing_col)

    same_rows, detail = True, ""
    for name in names[1:]:
        if frames[name]["Orig_Row"].tolist() != reference["Orig_Row"].tolist():
            same_rows = False
            detail = "%s and %s differ" % (first, name)
            break
    add("all models cover the same rows in the same order", same_rows,
        "%s rows" % f"{len(reference):,}" if same_rows else detail)

    class_columns = {n: [c for c in f.columns if c.startswith("p_")]
                     for n, f in frames.items()}
    expected = ["p_%s" % c for c in class_order]
    same_classes = all(cols == expected for cols in class_columns.values())
    add("all models use the same class order", same_classes,
        ", ".join(class_order) if same_classes
        else "expected %s" % expected)

    truth = {n: f["y_true"].tolist() for n, f in frames.items() if "y_true" in f}
    same_truth = len({tuple(v) for v in truth.values()}) <= 1
    add("the ground truth agrees across models", same_truth,
        "%s rows" % f"{len(reference):,}" if same_truth
        else "the y_true columns differ")

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table)

    if not same_rows:
        raise FusionError(
            "the model prediction frames do not cover the same rows in the same "
            "order (%s). Fusing them would combine different flows." % detail)
    if not same_classes:
        raise FusionError(
            "the model frames disagree about the class order; expected %s. "
            "Fusing them would add probabilities of different classes together."
            % expected)
    if not same_truth:
        raise FusionError(
            "the model frames disagree about the ground truth, so at least one is "
            "not describing the Step 5 test set")
    return table, all_ok


def check_fusion_models(verbose: bool = True) -> list[str]:
    """The FUSION_MODELS entries must exist in Step 12's EXPERIMENTS."""
    known = [e["name"] for e in config.EXPERIMENTS]
    unknown = [n for n in config.FUSION_MODELS if n not in known]
    if unknown:
        raise FusionError(
            "FUSION_MODELS names %s, which are not in config.EXPERIMENTS (%s). "
            "Use the exact Step 12 names." % (unknown, known))
    if verbose:
        show_note("Fusing %d models: %s. All are Step 12 configurations."
                  % (len(config.FUSION_MODELS), ", ".join(config.FUSION_MODELS)))
    return list(config.FUSION_MODELS)


# ===========================================================================
# 13.4  LOADING THE SAVED MODELS -- nothing is retrained
# ===========================================================================
def _spec_for(name: str) -> dict[str, Any]:
    for spec in config.EXPERIMENTS:
        if spec["name"] == name:
            return spec
    raise FusionError("no Step 12 configuration named %r" % name)


def _missing(path: Path, what: str, step: str) -> None:
    raise FusionError(
        "%s is missing:\n    %s\nRun %s first. Step 13 never retrains a model."
        % (what, path, step))


def load_model_bundle(name: str, verbose: bool = False) -> dict[str, Any]:
    """The saved classifier and feature pipeline for one Step 12 configuration.

    Reads only; if anything is absent it says which file and which step produces
    it, rather than retraining or guessing.
    """
    from . import experiments as xp
    from . import models as md
    from . import autoencoder as ae

    spec = _spec_for(name)
    folder12 = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder12 is None:
        raise FusionError("the Step 12 output folder is not usable here -- %s"
                          % config.why_unusable(config.STEP12_OUTPUT_DIR))
    folder = folder12 / name

    model_path = folder / "model.keras"
    if not model_path.exists():
        if spec.get("reuse_step7_10") and md.step10_outputs_exist():
            folder10 = config.usable_output_dir(config.STEP10_OUTPUT_DIR)
            model_path = folder10 / config.STEP10_FILES["model"]
        else:
            _missing(model_path, "the trained model for %r" % name,
                     "Step 12 (set OVERWRITE_STEP12 = True)")

    keras = md._keras()
    model = keras.models.load_model(model_path)

    source = xp.load_feature_source(spec["feature_source"], verbose=False)
    X_train, X_test = source["X_train"], source["X_test"]

    encode = None
    if spec["use_ae"]:
        if spec.get("reuse_step7_10"):
            folder7 = config.usable_output_dir(config.STEP7_OUTPUT_DIR)
            encoder_path = folder7 / config.STEP7_FILES["encoder"] if folder7 else None
            if encoder_path is None or not encoder_path.exists():
                _missing(encoder_path, "the Step 7 encoder", "Step 7")
            encode, _enc, order = ae.load_encoder(verbose=False)
            X_train, X_test = (encode(X_train[order], what="training features"),
                               encode(X_test[order], what="test features"))
        else:
            encoder_path = folder / "encoder.keras"
            if not encoder_path.exists():
                _missing(encoder_path, "the baseline autoencoder for %r" % name,
                         "Step 12 (set OVERWRITE_STEP12 = True)")
            encoder = keras.models.load_model(encoder_path)
            saved = json.loads((folder / "ae_config.json").read_text(encoding="utf-8"))
            order = list(saved["features"])
            encode = ae.make_encoder_function(encoder, order)
            X_train, X_test = (encode(X_train[order], what="training features"),
                               encode(X_test[order], what="test features"))

    if verbose:
        print("  loaded %-24s model=%s  features=%d"
              % (name, model_path.name, X_train.shape[1]))
    return {"name": name, "spec": spec, "model": model,
            "X_train": X_train, "X_test": X_test,
            "model_path": str(model_path)}


def model_probabilities(bundle: dict[str, Any], X: pd.DataFrame,
                        y: pd.DataFrame, meta: pd.DataFrame | None = None):
    """Softmax probabilities for one model over one matrix.  Returns (probs, idx)."""
    from . import models as md
    arranged, _labels, row_index = md.arrange_inputs(X, y, meta, verbose=False)
    return md.predict_proba(bundle["model"], arranged), row_index


def validation_probabilities(bundles: dict[str, dict], y_train: pd.DataFrame,
                             verbose: bool = True):
    """Each model's probabilities on the Step 9 VALIDATION rows.

    The Step 9 split is a pure function of (y_train, CLF_VAL_SIZE, RANDOM_STATE),
    so splitting each model's own training matrix with the same call selects the
    identical rows for every model -- which is what makes the reliabilities
    comparable.
    """
    from . import imbalance as im

    marker = pd.DataFrame({"row": np.arange(len(y_train))})
    _fit, _yf, val_rows, y_val, _w, _s = im.make_training_sets(marker, y_train,
                                                               verbose=False)
    positions = val_rows["row"].to_numpy()

    probs = {}
    for name, bundle in bundles.items():
        X_val = bundle["X_train"].iloc[positions].reset_index(drop=True)
        probs[name], _idx = model_probabilities(bundle, X_val, y_val)
    if verbose:
        show_note("Validation probabilities computed on **%s rows** -- the "
                  "validation part of the Step 9 split, identical for every model."
                  % f"{len(positions):,}")
    return probs, y_val, positions


def test_probabilities(bundles: dict[str, dict], y_test: pd.DataFrame,
                       meta_test: pd.DataFrame | None, verbose: bool = True):
    """Each model's probabilities on the untouched test set, cross-checked.

    Where Step 11/12 saved a predictions.csv the recomputed probabilities are
    compared against it: a mismatch means the saved model is not the one that
    produced those predictions, which would invalidate the comparison.
    """
    folder12 = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    probs, checks = {}, []
    for name, bundle in bundles.items():
        values, _idx = model_probabilities(bundle, bundle["X_test"], y_test,
                                           meta_test)
        probs[name] = values
        saved_path = (folder12 / name / "predictions.csv") if folder12 else None
        if saved_path is not None and saved_path.exists():
            saved = pd.read_csv(saved_path, low_memory=False)
            columns = [c for c in saved.columns if c.startswith("p_")]
            if len(columns) == values.shape[1] and len(saved) == len(values):
                difference = float(np.abs(saved[columns].to_numpy() - values).max())
                checks.append({"model": name, "source": saved_path.name,
                               "max abs difference": difference,
                               "result": "[OK]" if difference < 1e-6 else "[FAIL]"})
            else:
                checks.append({"model": name, "source": saved_path.name,
                               "max abs difference": float("nan"),
                               "result": "[FAIL]"})
        else:
            checks.append({"model": name, "source": "(not saved)",
                           "max abs difference": float("nan"), "result": "[SKIP]"})
    table = pd.DataFrame(checks)
    if verbose:
        show_table(table, note="Recomputed probabilities against the Step 12 "
                               "predictions.csv. A `[FAIL]` means the saved model "
                               "is not the one that produced those predictions.")
    failed = table[table["result"] == "[FAIL]"]
    if len(failed):
        raise FusionError(
            "the recomputed probabilities do not match the saved predictions for "
            "%s (tolerance 1e-6). Re-run Step 12 with OVERWRITE_STEP12 = True."
            % ", ".join(failed["model"]))
    return probs, table


# ===========================================================================
# 13.5  SCORING ONE SET OF PROBABILITIES
# ===========================================================================
def binary_metrics(risk: np.ndarray, y_binary: np.ndarray,
                   tau: float) -> dict[str, Any]:
    """Binary scores from the risk and the threshold, plus ROC-AUC and PR-AUC."""
    from sklearn.metrics import (average_precision_score,
                                 precision_recall_fscore_support, roc_auc_score)

    risk = np.asarray(risk, dtype="float64")
    truth = np.asarray(y_binary).astype("int64")
    predicted = (risk >= tau).astype("int64")

    precision, recall, f1, _s = precision_recall_fscore_support(
        truth, predicted, average="binary", pos_label=1, zero_division=0)
    negatives = int((truth == 0).sum())
    fp = int(((predicted == 1) & (truth == 0)).sum())
    two_classes = len(np.unique(truth)) == 2
    return {
        "threshold": float(tau),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fpr": float(fp / negatives) if negatives else 0.0,
        "roc_auc": float(roc_auc_score(truth, risk)) if two_classes else None,
        "pr_auc": float(average_precision_score(truth, risk)) if two_classes else None,
    }


def score_probabilities(name: str, probs: np.ndarray, y_test: pd.DataFrame,
                        benign_codes: Sequence[int], tau: float,
                        duplicate_flags: np.ndarray, label_names: dict[int, str],
                        n_classes: int, predicted: np.ndarray | None = None,
                        overhead_ms: float | None = None) -> dict[str, Any]:
    """Multiclass and binary metrics, on the full test set and duplicate-free.

    `predicted` overrides argmax for the majority-vote baseline, which produces a
    class directly rather than a distribution.
    """
    y_true = np.asarray(y_test[TARGET_COLUMN])
    y_binary = np.asarray(y_test["Label_binary"]).astype("int64")
    probs = np.asarray(probs, dtype="float64")
    y_pred = np.argmax(probs, axis=1) if predicted is None else np.asarray(predicted)
    risk = attack_risk(probs, benign_codes)
    clean = ~np.asarray(duplicate_flags, dtype=bool)

    full = ev.headline_metrics(y_true, y_pred, probs, n_classes)
    free = (ev.headline_metrics(y_true[clean], y_pred[clean], probs[clean], n_classes)
            if clean.sum() else {"rows": 0})
    return {
        "name": name, "full": full, "duplicate_free": free,
        "binary": binary_metrics(risk, y_binary, tau),
        "binary_duplicate_free": (binary_metrics(risk[clean], y_binary[clean], tau)
                                  if clean.sum() else {}),
        "per_class": ev.per_class_table(y_true, y_pred, label_names, n_classes),
        "risk": risk, "y_pred": y_pred, "overhead_ms_per_1000": overhead_ms,
    }


def comparison_row(result: dict[str, Any]) -> dict[str, Any]:
    full, free = result["full"], result.get("duplicate_free", {})
    binary = result["binary"]
    auc = (full.get("roc_auc") or {}).get("macro")
    return {
        "configuration": result["name"],
        "accuracy": full.get("accuracy"),
        "precision (macro)": full.get("precision_macro"),
        "recall (macro)": full.get("recall_macro"),
        "F1 (macro)": full.get("f1_macro"),
        "F1 (weighted)": full.get("f1_weighted"),
        "G-mean": full.get("g_mean"),
        "ROC-AUC (macro OvR)": auc,
        "binary F1": binary.get("f1"),
        "binary FPR": binary.get("fpr"),
        "binary ROC-AUC": binary.get("roc_auc"),
        "binary PR-AUC": binary.get("pr_auc"),
        "F1 macro (dup-free)": free.get("f1_macro"),
        "G-mean (dup-free)": free.get("g_mean"),
        "fusion overhead (ms/1k)": result.get("overhead_ms_per_1000"),
    }


def fuse_once(probs_list: Sequence[np.ndarray], model_names: Sequence[str],
              lookup: dict, macro: dict, contexts: np.ndarray,
              use_context: bool = True, confidence: bool | None = None,
              mode: str | None = None, temperature: float | None = None):
    """One complete fusion pass.  Returns (P, weights, reliability, confidence)."""
    conf = np.column_stack([
        normalised_entropy_confidence(p, enabled=confidence) for p in probs_list])
    rel = reliability_matrix(lookup, macro, model_names, contexts,
                             use_context=use_context)
    weights = dynamic_weights(rel, conf, mode=mode, temperature=temperature)
    return fuse_probabilities(probs_list, weights), weights, rel, conf


# ===========================================================================
# 13.6  CHARTS
# ===========================================================================
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
_SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
             "#0d366b"]


def _pyplot(verbose: bool = True):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        if verbose:
            print("  [SKIP]  matplotlib is not installed, so no chart was drawn.")
        return None


def _finish(fig, path, verbose, tight=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def _style(ax):
    ax.set_facecolor(dl._SURFACE)
    ax.yaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(dl._GRID)
    ax.tick_params(colors=dl._INK_SOFT, labelsize=9)


def plot_reliability_heatmap(table: pd.DataFrame, path, verbose: bool = True):
    """Rel_m(k) as model x context.  One measure in [0, 1], so one hue."""
    plt = _pyplot(verbose)
    if plt is None or table.empty:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    grid = table.pivot(index="model", columns="context", values="reliability")
    values = grid.to_numpy(dtype="float64")
    cmap = LinearSegmentedColormap.from_list("mscad_blue", _SEQ_RAMP)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * grid.shape[1] + 3.0),
                                    max(3.2, 0.8 * grid.shape[0] + 2.2)), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto",
                      interpolation="nearest")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(grid.columns, rotation=25, ha="right", fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index, fontsize=9, color=dl._INK_SOFT)
    ax.set_title("Context reliability Rel$_m(k)$ (validation rows only)",
                 color=dl._INK, fontsize=12, pad=14)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(values[i, j]):
                continue
            ax.annotate("%.3f" % values[i, j], (j, i), ha="center", va="center",
                        fontsize=8.5,
                        color="#ffffff" if values[i, j] > 0.55 else dl._INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("reliability", color=dl._INK_SOFT, fontsize=9)
    bar.ax.tick_params(colors=dl._INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)
    return _finish(fig, path, verbose)


def plot_mean_weights(summary: pd.DataFrame, model_names: Sequence[str], path,
                      verbose: bool = True):
    """Mean weight per model within each true class, stacked to 1."""
    plt = _pyplot(verbose)
    if plt is None or summary.empty:
        return None
    grid = summary.pivot(index="true class", columns="model", values="mean weight")
    grid = grid[[m for m in model_names if m in grid.columns]]
    labels = [str(v) for v in grid.index]
    positions = np.arange(len(labels), dtype="float64")

    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(labels) + 3.0), 5.0), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)

    bottom = np.zeros(len(labels))
    gap = 0.004
    for i, model in enumerate(grid.columns):
        values = grid[model].to_numpy(dtype="float64")
        ax.bar(positions, values, 0.62, bottom=bottom + i * gap, label=model,
               color=_SERIES[i % len(_SERIES)])
        for x, (value, base) in enumerate(zip(values, bottom)):
            if value > 0.06:
                ax.annotate("%.2f" % value, (x, base + value / 2), ha="center",
                            va="center", fontsize=8, color="#ffffff")
        bottom = bottom + values
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_ylabel("mean weight (sums to 1)", color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Mean fusion weight per model, by true class (test set)",
                 color=dl._INK, fontsize=12, pad=14)
    ax.set_ylim(0, 1.12)
    legend = ax.legend(frameon=False, loc="upper right", fontsize=9,
                       ncol=len(grid.columns))
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return _finish(fig, path, verbose)


def plot_risk_distribution(risk: np.ndarray, y_binary: np.ndarray, tau: float,
                           path, verbose: bool = True):
    """Where benign and attack flows sit on the risk scale, with tau marked."""
    plt = _pyplot(verbose)
    if plt is None:
        return None
    risk = np.asarray(risk, dtype="float64")
    truth = np.asarray(y_binary).astype("int64")

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    bins = np.linspace(0, 1, 41)
    ax.hist(risk[truth == 0], bins=bins, color=_SERIES[0], alpha=0.85,
            label="benign (%s rows)" % f"{int((truth == 0).sum()):,}")
    ax.hist(risk[truth == 1], bins=bins, color=_SERIES[1], alpha=0.75,
            label="attack (%s rows)" % f"{int((truth == 1).sum()):,}")
    ax.axvline(tau, color=dl._INK, linewidth=2, linestyle="--")
    ax.annotate(r"$\tau$ = %.4f" % tau, (tau, ax.get_ylim()[1] * 0.92),
                textcoords="offset points", xytext=(6, 0), fontsize=10,
                color=dl._INK)
    ax.set_yscale("log")
    ax.set_xlabel(r"fused risk $R_t$", color=dl._INK_SOFT, fontsize=10)
    ax.set_ylabel("flows (log scale)", color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Fused risk on the test set, with the validation threshold",
                 color=dl._INK, fontsize=12, pad=14)
    legend = ax.legend(frameon=False, loc="upper center", fontsize=9)
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return _finish(fig, path, verbose)


def plot_curves(series: dict[str, np.ndarray], y_binary: np.ndarray, path,
                verbose: bool = True):
    """ROC and PR curves for the fused risk against each single model.

    Each legend entry carries its own AUC: three of these hues sit below 3:1
    against the light surface, so the palette's relief rule requires a visible
    label rather than colour alone.
    """
    plt = _pyplot(verbose)
    if plt is None:
        return None
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    truth = np.asarray(y_binary).astype("int64")
    if len(np.unique(truth)) < 2:
        if verbose:
            print("  [SKIP]  the test set has only one binary class, so no curves.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    for ax in axes:
        _style(ax)

    for i, (name, score) in enumerate(series.items()):
        colour = _SERIES[i % len(_SERIES)]
        score = np.asarray(score, dtype="float64")
        fpr, tpr, _t = roc_curve(truth, score)
        axes[0].plot(fpr, tpr, linewidth=2.0, color=colour,
                     label="%s (AUC %.4f)" % (name, roc_auc_score(truth, score)))
        precision, recall, _t2 = precision_recall_curve(truth, score)
        axes[1].plot(recall, precision, linewidth=2.0, color=colour,
                     label="%s (AP %.4f)"
                           % (name, average_precision_score(truth, score)))

    axes[0].plot([0, 1], [0, 1], linewidth=1.0, color=dl._GRID, linestyle="--")
    axes[0].set_xlabel("false-positive rate", color=dl._INK_SOFT, fontsize=10)
    axes[0].set_ylabel("true-positive rate", color=dl._INK_SOFT, fontsize=10)
    axes[0].set_title("ROC (benign vs attack)", color=dl._INK, fontsize=11, pad=12)
    axes[1].set_xlabel("recall", color=dl._INK_SOFT, fontsize=10)
    axes[1].set_ylabel("precision", color=dl._INK_SOFT, fontsize=10)
    axes[1].set_title("Precision-recall", color=dl._INK, fontsize=11, pad=12)
    for ax in axes:
        legend = ax.legend(frameon=False, loc="lower left", fontsize=8.5)
        for text in legend.get_texts():
            text.set_color(dl._INK_SOFT)
    return _finish(fig, path, verbose)


def plot_comparison_bars(table: pd.DataFrame, columns, title, path,
                         verbose: bool = True):
    """Grouped bars across configurations, every bar directly labelled."""
    plt = _pyplot(verbose)
    if plt is None or table.empty:
        return None
    columns = [c for c in columns if c in table.columns]
    values = [pd.to_numeric(table[c], errors="coerce").to_numpy() for c in columns]
    names = [str(v) for v in table["configuration"]]
    positions = np.arange(len(names), dtype="float64")
    width = 0.80 / max(len(columns), 1)
    gap = 0.01

    fig, ax = plt.subplots(figsize=(max(9.0, 1.5 * len(names) + 4.0), 5.4), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    for i, (column, series) in enumerate(zip(columns, values)):
        offset = (i - (len(columns) - 1) / 2) * (width + gap)
        bars = ax.bar(positions + offset, np.nan_to_num(series), width,
                      label=column, color=_SERIES[i % len(_SERIES)])
        for bar, value in zip(bars, series):
            if np.isnan(value):
                continue
            ax.annotate("%.3f" % value,
                        (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=7, color=dl._INK)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5,
                       color=dl._INK_SOFT)
    ax.set_ylabel("score", color=dl._INK_SOFT, fontsize=10)
    ax.set_title(title, color=dl._INK, fontsize=12, pad=14)
    ax.set_ylim(0, 1.14)
    legend = ax.legend(frameon=False, loc="upper right", fontsize=8.5,
                       ncol=len(columns))
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return _finish(fig, path, verbose)


# ===========================================================================
# 13.7  THE TABLES THAT GO TO DISK
# ===========================================================================
def _context_labeller(label_names: dict[int, str]):
    """How to print a context key: a class name, or the protocol as it stands."""
    if config.CONTEXT_MODE == "consensus_class":
        def name_of(key):
            try:
                return label_names.get(int(key), str(key))
            except (TypeError, ValueError):
                return str(key)
        return name_of
    return lambda key: str(key)


def weights_summary_table(weights: np.ndarray, model_names: Sequence[str],
                          contexts: np.ndarray, y_true: np.ndarray,
                          label_names: dict[int, str]) -> pd.DataFrame:
    """Mean +- std of w_m, grouped by context and again by true class.

    Both groupings answer different questions: by context says what the fusion
    DECIDED to do, by true class says whether that decision was the right one --
    if the weights by true class are flat, the adaptation is not doing anything.
    """
    weights = np.asarray(weights, dtype="float64")
    contexts = np.asarray(contexts)
    y_true = np.asarray(y_true)
    if weights.shape != (len(contexts), len(model_names)):
        raise FusionError(
            "the weight matrix is %s but there are %d flows and %d models"
            % (weights.shape, len(contexts), len(model_names)))

    def class_name(key):
        return label_names.get(int(key), str(key))

    groups = [("context", contexts, _context_labeller(label_names)),
              ("true class", y_true, class_name)]
    rows = []
    for grouping, keys, to_name in groups:
        for key in sorted(set(keys.tolist()), key=str):
            selected = keys == key
            for j, model in enumerate(model_names):
                part = weights[selected, j]
                rows.append({
                    "grouping": grouping, "group": to_name(key),
                    "model": model, "rows": int(selected.sum()),
                    "mean weight": round(float(part.mean()), 6),
                    "std weight": round(float(part.std(ddof=0)), 6),
                    "min weight": round(float(part.min()), 6),
                    "max weight": round(float(part.max()), 6)})
    return pd.DataFrame(rows)


def fused_predictions_frame(out: dict[str, Any]) -> pd.DataFrame:
    """One row per test flow: the evidence, the weights, the fused risk.

    Rounded to 8 places, not 6: three weights rounded to 6 places can miss 1.0
    by 1.5e-6, and the saved file has to pass a 1e-6 sum check when it is read
    back in.
    """
    model_names = list(out["model_names"])
    label_names = out["label_names"]
    fused = np.asarray(out["fused"], dtype="float64")
    weights = np.asarray(out["weights"], dtype="float64")
    conf = np.asarray(out["conf_matrix"], dtype="float64")
    rel = np.asarray(out["rel_matrix"], dtype="float64")
    risk = np.asarray(out["risk"], dtype="float64")
    y_true = np.asarray(out["y_test"][TARGET_COLUMN])
    n_classes = int(out["n_classes"])
    meta = out.get("meta_test")
    tau = float(out["threshold"]["tau"])

    orig = (meta["Orig_Row"].to_numpy()
            if meta is not None and "Orig_Row" in meta.columns
            else np.arange(len(risk)))
    frame = pd.DataFrame({
        "Orig_Row": orig,
        "y_true": y_true,
        "y_true_label": [label_names.get(int(c), str(c)) for c in y_true],
    })
    if meta is not None and "Source_Capture" in meta.columns:
        frame["Source_Capture"] = meta["Source_Capture"].to_numpy()

    for j, model in enumerate(model_names):
        frame["r_%s" % model] = np.round(
            attack_risk(out["test_probs"][model], out["benign_codes"]), 8)
    for j, model in enumerate(model_names):
        frame["Rel_%s" % model] = np.round(rel[:, j], 8)
    for j, model in enumerate(model_names):
        frame["Conf_%s" % model] = np.round(conf[:, j], 8)
    for j, model in enumerate(model_names):
        frame["w_%s" % model] = np.round(weights[:, j], 8)

    name_of = _context_labeller(label_names)
    frame["context"] = [name_of(k) for k in np.asarray(out["contexts"])]
    for c in range(n_classes):
        frame["P_%s" % label_names.get(c, str(c))] = np.round(fused[:, c], 8)

    fused_class = np.argmax(fused, axis=1)
    frame["R_t"] = np.round(risk, 8)
    frame["fused_class"] = fused_class
    frame["fused_label"] = [label_names.get(int(c), str(c)) for c in fused_class]
    frame["attack_decision"] = (risk >= tau).astype("int64")
    return frame


def per_class_f1_frame(results: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Label x configuration, F1 per class, with the test support beside it."""
    if not results:
        return pd.DataFrame()
    columns = {r["name"]: r["per_class"].set_index("Label")["f1"] for r in results}
    table = pd.DataFrame(columns)
    support = results[0]["per_class"].set_index("Label")["support"]
    table.insert(0, "test rows", support)
    table.index.name = "Label"
    return table.reset_index()


def build_metrics(out: dict[str, Any], saved_files: dict[str, str],
                  summary: str | None = None) -> dict[str, Any]:
    """Everything a reader needs to reproduce the numbers, as one JSON object."""
    model_names = list(out["model_names"])
    fusion = out["results"][-1]
    weights = np.asarray(out["weights"], dtype="float64")
    return {
        "step": 13,
        "name": "Reliability-Aware Adaptive Evidence Fusion",
        "models_fused": model_names,
        "test_rows": int(len(out["risk"])),
        "duplicate_free_rows": int((~np.asarray(out["duplicate_flags"],
                                                dtype=bool)).sum()),
        "classes": int(out["n_classes"]),
        "benign_codes": [int(c) for c in out["benign_codes"]],
        "settings": {
            "RELIABILITY_METRIC": config.RELIABILITY_METRIC,
            "SHRINKAGE_K": config.SHRINKAGE_K,
            "CONFIDENCE_TERM": config.CONFIDENCE_TERM,
            "WEIGHT_MODE": config.WEIGHT_MODE,
            "WEIGHT_TEMPERATURE": config.WEIGHT_TEMPERATURE,
            "EPSILON": config.EPSILON,
            "CONTEXT_MODE": config.CONTEXT_MODE,
            "THRESHOLD_RULE": config.THRESHOLD_RULE,
            "TARGET_FPR": config.TARGET_FPR,
            "TEMPORAL_SMOOTHING": config.TEMPORAL_SMOOTHING,
            "EMA_ALPHA": config.EMA_ALPHA,
        },
        "validation_macro_f1": {k: round(float(v), 6)
                                for k, v in out["info"]["macro"].items()},
        "threshold": out["threshold"],
        "mean_weight": {m: round(float(weights[:, j].mean()), 6)
                        for j, m in enumerate(model_names)},
        "weight_spread": {m: round(float(weights[:, j].std(ddof=0)), 6)
                          for j, m in enumerate(model_names)},
        "fusion": {"full": fusion["full"],
                   "duplicate_free": fusion["duplicate_free"],
                   "binary": fusion["binary"],
                   "binary_duplicate_free": fusion["binary_duplicate_free"]},
        "fusion_overhead_ms_per_1000_flows": out.get("overhead_ms_per_1000"),
        "alignment_ok": bool((out["alignment"]["result"] == "[OK]").all()),
        "leakage_note": ("the per-context reliabilities, the weights and tau all "
                         "come from the validation part of the Step 9 split; the "
                         "test set is scored once with those values held fixed"),
        "files": saved_files,
        # the written summary travels inside metrics.json rather than in a file
        # of its own, so the notebook can render it on the resumable path too
        "summary": summary,
    }


def verify_saved_step13(folder: Path, out: dict[str, Any],
                        verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """Read the saved files back from disk and check them.  Nothing is trusted
    from memory here: a file that was not written, or written wrong, fails."""
    names = config.STEP13_FILES
    model_names = list(out["model_names"])
    expected_rows = int(len(out["risk"]))
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    path = folder / names["fused_predictions"]
    if not path.exists():
        add("fused_predictions.csv exists", False, "not written")
        table = pd.DataFrame(checks)
        return table, False

    predictions = pd.read_csv(path, low_memory=False)
    add("fused_predictions.csv exists", True, path.name)
    add("row count equals the test set", len(predictions) == expected_rows,
        "%s rows, expected %s" % (f"{len(predictions):,}", f"{expected_rows:,}"))

    weight_columns = ["w_%s" % m for m in model_names]
    have_weights = all(c in predictions.columns for c in weight_columns)
    if have_weights:
        error = float(np.abs(predictions[weight_columns].to_numpy(dtype="float64")
                             .sum(axis=1) - 1.0).max())
        add("the weights of every flow sum to 1", error < 1e-6,
            "max |sum - 1| = %.3e (tolerance 1e-6)" % error)
    else:
        add("the weights of every flow sum to 1", False,
            "missing %s" % [c for c in weight_columns
                            if c not in predictions.columns])

    probability_columns = [c for c in predictions.columns if c.startswith("P_")]
    add("one fused probability column per class",
        len(probability_columns) == int(out["n_classes"]),
        "%d columns for %d classes" % (len(probability_columns),
                                       int(out["n_classes"])))
    if probability_columns:
        error = float(np.abs(
            predictions[probability_columns].to_numpy(dtype="float64").sum(axis=1)
            - 1.0).max())
        add("every fused distribution sums to 1", error < 1e-6,
            "max |sum - 1| = %.3e (tolerance 1e-6)" % error)

    risk = predictions["R_t"].to_numpy(dtype="float64")
    add("R_t lies in [0, 1]",
        bool(np.isfinite(risk).all() and risk.min() >= 0.0 and risk.max() <= 1.0),
        "min %.6f, max %.6f" % (float(risk.min()), float(risk.max())))

    tau = float(out["threshold"]["tau"])
    decided = predictions["attack_decision"].to_numpy(dtype="int64")
    add("the attack decision is R_t >= tau",
        bool((decided == (risk >= tau).astype("int64")).all()),
        "tau = %.6f, %s flows flagged" % (tau, f"{int(decided.sum()):,}"))

    reliability_path = folder / names["reliability"]
    if reliability_path.exists():
        reliability = pd.read_csv(reliability_path)
        values = reliability["reliability"].to_numpy(dtype="float64")
        add("every reliability lies in [0, 1]",
            bool(values.min() >= 0.0 and values.max() <= 1.0),
            "%d rows, min %.6f, max %.6f"
            % (len(reliability), float(values.min()), float(values.max())))
    else:
        add("every reliability lies in [0, 1]", False, "reliability_table.csv missing")

    comparison_path = folder / names["comparison"]
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        add("the comparison table holds every configuration",
            len(comparison) == len(out["results"]),
            "%d rows" % len(comparison))
    else:
        add("the comparison table holds every configuration", False,
            "comparison_table.csv missing")

    threshold_path = folder / names["threshold"]
    if threshold_path.exists():
        stored = json.loads(threshold_path.read_text(encoding="utf-8"))
        add("threshold.json records the validation threshold",
            abs(float(stored.get("tau", -1)) - tau) < 1e-12,
            "tau = %.6f, rule = %s" % (float(stored.get("tau", float("nan"))),
                                       stored.get("rule")))
    else:
        add("threshold.json records the validation threshold", False, "missing")

    # the report itself is written after this check, so it cannot be in it
    expected = [k for k in names if k != "report"]
    missing = [names[k] for k in expected if not (folder / names[k]).exists()]
    add("every Step 13 file was written", not missing,
        "%d files" % len(expected) if not missing else "missing %s" % missing)

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table, note="Read back from disk after writing. A `[FAIL]` here "
                               "means the saved file is wrong, whatever the "
                               "in-memory numbers said.")
    return table, all_ok


def _row_value(table: pd.DataFrame, name: str, column: str) -> float:
    part = table[table["configuration"] == name]
    if not len(part) or column not in table.columns:
        return float("nan")
    return float(pd.to_numeric(part[column], errors="coerce").iloc[0])


def _leader(table: pd.DataFrame, column: str, tolerance: float = 1e-9):
    """(names, value) of the best rows in a column -- plural, so a tie reads as one.

    Step 12 learned this the hard way: calling one of four identical values the
    "highest" is a false statement, and starring all four is no better.
    """
    values = pd.to_numeric(table[column], errors="coerce")
    if not len(values.dropna()):
        return [], float("nan")
    best = float(values.max())
    names = table.loc[values >= best - tolerance, "configuration"].tolist()
    return names, best


def step13_summary(out: dict[str, Any]) -> str:
    """The Step 13 summary, composed only from numbers that were computed.

    Every figure below is read off the comparison, ablation and weight tables, so
    this text cannot contradict the tables it sits beside.
    """
    comparison = out["comparison"]
    ablation = out.get("ablation")
    fusion_name = "ADAPTIVE EVIDENCE FUSION"
    model_names = list(out["model_names"])
    weights = np.asarray(out["weights"], dtype="float64")
    threshold = out["threshold"]
    lines: list[str] = ["### Step 13 - what the numbers say", ""]

    singles = comparison[comparison["configuration"].str.startswith("single:")]
    fixed = comparison[comparison["configuration"].str.startswith("fixed:")]

    for column, scope in (("F1 (macro)", "full test set"),
                          ("F1 macro (dup-free)", "duplicate-free subset")):
        fused_value = _row_value(comparison, fusion_name, column)
        best_single, single_value = _leader(singles, column)
        best_fixed, fixed_value = _leader(fixed, column)
        if not np.isfinite(fused_value):
            continue
        verdict = ("**above** the best single model"
                   if fused_value > single_value + 1e-9 else
                   "**level with** the best single model"
                   if abs(fused_value - single_value) <= 1e-9 else
                   "**below** the best single model")
        lines.append(
            "- On the %s, adaptive fusion scores **%.4f** macro-F1, %s "
            "(%s, %.4f) and %s fixed-weight fusion (%s, %.4f)."
            % (scope, fused_value, verdict,
               ", ".join(best_single) or "none", single_value,
               "above" if fused_value > fixed_value + 1e-9 else
               "level with" if abs(fused_value - fixed_value) <= 1e-9 else "below",
               ", ".join(best_fixed) or "none", fixed_value))

    binary_f1 = _row_value(comparison, fusion_name, "binary F1")
    binary_fpr = _row_value(comparison, fusion_name, "binary FPR")
    pr_auc = _row_value(comparison, fusion_name, "binary PR-AUC")
    lines.append(
        "- At the validation threshold tau = **%.4f** (rule `%s`, chosen on "
        "validation and applied unchanged), the fused risk gives binary F1 "
        "**%.4f**, FPR **%.4f** and PR-AUC **%.4f** on the test set."
        % (threshold["tau"], threshold["rule"], binary_f1, binary_fpr, pr_auc))

    spread = [(name, float(weights[:, j].mean()), float(weights[:, j].std(ddof=0)))
              for j, name in enumerate(model_names)]
    lines.append(
        "- Mean weight per model: %s. The spread is what makes the fusion "
        "adaptive: a model whose weight never moves is being used like a fixed "
        "ensemble member."
        % ", ".join("`%s` %.3f +- %.3f" % row for row in spread))

    if ablation is not None and len(ablation) > 1:
        default = pd.to_numeric(ablation["F1 (macro)"], errors="coerce").iloc[0]
        rows = []
        for _i, row in ablation.iloc[1:].iterrows():
            value = float(pd.to_numeric(pd.Series([row["F1 (macro)"]]),
                                        errors="coerce").iloc[0])
            rows.append("`%s` %+.4f" % (row["ablation"], value - default))
        lines.append("- Ablations, as macro-F1 change from the default "
                     "configuration: %s." % ", ".join(rows))

    overhead = out.get("overhead_ms_per_1000")
    if overhead is not None:
        lines.append(
            "- Fusion overhead: **%.3f ms per 1,000 flows**, excluding the "
            "models' own inference time -- the weighting is arithmetic on three "
            "probability vectors, not another model." % overhead)

    rows = int(len(out["risk"]))
    duplicate_free = int((~np.asarray(out["duplicate_flags"], dtype=bool)).sum())
    duplicate_share = 100.0 * (rows - duplicate_free) / rows if rows else 0.0
    per_class = out["results"][-1]["per_class"]
    support = pd.to_numeric(per_class["support"], errors="coerce")
    smallest = per_class.loc[support.idxmin()] if len(support.dropna()) else None
    best_single_full = _leader(singles, "F1 (macro)")[1]

    lines += [
        "",
        "**Read these with four caveats.**",
        "",
        "1. The reliabilities come from the validation part of the Step 9 split, "
        "which Step 10 also used for early stopping, so they are slightly "
        "optimistic. A third split held out for fusion alone would be cleaner; "
        "with the smallest class as small as it is here there are not enough rows "
        "to estimate a per-context F1 from one.",
    ]
    if smallest is not None:
        lines.append(
            "2. The smallest class in the test set is `%s` with **%d rows**, so "
            "any per-class figure for it moves a long way on one flow changing "
            "hands." % (smallest["Label"], int(smallest["support"])))
    else:
        lines.append("2. The rarest classes have very few test rows, so their "
                     "per-class figures carry high variance.")
    lines.append(
        "3. The duplicate-free columns cover **%s of %s** test rows: %.1f%% of "
        "this test set has a feature vector that also occurs in training, so "
        "every full-test score is inflated by that much memorisable material. "
        "Those columns are the conservative comparison."
        % (f"{duplicate_free:,}", f"{rows:,}", duplicate_share))
    lines.append(
        "4. The best single model already reaches **%.4f** macro-F1 on the full "
        "test set, so there is %s headroom above it; fusion cannot show a gain it "
        "has no room to make. The duplicate-free subset and the per-class table "
        "are where a difference, if there is one, shows up."
        % (best_single_full,
           "very little" if best_single_full > 0.95 else "some"))
    return "\n".join(lines)


def save_step13(out: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Write every Step 13 output, then read it back and check it."""
    model_names = list(out["model_names"])
    label_names = out["label_names"]
    y_true = np.asarray(out["y_test"][TARGET_COLUMN])
    y_binary = np.asarray(out["y_test"]["Label_binary"]).astype("int64")
    tau = float(out["threshold"]["tau"])

    summary = weights_summary_table(out["weights"], model_names, out["contexts"],
                                    y_true, label_names)
    predictions = fused_predictions_frame(out)
    per_class = per_class_f1_frame(out["results"])
    result: dict[str, Any] = {"weights_summary": summary,
                              "fused_predictions": predictions,
                              "per_class": per_class}

    folder = prepare_output_dir(config.STEP13_OUTPUT_DIR, "the Step 13 outputs",
                                verbose=verbose)
    if folder is None:
        result.update({"status": "skipped_not_windows", "ok": False})
        return result

    names = config.STEP13_FILES
    out["reliability"].to_csv(folder / names["reliability"], index=False)
    summary.to_csv(folder / names["weights_summary"], index=False)
    predictions.to_csv(folder / names["fused_predictions"], index=False)
    out["comparison"].to_csv(folder / names["comparison"], index=False)
    out["ablation"].to_csv(folder / names["ablation"], index=False)
    if len(per_class):
        per_class.to_csv(folder / names["per_class"], index=False)
    (folder / names["threshold"]).write_text(
        json.dumps(out["threshold"], indent=2, default=str), encoding="utf-8")

    # ---- charts ----------------------------------------------------------
    charts: dict[str, str] = {}
    drawn = plot_reliability_heatmap(out["reliability"],
                                     folder / names["chart_reliability"],
                                     verbose=verbose)
    if drawn:
        charts["reliability"] = str(drawn)
    by_class = (summary[summary["grouping"] == "true class"]
                .rename(columns={"group": "true class"}))
    drawn = plot_mean_weights(by_class, model_names,
                              folder / names["chart_weights"], verbose=verbose)
    if drawn:
        charts["weights"] = str(drawn)
    drawn = plot_risk_distribution(out["risk"], y_binary, tau,
                                   folder / names["chart_risk"], verbose=verbose)
    if drawn:
        charts["risk"] = str(drawn)
    series = {"adaptive fusion": out["risk"]}
    for model in model_names:
        series[model] = attack_risk(out["test_probs"][model], out["benign_codes"])
    series["simple average"] = attack_risk(
        simple_average([out["test_probs"][m] for m in model_names]),
        out["benign_codes"])
    drawn = plot_curves(series, y_binary, folder / names["chart_curves"],
                        verbose=verbose)
    if drawn:
        charts["curves"] = str(drawn)
    drawn = plot_comparison_bars(
        out["comparison"], ["F1 (macro)", "G-mean", "binary F1", "binary FPR"],
        "Full test set -- F1 and G-mean higher is better, FPR lower",
        folder / names["chart_comparison"], verbose=verbose)
    if drawn:
        charts["comparison"] = str(drawn)
    drawn = plot_comparison_bars(
        out["comparison"], ["F1 macro (dup-free)", "G-mean (dup-free)"],
        "Duplicate-free test subset -- the conservative comparison",
        folder / names["chart_comparison_free"], verbose=verbose)
    if drawn:
        charts["comparison_duplicate_free"] = str(drawn)
    result["charts"] = charts

    # ---- metrics.json, written before the check so the check can read it --
    saved_files = {key: names[key] for key in names
                   if (folder / names[key]).exists()}
    summary = step13_summary(out)
    metrics = build_metrics(out, saved_files, summary=summary)
    result["summary"] = summary
    (folder / names["metrics"]).write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    result["metrics"] = metrics

    report, ok = verify_saved_step13(folder, out, verbose=verbose)
    report.to_csv(folder / names["report"], index=False)
    result.update({"report": report, "ok": ok, "status": "written",
                   "folder": str(folder)})
    if verbose:
        print("  [%s]  Step 13 outputs written to %s"
              % ("OK" if ok else "FAIL", folder))
    return result


# ===========================================================================
# 13.8  RUN EVERYTHING
# ===========================================================================
def step13_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP13_OUTPUT_DIR)
    if folder is None:
        return False
    return all((folder / config.STEP13_FILES[k]).exists()
               for k in ("metrics", "comparison", "fused_predictions", "threshold"))


def load_step13(verbose: bool = True) -> dict[str, Any]:
    folder = config.usable_output_dir(config.STEP13_OUTPUT_DIR)
    if folder is None:
        raise FusionError("the Step 13 folder is not usable here -- %s"
                          % config.why_unusable(config.STEP13_OUTPUT_DIR))
    names = config.STEP13_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    for key in ("reliability", "weights_summary", "fused_predictions", "comparison",
                "ablation", "per_class", "report"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    for key in ("threshold", "metrics"):
        path = folder / names[key]
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP13 = False)"),
            ("folder", str(folder)),
            ("threshold tau", "%.6f" % out.get("threshold", {}).get("tau", float("nan"))),
            ("configurations compared", len(out.get("comparison", []))),
        ]), note="Set `OVERWRITE_STEP13 = True` in config.py to recompute.")
    return out


def run_step13(overwrite: bool | None = None, verbose: bool = True) -> dict[str, Any]:
    """STEP 13 end to end: reliability -> weights -> fusion -> comparison."""
    overwrite = config.OVERWRITE_STEP13 if overwrite is None else overwrite
    if not overwrite and step13_outputs_exist():
        return load_step13(verbose=verbose)

    model_names = check_fusion_models(verbose=verbose)

    # ---- inputs -------------------------------------------------------
    loaded = ev.load_evaluation_inputs(verbose=False)
    y_test, meta_test = loaded["y_test"], loaded.get("meta_test")
    label_names = ev.label_names_from_mapping(loaded["label_mapping"])
    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    y_train = pd.read_csv(folder8 / config.STEP8_FILES["y_train"], low_memory=False)
    n_classes = int(max(int(y_train[TARGET_COLUMN].max()),
                        int(y_test[TARGET_COLUMN].max())) + 1)
    benign_codes = ev.benign_codes_from(y_test, label_names)
    class_order = [label_names.get(c, str(c)) for c in range(n_classes)]

    duplicate_flags = np.zeros(len(y_test), dtype=bool)
    if loaded["X_train"] is not None and loaded["X_test"] is not None:
        duplicate_flags = ev.duplicate_mask(loaded["X_train"], loaded["X_test"])

    if verbose:
        show_title("Step 13.2 -- validation predictions and context reliability",
                   level=3)
    bundles = {name: load_model_bundle(name, verbose=verbose)
               for name in model_names}
    val_probs, y_val, _positions = validation_probabilities(bundles, y_train,
                                                            verbose=verbose)
    test_probs, match_table = test_probabilities(bundles, y_test, meta_test,
                                                 verbose=verbose)

    # alignment, expressed as the per-model frames Step 11/12 would have written
    frames = {name: pd.DataFrame(
        {"Orig_Row": (meta_test["Orig_Row"].to_numpy() if meta_test is not None
                      else np.arange(len(y_test))),
         "y_true": y_test[TARGET_COLUMN].to_numpy(),
         **{"p_%s" % class_order[c]: test_probs[name][:, c]
            for c in range(n_classes)}})
        for name in model_names}
    alignment, _ok = check_alignment(frames, class_order, verbose=verbose)

    # ---- reliability, on validation only --------------------------------
    val_list = [val_probs[n] for n in model_names]
    test_list = [test_probs[n] for n in model_names]
    val_contexts, test_contexts, context_names = context_sets(
        config.CONTEXT_MODE, val_list, test_list, y_train, meta_test,
        label_names, n_classes)
    if val_contexts is None:
        raise FusionError(
            "CONTEXT_MODE = \"protocol\" needs the Protocol column in the Step 5 "
            "meta_train.csv and meta_test.csv. Run Step 5, or set CONTEXT_MODE = "
            "\"consensus_class\".")

    reliability, info = build_reliability(
        val_probs, y_val[TARGET_COLUMN].to_numpy(), val_contexts, context_names,
        n_classes, verbose=verbose)

    # ---- the threshold, on validation only -------------------------------
    if verbose:
        show_title("Step 13.4 -- fused risk and the decision threshold", level=3)
    val_fused, val_weights, _r, _c = fuse_once(
        val_list, model_names, info["lookup"], info["macro"], val_contexts)
    val_risk = attack_risk(val_fused, benign_codes)
    threshold = choose_threshold(val_risk,
                                 y_val["Label_binary"].to_numpy())
    tau = threshold["tau"]
    if verbose:
        show_table(kv_table([
            ("rule", threshold["rule"]),
            ("tau", "%.6f" % tau),
            ("validation F1 at tau", "%.4f" % threshold["validation_f1"]),
            ("validation FPR at tau", "%.4f" % threshold["validation_fpr"]),
        ]), note="Chosen on the validation part and then applied unchanged to the "
                 "test set.")

    # ---- the test set, once ----------------------------------------------
    started = time.perf_counter()
    fused, weights, rel_matrix, conf_matrix = fuse_once(
        test_list, model_names, info["lookup"], info["macro"], test_contexts)
    risk = attack_risk(fused, benign_codes)
    if config.TEMPORAL_SMOOTHING and meta_test is not None:
        risk = ema_within_capture(risk, meta_test["Source_Capture"],
                                  meta_test.get("Timestamp"))
    overhead = (time.perf_counter() - started) / max(len(fused), 1) * 1e6

    # ---- the comparison ---------------------------------------------------
    if verbose:
        show_title("Step 13.5 -- comparison with single models and fixed fusion",
                   level=3)
    results = []
    for name in model_names:
        results.append(score_probabilities(
            "single: %s" % name, test_probs[name], y_test, benign_codes, tau,
            duplicate_flags, label_names, n_classes))
    average = simple_average(test_list)
    results.append(score_probabilities("fixed: simple average", average, y_test,
                                       benign_codes, tau, duplicate_flags,
                                       label_names, n_classes))
    results.append(score_probabilities(
        "fixed: majority vote", average, y_test, benign_codes, tau,
        duplicate_flags, label_names, n_classes,
        predicted=majority_vote(test_list)))
    static = static_weights([info["macro"][n] for n in model_names])
    static_fused = fuse_probabilities(
        test_list, np.tile(static, (len(y_test), 1)))
    results.append(score_probabilities("fixed: static macro-F1 weights",
                                       static_fused, y_test, benign_codes, tau,
                                       duplicate_flags, label_names, n_classes))
    results.append(score_probabilities(
        "ADAPTIVE EVIDENCE FUSION", fused, y_test, benign_codes, tau,
        duplicate_flags, label_names, n_classes, overhead_ms=overhead))

    comparison = pd.DataFrame([comparison_row(r) for r in results])

    # ---- ablations --------------------------------------------------------
    if verbose:
        show_title("Step 13.6 -- ablation study", level=3)
    ablations = [
        ("no confidence term", dict(confidence=False)),
        ("no context (global reliability)", dict(use_context=False)),
        ("softmax weights, T = 0.2", dict(mode="softmax", temperature=0.2)),
        ("softmax weights, T = 1.0", dict(mode="softmax", temperature=1.0)),
    ]
    ablation_rows = [comparison_row(results[-1]) |
                     {"ablation": "default (context = %s, %s weights)"
                      % (config.CONTEXT_MODE, config.WEIGHT_MODE)}]
    for label, kwargs in ablations:
        alt_fused, _w, _r2, _c2 = fuse_once(
            test_list, model_names, info["lookup"], info["macro"], test_contexts,
            **kwargs)
        scored = score_probabilities(label, alt_fused, y_test, benign_codes, tau,
                                     duplicate_flags, label_names, n_classes)
        ablation_rows.append(comparison_row(scored) | {"ablation": label})

    # the other CONTEXT_MODE needs its own reliability table, so it cannot go
    # through the loop above -- the contexts themselves change
    other = ("protocol" if config.CONTEXT_MODE == "consensus_class"
             else "consensus_class")
    val_other, test_other, names_other = context_sets(
        other, val_list, test_list, y_train, meta_test, label_names, n_classes)
    if val_other is None:
        if verbose:
            show_note("The `context = %s` ablation is **skipped**: the Protocol "
                      "column is not in the Step 5 metadata on this machine, and "
                      "Step 13 does not guess a context." % other)
    else:
        _table, info_other = build_reliability(
            val_probs, y_val[TARGET_COLUMN].to_numpy(), val_other, names_other,
            n_classes, context_mode=other, verbose=False)
        alt_fused, _w, _r3, _c3 = fuse_once(
            test_list, model_names, info_other["lookup"], info_other["macro"],
            test_other)
        label = "context = %s" % other
        scored = score_probabilities(label, alt_fused, y_test, benign_codes, tau,
                                     duplicate_flags, label_names, n_classes)
        ablation_rows.append(comparison_row(scored) | {"ablation": label})
    ablation = pd.DataFrame(ablation_rows)

    out: dict[str, Any] = {
        "model_names": model_names, "reliability": reliability,
        "threshold": threshold, "comparison": comparison, "ablation": ablation,
        "results": results, "fused": fused, "weights": weights, "risk": risk,
        "contexts": test_contexts, "alignment": alignment,
        "probability_match": match_table, "info": info,
        "benign_codes": benign_codes, "label_names": label_names,
        "n_classes": n_classes, "duplicate_flags": duplicate_flags,
        "test_probs": test_probs, "overhead_ms_per_1000": overhead,
        "conf_matrix": conf_matrix, "rel_matrix": rel_matrix,
        "y_test": y_test, "meta_test": meta_test,
    }
    saved = save_step13(out, verbose=verbose)
    out.update({"status": saved.get("status", "written"), "saved": saved,
                "ok": bool(saved.get("ok"))})
    # lift the saved tables to the top level, under the same keys load_step13
    # uses, so the notebook cells read the same dictionary on both paths
    for key in ("weights_summary", "fused_predictions", "per_class", "report",
                "metrics", "summary", "charts", "folder"):
        if key in saved:
            out[key] = saved[key]

    if verbose:
        show_title("Step 13 -- comparison", level=3)
        show_table(comparison, all_columns=True,
                   note="Single models, fixed-weight fusion and the adaptive "
                        "fusion, all on the same untouched test set with the same "
                        "evaluator. The duplicate-free columns are the "
                        "conservative comparison.")
    return out
