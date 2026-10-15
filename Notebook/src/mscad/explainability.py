r"""
src/mscad/explainability.py -- STEP 14: the explainability layer.

NEW FILE.  Steps 1-13 are finished: nothing in converter.py, data_loading.py,
preprocessing.py, splitting.py, feature_selection.py, autoencoder.py,
imbalance.py, models.py, evaluation.py, experiments.py or fusion.py is
modified.  This module calls those functions and adds its own beside them.

What this does
--------------
Step 13 answers "how suspicious is this flow" with one number.  An analyst
needs the reason.  For every ATTACK or SUSPICIOUS alert this builds an evidence
report:

    which input features moved each model          -> Integrated Gradients
    how the fusion combined them into R(x)         -> C_m = w_m r_m, R = sum C_m
    how much the evidence can be trusted           -> completeness, agreement
    what to look at next                           -> a suggested analyst check

Nothing here changes a Step 13 decision.  The models, the encoders, the fusion
weights and tau are fixed; this module only reads them.

Three honesty rules the code enforces rather than merely stating
---------------------------------------------------------------
1. **IG explains the MODEL, not the world.**  An attribution says which input
   the network leaned on, which is not the same as what caused the attack.
2. **The attack stage is a lookup, not an inference.**  ATTACK_STAGE_MAP turns
   a predicted label into a tactic name.  A label outside the map is reported
   as "Unmapped"; the code never invents one, and nothing here claims a
   multi-stage progression.
3. **Identifiers are context only.**  Src IP, Dst IP, ports and timestamps are
   printed for the analyst and never reach a model -- the MSCAD labels were
   assigned by attacker IP, so a model given Src IP would simply read the
   answer off it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from . import data_loading as dl  # noqa: E402
from . import evaluation as ev  # noqa: E402
from . import fusion as fz  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = "Label_encoded"

# the two things an attribution can be computed FOR
TARGET_CLASS = "class_probability"      # p_m(predicted class | x)
TARGET_RISK = "attack_risk"             # r_m(x) = 1 - sum of benign probability
TARGETS = (TARGET_CLASS, TARGET_RISK)

BAND_ATTACK = "ATTACK"
BAND_SUSPICIOUS = "SUSPICIOUS"
BAND_BENIGN = "BENIGN"

UNMAPPED = "Unmapped"


class ExplainabilityError(Exception):
    """Raised when Step 14 cannot continue without guessing."""


# ===========================================================================
# 14.0  SMALL PURE HELPERS -- no TensorFlow, no file I/O
# ===========================================================================
def rank_values(values: Sequence[float]) -> np.ndarray:
    """Ascending ranks with ties averaged, which is what Spearman needs."""
    values = np.asarray(values, dtype="float64")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype="float64")
    ranks[order] = np.arange(1, len(values) + 1, dtype="float64")
    # average the ranks of tied values, so a tie cannot be broken by position
    unique, inverse, counts = np.unique(values, return_inverse=True,
                                        return_counts=True)
    if len(unique) < len(values):
        sums = np.zeros(len(unique), dtype="float64")
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation.  Returns 0.0 when either side is constant.

    Written out rather than imported so the behaviour on a constant vector is
    ours and documented: numpy would return nan, which would then quietly
    poison every mean computed from it.  A constant ranking carries no ordering
    information, so "no correlation" is the honest answer.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if len(a) != len(b):
        raise ExplainabilityError(
            "spearman needs two sequences of the same length; got %d and %d"
            % (len(a), len(b)))
    if len(a) < 2:
        return 0.0
    ra, rb = rank_values(a), rank_values(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denominator < 1e-12:
        return 0.0
    return float((ra * rb).sum() / denominator)


def top_k(values: Sequence[float], k: int) -> list[int]:
    """Indices of the k largest values, ties broken by index so it is stable."""
    values = np.asarray(values, dtype="float64")
    k = int(min(max(k, 0), len(values)))
    if k == 0:
        return []
    order = np.lexsort((np.arange(len(values)), -values))
    return [int(i) for i in order[:k]]


def top_k_overlap(a: Sequence[float], b: Sequence[float], k: int) -> float:
    """Share of the top-k that the two rankings have in common, in [0, 1]."""
    first, second = set(top_k(a, k)), set(top_k(b, k))
    if not first:
        return 0.0
    return len(first & second) / len(first)


# ===========================================================================
# 14.1  THE DECISION BANDS
# ===========================================================================
def choose_suspicious_threshold(risk: Sequence[float], y_binary: Sequence[int],
                                target_recall: float | None = None,
                                tau: float | None = None) -> dict[str, Any]:
    """tau_s: the LOWEST risk at which validation attack recall still reaches
    SUSPICIOUS_RECALL_TARGET.

    Chosen on the VALIDATION split only.  Lowest, because any higher threshold
    meeting the same recall would widen the suspicious band for nothing -- the
    point of the band is to catch what sits just under tau, not to flag more.
    """
    target_recall = (config.SUSPICIOUS_RECALL_TARGET if target_recall is None
                     else target_recall)
    risk = np.asarray(risk, dtype="float64")
    truth = np.asarray(y_binary).astype("int64")
    if len(risk) == 0:
        raise ExplainabilityError(
            "tau_s cannot be chosen from an empty validation set")
    positives = int((truth == 1).sum())
    if positives == 0:
        raise ExplainabilityError(
            "the validation split holds no attack rows, so an attack recall "
            "target is meaningless. Check the Step 9 split.")

    candidates = np.unique(np.concatenate([[0.0], risk, [1.0]]))
    reached = []
    for candidate in candidates:
        recall = float(((risk >= candidate) & (truth == 1)).sum() / positives)
        if recall >= target_recall:
            reached.append((candidate, recall))
    if not reached:
        raise ExplainabilityError(
            "no threshold reaches an attack recall of %.4f on the validation "
            "split (the best achievable is %.4f). Lower "
            "SUSPICIOUS_RECALL_TARGET."
            % (target_recall, float(max(
                ((risk >= c) & (truth == 1)).sum() / positives
                for c in candidates))))
    tau_s, recall = max(reached, key=lambda pair: pair[0])
    negatives = int((truth == 0).sum())
    out = {
        "tau_s": float(tau_s),
        "target_recall": float(target_recall),
        "validation_recall": float(recall),
        "validation_fpr": float(((risk >= tau_s) & (truth == 0)).sum() / negatives)
                          if negatives else 0.0,
        "chosen_on": "the validation part of the Step 9 split",
        "rows": int(len(risk)),
    }
    if tau is not None and tau_s > tau:
        # a band whose floor is above tau would be empty; say so rather than
        # silently producing zero SUSPICIOUS alerts
        out["note"] = ("tau_s (%.6f) is above tau (%.6f), so the suspicious "
                       "band is empty: every flow that reaches tau_s is already "
                       "an ATTACK." % (tau_s, tau))
    return out


def decision_band(risk: Sequence[float], tau: float, tau_s: float | None = None,
                  suspicious: bool | None = None) -> np.ndarray:
    """ATTACK / SUSPICIOUS / BENIGN per flow.  The ATTACK rule is Step 13's.

    Step 13 decided ATTACK with `R >= tau` and this repeats that rule exactly;
    the suspicious band only subdivides what Step 13 already called benign, so
    no Step 13 decision changes.
    """
    suspicious = config.SUSPICIOUS_BAND if suspicious is None else suspicious
    risk = np.asarray(risk, dtype="float64")
    bands = np.where(risk >= tau, BAND_ATTACK, BAND_BENIGN).astype(object)
    if suspicious and tau_s is not None:
        middle = (risk >= tau_s) & (risk < tau)
        bands[middle] = BAND_SUSPICIOUS
    return bands


def select_flows(bands: Sequence[str], fused_class: Sequence[int],
                 per_class: int | None = None, random_state: int | None = None,
                 benign_sample: int = 25) -> np.ndarray:
    """Which test rows to explain: every alert up to `per_class` per fused class.

    Deterministic: the rows of a class are taken in a fixed shuffled order from
    a seeded generator, so two runs explain the same flows.  A class with fewer
    alerts than `per_class` contributes all of them, which is how the rare
    classes stay represented.  A small benign sample is added for contrast.
    """
    per_class = (config.EXPLAIN_SAMPLE_PER_CLASS if per_class is None
                 else per_class)
    random_state = (config.EXPLAIN_RANDOM_STATE if random_state is None
                    else random_state)
    bands = np.asarray(bands, dtype=object)
    fused_class = np.asarray(fused_class)
    rng = np.random.default_rng(random_state)

    chosen: list[int] = []
    alert = np.isin(bands, [BAND_ATTACK, BAND_SUSPICIOUS])
    for code in sorted({int(c) for c in fused_class[alert].tolist()}):
        rows = np.flatnonzero(alert & (fused_class == code))
        if len(rows) > per_class:
            rows = np.sort(rng.choice(rows, size=per_class, replace=False))
        chosen.extend(int(r) for r in rows)

    benign_rows = np.flatnonzero(~alert)
    if len(benign_rows) and benign_sample > 0:
        take = int(min(benign_sample, len(benign_rows)))
        chosen.extend(int(r) for r in
                      np.sort(rng.choice(benign_rows, size=take, replace=False)))
    return np.array(sorted(set(chosen)), dtype="int64")


# ===========================================================================
# 14.5  THE ANALYST-FACING VOCABULARY
# ===========================================================================
def attack_stage(label: str, mapping: dict[str, str] | None = None) -> str:
    """The tactic for a label.  A label outside the map is "Unmapped".

    This is a LOOKUP of the predicted label, not an inference about where an
    attack has reached.  Guessing a tactic for an unknown label would be the
    one failure mode worth avoiding here, so it is not done.
    """
    mapping = config.ATTACK_STAGE_MAP if mapping is None else mapping
    return str(mapping.get(str(label), UNMAPPED))


def recommended_action(label: str, src_ip: Any, dst_ip: Any,
                       templates: dict[str, str] | None = None) -> str:
    """A suggested check for an analyst.  Never an automated action."""
    templates = config.ACTION_TEMPLATES if templates is None else templates
    template = templates.get(str(label))
    if template is None:
        return ("Suggested analyst check: review the flow from %s to %s manually; "
                "no action template is defined for class %r."
                % (src_ip, dst_ip, str(label)))
    return "Suggested analyst check: " + template.format(
        src_ip=src_ip, dst_ip=dst_ip)


def agreement_status(predicted: Sequence[int]) -> str:
    """unanimous / majority / disagreement across the models for one flow."""
    values = [int(p) for p in predicted]
    if not values:
        return "disagreement"
    counts = {v: values.count(v) for v in set(values)}
    best = max(counts.values())
    if best == len(values):
        return "unanimous"
    if best > len(values) / 2:
        return "majority"
    return "disagreement"


def confidence_label(fused_max_probability: float, agreement: str,
                     completeness_ok: bool,
                     high: float | None = None,
                     medium: float | None = None) -> str:
    """High / Medium / Low, by the documented rule and nothing else.

    High   = fused max probability >= CONF_HIGH_PROB AND all models agree
             AND the IG completeness check passed
    Medium = fused max probability >= CONF_MEDIUM_PROB OR a majority agrees
    Low    = otherwise
    """
    high = config.CONF_HIGH_PROB if high is None else high
    medium = config.CONF_MEDIUM_PROB if medium is None else medium
    probability = float(fused_max_probability)
    if (probability >= high and agreement == "unanimous" and completeness_ok):
        return "High"
    if probability >= medium or agreement in ("unanimous", "majority"):
        return "Medium"
    return "Low"


def uncertainty_notes(fused_max_probability: float, agreement: str,
                      completeness_ok: bool, band: str,
                      training_rows: int | None = None,
                      rare_threshold: int = 100) -> list[str]:
    """Everything about this alert that should make a reader hesitate."""
    notes: list[str] = []
    if float(fused_max_probability) < config.CONF_MEDIUM_PROB:
        notes.append("low fused probability (%.3f < %.2f)"
                     % (fused_max_probability, config.CONF_MEDIUM_PROB))
    if agreement != "unanimous":
        notes.append("model %s" % agreement)
    if not completeness_ok:
        notes.append("IG completeness check failed, so the attributions do not "
                     "fully account for the prediction")
    if band == BAND_SUSPICIOUS:
        notes.append("SUSPICIOUS band: below tau, so Step 13 did not raise it "
                     "as an attack")
    if training_rows is not None and training_rows < rare_threshold:
        notes.append("rare class: %d training rows" % int(training_rows))
    return notes


# ===========================================================================
# 14.2  THE COMPOSED MODEL -- one differentiable function per configuration
# ===========================================================================
def _keras():
    from . import autoencoder as ae
    return ae._keras()


def _tf():
    """TensorFlow itself, for GradientTape.  Imported lazily, like Keras."""
    _keras()                                   # same error message if missing
    import tensorflow as tf
    return tf


def build_composed_model(classifier, encoder=None, n_features: int | None = None,
                         mode: str | None = None, window: int | None = None,
                         name: str = "composed"):
    """x (ORIGINAL features) -> [encoder] -> reshape -> classifier -> softmax.

    Why compose rather than attribute on the latent units: the 15 autoencoder
    outputs have no meaning an analyst can act on.  Putting the encoder inside
    the differentiated function moves the attribution back onto the 30 named
    CICFlowMeter features, which is what the report has to talk about.

    The softmax is applied here because Step 10's classifier emits LOGITS; it is
    applied exactly once, as everywhere else in this pipeline.  Every weight is
    frozen: this is a wrapper, not a model to train.
    """
    keras = _keras()
    mode = config.INPUT_MODE if mode is None else mode
    window = config.WINDOW_SIZE if window is None else window
    if n_features is None:
        raise ExplainabilityError("build_composed_model needs n_features")

    if mode == "feature_seq":
        inputs = keras.Input(shape=(int(n_features),), name="original_features")
        hidden = encoder(inputs) if encoder is not None else inputs
        width = int(hidden.shape[-1])
        hidden = keras.layers.Reshape((width, 1))(hidden)
    elif mode == "time_window":
        inputs = keras.Input(shape=(int(window), int(n_features)),
                             name="original_feature_window")
        hidden = (keras.layers.TimeDistributed(encoder)(inputs)
                  if encoder is not None else inputs)
    else:
        raise ExplainabilityError(
            "INPUT_MODE is %r; Step 14 composes models for \"feature_seq\" and "
            "\"time_window\"" % mode)

    logits = classifier(hidden)
    probabilities = keras.layers.Activation("softmax", name="probabilities")(logits)
    composed = keras.Model(inputs, probabilities, name=name)
    composed.trainable = False
    return composed


def composed_probabilities(composed, X, batch_size: int | None = None) -> np.ndarray:
    """Probabilities straight out of the composed model (softmax included)."""
    batch_size = config.IG_BATCH_SIZE if batch_size is None else batch_size
    values = (X.to_numpy(dtype="float32") if isinstance(X, pd.DataFrame)
              else np.asarray(X, dtype="float32"))
    return np.asarray(composed.predict(values, batch_size=batch_size, verbose=0),
                      dtype="float64")


def check_composed_model(composed, X, reference: np.ndarray,
                         what: str, tolerance: float = 1e-5,
                         verbose: bool = True) -> dict[str, Any]:
    """The composed model must reproduce the saved probabilities, or stop.

    If it does not, every attribution below would explain a DIFFERENT model from
    the one that raised the alert, which is worse than no explanation at all.
    """
    recomputed = composed_probabilities(composed, X)
    reference = np.asarray(reference, dtype="float64")
    if recomputed.shape != reference.shape:
        raise ExplainabilityError(
            "the composed model for %s returns %s probabilities but the saved "
            "predictions are %s" % (what, recomputed.shape, reference.shape))
    difference = float(np.abs(recomputed - reference).max())
    row = {"model": what, "max abs difference": difference,
           "tolerance": tolerance,
           "result": "[OK]" if difference <= tolerance else "[FAIL]"}
    if difference > tolerance:
        raise ExplainabilityError(
            "the composed model for %s does not reproduce the saved "
            "probabilities (max difference %.3e > %.0e). Attributions from it "
            "would explain a different model than the one that raised the "
            "alert. Re-run Step 12 with OVERWRITE_STEP12 = True."
            % (what, difference, tolerance))
    if verbose:
        print("  [OK]    %-26s composed model matches the saved probabilities "
              "(max difference %.2e)" % (what, difference))
    return row


# ===========================================================================
# 14.2  THE BASELINE x'
# ===========================================================================
def baseline_vector(X_train: pd.DataFrame, y_train: pd.DataFrame | None = None,
                    benign_codes: Sequence[int] | None = None,
                    mode: str | None = None) -> np.ndarray:
    """x' -- the reference point every attribution is measured against.

    "benign_median"  the per-feature median of the TRAINING benign rows: "a
                     typical normal flow", which is the comparison an analyst
                     makes anyway ("how is this different from normal?")
    "zeros"          the origin of the scaled space
    "train_mean"     the average training row

    Training rows only.  A baseline computed from the test set would be a
    setting chosen on the data the explanation is then judged on.
    """
    mode = config.IG_BASELINE if mode is None else mode
    values = X_train.to_numpy(dtype="float64")
    if mode == "zeros":
        return np.zeros(values.shape[1], dtype="float32")
    if mode == "train_mean":
        return values.mean(axis=0).astype("float32")
    if mode != "benign_median":
        raise ExplainabilityError(
            "IG_BASELINE is %r; it must be \"benign_median\", \"zeros\" or "
            "\"train_mean\"" % mode)

    if y_train is None or not len(y_train):
        raise ExplainabilityError(
            "IG_BASELINE = \"benign_median\" needs y_train to find the benign "
            "rows")
    if "Label_binary" in y_train.columns:
        benign = np.asarray(y_train["Label_binary"]).astype("int64") == 0
    else:
        codes = {int(c) for c in (benign_codes or [])}
        benign = np.isin(np.asarray(y_train[TARGET_COLUMN]), list(codes))
    if not benign.any():
        raise ExplainabilityError(
            "no benign training rows, so a \"typical normal flow\" baseline "
            "cannot be built. Use IG_BASELINE = \"train_mean\" or \"zeros\".")
    return np.median(values[benign], axis=0).astype("float32")


# ===========================================================================
# 14.2  INTEGRATED GRADIENTS
# ===========================================================================
def _target_function(target: str, class_index: np.ndarray | None,
                     benign_codes: Sequence[int] | None):
    """probs (n, C) -> the scalar per row that IG attributes."""
    tf = _tf()
    if target == TARGET_CLASS:
        if class_index is None:
            raise ExplainabilityError(
                "target %r needs the class index of each flow" % target)
        indices = tf.constant(np.asarray(class_index, dtype="int32"))

        def picked(probabilities):
            return tf.gather(probabilities, indices[:tf.shape(probabilities)[0]],
                             batch_dims=1)
        return picked

    if target == TARGET_RISK:
        codes = [int(c) for c in (benign_codes or [])]

        def risk(probabilities):
            if not codes:
                return tf.ones(tf.shape(probabilities)[0],
                               dtype=probabilities.dtype)
            benign = tf.gather(probabilities, tf.constant(codes, dtype=tf.int32),
                               axis=1)
            return 1.0 - tf.reduce_sum(benign, axis=1)
        return risk

    raise ExplainabilityError(
        "target is %r; it must be %r or %r" % (target, TARGET_CLASS, TARGET_RISK))


def integrated_gradients(composed, X, baseline: np.ndarray,
                         target: str = TARGET_CLASS,
                         class_index: np.ndarray | None = None,
                         benign_codes: Sequence[int] | None = None,
                         steps: int | None = None,
                         batch_size: int | None = None) -> dict[str, Any]:
    r"""IG_i(x) = (x_i - x'_i) * \int_0^1 dF(x' + a(x - x'))/dx_i da.

    The integral is evaluated with the TRAPEZOIDAL rule over `steps` intervals,
    so `steps + 1` gradients are taken per flow and the two endpoints carry half
    weight.  Returns the attributions, F(x), F(x') and the completeness gap.

    Written directly on tf.GradientTape: no explainability library is installed
    for this, and the whole method is six lines of arithmetic around one
    gradient call.
    """
    tf = _tf()
    steps = config.IG_STEPS if steps is None else int(steps)
    batch_size = config.IG_BATCH_SIZE if batch_size is None else int(batch_size)
    if steps < 1:
        raise ExplainabilityError("IG_STEPS must be at least 1; got %r" % steps)

    values = (X.to_numpy(dtype="float32") if isinstance(X, pd.DataFrame)
              else np.asarray(X, dtype="float32"))
    baseline = np.asarray(baseline, dtype="float32")
    if baseline.shape != values.shape[1:]:
        raise ExplainabilityError(
            "the baseline has shape %s but each flow has shape %s"
            % (baseline.shape, values.shape[1:]))
    n_flows = len(values)
    if n_flows == 0:
        return {"attributions": np.zeros((0,) + values.shape[1:]),
                "f_x": np.zeros(0), "f_baseline": 0.0,
                "completeness_error": np.zeros(0), "steps": steps}

    reference = np.broadcast_to(baseline, values.shape).astype("float32")
    difference = values - reference
    alphas = np.linspace(0.0, 1.0, steps + 1, dtype="float32")
    weights = np.full(steps + 1, 1.0 / steps, dtype="float64")
    weights[0] *= 0.5                                   # trapezoidal endpoints
    weights[-1] *= 0.5

    target_fn = _target_function(target, class_index, benign_codes)
    accumulated = np.zeros(values.shape, dtype="float64")

    for alpha, weight in zip(alphas, weights):
        point = reference + alpha * difference
        for start in range(0, n_flows, batch_size):
            chunk = point[start:start + batch_size]
            index = (None if class_index is None
                     else np.asarray(class_index)[start:start + batch_size])
            local_target = _target_function(target, index, benign_codes)
            inputs = tf.convert_to_tensor(chunk)
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                probabilities = composed(inputs, training=False)
                scores = local_target(probabilities)
            gradient = tape.gradient(scores, inputs)
            if gradient is None:
                raise ExplainabilityError(
                    "the composed model returned no gradient; it is not "
                    "differentiable with respect to its input")
            accumulated[start:start + batch_size] += (
                weight * np.asarray(gradient, dtype="float64"))

    attributions = difference.astype("float64") * accumulated

    # F(x) and F(x'), for the completeness identity
    probabilities_x = composed_probabilities(composed, values, batch_size)
    probabilities_b = composed_probabilities(
        composed, reference[:1], batch_size)
    if target == TARGET_CLASS:
        index = np.asarray(class_index, dtype="int64")
        f_x = probabilities_x[np.arange(n_flows), index]
        f_baseline = probabilities_b[0][index]
    else:
        codes = [int(c) for c in (benign_codes or [])
                 if 0 <= int(c) < probabilities_x.shape[1]]
        f_x = (1.0 - probabilities_x[:, codes].sum(axis=1) if codes
               else np.ones(n_flows))
        f_baseline = np.full(n_flows,
                             (1.0 - probabilities_b[0][codes].sum()) if codes
                             else 1.0)

    axes = tuple(range(1, attributions.ndim))
    total = attributions.sum(axis=axes)
    gap = (f_x - f_baseline) - total
    return {"attributions": attributions, "f_x": f_x,
            "f_baseline": f_baseline, "total_attribution": total,
            "completeness_gap": gap, "steps": steps, "target": target}


def completeness_table(result: dict[str, Any], model: str, target: str,
                       tolerance: float | None = None) -> pd.DataFrame:
    """|sum IG - (F(x) - F(x'))| per flow, absolute and relative.

    The relative error divides by |F(x) - F(x')|, which is the quantity being
    explained; where that is ~0 there is nothing to attribute, so the row is
    marked and excluded from the share rather than counted as a failure.
    """
    tolerance = config.COMPLETENESS_TOL if tolerance is None else tolerance
    gap = np.abs(np.asarray(result["completeness_gap"], dtype="float64"))
    span = np.abs(np.asarray(result["f_x"], dtype="float64")
                  - np.asarray(result["f_baseline"], dtype="float64"))
    trivial = span < 1e-9
    relative = np.where(trivial, 0.0, gap / np.where(trivial, 1.0, span))
    return pd.DataFrame({
        "model": model, "target": target,
        "f_x": np.round(result["f_x"], 8),
        "f_baseline": np.round(result["f_baseline"], 8),
        "sum_ig": np.round(result["total_attribution"], 8),
        "absolute_error": np.round(gap, 8),
        "relative_error": np.round(relative, 8),
        "nothing_to_explain": trivial,
        "within_tolerance": (relative <= tolerance) | trivial,
    })


def global_importance(attributions: np.ndarray, feature_names: Sequence[str],
                      model: str, target: str,
                      predicted: Sequence[int] | None = None,
                      label_names: dict[int, str] | None = None) -> pd.DataFrame:
    """Mean |IG| and mean signed IG per feature, overall and per predicted class.

    Both are reported because they answer different questions: mean |IG| says
    how much a feature mattered, mean signed IG says which way it pushed.
    """
    attributions = np.asarray(attributions, dtype="float64")
    if attributions.ndim > 2:                   # time_window: sum over the steps
        attributions = attributions.sum(axis=1)
    rows = []

    def add(scope: str, values: np.ndarray) -> None:
        if not len(values):
            return
        for position, feature in enumerate(feature_names):
            column = values[:, position]
            rows.append({
                "model": model, "target": target, "scope": scope,
                "feature": feature,
                "mean_abs_ig": round(float(np.abs(column).mean()), 8),
                "mean_signed_ig": round(float(column.mean()), 8),
                "flows": int(len(column))})

    add("all explained flows", attributions)
    if predicted is not None:
        predicted = np.asarray(predicted)
        names = label_names or {}
        for code in sorted({int(c) for c in predicted.tolist()}):
            add("predicted: %s" % names.get(code, str(code)),
                attributions[predicted == code])
    return pd.DataFrame(rows)


# ===========================================================================
# 14.3  CHECKING THE EXPLANATIONS BY PERTURBATION
# ===========================================================================
def _replace_feature(values: np.ndarray, position: int,
                     baseline: np.ndarray) -> np.ndarray:
    """A copy of `values` with ONE feature set to its baseline value.

    In time_window mode the feature is replaced at every step of the window,
    because the feature is the thing being tested, not one moment of it.
    """
    changed = values.copy()
    if changed.ndim == 2:
        changed[:, position] = baseline[position]
    elif changed.ndim == 3:
        changed[:, :, position] = baseline[:, position]
    else:
        raise ExplainabilityError(
            "a flow must be (features,) or (steps, features); got %s"
            % (values.shape[1:],))
    return changed


def _scores(composed, values: np.ndarray, target: str,
            class_index: np.ndarray | None, benign_codes: Sequence[int] | None,
            batch_size: int | None = None) -> np.ndarray:
    """F(x) for a whole batch, as the same scalar IG attributes."""
    probabilities = composed_probabilities(composed, values, batch_size)
    if target == TARGET_CLASS:
        index = np.asarray(class_index, dtype="int64")
        return probabilities[np.arange(len(probabilities)), index]
    codes = [int(c) for c in (benign_codes or [])
             if 0 <= int(c) < probabilities.shape[1]]
    if not codes:
        return np.ones(len(probabilities))
    return 1.0 - probabilities[:, codes].sum(axis=1)


def perturbation_drops(composed, X, baseline: np.ndarray,
                       target: str = TARGET_CLASS,
                       class_index: np.ndarray | None = None,
                       benign_codes: Sequence[int] | None = None,
                       batch_size: int | None = None) -> dict[str, Any]:
    """Replace one feature at a time with its baseline value and watch F fall.

    This is the independent check on IG: it needs no gradients and makes no
    assumption about the model, so where the two agree the attribution is
    saying something about the model rather than about the method.
    """
    if config.PERTURB_MODE != "baseline_replace":
        raise ExplainabilityError(
            "PERTURB_MODE is %r; this module implements \"baseline_replace\""
            % config.PERTURB_MODE)
    values = (X.to_numpy(dtype="float32") if isinstance(X, pd.DataFrame)
              else np.asarray(X, dtype="float32"))
    baseline = np.asarray(baseline, dtype="float32")
    n_features = values.shape[-1]

    original = _scores(composed, values, target, class_index, benign_codes,
                       batch_size)
    drops = np.zeros((len(values), n_features), dtype="float64")
    for position in range(n_features):
        changed = _replace_feature(values, position, baseline)
        drops[:, position] = original - _scores(
            composed, changed, target, class_index, benign_codes, batch_size)
    return {"drops": drops, "original": original}


def deletion_check(composed, X, baseline: np.ndarray, attributions: np.ndarray,
                   k: int | None = None, target: str = TARGET_RISK,
                   class_index: np.ndarray | None = None,
                   benign_codes: Sequence[int] | None = None,
                   random_state: int | None = None,
                   batch_size: int | None = None) -> dict[str, Any]:
    """Set the top-K attributed features to baseline, and K random ones.

    If the attribution is informative the top-K removal costs the model more
    than a random removal of the same size.  If it does not, the ranking is not
    telling the analyst anything, and this says so.
    """
    k = config.TOP_K_FEATURES if k is None else int(k)
    random_state = (config.EXPLAIN_RANDOM_STATE if random_state is None
                    else random_state)
    values = (X.to_numpy(dtype="float32") if isinstance(X, pd.DataFrame)
              else np.asarray(X, dtype="float32"))
    baseline = np.asarray(baseline, dtype="float32")
    attributions = np.asarray(attributions, dtype="float64")
    if attributions.ndim > 2:
        attributions = attributions.sum(axis=1)
    n_features = values.shape[-1]
    k = int(min(max(k, 1), n_features))
    rng = np.random.default_rng(random_state)

    original = _scores(composed, values, target, class_index, benign_codes,
                       batch_size)
    best = values.copy()
    chance = values.copy()
    for row in range(len(values)):
        for position in top_k(np.abs(attributions[row]), k):
            if best.ndim == 2:
                best[row, position] = baseline[position]
            else:
                best[row, :, position] = baseline[:, position]
        for position in rng.choice(n_features, size=k, replace=False):
            if chance.ndim == 2:
                chance[row, position] = baseline[position]
            else:
                chance[row, :, position] = baseline[:, position]

    top_scores = _scores(composed, best, target, class_index, benign_codes,
                         batch_size)
    random_scores = _scores(composed, chance, target, class_index, benign_codes,
                            batch_size)
    top_drop = original - top_scores
    random_drop = original - random_scores
    return {
        "k": k, "target": target,
        "mean_drop_top_k": float(top_drop.mean()),
        "mean_drop_random_k": float(random_drop.mean()),
        "top_k_drops_more": bool(top_drop.mean() > random_drop.mean()),
        "margin": float(top_drop.mean() - random_drop.mean()),
        "per_flow_top_k": top_drop, "per_flow_random_k": random_drop,
    }


def attribution_agreement(attributions: np.ndarray, drops: np.ndarray,
                          k: int | None = None) -> pd.DataFrame:
    """Per flow: Spearman between |IG| and the perturbation drops, plus overlap."""
    k = config.TOP_K_FEATURES if k is None else int(k)
    attributions = np.asarray(attributions, dtype="float64")
    if attributions.ndim > 2:
        attributions = attributions.sum(axis=1)
    drops = np.asarray(drops, dtype="float64")
    rows = []
    for position in range(len(attributions)):
        magnitude = np.abs(attributions[position])
        rows.append({
            "flow": position,
            "spearman": round(spearman(magnitude, drops[position]), 6),
            "top_k_overlap": round(top_k_overlap(magnitude, drops[position], k), 6),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 14.6  STABILITY UNDER SMALL NOISE
# ===========================================================================
def stability(composed, X, baseline: np.ndarray, attributions: np.ndarray,
              target: str = TARGET_CLASS, class_index: np.ndarray | None = None,
              benign_codes: Sequence[int] | None = None,
              noise_std: float | None = None, repeats: int | None = None,
              k: int | None = None, steps: int | None = None,
              random_state: int | None = None) -> dict[str, Any]:
    """Re-attribute slightly noisy copies and compare the rankings.

    The features are min-max scaled, so the noise is clipped back into [0, 1]:
    a "perturbed" flow that leaves the feasible range would be testing the
    model outside the space it was fitted on.
    """
    noise_std = config.STABILITY_NOISE_STD if noise_std is None else noise_std
    repeats = config.STABILITY_N_REPEATS if repeats is None else int(repeats)
    k = config.TOP_K_FEATURES if k is None else int(k)
    random_state = (config.EXPLAIN_RANDOM_STATE if random_state is None
                    else random_state)
    values = (X.to_numpy(dtype="float32") if isinstance(X, pd.DataFrame)
              else np.asarray(X, dtype="float32"))
    reference = np.asarray(attributions, dtype="float64")
    if reference.ndim > 2:
        reference = reference.sum(axis=1)
    rng = np.random.default_rng(random_state)

    correlations, overlaps = [], []
    for _repeat in range(max(repeats, 0)):
        noisy = np.clip(values + rng.normal(0.0, noise_std, values.shape),
                        0.0, 1.0).astype("float32")
        again = integrated_gradients(composed, noisy, baseline, target=target,
                                     class_index=class_index,
                                     benign_codes=benign_codes, steps=steps)
        repeated = np.asarray(again["attributions"], dtype="float64")
        if repeated.ndim > 2:
            repeated = repeated.sum(axis=1)
        for row in range(len(values)):
            correlations.append(spearman(np.abs(reference[row]),
                                         np.abs(repeated[row])))
            overlaps.append(top_k_overlap(np.abs(reference[row]),
                                          np.abs(repeated[row]), k))
    if not correlations:
        return {"repeats": 0, "noise_std": noise_std}
    return {
        "repeats": int(repeats), "noise_std": float(noise_std), "top_k": k,
        "mean_spearman": float(np.mean(correlations)),
        "median_spearman": float(np.median(correlations)),
        "mean_top_k_overlap": float(np.mean(overlaps)),
        "comparisons": int(len(correlations)),
    }


# ===========================================================================
# 14.4  EXPLAINING THE ADAPTIVE FUSION, EXACTLY
# ===========================================================================
def fusion_breakdown(rows: Sequence[int], model_names: Sequence[str],
                     probabilities: dict[str, np.ndarray],
                     weights: np.ndarray, reliability: np.ndarray,
                     confidence: np.ndarray, benign_codes: Sequence[int],
                     label_names: dict[int, str],
                     orig_row: Sequence[Any] | None = None) -> pd.DataFrame:
    r"""One row per (flow, model): p_m, r_m, Conf_m, Rel_m, w_m and C_m.

    C_m = w_m r_m, and R(x) = sum_m C_m exactly, because the weights sum to 1:

        sum_m w_m r_m = sum_m w_m (1 - sum_{c in B} p_m(c))
                      = 1 - sum_{c in B} sum_m w_m p_m(c)
                      = 1 - sum_{c in B} P(c) = R(x)

    so the contributions are not an approximation of the risk -- they are the
    risk, split by model.
    """
    rows = np.asarray(rows, dtype="int64")
    weights = np.asarray(weights, dtype="float64")
    reliability = np.asarray(reliability, dtype="float64")
    confidence = np.asarray(confidence, dtype="float64")
    records = []
    for position, row in enumerate(rows):
        for index, name in enumerate(model_names):
            probability = np.asarray(probabilities[name], dtype="float64")[row]
            predicted = int(np.argmax(probability))
            risk = float(fz.attack_risk(probability[None, :], benign_codes)[0])
            records.append({
                "flow": position,
                "Orig_Row": (orig_row[position] if orig_row is not None else row),
                "test_row": int(row),
                "model": name,
                "predicted": predicted,
                "predicted_label": label_names.get(predicted, str(predicted)),
                "p_max": round(float(probability.max()), 8),
                "r_m": round(risk, 8),
                "Conf_m": round(float(confidence[row, index]), 8),
                "Rel_m": round(float(reliability[row, index]), 8),
                "w_m": round(float(weights[row, index]), 8),
                "C_m": round(float(weights[row, index] * risk), 8),
            })
    return pd.DataFrame(records)


def fusion_consistency(breakdown: pd.DataFrame, risk: Sequence[float],
                       rows: Sequence[int],
                       tolerance: float = 1e-6) -> dict[str, Any]:
    """R(x) from Step 13 against the sum of the contributions explained here."""
    risk = np.asarray(risk, dtype="float64")
    rows = np.asarray(rows, dtype="int64")
    totals = (breakdown.groupby("flow")["C_m"].sum()
              .reindex(range(len(rows)), fill_value=np.nan).to_numpy())
    expected = risk[rows]
    difference = np.abs(totals - expected)
    worst = float(np.nanmax(difference)) if len(difference) else 0.0
    return {"max_abs_difference": worst, "tolerance": tolerance,
            "ok": bool(worst <= tolerance),
            "result": "[OK]" if worst <= tolerance else "[FAIL]",
            "per_flow_difference": difference}


def agreement_table(breakdown: pd.DataFrame, fused_class: Sequence[int],
                    average_class: Sequence[int] | None,
                    rows: Sequence[int], label_names: dict[int, str],
                    orig_row: Sequence[Any] | None = None) -> pd.DataFrame:
    """Per flow: who agreed with whom, who dominated, and what fusion changed."""
    rows = np.asarray(rows, dtype="int64")
    fused_class = np.asarray(fused_class)
    records = []
    for position, row in enumerate(rows):
        part = breakdown[breakdown["flow"] == position]
        predicted = part["predicted"].tolist()
        dominant = (part.loc[part["C_m"].idxmax(), "model"]
                    if len(part) else None)
        fused = int(fused_class[row])
        record = {
            "flow": position,
            "Orig_Row": (orig_row[position] if orig_row is not None else row),
            "test_row": int(row),
            "agreement": agreement_status(predicted),
            "dominant_model": dominant,
            "dominant_contribution": (round(float(part["C_m"].max()), 8)
                                      if len(part) else np.nan),
            "fused_class": fused,
            "fused_label": label_names.get(fused, str(fused)),
            "models_agreeing_with_fusion": int(sum(p == fused for p in predicted)),
            "models": len(predicted),
        }
        if average_class is not None:
            simple = int(np.asarray(average_class)[row])
            record["simple_average_class"] = simple
            record["simple_average_label"] = label_names.get(simple, str(simple))
            record["fusion_changed_the_class"] = bool(simple != fused)
        records.append(record)
    return pd.DataFrame(records)


# ===========================================================================
# 14.5  THE PER-ALERT EVIDENCE REPORT
# ===========================================================================
# The rows every report carries, in order.  Checked by the tests, so a row
# cannot quietly disappear -- and nothing outside this list is ever added.
REPORT_ROWS = [
    "Alert ID", "Risk score", "Model 1 Proposed (FS->AE->CNN-LSTM)",
    "Model 2 AE->LSTM", "Model 3 CNN->LSTM", "Fusion", "Attack family",
    "Attack stage", "Flow context", "Source context", "Uncertainty",
    "Recommended action",
]

# Rows this pipeline cannot support, listed so the omission is deliberate and
# testable rather than an oversight.  There is no sequence model, no tokeniser
# and no graph here, so a report must not imply one.
FORBIDDEN_ROWS = [
    "Sequential pattern", "Sequence", "Token", "Graph path", "Graph",
    "Transition consistency", "Transition", "Attack progression",
    "Analyst triage time", "Triage time",
]

MODEL_ROW_TITLES = {
    "proposed": "Model 1 Proposed (FS->AE->CNN-LSTM)",
    "baseline2_ae_lstm": "Model 2 AE->LSTM",
    "baseline3_cnn_lstm": "Model 3 CNN->LSTM",
}


def model_row_title(name: str, position: int) -> str:
    """The report row heading for a model, by its Step 12 name."""
    return MODEL_ROW_TITLES.get(name, "Model %d %s" % (position + 1, name))


def top_feature_sentence(attribution: np.ndarray, feature_names: Sequence[str],
                         unscaled: pd.Series | None, k: int | None = None) -> str:
    """The top-k attributed features, each with its direction and real value.

    The value printed is the ORIGINAL unscaled one, because "Flow Duration =
    0.0021" in min-max space means nothing to an analyst and "Flow Duration =
    184 us" means something.
    """
    k = config.TOP_K_FEATURES if k is None else int(k)
    attribution = np.asarray(attribution, dtype="float64")
    if attribution.ndim > 1:
        attribution = attribution.sum(axis=0)
    pieces = []
    for position in top_k(np.abs(attribution), k):
        name = feature_names[position]
        direction = "raises" if attribution[position] > 0 else "lowers"
        value = None
        if unscaled is not None and name in unscaled.index:
            value = unscaled[name]
        pieces.append("%s = %s (%s, IG %+.4f)"
                      % (name,
                         ("%.6g" % value) if isinstance(value, (int, float,
                                                                np.floating,
                                                                np.integer))
                         else ("n/a" if value is None else str(value)),
                         direction, attribution[position]))
    return "; ".join(pieces) if pieces else "no attribution available"


def source_context(meta: pd.DataFrame, bands: Sequence[str],
                   risk: Sequence[float], fused_class: Sequence[int],
                   label_names: dict[int, str], row: int,
                   window_minutes: int | None = None,
                   limit: int = 8) -> pd.DataFrame:
    """Other TEST-SET alerts from the same Src IP, within the time window.

    Descriptive context only.  No model sees it, and the test set is a random
    20% of the flows, so quiet gaps here mean "not in this subset", never "the
    source went quiet".
    """
    window_minutes = (config.SOURCE_CONTEXT_WINDOW_MIN if window_minutes is None
                      else window_minutes)
    if meta is None or "Src IP" not in meta.columns:
        return pd.DataFrame()
    bands = np.asarray(bands, dtype=object)
    risk = np.asarray(risk, dtype="float64")
    fused_class = np.asarray(fused_class)

    same_source = np.asarray(meta["Src IP"] == meta["Src IP"].iloc[row])
    alert = np.isin(bands, [BAND_ATTACK, BAND_SUSPICIOUS])
    candidates = np.flatnonzero(same_source & alert)
    candidates = candidates[candidates != row]
    if not len(candidates):
        return pd.DataFrame()

    stamps = None
    for column in ("Timestamp_dt", "Timestamp"):
        if column in meta.columns:
            stamps = pd.to_datetime(meta[column], errors="coerce")
            if stamps.notna().any():
                break
    if stamps is not None and pd.notna(stamps.iloc[row]):
        delta = (stamps.iloc[candidates] - stamps.iloc[row]).abs()
        inside = delta <= pd.Timedelta(minutes=window_minutes)
        candidates = candidates[np.asarray(inside.fillna(False))]
    if not len(candidates):
        return pd.DataFrame()

    frame = pd.DataFrame({
        "Orig_Row": (meta["Orig_Row"].iloc[candidates].to_numpy()
                     if "Orig_Row" in meta.columns else candidates),
        "class": [label_names.get(int(c), str(c))
                  for c in fused_class[candidates]],
        "band": bands[candidates],
        "R": np.round(risk[candidates], 6),
    })
    if stamps is not None:
        frame.insert(1, "time", stamps.iloc[candidates].to_numpy())
        frame = frame.sort_values("time", kind="stable")
    return frame.head(limit).reset_index(drop=True)


def build_alert_report(position: int, row: int, context: dict[str, Any]
                       ) -> dict[str, Any]:
    """One alert's evidence table, plus the flat record saved to CSV.

    `context` carries everything already computed for the whole batch; this
    function only assembles, so the report cannot disagree with the tables.
    """
    label_names = context["label_names"]
    model_names = list(context["model_names"])
    breakdown = context["breakdown"]
    part = breakdown[breakdown["flow"] == position]
    meta = context.get("meta")
    tau, tau_s = context["tau"], context.get("tau_s")
    risk = float(np.asarray(context["risk"])[row])
    band = str(np.asarray(context["bands"], dtype=object)[row])
    fused = int(np.asarray(context["fused_class"])[row])
    fused_label = label_names.get(fused, str(fused))
    fused_probability = float(np.asarray(context["fused_probabilities"])[row].max())

    agreement = agreement_status(part["predicted"].tolist())
    completeness_ok = bool(context["completeness_ok"].get(position, True))
    confidence = confidence_label(fused_probability, agreement, completeness_ok)
    orig = (meta["Orig_Row"].iloc[row] if meta is not None
            and "Orig_Row" in meta.columns else row)
    alert_id = "A_%s" % orig

    rows: list[dict[str, str]] = [
        {"Evidence component": "Alert ID",
         "Detail": "%s | decision band: %s" % (alert_id, band),
         "Confidence": confidence},
        {"Evidence component": "Risk score",
         "Detail": ("R = %.3f vs tau = %.3f%s" %
                    (risk, tau,
                     "" if tau_s is None else " (suspicious band from tau_s = "
                                              "%.3f)" % tau_s)),
         "Confidence": confidence},
    ]

    for index, name in enumerate(model_names):
        line = part[part["model"] == name]
        if not len(line):
            continue
        line = line.iloc[0]
        attribution = context["attributions"].get(name)
        features = context["feature_names"].get(name, [])
        unscaled = context.get("unscaled_rows", {}).get(row)
        detail = ("P = %.2f (%s) | w = %.2f | contribution w*r = %.2f | "
                  "r = %.3f | top-%d features: %s"
                  % (line["p_max"], line["predicted_label"], line["w_m"],
                     line["C_m"], line["r_m"], config.TOP_K_FEATURES,
                     top_feature_sentence(
                         attribution[position] if attribution is not None
                         else np.zeros(len(features)), features, unscaled)))
        rows.append({"Evidence component": model_row_title(name, index),
                     "Detail": detail,
                     "Confidence": ("High" if line["p_max"] >= config.CONF_HIGH_PROB
                                    else "Medium" if line["p_max"]
                                    >= config.CONF_MEDIUM_PROB else "Low")})

    reliabilities = "; ".join(
        "%s %.3f" % (r["model"], r["Rel_m"]) for _i, r in part.iterrows())
    dominant = (part.loc[part["C_m"].idxmax(), "model"] if len(part) else "n/a")
    total = float(part["C_m"].sum())
    context_label = context.get("context_labels", {}).get(row, "n/a")
    rows.append({
        "Evidence component": "Fusion",
        "Detail": ("context = %s | reliability: %s | dominant model: %s | "
                   "models %s | sum of contributions = %.3f (R = %.3f)"
                   % (context_label, reliabilities, dominant, agreement,
                      total, risk)),
        "Confidence": confidence})
    rows.append({"Evidence component": "Attack family",
                 "Detail": "%s (fused predicted class)" % fused_label,
                 "Confidence": confidence})
    rows.append({
        "Evidence component": "Attack stage",
        "Detail": ("%s -- label-to-tactic mapping, not an inferred attack "
                   "progression" % attack_stage(fused_label)),
        "Confidence": confidence})

    flow_context = "identifiers unavailable"
    src_ip = dst_ip = "unknown"
    if meta is not None:
        line = meta.iloc[row]
        src_ip = line.get("Src IP", "unknown")
        dst_ip = line.get("Dst IP", "unknown")
        flow_context = ("%s:%s -> %s:%s | Protocol %s | %s | capture %s | "
                        "identifiers are context only, never model features"
                        % (src_ip, line.get("Src Port", "?"), dst_ip,
                           line.get("Dst Port", "?"), line.get("Protocol", "?"),
                           line.get("Timestamp", "?"),
                           line.get("Source_Capture", "?")))
    rows.append({"Evidence component": "Flow context", "Detail": flow_context,
                 "Confidence": confidence})

    nearby = pd.DataFrame()
    if config.SOURCE_CONTEXT and meta is not None:
        nearby = source_context(meta, context["bands"], context["risk"],
                                context["fused_class"], label_names, row)
    if len(nearby):
        listed = "; ".join(
            "%s at %s (R %.3f, %s)"
            % (r["class"], r.get("time", "?"), r["R"], r["band"])
            for _i, r in nearby.iterrows())
    else:
        listed = "no other alert from this source in the window"
    rows.append({
        "Evidence component": "Source context",
        "Detail": ("%d other alert(s) from %s within %d minutes: %s -- "
                   "descriptive context from the test subset; not used by the "
                   "models; the random split means some flows in between are "
                   "not shown"
                   % (len(nearby), src_ip, config.SOURCE_CONTEXT_WINDOW_MIN,
                      listed)),
        "Confidence": "Medium" if len(nearby) else "Low"})

    notes = uncertainty_notes(fused_probability, agreement, completeness_ok,
                              band, context.get("training_rows", {}).get(fused))
    rows.append({"Evidence component": "Uncertainty",
                 "Detail": "; ".join(notes) if notes else "none noted",
                 "Confidence": confidence})
    rows.append({"Evidence component": "Recommended action",
                 "Detail": recommended_action(fused_label, src_ip, dst_ip),
                 "Confidence": confidence})

    table = pd.DataFrame(rows)
    flat: dict[str, Any] = {
        "alert_id": alert_id, "Orig_Row": orig, "test_row": int(row),
        "band": band, "R": round(risk, 6), "tau": round(float(tau), 6),
        "tau_s": (None if tau_s is None else round(float(tau_s), 6)),
        "fused_class": fused, "fused_label": fused_label,
        "fused_max_probability": round(fused_probability, 6),
        "agreement": agreement, "dominant_model": dominant,
        "sum_contributions": round(total, 6),
        "completeness_ok": completeness_ok, "confidence": confidence,
        "attack_stage": attack_stage(fused_label),
        "context": context_label,
        "src_ip": src_ip, "dst_ip": dst_ip,
        "source_context_alerts": int(len(nearby)),
        "uncertainty": "; ".join(notes) if notes else "none noted",
        "recommended_action": recommended_action(fused_label, src_ip, dst_ip),
    }
    for component, detail in zip(table["Evidence component"], table["Detail"]):
        flat["row: %s" % component] = detail
    return {"table": table, "flat": flat, "alert_id": alert_id,
            "fused_label": fused_label, "band": band}


def report_as_text(report: dict[str, Any]) -> str:
    """The same table as plain text, for a terminal or a .md file."""
    table = report["table"]
    width = max(len(str(v)) for v in table["Evidence component"])
    lines = ["ALERT %s  [%s]" % (report["alert_id"], report["band"]),
             "-" * 78]
    for _i, row in table.iterrows():
        lines.append("%-*s | %s  [%s]"
                     % (width, row["Evidence component"], row["Detail"],
                        row["Confidence"]))
    return "\n".join(lines)


def examples_markdown(reports: Sequence[dict[str, Any]]) -> str:
    """One example alert per attack class, as a markdown document."""
    parts = ["# Step 14 -- example alert evidence reports", "",
             "One alert per fused attack class, generated from the computed "
             "tables. Every number here is read off `alert_explanations.csv` "
             "and `explanations.csv`.", ""]
    for report in reports:
        parts += ["## %s -- %s (%s)" % (report["alert_id"],
                                        report["fused_label"], report["band"]),
                  "",
                  "| Evidence component | Detail | Confidence |",
                  "|---|---|---|"]
        for _i, row in report["table"].iterrows():
            detail = str(row["Detail"]).replace("|", "\\|")
            parts.append("| %s | %s | %s |" % (row["Evidence component"],
                                               detail, row["Confidence"]))
        parts += ["", "```", report_as_text(report), "```", ""]
    return "\n".join(parts)


# ===========================================================================
# 14.1  LOADING -- Step 1-13 outputs only, nothing retrained
# ===========================================================================
# Stability re-attributes every selected flow STABILITY_N_REPEATS times, which
# is the most expensive check here.  It is measured on at most this many flows
# and the number used is reported, so the figure is never quietly partial.
STABILITY_MAX_FLOWS = 50


def _missing(path, what: str, step: str) -> None:
    raise ExplainabilityError(
        "%s is missing:\n    %s\nRun %s first. Step 14 never retrains a model."
        % (what, path, step))


def load_composed(name: str, verbose: bool = False) -> dict[str, Any]:
    """The composed model for one Step 12 configuration, on ORIGINAL features.

    Mirrors fusion.load_model_bundle, but keeps the encoder and the raw feature
    matrix apart instead of encoding up front -- the whole point here is to put
    the encoder INSIDE the differentiated function.
    """
    from . import experiments as xp

    spec = fz._spec_for(name)
    folder12 = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder12 is None:
        raise ExplainabilityError("the Step 12 output folder is not usable "
                                  "here -- %s"
                                  % config.why_unusable(config.STEP12_OUTPUT_DIR))
    folder = folder12 / name
    keras = _keras()

    model_path = folder / "model.keras"
    if not model_path.exists():
        from . import models as md
        if spec.get("reuse_step7_10") and md.step10_outputs_exist():
            folder10 = config.usable_output_dir(config.STEP10_OUTPUT_DIR)
            model_path = folder10 / config.STEP10_FILES["model"]
        else:
            _missing(model_path, "the trained model for %r" % name,
                     "Step 12 (set OVERWRITE_STEP12 = True)")
    classifier = keras.models.load_model(model_path)

    source = xp.load_feature_source(spec["feature_source"], verbose=False)
    X_train, X_test = source["X_train"], source["X_test"]

    encoder = None
    if spec["use_ae"]:
        if spec.get("reuse_step7_10"):
            from . import autoencoder as ae
            folder7 = config.usable_output_dir(config.STEP7_OUTPUT_DIR)
            encoder_path = (folder7 / config.STEP7_FILES["encoder"]
                            if folder7 else None)
            if encoder_path is None or not encoder_path.exists():
                _missing(encoder_path, "the Step 7 encoder", "Step 7")
            _encode, encoder, order = ae.load_encoder(verbose=False)
        else:
            encoder_path = folder / "encoder.keras"
            if not encoder_path.exists():
                _missing(encoder_path, "the baseline autoencoder for %r" % name,
                         "Step 12 (set OVERWRITE_STEP12 = True)")
            encoder = keras.models.load_model(encoder_path)
            saved = json.loads((folder / "ae_config.json").read_text(
                encoding="utf-8"))
            order = list(saved["features"])
        X_train, X_test = X_train[order], X_test[order]

    features = list(X_test.columns)
    composed = build_composed_model(classifier, encoder=encoder,
                                    n_features=len(features),
                                    name="composed_%s" % name)
    saved_probabilities = None
    saved_path = folder / "predictions.csv"
    if saved_path.exists():
        saved = pd.read_csv(saved_path, low_memory=False)
        columns = [c for c in saved.columns if c.startswith("p_")]
        if columns:
            saved_probabilities = saved[columns].to_numpy(dtype="float64")
    if verbose:
        print("  loaded %-24s %d features -> %s"
              % (name, len(features),
                 "encoder -> classifier" if encoder is not None
                 else "classifier"))
    return {"name": name, "spec": spec, "composed": composed,
            "classifier": classifier, "encoder": encoder,
            "X_train": X_train, "X_test": X_test, "features": features,
            "saved_probabilities": saved_probabilities,
            "model_path": str(model_path)}


def load_step14_inputs(verbose: bool = True) -> dict[str, Any]:
    """Everything Step 14 reads, with the alignment checks that make it usable."""
    model_names = fz.check_fusion_models(verbose=False)
    loaded = ev.load_evaluation_inputs(verbose=False)
    y_test, meta_test = loaded["y_test"], loaded.get("meta_test")
    label_names = ev.label_names_from_mapping(loaded["label_mapping"])
    benign_codes = ev.benign_codes_from(y_test, label_names)

    folder5 = config.usable_output_dir(config.STEP5_OUTPUT_DIR)
    if folder5 is None:
        raise ExplainabilityError("the Step 5 output folder is not usable here "
                                  "-- %s"
                                  % config.why_unusable(config.STEP5_OUTPUT_DIR))
    unscaled_path = folder5 / config.STEP5_FILES["X_test_unscaled"]
    if not unscaled_path.exists():
        _missing(unscaled_path, "the unscaled test features", "Step 5")
    X_test_unscaled = pd.read_csv(unscaled_path, low_memory=False)

    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    y_train = pd.read_csv(folder8 / config.STEP8_FILES["y_train"],
                          low_memory=False)

    if not fz.step13_outputs_exist():
        _missing(config.STEP13_OUTPUT_DIR, "the Step 13 fusion outputs",
                 "Step 13 (set OVERWRITE_STEP13 = True)")
    step13 = fz.load_step13(verbose=False)
    for key in ("fused_predictions", "threshold"):
        if key not in step13:
            _missing(config.STEP13_OUTPUT_DIR / config.STEP13_FILES[key],
                     "the Step 13 %s" % key, "Step 13")
    fused = step13["fused_predictions"]
    tau = float(step13["threshold"]["tau"])

    n_classes = int(max(int(y_train[TARGET_COLUMN].max()),
                        int(y_test[TARGET_COLUMN].max())) + 1)
    class_order = [label_names.get(c, str(c)) for c in range(n_classes)]

    checks: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        checks.append({"check": check, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    same_rows = len(fused) == len(y_test)
    add("Step 13 predictions cover the Step 5 test set", same_rows,
        "%s rows" % f"{len(fused):,}" if same_rows
        else "%d vs %d" % (len(fused), len(y_test)))
    if not same_rows:
        raise ExplainabilityError(
            "Step 13's fused_predictions.csv has %d rows but the test set has "
            "%d. Re-run Step 13." % (len(fused), len(y_test)))

    if meta_test is not None and "Orig_Row" in meta_test.columns:
        aligned = (fused["Orig_Row"].tolist()
                   == meta_test["Orig_Row"].tolist())
        add("Orig_Row matches the Step 5 metadata row for row", aligned,
            "%s rows in the same order" % f"{len(fused):,}" if aligned
            else "the order differs")
        if not aligned:
            raise ExplainabilityError(
                "Step 13's fused_predictions.csv and Step 5's meta_test.csv do "
                "not list the same rows in the same order, so an alert cannot "
                "be tied to a flow.")

    truth = (fused["y_true"].tolist() == y_test[TARGET_COLUMN].tolist()
             if "y_true" in fused.columns else True)
    add("the ground truth agrees with Step 5", bool(truth),
        "%s rows" % f"{len(fused):,}" if truth else "y_true differs")

    probability_columns = ["P_%s" % name for name in class_order]
    present = [c for c in probability_columns if c in fused.columns]
    add("one fused probability column per class",
        len(present) == n_classes,
        "%d of %d: %s" % (len(present), n_classes, ", ".join(class_order)))
    if len(present) != n_classes:
        raise ExplainabilityError(
            "Step 13 saved %d fused probability columns for %d classes; the "
            "class order has changed since Step 13 ran. Re-run Step 13."
            % (len(present), n_classes))

    missing_models = [name for name in model_names
                      if "w_%s" % name not in fused.columns]
    add("every fused model has its weight column", not missing_models,
        ", ".join(model_names) if not missing_models
        else "missing %s" % missing_models)
    if missing_models:
        raise ExplainabilityError(
            "Step 13's fused_predictions.csv has no weight column for %s. "
            "FUSION_MODELS changed since Step 13 ran; re-run Step 13."
            % missing_models)

    table = pd.DataFrame(checks)
    if verbose:
        show_table(table, note="Everything is read from Steps 1-13 and aligned "
                               "on `Orig_Row`. A `[FAIL]` here means an alert "
                               "could be tied to the wrong flow, so it raises.")
    return {
        "model_names": model_names, "y_test": y_test, "y_train": y_train,
        "meta_test": meta_test, "label_names": label_names,
        "benign_codes": benign_codes, "n_classes": n_classes,
        "class_order": class_order, "X_test_unscaled": X_test_unscaled,
        "X_train_step5": loaded.get("X_train"), "X_test_step5": loaded.get("X_test"),
        "step13": step13, "fused": fused, "tau": tau, "alignment": table,
    }


def step13_matrices(fused: pd.DataFrame, model_names: Sequence[str],
                    class_order: Sequence[str]) -> dict[str, np.ndarray]:
    """The weights, reliabilities, confidences and fused risk Step 13 saved.

    Read back rather than recomputed on purpose: the report then explains the
    numbers that actually produced the alert, and `R = sum C_m` becomes a check
    on Step 13's own file instead of a restatement of this module's arithmetic.
    """
    def column(prefix: str) -> np.ndarray:
        return np.column_stack([fused["%s%s" % (prefix, name)].to_numpy(
            dtype="float64") for name in model_names])

    return {
        "weights": column("w_"),
        "reliability": (column("Rel_") if "Rel_%s" % model_names[0] in fused.columns
                        else np.full((len(fused), len(model_names)), np.nan)),
        "confidence": column("Conf_"),
        "model_risk": column("r_"),
        "risk": fused["R_t"].to_numpy(dtype="float64"),
        "fused_class": fused["fused_class"].to_numpy(dtype="int64"),
        "fused_probabilities": np.column_stack(
            [fused["P_%s" % name].to_numpy(dtype="float64")
             for name in class_order]),
        "context": fused["context"].to_numpy(dtype=object),
    }


# ===========================================================================
# 14.7  CHARTS
# ===========================================================================
# The palette, the surface colours and the save/style helpers are Step 13's:
# one visual system across the whole notebook, and no second copy to drift.
_SERIES = fz._SERIES
_SEQ_RAMP = fz._SEQ_RAMP
_pyplot = fz._pyplot
_finish = fz._finish
_style = fz._style


def _label_bars(ax, bars, values, fmt="%.3f", inside=False):
    """Every bar carries its value, so no one has to read it off an axis."""
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        height = bar.get_height()
        ax.annotate(fmt % value,
                    (bar.get_x() + bar.get_width() / 2,
                     height - abs(height) * 0.5 if inside else height),
                    ha="center",
                    va="center" if inside else ("bottom" if height >= 0 else "top"),
                    fontsize=8, color="white" if inside else dl._INK_SOFT,
                    xytext=(0, 0 if inside else 3), textcoords="offset points")


def plot_global_importance(table: pd.DataFrame, path, top: int = 8,
                           verbose: bool = True):
    """The features each model leaned on most, one small panel per model."""
    plt = _pyplot(verbose)
    if plt is None or table.empty:
        return None
    overall = table[(table["scope"] == "all explained flows")
                    & (table["target"] == TARGET_RISK)]
    if overall.empty:
        overall = table[table["scope"] == "all explained flows"]
    models = list(dict.fromkeys(overall["model"]))
    if not models:
        return None

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.0),
                             dpi=150)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(dl._SURFACE)
    for index, (ax, model) in enumerate(zip(axes, models)):
        _style(ax)
        part = (overall[overall["model"] == model]
                .sort_values("mean_abs_ig", ascending=True).tail(top))
        positions = np.arange(len(part), dtype="float64")
        bars = ax.barh(positions, part["mean_abs_ig"].to_numpy(),
                       color=_SERIES[index % len(_SERIES)], height=0.68)
        ax.set_yticks(positions)
        ax.set_yticklabels(part["feature"], fontsize=8.5, color=dl._INK_SOFT)
        ax.xaxis.grid(True, color=dl._GRID, linewidth=0.8)
        ax.yaxis.grid(False)
        for bar, value in zip(bars, part["mean_abs_ig"].to_numpy()):
            # %.3g, not %.4f: an attribution can be 1e-5, and a column of
            # "0.0000" labels tells a reader nothing
            ax.annotate("%.3g" % value,
                        (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        ha="left", va="center", fontsize=8, color=dl._INK_SOFT,
                        xytext=(3, 0), textcoords="offset points")
        ax.set_title(model, color=dl._INK, fontsize=11, pad=10)
        ax.set_xlabel("mean |IG| on the attack risk", color=dl._INK_SOFT,
                      fontsize=9.5)
        ax.set_xlim(0, float(part["mean_abs_ig"].max()) * 1.25 or 1.0)
    fig.suptitle("What moved each model, averaged over the explained flows",
                 color=dl._INK, fontsize=12.5, y=0.99)
    return _finish(fig, path, verbose)


def plot_class_feature_heatmap(table: pd.DataFrame, model: str, path,
                               top: int = 10, verbose: bool = True):
    """Predicted class x feature, mean |IG| for one model.  One measure, one hue."""
    plt = _pyplot(verbose)
    if plt is None or table.empty:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    part = table[(table["model"] == model) & (table["target"] == TARGET_RISK)
                 & (table["scope"].str.startswith("predicted: "))]
    if part.empty:
        return None
    best = (table[(table["model"] == model)
                  & (table["scope"] == "all explained flows")
                  & (table["target"] == TARGET_RISK)]
            .nlargest(top, "mean_abs_ig")["feature"].tolist())
    grid = (part[part["feature"].isin(best)]
            .pivot(index="scope", columns="feature", values="mean_abs_ig")
            .reindex(columns=best))
    if grid.empty:
        return None

    values = grid.to_numpy(dtype="float64")
    cmap = LinearSegmentedColormap.from_list("mscad_blue", _SEQ_RAMP)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.0 * grid.shape[1] + 4.0),
                                    max(4.0, 0.62 * grid.shape[0] + 2.4)),
                           dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    image = ax.imshow(values, cmap=cmap, aspect="auto",
                      vmin=0.0, vmax=float(np.nanmax(values)) or 1.0)
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_xticklabels(grid.columns, rotation=28, ha="right", fontsize=8.5,
                       color=dl._INK_SOFT)
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_yticklabels([s.replace("predicted: ", "") for s in grid.index],
                       fontsize=9, color=dl._INK_SOFT)
    limit = float(np.nanmax(values)) or 1.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            ax.text(column, row, "%.2g" % value, ha="center", va="center",
                    fontsize=7.5,
                    color="white" if value > 0.55 * limit else dl._INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("mean |IG|", color=dl._INK_SOFT, fontsize=9.5)
    bar.ax.tick_params(colors=dl._INK_SOFT, labelsize=8)
    ax.set_title("Which features drive which predicted class -- %s" % model,
                 color=dl._INK, fontsize=12, pad=14)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return _finish(fig, path, verbose)


def plot_signed_box(attributions: np.ndarray, feature_names: Sequence[str],
                    path, top: int = 8, verbose: bool = True):
    """Signed IG spread for the most important features: which way, how reliably."""
    plt = _pyplot(verbose)
    if plt is None or not len(attributions):
        return None
    values = np.asarray(attributions, dtype="float64")
    if values.ndim > 2:
        values = values.sum(axis=1)
    order = top_k(np.abs(values).mean(axis=0), top)
    if not order:
        return None

    fig, ax = plt.subplots(figsize=(max(8.5, 1.2 * len(order) + 3.0), 5.2),
                           dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    box = ax.boxplot([values[:, i] for i in order], patch_artist=True,
                     widths=0.6, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor(_SERIES[0])
        patch.set_alpha(0.75)
        patch.set_edgecolor(_SERIES[0])
    for piece in ("whiskers", "caps"):
        for line in box[piece]:
            line.set_color(dl._GRID)
    for line in box["medians"]:
        line.set_color(dl._INK)
        line.set_linewidth(1.6)
    ax.axhline(0.0, color=dl._INK_SOFT, linewidth=1.0, linestyle="--")
    ax.set_xticklabels([feature_names[i] for i in order], rotation=24,
                       ha="right", fontsize=8.5, color=dl._INK_SOFT)
    ax.set_ylabel("signed IG on the attack risk", color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Above the line the feature RAISES the risk, below it lowers it",
                 color=dl._INK, fontsize=12, pad=14)
    return _finish(fig, path, verbose)


def plot_ig_vs_perturbation(attributions: np.ndarray, drops: np.ndarray, path,
                            verbose: bool = True, sample: int = 4000):
    """One point per (flow, feature): the gradient method against the direct one."""
    plt = _pyplot(verbose)
    if plt is None or not len(attributions):
        return None
    values = np.asarray(attributions, dtype="float64")
    if values.ndim > 2:
        values = values.sum(axis=1)
    x = np.abs(values).ravel()
    y = np.asarray(drops, dtype="float64").ravel()
    if len(x) > sample:
        rng = np.random.default_rng(config.EXPLAIN_RANDOM_STATE)
        pick = rng.choice(len(x), size=sample, replace=False)
        x, y = x[pick], y[pick]

    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    ax.xaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.scatter(x, y, s=9, alpha=0.35, color=_SERIES[0], linewidths=0)
    ax.axhline(0.0, color=dl._GRID, linewidth=1.0)
    ax.set_xlabel("|IG| (gradient method)", color=dl._INK_SOFT, fontsize=10)
    ax.set_ylabel("risk drop when the feature is set to its baseline",
                  color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Spearman %.3f over %s (flow, feature) pairs"
                 % (spearman(x, y), f"{len(x):,}"),
                 color=dl._INK, fontsize=12, pad=14)
    return _finish(fig, path, verbose)


def plot_deletion_check(results: dict[str, dict[str, Any]], path,
                        verbose: bool = True):
    """Top-K removal against random removal, per model."""
    plt = _pyplot(verbose)
    if plt is None or not results:
        return None
    models = list(results)
    positions = np.arange(len(models), dtype="float64")
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(7.5, 2.4 * len(models) + 3.0), 5.2),
                           dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    first = [results[m]["mean_drop_top_k"] for m in models]
    second = [results[m]["mean_drop_random_k"] for m in models]
    bars_a = ax.bar(positions - width / 2 - 0.01, first, width,
                    color=_SERIES[0], label="top-K by |IG|")
    bars_b = ax.bar(positions + width / 2 + 0.01, second, width,
                    color=_SERIES[1], label="K random features")
    _label_bars(ax, bars_a, first, "%.3g")
    _label_bars(ax, bars_b, second, "%.3g")
    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=12, ha="right", fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_ylabel("mean drop in attack risk", color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Removing the attributed features should cost the model more",
                 color=dl._INK, fontsize=12, pad=14)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return _finish(fig, path, verbose)


def plot_contributions(breakdown: pd.DataFrame, reports: Sequence[dict[str, Any]],
                       path, verbose: bool = True):
    """C_m stacked per example alert: the bar's height IS the risk."""
    plt = _pyplot(verbose)
    if plt is None or breakdown.empty or not reports:
        return None
    flows = [int(breakdown[breakdown["Orig_Row"] == r["flat"]["Orig_Row"]]
                 ["flow"].iloc[0]) for r in reports
             if (breakdown["Orig_Row"] == r["flat"]["Orig_Row"]).any()]
    if not flows:
        return None
    labels = ["%s\n%s" % (r["alert_id"], r["fused_label"]) for r in reports]
    models = list(dict.fromkeys(breakdown["model"]))
    positions = np.arange(len(flows), dtype="float64")

    fig, ax = plt.subplots(figsize=(max(8.0, 1.9 * len(flows) + 3.0), 5.4),
                           dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    bottom = np.zeros(len(flows))
    for index, model in enumerate(models):
        values = np.array([
            float(breakdown[(breakdown["flow"] == flow)
                            & (breakdown["model"] == model)]["C_m"].sum())
            for flow in flows])
        bars = ax.bar(positions, values, 0.6, bottom=bottom,
                      color=_SERIES[index % len(_SERIES)], label=model)
        for bar, value in zip(bars, values):
            if value > 0.04:
                ax.annotate("%.2f" % value,
                            (bar.get_x() + bar.get_width() / 2,
                             bar.get_y() + value / 2),
                            ha="center", va="center", fontsize=8, color="white")
        bottom += values
    for position, total in zip(positions, bottom):
        ax.annotate("R = %.3f" % total, (position, total), ha="center",
                    va="bottom", fontsize=9, color=dl._INK,
                    xytext=(0, 4), textcoords="offset points")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5, color=dl._INK_SOFT)
    ax.set_ylabel("contribution to the fused risk (C_m = w_m r_m)",
                  color=dl._INK_SOFT, fontsize=10)
    ax.set_ylim(0, max(1.0, float(bottom.max()) * 1.18))
    ax.set_title("Who produced the risk, alert by alert", color=dl._INK,
                 fontsize=12, pad=14)
    legend = ax.legend(frameon=False, fontsize=8.5, loc="upper right",
                       ncol=len(models))
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return _finish(fig, path, verbose)


def plot_risk_bands(risk: Sequence[float], bands: Sequence[str], tau: float,
                    tau_s: float | None, path, verbose: bool = True):
    """Where the test flows sit, with both decision lines marked."""
    plt = _pyplot(verbose)
    if plt is None:
        return None
    risk = np.asarray(risk, dtype="float64")
    bands = np.asarray(bands, dtype=object)

    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    edges = np.linspace(0.0, 1.0, 41)
    for index, name in enumerate((BAND_BENIGN, BAND_SUSPICIOUS, BAND_ATTACK)):
        selected = risk[bands == name]
        if not len(selected):
            continue
        ax.hist(selected, bins=edges, color=_SERIES[index % len(_SERIES)],
                alpha=0.78, label="%s (%s flows)" % (name, f"{len(selected):,}"))
    ax.set_yscale("log")
    for value, text in ((tau, "tau = %.3f" % tau),
                        (tau_s, None if tau_s is None else "tau_s = %.3f" % tau_s)):
        if value is None or text is None:
            continue
        ax.axvline(value, color=dl._INK, linewidth=1.8, linestyle="--")
        ax.annotate(text, (value, ax.get_ylim()[1]), ha="left", va="top",
                    fontsize=9.5, color=dl._INK, xytext=(5, -6),
                    textcoords="offset points")
    ax.set_xlabel("fused risk R", color=dl._INK_SOFT, fontsize=10)
    ax.set_ylabel("flows (log scale)", color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Test flows by decision band", color=dl._INK, fontsize=12, pad=14)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper center")
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return _finish(fig, path, verbose)


def plot_temporal_attribution(attributions: np.ndarray, path,
                              verbose: bool = True):
    """Mean |IG| per time step -- only meaningful in time_window mode."""
    plt = _pyplot(verbose)
    if plt is None:
        return None
    values = np.asarray(attributions, dtype="float64")
    if values.ndim != 3:
        if verbose:
            print("  [SKIP]  INPUT_MODE is %r, so a flow has no time steps and "
                  "there is no temporal attribution to draw."
                  % config.INPUT_MODE)
        return None
    per_step = np.abs(values).sum(axis=2).mean(axis=0)
    positions = np.arange(len(per_step), dtype="float64")

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    bars = ax.bar(positions, per_step, 0.64, color=_SERIES[0])
    _label_bars(ax, bars, per_step, "%.3g")
    ax.set_xticks(positions)
    ax.set_xticklabels(["t-%d" % (len(per_step) - 1 - i)
                        for i in range(len(per_step))], fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_ylabel("mean |IG| summed over features", color=dl._INK_SOFT,
                  fontsize=10)
    ax.set_title("Which step of the window carried the evidence",
                 color=dl._INK, fontsize=12, pad=14)
    return _finish(fig, path, verbose)


# ===========================================================================
# 14.7  SAVE, VERIFY, RUN
# ===========================================================================
def explanations_frame(rows: Sequence[int], orig_row: Sequence[Any],
                       attributions: dict[str, dict[str, np.ndarray]],
                       feature_names: dict[str, Sequence[str]],
                       drops: dict[str, np.ndarray] | None = None
                       ) -> pd.DataFrame:
    """Long format: one row per (flow, model, feature, target).

    Long rather than wide because the three models have different feature sets,
    and a wide table would be mostly empty columns.
    """
    records = []
    drops = drops or {}
    for model, by_target in attributions.items():
        names = list(feature_names[model])
        for target, values in by_target.items():
            matrix = np.asarray(values, dtype="float64")
            if matrix.ndim > 2:
                matrix = matrix.sum(axis=1)
            drop = drops.get(model) if target == TARGET_RISK else None
            for position in range(len(matrix)):
                ranks = {int(i): place + 1 for place, i
                         in enumerate(top_k(np.abs(matrix[position]),
                                            len(names)))}
                for index, feature in enumerate(names):
                    records.append({
                        "Orig_Row": orig_row[position],
                        "test_row": int(rows[position]),
                        "model": model, "target": target, "feature": feature,
                        "ig": round(float(matrix[position, index]), 8),
                        "abs_rank": ranks.get(index, len(names)),
                        "perturbation_drop": (None if drop is None
                                              else round(float(
                                                  drop[position, index]), 8)),
                    })
    return pd.DataFrame(records)


def evidence_frame(rows: Sequence[int], context: dict[str, Any]) -> pd.DataFrame:
    """One row per EXPLAINED flow: what was selected, and why it qualified."""
    rows = np.asarray(rows, dtype="int64")
    meta = context.get("meta")
    label_names = context["label_names"]
    risk = np.asarray(context["risk"], dtype="float64")
    bands = np.asarray(context["bands"], dtype=object)
    fused_class = np.asarray(context["fused_class"])
    y_true = np.asarray(context["y_true"])
    frame = pd.DataFrame({
        "Orig_Row": (meta["Orig_Row"].to_numpy()[rows] if meta is not None
                     and "Orig_Row" in meta.columns else rows),
        "test_row": rows,
        "band": bands[rows],
        "R": np.round(risk[rows], 8),
        "fused_class": fused_class[rows],
        "fused_label": [label_names.get(int(c), str(c))
                        for c in fused_class[rows]],
        "true_label": [label_names.get(int(c), str(c)) for c in y_true[rows]],
        "correct": fused_class[rows] == y_true[rows],
        "context": np.asarray(context["context_label_column"], dtype=object)[rows],
        "fused_max_probability": np.round(
            np.asarray(context["fused_probabilities"])[rows].max(axis=1), 8),
    })
    if meta is not None:
        for column in ("Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol",
                       "Timestamp", "Source_Capture"):
            if column in meta.columns:
                frame[column] = meta[column].to_numpy()[rows]
    return frame


def verify_saved_step14(folder: Path, out: dict[str, Any],
                        verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """Read the saved files back and check them.  Nothing is trusted from memory."""
    names = config.STEP14_FILES
    checks: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        checks.append({"check": check, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    expected = int(len(out["explained_rows"]))
    path = folder / names["evidence"]
    if path.exists():
        evidence = pd.read_csv(path, low_memory=False)
        add("evidence_records.csv covers every explained flow",
            len(evidence) == expected,
            "%d rows, expected %d" % (len(evidence), expected))
        add("no NaN in the evidence risk column",
            bool(evidence["R"].notna().all()),
            "%d rows checked" % len(evidence))
    else:
        add("evidence_records.csv covers every explained flow", False,
            "not written")

    path = folder / names["alerts"]
    if path.exists():
        alerts = pd.read_csv(path, low_memory=False)
        add("alert_explanations.csv holds one row per alert",
            len(alerts) == int(out["n_alerts"]),
            "%d rows, expected %d" % (len(alerts), int(out["n_alerts"])))
        if len(alerts):
            gap = float((alerts["sum_contributions"] - alerts["R"]).abs().max())
            add("R equals the sum of the model contributions", gap <= 1e-5,
                "max |R - sum C_m| = %.3e (tolerance 1e-5 after rounding)" % gap)
            missing_rows = [row for row in REPORT_ROWS
                            if "row: %s" % row not in alerts.columns]
            add("every report row is present", not missing_rows,
                "%d rows" % len(REPORT_ROWS) if not missing_rows
                else "missing %s" % missing_rows)
            forbidden = [c for c in alerts.columns
                         if any(c.lower().startswith("row: %s" % f.lower())
                                for f in FORBIDDEN_ROWS)]
            add("no sequence, token or graph row is claimed", not forbidden,
                "none" if not forbidden else "found %s" % forbidden)
            add("no NaN in the risk or confidence columns",
                bool(alerts["R"].notna().all()
                     and alerts["confidence"].notna().all()),
                "%d alerts checked" % len(alerts))
    else:
        add("alert_explanations.csv holds one row per alert", False,
            "not written")

    path = folder / names["explanations"]
    if path.exists():
        explanations = pd.read_csv(path, low_memory=False)
        add("explanations.csv has no NaN attribution",
            bool(explanations["ig"].notna().all()),
            "%s attribution rows" % f"{len(explanations):,}")
        add("explanations.csv covers both targets",
            set(explanations["target"]) == set(TARGETS),
            ", ".join(sorted(set(explanations["target"]))))
    else:
        add("explanations.csv has no NaN attribution", False, "not written")

    path = folder / names["thresholds"]
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        same = abs(float(stored.get("tau", -1)) - float(out["tau"])) < 1e-12
        add("thresholds_step14.json keeps Step 13's tau unchanged", same,
            "tau = %.6f, tau_s = %s" % (float(stored.get("tau", float("nan"))),
                                        stored.get("tau_s")))
    else:
        add("thresholds_step14.json keeps Step 13's tau unchanged", False,
            "not written")

    expected_files = [k for k in names
                      if k != "report"
                      and not (k == "chart_temporal"
                               and config.INPUT_MODE != "time_window")]
    missing = [names[k] for k in expected_files if not (folder / names[k]).exists()]
    add("every Step 14 file was written", not missing,
        "%d files" % len(expected_files) if not missing
        else "missing %s" % missing)

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table, note="Read back from disk after writing. A `[FAIL]` "
                               "here means the saved file is wrong, whatever "
                               "the in-memory numbers said.")
    return table, all_ok


def save_step14(out: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Write every Step 14 output, then read it back and check it."""
    result: dict[str, Any] = {}
    folder = prepare_output_dir(config.STEP14_OUTPUT_DIR, "the Step 14 outputs",
                                verbose=verbose)
    if folder is None:
        result.update({"status": "skipped_not_windows", "ok": False})
        return result

    names = config.STEP14_FILES
    out["evidence"].to_csv(folder / names["evidence"], index=False)
    out["explanations"].to_csv(folder / names["explanations"], index=False)
    out["global_importance"].to_csv(folder / names["global_importance"],
                                    index=False)
    out["validation"].to_csv(folder / names["validation"], index=False)
    out["agreement"].to_csv(folder / names["agreement"], index=False)
    out["alerts"].to_csv(folder / names["alerts"], index=False)
    (folder / names["examples"]).write_text(out["examples_markdown"],
                                            encoding="utf-8")
    (folder / names["thresholds"]).write_text(
        json.dumps(out["thresholds"], indent=2, default=str), encoding="utf-8")

    charts: dict[str, str] = {}
    drawn = plot_global_importance(out["global_importance"],
                                   folder / names["chart_global"], verbose=verbose)
    if drawn:
        charts["global"] = str(drawn)
    drawn = plot_class_feature_heatmap(out["global_importance"],
                                       out["model_names"][0],
                                       folder / names["chart_heatmap"],
                                       verbose=verbose)
    if drawn:
        charts["heatmap"] = str(drawn)
    first = out["model_names"][0]
    drawn = plot_signed_box(out["attributions"][first][TARGET_RISK],
                            out["feature_names"][first],
                            folder / names["chart_signed"], verbose=verbose)
    if drawn:
        charts["signed"] = str(drawn)
    drawn = plot_ig_vs_perturbation(out["attributions"][first][TARGET_RISK],
                                    out["drops"][first],
                                    folder / names["chart_scatter"],
                                    verbose=verbose)
    if drawn:
        charts["scatter"] = str(drawn)
    drawn = plot_deletion_check(out["deletion"], folder / names["chart_deletion"],
                                verbose=verbose)
    if drawn:
        charts["deletion"] = str(drawn)
    drawn = plot_contributions(out["breakdown"], out["example_reports"],
                               folder / names["chart_contributions"],
                               verbose=verbose)
    if drawn:
        charts["contributions"] = str(drawn)
    drawn = plot_risk_bands(out["risk"], out["bands"], out["tau"],
                            out.get("tau_s"), folder / names["chart_bands"],
                            verbose=verbose)
    if drawn:
        charts["bands"] = str(drawn)
    if config.INPUT_MODE == "time_window":
        drawn = plot_temporal_attribution(out["attributions"][first][TARGET_RISK],
                                          folder / names["chart_temporal"],
                                          verbose=verbose)
        if drawn:
            charts["temporal"] = str(drawn)
    result["charts"] = charts

    metrics = out["metrics"] | {
        "files": {key: names[key] for key in names
                  if (folder / names[key]).exists()}}
    (folder / names["metrics"]).write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    result["metrics"] = metrics

    report, ok = verify_saved_step14(folder, out, verbose=verbose)
    report.to_csv(folder / names["report"], index=False)
    result.update({"report": report, "ok": ok, "status": "written",
                   "folder": str(folder)})
    if verbose:
        print("  [%s]  Step 14 outputs written to %s"
              % ("OK" if ok else "FAIL", folder))
    return result


def step14_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP14_OUTPUT_DIR)
    if folder is None:
        return False
    return all((folder / config.STEP14_FILES[k]).exists()
               for k in ("alerts", "explanations", "metrics", "thresholds"))


def load_step14(verbose: bool = True) -> dict[str, Any]:
    folder = config.usable_output_dir(config.STEP14_OUTPUT_DIR)
    if folder is None:
        raise ExplainabilityError("the Step 14 folder is not usable here -- %s"
                                  % config.why_unusable(config.STEP14_OUTPUT_DIR))
    names = config.STEP14_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    for key in ("evidence", "explanations", "global_importance", "validation",
                "agreement", "alerts", "report"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    for key in ("thresholds", "metrics"):
        path = folder / names[key]
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    path = folder / names["examples"]
    if path.exists():
        out["examples_markdown"] = path.read_text(encoding="utf-8")
    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP14 = False)"),
            ("folder", str(folder)),
            ("alerts explained", len(out.get("alerts", []))),
            ("attribution rows", f"{len(out.get('explanations', [])):,}"),
        ]), note="Set `OVERWRITE_STEP14 = True` in config.py to recompute.")
    return out


def run_step14(overwrite: bool | None = None, verbose: bool = True) -> dict[str, Any]:
    """STEP 14 end to end: attribute -> validate -> explain the fusion -> report."""
    overwrite = config.OVERWRITE_STEP14 if overwrite is None else overwrite
    if not overwrite and step14_outputs_exist():
        return load_step14(verbose=verbose)

    clock = time.perf_counter()
    if verbose:
        show_title("Step 14.1 -- inputs, alignment and the decision bands",
                   level=3)
    inputs = load_step14_inputs(verbose=verbose)
    model_names = list(inputs["model_names"])
    y_test, y_train = inputs["y_test"], inputs["y_train"]
    meta = inputs["meta_test"]
    label_names, benign = inputs["label_names"], inputs["benign_codes"]
    n_classes, class_order = inputs["n_classes"], inputs["class_order"]
    tau = inputs["tau"]
    matrices = step13_matrices(inputs["fused"], model_names, class_order)
    checks = list(inputs["alignment"].to_dict("records"))

    # ---- the composed models, each checked against its saved probabilities --
    bundles: dict[str, dict[str, Any]] = {}
    for name in model_names:
        bundle = load_composed(name, verbose=verbose)
        if bundle["saved_probabilities"] is not None:
            row = check_composed_model(bundle["composed"], bundle["X_test"],
                                       bundle["saved_probabilities"], name,
                                       verbose=verbose)
            checks.append({"check": "composed model reproduces %s" % name,
                           "result": row["result"],
                           "detail": "max difference %.2e"
                                     % row["max abs difference"]})
        else:
            checks.append({"check": "composed model reproduces %s" % name,
                           "result": "[SKIP]",
                           "detail": "no saved predictions.csv to compare with"})
        bundles[name] = bundle

    probabilities = {name: composed_probabilities(b["composed"], b["X_test"])
                     for name, b in bundles.items()}
    test_list = [probabilities[name] for name in model_names]

    # ---- tau_s, on the VALIDATION split only -------------------------------
    positions = fz.validation_positions(y_train)
    val_probabilities = {
        name: composed_probabilities(
            b["composed"], b["X_train"].iloc[positions].reset_index(drop=True))
        for name, b in bundles.items()}
    y_val = y_train.iloc[positions].reset_index(drop=True)
    val_list = [val_probabilities[name] for name in model_names]
    val_contexts, test_contexts, context_names = fz.context_sets(
        config.CONTEXT_MODE, val_list, test_list, y_train, meta, label_names,
        n_classes)
    if val_contexts is None:
        raise ExplainabilityError(
            "CONTEXT_MODE = \"protocol\" needs the Protocol column in the Step 5 "
            "metadata, which is what Step 13 fused with. Run Step 5, or set "
            "CONTEXT_MODE = \"consensus_class\".")
    reliability, info = fz.build_reliability(
        val_probabilities, y_val[TARGET_COLUMN].to_numpy(), val_contexts,
        context_names, n_classes, verbose=False)

    # the recomputed reliabilities must match the ones Step 13 fused with
    saved_reliability = inputs["step13"].get("reliability")
    if saved_reliability is not None and len(saved_reliability):
        merged = saved_reliability.merge(
            reliability[["model", "context", "reliability"]],
            on=["model", "context"], how="inner",
            suffixes=("_step13", "_step14"))
        gap = (float((merged["reliability_step13"]
                      - merged["reliability_step14"]).abs().max())
               if len(merged) else float("nan"))
        checks.append({
            "check": "the reliabilities match the ones Step 13 fused with",
            "result": "[OK]" if np.isfinite(gap) and gap <= 1e-6 else "[FAIL]",
            "detail": "max difference %.2e over %d (model, context) pairs"
                      % (gap, len(merged))})

    val_fused, _w, _r, _c = fz.fuse_once(val_list, model_names, info["lookup"],
                                         info["macro"], val_contexts)
    val_risk = fz.attack_risk(val_fused, benign)
    suspicious = choose_suspicious_threshold(
        val_risk, y_val["Label_binary"].to_numpy(), tau=tau)
    tau_s = suspicious["tau_s"] if config.SUSPICIOUS_BAND else None

    risk = matrices["risk"]
    bands = decision_band(risk, tau, tau_s)
    rows = select_flows(bands, matrices["fused_class"])
    orig_row = (meta["Orig_Row"].to_numpy()[rows] if meta is not None
                and "Orig_Row" in meta.columns else rows)
    if verbose:
        show_table(kv_table([
            ("tau (Step 13, unchanged)", "%.6f" % tau),
            ("tau_s (validation, recall >= %.2f)"
             % config.SUSPICIOUS_RECALL_TARGET,
             "n/a" if tau_s is None else "%.6f" % tau_s),
            ("ATTACK flows", f"{int((bands == BAND_ATTACK).sum()):,}"),
            ("SUSPICIOUS flows", f"{int((bands == BAND_SUSPICIOUS).sum()):,}"),
            ("BENIGN flows", f"{int((bands == BAND_BENIGN).sum()):,}"),
            ("flows selected to explain", f"{len(rows):,}"),
        ]), note=suspicious.get("note", "The ATTACK rule is Step 13's `R >= "
                                        "tau`, unchanged. The suspicious band "
                                        "only subdivides what Step 13 already "
                                        "called benign."))

    # ---- 14.2/14.3  attribution and its validation, per model ---------------
    if verbose:
        show_title("Step 14.2 -- Integrated Gradients on the original features",
                   level=3)
    attributions: dict[str, dict[str, np.ndarray]] = {}
    feature_names: dict[str, list[str]] = {}
    baselines: dict[str, np.ndarray] = {}
    completeness_frames, validation_rows, deletion = [], [], {}
    drops_by_model: dict[str, np.ndarray] = {}
    global_frames, stability_by_model, latency = [], {}, {}
    wanted = model_names if config.EXPLAIN_OTHER_MODELS else model_names[:1]

    for name in wanted:
        bundle = bundles[name]
        X_selected = bundle["X_test"].iloc[rows].reset_index(drop=True)
        feature_names[name] = list(bundle["features"])
        baseline = baseline_vector(bundle["X_train"], y_train, benign)
        baselines[name] = baseline
        predicted = np.argmax(probabilities[name][rows], axis=1)

        started = time.perf_counter()
        by_class = integrated_gradients(bundle["composed"], X_selected, baseline,
                                        target=TARGET_CLASS,
                                        class_index=predicted)
        by_risk = integrated_gradients(bundle["composed"], X_selected, baseline,
                                       target=TARGET_RISK, benign_codes=benign)
        ig_seconds = time.perf_counter() - started
        attributions[name] = {TARGET_CLASS: by_class["attributions"],
                              TARGET_RISK: by_risk["attributions"]}

        for target, result in ((TARGET_CLASS, by_class), (TARGET_RISK, by_risk)):
            completeness_frames.append(
                completeness_table(result, name, target))
            global_frames.append(global_importance(
                result["attributions"], feature_names[name], name, target,
                predicted=predicted, label_names=label_names))

        started = time.perf_counter()
        perturbed = perturbation_drops(bundle["composed"], X_selected, baseline,
                                       target=TARGET_RISK, benign_codes=benign)
        perturb_seconds = time.perf_counter() - started
        drops_by_model[name] = perturbed["drops"]
        agreement = attribution_agreement(by_risk["attributions"],
                                          perturbed["drops"])
        agreement.insert(0, "model", name)
        agreement["Orig_Row"] = orig_row
        validation_rows.append(agreement)

        deletion[name] = deletion_check(bundle["composed"], X_selected, baseline,
                                        by_risk["attributions"],
                                        target=TARGET_RISK, benign_codes=benign)
        capped = min(STABILITY_MAX_FLOWS, len(rows))
        stability_by_model[name] = stability(
            bundle["composed"], X_selected.iloc[:capped], baseline,
            by_risk["attributions"][:capped], target=TARGET_RISK,
            benign_codes=benign) | {"flows": int(capped)}

        flows = max(len(rows), 1)
        latency[name] = {
            "ig_ms_per_flow": round(ig_seconds / flows * 1000 / 2, 4),
            "ig_seconds_total": round(ig_seconds, 3),
            "perturbation_ms_per_flow": round(perturb_seconds / flows * 1000, 4),
            "baseline_risk": round(float(_scores(
                bundle["composed"],
                np.broadcast_to(baseline, (1,) + baseline.shape).astype("float32"),
                TARGET_RISK, None, benign)[0]), 6),
        }
        if verbose:
            share = float(completeness_frames[-1]["within_tolerance"].mean())
            print("  %-26s IG %.1f s | completeness within %.0f%%: %.1f%% of "
                  "flows | baseline risk %.3f"
                  % (name, ig_seconds, config.COMPLETENESS_TOL * 100,
                     share * 100, latency[name]["baseline_risk"]))

    completeness = pd.concat(completeness_frames, ignore_index=True)
    completeness.insert(0, "Orig_Row", np.tile(orig_row,
                                               len(completeness) // len(rows)))
    global_table = pd.concat(global_frames, ignore_index=True)
    validation = pd.concat(validation_rows, ignore_index=True)

    # a flow's attributions are trustworthy only if EVERY model's are
    risk_completeness = completeness[completeness["target"] == TARGET_RISK]
    completeness_ok = (risk_completeness.groupby(
        risk_completeness.index % len(rows))["within_tolerance"].all()
        if len(risk_completeness) else pd.Series(dtype=bool))
    completeness_ok = {int(i): bool(v) for i, v in completeness_ok.items()}

    # ---- 14.4  the fusion, explained exactly --------------------------------
    if verbose:
        show_title("Step 14.4 -- how the fusion produced R(x)", level=3)
    started = time.perf_counter()
    breakdown = fusion_breakdown(rows, model_names, probabilities,
                                 matrices["weights"], matrices["reliability"],
                                 matrices["confidence"], benign, label_names,
                                 orig_row=orig_row)
    consistency = fusion_consistency(breakdown, risk, rows)
    average_class = np.argmax(fz.simple_average(test_list), axis=1)
    agreement_frame = agreement_table(breakdown, matrices["fused_class"],
                                      average_class, rows, label_names,
                                      orig_row=orig_row)
    fusion_seconds = time.perf_counter() - started
    checks.append({"check": "R(x) equals the sum of the model contributions",
                   "result": consistency["result"],
                   "detail": "max |R - sum C_m| = %.3e over %d explained flows"
                             % (consistency["max_abs_difference"], len(rows))})
    if verbose:
        show_table(agreement_frame.head(10),
                   note="The first ten explained flows. `dominant_model` is the "
                        "largest contribution C_m = w_m r_m, not the most "
                        "confident model.")

    # ---- 14.5  the per-alert reports ----------------------------------------
    if verbose:
        show_title("Step 14.5 -- the per-alert evidence reports", level=3)
    training_counts = y_train[TARGET_COLUMN].value_counts().to_dict()
    context_dict = {
        "label_names": label_names, "model_names": model_names,
        "breakdown": breakdown, "meta": meta, "tau": tau, "tau_s": tau_s,
        "risk": risk, "bands": bands, "fused_class": matrices["fused_class"],
        "fused_probabilities": matrices["fused_probabilities"],
        "completeness_ok": completeness_ok,
        "attributions": {name: attributions[name][TARGET_RISK]
                         for name in attributions},
        "feature_names": feature_names,
        "unscaled_rows": {int(r): inputs["X_test_unscaled"].iloc[int(r)]
                          for r in rows},
        "context_labels": {int(r): matrices["context"][int(r)] for r in rows},
        "context_label_column": matrices["context"],
        "training_rows": {int(k): int(v) for k, v in training_counts.items()},
        "y_true": y_test[TARGET_COLUMN].to_numpy(),
    }
    started = time.perf_counter()
    reports = []
    for position, row in enumerate(rows):
        if bands[row] not in (BAND_ATTACK, BAND_SUSPICIOUS):
            continue
        reports.append(build_alert_report(position, int(row), context_dict))
    report_seconds = time.perf_counter() - started
    alerts = pd.DataFrame([r["flat"] for r in reports])

    examples, seen = [], set()
    for report in reports:
        family = report["fused_label"]
        if family in seen or int(np.asarray(matrices["fused_class"])[
                int(report["flat"]["test_row"])]) in benign:
            continue
        seen.add(family)
        examples.append(report)
    if verbose and examples:
        show_note("One example per fused attack class: %s."
                  % ", ".join(sorted(seen)))

    # ---- 14.6  the explainability metrics -----------------------------------
    metrics: dict[str, Any] = {
        "step": 14, "name": "Explainability layer",
        "models_explained": wanted,
        "test_rows": int(len(risk)),
        "explained_flows": int(len(rows)),
        "alerts_explained": int(len(alerts)),
        "bands": {band: int((bands == band).sum())
                  for band in (BAND_ATTACK, BAND_SUSPICIOUS, BAND_BENIGN)},
        "settings": {
            "IG_STEPS": config.IG_STEPS, "IG_BASELINE": config.IG_BASELINE,
            "TOP_K_FEATURES": config.TOP_K_FEATURES,
            "COMPLETENESS_TOL": config.COMPLETENESS_TOL,
            "STABILITY_NOISE_STD": config.STABILITY_NOISE_STD,
            "STABILITY_N_REPEATS": config.STABILITY_N_REPEATS,
            "PERTURB_MODE": config.PERTURB_MODE,
            "SUSPICIOUS_RECALL_TARGET": config.SUSPICIOUS_RECALL_TARGET,
            "CONF_HIGH_PROB": config.CONF_HIGH_PROB,
            "CONF_MEDIUM_PROB": config.CONF_MEDIUM_PROB,
            "INPUT_MODE": config.INPUT_MODE,
        },
        "thresholds": {"tau": tau, "tau_s": tau_s,
                       "tau_source": "Step 13, unchanged",
                       "tau_s_choice": suspicious},
        "completeness": {
            model: {
                target: {
                    "share_within_tolerance": round(float(part["within_tolerance"]
                                                          .mean()), 6),
                    "mean_relative_error": round(float(part["relative_error"]
                                                       .mean()), 8),
                    "median_relative_error": round(float(part["relative_error"]
                                                         .median()), 8),
                    "flows": int(len(part)),
                }
                for target, part in group.groupby("target")}
            for model, group in completeness.groupby("model")},
        "stability": stability_by_model,
        "ig_vs_perturbation": {
            model: {
                "mean_spearman": round(float(part["spearman"].mean()), 6),
                "median_spearman": round(float(part["spearman"].median()), 6),
                "mean_top_k_overlap": round(float(part["top_k_overlap"].mean()), 6),
                "flows_with_negative_spearman": int((part["spearman"] < 0).sum()),
            }
            for model, part in validation.groupby("model")},
        "deletion_check": {
            model: {k: v for k, v in result.items()
                    if not isinstance(v, np.ndarray)}
            for model, result in deletion.items()},
        "fusion_consistency": {
            "max_abs_difference": consistency["max_abs_difference"],
            "tolerance": consistency["tolerance"], "ok": consistency["ok"]},
        "agreement": {
            status: int((agreement_frame["agreement"] == status).sum())
            for status in ("unanimous", "majority", "disagreement")},
        "fusion_changed_the_class": int(
            agreement_frame.get("fusion_changed_the_class",
                                pd.Series(dtype=bool)).sum()),
        "confidence": {level: int((alerts["confidence"] == level).sum())
                       for level in ("High", "Medium", "Low")} if len(alerts)
                      else {},
        "latency": latency | {
            "fusion_explanation_ms_per_flow":
                round(fusion_seconds / max(len(rows), 1) * 1000, 4),
            "report_generation_ms_per_alert":
                round(report_seconds / max(len(alerts), 1) * 1000, 4),
            "note": ("generation time, not analyst triage time -- this measures "
                     "how long the code takes to produce an explanation"),
        },
        "temporal_attribution": (
            "computed" if config.INPUT_MODE == "time_window" else
            "not applicable: INPUT_MODE is %r, so a flow is one vector with no "
            "time steps" % config.INPUT_MODE),
        "settings_chosen_on": ("the validation part of the Step 9 split; no "
                               "test label took part in any Step 14 setting"),
        "total_seconds": round(time.perf_counter() - clock, 2),
    }

    out: dict[str, Any] = {
        "model_names": wanted, "inputs": inputs, "bundles": bundles,
        "probabilities": probabilities, "attributions": attributions,
        "feature_names": feature_names, "baselines": baselines,
        "drops": drops_by_model, "deletion": deletion,
        "completeness": completeness, "validation": validation,
        "global_importance": global_table, "breakdown": breakdown,
        "agreement": agreement_frame, "consistency": consistency,
        "risk": risk, "bands": bands, "tau": tau, "tau_s": tau_s,
        "explained_rows": rows, "orig_row": orig_row,
        "reports": reports, "example_reports": examples,
        "alerts": alerts, "n_alerts": len(alerts),
        "evidence": evidence_frame(rows, context_dict),
        "explanations": explanations_frame(rows, orig_row, attributions,
                                           feature_names, drops_by_model),
        "examples_markdown": examples_markdown(examples),
        "thresholds": {"tau": tau, "tau_s": tau_s,
                       "tau_source": "Step 13 threshold.json, unchanged",
                       "tau_s_choice": suspicious},
        "metrics": metrics, "checks": pd.DataFrame(checks),
        "stability": stability_by_model, "latency": latency,
    }
    out["summary"] = step14_summary(out)
    out["metrics"]["summary"] = out["summary"]

    saved = save_step14(out, verbose=verbose)
    out.update({"status": saved.get("status", "written"),
                "ok": bool(saved.get("ok")), "saved": saved})
    for key in ("report", "charts", "folder"):
        if key in saved:
            out[key] = saved[key]

    if verbose:
        show_title("Step 14 -- checks", level=3)
        show_table(out["checks"], note="Asserted at run time. Attribution that "
                                       "explains a different model than the one "
                                       "that raised the alert is worse than no "
                                       "attribution, so these raise rather than "
                                       "warn.")
    return out


def step14_summary(out: dict[str, Any]) -> str:
    """The Step 14 summary, composed only from numbers that were computed."""
    metrics = out["metrics"]
    lines = ["### Step 14 - what the explanations say", ""]
    bands = metrics["bands"]
    lines.append(
        "- Of %s test flows, **%s** are ATTACK (R >= tau = %.4f), **%s** fall in "
        "the SUSPICIOUS band (tau_s = %s) and **%s** are benign. %s flows were "
        "explained, producing **%s alert reports**."
        % (f"{metrics['test_rows']:,}", f"{bands[BAND_ATTACK]:,}", out["tau"],
           f"{bands[BAND_SUSPICIOUS]:,}",
           "n/a" if out["tau_s"] is None else "%.4f" % out["tau_s"],
           f"{bands[BAND_BENIGN]:,}", f"{metrics['explained_flows']:,}",
           f"{metrics['alerts_explained']:,}"))

    ordered = [m for m in out["model_names"] if m in metrics["completeness"]]
    for model in ordered:
        by_target = metrics["completeness"][model]
        risky = by_target.get(TARGET_RISK, {})
        top = (out["global_importance"][
            (out["global_importance"]["model"] == model)
            & (out["global_importance"]["scope"] == "all explained flows")
            & (out["global_importance"]["target"] == TARGET_RISK)]
            .nlargest(3, "mean_abs_ig")["feature"].tolist())
        lines.append(
            "- `%s`: top features by mean |IG| on the risk are %s; IG "
            "completeness holds for **%.1f%%** of flows (mean relative error "
            "%.4f); stability under %.3f noise: mean Spearman **%.3f**, top-%d "
            "overlap %.3f."
            % (model, ", ".join("`%s`" % t for t in top) or "n/a",
               100 * risky.get("share_within_tolerance", float("nan")),
               risky.get("mean_relative_error", float("nan")),
               metrics["stability"][model].get("noise_std", float("nan")),
               metrics["stability"][model].get("mean_spearman", float("nan")),
               config.TOP_K_FEATURES,
               metrics["stability"][model].get("mean_top_k_overlap",
                                               float("nan"))))

    for model in [m for m in out["model_names"]
                  if m in metrics["ig_vs_perturbation"]]:
        scores = metrics["ig_vs_perturbation"][model]
        check = metrics["deletion_check"][model]
        lines.append(
            "- `%s`: IG agrees with direct perturbation at mean Spearman "
            "**%.3f** (top-%d overlap %.3f; %d of %s flows disagree outright). "
            "Removing the top-%d attributed features costs %.4f risk against "
            "%.4f for %d random features -- %s."
            % (model, scores["mean_spearman"], config.TOP_K_FEATURES,
               scores["mean_top_k_overlap"],
               scores["flows_with_negative_spearman"],
               f"{metrics['explained_flows']:,}", check["k"],
               check["mean_drop_top_k"], check["mean_drop_random_k"],
               check["k"],
               "the attribution is informative" if check["top_k_drops_more"]
               else "**the attribution is NOT beating chance here**"))

    agreement = metrics["agreement"]
    lines.append(
        "- Across the explained flows the three models were unanimous on %s, a "
        "majority on %s and in outright disagreement on %s; the adaptive "
        "weighting changed the predicted class away from the simple average on "
        "**%s** of them."
        % (f"{agreement['unanimous']:,}", f"{agreement['majority']:,}",
           f"{agreement['disagreement']:,}",
           f"{metrics['fusion_changed_the_class']:,}"))
    lines.append(
        "- R(x) = sum_m C_m holds to **%.1e** across every explained flow, so "
        "the per-model contributions in the reports add up to exactly the risk "
        "Step 13 published."
        % metrics["fusion_consistency"]["max_abs_difference"])
    if metrics["confidence"]:
        lines.append("- Report confidence: %s."
                     % ", ".join("%s **%s**" % (k, f"{v:,}")
                                 for k, v in metrics["confidence"].items()))
    latency = metrics["latency"]
    per_model = "; ".join(
        "`%s` %.2f ms" % (model, values["ig_ms_per_flow"])
        for model, values in latency.items() if isinstance(values, dict))
    lines.append(
        "- Generation time per flow: IG %s; perturbation %s; fusion explanation "
        "%.2f ms; report %.2f ms per alert. This is how long the CODE takes, "
        "not how long an analyst takes."
        % (per_model,
           "; ".join("`%s` %.2f ms" % (m, v["perturbation_ms_per_flow"])
                     for m, v in latency.items() if isinstance(v, dict)),
           latency["fusion_explanation_ms_per_flow"],
           latency["report_generation_ms_per_alert"]))

    lines += [
        "",
        "**Four caveats, and they matter more here than anywhere else in this "
        "pipeline.**",
        "",
        "1. **IG explains the model, not the world.** An attribution says which "
        "input the network leaned on. It is not evidence that the feature "
        "caused the attack, and a model that learned a shortcut will be "
        "explained faithfully as using that shortcut.",
        "2. **The attack stage is a lookup.** `ATTACK_STAGE_MAP` turns the "
        "predicted label into a tactic name; an unmapped label is reported as "
        "`Unmapped`. Nothing here infers a multi-stage progression, and no "
        "report claims one.",
        "3. **Source context is descriptive.** It lists other alerts from the "
        "same Src IP in the test subset, which is a random 20% of the flows, "
        "so a gap means \"not in this subset\", never \"the source stopped\". "
        "No model sees it.",
    ]
    if len(out["alerts"]):
        per_class = "; ".join(
            "%s %d" % (label, count) for label, count in
            sorted(out["alerts"]["fused_label"].value_counts().items()))
        lines.append(
            "4. **The rare classes have few explained alerts**, so a per-class "
            "attribution for them rests on a handful of flows: %s." % per_class)
    else:
        lines.append(
            "4. **No alert reached the ATTACK or SUSPICIOUS band**, so there is "
            "no per-alert evidence to read: every figure above describes the "
            "benign contrast sample only.")
    return "\n".join(lines)
