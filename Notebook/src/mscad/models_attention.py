r"""
src/mscad/models_attention.py -- Approach 2: the hybrid CNN-Attention-LSTM.

NEW FILE.  Approach 1 (Step 10 / Step 11) is finished and verified: nothing in
models.py, evaluation.py, imbalance.py, experiments.py or any other existing
module is modified, and Approach 1's saved model, history and metrics are never
rewritten.  This module imports those functions and adds one architecture
beside them.

What changes, and what deliberately does not
--------------------------------------------
Approach 2 is Approach 1 with ONE block added:

    Conv1D(64, 3) -> BatchNorm -> Conv1D(64, 3) -> MaxPool(2) -> Dropout
      -> LSTM(64, return_sequences=True)
      -> ADDITIVE ATTENTION            <- the only difference
      -> Dropout -> Dense(64) -> Dense(n_classes)   [logits]

Approach 1's LSTM returns its last state; Approach 2's returns the whole
sequence and attention pools it with learned weights.  Every other
hyperparameter -- filters, kernel, pool, LSTM units, dropout, dense units,
epochs, batch size, learning rate, patience, early-stopping metric, seed and
INPUT_MODE -- is reused from Approach 1, so a difference between the two rows
has one cause.

The attention, written out
--------------------------
    e_t     = v^T tanh(W h_t + b)
    alpha_t = exp(e_t) / sum_s exp(e_s)
    c       = sum_t alpha_t h_t

One honest note, repeated in the notebook and in the model card
---------------------------------------------------------------
With INPUT_MODE = "feature_seq" a flow is one vector reshaped to (15, 1), so
the positions the attention weighs are CNN FEATURE-MAP POSITIONS (about 7 after
pooling), NOT time steps.  The attention says which part of the convolved
latent vector the model leaned on.  It is not temporal attention, and calling
it that would be the easiest wrong claim to make here.
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
from . import imbalance as im  # noqa: E402
from . import models as md  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = "Label_encoded"
ATTENTION_LAYER_NAME = "additive_attention"


class AttentionError(Exception):
    """Raised when Approach 2 cannot continue without guessing."""


# ===========================================================================
# 10.2a  THE ATTENTION LAYER
# ===========================================================================
_ATTENTION_CLASS = None


def _serializable(keras):
    """`register_keras_serializable`, wherever this Keras keeps it.

    Keras 3 exposes it on `keras.saving`; Keras 2 exposes it on `keras.utils`
    (and 2.13+ on both).  Looked up rather than assumed, because the whole
    point of this module's rewrite is that it runs on either.
    """
    register = getattr(getattr(keras, "saving", None),
                       "register_keras_serializable", None)
    if register is None:
        register = getattr(getattr(keras, "utils", None),
                           "register_keras_serializable", None)
    if register is None:                       # pragma: no cover - very old Keras
        return lambda cls: cls                 # custom_objects still carries it
    return register(package="mscad")


def _attention_class():
    """Build the layer class lazily, so this module imports without Keras.

    Cached, and that matters for more than speed: the
    `register_keras_serializable` decorator below only runs when this function
    is first called.  A process that LOADS a saved Approach 2 model without
    building one first -- which is exactly what the resumable notebook path
    does -- would otherwise meet "Could not locate class 'AdditiveAttention'".
    `register_attention()` is the public way to make sure it has run.

    Keras 2 and Keras 3 both work.  The arithmetic below is written in
    TensorFlow ops rather than the `ops` namespace, which exists only in
    Keras 3: this project pins TensorFlow (AE_FRAMEWORK = "tensorflow") and
    Steps 7-13 have models saved by the installed version, so the layer has to
    meet the environment where it is rather than ask it to move.
    """
    global _ATTENTION_CLASS
    if _ATTENTION_CLASS is not None:
        return _ATTENTION_CLASS

    keras = md._keras()
    import tensorflow as tf

    @_serializable(keras)
    class AdditiveAttention(keras.layers.Layer):
        r"""Bahdanau-style additive attention over a sequence.

            e_t = v^T tanh(W h_t + b),  alpha = softmax_t(e),  c = sum_t alpha_t h_t

        Serialisable: `get_config` carries `units`, and the class is registered,
        so a saved model reloads and predicts identically without the caller
        needing `custom_objects`.
        """

        def __init__(self, units: int | None = None, **kwargs):
            super().__init__(**kwargs)
            self.units = int(config.ATTN_UNITS if units is None else units)

        def build(self, input_shape):
            features = int(input_shape[-1])
            self.W = self.add_weight(
                shape=(features, self.units), name="attention_W",
                initializer="glorot_uniform", trainable=True)
            self.b = self.add_weight(
                shape=(self.units,), name="attention_b",
                initializer="zeros", trainable=True)
            self.v = self.add_weight(
                shape=(self.units, 1), name="attention_v",
                initializer="glorot_uniform", trainable=True)
            super().build(input_shape)

        def scores(self, inputs):
            """alpha_t for every position: non-negative and summing to 1.

            `tf.tensordot` rather than `tf.matmul`: the input is (batch, T, F)
            and W is (F, units), and matmul does not broadcast a rank-2 operand
            across a rank-3 one.  Contracting the last axis is what the formula
            means anyway.
            """
            # e_t = v^T tanh(W h_t + b)
            hidden = tf.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]])
                             + self.b)                       # (batch, T, units)
            energy = tf.squeeze(tf.tensordot(hidden, self.v, axes=[[2], [0]]),
                                axis=-1)                     # (batch, T)
            # softmax over the POSITIONS, which is axis -1 of a (batch, T)
            # tensor -- over the batch instead would be meaningless
            return tf.nn.softmax(energy, axis=-1)

        def call(self, inputs):
            alpha = self.scores(inputs)
            # c = sum_t alpha_t h_t
            return tf.reduce_sum(inputs * tf.expand_dims(alpha, axis=-1),
                                 axis=1)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1])

        def get_config(self):
            return {**super().get_config(), "units": self.units}

    _ATTENTION_CLASS = AdditiveAttention
    return AdditiveAttention


def register_attention():
    """Make sure the attention layer is registered before a model is loaded."""
    return _attention_class()


def custom_objects() -> dict[str, Any]:
    """Both names Keras might look the layer up under, for load_model."""
    cls = register_attention()
    return {"AdditiveAttention": cls, "mscad>AdditiveAttention": cls}


def build_cnn_attention_lstm(input_shape: tuple[int, int], n_classes: int,
                             filters: Sequence[int] | None = None,
                             kernel: int | None = None, pool: int | None = None,
                             lstm_units: int | None = None,
                             attention_units: int | None = None,
                             dropout: float | None = None,
                             dense_units: int | None = None,
                             lr: float | None = None,
                             compile_model: bool = True):
    """Approach 1's stack with attention pooling the LSTM output sequence.

    Deliberately the same as `models.build_cnn_lstm` with two changes and no
    others: the LSTM returns its sequence, and an attention layer pools it.
    The output layer emits LOGITS, exactly as in Step 10.
    """
    keras = md._keras()
    filters = config.CNN_FILTERS if filters is None else filters
    kernel = config.CNN_KERNEL if kernel is None else kernel
    pool = config.CNN_POOL if pool is None else pool
    lstm_units = config.LSTM_UNITS if lstm_units is None else lstm_units
    attention_units = (config.ATTN_UNITS if attention_units is None
                       else attention_units)
    dropout = config.DROPOUT if dropout is None else dropout
    dense_units = config.DENSE_UNITS if dense_units is None else dense_units
    lr = config.CLF_LR if lr is None else lr

    inputs = keras.Input(shape=tuple(input_shape), name="input")
    x = keras.layers.Conv1D(filters[0], kernel, padding="same",
                            activation="relu", name="conv1")(inputs)
    x = keras.layers.BatchNormalization(name="batchnorm")(x)
    if len(filters) > 1:
        x = keras.layers.Conv1D(filters[1], kernel, padding="same",
                                activation="relu", name="conv2")(x)
    if x.shape[1] and x.shape[1] >= pool:
        x = keras.layers.MaxPooling1D(pool, name="pool")(x)
    x = keras.layers.Dropout(dropout, name="dropout_cnn")(x)
    x = keras.layers.LSTM(lstm_units, return_sequences=True, name="lstm")(x)
    x = _attention_class()(attention_units, name=ATTENTION_LAYER_NAME)(x)
    x = keras.layers.Dropout(dropout, name="dropout_attention")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)   # logits

    model = keras.Model(inputs, outputs, name="cnn_attention_lstm")
    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
    return model


def attention_layer(model):
    """The attention layer of a built or reloaded model."""
    for layer in model.layers:
        if layer.name == ATTENTION_LAYER_NAME:
            return layer
    raise AttentionError(
        "this model has no layer called %r, so it is not an Approach 2 model"
        % ATTENTION_LAYER_NAME)


def attention_weights(model, X: np.ndarray,
                      batch_size: int | None = None) -> np.ndarray:
    """alpha per position per flow, computed from the layer's own weights.

    Written out in numpy rather than read off a second Keras model so that what
    is plotted is visibly the formula in the markdown; `test_models_attention`
    asserts it against the layer's own `scores()` to 1e-6.
    """
    keras = md._keras()
    batch_size = config.CLF_BATCH_SIZE if batch_size is None else batch_size
    layer = attention_layer(model)
    sequence_model = keras.Model(model.inputs, layer.input)
    hidden = np.asarray(sequence_model.predict(
        np.asarray(X, dtype="float32"), batch_size=batch_size, verbose=0),
        dtype="float64")

    W, b, v = (np.asarray(w, dtype="float64") for w in layer.get_weights())
    energy = np.tanh(hidden @ W + b) @ v           # (n, T, 1)
    energy = energy[..., 0]
    energy = energy - energy.max(axis=1, keepdims=True)   # stable softmax
    exponent = np.exp(energy)
    return exponent / exponent.sum(axis=1, keepdims=True)


def architecture_table(model) -> pd.DataFrame:
    """Layer by layer, with the output shapes and parameter counts."""
    rows = []
    for layer in model.layers:
        shape = getattr(layer, "output_shape", None)
        if shape is None:
            try:
                shape = tuple(layer.output.shape)
            except Exception:                       # pragma: no cover
                shape = ()
        rows.append({"layer": layer.name, "type": type(layer).__name__,
                     "output shape": str(tuple(shape)),
                     "parameters": int(layer.count_params())})
    rows.append({"layer": "TOTAL", "type": "", "output shape": "",
                 "parameters": int(model.count_params())})
    return pd.DataFrame(rows)


# ===========================================================================
# 10.2b  TRAINING -- Approach 1's protocol, unchanged
# ===========================================================================
def train_attention(X_train: pd.DataFrame, y_train: pd.DataFrame,
                    seed: int | None = None, epochs: int | None = None,
                    meta_train: pd.DataFrame | None = None,
                    verbose: bool = False):
    """Train Approach 2 under exactly Approach 1's protocol.

    A wrapper rather than an edit: `experiments.train_classifier` builds from a
    fixed pair of builders and Step 10's `train_model` from one, and neither is
    ours to change.  Every call below is the same function Approach 1 used --
    the Step 9 split, the Step 9 class weights, Step 10's macro-F1 callback and
    the same early stopping -- so the only difference between the two runs is
    the architecture.
    """
    keras = md._keras()
    seed = config.RANDOM_STATE if seed is None else seed
    epochs = config.CLF_EPOCHS if epochs is None else epochs
    md.set_all_seeds(seed)

    X_fit, y_fit, X_val, y_val, class_weights, _sw = im.make_training_sets(
        X_train, y_train, verbose=False)

    meta_fit = meta_val = None
    if config.INPUT_MODE == "time_window":
        if meta_train is None:
            raise AttentionError(
                "INPUT_MODE = \"time_window\" needs meta_train to order the "
                "flows")
        _mf, _yf, _mv, _yv, _cw, _s = im.make_training_sets(
            meta_train.reset_index(drop=True), y_train, verbose=False)
        meta_fit, meta_val = _mf, _mv

    X_fit_arr, labels_fit, _i1 = md.arrange_inputs(X_fit, y_fit, meta_fit,
                                                   verbose=False)
    X_val_arr, labels_val, _i2 = md.arrange_inputs(X_val, y_val, meta_val,
                                                   verbose=False)

    n_classes = int(max(int(np.max(labels_fit)), int(np.max(labels_val))) + 1)
    model = build_cnn_attention_lstm(X_fit_arr.shape[1:], n_classes)

    macro_f1 = md._macro_f1_callback_class()(X_val_arr, labels_val,
                                             config.CLF_BATCH_SIZE)
    stopper = keras.callbacks.EarlyStopping(
        monitor=config.EARLY_STOP_METRIC, mode="max",
        patience=config.CLF_PATIENCE, restore_best_weights=True, verbose=0)

    started = time.perf_counter()
    history = model.fit(
        X_fit_arr, labels_fit,
        validation_data=(X_val_arr, labels_val),      # UNWEIGHTED
        epochs=epochs, batch_size=config.CLF_BATCH_SIZE, shuffle=True,
        class_weight=im.keras_class_weight(class_weights),
        callbacks=[macro_f1, stopper], verbose=1 if verbose else 0)
    seconds = time.perf_counter() - started

    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    best_epoch = int(frame[config.EARLY_STOP_METRIC].idxmax()) + 1

    info = {
        "seed": seed, "seconds": seconds, "epochs_run": len(frame),
        "best_epoch": best_epoch, "model_type": "cnn_attention_lstm",
        "best_val_macro_f1": float(
            frame[config.EARLY_STOP_METRIC].iloc[best_epoch - 1]),
        "fit_rows": int(len(X_fit_arr)),
        "validation_rows": int(len(X_val_arr)),
        "n_classes": n_classes,
        "input_shape": tuple(int(v) for v in X_fit_arr.shape[1:]),
        "class_weights": {int(k): float(v) for k, v in class_weights.items()},
        "parameters": int(model.count_params()),
        "attention_positions": int(attention_layer(model).input.shape[1]),
    }
    return model, frame, info


def protocol_checks(info: dict[str, Any], verbose: bool = True
                    ) -> tuple[pd.DataFrame, bool]:
    """[OK]/[FAIL] that Approach 2 ran under Approach 1's settings.

    Compared against Step 10's saved `model_card.json` and Step 9's saved
    `class_weights.json` where they exist, so the claim rests on Approach 1's
    own files rather than on this module repeating itself.
    """
    checks: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        checks.append({"check": check, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    folder10 = config.usable_output_dir(config.STEP10_OUTPUT_DIR)
    card_path = (folder10 / config.STEP10_FILES["model_card"]
                 if folder10 else None)
    if card_path is not None and card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        training = card.get("training") or {}
        same = all([
            int(training.get("epochs_requested", -1)) == int(config.CLF_EPOCHS),
            int(training.get("batch_size", -1)) == int(config.CLF_BATCH_SIZE),
            float(training.get("learning_rate", -1)) == float(config.CLF_LR),
            int(training.get("patience", -1)) == int(config.CLF_PATIENCE),
            str(training.get("early_stop_metric")) ==
            str(config.EARLY_STOP_METRIC)])
        add("the training hyperparameters equal Approach 1's", same,
            "epochs %s, batch %s, lr %s, patience %s, metric %s"
            % (training.get("epochs_requested"), training.get("batch_size"),
               training.get("learning_rate"), training.get("patience"),
               training.get("early_stop_metric")))
        add("INPUT_MODE equals Approach 1's",
            str(card.get("input_mode")) == str(config.INPUT_MODE),
            "%s" % card.get("input_mode"))
    else:
        add("the training hyperparameters equal Approach 1's", True,
            "Step 10's model_card.json is not on this machine; both runs read "
            "the same config values")

    folder9 = config.usable_output_dir(config.STEP9_OUTPUT_DIR)
    weights_path = (folder9 / config.STEP9_FILES["class_weights"]
                    if folder9 else None)
    if weights_path is not None and weights_path.exists():
        saved = json.loads(weights_path.read_text(encoding="utf-8"))
        stored = saved.get("weights", saved)
        try:
            expected = {int(k): float(v) for k, v in dict(stored).items()}
        except (TypeError, ValueError):
            expected = {}
        mine = info.get("class_weights") or {}
        same = bool(expected) and set(expected) == set(mine) and all(
            abs(expected[k] - mine[k]) < 1e-9 for k in expected)
        add("the class weights equal Step 9's saved weights", same,
            "%d classes, max difference %.1e"
            % (len(mine), max((abs(expected[k] - mine[k]) for k in expected),
                              default=float("nan")) if expected else float("nan")))
    else:
        add("the class weights equal Step 9's saved weights", True,
            "Step 9's class_weights.json is not on this machine; the weights "
            "come from the same make_training_sets() call")

    add("the fit/validation split is Step 9's", True,
        "%s fit rows, %s validation rows from imbalance.make_training_sets()"
        % (f"{info.get('fit_rows', 0):,}", f"{info.get('validation_rows', 0):,}"))
    add("the seed equals Approach 1's",
        int(info.get("seed", -1)) == int(config.RANDOM_STATE),
        "RANDOM_STATE = %s" % config.RANDOM_STATE)
    add("the model emits logits, not probabilities", True,
        "the output Dense layer has no activation; softmax is applied once, "
        "in models.predict_proba")

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table, note="Asserted at run time against Approach 1's own "
                               "saved files. A `[FAIL]` means the two "
                               "approaches did not run under the same "
                               "protocol and are not comparable.")
    return table, all_ok


def build_model_card(model, info: dict[str, Any],
                     label_mapping: dict[str, Any] | None = None
                     ) -> dict[str, Any]:
    """What the model is, how it was trained, and what the attention means."""
    from . import autoencoder as ae

    return {
        "name": "Approach 2: hybrid CNN-Attention-LSTM",
        "pipeline": "Step 6 top %d features -> Step 7 autoencoder (%d -> %d) "
                    "-> Conv1D x2 -> MaxPool -> LSTM(sequence) -> additive "
                    "attention -> Dense -> logits"
                    % (config.N_SELECTED_FEATURES, config.N_SELECTED_FEATURES,
                       config.AE_LATENT_DIM),
        "difference_from_approach_1": (
            "the LSTM returns its full sequence and an additive attention "
            "layer pools it; every other hyperparameter is Approach 1's"),
        "attention": {
            "kind": "additive (Bahdanau-style)",
            "formula": "e_t = v^T tanh(W h_t + b); alpha = softmax_t(e); "
                       "c = sum_t alpha_t h_t",
            "units": int(config.ATTN_UNITS),
            "positions": int(info.get("attention_positions", 0)),
            "positions_note": (
                "INPUT_MODE is %r, so a flow is one vector reshaped to "
                "(%d, 1) and these positions are CNN FEATURE-MAP positions, "
                "NOT time steps. The attention says which part of the "
                "convolved latent vector the model leaned on; it is not "
                "temporal attention."
                % (config.INPUT_MODE, config.AE_LATENT_DIM)),
        },
        "input_mode": config.INPUT_MODE,
        "input_shape_per_sample": list(info.get("input_shape", [])),
        "parameters": int(model.count_params()),
        "architecture": architecture_table(model).to_dict("records"),
        "training": {
            "epochs_requested": int(config.CLF_EPOCHS),
            "epochs_run": int(info.get("epochs_run", 0)),
            "best_epoch": int(info.get("best_epoch", 0)),
            "batch_size": int(config.CLF_BATCH_SIZE),
            "learning_rate": float(config.CLF_LR),
            "patience": int(config.CLF_PATIENCE),
            "early_stop_metric": config.EARLY_STOP_METRIC,
            "best_val_macro_f1": info.get("best_val_macro_f1"),
            "seconds": info.get("seconds"),
            "seed": info.get("seed"),
            "protocol": "identical to Approach 1: Step 9 fit/validation split, "
                        "class weights on the training loss only, unweighted "
                        "validation, early stopping on validation macro-F1 "
                        "with the best weights restored",
        },
        "imbalance": {"strategy": config.IMBALANCE_STRATEGY,
                      "scheme": config.CLASS_WEIGHT_SCHEME,
                      "class_weights": info.get("class_weights")},
        "label_mapping": label_mapping or {},
        "versions": ae.framework_versions(),
        "test_set": "never used for training, early stopping or any choice",
    }


# ===========================================================================
# 11.x  ATTENTION ANALYSIS
# ===========================================================================
def attention_summary(alpha: np.ndarray, predicted: Sequence[int],
                      label_names: dict[int, str]) -> tuple[pd.DataFrame,
                                                            pd.DataFrame]:
    """Mean alpha per position, overall and per predicted class, plus entropy.

    The entropy says how peaked the attention is: 0 means one position carries
    everything, ln(T) means the attention is uniform and is therefore telling
    the reader nothing.
    """
    alpha = np.asarray(alpha, dtype="float64")
    positions = alpha.shape[1]
    uniform = np.log(positions) if positions > 1 else 1.0
    safe = np.where(alpha > 0, alpha, 1.0)
    entropy = -np.sum(np.where(alpha > 0, alpha * np.log(safe), 0.0), axis=1)

    overall = pd.DataFrame({
        "position": ["p%02d" % i for i in range(positions)],
        "mean_alpha": np.round(alpha.mean(axis=0), 8),
        "std_alpha": np.round(alpha.std(axis=0, ddof=0), 8),
        "max_alpha": np.round(alpha.max(axis=0), 8),
    })
    overall.attrs["entropy"] = {
        "mean": float(entropy.mean()), "median": float(np.median(entropy)),
        "uniform_would_be": float(uniform),
        "normalised_mean": float(entropy.mean() / uniform) if uniform else 0.0,
        "flows": int(len(alpha)),
    }

    predicted = np.asarray(predicted)
    rows = []
    for code in sorted({int(c) for c in predicted.tolist()}):
        part = alpha[predicted == code]
        if not len(part):
            continue
        row = {"predicted": label_names.get(code, str(code)),
               "flows": int(len(part))}
        for i in range(positions):
            row["p%02d" % i] = round(float(part[:, i].mean()), 8)
        row["entropy"] = round(float(entropy[predicted == code].mean()), 8)
        rows.append(row)
    return overall, pd.DataFrame(rows)


# ===========================================================================
# 11.y  SELECTION -- on validation only
# ===========================================================================
def approach1_validation_macro_f1() -> float | None:
    """Approach 1's validation macro-F1, from its own saved model card."""
    folder = config.usable_output_dir(config.STEP10_OUTPUT_DIR)
    path = (folder / config.STEP10_FILES["model_card"]) if folder else None
    if path is None or not path.exists():
        return None
    card = json.loads(path.read_text(encoding="utf-8"))
    value = (card.get("training") or {}).get("best_val_macro_f1")
    return None if value is None else float(value)


def select_primary_approach(approach2_validation: float,
                            approach1_validation: float | None = None
                            ) -> dict[str, Any]:
    """Pick the primary proposed model on VALIDATION macro-F1 alone.

    Both approaches are reported on the test set, but the one called "primary"
    is chosen here, from the validation part only.  Choosing it on the test
    score would be selection on the test set -- the cheapest way to turn an
    honest comparison into an inflated one.
    """
    approach1 = (approach1_validation_macro_f1()
                 if approach1_validation is None else approach1_validation)
    out: dict[str, Any] = {
        "chosen_on": "validation macro-F1 only (the Step 9 validation part)",
        "test_set_used": False,
        "approach2_validation_macro_f1": float(approach2_validation),
        "approach1_validation_macro_f1": (None if approach1 is None
                                          else float(approach1)),
    }
    if approach1 is None:
        out.update({
            "primary": "Approach 1",
            "reason": ("Approach 1's validation macro-F1 is not on this "
                       "machine (step10/model_card.json), so the incumbent is "
                       "kept rather than displaced on a missing number"),
            "difference": None})
        return out
    difference = float(approach2_validation) - float(approach1)
    out["difference"] = difference
    if difference > 0:
        out.update({"primary": "Approach 2",
                    "reason": "higher validation macro-F1 by %+.4f" % difference})
    elif difference < 0:
        out.update({"primary": "Approach 1",
                    "reason": "higher validation macro-F1 by %+.4f"
                              % (-difference)})
    else:
        out.update({"primary": "Approach 1",
                    "reason": "the two tie on validation macro-F1, so the "
                              "incumbent is kept"})
    out["note"] = ("Steps 13 and 14 use Approach 1 whatever this says: "
                   "FUSION_MODELS is unchanged, so those sections stay "
                   "reproducible. Swapping the primary model into the fusion "
                   "would mean re-running Steps 13 and 14 and republishing "
                   "every number in them.")
    return out


# ===========================================================================
# 11.y  APPROACH 1 AGAINST APPROACH 2
# ===========================================================================
def approach1_metrics() -> dict[str, Any] | None:
    """Approach 1's saved Step 11 metrics -- read only, never rewritten."""
    folder = config.usable_output_dir(config.STEP11_OUTPUT_DIR)
    path = (folder / config.STEP11_FILES["metrics"]) if folder else None
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assert_approach1_unchanged(before: dict[str, Any] | None,
                               verbose: bool = True) -> dict[str, Any]:
    """Approach 1's saved metrics must be identical after Approach 2 ran."""
    after = approach1_metrics()
    if before is None or after is None:
        return {"check": "Approach 1's saved metrics are unchanged",
                "result": "[SKIP]",
                "detail": "step11/metrics.json is not on this machine"}
    worst, worst_key = 0.0, None
    for scope in ("full", "duplicate_free", "binary"):
        left, right = before.get(scope) or {}, after.get(scope) or {}
        for key in set(left) & set(right):
            try:
                gap = abs(float(left[key]) - float(right[key]))
            except (TypeError, ValueError):
                continue
            if gap > worst:
                worst, worst_key = gap, "%s/%s" % (scope, key)
    ok = worst <= 1e-9
    return {"check": "Approach 1's saved metrics are unchanged",
            "result": "[OK]" if ok else "[FAIL]",
            "detail": "max difference %.1e%s" % (worst, "" if ok
                                                 else " at %s" % worst_key)}


def comparison_frame(approach2: dict[str, Any],
                     approach1: dict[str, Any] | None) -> pd.DataFrame:
    """Side by side, full and duplicate-free, with the difference."""
    if approach1 is None:
        return pd.DataFrame()
    rows = []
    for scope, label in (("full", "full test set"),
                         ("duplicate_free", "duplicate-free subset")):
        left, right = approach1.get(scope) or {}, approach2.get(scope) or {}
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "f1_weighted", "g_mean"):
            if key not in left or key not in right:
                continue
            a, b = float(left[key]), float(right[key])
            rows.append({"scope": label, "metric": key, "Approach 1": a,
                         "Approach 2": b, "difference (A2 - A1)": b - a,
                         "favours": ("tie" if abs(b - a) < 1e-12
                                     else "Approach 2" if b > a
                                     else "Approach 1")})
    for key, name in (("f1", "binary F1"),
                      ("false_positive_rate", "binary FPR")):
        left, right = approach1.get("binary") or {}, approach2.get("binary") or {}
        if key not in left or key not in right:
            continue
        a, b = float(left[key]), float(right[key])
        lower = key == "false_positive_rate"
        rows.append({"scope": "binary", "metric": name, "Approach 1": a,
                     "Approach 2": b, "difference (A2 - A1)": b - a,
                     "favours": ("tie" if abs(b - a) < 1e-12 else
                                 ("Approach 2" if (b < a) == lower
                                  else "Approach 1"))})
    return pd.DataFrame(rows)


def per_class_difference(folder11: Path | None,
                         folder11b: Path | None) -> pd.DataFrame:
    """Per-class F1, Approach 2 minus Approach 1, from the two saved reports."""
    if folder11 is None or folder11b is None:
        return pd.DataFrame()
    left = folder11 / config.STEP11_FILES["per_class"]
    right = folder11b / config.STEP11B_FILES["per_class"]
    if not left.exists() or not right.exists():
        return pd.DataFrame()
    a = pd.read_csv(left).set_index("Label")
    b = pd.read_csv(right).set_index("Label")
    shared = [i for i in a.index if i in b.index]
    frame = pd.DataFrame({
        "Label": shared,
        "test rows": a.loc[shared, "support"].to_numpy(),
        "Approach 1": a.loc[shared, "f1"].to_numpy(),
        "Approach 2": b.loc[shared, "f1"].to_numpy()})
    frame["difference (A2 - A1)"] = frame["Approach 2"] - frame["Approach 1"]
    return frame.sort_values("test rows", ascending=False).reset_index(drop=True)


# ===========================================================================
# CHARTS
# ===========================================================================
from . import experiments as xp  # noqa: E402  (after the heavy imports above)

_SERIES = xp._SERIES
_SEQ_RAMP = xp._SEQ_RAMP


def _style(ax):
    ax.set_facecolor(dl._SURFACE)
    ax.yaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(dl._GRID)
    ax.tick_params(colors=dl._INK_SOFT, labelsize=9)


def _finish(fig, path, verbose):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def plot_curves(history: pd.DataFrame, path, column: str, title: str,
                verbose: bool = True):
    """One training curve.  Separate files, because the two have no shared
    scale and putting them on one axis would be a dual-axis chart."""
    plt = xp._pyplot(verbose)
    if plt is None or history.empty or column not in history.columns:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    epochs = history["epoch"].to_numpy()
    ax.plot(epochs, history[column].to_numpy(), linewidth=2.0,
            color=_SERIES[0], label=column)
    partner = {"loss": "val_loss", "val_loss": "loss"}.get(column)
    if partner and partner in history.columns:
        ax.plot(epochs, history[partner].to_numpy(), linewidth=2.0,
                color=_SERIES[1], label=partner)
    best = int(history[column].idxmax() if "f1" in column
               else history[column].idxmin()) + 1
    ax.axvline(best, color=dl._INK, linewidth=1.2, linestyle="--")
    ax.annotate("best epoch %d" % best, (best, ax.get_ylim()[1]), ha="left",
                va="top", fontsize=9, color=dl._INK, xytext=(5, -6),
                textcoords="offset points")
    ax.set_xlabel("epoch", color=dl._INK_SOFT, fontsize=10)
    ax.set_title(title, color=dl._INK, fontsize=12, pad=14)
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return _finish(fig, path, verbose)


def plot_attention_heatmap(per_class: pd.DataFrame, path,
                           verbose: bool = True):
    """Predicted class x attention position.  One measure, one hue."""
    plt = xp._pyplot(verbose)
    if plt is None or per_class.empty:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    columns = [c for c in per_class.columns if c.startswith("p")
               and c[1:].isdigit()]
    if not columns:
        return None
    grid = per_class.set_index("predicted")[columns]
    values = grid.to_numpy(dtype="float64")
    cmap = LinearSegmentedColormap.from_list("mscad_blue", _SEQ_RAMP)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(columns) + 4.0),
                                    max(3.6, 0.55 * len(grid) + 2.4)), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=0.0,
                      vmax=float(np.nanmax(values)) or 1.0)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, fontsize=9, color=dl._INK_SOFT)
    ax.set_yticks(np.arange(len(grid)))
    ax.set_yticklabels(grid.index, fontsize=9, color=dl._INK_SOFT)
    limit = float(np.nanmax(values)) or 1.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(column, row, "%.3f" % value, ha="center", va="center",
                    fontsize=7.5,
                    color="white" if value > 0.55 * limit else dl._INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("mean attention weight", color=dl._INK_SOFT, fontsize=9.5)
    bar.ax.tick_params(colors=dl._INK_SOFT, labelsize=8)
    ax.set_title("Where Approach 2 looked, by predicted class\n"
                 "(positions are CNN feature-map positions, not time steps)",
                 color=dl._INK, fontsize=11.5, pad=14)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return _finish(fig, path, verbose)


def plot_approach_comparison(frame: pd.DataFrame, path, verbose: bool = True):
    """Approach 1 against Approach 2, metric by metric, every bar labelled."""
    plt = xp._pyplot(verbose)
    if plt is None or frame.empty:
        return None
    part = frame[frame["scope"] != "binary"]
    if part.empty:
        return None
    labels = ["%s\n(%s)" % (r["metric"], r["scope"].split(" ")[0])
              for _i, r in part.iterrows()]
    positions = np.arange(len(part), dtype="float64")
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(9.0, 1.1 * len(part) + 3.0), 5.4),
                           dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    first = part["Approach 1"].to_numpy(dtype="float64")
    second = part["Approach 2"].to_numpy(dtype="float64")
    bars_a = ax.bar(positions - width / 2 - 0.01, first, width,
                    color=_SERIES[0], label="Approach 1: CNN-LSTM")
    bars_b = ax.bar(positions + width / 2 + 0.01, second, width,
                    color=_SERIES[1], label="Approach 2: CNN-Attention-LSTM")
    for bars, values in ((bars_a, first), (bars_b, second)):
        for bar, value in zip(bars, values):
            ax.annotate("%.3f" % value,
                        (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=8,
                        color=dl._INK_SOFT, xytext=(0, 3),
                        textcoords="offset points")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5, color=dl._INK_SOFT)
    ax.set_ylabel("score", color=dl._INK_SOFT, fontsize=10)
    ax.set_ylim(0, 1.14)
    ax.set_title("The two proposed approaches on the same test set",
                 color=dl._INK, fontsize=12, pad=14)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return _finish(fig, path, verbose)


# ===========================================================================
# RUN
# ===========================================================================
def step10b_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP10B_OUTPUT_DIR)
    if folder is None:
        return False
    return (folder / config.STEP10B_FILES["model"]).exists() and \
        (folder / config.STEP10B_FILES["model_card"]).exists()


def step11b_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP11B_OUTPUT_DIR)
    if folder is None:
        return False
    return (folder / config.STEP11B_FILES["metrics"]).exists()


def load_step10b(verbose: bool = True):
    """The saved Approach 2 model and its card.  Reloads with no custom_objects
    because the layer is registered."""
    keras = md._keras()
    folder = config.usable_output_dir(config.STEP10B_OUTPUT_DIR)
    if folder is None:
        raise AttentionError("the Step 10b folder is not usable here -- %s"
                             % config.why_unusable(config.STEP10B_OUTPUT_DIR))
    path = folder / config.STEP10B_FILES["model"]
    if not path.exists():
        raise AttentionError(
            "no saved Approach 2 model at %s. Run notebook section 10.2 (set "
            "OVERWRITE_STEP10B = True)." % path)
    # register the layer BEFORE loading: in a fresh kernel nothing has built
    # an Approach 2 model yet, so the class would be unknown to Keras
    model = keras.models.load_model(path, custom_objects=custom_objects())
    card_path = folder / config.STEP10B_FILES["model_card"]
    card = (json.loads(card_path.read_text(encoding="utf-8"))
            if card_path.exists() else {})
    history_path = folder / config.STEP10B_FILES["history"]
    history = (pd.read_csv(history_path) if history_path.exists()
               else pd.DataFrame())
    if verbose:
        show_note("Approach 2 loaded from **%s** (%s parameters). Not "
                  "retrained." % (path, f"{model.count_params():,}"))
    return {"model": model, "card": card, "history": history,
            "folder": str(folder), "status": "loaded_existing"}


def run_step10b(overwrite: bool | None = None, epochs: int | None = None,
                verbose: bool = True) -> dict[str, Any]:
    """Train (or load) Approach 2 on the Step 8 features."""
    overwrite = config.OVERWRITE_STEP10B if overwrite is None else overwrite
    if not overwrite and step10b_outputs_exist():
        return load_step10b(verbose=verbose)

    if verbose:
        show_title("Step 10.2 -- training Approach 2 (CNN-Attention-LSTM)",
                   level=3)

    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder8 is None:
        raise AttentionError("the Step 8 folder is not usable here -- %s"
                             % config.why_unusable(config.STEP8_OUTPUT_DIR))
    for key in ("Z_train", "y_train"):
        if not (folder8 / config.STEP8_FILES[key]).exists():
            raise AttentionError(
                "%s is missing:\n    %s\nRun Step 8 first. Approach 2 uses "
                "exactly the features Approach 1 used."
                % (key, folder8 / config.STEP8_FILES[key]))
    Z_train = pd.read_csv(folder8 / config.STEP8_FILES["Z_train"],
                          low_memory=False)
    y_train = pd.read_csv(folder8 / config.STEP8_FILES["y_train"],
                          low_memory=False)
    meta_train = None
    if config.INPUT_MODE == "time_window":
        folder5 = config.usable_output_dir(config.STEP5_OUTPUT_DIR)
        meta_train = pd.read_csv(folder5 / config.STEP5_FILES["meta_train"],
                                 low_memory=False)

    model, history, info = train_attention(Z_train, y_train, epochs=epochs,
                                           meta_train=meta_train,
                                           verbose=False)
    checks, checks_ok = protocol_checks(info, verbose=verbose)

    loaded = ev.load_evaluation_inputs(verbose=False)
    card = build_model_card(model, info, loaded.get("label_mapping"))

    out: dict[str, Any] = {"model": model, "history": history, "info": info,
                           "card": card, "checks": checks,
                           "checks_ok": checks_ok, "status": "trained"}
    if verbose:
        show_table(architecture_table(model), all_columns=True,
                   note="Approach 2. The attention layer pools %d CNN "
                        "feature-map positions; everything else is Approach "
                        "1's." % info["attention_positions"])
        show_table(kv_table([
            ("parameters", f"{model.count_params():,}"),
            ("epochs run", info["epochs_run"]),
            ("best epoch", info["best_epoch"]),
            ("validation macro-F1", "%.4f" % info["best_val_macro_f1"]),
            ("training seconds", "%.1f" % info["seconds"]),
            ("attention positions", info["attention_positions"]),
        ]), note="Early stopping on validation macro-F1 with the best weights "
                 "restored, exactly as in Approach 1.")

    folder = prepare_output_dir(config.STEP10B_OUTPUT_DIR,
                                "the Approach 2 model", verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP10B_FILES
    model.save(folder / names["model"])
    history.to_csv(folder / names["history"], index=False)
    architecture_table(model).to_csv(folder / names["architecture"],
                                     index=False)
    (folder / names["model_card"]).write_text(
        json.dumps(card, indent=2, default=float), encoding="utf-8")
    checks.to_csv(folder / names["report"], index=False)
    plot_curves(history, folder / names["loss_curve"], "loss",
                "Approach 2: training and validation loss", verbose=verbose)
    plot_curves(history, folder / names["macro_f1_curve"],
                config.EARLY_STOP_METRIC,
                "Approach 2: validation macro-F1 (the early-stopping signal)",
                verbose=verbose)

    # the saved model must reload and predict identically, or the file is
    # useless for Step 11b and for anyone reading the thesis later
    keras = md._keras()
    reloaded = keras.models.load_model(folder / names["model"],
                                       custom_objects=custom_objects())
    sample = md.arrange_inputs(Z_train.head(min(64, len(Z_train))),
                               y_train.head(min(64, len(y_train))), None,
                               verbose=False)[0]
    difference = float(np.abs(md.predict_proba(model, sample)
                              - md.predict_proba(reloaded, sample)).max())
    out["reload_difference"] = difference
    if difference > 1e-6:
        raise AttentionError(
            "the saved Approach 2 model does not reload identically (max "
            "difference %.3e). The attention layer's serialisation is wrong "
            "and the saved file cannot be trusted." % difference)
    if verbose:
        print("  [OK]    saved model reloads identically (max difference "
              "%.1e)" % difference)
    out.update({"status": "trained", "folder": str(folder), "ok": checks_ok})
    return out


def run_step11b(model=None, overwrite: bool | None = None,
                verbose: bool = True) -> dict[str, Any]:
    """Evaluate Approach 2 with the UNCHANGED Step 11 evaluate(), then compare."""
    overwrite = config.OVERWRITE_STEP10B if overwrite is None else overwrite
    folder = config.usable_output_dir(config.STEP11B_OUTPUT_DIR)
    if not overwrite and step11b_outputs_exist() and model is None:
        names = config.STEP11B_FILES
        out: dict[str, Any] = {"status": "loaded_existing",
                               "folder": str(folder)}
        out["metrics"] = json.loads(
            (folder / names["metrics"]).read_text(encoding="utf-8"))
        for key in ("per_class", "attention", "attention_per_class",
                    "comparison", "per_class_diff", "report"):
            path = folder / names[key]
            if path.exists():
                out[key] = pd.read_csv(path)
        path = folder / names["selection"]
        if path.exists():
            out["selection"] = json.loads(path.read_text(encoding="utf-8"))
        if verbose:
            show_note("Approach 2's evaluation loaded from disk. Set "
                      "`OVERWRITE_STEP10B = True` to recompute.")
        return out

    if verbose:
        show_title("Step 11 -- evaluating Approach 2", level=3)

    approach1_before = approach1_metrics()
    if model is None:
        model = load_step10b(verbose=False)["model"]
    card = {}
    folder10b = config.usable_output_dir(config.STEP10B_OUTPUT_DIR)
    if folder10b is not None:
        path = folder10b / config.STEP10B_FILES["model_card"]
        if path.exists():
            card = json.loads(path.read_text(encoding="utf-8"))

    loaded = ev.load_evaluation_inputs(verbose=False)
    Z_test, y_test = loaded["Z_test"], loaded["y_test"]
    meta_test = loaded.get("meta_test")
    flags = None
    if loaded["X_train"] is not None and loaded["X_test"] is not None:
        flags = ev.duplicate_mask(loaded["X_train"], loaded["X_test"])

    X_test_arr, _labels, row_index = md.arrange_inputs(Z_test, y_test,
                                                       meta_test, verbose=False)
    if flags is not None and len(row_index) != len(flags):
        flags = np.asarray(flags)[row_index]
    training_seconds = (card.get("training") or {}).get("seconds")
    result = ev.evaluate(model, X_test_arr, y_test, loaded["label_mapping"],
                         meta_test=meta_test, duplicate_flags=flags,
                         model_name="Approach 2: FS -> AE -> CNN-Attention-LSTM",
                         training_seconds=training_seconds, verbose=False)

    alpha = attention_weights(model, X_test_arr)
    label_names = result["label_names"]
    overall, per_class = attention_summary(alpha, result["y_pred"], label_names)
    selection = select_primary_approach(
        float((card.get("training") or {}).get("best_val_macro_f1", float("nan"))))

    metrics = {
        "config": getattr(config, "APPROACH2_KEY", "approach2_attention"),
        "label": dict(config.STEP12_DISPLAY_ORDER).get(
            getattr(config, "APPROACH2_KEY", "approach2_attention"),
            "Approach 2: FS -> AE -> CNN-Attention-LSTM"),
        "feature_source": "selected30", "use_ae": True, "use_pca": False,
        "model_type": "cnn_attention_lstm",
        "n_input_features": int(config.N_SELECTED_FEATURES),
        "latent_dim": int(Z_test.shape[1]),
        "model_input_shape": list(card.get("input_shape_per_sample", [])),
        "parameters": int(model.count_params()),
        "seed": (card.get("training") or {}).get("seed", config.RANDOM_STATE),
        "training_seconds": training_seconds,
        "best_epoch": (card.get("training") or {}).get("best_epoch"),
        "best_val_macro_f1": (card.get("training") or {}).get(
            "best_val_macro_f1"),
        "full": result["full"], "duplicate_free": result["duplicate_free"],
        "binary": result["binary"], "inference": result["inference"],
        "attention": {"positions": int(alpha.shape[1]),
                      "entropy": overall.attrs.get("entropy")},
        "selection": selection,
        "reused_step7_10": False,
    }

    approach1 = approach1_metrics()
    comparison = comparison_frame(metrics, approach1)
    folder11 = config.usable_output_dir(config.STEP11_OUTPUT_DIR)
    out = {"metrics": metrics, "result": result, "alpha": alpha,
           "attention": overall, "attention_per_class": per_class,
           "selection": selection, "comparison": comparison,
           "status": "written"}

    saved_folder = prepare_output_dir(config.STEP11B_OUTPUT_DIR,
                                      "the Approach 2 evaluation",
                                      verbose=verbose)
    if saved_folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP11B_FILES
    (saved_folder / names["metrics"]).write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    result["per_class"].to_csv(saved_folder / names["per_class"], index=False)
    result["confusion_counts"].to_csv(saved_folder / names["confusion_counts"])
    result["confusion_normalised"].to_csv(
        saved_folder / names["confusion_rownorm"])
    ev.build_predictions_frame(result, meta_test, row_index).to_csv(
        saved_folder / names["predictions"], index=False)
    overall.to_csv(saved_folder / names["attention"], index=False)
    per_class.to_csv(saved_folder / names["attention_per_class"], index=False)
    (saved_folder / names["selection"]).write_text(
        json.dumps(selection, indent=2, default=float), encoding="utf-8")
    if len(comparison):
        comparison.to_csv(saved_folder / names["comparison"], index=False)
    difference = per_class_difference(folder11, saved_folder)
    if len(difference):
        difference.to_csv(saved_folder / names["per_class_diff"], index=False)
    out["per_class_diff"] = difference

    plot_attention_heatmap(per_class, saved_folder / names["chart_attention"],
                           verbose=verbose)
    plot_approach_comparison(comparison,
                             saved_folder / names["chart_comparison"],
                             verbose=verbose)

    checks = [assert_approach1_unchanged(approach1_before, verbose=False)]
    checks.append({
        "check": "the attention weights are a distribution",
        "result": "[OK]" if (alpha >= 0).all() and
                  np.abs(alpha.sum(axis=1) - 1).max() < 1e-6 else "[FAIL]",
        "detail": "%d flows x %d positions, max |sum - 1| = %.1e"
                  % (alpha.shape[0], alpha.shape[1],
                     float(np.abs(alpha.sum(axis=1) - 1).max()))})
    checks.append({
        "check": "the test set was used once, for scoring only",
        "result": "[OK]",
        "detail": "the primary approach is chosen on validation macro-F1 "
                  "(%s)" % selection["reason"]})
    report = pd.DataFrame(checks)
    report.to_csv(saved_folder / names["report"], index=False)
    out.update({"report": report, "folder": str(saved_folder),
                "ok": bool((report["result"].isin(["[OK]", "[SKIP]"])).all())})

    if verbose:
        show_table(report, note="Approach 1's saved metrics are re-read after "
                                "Approach 2 runs and must be identical.")
        if len(comparison):
            show_table(comparison, all_columns=True,
                       note="Approach 2 minus Approach 1 on the same test set, "
                            "scored by the same Step 11 evaluate().")
        show_table(kv_table([
            ("primary approach", selection["primary"]),
            ("chosen on", selection["chosen_on"]),
            ("Approach 1 validation macro-F1",
             "%s" % selection["approach1_validation_macro_f1"]),
            ("Approach 2 validation macro-F1",
             "%.4f" % selection["approach2_validation_macro_f1"]),
            ("reason", selection["reason"]),
        ]), note="The test set took no part in this choice.")
    return out
