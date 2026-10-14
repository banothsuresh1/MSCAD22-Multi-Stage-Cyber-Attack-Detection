"""
src/mscad/models.py -- STEP 10: the proposed hybrid CNN-LSTM.

Input   C:\\MSCAD2022\\outputs\\step8\\   Z_train.csv, y_train.csv  (and Z_test for
                                        Step 11 only -- never read while training)
        C:\\MSCAD2022\\outputs\\step5\\   meta_train.csv / meta_test.csv, for
                                        INPUT_MODE = "time_window"
        Step 9's make_training_sets() and class weights
Output  C:\\MSCAD2022\\outputs\\step10\\

Everything here is a REUSABLE function.  Step 12's baselines call the same
builder, the same training loop and the same evaluation, so a comparison between
models is a comparison of architectures and not of harnesses.

Three things that are easy to get wrong, and how they are handled
-----------------------------------------------------------------
1. LOGITS, NOT PROBABILITIES.  The last Dense layer has no activation, and the
   loss is SparseCategoricalCrossentropy(from_logits=True), which applies the
   softmax internally.  Softmax is applied exactly once, in predict_proba().
   Putting softmax on the output layer AND using from_logits=True would apply it
   twice: the gradients would be wrong and the model would train badly while
   still producing plausible-looking probabilities.
2. THE VALIDATION LOSS IS UNWEIGHTED.  Keras applies `class_weight` to training
   batches only, which is what we want: the validation signal must measure the
   model on the data as it really is.
3. EARLY STOPPING WATCHES MACRO-F1, NOT LOSS.  With a 3849:1 imbalance the loss
   is dominated by the majority class, and a model that quietly stops predicting
   the 23-row classes can still improve it.  MacroF1Callback computes macro-F1 on
   the (unweighted) validation part each epoch and EarlyStopping maximises it.

The test set is never read here
-------------------------------
Training, early stopping and the class weights all come from the training rows.
Step 11 is the only place X_test is touched.
"""

from __future__ import annotations

import json
import os
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

from . import autoencoder as ae  # noqa: E402
from . import data_loading as dl  # noqa: E402
from . import imbalance as im  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = "Label_encoded"
VALID_INPUT_MODES = ("feature_seq", "time_window")


class ModelError(Exception):
    """Raised when Step 10 cannot continue without guessing."""


_keras = ae._keras
set_all_seeds = ae.set_all_seeds
framework_versions = ae.framework_versions


# ===========================================================================
# 10.1  ARRANGING THE INPUT
# ===========================================================================
def as_feature_sequence(Z: pd.DataFrame) -> np.ndarray:
    """(n, d) -> (n, d, 1).  Each latent feature becomes one "step".

    Not a time sequence: see the INPUT_MODE note in config.py.  The latent units
    have no natural order, so which feature sits next to which is an arbitrary
    consequence of how the encoder initialised.
    """
    values = Z.to_numpy(dtype="float32")
    return values.reshape(values.shape[0], values.shape[1], 1)


def build_time_windows(Z: pd.DataFrame, y: pd.DataFrame, meta: pd.DataFrame,
                       window: int | None = None, verbose: bool = True):
    """Windows of consecutive flows within one capture, labelled by the last.

    Returns (X, y_encoded, index) where `index` gives the row of Z that each
    window ends on, so a prediction can still be traced back to its flow.

    Windows never cross a capture boundary, and this function is called
    SEPARATELY on the training rows and on the test rows, so no window can ever
    straddle the train/test split.
    """
    window = config.WINDOW_SIZE if window is None else window
    needed = ["Source_Capture", "Timestamp"]
    missing = [c for c in needed if c not in meta.columns]
    if missing:
        raise ModelError(
            "INPUT_MODE = \"time_window\" needs %s in the Step 5 metadata; got %s"
            % (missing, list(meta.columns)))
    if len(meta) != len(Z) or len(y) != len(Z):
        raise ModelError(
            "Z, y and the metadata must have the same number of rows: %s"
            % {"Z": len(Z), "y": len(y), "meta": len(meta)})

    values = Z.to_numpy(dtype="float32")
    labels = y[TARGET_COLUMN].to_numpy()
    order_key = pd.DataFrame({
        "capture": meta["Source_Capture"].to_numpy(),
        "time": pd.to_datetime(meta["Timestamp"], errors="coerce").to_numpy(),
        "row": np.arange(len(meta)),
    })

    windows, window_labels, end_rows = [], [], []
    skipped = {}
    for capture, part in order_key.groupby("capture", sort=True):
        part = part.sort_values(["time", "row"], kind="stable")
        rows = part["row"].to_numpy()
        if len(rows) < window:
            skipped[str(capture)] = int(len(rows))
            continue
        for stop in range(window - 1, len(rows)):
            selected = rows[stop - window + 1:stop + 1]
            windows.append(values[selected])
            window_labels.append(labels[rows[stop]])
            end_rows.append(int(rows[stop]))

    if not windows:
        raise ModelError(
            "no window of %d consecutive flows fits inside any capture on this "
            "side of the split (rows per capture: %s). Lower WINDOW_SIZE or use "
            "INPUT_MODE = \"feature_seq\"." % (window, skipped))

    X = np.stack(windows).astype("float32")
    if verbose:
        show_table(kv_table([
            ("window size", window),
            ("rows in", f"{len(Z):,}"),
            ("windows out", f"{len(X):,}"),
            ("shape", str(X.shape)),
            ("captures too short to fill a window", skipped or "(none)"),
        ]), note="Each window is %d consecutive flows from ONE capture, ordered by "
                 "Timestamp and labelled by the last flow. Built separately on each "
                 "side of the split, so no window crosses it." % window)
    return X, np.asarray(window_labels), np.asarray(end_rows, dtype="int64")


def arrange_inputs(Z: pd.DataFrame, y: pd.DataFrame, meta: pd.DataFrame | None = None,
                   mode: str | None = None, verbose: bool = False):
    """Dispatch on INPUT_MODE.  Returns (X, y_encoded, row_index)."""
    mode = config.INPUT_MODE if mode is None else mode
    if mode not in VALID_INPUT_MODES:
        raise ModelError(
            "INPUT_MODE is %r; it must be one of %s"
            % (mode, ", ".join(repr(m) for m in VALID_INPUT_MODES)))
    if mode == "feature_seq":
        return (as_feature_sequence(Z), y[TARGET_COLUMN].to_numpy(),
                np.arange(len(Z), dtype="int64"))
    if meta is None:
        raise ModelError(
            "INPUT_MODE = \"time_window\" needs the Step 5 metadata "
            "(meta_train.csv / meta_test.csv) to order the flows.")
    return build_time_windows(Z, y, meta, verbose=verbose)


# ===========================================================================
# 10.2  THE ARCHITECTURE
# ===========================================================================
def build_cnn_lstm(input_shape: tuple[int, int], n_classes: int,
                   filters: Sequence[int] | None = None,
                   kernel: int | None = None, pool: int | None = None,
                   lstm_units: int | None = None, dropout: float | None = None,
                   dense_units: int | None = None, lr: float | None = None,
                   compile_model: bool = True):
    """Conv1D -> BN -> Conv1D -> MaxPool -> Dropout -> LSTM -> Dropout -> Dense.

    The output layer has NO activation: it emits logits, and the loss applies the
    softmax.  predict_proba() is the only place a softmax is applied.
    """
    keras = _keras()
    filters = list(config.CNN_FILTERS if filters is None else filters)
    kernel = config.CNN_KERNEL if kernel is None else kernel
    pool = config.CNN_POOL if pool is None else pool
    lstm_units = config.LSTM_UNITS if lstm_units is None else lstm_units
    dropout = config.DROPOUT if dropout is None else dropout
    dense_units = config.DENSE_UNITS if dense_units is None else dense_units
    lr = config.CLF_LR if lr is None else lr

    steps = int(input_shape[0])
    if steps < kernel:
        raise ModelError(
            "the input has %d steps but the convolution kernel is %d; lower "
            "CNN_KERNEL or use a longer input." % (steps, kernel))

    inputs = keras.Input(shape=tuple(input_shape), name="input")
    x = inputs
    for i, n_filters in enumerate(filters, start=1):
        x = keras.layers.Conv1D(n_filters, kernel, activation="relu",
                                padding="same", name="conv%d" % i)(x)
        if i == 1:
            x = keras.layers.BatchNormalization(name="batchnorm")(x)
    # pooling halves the steps, so only pool when there is something to halve
    if steps >= pool * 2:
        x = keras.layers.MaxPooling1D(pool, name="maxpool")(x)
    x = keras.layers.Dropout(dropout, name="dropout_cnn")(x)
    x = keras.layers.LSTM(lstm_units, name="lstm")(x)
    x = keras.layers.Dropout(dropout, name="dropout_lstm")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    # no activation: these are LOGITS
    outputs = keras.layers.Dense(n_classes, name="logits")(x)

    model = keras.Model(inputs, outputs, name="cnn_lstm")
    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
    return model


def architecture_table(model) -> pd.DataFrame:
    """One row per layer: output shape, activation and parameter count."""
    rows = []
    for layer in model.layers:
        cfg = layer.get_config()
        try:
            shape = tuple(layer.output.shape)
        except Exception:
            shape = "-"
        rows.append({"layer": layer.name, "type": type(layer).__name__,
                     "output shape": str(shape),
                     "activation": cfg.get("activation", "-"),
                     "parameters": int(layer.count_params())})
    table = pd.DataFrame(rows)
    return pd.concat([table, pd.DataFrame([{
        "layer": "TOTAL", "type": "", "output shape": "", "activation": "",
        "parameters": int(table["parameters"].sum())}])], ignore_index=True)


def predict_logits(model, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """Raw model output.  Rows do NOT sum to 1 -- these are logits."""
    batch_size = config.CLF_BATCH_SIZE if batch_size is None else batch_size
    return np.asarray(model.predict(X, batch_size=batch_size, verbose=0))


def predict_proba(model, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """Softmax of the logits -- the ONLY place a softmax is applied."""
    logits = predict_logits(model, X, batch_size=batch_size)
    shifted = logits - logits.max(axis=1, keepdims=True)   # stable softmax
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


# ===========================================================================
# 10.3  TRAINING
# ===========================================================================
def _macro_f1_callback_class():
    """Build the callback class lazily, so this module imports without Keras.

    It puts an UNWEIGHTED validation macro-F1 into the epoch logs, which is what
    EarlyStopping then maximises.
    """
    keras = _keras()
    from sklearn.metrics import f1_score

    class _MacroF1(keras.callbacks.Callback):
        def __init__(self, X_val, y_val, batch_size):
            super().__init__()
            self.X_val = X_val
            self.y_val = np.asarray(y_val)
            self.batch_size = batch_size

        def on_epoch_end(self, epoch, logs=None):
            logs = logs if logs is not None else {}
            logits = self.model.predict(self.X_val, batch_size=self.batch_size,
                                        verbose=0)
            predicted = np.argmax(logits, axis=1)
            # unweighted, on the validation part exactly as it is
            logs["val_macro_f1"] = float(
                f1_score(self.y_val, predicted, average="macro", zero_division=0))

    return _MacroF1


def train_model(Z_train: pd.DataFrame, y_train: pd.DataFrame,
                meta_train: pd.DataFrame | None = None,
                seed: int | None = None, epochs: int | None = None,
                input_mode: str | None = None, verbose: bool = True):
    """Build and fit the CNN-LSTM on the Step 9 fit part.  Returns (model, info).

    The test set is not read, not referenced and not available to this function.
    """
    keras = _keras()
    seed = config.RANDOM_STATE if seed is None else seed
    epochs = config.CLF_EPOCHS if epochs is None else epochs
    input_mode = config.INPUT_MODE if input_mode is None else input_mode
    set_all_seeds(seed)

    # Step 9 owns the fit/validation split and the class weights
    Z_fit, y_fit, Z_val, y_val, class_weights, _sample_w = im.make_training_sets(
        Z_train, y_train, verbose=False)
    meta_fit = meta_val = None
    if input_mode == "time_window":
        if meta_train is None:
            raise ModelError(
                "INPUT_MODE = \"time_window\" needs meta_train.csv so the flows can "
                "be ordered inside each capture.")
        # Split the metadata with the SAME call, so it lands on the same rows:
        # make_training_sets is deterministic in (y, val_size, random_state) and
        # the strategy only affects the weights, never the positions.
        _zf, _yf, _zv, _yv, _cw, _sw = im.make_training_sets(
            meta_train.reset_index(drop=True), y_train, verbose=False)
        meta_fit, meta_val = _zf, _zv
        if len(meta_fit) != len(Z_fit) or len(meta_val) != len(Z_val):
            raise ModelError(
                "the metadata split does not line up with the feature split "
                "(%d/%d vs %d/%d rows)"
                % (len(meta_fit), len(meta_val), len(Z_fit), len(Z_val)))
        if not y_fit[TARGET_COLUMN].equals(_yf[TARGET_COLUMN]):
            raise ModelError(
                "the metadata split selected different rows from the feature split")

    X_fit, labels_fit, _idx_fit = arrange_inputs(Z_fit, y_fit, meta_fit,
                                                 mode=input_mode, verbose=False)
    X_val, labels_val, _idx_val = arrange_inputs(Z_val, y_val, meta_val,
                                                 mode=input_mode, verbose=False)

    n_classes = int(max(int(np.max(labels_fit)), int(np.max(labels_val))) + 1)
    model = build_cnn_lstm(X_fit.shape[1:], n_classes)

    macro_f1 = _macro_f1_callback_class()(X_val, labels_val, config.CLF_BATCH_SIZE)
    stopper = keras.callbacks.EarlyStopping(
        monitor=config.EARLY_STOP_METRIC, mode="max", patience=config.CLF_PATIENCE,
        restore_best_weights=True, verbose=0)

    started = time.perf_counter()
    history = model.fit(
        X_fit, labels_fit,
        validation_data=(X_val, labels_val),     # UNWEIGHTED, on purpose
        epochs=epochs, batch_size=config.CLF_BATCH_SIZE, shuffle=True,
        class_weight=im.keras_class_weight(class_weights),   # training only
        callbacks=[macro_f1, stopper],           # metric first, then the stopper
        verbose=0)
    seconds = time.perf_counter() - started

    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    best_epoch = int(frame[config.EARLY_STOP_METRIC].idxmax()) + 1

    info = {
        "seed": seed, "seconds": seconds, "epochs_run": len(frame),
        "epochs_requested": epochs, "best_epoch": best_epoch,
        "stopped_early": len(frame) < epochs,
        "best_val_macro_f1": float(frame[config.EARLY_STOP_METRIC].iloc[best_epoch - 1]),
        "val_loss_at_best": float(frame["val_loss"].iloc[best_epoch - 1]),
        "train_loss_at_best": float(frame["loss"].iloc[best_epoch - 1]),
        "fit_rows": len(X_fit), "validation_rows": len(X_val),
        "n_classes": n_classes, "input_mode": input_mode,
        "input_shape": tuple(int(v) for v in X_fit.shape[1:]),
        "class_weights": {int(k): float(v) for k, v in class_weights.items()},
    }

    if verbose:
        show_table(kv_table([
            ("input mode", input_mode),
            ("input shape per sample", str(info["input_shape"])),
            ("fit rows", f"{info['fit_rows']:,}"),
            ("validation rows", f"{info['validation_rows']:,}"),
            ("classes", n_classes),
            ("epochs run", "%d of %d" % (info["epochs_run"], epochs)),
            ("stopped early", "yes, patience %d" % config.CLF_PATIENCE
             if info["stopped_early"] else "no"),
            ("best epoch", best_epoch),
            ("best validation macro-F1", "%.4f" % info["best_val_macro_f1"]),
            ("validation loss there", "%.4f" % info["val_loss_at_best"]),
            ("training time", "%.1f s" % seconds),
            ("seed", seed),
        ]), note="The validation macro-F1 is UNWEIGHTED and early stopping "
                 "maximises it. The class weights applied to the training loss "
                 "came from Step 9, computed on the fit part alone.")
    return model, frame, info


# ===========================================================================
# 10.4  CURVES
# ===========================================================================
_TRAIN_COLOUR = "#2a78d6"       # categorical slot 1
_VAL_COLOUR = "#eb6834"         # categorical slot 2


def _curve(history: pd.DataFrame, columns, labels, title, ylabel, path,
           best_epoch=None, log=False, verbose=True):
    """One line per series, a legend, and the best epoch marked."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("  [SKIP]  matplotlib is not installed, so no chart was drawn.")
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colours = [_TRAIN_COLOUR, _VAL_COLOUR]

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    for column, label, colour in zip(columns, labels, colours):
        if column in history.columns:
            ax.plot(history["epoch"], history[column], linewidth=2.0,
                    color=colour, label=label)

    if best_epoch is not None and columns[-1] in history.columns:
        value = float(history[columns[-1]].iloc[best_epoch - 1])
        ax.scatter([best_epoch], [value], s=42, color=colours[len(columns) - 1],
                   zorder=5, edgecolor=dl._SURFACE, linewidth=2)
        ax.annotate("best epoch %d\n%.4f" % (best_epoch, value),
                    (best_epoch, value), textcoords="offset points",
                    xytext=(10, 8), fontsize=9, color=dl._INK)

    if log:
        ax.set_yscale("log")
    ax.set_xlabel("epoch", color=dl._INK_SOFT, fontsize=10)
    ax.set_ylabel(ylabel, color=dl._INK_SOFT, fontsize=10)
    ax.set_title(title, color=dl._INK, fontsize=12, pad=14)
    ax.set_xlim(0, float(history["epoch"].max()) * 1.16)

    ax.yaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(dl._GRID)
    ax.tick_params(colors=dl._INK_SOFT, labelsize=9)

    if len([c for c in columns if c in history.columns]) > 1:
        legend = ax.legend(frameon=False, loc="best", fontsize=9)
        for text in legend.get_texts():
            text.set_color(dl._INK_SOFT)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def plot_curves(history: pd.DataFrame, info: dict[str, Any], folder: Path,
                verbose: bool = True) -> dict[str, str]:
    """The loss curve and the validation macro-F1 curve."""
    out = {}
    loss = _curve(history, ["loss", "val_loss"],
                  ["train loss (class-weighted)", "validation loss (unweighted)"],
                  "CNN-LSTM training loss", "cross-entropy (log scale)",
                  folder / config.STEP10_FILES["loss_curve"],
                  best_epoch=info.get("best_epoch"), log=True, verbose=verbose)
    if loss is not None:
        out["loss_curve"] = str(loss)
    f1 = _curve(history, [config.EARLY_STOP_METRIC],
                ["validation macro-F1"],
                "Validation macro-F1 -- what early stopping maximises",
                "macro-F1", folder / config.STEP10_FILES["f1_curve"],
                best_epoch=info.get("best_epoch"), verbose=verbose)
    if f1 is not None:
        out["f1_curve"] = str(f1)
    return out


# ===========================================================================
# 10.5  SAVE / LOAD / RUN
# ===========================================================================
def build_model_card(info: dict[str, Any], architecture: pd.DataFrame,
                     label_mapping: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything needed to reproduce or to judge this model."""
    keras = _keras()
    import tensorflow as tf
    devices = [d.device_type for d in tf.config.list_physical_devices()]
    return {
        "name": "proposed hybrid CNN-LSTM",
        "pipeline": "Step 6 feature selection -> Step 7 autoencoder 30->15 -> "
                    "Step 10 CNN-LSTM",
        "input_mode": info["input_mode"],
        "input_shape_per_sample": list(info["input_shape"]),
        "input_mode_note":
            "feature_seq lays the 15 autoencoder features out as 15 steps of one "
            "channel. This is NOT a time sequence: the Conv1D layers learn "
            "patterns across neighbouring LATENT FEATURES and the LSTM reads "
            "their feature map as a sequence of learned patterns. The latent "
            "units have no natural order -- which feature sits at position 3 is "
            "an arbitrary consequence of how the encoder initialised -- so any "
            "claim about temporal attack structure needs "
            "INPUT_MODE = \"time_window\" instead."
            if info["input_mode"] == "feature_seq" else
            "time_window uses %d consecutive flows from one capture, ordered by "
            "Timestamp and labelled by the last flow. Windows are built "
            "separately on each side of the split. The Step 5 split was random "
            "and stratified, so consecutive rows on one side were not "
            "consecutive in the capture: a window spans more wall-clock time "
            "than it appears to." % config.WINDOW_SIZE,
        "architecture": {
            "cnn_filters": list(config.CNN_FILTERS), "cnn_kernel": config.CNN_KERNEL,
            "cnn_pool": config.CNN_POOL, "lstm_units": config.LSTM_UNITS,
            "dropout": config.DROPOUT, "dense_units": config.DENSE_UNITS,
            "output": "logits (no activation); the loss applies the softmax",
            "total_parameters": int(architecture["parameters"].iloc[-1]),
        },
        "training": {
            "epochs_requested": info["epochs_requested"],
            "epochs_run": info["epochs_run"], "best_epoch": info["best_epoch"],
            "batch_size": config.CLF_BATCH_SIZE, "learning_rate": config.CLF_LR,
            "patience": config.CLF_PATIENCE,
            "early_stop_metric": config.EARLY_STOP_METRIC,
            "early_stop_note": "macro-F1 on the UNWEIGHTED validation part. With a "
                               "3849:1 imbalance the loss is dominated by the "
                               "majority class, so a model that stops predicting "
                               "the rare classes can still improve it.",
            "best_val_macro_f1": info["best_val_macro_f1"],
            "training_seconds": round(info["seconds"], 3),
            "seed": info["seed"], "fit_rows": info["fit_rows"],
            "validation_rows": info["validation_rows"],
        },
        "imbalance": {
            "strategy": config.IMBALANCE_STRATEGY,
            "scheme": config.CLASS_WEIGHT_SCHEME,
            "max_weight": config.CLASS_WEIGHT_MAX,
            "normalised": config.NORMALISE_WEIGHTS,
            "class_weights": info["class_weights"],
            "note": "class weighting only -- no SMOTE and no resampling anywhere. "
                    "Weights apply to the training loss; the validation loss is "
                    "unweighted and the test set is never reweighted.",
        },
        "label_mapping": label_mapping or {},
        "versions": framework_versions(),
        "devices": devices,
        "test_set": "never read during training, early stopping, tuning or the "
                    "class weights",
    }


def step10_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP10_OUTPUT_DIR)
    if folder is None:
        return False
    return all((folder / config.STEP10_FILES[k]).exists()
               for k in ("model", "history", "model_card"))


def load_step10(verbose: bool = True) -> dict[str, Any]:
    """Read the trained model and its reports back instead of retraining."""
    folder = config.usable_output_dir(config.STEP10_OUTPUT_DIR)
    if folder is None:
        raise ModelError("the Step 10 folder is not usable here -- %s"
                         % config.why_unusable(config.STEP10_OUTPUT_DIR))
    keras = _keras()
    names = config.STEP10_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    out["model"] = keras.models.load_model(folder / names["model"])
    out["model_card"] = json.loads(
        (folder / names["model_card"]).read_text(encoding="utf-8"))
    for key in ("history", "architecture", "report"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    card = out["model_card"]
    out["info"] = {
        "input_mode": card["input_mode"],
        "input_shape": tuple(card["input_shape_per_sample"]),
        "best_epoch": card["training"]["best_epoch"],
        "epochs_run": card["training"]["epochs_run"],
        "best_val_macro_f1": card["training"]["best_val_macro_f1"],
        "seconds": card["training"]["training_seconds"],
        "seed": card["training"]["seed"],
        "class_weights": {int(k): float(v)
                          for k, v in card["imbalance"]["class_weights"].items()},
    }
    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP10 = False)"),
            ("folder", str(folder)),
            ("input mode", card["input_mode"]),
            ("parameters", f"{card['architecture']['total_parameters']:,}"),
            ("best epoch", card["training"]["best_epoch"]),
            ("best validation macro-F1",
             "%.4f" % card["training"]["best_val_macro_f1"]),
            ("training time", "%.1f s" % card["training"]["training_seconds"]),
        ]), note="Set `OVERWRITE_STEP10 = True` in config.py to retrain.")
    return out


def run_step10(overwrite: bool | None = None, seed: int | None = None,
               verbose: bool = True) -> dict[str, Any]:
    """STEP 10 end to end: arrange -> build -> train -> save."""
    overwrite = config.OVERWRITE_STEP10 if overwrite is None else overwrite
    if not overwrite and step10_outputs_exist():
        return load_step10(verbose=verbose)

    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder8 is None:
        raise ModelError("the Step 8 output folder is not usable here -- %s"
                         % config.why_unusable(config.STEP8_OUTPUT_DIR))
    names8 = config.STEP8_FILES
    missing = [names8[k] for k in ("Z_train", "y_train")
               if not (folder8 / names8[k]).exists()]
    if missing:
        raise ModelError("Step 10 needs %s in %s. Run Step 8 first."
                         % (missing, folder8))

    Z_train = pd.read_csv(folder8 / names8["Z_train"], low_memory=False)
    y_train = pd.read_csv(folder8 / names8["y_train"], low_memory=False)

    meta_train = None
    if config.INPUT_MODE == "time_window":
        folder5 = config.usable_output_dir(config.STEP5_OUTPUT_DIR)
        path = folder5 / config.STEP5_FILES["meta_train"] if folder5 else None
        if path is None or not path.exists():
            raise ModelError(
                "INPUT_MODE = \"time_window\" needs %s; run Step 5 first." % path)
        meta_train = pd.read_csv(path, low_memory=False)

    label_mapping = {}
    folder4 = config.usable_output_dir(config.STEP4_OUTPUT_DIR)
    if folder4 is not None:
        path = folder4 / config.STEP4_FILES["label_mapping"]
        if path.exists():
            label_mapping = json.loads(path.read_text(encoding="utf-8"))

    if verbose:
        show_title("Step 10 -- train the hybrid CNN-LSTM", level=3)
    model, history, info = train_model(Z_train, y_train, meta_train, seed=seed,
                                       verbose=verbose)
    architecture = architecture_table(model)
    if verbose:
        show_table(architecture,
                   note="The last layer has NO activation: it emits logits, and "
                        "the loss applies the softmax. `predict_proba()` is the "
                        "only place a softmax is applied.")

    card = build_model_card(info, architecture, label_mapping)
    report = pd.DataFrame([
        {"step": "10.1 arrange", "action": "input mode %s" % info["input_mode"],
         "detail": "shape per sample %s" % (info["input_shape"],)},
        {"step": "10.2 build", "action": "Conv1D x%d -> BN -> pool -> LSTM -> Dense"
                                         % len(config.CNN_FILTERS),
         "detail": "%s parameters, output = logits"
                   % f"{int(architecture['parameters'].iloc[-1]):,}"},
        {"step": "10.3 train", "action": "class-weighted loss, unweighted validation",
         "detail": "%d of %d epochs, best %d, early stop on %s"
                   % (info["epochs_run"], info["epochs_requested"],
                      info["best_epoch"], config.EARLY_STOP_METRIC)},
        {"step": "10.3 train", "action": "best validation macro-F1",
         "detail": "%.4f in %.1f s (seed %s)"
                   % (info["best_val_macro_f1"], info["seconds"], info["seed"])},
        {"step": "10.x not done", "action": "the test set was not read",
         "detail": "no training, early stopping, tuning or weighting used it"},
    ])

    out: dict[str, Any] = {"model": model, "history": history, "info": info,
                           "architecture": architecture, "model_card": card,
                           "report": report, "label_mapping": label_mapping}

    folder = prepare_output_dir(config.STEP10_OUTPUT_DIR, "the Step 10 outputs",
                                verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP10_FILES
    model.save(folder / names["model"])
    history.to_csv(folder / names["history"], index=False)
    architecture.to_csv(folder / names["architecture"], index=False)
    report.to_csv(folder / names["report"], index=False)
    (folder / names["model_card"]).write_text(json.dumps(card, indent=2),
                                              encoding="utf-8")
    out["charts"] = plot_curves(history, info, folder, verbose=verbose)
    out.update({"status": "written", "ok": True, "folder": str(folder)})

    if verbose:
        show_table(kv_table([
            ("parameters", f"{int(architecture['parameters'].iloc[-1]):,}"),
            ("best epoch", info["best_epoch"]),
            ("best validation macro-F1", "%.4f" % info["best_val_macro_f1"]),
            ("training time", "%.1f s" % info["seconds"]),
            ("output folder", str(folder)),
        ]), note="Step 11 evaluates this model on the untouched test set.")
    return out
