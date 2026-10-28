"""
src/mscad/autoencoder.py -- STEP 7: train an autoencoder.  STEP 8: encode both sets.

Input   C:\\MSCAD2022\\outputs\\step6\\   (Step 6's outputs)
            X_train_selected.csv, X_test_selected.csv   30 features, [0, 1]
            y_train.csv, y_test.csv, selected_features.json
Output  C:\\MSCAD2022\\outputs\\step7\\   the trained autoencoder and its reports
        C:\\MSCAD2022\\outputs\\step8\\   Z_train.csv / Z_test.csv (15 columns)

What the autoencoder is for
---------------------------
It compresses the 30 selected features into AE_LATENT_DIM numbers by learning to
rebuild its own input from them.  Whatever survives that squeeze is the structure
the 30 features share; whatever does not was redundant or noise.  The result is a
denser input for the classifiers in Steps 10-12.

It is UNSUPERVISED: the loss is reconstruction error against the input itself, so
no label is ever a target.  Labels appear in Step 7 for exactly one purpose -- to
STRATIFY the inner split used for early stopping, so the 23-row classes are
represented in the validation part instead of landing entirely in the fit part by
chance.  Nothing about the encoding depends on them.

Why sigmoid on the output
-------------------------
Step 5's MinMax scaler put every training feature into [0, 1], so a sigmoid can
reach every legal value and nothing else.  A linear output would be free to
predict -3 or 7, which no input can be, and would waste capacity learning not to.

Nothing here touches the test set in Step 7
-------------------------------------------
Training, the inner split and early stopping all run on X_train_selected alone.
X_test_selected is read for the first time in Step 8, and only to run the frozen
encoder over it.
"""

from __future__ import annotations

import json
import os
import random
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

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

LABEL_COLUMNS = ["Label", "Label_encoded", "Label_binary"]
TARGET_COLUMN = "Label_encoded"


class AutoencoderError(Exception):
    """Raised when Step 7 or Step 8 cannot continue without guessing."""


# ===========================================================================
# framework and determinism
# ===========================================================================
def _keras():
    """Import Keras, or explain exactly how to install it."""
    try:
        # quieten the C++ logger before TensorFlow is imported, or every run
        # starts with a page of oneDNN and AVX notices
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras
        return keras
    except ImportError as exc:
        raise AutoencoderError(
            "TensorFlow/Keras is needed for Steps 7-8.  Install it with:\n"
            '    "%s" -m pip install "tensorflow>=2.21,<3"\n'
            "On Windows that wheel is CPU-only, which is all these steps need."
            % sys.executable) from exc


def framework_versions() -> dict[str, str]:
    """What actually ran, recorded in ae_config.json so a result is reproducible."""
    keras = _keras()
    import tensorflow as tf
    return {"framework": config.AE_FRAMEWORK, "tensorflow": tf.__version__,
            "keras": keras.__version__, "python": sys.version.split()[0],
            "numpy": np.__version__}


def set_all_seeds(seed: int | None = None) -> int:
    """Seed python, numpy and Keras, and ask for deterministic kernels.

    Without the op-determinism call the same seed still gives different results
    run to run: several CPU kernels reduce in a non-deterministic order.
    """
    seed = config.RANDOM_STATE if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras = _keras()
    keras.utils.set_random_seed(seed)
    import tensorflow as tf
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:                      # older builds without the call
        pass
    return seed


def z_columns(n: int | None = None) -> list[str]:
    """z01 ... z15 -- zero-padded, so they sort correctly as text."""
    n = config.AE_LATENT_DIM if n is None else n
    width = max(2, len(str(n)))
    return ["%s%0*d" % (config.Z_COLUMN_PREFIX, width, i + 1) for i in range(n)]


# ===========================================================================
# 7.1  INPUT
# ===========================================================================
def load_step6_outputs(verbose: bool = True) -> dict[str, Any]:
    """Read the selected features, the labels and selected_features.json."""
    folder = config.usable_output_dir(config.STEP6_OUTPUT_DIR)
    if folder is None:
        raise AutoencoderError("the Step 6 output folder is not usable here -- %s"
                               % config.why_unusable(config.STEP6_OUTPUT_DIR))

    names = config.STEP6_FILES
    wanted = ("X_train_selected", "X_test_selected", "y_train", "y_test", "selected")
    missing = [names[k] for k in wanted if not (folder / names[k]).exists()]
    if missing:
        raise AutoencoderError(
            "Steps 7-8 need %s in %s. Run Step 6 first (set OVERWRITE_STEP6 = True "
            "if its outputs are stale)." % (missing, folder))

    out: dict[str, Any] = {"folder": str(folder)}
    for key in ("X_train_selected", "X_test_selected", "y_train", "y_test"):
        out[key] = pd.read_csv(folder / names[key], low_memory=False)
    picked = json.loads((folder / names["selected"]).read_text(encoding="utf-8"))
    out["feature_order"] = list(picked["features"])
    return out


def check_inputs(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.DataFrame, y_test: pd.DataFrame,
                 feature_order: Sequence[str], verbose: bool = True):
    """[OK]/[FAIL] on everything that must hold before the autoencoder is built."""
    feature_order = list(feature_order)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    train_ok = list(X_train.columns) == feature_order
    add("X_train_selected matches selected_features.json, in order", train_ok,
        "%d features" % len(feature_order) if train_ok
        else "got %s..." % list(X_train.columns)[:4])
    test_ok = list(X_test.columns) == feature_order
    add("X_test_selected matches it too, in the same order", test_ok,
        "%d features" % X_test.shape[1] if test_ok
        else "got %s..." % list(X_test.columns)[:4])

    rows_ok = len(X_train) == len(y_train) and len(X_test) == len(y_test)
    add("features and labels have matching row counts", rows_ok,
        "train %s/%s, test %s/%s" % (f"{len(X_train):,}", f"{len(y_train):,}",
                                     f"{len(X_test):,}", f"{len(y_test):,}"))

    leaked = [c for c in LABEL_COLUMNS + list(config.ID_COLUMNS_NOT_FOR_TRAINING)
              if c in X_train.columns]
    add("no label or identifier column among the features", not leaked,
        "none" if not leaked else "found %s" % leaked)

    block = X_train.select_dtypes(include=[np.number]).to_numpy(dtype="float64")
    n_nan, n_inf = int(np.isnan(block).sum()), int(np.isinf(block).sum())
    add("no NaN or inf in X_train_selected", n_nan == 0 and n_inf == 0,
        "%d NaN, %d inf" % (n_nan, n_inf))

    low, high = float(block.min()), float(block.max())
    in_unit = low >= -1e-9 and high <= 1 + 1e-9
    add("training features are inside [0, 1] (a sigmoid output can reach them)",
        in_unit, "[%.6f, %.6f]" % (low, high))

    enough = len(X_train) > config.AE_BATCH_SIZE
    add("more training rows than one batch", enough,
        "%s rows, batch %d" % (f"{len(X_train):,}", config.AE_BATCH_SIZE))

    latent_ok = config.AE_LATENT_DIM <= len(feature_order)
    add("the latent dimension is no larger than the input "
        "(equal means a degenerate, non-compressing autoencoder -- by design "
        "for the AE=30 sensitivity run)", latent_ok,
        "%d -> %s -> %d" % (len(feature_order), config.AE_HIDDEN,
                            config.AE_LATENT_DIM))

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table, note="A `[FAIL]` on the first two rows means Step 6 and "
                               "Step 7 disagree about the features; re-run Step 6.")

    # Ordered by how specific the diagnosis is, not by the order of the checks.
    # A leaked column also makes the frames disagree about their columns, and
    # "Label_encoded is among the features" sends you to the right place.
    if leaked:
        raise AutoencoderError(
            "[FAIL] %s is among the features. The autoencoder would learn to "
            "reconstruct the answer." % leaked)
    if not train_ok or not test_ok:
        raise AutoencoderError(
            "[FAIL] the selected-feature columns do not match selected_features.json "
            "in name and order, so the encoder would be fed the wrong columns. "
            "Re-run Step 6 with OVERWRITE_STEP6 = True.")
    if not rows_ok:
        raise AutoencoderError(
            "[FAIL] the features and labels disagree on their row counts: %s"
            % {"X_train": len(X_train), "y_train": len(y_train),
               "X_test": len(X_test), "y_test": len(y_test)})
    if not latent_ok:
        raise AutoencoderError(
            "[FAIL] AE_LATENT_DIM is %d but there are only %d features -- the "
            "latent width cannot exceed the input."
            % (config.AE_LATENT_DIM, len(feature_order)))
    return table, all_ok
# ===========================================================================
# 7.2  BUILD
# ===========================================================================
def build_autoencoder(n_features: int, latent_dim: int | None = None,
                      hidden: Sequence[int] | None = None):
    """Return (autoencoder, encoder).  The encoder SHARES the trained weights.

    Both models are built over the same layer objects, so after fit() the encoder
    is already trained -- it is the first half of the autoencoder, not a copy.
    """
    keras = _keras()
    latent_dim = config.AE_LATENT_DIM if latent_dim is None else latent_dim
    hidden = list(config.AE_HIDDEN if hidden is None else hidden)

    inputs = keras.Input(shape=(n_features,), name="input")
    x = inputs
    for i, units in enumerate(hidden, start=1):
        x = keras.layers.Dense(units, activation=config.AE_ACTIVATION,
                               name="encoder_%d" % i)(x)
    latent = keras.layers.Dense(latent_dim, activation=config.AE_ACTIVATION,
                                name="latent")(x)

    y = latent
    for i, units in enumerate(reversed(hidden), start=1):
        y = keras.layers.Dense(units, activation=config.AE_ACTIVATION,
                               name="decoder_%d" % i)(y)
    outputs = keras.layers.Dense(n_features, activation=config.AE_OUTPUT_ACT,
                                 name="reconstruction")(y)

    autoencoder = keras.Model(inputs, outputs, name="autoencoder")
    encoder = keras.Model(inputs, latent, name="encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=config.AE_LR),
                        loss=config.AE_LOSS)
    return autoencoder, encoder


def architecture_table(autoencoder, latent_name: str = "latent") -> pd.DataFrame:
    """One row per layer: units, activation, parameter count, and its half."""
    rows = []
    half = "encoder"
    for layer in autoencoder.layers:
        cfg = layer.get_config()
        units = cfg.get("units", "-")
        activation = cfg.get("activation", "-")
        rows.append({"layer": layer.name, "units": units, "activation": activation,
                     "parameters": int(layer.count_params()), "half": half})
        if layer.name == latent_name:
            half = "decoder"
    table = pd.DataFrame(rows)
    table = pd.concat([table, pd.DataFrame([{
        "layer": "TOTAL", "units": "", "activation": "",
        "parameters": int(table["parameters"].sum()), "half": ""}])],
        ignore_index=True)
    return table


# ===========================================================================
# 7.3  TRAIN -- training rows only
# ===========================================================================
def train_autoencoder(X_train: pd.DataFrame, y_train: pd.DataFrame,
                      epochs: int | None = None, latent_dim: int | None = None,
                      verbose: bool = True):
    """Fit on the training rows, early-stopping on an inner stratified split.

    `latent_dim` defaults to None, which means config.AE_LATENT_DIM -- the
    behaviour every existing caller already had.  It is a parameter so the
    bottleneck-sensitivity run can ask for a width explicitly instead of
    depending on what config happens to hold at the time.
    """
    keras = _keras()
    from sklearn.model_selection import train_test_split

    epochs = config.AE_EPOCHS if epochs is None else epochs
    set_all_seeds()

    values = X_train.to_numpy(dtype="float32")
    y = y_train[TARGET_COLUMN].to_numpy()

    counts = pd.Series(y).value_counts()
    stratify = y if int(counts.min()) >= 2 else None
    if stratify is None and verbose:
        show_note("**[WARN]** a class has fewer than 2 rows, so the inner split "
                  "cannot be stratified and falls back to a plain random split.")

    positions = np.arange(len(values), dtype="int64")
    fit_idx, val_idx = train_test_split(
        positions, test_size=config.AE_VAL_SIZE, random_state=config.RANDOM_STATE,
        shuffle=True, stratify=stratify)
    X_fit, X_val = values[fit_idx], values[val_idx]

    autoencoder, encoder = build_autoencoder(X_train.shape[1],
                                             latent_dim=latent_dim)
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.AE_PATIENCE,
        restore_best_weights=True, verbose=0)

    started = time.perf_counter()
    history = autoencoder.fit(
        X_fit, X_fit,                     # the target IS the input: unsupervised
        validation_data=(X_val, X_val),
        epochs=epochs, batch_size=config.AE_BATCH_SIZE,
        shuffle=True, callbacks=[stopper], verbose=0)
    seconds = time.perf_counter() - started

    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))

    best_epoch = int(frame["val_loss"].idxmin()) + 1
    info = {
        "seconds": seconds,
        "epochs_run": len(frame),
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "stopped_early": len(frame) < epochs,
        "train_mse": float(frame["loss"].iloc[best_epoch - 1]),
        "val_mse": float(frame["val_loss"].iloc[best_epoch - 1]),
        "final_train_mse": float(frame["loss"].iloc[-1]),
        "final_val_mse": float(frame["val_loss"].iloc[-1]),
        "fit_rows": len(fit_idx), "validation_rows": len(val_idx),
        "fit_idx": fit_idx, "val_idx": val_idx,
    }

    if verbose:
        show_table(kv_table([
            ("rows trained on", f"{len(fit_idx):,}"),
            ("inner validation rows", f"{len(val_idx):,}"),
            ("epochs requested", epochs),
            ("epochs run", info["epochs_run"]),
            ("stopped early", "yes, patience %d" % config.AE_PATIENCE
             if info["stopped_early"] else "no, ran the full %d" % epochs),
            ("best epoch (weights restored to it)", best_epoch),
            ("train reconstruction MSE", "%.6f" % info["train_mse"]),
            ("validation reconstruction MSE", "%.6f" % info["val_mse"]),
            ("training time", "%.1f s" % seconds),
        ]), note="The target is the input itself, so these are reconstruction "
                 "errors, not classification scores. No label was a target.")
    return autoencoder, encoder, frame, info


def per_feature_mse(autoencoder, X: pd.DataFrame,
                    verbose: bool = True) -> pd.DataFrame:
    """Reconstruction MSE per feature -- which of the 30 the AE rebuilds worst."""
    values = X.to_numpy(dtype="float32")
    rebuilt = autoencoder.predict(values, verbose=0)
    errors = np.mean((values - rebuilt) ** 2, axis=0)
    table = pd.DataFrame({"feature": list(X.columns), "mse": errors})
    table = table.sort_values("mse", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    if verbose:
        show_table(table, note="Per feature, on the inner validation part. A high "
                               "value means the %d-number bottleneck could not keep "
                               "that feature -- either it is noisy, or it carries "
                               "something the others do not."
                               % config.AE_LATENT_DIM)
    return table


# ===========================================================================
# the loss curve
# ===========================================================================
_TRAIN_COLOUR = "#2a78d6"       # categorical slot 1
_VAL_COLOUR = "#eb6834"         # categorical slot 2


def plot_loss_curve(history: pd.DataFrame, best_epoch: int | None = None,
                    n_features: int | None = None, path=None, verbose: bool = True):
    """Train and validation reconstruction MSE per epoch.  Two series, so a legend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("  [SKIP]  matplotlib is not installed, so no chart was drawn.")
        return None

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        folder = prepare_output_dir(config.STEP7_OUTPUT_DIR, "the loss curve",
                                    verbose=verbose)
        if folder is None:
            return None
        path = folder / config.STEP7_FILES["loss_curve"]

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    ax.plot(history["epoch"], history["loss"], linewidth=2.0,
            color=_TRAIN_COLOUR, label="train")
    if "val_loss" in history.columns:
        ax.plot(history["epoch"], history["val_loss"], linewidth=2.0,
                color=_VAL_COLOUR, label="inner validation")

    if best_epoch is not None and "val_loss" in history.columns:
        best = float(history["val_loss"].iloc[best_epoch - 1])
        ax.scatter([best_epoch], [best], s=42, color=_VAL_COLOUR, zorder=5,
                   edgecolor=dl._SURFACE, linewidth=2)
        ax.annotate("best epoch %d\n%.5f" % (best_epoch, best),
                    (best_epoch, best), textcoords="offset points",
                    xytext=(10, 10), fontsize=9, color=dl._INK)

    ax.set_yscale("log")
    ax.set_xlabel("epoch", color=dl._INK_SOFT, fontsize=10)
    ax.set_ylabel("reconstruction MSE (log scale)", color=dl._INK_SOFT, fontsize=10)
    width = config.N_SELECTED_FEATURES if n_features is None else n_features
    ax.set_title("Autoencoder training: %d -> %s -> %d and back"
                 % (width, " -> ".join(str(u) for u in config.AE_HIDDEN),
                    config.AE_LATENT_DIM),
                 color=dl._INK, fontsize=12, pad=14)

    ax.yaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(dl._GRID)
    ax.tick_params(colors=dl._INK_SOFT, labelsize=9)
    # headroom on the right so the best-epoch annotation never clips
    ax.set_xlim(0, float(history["epoch"].max()) * 1.16)

    legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path



# ===========================================================================
# 7.4  SAVE / LOAD -- Step 7
# ===========================================================================
def build_step7_report(info: dict[str, Any], architecture: pd.DataFrame,
                       n_features: int) -> pd.DataFrame:
    """step7_report.csv -- one row per action, in the order they happened."""
    rows = [
        {"step": "7.1 input", "action": "read the %d selected features of "
                                        "X_train_selected.csv" % n_features,
         "detail": "X_test_selected was NOT read in Step 7"},
        {"step": "7.2 build", "action": "%d -> %s -> %d, decoder mirrored"
                                        % (n_features,
                                           " -> ".join(str(u) for u in config.AE_HIDDEN),
                                           config.AE_LATENT_DIM),
         "detail": "%s hidden, %s output, %s loss, %s parameters"
                   % (config.AE_ACTIVATION, config.AE_OUTPUT_ACT, config.AE_LOSS,
                      f"{int(architecture['parameters'].iloc[-1]):,}")},
        {"step": "7.3 train", "action": "unsupervised: the target is the input",
         "detail": "no label was ever a target; labels only stratified the inner "
                   "split"},
        {"step": "7.3 train", "action": "inner split for early stopping",
         "detail": "%s fit rows, %s validation rows, AE_VAL_SIZE=%s, stratified on %s"
                   % (f"{info['fit_rows']:,}", f"{info['validation_rows']:,}",
                      config.AE_VAL_SIZE, TARGET_COLUMN)},
        {"step": "7.3 train", "action": "early stopping, best weights restored",
         "detail": "ran %d of %d epochs, best epoch %d, patience %d"
                   % (info["epochs_run"], info["epochs_requested"],
                      info["best_epoch"], config.AE_PATIENCE)},
        {"step": "7.3 train", "action": "reconstruction MSE at the best epoch",
         "detail": "train %.6f, inner validation %.6f"
                   % (info["train_mse"], info["val_mse"])},
        {"step": "7.x not done", "action": "no class-imbalance handling, no classifier",
         "detail": "no SMOTE, no resampling -- later steps"},
    ]
    return pd.DataFrame(rows)


def save_step7(autoencoder, encoder, history: pd.DataFrame, info: dict[str, Any],
               architecture: pd.DataFrame, feature_mse: pd.DataFrame,
               feature_order: Sequence[str], verbose: bool = True) -> dict[str, Any]:
    """Write the models, the history, the config and the reports; then re-read."""
    out: dict[str, Any] = {"folder": str(config.STEP7_OUTPUT_DIR), "files": {}}
    folder = prepare_output_dir(config.STEP7_OUTPUT_DIR, "the Step 7 outputs",
                                verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP7_FILES
    autoencoder.save(folder / names["autoencoder"])
    encoder.save(folder / names["encoder"])
    out["files"]["autoencoder"] = str(folder / names["autoencoder"])
    out["files"]["encoder"] = str(folder / names["encoder"])

    for key, frame in (("history", history), ("architecture", architecture),
                       ("per_feature_mse", feature_mse),
                       ("report", build_step7_report(info, architecture,
                                                     len(feature_order)))):
        frame.to_csv(folder / names[key], index=False)
        out["files"][key] = str(folder / names[key])

    (folder / names["ae_config"]).write_text(json.dumps({
        "features": list(feature_order),
        "n_features": len(feature_order),
        "latent_dim": config.AE_LATENT_DIM,
        "hidden": list(config.AE_HIDDEN),
        "activation": config.AE_ACTIVATION,
        "output_activation": config.AE_OUTPUT_ACT,
        "loss": config.AE_LOSS,
        "epochs_requested": config.AE_EPOCHS,
        "epochs_run": info["epochs_run"],
        "best_epoch": info["best_epoch"],
        "batch_size": config.AE_BATCH_SIZE,
        "learning_rate": config.AE_LR,
        "val_size": config.AE_VAL_SIZE,
        "patience": config.AE_PATIENCE,
        "random_state": config.RANDOM_STATE,
        "z_columns": z_columns(),
        "train_mse": info["train_mse"],
        "val_mse": info["val_mse"],
        "training_seconds": round(info["seconds"], 3),
        "versions": framework_versions(),
        "trained_on": "the TRAINING rows of Step 6 only; X_test_selected was not "
                      "read in Step 7",
        "supervised": False,
    }, indent=2), encoding="utf-8")
    out["files"]["ae_config"] = str(folder / names["ae_config"])

    curve = plot_loss_curve(history, best_epoch=info["best_epoch"],
                            n_features=len(feature_order),
                            path=folder / names["loss_curve"], verbose=verbose)
    if curve is not None:
        out["files"]["loss_curve"] = str(curve)

    out["status"] = "written"
    verification, ok = verify_step7(info, feature_order, folder=folder,
                                    verbose=verbose)
    out["verification"] = verification
    out["ok"] = ok
    return out


def verify_step7(info: dict[str, Any], feature_order: Sequence[str],
                 folder: Path | None = None, verbose: bool = True):
    """Re-read the saved model and confirm it is the one that was trained."""
    folder = Path(folder) if folder is not None else config.STEP7_OUTPUT_DIR
    names = config.STEP7_FILES
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    present = {key: (folder / names[key]).exists() for key in names
               if key != "loss_curve"}
    add("every expected file is on disk", all(present.values()),
        "%d of %d" % (sum(present.values()), len(present)) if all(present.values())
        else "missing %s" % [names[k] for k, v in present.items() if not v])

    saved = json.loads((folder / names["ae_config"]).read_text(encoding="utf-8"))
    add("ae_config.json records the feature order",
        saved["features"] == list(feature_order),
        "%d features" % len(saved["features"]))
    add("ae_config.json records the framework and versions",
        bool(saved.get("versions", {}).get("keras")),
        ", ".join("%s %s" % kv for kv in saved.get("versions", {}).items()))

    keras = _keras()
    encoder = keras.models.load_model(folder / names["encoder"])
    out_shape = encoder.output_shape
    add("the reloaded encoder outputs %d numbers" % config.AE_LATENT_DIM,
        out_shape[-1] == config.AE_LATENT_DIM, "output shape %s" % (out_shape,))
    in_shape = encoder.input_shape
    add("the reloaded encoder takes %d features" % len(feature_order),
        in_shape[-1] == len(feature_order), "input shape %s" % (in_shape,))

    history = pd.read_csv(folder / names["history"], low_memory=False)
    add("the history covers every epoch that ran",
        len(history) == info["epochs_run"],
        "%d epochs" % len(history))
    add("the best epoch is the lowest validation loss",
        int(history["val_loss"].idxmin()) + 1 == info["best_epoch"],
        "epoch %d, val_loss %.6f" % (info["best_epoch"], info["val_mse"]))

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_title("Step 7 -- files written, then re-read from disk")
        show_table(table, note="Folder: %s" % folder)
    return table, all_ok


def step7_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP7_OUTPUT_DIR)
    if folder is None:
        return False
    return all((folder / config.STEP7_FILES[k]).exists()
               for k in ("autoencoder", "encoder", "ae_config", "history"))


def load_step7(verbose: bool = True) -> dict[str, Any]:
    """Read the saved autoencoder and its reports back instead of retraining."""
    folder = config.usable_output_dir(config.STEP7_OUTPUT_DIR)
    if folder is None:
        raise AutoencoderError("the Step 7 folder is not usable here -- %s"
                               % config.why_unusable(config.STEP7_OUTPUT_DIR))
    keras = _keras()
    names = config.STEP7_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    out["autoencoder"] = keras.models.load_model(folder / names["autoencoder"])
    out["encoder"] = keras.models.load_model(folder / names["encoder"])
    out["ae_config"] = json.loads(
        (folder / names["ae_config"]).read_text(encoding="utf-8"))
    out["feature_order"] = list(out["ae_config"]["features"])
    for key in ("history", "architecture", "per_feature_mse", "report"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)

    saved = out["ae_config"]
    out["info"] = {"train_mse": saved.get("train_mse"), "val_mse": saved.get("val_mse"),
                   "best_epoch": saved.get("best_epoch"),
                   "epochs_run": saved.get("epochs_run"),
                   "seconds": saved.get("training_seconds")}

    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP7 = False)"),
            ("folder", str(folder)),
            ("features in", saved.get("n_features")),
            ("latent dimension", saved.get("latent_dim")),
            ("best epoch", saved.get("best_epoch")),
            ("train / validation MSE", "%.6f / %.6f"
             % (saved.get("train_mse", float("nan")),
                saved.get("val_mse", float("nan")))),
            ("trained with", ", ".join("%s %s" % kv
                                       for kv in saved.get("versions", {}).items())),
        ]), note="Set `OVERWRITE_STEP7 = True` in config.py to retrain.")
    return out


def run_step7(overwrite: bool | None = None, verbose: bool = True) -> dict[str, Any]:
    """STEP 7 end to end: check -> build -> train -> report -> save."""
    overwrite = config.OVERWRITE_STEP7 if overwrite is None else overwrite
    if not overwrite and step7_outputs_exist():
        return load_step7(verbose=verbose)

    if verbose:
        show_title("Step 7.1 -- input from Step 6", level=3)
    loaded = load_step6_outputs(verbose=verbose)
    X_train, X_test = loaded["X_train_selected"], loaded["X_test_selected"]
    y_train, y_test = loaded["y_train"], loaded["y_test"]
    feature_order = loaded["feature_order"]

    if verbose:
        show_table(kv_table([
            ("X_train_selected", "%s rows x %d columns"
             % (f"{len(X_train):,}", X_train.shape[1])),
            ("X_test_selected", "%s rows x %d columns (not used until Step 8)"
             % (f"{len(X_test):,}", X_test.shape[1])),
            ("latent dimension", config.AE_LATENT_DIM),
            ("framework", "%s %s / keras %s"
             % (config.AE_FRAMEWORK, framework_versions()["tensorflow"],
                framework_versions()["keras"])),
            ("read from", loaded["folder"]),
        ]))
    check_table, _ok = check_inputs(X_train, X_test, y_train, y_test,
                                    feature_order, verbose=verbose)

    if verbose:
        show_title("Step 7.2 -- architecture", level=3)
    autoencoder, encoder, history, info = train_autoencoder(
        X_train, y_train, verbose=False)
    architecture = architecture_table(autoencoder)
    if verbose:
        show_table(architecture,
                   note="The encoder and the decoder share no weights -- the decoder "
                        "mirrors the encoder's SHAPE, not its values. Only the "
                        "encoder half is used in Step 8.")
        show_title("Step 7.3 -- training (training rows only)", level=3)
        show_table(kv_table([
            ("rows trained on", f"{info['fit_rows']:,}"),
            ("inner validation rows", f"{info['validation_rows']:,}"),
            ("epochs run", "%d of %d" % (info["epochs_run"], info["epochs_requested"])),
            ("stopped early", "yes, patience %d" % config.AE_PATIENCE
             if info["stopped_early"] else "no"),
            ("best epoch", info["best_epoch"]),
            ("train reconstruction MSE", "%.6f" % info["train_mse"]),
            ("validation reconstruction MSE", "%.6f" % info["val_mse"]),
            ("training time", "%.1f s" % info["seconds"]),
        ]), note="Reconstruction errors, not classification scores: the target is "
                 "the input itself.")

    X_val = X_train.iloc[info["val_idx"]]
    feature_mse = per_feature_mse(autoencoder, X_val, verbose=verbose)

    saved = save_step7(autoencoder, encoder, history, info, architecture,
                       feature_mse, feature_order, verbose=verbose)

    return {"status": saved.get("status", "written"), "autoencoder": autoencoder,
            "encoder": encoder, "history": history, "info": info,
            "architecture": architecture, "per_feature_mse": feature_mse,
            "feature_order": feature_order, "check_table": check_table,
            "saved": saved, "ok": bool(saved.get("ok"))}


# ===========================================================================
# STEP 8 -- transform both sets with the SAME frozen encoder
# ===========================================================================
def check_feature_frame(X: pd.DataFrame, feature_order: Sequence[str],
                        what: str = "the frame") -> None:
    """Refuse to encode anything whose columns are not exactly right, in order.

    A Dense layer takes positions, not names: hand it the same 30 columns in a
    different order and it will happily return numbers, silently wrong.  This is
    the check that makes that impossible.
    """
    feature_order = list(feature_order)
    actual = list(X.columns)
    if actual == feature_order:
        return
    missing = [c for c in feature_order if c not in actual]
    extra = [c for c in actual if c not in feature_order]
    if missing or extra:
        raise AutoencoderError(
            "%s does not carry the %d features the encoder was trained on.%s%s"
            % (what, len(feature_order),
               "\n  missing: %s" % missing if missing else "",
               "\n  unexpected: %s" % extra if extra else ""))
    first = next(i for i, (a, b) in enumerate(zip(actual, feature_order)) if a != b)
    raise AutoencoderError(
        "%s has the right %d features but in the WRONG ORDER: position %d is %r, "
        "the encoder expects %r. A Dense layer reads positions, not names, so "
        "encoding this would return numbers that look fine and are wrong."
        % (what, len(feature_order), first, actual[first], feature_order[first]))


def make_encoder_function(encoder, feature_order: Sequence[str]):
    """Return encode(X) -> Z, which checks the columns before every call."""
    feature_order = list(feature_order)
    columns = z_columns(int(encoder.output_shape[-1]))

    def encode(X: pd.DataFrame, what: str = "the frame") -> pd.DataFrame:
        check_feature_frame(X, feature_order, what=what)
        values = X.to_numpy(dtype="float32")
        latent = encoder.predict(values, verbose=0)
        return pd.DataFrame(latent, columns=columns, index=X.index)

    encode.feature_order = feature_order        # type: ignore[attr-defined]
    encode.z_columns = columns                  # type: ignore[attr-defined]
    return encode


def load_encoder(folder: Path | None = None, verbose: bool = True):
    """Load the saved encoder and return (encode_function, encoder, feature_order)."""
    folder = config.usable_output_dir(
        config.STEP7_OUTPUT_DIR if folder is None else folder)
    if folder is None:
        raise AutoencoderError("the Step 7 folder is not usable here -- %s"
                               % config.why_unusable(config.STEP7_OUTPUT_DIR))
    names = config.STEP7_FILES
    if not (folder / names["encoder"]).exists():
        raise AutoencoderError(
            "no saved encoder at %s. Run Step 7 first (set OVERWRITE_STEP7 = True "
            "if its outputs are stale)." % (folder / names["encoder"]))

    keras = _keras()
    encoder = keras.models.load_model(folder / names["encoder"])
    saved = json.loads((folder / names["ae_config"]).read_text(encoding="utf-8"))
    feature_order = list(saved["features"])
    if verbose:
        show_note("Encoder loaded from **%s**: %d features in, %d numbers out."
                  % (folder / names["encoder"], len(feature_order),
                     int(encoder.output_shape[-1])))
    return make_encoder_function(encoder, feature_order), encoder, feature_order


def reconstruction_mse(autoencoder, X: pd.DataFrame) -> float:
    """Mean squared reconstruction error over every cell of X."""
    values = X.to_numpy(dtype="float32")
    rebuilt = autoencoder.predict(values, verbose=0)
    return float(np.mean((values - rebuilt) ** 2))


def verify_step8(n_train: int, n_test: int, folder: Path | None = None,
                 verbose: bool = True):
    """Re-read Z_train / Z_test from disk and check them.  [OK]/[FAIL]."""
    folder = Path(folder) if folder is not None else config.STEP8_OUTPUT_DIR
    names = config.STEP8_FILES
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    present = {key: (folder / names[key]).exists() for key in names}
    add("every expected file is on disk", all(present.values()),
        "%d of %d" % (sum(present.values()), len(present)) if all(present.values())
        else "missing %s" % [names[k] for k, v in present.items() if not v])

    Z_train = pd.read_csv(folder / names["Z_train"], low_memory=False)
    Z_test = pd.read_csv(folder / names["Z_test"], low_memory=False)
    y_train = pd.read_csv(folder / names["y_train"], low_memory=False)
    y_test = pd.read_csv(folder / names["y_test"], low_memory=False)
    expected = z_columns()

    add("Z_train is (%s, %d)" % (f"{n_train:,}", config.AE_LATENT_DIM),
        Z_train.shape == (n_train, config.AE_LATENT_DIM), "%s" % (Z_train.shape,))
    add("Z_test is (%s, %d)" % (f"{n_test:,}", config.AE_LATENT_DIM),
        Z_test.shape == (n_test, config.AE_LATENT_DIM), "%s" % (Z_test.shape,))
    add("both carry z01 ... %s in the same order" % expected[-1],
        list(Z_train.columns) == expected and list(Z_test.columns) == expected,
        ", ".join(expected[:3]) + ", ..., " + expected[-1])

    for name, frame in (("Z_train", Z_train), ("Z_test", Z_test)):
        block = frame.to_numpy(dtype="float64")
        n_nan, n_inf = int(np.isnan(block).sum()), int(np.isinf(block).sum())
        add("no NaN or inf in %s" % name, n_nan == 0 and n_inf == 0,
            "%d NaN, %d inf" % (n_nan, n_inf))

    add("labels are still aligned with the encoded rows",
        len(y_train) == len(Z_train) and len(y_test) == len(Z_test),
        "y_train %s, y_test %s" % (f"{len(y_train):,}", f"{len(y_test):,}"))
    add("the label columns were copied unchanged",
        list(y_train.columns) == LABEL_COLUMNS and list(y_test.columns) == LABEL_COLUMNS,
        ", ".join(y_train.columns))

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_title("Step 8 -- files written, then re-read from disk")
        show_table(table, note="Folder: %s" % folder)
        if not all_ok:
            show_note("**[FAIL]** -- the Step 8 outputs do not agree; do not train "
                      "on them.")
    return table, all_ok


def step8_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder is None:
        return False
    return all((folder / config.STEP8_FILES[k]).exists()
               for k in ("Z_train", "Z_test", "y_train", "y_test"))


def load_step8(verbose: bool = True) -> dict[str, Any]:
    """Read the saved Z matrices back instead of re-encoding."""
    folder = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder is None:
        raise AutoencoderError("the Step 8 folder is not usable here -- %s"
                               % config.why_unusable(config.STEP8_OUTPUT_DIR))
    names = config.STEP8_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    for key in ("Z_train", "Z_test", "y_train", "y_test", "recon_mse", "report"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP8 = False)"),
            ("folder", str(folder)),
            ("Z_train", "%s rows x %d columns"
             % (f"{len(out['Z_train']):,}", out["Z_train"].shape[1])),
            ("Z_test", "%s rows x %d columns"
             % (f"{len(out['Z_test']):,}", out["Z_test"].shape[1])),
        ]), note="Set `OVERWRITE_STEP8 = True` in config.py to encode again.")
    return out


def run_step8(overwrite: bool | None = None, encoder=None, autoencoder=None,
              feature_order: Sequence[str] | None = None,
              verbose: bool = True) -> dict[str, Any]:
    """STEP 8: run the SAME frozen encoder over the training and test sets."""
    overwrite = config.OVERWRITE_STEP8 if overwrite is None else overwrite
    if not overwrite and step8_outputs_exist():
        return load_step8(verbose=verbose)

    loaded = load_step6_outputs(verbose=False)
    X_train, X_test = loaded["X_train_selected"], loaded["X_test_selected"]
    y_train, y_test = loaded["y_train"], loaded["y_test"]

    if encoder is None or feature_order is None:
        encode, encoder, feature_order = load_encoder(verbose=verbose)
    else:
        encode = make_encoder_function(encoder, feature_order)

    # the column check runs on BOTH frames before either is encoded
    Z_train = encode(X_train, what="X_train_selected.csv")
    Z_test = encode(X_test, what="X_test_selected.csv")

    test_mse = train_mse = None
    if autoencoder is None:
        folder7 = config.usable_output_dir(config.STEP7_OUTPUT_DIR)
        path = folder7 / config.STEP7_FILES["autoencoder"] if folder7 else None
        if path is not None and path.exists():
            autoencoder = _keras().models.load_model(path)
    if autoencoder is not None:
        train_mse = reconstruction_mse(autoencoder, X_train)
        test_mse = reconstruction_mse(autoencoder, X_test)

    mse_table = pd.DataFrame([
        {"set": "train", "rows": len(X_train), "reconstruction MSE": train_mse},
        {"set": "test", "rows": len(X_test), "reconstruction MSE": test_mse},
    ])

    report = pd.DataFrame([
        {"step": "8.1 encode", "action": "the SAME frozen encoder over both sets",
         "detail": "%d features -> %d numbers; row order unchanged"
                   % (len(feature_order), config.AE_LATENT_DIM)},
        {"step": "8.1 encode", "action": "column names and order checked first",
         "detail": "a Dense layer reads positions, so a reordered frame would "
                   "encode silently wrong"},
        {"step": "8.2 report", "action": "test reconstruction MSE",
         "detail": "%s -- REPORTED ONLY, no decision depends on it"
                   % ("%.6f" % test_mse if test_mse is not None else "unavailable")},
        {"step": "8.x not done", "action": "no class-imbalance handling, no classifier",
         "detail": "later steps, fitted on the training side only"},
    ])

    out: dict[str, Any] = {"Z_train": Z_train, "Z_test": Z_test,
                           "y_train": y_train, "y_test": y_test,
                           "recon_mse": mse_table, "report": report,
                           "train_mse": train_mse, "test_mse": test_mse,
                           "feature_order": list(feature_order)}

    folder = prepare_output_dir(config.STEP8_OUTPUT_DIR, "the Step 8 outputs",
                                verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP8_FILES
    for key, frame in (("Z_train", Z_train), ("Z_test", Z_test),
                       ("y_train", y_train), ("y_test", y_test),
                       ("recon_mse", mse_table), ("report", report)):
        frame.to_csv(folder / names[key], index=False)

    if verbose:
        show_title("Step 8 -- encoded", level=3)
        show_table(kv_table([
            ("Z_train", "%s rows x %d columns"
             % (f"{Z_train.shape[0]:,}", Z_train.shape[1])),
            ("Z_test", "%s rows x %d columns"
             % (f"{Z_test.shape[0]:,}", Z_test.shape[1])),
            ("columns", "%s ... %s" % (Z_train.columns[0], Z_train.columns[-1])),
            ("train reconstruction MSE",
             "%.6f" % train_mse if train_mse is not None else "unavailable"),
            ("test reconstruction MSE",
             "%.6f" % test_mse if test_mse is not None else "unavailable"),
            ("output folder", str(folder)),
        ]), note="The test MSE is **reported only**. Nothing in Steps 7-8 was "
                 "chosen by looking at it -- the architecture, the epochs and the "
                 "early stopping all came from the training rows.")

    verification, ok = verify_step8(len(X_train), len(X_test), folder=folder,
                                    verbose=verbose)
    out.update({"status": "written", "verification": verification, "ok": ok,
                "folder": str(folder)})
    return out
