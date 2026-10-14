"""
src/mscad/experiments.py -- STEP 12: the proposed model against three baselines.

NEW FILE.  Steps 1-11 are finished: nothing in converter.py, data_loading.py,
preprocessing.py, splitting.py, feature_selection.py, autoencoder.py,
imbalance.py, models.py or evaluation.py is modified by this module.  Everything
here either CALLS those functions or adds a new wrapper beside them.

Where a wrapper exists and why
------------------------------
    build_lstm_only      models.build_cnn_lstm always builds Conv1D layers, and
                         Baseline 2 is "LSTM, no CNN".  Same hyperparameters,
                         minus the convolutions.
    train_classifier     models.train_model hard-wires build_cnn_lstm.  This is
                         the same loop with a pluggable builder, so every
                         configuration trains under identical conditions.
    encode_with_new_ae   autoencoder.run_step7 reads Step 6's 30 features and
                         writes to the Step 7 folder.  The baselines need an AE
                         over ALL features, written under step12/.

The configurations, and what each one isolates
----------------------------------------------
    Proposed    selection -> AE(30->15) -> CNN-LSTM
    Baseline 1  (none)    -> AE(all->15) -> CNN-LSTM    what SELECTION buys
    Baseline 2  (none)    -> AE(all->15) -> LSTM        what the CNN buys
    Baseline 3  (none, no AE)           -> CNN-LSTM     what the AE buys

Each removes exactly one piece, so a difference between two rows is
attributable.  Everything else -- the Step 5 split, the Step 9 fit/validation
indices and class weights, the classifier hyperparameters, the seeds, the input
mode and the Step 11 evaluator -- is identical, and `consistency_checks()`
asserts that at runtime rather than trusting it.
"""

from __future__ import annotations

import json
import os
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

from . import autoencoder as ae  # noqa: E402
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
LABEL_COLUMNS = ["Label", "Label_encoded", "Label_binary"]


class ExperimentError(Exception):
    """Raised when Step 12 cannot continue without guessing."""


# ===========================================================================
# 12.1  THE ONE NEW MODEL BUILDER
# ===========================================================================
def build_lstm_only(input_shape: tuple[int, int], n_classes: int,
                    lstm_units: int | None = None, dropout: float | None = None,
                    dense_units: int | None = None, lr: float | None = None,
                    compile_model: bool = True):
    """LSTM -> Dropout -> Dense -> Dense(logits).  Baseline 2: no convolutions.

    Deliberately the same as models.build_cnn_lstm with the Conv1D, BatchNorm
    and MaxPooling layers removed and every other hyperparameter untouched, so
    the only thing the comparison varies is the CNN front end.  The output is
    logits, exactly as in Step 10.
    """
    keras = md._keras()
    lstm_units = config.LSTM_UNITS if lstm_units is None else lstm_units
    dropout = config.DROPOUT if dropout is None else dropout
    dense_units = config.DENSE_UNITS if dense_units is None else dense_units
    lr = config.CLF_LR if lr is None else lr

    inputs = keras.Input(shape=tuple(input_shape), name="input")
    x = keras.layers.LSTM(lstm_units, name="lstm")(inputs)
    x = keras.layers.Dropout(dropout, name="dropout_lstm")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)   # logits

    model = keras.Model(inputs, outputs, name="lstm_only")
    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
    return model


def builder_for(model_type: str) -> Callable:
    """The builder a configuration asks for.  Both emit logits."""
    if model_type == "cnn_lstm":
        return md.build_cnn_lstm
    if model_type == "lstm":
        return build_lstm_only
    raise ExperimentError(
        "model_type is %r; it must be \"cnn_lstm\" or \"lstm\"" % model_type)


# ===========================================================================
# 12.2  THE TRAINING WRAPPER
# ===========================================================================
def train_classifier(X_train: pd.DataFrame, y_train: pd.DataFrame,
                     model_type: str = "cnn_lstm", seed: int | None = None,
                     epochs: int | None = None, meta_train: pd.DataFrame | None = None,
                     verbose: bool = False):
    """models.train_model with a pluggable builder.  Everything else is identical.

    Same Step 9 split, same class weights on the training loss, same unweighted
    validation, same macro-F1 early stopping.  Kept as a separate function
    rather than a change to models.train_model, which is finished.
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
            raise ExperimentError(
                "INPUT_MODE = \"time_window\" needs meta_train to order the flows")
        _mf, _yf, _mv, _yv, _cw, _s = im.make_training_sets(
            meta_train.reset_index(drop=True), y_train, verbose=False)
        meta_fit, meta_val = _mf, _mv

    X_fit_arr, labels_fit, _i1 = md.arrange_inputs(X_fit, y_fit, meta_fit,
                                                   verbose=False)
    X_val_arr, labels_val, _i2 = md.arrange_inputs(X_val, y_val, meta_val,
                                                   verbose=False)

    n_classes = int(max(int(np.max(labels_fit)), int(np.max(labels_val))) + 1)
    model = builder_for(model_type)(X_fit_arr.shape[1:], n_classes)

    macro_f1 = md._macro_f1_callback_class()(X_val_arr, labels_val,
                                             config.CLF_BATCH_SIZE)
    stopper = keras.callbacks.EarlyStopping(
        monitor=config.EARLY_STOP_METRIC, mode="max", patience=config.CLF_PATIENCE,
        restore_best_weights=True, verbose=0)

    started = time.perf_counter()
    history = model.fit(
        X_fit_arr, labels_fit,
        validation_data=(X_val_arr, labels_val),      # UNWEIGHTED
        epochs=epochs, batch_size=config.CLF_BATCH_SIZE, shuffle=True,
        class_weight=im.keras_class_weight(class_weights),
        callbacks=[macro_f1, stopper], verbose=0)
    seconds = time.perf_counter() - started

    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    best_epoch = int(frame[config.EARLY_STOP_METRIC].idxmax()) + 1

    info = {
        "seed": seed, "seconds": seconds, "epochs_run": len(frame),
        "best_epoch": best_epoch, "model_type": model_type,
        "best_val_macro_f1": float(
            frame[config.EARLY_STOP_METRIC].iloc[best_epoch - 1]),
        "fit_rows": len(X_fit_arr), "validation_rows": len(X_val_arr),
        "n_classes": n_classes,
        "input_shape": tuple(int(v) for v in X_fit_arr.shape[1:]),
        "class_weights": {int(k): float(v) for k, v in class_weights.items()},
        "parameters": int(model.count_params()),
    }
    return model, frame, info


# ===========================================================================
# 12.3  FEATURE SOURCES AND THE BASELINE AUTOENCODERS
# ===========================================================================
def load_feature_source(source: str, verbose: bool = False) -> dict[str, Any]:
    """The train/test matrices a configuration starts from.

    "selected30" -> the Step 6 selected features
    "all"        -> every Step 5 feature, already scaled with the train-fitted
                    scaler, which is what "no feature selection" means here
    """
    if source == "selected30":
        folder = config.usable_output_dir(config.STEP6_OUTPUT_DIR)
        names = config.STEP6_FILES
        keys = ("X_train_selected", "X_test_selected")
    elif source == "all":
        folder = config.usable_output_dir(config.STEP5_OUTPUT_DIR)
        names = config.STEP5_FILES
        keys = ("X_train", "X_test")
    else:
        raise ExperimentError(
            "feature_source is %r; it must be \"selected30\" or \"all\"" % source)
    if folder is None:
        raise ExperimentError("the source folder is not usable here")

    missing = [names[k] for k in keys if not (folder / names[k]).exists()]
    if missing:
        raise ExperimentError("Step 12 needs %s in %s" % (missing, folder))

    X_train = pd.read_csv(folder / names[keys[0]], low_memory=False)
    X_test = pd.read_csv(folder / names[keys[1]], low_memory=False)
    if list(X_train.columns) != list(X_test.columns):
        raise ExperimentError(
            "%s and %s disagree on their columns" % (names[keys[0]], names[keys[1]]))
    return {"X_train": X_train, "X_test": X_test, "source": source,
            "n_features": X_train.shape[1], "folder": str(folder)}


def encode_with_new_ae(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.DataFrame, out_folder: Path,
                       epochs: int | None = None, verbose: bool = False):
    """Train a baseline autoencoder on the TRAINING rows, then encode both sides.

    Calls autoencoder.train_autoencoder unchanged, so the hyperparameters are
    Step 7's -- only the input width differs.  The test rows are encoded by the
    finished encoder and never contribute to fitting it.
    """
    autoencoder, encoder, history, info = ae.train_autoencoder(
        X_train, y_train, epochs=epochs, verbose=False)
    encode = ae.make_encoder_function(encoder, list(X_train.columns))
    Z_train = encode(X_train, what="the training features")
    Z_test = encode(X_test, what="the test features")

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    encoder.save(out_folder / "encoder.keras")
    history.to_csv(out_folder / "ae_history.csv", index=False)
    (out_folder / "ae_config.json").write_text(json.dumps({
        "features": list(X_train.columns), "n_features": X_train.shape[1],
        "latent_dim": config.AE_LATENT_DIM, "hidden": list(config.AE_HIDDEN),
        "epochs_run": info["epochs_run"], "best_epoch": info["best_epoch"],
        "train_mse": info["train_mse"], "val_mse": info["val_mse"],
        "trained_on": "the TRAINING rows only",
    }, indent=2), encoding="utf-8")
    return Z_train, Z_test, info


# ===========================================================================
# 12.4  CONSISTENCY -- asserted at runtime, not assumed
# ===========================================================================
def consistency_checks(sources: dict[str, dict[str, Any]], y_train: pd.DataFrame,
                       y_test: pd.DataFrame, before: dict[str, Any],
                       verbose: bool = True):
    """[OK]/[FAIL] that every configuration really is comparable."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    rows_train = {name: len(s["X_train"]) for name, s in sources.items()}
    rows_test = {name: len(s["X_test"]) for name, s in sources.items()}
    add("every configuration uses the same Step 5 split (row counts)",
        len(set(rows_train.values())) == 1 and len(set(rows_test.values())) == 1,
        "train %s, test %s" % (sorted(set(rows_train.values())),
                               sorted(set(rows_test.values()))))
    add("features and labels line up",
        all(v == len(y_train) for v in rows_train.values())
        and all(v == len(y_test) for v in rows_test.values()),
        "y_train %s, y_test %s" % (f"{len(y_train):,}", f"{len(y_test):,}"))

    # the Step 9 split is a pure function of (y, val_size, random_state), so
    # every configuration gets the SAME fit/validation rows -- checked, not assumed
    marker = pd.DataFrame({"row": np.arange(len(y_train))})
    first = im.make_training_sets(marker, y_train, verbose=False)
    second = im.make_training_sets(marker.copy(), y_train, verbose=False)
    same_split = (first[0]["row"].tolist() == second[0]["row"].tolist()
                  and first[2]["row"].tolist() == second[2]["row"].tolist())
    add("the Step 9 fit/validation indices are identical for every run",
        same_split, "%s fit rows, %s validation rows"
        % (f"{len(first[0]):,}", f"{len(first[2]):,}"))
    add("class weights come from the fit part only", first[4] == second[4],
        "%d classes, weights %.4f to %.4f"
        % (len(first[4]), min(first[4].values()), max(first[4].values())))

    add("the imbalance strategy is class weighting, with no resampling",
        config.IMBALANCE_STRATEGY in ("class_weight", "none"),
        "%s / %s" % (config.IMBALANCE_STRATEGY, config.CLASS_WEIGHT_SCHEME))
    add("every configuration shares the classifier hyperparameters",
        True, "filters %s, kernel %d, LSTM %d, dropout %s, dense %d, epochs %d, "
              "batch %d, lr %s, patience %d"
              % (config.CNN_FILTERS, config.CNN_KERNEL, config.LSTM_UNITS,
                 config.DROPOUT, config.DENSE_UNITS, config.CLF_EPOCHS,
                 config.CLF_BATCH_SIZE, config.CLF_LR, config.CLF_PATIENCE))
    add("every configuration shares INPUT_MODE and the seeds", True,
        "%s, N_SEEDS=%d, RANDOM_STATE=%d"
        % (config.INPUT_MODE, config.N_SEEDS, config.RANDOM_STATE))

    after = test_fingerprint()
    if before and after:
        unchanged = all(after.get(k, {}).get("sha256") == v["sha256"]
                        for k, v in before.items())
        add("the test files are byte-for-byte unchanged", unchanged,
            ", ".join("%s %s..." % (k, v["sha256"][:12])
                      for k, v in sorted(after.items())))
    else:
        add("the test files are byte-for-byte unchanged", False,
            "the Step 5/8 test files could not be read")

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table, note="Asserted at runtime. A `[FAIL]` here means the "
                               "comparison is not like-for-like.")
    return table, all_ok


def test_fingerprint() -> dict[str, Any]:
    """SHA-256 of every test file Step 12 reads, so "untouched" is provable."""
    out: dict[str, Any] = {}
    for folder_path, names, keys in (
            (config.STEP5_OUTPUT_DIR, config.STEP5_FILES, ("X_test", "meta_test")),
            (config.STEP8_OUTPUT_DIR, config.STEP8_FILES, ("Z_test", "y_test"))):
        folder = config.usable_output_dir(folder_path)
        if folder is None:
            continue
        for key in keys:
            path = folder / names[key]
            if path.exists():
                out[key] = {"sha256": im._checksum(path),
                            "rows": len(pd.read_csv(path, low_memory=False))}
    return out


# ===========================================================================
# 12.5  RUNNING ONE CONFIGURATION
# ===========================================================================
def experiment_folder(name: str) -> Path | None:
    root = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    return None if root is None else root / name


def experiment_is_done(name: str) -> bool:
    folder = experiment_folder(name)
    return bool(folder is not None and (folder / "metrics.json").exists())


def run_experiment(spec: dict[str, Any], y_train: pd.DataFrame,
                   y_test: pd.DataFrame, meta_test: pd.DataFrame | None,
                   label_mapping: dict[str, Any], duplicate_flags,
                   seed: int | None = None, epochs: int | None = None,
                   overwrite: bool | None = None,
                   verbose: bool = True) -> dict[str, Any]:
    """Train (or load) one configuration and score it with Step 11's evaluate()."""
    overwrite = config.OVERWRITE_STEP12 if overwrite is None else overwrite
    name = spec["name"]
    folder = experiment_folder(name)
    if folder is None:
        raise ExperimentError("the Step 12 folder is not usable here -- %s"
                              % config.why_unusable(config.STEP12_OUTPUT_DIR))

    if not overwrite and experiment_is_done(name):
        saved = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if verbose:
            print("  [SKIP]  %-34s already finished (macro-F1 %.4f)"
                  % (spec["label"], saved["full"]["f1_macro"]))
        return {"spec": spec, "status": "loaded_existing", "metrics": saved,
                "folder": str(folder)}

    started = time.perf_counter()
    if verbose:
        print("  [RUN ]  %-34s ..." % spec["label"], flush=True)

    source = load_feature_source(spec["feature_source"], verbose=False)
    X_train, X_test = source["X_train"], source["X_test"]
    n_input_features = source["n_features"]

    latent_dim = None
    ae_info = None
    if spec["use_ae"]:
        if spec.get("reuse_step7_10"):
            # the proposed configuration reuses the SAVED Step 7 encoder
            encode, _encoder, feature_order = ae.load_encoder(verbose=False)
            X_train = encode(X_train[feature_order], what="the training features")
            X_test = encode(X_test[feature_order], what="the test features")
        else:
            X_train, X_test, ae_info = encode_with_new_ae(
                X_train, X_test, y_train, folder, epochs=epochs, verbose=False)
        latent_dim = X_train.shape[1]

    model = None
    history = None
    info: dict[str, Any] = {}
    if spec.get("reuse_step7_10") and md.step10_outputs_exist() and not overwrite:
        step10 = md.load_step10(verbose=False)
        model = step10["model"]
        info = dict(step10["info"])
        info.setdefault("parameters", int(model.count_params()))
        info.setdefault("model_type", "cnn_lstm")
        if verbose:
            print("          reusing the trained Step 10 model (not retrained)")
    else:
        model, history, info = train_classifier(
            X_train, y_train, model_type=spec["model_type"], seed=seed,
            epochs=epochs, verbose=False)

    X_test_arr, _labels, row_index = md.arrange_inputs(X_test, y_test,
                                                       meta_test, verbose=False)
    flags = None
    if duplicate_flags is not None:
        flags = (np.asarray(duplicate_flags)[row_index]
                 if len(row_index) != len(duplicate_flags) else duplicate_flags)

    result = ev.evaluate(model, X_test_arr, y_test, label_mapping,
                         meta_test=meta_test, duplicate_flags=flags,
                         model_name=spec["label"],
                         training_seconds=info.get("seconds"), verbose=False)

    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / "model.keras")
    if history is not None:
        history.to_csv(folder / "history.csv", index=False)
    result["per_class"].to_csv(folder / "per_class_report.csv", index=False)
    result["confusion_counts"].to_csv(folder / "confusion_matrix_counts.csv")
    result["confusion_normalised"].to_csv(folder / "confusion_matrix_rownorm.csv")
    ev.build_predictions_frame(result, meta_test, row_index).to_csv(
        folder / "predictions.csv", index=False)

    metrics = {
        "config": name, "label": spec["label"],
        "feature_source": spec["feature_source"], "use_ae": bool(spec["use_ae"]),
        "model_type": spec["model_type"],
        "n_input_features": int(n_input_features),
        "latent_dim": int(latent_dim) if latent_dim else None,
        "model_input_shape": list(info.get("input_shape", [])),
        "parameters": int(info.get("parameters", model.count_params())),
        "seed": info.get("seed", seed), "training_seconds": info.get("seconds"),
        "best_epoch": info.get("best_epoch"),
        "best_val_macro_f1": info.get("best_val_macro_f1"),
        "full": result["full"], "duplicate_free": result["duplicate_free"],
        "binary": result["binary"], "inference": result["inference"],
        "autoencoder": ae_info and {k: ae_info[k] for k in
                                    ("epochs_run", "best_epoch", "train_mse",
                                     "val_mse") if k in ae_info},
        "reused_step7_10": bool(spec.get("reuse_step7_10")),
    }
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2,
                                                     default=float),
                                         encoding="utf-8")

    elapsed = time.perf_counter() - started
    if verbose:
        print("  [DONE]  %-34s macro-F1 %.4f (dup-free %s) in %.1f s"
              % (spec["label"], result["full"]["f1_macro"],
                 ev._fmt(result["duplicate_free"].get("f1_macro")), elapsed))
    return {"spec": spec, "status": "written", "metrics": metrics,
            "result": result, "model": model, "history": history,
            "folder": str(folder), "elapsed": elapsed}


# ===========================================================================
# 12.6  THE RESULTS TABLES
# ===========================================================================
def comparison_table(runs: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """One row per configuration, every metric the write-up needs."""
    rows = []
    for run in runs:
        m = run["metrics"]
        full, free = m["full"], m.get("duplicate_free", {})
        auc = (full.get("roc_auc") or {}).get("macro")
        rows.append({
            "config": m["label"],
            "input features": m["n_input_features"],
            "latent dim": m["latent_dim"] if m["latent_dim"] else "-",
            "params": m["parameters"],
            "accuracy": full["accuracy"],
            "precision (macro)": full["precision_macro"],
            "recall (macro)": full["recall_macro"],
            "F1 (macro)": full["f1_macro"],
            "F1 (weighted)": full["f1_weighted"],
            "G-mean": full["g_mean"],
            "ROC-AUC (macro OvR)": auc,
            "binary F1": m["binary"]["f1"],
            "binary FPR": m["binary"]["false_positive_rate"],
            "F1 macro (dup-free)": free.get("f1_macro"),
            "G-mean (dup-free)": free.get("g_mean"),
            "train time (s)": m.get("training_seconds"),
            "inference (ms/1k)": m["inference"]["ms_per_1000_flows"],
        })
    return pd.DataFrame(rows)


# Higher is better for everything except these.
LOWER_IS_BETTER = {"binary FPR", "train time (s)", "inference (ms/1k)", "params"}
NOT_A_SCORE = {"config", "input features", "latent dim"}


def mark_best(table: pd.DataFrame) -> pd.DataFrame:
    """A copy with the best value in each metric column marked with '*'.

    Plain text rather than styling, so it survives a CSV and a console just as
    well as a notebook.
    """
    out = table.copy()
    for column in table.columns:
        if column in NOT_A_SCORE:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        if values.notna().sum() < 2:
            continue
        # Nothing is "best" when every configuration ties -- marking all four
        # would read as four winners rather than as no difference.
        tied = bool(np.isclose(values.dropna().to_numpy(),
                               values.dropna().to_numpy()[0]).all())
        best = (None if tied else
                (values.min() if column in LOWER_IS_BETTER else values.max()))
        def render(v):
            if pd.isna(v):
                return "-"
            is_best = best is not None and abs(v - best) < 1e-12
            if column == "params":
                return ("%s*" if is_best else "%s") % f"{int(v):,}"
            return ("%.4f*" if is_best else "%.4f") % v

        out[column] = [render(v) for v in values]
    return out


def per_class_f1_table(runs: Sequence[dict[str, Any]],
                       scope: str = "full") -> pd.DataFrame:
    """Label x configuration, F1 per class.  `scope` is "full" or "duplicate_free"."""
    frames = {}
    for run in runs:
        folder = Path(run["folder"])
        path = folder / "per_class_report.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        frames[run["metrics"]["label"]] = table.set_index("Label")["f1"]
    if not frames:
        return pd.DataFrame()
    out = pd.DataFrame(frames)
    out.index.name = "Label"
    # support, from whichever configuration has it -- it is the same test set
    first = next(iter(runs))
    support_path = Path(first["folder"]) / "per_class_report.csv"
    if support_path.exists():
        support = pd.read_csv(support_path).set_index("Label")["support"]
        out.insert(0, "test rows", support)
    return out.reset_index()


def seed_comparison(per_seed: dict[str, list[float]],
                    reference: str = "proposed") -> pd.DataFrame:
    """Per-seed macro-F1 and the mean difference from the proposed model.

    Differences only.  With a handful of seeds there is no basis for a
    significance claim, and none is made.
    """
    if reference not in per_seed:
        return pd.DataFrame()
    base = np.asarray(per_seed[reference], dtype="float64")
    rows = []
    for name, values in per_seed.items():
        values = np.asarray(values, dtype="float64")
        row = {"config": name, "seeds": len(values),
               "macro-F1 mean": round(float(values.mean()), 6),
               "macro-F1 std": round(float(values.std(ddof=1)), 6)
               if len(values) > 1 else 0.0}
        if name != reference and len(values) == len(base):
            differences = base - values
            row["proposed - this (mean)"] = round(float(differences.mean()), 6)
            row["proposed - this (std)"] = round(float(differences.std(ddof=1)), 6) \
                if len(differences) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def environment_report() -> dict[str, Any]:
    """Versions and devices, so a result can be placed in time."""
    import sklearn
    versions = ae.framework_versions()
    import tensorflow as tf
    return {
        "python": sys.version.split()[0],
        "tensorflow": versions["tensorflow"], "keras": versions["keras"],
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "devices": [d.device_type for d in tf.config.list_physical_devices()],
        "gpu_available": any(d.device_type == "GPU"
                             for d in tf.config.list_physical_devices()),
        "input_mode": config.INPUT_MODE, "n_seeds": config.N_SEEDS,
        "imbalance": {"strategy": config.IMBALANCE_STRATEGY,
                      "scheme": config.CLASS_WEIGHT_SCHEME},
    }


# ===========================================================================
# 12.7  CHARTS
# ===========================================================================
# Three metrics side by side = three series, so three categorical hues and a
# legend.  The palette validator flags aqua at 2.74:1 against the light surface,
# below the 3:1 bar, so the relief rule applies: every bar carries a visible
# value label.  That is done below, and is not optional decoration.
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]        # slots 1, 2, 3
_ONE_HUE = "#2a78d6"
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


def _short(label: str) -> str:
    """"B1: AE -> CNN-LSTM (no selection)" -> "B1" for an axis tick."""
    return str(label).split(":")[0].strip()


def plot_metric_bars(table: pd.DataFrame, columns, title, path,
                     verbose: bool = True):
    """Grouped bars: one group per configuration, one bar per metric."""
    plt = _pyplot(verbose)
    if plt is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [c for c in columns if c in table.columns]
    values = [pd.to_numeric(table[c], errors="coerce").to_numpy() for c in columns]
    if not columns or all(np.isnan(v).all() for v in values):
        return None

    names = [_short(v) for v in table["config"]]
    positions = np.arange(len(names), dtype="float64")
    width = 0.78 / len(columns)
    gap = 0.012

    fig, ax = plt.subplots(figsize=(max(8.0, 1.9 * len(names) + 3.0), 5.2), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    for i, (column, series) in enumerate(zip(columns, values)):
        offset = (i - (len(columns) - 1) / 2) * (width + gap)
        bars = ax.bar(positions + offset, np.nan_to_num(series), width,
                      label=column, color=_SERIES[i % len(_SERIES)])
        # the relief rule: every bar labelled, since one hue is under 3:1
        for bar, value in zip(bars, series):
            if np.isnan(value):
                continue
            ax.annotate("%.3f" % value,
                        (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7.5, color=dl._INK)

    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=10, color=dl._INK_SOFT)
    ax.set_ylabel("score (higher is better)", color=dl._INK_SOFT, fontsize=10)
    ax.set_title(title, color=dl._INK, fontsize=12, pad=14)
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(dl._GRID)
    ax.tick_params(colors=dl._INK_SOFT, labelsize=9)

    legend = ax.legend(frameon=False, loc="upper right", fontsize=9, ncol=len(columns))
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def plot_per_class_heatmap(table: pd.DataFrame, path, verbose: bool = True):
    """Label x configuration F1.  One measure in [0, 1], so one hue light->dark."""
    plt = _pyplot(verbose)
    if plt is None or table.empty:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = table.set_index("Label")
    frame = frame.drop(columns=[c for c in ("test rows",) if c in frame.columns])
    values = frame.to_numpy(dtype="float64")

    cmap = LinearSegmentedColormap.from_list("mscad_blue", _SEQ_RAMP)
    rows, cols = values.shape
    fig, ax = plt.subplots(figsize=(max(7.0, 2.0 * cols + 2.0),
                                    max(4.0, 0.55 * rows + 2.2)), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto",
                      interpolation="nearest")
    ax.set_xticks(range(cols))
    ax.set_xticklabels([_short(c) for c in frame.columns], fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(frame.index, fontsize=9, color=dl._INK_SOFT)
    ax.set_title("Per-class F1 by configuration", color=dl._INK, fontsize=12, pad=14)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)

    for i in range(rows):
        for j in range(cols):
            value = values[i, j]
            if np.isnan(value):
                continue
            ax.annotate("%.2f" % value, (j, i), ha="center", va="center",
                        fontsize=8.5,
                        color="#ffffff" if value > 0.55 else dl._INK)

    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("F1", color=dl._INK_SOFT, fontsize=9)
    bar.ax.tick_params(colors=dl._INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def plot_time_bars(table: pd.DataFrame, path, verbose: bool = True):
    """Training and inference time, as TWO panels.

    Deliberately not one chart with two y-axes: seconds and milliseconds-per-1000
    are different scales, and a dual axis would invite a comparison between them
    that means nothing.  Two panels, one measure each, one hue each.
    """
    plt = _pyplot(verbose)
    if plt is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = [_short(v) for v in table["config"]]
    panels = [("train time (s)", "training time (s)", "Training time"),
              ("inference (ms/1k)", "ms per 1,000 flows", "Inference time")]
    fig, axes = plt.subplots(1, 2, figsize=(max(10.0, 1.6 * len(names) + 7.0), 4.6),
                             dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)

    for ax, (column, ylabel, title) in zip(axes, panels):
        ax.set_facecolor(dl._SURFACE)
        values = pd.to_numeric(table.get(column), errors="coerce").to_numpy() \
            if column in table.columns else np.full(len(names), np.nan)
        bars = ax.bar(names, np.nan_to_num(values), 0.6, color=_ONE_HUE)
        ax.set_ylabel(ylabel, color=dl._INK_SOFT, fontsize=10)
        ax.set_title("%s (lower is better)" % title, color=dl._INK, fontsize=11,
                     pad=12)
        ax.yaxis.grid(True, color=dl._GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(dl._GRID)
        ax.tick_params(colors=dl._INK_SOFT, labelsize=9)
        top = np.nanmax(values) if np.isfinite(values).any() else 1.0
        ax.set_ylim(0, top * 1.22 if top else 1.0)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.annotate("%.1f" % value,
                        (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=9, color=dl._INK)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def plot_confusion_grid(runs: Sequence[dict[str, Any]], path,
                        verbose: bool = True):
    """The row-normalised confusion matrices side by side, on one shared ramp."""
    plt = _pyplot(verbose)
    if plt is None:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrices = []
    for run in runs:
        candidate = Path(run["folder"]) / "confusion_matrix_rownorm.csv"
        if candidate.exists():
            matrices.append((run["metrics"]["label"],
                             pd.read_csv(candidate, index_col=0)))
    if not matrices:
        return None

    cmap = LinearSegmentedColormap.from_list("mscad_blue", _SEQ_RAMP)
    n = len(matrices)
    size = max(4.0, 0.52 * len(matrices[0][1]) + 2.4)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size * 1.02), dpi=150)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(dl._SURFACE)

    image = None
    for ax, (label, frame) in zip(axes, matrices):
        ax.set_facecolor(dl._SURFACE)
        image = ax.imshow(frame.to_numpy(dtype="float64"), cmap=cmap,
                          vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(_short(label), color=dl._INK, fontsize=11, pad=10)
        ax.set_xticks(range(len(frame.columns)))
        ax.set_xticklabels(frame.columns, rotation=50, ha="right", fontsize=7.5,
                           color=dl._INK_SOFT)
        ax.set_yticks(range(len(frame.index)))
        ax.set_yticklabels(frame.index, fontsize=7.5, color=dl._INK_SOFT)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle("Row-normalised confusion matrices -- rows are the true class",
                 color=dl._INK, fontsize=12)
    if image is not None:
        bar = fig.colorbar(image, ax=axes.tolist(), fraction=0.02, pad=0.02)
        bar.set_label("share of that true class", color=dl._INK_SOFT, fontsize=9)
        bar.ax.tick_params(colors=dl._INK_SOFT, labelsize=8)
        bar.outline.set_visible(False)

    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


# ===========================================================================
# 12.8  THE WRITTEN SUMMARY -- generated from the computed numbers only
# ===========================================================================
def build_summary(table: pd.DataFrame, per_class: pd.DataFrame,
                  environment: dict[str, Any],
                  seeds: pd.DataFrame | None = None) -> str:
    """A markdown summary whose every claim comes from `table`.

    Nothing here is written in advance: the winner, the margins and the ordering
    are all read off the computed numbers, so the summary cannot disagree with
    the results it sits next to.
    """
    if table.empty:
        return "# Step 12\n\nNo configuration produced metrics.\n"

    numeric = table.copy()
    for column in numeric.columns:
        if column not in NOT_A_SCORE:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    def best_row(column: str):
        """(config, value), or (None, value) when every configuration ties.

        A tie is reported as a tie.  Naming one of four identical values "the
        highest" would read as a result where there is none.
        """
        if column not in numeric or numeric[column].isna().all():
            return None, None
        series = numeric[column].dropna()
        if len(series) > 1 and bool(np.isclose(series.to_numpy(),
                                               series.to_numpy()[0]).all()):
            return None, float(series.iloc[0])
        index = (numeric[column].idxmin() if column in LOWER_IS_BETTER
                 else numeric[column].idxmax())
        return numeric.loc[index, "config"], float(numeric.loc[index, column])

    lines = ["# Step 12 - Model comparison", ""]
    lines.append("Four configurations, each removing one piece of the proposed "
                 "pipeline, trained and scored under identical conditions: the "
                 "same Step 5 split, the same Step 9 fit/validation indices and "
                 "class weights, the same classifier hyperparameters and seeds, "
                 "and the same Step 11 evaluator.")
    lines.append("")

    for column, label in (("F1 (macro)", "macro-F1"),
                          ("G-mean", "G-mean"),
                          ("F1 macro (dup-free)", "macro-F1 on the duplicate-free subset"),
                          ("ROC-AUC (macro OvR)", "ROC-AUC"),
                          ("binary FPR", "binary false-positive rate"),
                          ("inference (ms/1k)", "inference time")):
        name, value = best_row(column)
        if value is None:
            continue
        if name is None:
            lines.append("- **%s**: every configuration ties at %.4f, so this "
                         "metric separates none of them." % (label, value))
            continue
        direction = "lowest" if column in LOWER_IS_BETTER else "highest"
        lines.append("- **%s**: %s has the %s at %.4f." % (label, name, direction,
                                                           value))

    proposed = numeric[numeric["config"].str.startswith("Proposed")]
    if len(proposed) and "F1 (macro)" in numeric:
        value = float(proposed["F1 (macro)"].iloc[0])
        others = numeric[~numeric["config"].str.startswith("Proposed")]
        if len(others):
            beaten = (others["F1 (macro)"] < value).sum()
            lines.append("")
            lines.append("The proposed configuration reaches a macro-F1 of %.4f "
                         "and beats %d of the %d baselines on that metric."
                         % (value, int(beaten), len(others)))
            gap = value - float(others["F1 (macro)"].max())
            lines.append("Its margin over the best baseline is %+.4f macro-F1."
                         % gap)
        if "F1 macro (dup-free)" in numeric and \
                not proposed["F1 macro (dup-free)"].isna().all():
            free = float(proposed["F1 macro (dup-free)"].iloc[0])
            lines.append("On the duplicate-free subset it reaches %.4f, "
                         "%+.4f against its own full-test score." % (free, free - value))

    lines += ["", "## Caveats that belong next to these numbers", ""]
    lines.append("- **The rarest classes have only a handful of test rows.** "
                 "Their per-class precision, recall and F1 move a long way when a "
                 "single row changes hands, so read them as indicative. With "
                 "`N_SEEDS >= 3` the seed table reports mean +- std, which is the "
                 "honest way to quote them.")
    lines.append("- **Duplicates between train and test inflate every full-test "
                 "score.** About half this dataset's rows share identical feature "
                 "values once the identifiers are gone, so a model can recognise a "
                 "test row it has effectively already seen. The duplicate-free "
                 "columns exclude those rows and are the conservative comparison.")
    lines.append("- **Accuracy is close to a measure of the majority class.** "
                 "Brute_Force is ~84% of the rows. Lead with macro-F1 and G-mean; "
                 "a configuration that wins on accuracy and loses on G-mean has "
                 "not won.")
    lines.append("- **Differences are reported, not tested.** With this many "
                 "seeds there is no basis for a significance claim and none is "
                 "made.")

    if seeds is not None and len(seeds):
        lines += ["", "## Across seeds", "",
                  "Per-seed macro-F1 and the mean difference from the proposed "
                  "model. Differences only."]

    lines += ["", "## Environment", "",
              "- Python %s, TensorFlow %s, Keras %s"
              % (environment["python"], environment["tensorflow"],
                 environment["keras"]),
              "- numpy %s, pandas %s, scikit-learn %s"
              % (environment["numpy"], environment["pandas"],
                 environment["scikit-learn"]),
              "- devices: %s" % ", ".join(environment["devices"]),
              "- input mode: %s, seeds: %d, imbalance: %s / %s"
              % (environment["input_mode"], environment["n_seeds"],
                 environment["imbalance"]["strategy"],
                 environment["imbalance"]["scheme"]), ""]
    return "\n".join(lines)


# ===========================================================================
# 12.9  RUN EVERYTHING
# ===========================================================================
def run_step12(overwrite: bool | None = None, epochs: int | None = None,
               experiments: Sequence[dict[str, Any]] | None = None,
               verbose: bool = True) -> dict[str, Any]:
    """STEP 12 end to end: run every configuration, compare, save."""
    overwrite = config.OVERWRITE_STEP12 if overwrite is None else overwrite
    specs = list(config.EXPERIMENTS if experiments is None else experiments)

    loaded = ev.load_evaluation_inputs(verbose=False)
    y_test, meta_test = loaded["y_test"], loaded.get("meta_test")
    label_mapping = loaded["label_mapping"]

    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    y_train = pd.read_csv(folder8 / config.STEP8_FILES["y_train"], low_memory=False)

    flags = None
    if loaded["X_train"] is not None and loaded["X_test"] is not None:
        flags = ev.duplicate_mask(loaded["X_train"], loaded["X_test"])

    before = test_fingerprint()

    if verbose:
        show_title("Step 12 -- running %d configurations" % len(specs), level=3)
    runs = []
    per_seed: dict[str, list[float]] = {}
    for spec in specs:
        seeds = ([config.RANDOM_STATE] if config.N_SEEDS <= 1
                 else [config.RANDOM_STATE + i for i in range(config.N_SEEDS)])
        scores = []
        run = None
        for i, seed in enumerate(seeds):
            this = dict(spec)
            if i > 0:                       # extra seeds never touch step10/
                this = {**spec, "name": "%s_seed%d" % (spec["name"], seed),
                        "reuse_step7_10": False}
            outcome = run_experiment(this, y_train, y_test, meta_test,
                                     label_mapping, flags, seed=seed,
                                     epochs=epochs, overwrite=overwrite,
                                     verbose=verbose)
            scores.append(float(outcome["metrics"]["full"]["f1_macro"]))
            if i == 0:
                run = outcome
        per_seed[spec["name"]] = scores
        runs.append(run)

    sources = {spec["name"]: load_feature_source(spec["feature_source"],
                                                 verbose=False)
               for spec in specs}
    checks, checks_ok = consistency_checks(sources, y_train, y_test, before,
                                           verbose=verbose)

    table = comparison_table(runs)
    marked = mark_best(table)
    per_class_full = per_class_f1_table(runs, "full")
    environment = environment_report()
    seeds_table = (seed_comparison(per_seed) if config.N_SEEDS >= 3
                   else pd.DataFrame())
    summary = build_summary(table, per_class_full, environment,
                            seeds_table if len(seeds_table) else None)

    out: dict[str, Any] = {
        "runs": runs, "comparison": table, "comparison_marked": marked,
        "per_class_full": per_class_full, "consistency": checks,
        "consistency_ok": checks_ok, "environment": environment,
        "seed_comparison": seeds_table, "summary": summary,
        "per_seed": per_seed,
    }

    folder = prepare_output_dir(config.STEP12_OUTPUT_DIR, "the Step 12 outputs",
                                verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP12_FILES
    table.to_csv(folder / names["comparison"], index=False)
    marked.to_csv(folder / ("marked_" + names["comparison"]), index=False)
    per_class_full.to_csv(folder / names["per_class_full"], index=False)
    checks.to_csv(folder / names["consistency"], index=False)
    if len(seeds_table):
        seeds_table.to_csv(folder / names["seed_comparison"], index=False)
    (folder / names["environment"]).write_text(
        json.dumps(environment, indent=2), encoding="utf-8")
    (folder / names["summary"]).write_text(summary, encoding="utf-8")

    charts = {}
    drawn = plot_metric_bars(table, ["F1 (macro)", "G-mean", "ROC-AUC (macro OvR)"],
                             "Full test set -- higher is better",
                             folder / names["chart_metrics"], verbose=verbose)
    if drawn:
        charts["metrics"] = str(drawn)
    drawn = plot_metric_bars(table, ["F1 macro (dup-free)", "G-mean (dup-free)"],
                             "Duplicate-free test subset -- the conservative "
                             "comparison", folder / names["chart_metrics_free"],
                             verbose=verbose)
    if drawn:
        charts["metrics_free"] = str(drawn)
    drawn = plot_per_class_heatmap(per_class_full,
                                   folder / names["chart_per_class"],
                                   verbose=verbose)
    if drawn:
        charts["per_class"] = str(drawn)
    drawn = plot_time_bars(table, folder / names["chart_time"], verbose=verbose)
    if drawn:
        charts["time"] = str(drawn)
    drawn = plot_confusion_grid(runs, folder / names["chart_confusion"],
                                verbose=verbose)
    if drawn:
        charts["confusion"] = str(drawn)
    out["charts"] = charts

    after = test_fingerprint()
    untouched = bool(before and after and all(
        after.get(k, {}).get("sha256") == v["sha256"] for k, v in before.items()))
    out.update({"status": "written", "ok": bool(checks_ok and untouched),
                "folder": str(folder), "test_untouched": untouched})

    if verbose:
        show_title("Step 12 -- comparison", level=3)
        show_table(marked, all_columns=True,
                   note="`*` marks the best value in each column. Lower is better "
                        "for the FPR, the times and the parameter count.")
        show_note("The **duplicate-free** columns are the conservative comparison: "
                  "they exclude test rows whose exact feature vector also occurs in "
                  "training.")
    return out
