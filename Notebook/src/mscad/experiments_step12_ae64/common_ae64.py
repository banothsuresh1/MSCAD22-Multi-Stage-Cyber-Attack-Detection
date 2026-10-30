r"""
src/mscad/experiments_step12_ae64/common_ae64.py -- plumbing for the AE=64
bottleneck-sensitivity run (B1, B2, B4, B6; B3 copied; B5 skipped).

NEW FILE.  It mirrors experiments_step12_ae30/common_ae30.py: a dedicated
output root (config.OUTPUTS_ROOT_AE64), this run's own lock
(config.LOCK_TRAINED_MODELS_AE64), a context-manager redirect, and a
`run_one_ae64` entry point.

The AE and the B2/B4 LSTM are NOT re-implemented here: the two builders are
the published paper's, reused from experiments_step12_paper_replication
(build_paper_autoencoder: 66 -> 64, no hidden layer, ReLU bottleneck, sigmoid
output; build_paper_lstm: 1024 -> dropout 0.6 -> 512, tanh).  B1 and B6 keep
their EXISTING CNN-LSTM classifier through experiments.train_classifier,
unchanged -- only the upstream AE width (-> 64) changes for them.  The shared
evaluator (evaluation.evaluate) is reused so every metrics.json matches the
schema comparison.py already reads.

Scope, enforced by construction:
  * inputs (Steps 4, 5, 8) are read from the active official root, so the run
    is scored on the same test rows as the AE=15 / AE=30 tables;
  * every write lands in config.OUTPUTS_ROOT_AE64\<name>\, nowhere else;
  * the AE, the scaler (Step 5, already fit) and PCA are fit on TRAIN rows
    only, verified by assert_pca_fitted_on_train_only and by the AE never
    seeing the test rows during fit.

    python -m src.mscad.experiments_step12_ae64.baseline1_ae64
"""

from __future__ import annotations

import contextlib
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from .. import comparison as cmp  # noqa: E402
from .. import data_loading as dl  # noqa: E402
from .. import evaluation as ev  # noqa: E402
from .. import experiments as xp  # noqa: E402
from .. import experiments_pca as xpca  # noqa: E402
from .. import models as md  # noqa: E402
from ..experiments_step12 import common as base  # noqa: E402
from ..experiments_step12_paper_replication import (  # noqa: E402
    baseline_ae64_pca30_lstm as paper)

Step12ConfigError = base.Step12ConfigError
LOADED, TRAINED, PENDING = base.LOADED, base.TRAINED, base.PENDING

# the paper's hyperparameters, reused (NOT redefined here)
AE_LATENT = paper.AE_LATENT          # 64
AE_BATCH, AE_EPOCHS = paper.AE_BATCH, paper.AE_EPOCHS      # 850, 20 fixed
CLF_BATCH, CLF_EPOCHS = paper.CLF_BATCH, paper.CLF_EPOCHS  # 850, 20 fixed
PCA_COMPONENTS = paper.PCA_COMPONENTS  # 30
SEED = int(getattr(config, "RANDOM_STATE", 42))


class Ae64Error(Exception):
    """A problem with the AE=64 run itself, not with a trainer."""


# ===========================================================================
# 1. where this run writes
# ===========================================================================
def root() -> Path | None:
    return config.usable_output_dir(config.OUTPUTS_ROOT_AE64)


def lock_is_on() -> bool:
    return bool(getattr(config, "LOCK_TRAINED_MODELS_AE64", True))


def folder(name: str) -> Path | None:
    place = root()
    return None if place is None else place / name


def metrics_path(name: str) -> Path | None:
    place = folder(name)
    return None if place is None else place / "metrics.json"


def is_finished(name: str) -> bool:
    path = metrics_path(name)
    return bool(path is not None and path.exists())


def inside_ae64(path: Path | str | None) -> bool:
    if path is None:
        return False
    place = root()
    if place is None:
        return False
    parts, root_parts = Path(path).parts, Path(place).parts
    return bool(root_parts) and parts[:len(root_parts)] == root_parts


# ===========================================================================
# 2. the redirect -- only the root and this run's lock
# ===========================================================================
@contextlib.contextmanager
def ae64() -> Iterator[Path]:
    """Point config's output location at the AE=64 root for one call.

    Only OUTPUTS_ROOT/OUTPUTS_DIR and the lock: inputs (Steps 4/5/8) stay on
    the official root, and every write is addressed explicitly under
    folder(name).  Restored afterwards, exception or not.
    """
    place = root()
    if place is None:
        raise Ae64Error(
            "%s cannot be written on this machine, so nothing was trained.\n"
            "    %s" % (config.OUTPUTS_ROOT_AE64,
                        config.why_unusable(config.OUTPUTS_ROOT_AE64)))
    wanted = {"OUTPUTS_ROOT": Path(place), "OUTPUTS_DIR": Path(place),
              "LOCK_TRAINED_MODELS": lock_is_on()}
    missing = object()
    before = {k: getattr(config, k, missing) for k in wanted}
    for k, v in wanted.items():
        setattr(config, k, v)
    try:
        Path(place).mkdir(parents=True, exist_ok=True)
        yield Path(place)
    finally:
        for k, v in before.items():
            if v is missing:                           # pragma: no cover
                delattr(config, k)
            else:
                setattr(config, k, v)


def inputs(verbose: bool = False) -> dict[str, Any]:
    """Labels, test set, duplicate mask -- from the active official root."""
    return base.evaluation_inputs(verbose=verbose)


# ===========================================================================
# 3. the shared pipeline pieces (fit on TRAIN rows only)
# ===========================================================================
def feature_frame_66():
    """Step 5 'all' features with zero_duration dropped -> 66 columns."""
    source = xp.load_feature_source("all", verbose=False)
    X_train, X_test = source["X_train"], source["X_test"]
    col = getattr(config, "ZERO_DURATION_COLUMN", "zero_duration")
    dropped = None
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])
        dropped = col
    return X_train, X_test, dropped


def encode_ae64(X_train, X_test, y_train, place, verbose=False):
    """Paper AE 66 -> 64 (no hidden), fit on the TRAINING rows only.

    The training matrix is converted to float32 ONCE and reused for the fit
    (input and target) and the encode, instead of three separate .to_numpy()
    copies -- same result, lower peak memory.  The autoencoder (its decoder
    head and Adam state) is dropped as soon as the fit is done, before the
    encodings are built, so only the encoder survives.
    """
    Xtr = X_train.to_numpy("float32")          # one copy, reused below
    autoencoder, encoder = paper.build_paper_autoencoder(Xtr.shape[1])
    autoencoder.fit(Xtr, Xtr, epochs=AE_EPOCHS, batch_size=AE_BATCH,
                    shuffle=True, verbose=2 if verbose else 0)
    del autoencoder                            # decoder + optimizer state done
    gc.collect()
    cols = ["ae_%d" % i for i in range(AE_LATENT)]
    Z_train = pd.DataFrame(encoder.predict(Xtr, batch_size=AE_BATCH, verbose=0),
                           columns=cols, index=X_train.index)
    del Xtr
    Z_test = pd.DataFrame(encoder.predict(X_test.to_numpy("float32"),
                                          batch_size=AE_BATCH, verbose=0),
                          columns=cols, index=X_test.index)
    encoder.save(Path(place) / "encoder.keras")
    gc.collect()
    return Z_train, Z_test, encoder


def pca30(Z_train, Z_test, place, verbose=False):
    """PCA -> 30 components, fit on the TRAINING latent only (leak-checked)."""
    pca, info = xpca.fit_pca_on_train(Z_train, setting=PCA_COMPONENTS, seed=SEED)
    leak = xpca.assert_pca_fitted_on_train_only(pca, Z_train, Z_test,
                                                setting=PCA_COMPONENTS, seed=SEED)
    if leak["result"] != "[OK]":
        raise Ae64Error("PCA was not fitted on the training rows alone: %s"
                        % leak["detail"])
    variance = xpca.pca_variance_table(pca)
    P_train = xpca.transform_with_pca(pca, Z_train, "the training latent")
    P_test = xpca.transform_with_pca(pca, Z_test, "the test latent")
    from joblib import dump
    dump(pca, Path(place) / "pca.joblib")
    variance.to_csv(Path(place) / "pca_variance.csv", index=False)
    return P_train, P_test, pca, info, variance


@contextlib.contextmanager
def _classifier_fit_verbose(level: int = 2) -> Iterator[None]:
    """Make the CNN-LSTM classifier's fit() log per-epoch progress (AE=64 only).

    experiments.train_classifier is shared by every baseline, approach and AE
    width, and it hardcodes model.fit(..., verbose=0): its CNN-LSTM stage prints
    nothing for the whole run, so from the outside a long fit is indistinguishable
    from a hang.  We must NOT change that shared call (doing so would alter every
    other width's verbosity), so for the AE=64 run only we wrap keras.Model.fit
    for the duration of the one train_classifier call and raise its fit verbosity
    to `level` -- 2 is one line per epoch, exactly what encode_ae64's AE stage
    already prints with verbose=2.  The original method is restored on the way
    out, exception or not; nothing outside this context, and no other
    baseline/approach/width, is touched.
    """
    keras = md._keras()
    original = keras.Model.fit

    def _fit(self, *args, **kwargs):
        kwargs["verbose"] = level
        return original(self, *args, **kwargs)

    keras.Model.fit = _fit
    try:
        yield
    finally:
        keras.Model.fit = original


def train_cnn_lstm(P_train, y_train, verbose=False):
    """B1/B6's EXISTING CNN-LSTM classifier, unchanged (experiments.train_classifier).

    train_classifier hardcodes its fit() verbosity to 0, so when `verbose` is on
    we wrap that single call (AE=64 run only) so the classifier logs per-epoch
    progress like the AE stage.  The classifier and its training are unchanged."""
    if not verbose:
        return xp.train_classifier(P_train, y_train, model_type="cnn_lstm",
                                   seed=SEED, verbose=verbose)
    with _classifier_fit_verbose(2):
        return xp.train_classifier(P_train, y_train, model_type="cnn_lstm",
                                   seed=SEED, verbose=verbose)


def train_paper_lstm(P_train, y_train, n_classes, verbose=False):
    """B2/B4's paper LSTM: 1024 -> dropout 0.6 -> 512, 20 fixed epochs, batch 850."""
    X_seq, y_enc, _ = md.arrange_inputs(P_train, y_train, meta=None,
                                        mode="feature_seq")
    model = paper.build_paper_lstm((X_seq.shape[1], X_seq.shape[2]), n_classes)
    gc.collect()                        # sweep transients before the heavy fit
    started = time.perf_counter()
    model.fit(X_seq, np.asarray(y_enc), epochs=CLF_EPOCHS, batch_size=CLF_BATCH,
              shuffle=True, verbose=2 if verbose else 0)
    info = {"seconds": time.perf_counter() - started,
            "input_shape": [int(X_seq.shape[1]), int(X_seq.shape[2])],
            "parameters": int(model.count_params()), "seed": SEED}
    return model, None, info


# ===========================================================================
# 4. evaluate + save, in the schema comparison.py reads
# ===========================================================================
def evaluate_and_save(name, label, model, P_test, y_test, meta_test,
                      label_mapping, duplicate_flags, descriptors, info,
                      place, verbose=False):
    X_seq, _labels, row_index = md.arrange_inputs(P_test, y_test, meta_test,
                                                  mode="feature_seq")
    flags = None
    if duplicate_flags is not None:
        flags = (np.asarray(duplicate_flags)[row_index]
                 if len(row_index) != len(duplicate_flags) else duplicate_flags)
    result = ev.evaluate(model, X_seq, y_test, label_mapping,
                         meta_test=meta_test, duplicate_flags=flags,
                         model_name=label,
                         training_seconds=info.get("seconds"), verbose=False)
    model.save(Path(place) / "model.keras")
    result["per_class"].to_csv(Path(place) / "per_class_report.csv", index=False)
    ev.build_predictions_frame(result, meta_test, row_index).to_csv(
        Path(place) / "predictions.csv", index=False)
    metrics = {
        "config": name, "label": label, "ae64_sensitivity": True,
        "latent_dim": int(AE_LATENT),
        "model_input_shape": info.get("input_shape"),
        "parameters": int(info.get("parameters", model.count_params())),
        "seed": info.get("seed", SEED),
        "training_seconds": info.get("seconds"),
        "best_epoch": info.get("best_epoch"),
        "hyperparameters": {"source": "paper AE + per-baseline classifier",
                            "ae": {"latent": AE_LATENT, "hidden": [],
                                   "batch": AE_BATCH, "epochs": AE_EPOCHS,
                                   "early_stopping": False}},
        **descriptors,
        "full": result["full"], "duplicate_free": result["duplicate_free"],
        "binary": result["binary"], "inference": result["inference"],
    }
    (Path(place) / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return {"metrics": metrics, "result": result, "status": "written"}


# ===========================================================================
# 5. the entry point, mirroring run_one_ae30
# ===========================================================================
def run_one_ae64(name: str, trainer: Callable[..., Any],
                 verbose: bool = True) -> dict[str, Any]:
    """Load the AE=64 result if it exists, else train it inside the redirect."""
    path = metrics_path(name)
    if verbose:
        dl.show_title("AE=64: %s" % name, level=4)
    if path is None:
        raise Ae64Error(
            "%s cannot be read on this machine, so nothing was loaded and "
            "nothing was trained.\n    %s"
            % (name, config.why_unusable(config.OUTPUTS_ROOT_AE64)))
    if path.exists():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        status, detail = LOADED, "loaded from %s" % path
    elif lock_is_on():
        return {"name": name, "status": PENDING, "pending": True,
                "status_detail": "not trained (LOCK_TRAINED_MODELS_AE64 is True)",
                "metrics": {}, "metrics_path": str(path), "ok": False}
    else:
        with ae64():
            out = trainer() or {}
        metrics = out.get("metrics")
        if metrics is None:
            raise Ae64Error("%s finished but returned no metrics" % name)
        status, detail = TRAINED, "trained into %s" % path.parent
    result = {"name": name, "status": status, "pending": False,
              "status_detail": detail, "metrics": metrics,
              "metrics_path": str(path), "ok": True}
    if verbose:
        print("  [%s] %s: %s"
              % ("LOAD" if status == LOADED else "TRAIN", name, detail))
        dl.show_table(base.headline(metrics), note="%s -- %s" % (name, detail))
    return result


# ===========================================================================
# 6. the AE=64 comparison table, built by the SAME builder as AE=15 / AE=30
# ===========================================================================
# The five configurations this run covers, in the AE=15 display order.  The
# two approaches and B5 are NOT part of the AE=64 run -- the approaches were
# not re-bottlenecked, and B5 is skipped (see SKIP_REASON) -- so they are not
# builder rows here; B5 is appended afterwards as an explicit "skipped" row.
AE64_KEYS = {"baseline1_ae_cnn_lstm", "baseline2_ae_lstm", "baseline3_cnn_lstm",
             "baseline4_ae_pca_lstm", "baseline6_ae_pca_cnn_lstm"}
AE64_ORDER = [(k, l) for k, l in config.STEP12_DISPLAY_ORDER if k in AE64_KEYS]
SKIPPED_KEY = "baseline5"
SKIP_REASON = ("skipped: its AE trains on 30 MI-selected features, so a 64 "
               "bottleneck would expand 30 -> 64, not compress")


@contextlib.contextmanager
def _table_redirect() -> Iterator[Path | None]:
    """Map comparison.result_path onto the FLAT AE=64 layout for one read.

    The AE=64 results are written flat -- <root>\\<NAME>\\metrics.json, one
    folder per configuration -- not the nested step11/step12 layout the AE=15
    and AE=30 roots use.  comparison.build_full_comparison resolves each key
    through comparison.result_path, which reads STEP12_OUTPUT_DIR,
    STEP12_B6_OUTPUT_DIR, STEP11_OUTPUT_DIR and STEP11B_OUTPUT_DIR.  Pointing
    those at the AE=64 root (and B6's own folder beneath it) makes the EXISTING
    builder read exactly the flat files for B1-B4 and B6; the keys with no
    AE=64 result resolve to a path that does not exist, so they come back as
    "not yet trained" instead of falling through to the AE=15 root.  Read-only:
    nothing is written, every attribute is restored afterwards, and it works
    even where the root is unusable (every key then resolves to None)."""
    place = config.OUTPUTS_ROOT_AE64
    wanted = {
        "STEP12_OUTPUT_DIR": Path(place),
        "STEP12_B6_OUTPUT_DIR": Path(place) / "baseline6_ae_pca_cnn_lstm",
        "STEP11_OUTPUT_DIR": Path(place),
        "STEP11B_OUTPUT_DIR": Path(place),
    }
    missing = object()
    before = {k: getattr(config, k, missing) for k in wanted}
    for k, v in wanted.items():
        setattr(config, k, v)
    try:
        yield root()
    finally:
        for k, v in before.items():
            if v is missing:                           # pragma: no cover
                delattr(config, k)
            else:
                setattr(config, k, v)


def comparison_table(order: Sequence[tuple[str, str]] | None = None
                     ) -> "pd.DataFrame":
    """The AE=64 table, built by the SAME builder the AE=15/AE=30 tables use."""
    order = AE64_ORDER if order is None else order
    with _table_redirect():
        built = cmp.build_full_comparison(order=order, verbose=False)
    return built["table"]


def _b5_skip_row() -> dict[str, Any]:
    """B5's explicit 'skipped' row, in the builder's own columns."""
    row = {column: "-" for column in cmp.COLUMNS}
    row["config"] = base.display_label(SKIPPED_KEY)
    row["input features"] = SKIP_REASON
    return row


def marked_comparison_table(order: Sequence[tuple[str, str]] | None = None,
                            include_skipped: bool = True) -> "pd.DataFrame":
    """The AE=64 table with best-value `*` marks, plus the explicit B5 row.

    mark_best is the SAME marker the AE=15/AE=30 tables use, so columns and the
    `*` convention match across all three widths.  The B5 row is appended AFTER
    marking (it carries no values to compare), so B5 reads as skipped with its
    reason rather than as a blank 'not trained' row."""
    order = AE64_ORDER if order is None else order
    with _table_redirect():
        built = cmp.build_full_comparison(order=order, verbose=False)
        marked = cmp.mark_best(built["table"])
    if include_skipped:
        marked = pd.concat([marked, pd.DataFrame([_b5_skip_row()])],
                           ignore_index=True)[cmp.COLUMNS]
    return marked


def provenance(order: Sequence[tuple[str, str]] | None = None) -> "pd.DataFrame":
    """Every AE=64 row, its state, and the exact file it was read from."""
    order = AE64_ORDER if order is None else order
    rows = []
    with _table_redirect():
        for key, label in order:
            path = cmp.result_path(key)
            exists = bool(path is not None and Path(path).exists())
            rows.append({
                "config": label,
                "state": "read from disk" if exists else "not trained at AE=64",
                "metrics.json": "(unreachable here)" if path is None else str(path),
                "file exists": "yes" if exists else "no",
            })
    rows.append({"config": base.display_label(SKIPPED_KEY), "state": "skipped",
                 "metrics.json": "(not run)", "file exists": "n/a"})
    return pd.DataFrame(rows)
