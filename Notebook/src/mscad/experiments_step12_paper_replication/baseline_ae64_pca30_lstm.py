r"""
baseline_ae64_pca30_lstm.py -- the paper's AE(->64) + PCA(->30) + LSTM(1024/512)
pipeline, replicated on the official MSCAD dataset.

NO hyperparameter here is read from the shared config.AE_* or classifier
constants: every one is the PAPER's, hardcoded below, so this variant is
independent of the AE=15 and AE=30 pipelines.  The autoencoder and the LSTM
are built HERE (not through models.build_cnn_lstm / autoencoder.build_
autoencoder, which read the shared config), because the paper's shapes differ
from every existing configuration.  The EVALUATION is reused unchanged --
evaluation.evaluate -- so the metrics.json matches the schema comparison.py
already reads, with the same macro-ROC-AUC, G-mean, binary-FPR and inference
conventions as every other row.

Where the paper's spec and this codebase conflict, the call made is recorded
both here and in docs/paper_replication_notes.md:

1. FEATURE COUNT -- the paper says 66; Step 4 (INF_STRATEGY = "cap") adds a
   67th column, "zero_duration".  DECISION: drop zero_duration, train on 66,
   to match the paper exactly.  Rationale: this baseline exists to compare
   against the paper's REPORTED numbers, the paper is explicit about 66, and
   zero_duration is a project artifact the paper's authors never had.  B4
   keeps all 67; that is the deviation this row is meant to isolate.

2. AE DEPTH -- the paper's Algorithm 1 is read as a single-layer encoder and
   decoder, so there are NO hidden layers between input and bottleneck:
   66 -> 64 -> 66.  (At 66 -> 64 the "bottleneck" removes only two dimensions;
   that is what the paper specifies, implemented as written.)  CONFIRM against
   the paper text if it shows a staged encoder.

3. AE OUTPUT ACTIVATION -- the paper says ReLU.  This project's Step 5 scaler
   is MinMax into [0, 1], and a sigmoid output reconstructs [0, 1] cleanly; a
   ReLU output has no upper bound and fits [0, 1] data poorly.  DECISION: ReLU
   on the encoder/bottleneck (matches the paper), SIGMOID on the reconstruction
   output (sound for the MinMax scaler).  The paper's "ReLU" is taken as the
   hidden/bottleneck activation.  Flag: if the paper means a ReLU OUTPUT, it
   implies a different input scaling than this project uses.

4. CLASSIFIER OUTPUT / SOFTMAX -- the LSTM emits LOGITS (a linear Dense head);
   the softmax is applied by evaluation.predict_proba, the one place this
   project applies it, for every row.  Functionally identical to the paper's
   "Softmax output layer": the probabilities and the argmax are the same.

5. SEQUENCE SHAPE -- the 30 PCA components are arranged as a sequence of 30
   steps x 1 feature (models.as_feature_sequence, INPUT_MODE "feature_seq"),
   this project's convention for every LSTM row, so the comparison against B4
   is on the same input arrangement.  The paper's "input size = 30" is read as
   this.

6. CLASSIFIER OPTIMIZER -- the paper names Adam for the AE and does not state
   the classifier's optimizer in the spec provided.  DECISION: Adam for the
   classifier too, matching the AE and this project's default.  Flag if the
   paper specifies otherwise.

Paper hyperparameters, fixed:
  AE    : 66 -> 64, no hidden layer, ReLU bottleneck, sigmoid output, MSE,
          Adam, batch 850, 20 fixed epochs, NO early stopping
  PCA   : 64 -> 30 components (exact)
  LSTM  : 30 -> LSTM(1024, tanh) -> Dropout(0.6) -> LSTM(512, tanh) -> logits,
          Adam, batch 850, 20 fixed epochs, NO early stopping

    python -m src.mscad.experiments_step12_paper_replication.baseline_ae64_pca30_lstm
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from . import common_paper as cp  # noqa: E402
from .. import evaluation as ev  # noqa: E402
from .. import experiments as xp  # noqa: E402
from .. import experiments_pca as xpca  # noqa: E402
from .. import models as md  # noqa: E402

# ---- the paper's hyperparameters, hardcoded (NOT from shared config) -------
NAME = "paper_ae64_pca30_lstm"
LABEL = "Paper: AE(66->64) -> PCA(30) -> LSTM(1024/512)"
DROP_ZERO_DURATION = True        # decision 1: 66 features, not 67
AE_LATENT = 64                   # decision 2
AE_HIDDEN: list[int] = []        # decision 2: single-layer encoder/decoder
AE_BOTTLENECK_ACT = "relu"       # decision 3: paper's ReLU, on the bottleneck
AE_OUTPUT_ACT = "sigmoid"        # decision 3: sound for the MinMax [0,1] scaler
AE_LOSS = "mse"
AE_LR = 1e-3                     # Adam default; paper names Adam, not a rate
AE_BATCH = 850
AE_EPOCHS = 20                   # fixed, no early stopping
PCA_COMPONENTS = 30              # decision: exact, from the 64-dim bottleneck
LSTM_UNITS_1 = 1024
LSTM_UNITS_2 = 512
LSTM_DROPOUT = 0.6
LSTM_ACT = "tanh"
CLF_LR = 1e-3                   # decision 6: Adam for the classifier too
CLF_BATCH = 850
CLF_EPOCHS = 20                 # fixed, no early stopping
SEED = int(getattr(config, "RANDOM_STATE", 42))


# ===========================================================================
# the models, built here at the paper's shapes
# ===========================================================================
def build_paper_autoencoder(n_features: int):
    """input -> 64 (ReLU) -> input (sigmoid).  No hidden layer.  Returns
    (autoencoder, encoder) sharing the trained bottleneck weights."""
    keras = md._keras()
    inputs = keras.Input(shape=(n_features,), name="input")
    x = inputs
    for i, units in enumerate(AE_HIDDEN, start=1):           # empty: no hidden
        x = keras.layers.Dense(units, activation=AE_BOTTLENECK_ACT,
                               name="encoder_%d" % i)(x)
    latent = keras.layers.Dense(AE_LATENT, activation=AE_BOTTLENECK_ACT,
                                name="latent")(x)
    y = latent
    for i, units in enumerate(reversed(AE_HIDDEN), start=1):
        y = keras.layers.Dense(units, activation=AE_BOTTLENECK_ACT,
                               name="decoder_%d" % i)(y)
    outputs = keras.layers.Dense(n_features, activation=AE_OUTPUT_ACT,
                                 name="reconstruction")(y)
    autoencoder = keras.Model(inputs, outputs, name="paper_autoencoder")
    encoder = keras.Model(inputs, latent, name="paper_encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=AE_LR),
                        loss=AE_LOSS)
    return autoencoder, encoder


def build_paper_lstm(input_shape: tuple[int, int], n_classes: int):
    """LSTM(1024, tanh) -> Dropout(0.6) -> LSTM(512, tanh) -> logits.

    The head is linear: evaluation.predict_proba applies the softmax, as it
    does for every row (decision 4).
    """
    keras = md._keras()
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.LSTM(LSTM_UNITS_1, activation=LSTM_ACT,
                          return_sequences=True, name="lstm_1"),
        keras.layers.Dropout(LSTM_DROPOUT, name="dropout"),
        keras.layers.LSTM(LSTM_UNITS_2, activation=LSTM_ACT, name="lstm_2"),
        keras.layers.Dense(n_classes, name="logits"),
    ], name="paper_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CLF_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    return model


# ===========================================================================
# the run
# ===========================================================================
def _feature_frame():
    """The Step 5 'all' features, with zero_duration dropped per decision 1."""
    source = xp.load_feature_source("all", verbose=False)
    X_train, X_test = source["X_train"], source["X_test"]
    dropped = None
    col = getattr(config, "ZERO_DURATION_COLUMN", "zero_duration")
    if DROP_ZERO_DURATION and col in X_train.columns:
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])
        dropped = col
    return X_train, X_test, dropped


def train(verbose: bool = True) -> dict[str, Any]:
    """Train the paper pipeline and score it with the shared evaluator.

    Only reached with LOCK_TRAINED_MODELS_PAPER = False and no saved result,
    and only from inside common_paper.paper_root(), so every output lands under
    the replication root.
    """
    keras = md._keras()
    np.random.seed(SEED)
    try:
        keras.utils.set_random_seed(SEED)
    except Exception:                                        # pragma: no cover
        pass

    data = cp.inputs(verbose=verbose)
    y_train, y_test = data["y_train"], data["y_test"]
    meta_test, label_mapping = data["meta_test"], data["label_mapping"]
    duplicate_flags = data["duplicate_flags"]

    X_train, X_test, dropped = _feature_frame()
    n_features = X_train.shape[1]
    place = cp.folder(NAME)
    place.mkdir(parents=True, exist_ok=True)

    # ---- AE: train on the TRAINING rows only, then encode both sides -------
    started = time.perf_counter()
    autoencoder, encoder = build_paper_autoencoder(n_features)
    autoencoder.fit(X_train.to_numpy(dtype="float32"),
                    X_train.to_numpy(dtype="float32"),
                    epochs=AE_EPOCHS, batch_size=AE_BATCH, shuffle=True,
                    verbose=2 if verbose else 0)
    cols = ["ae_%d" % i for i in range(AE_LATENT)]
    Z_train = pd.DataFrame(encoder.predict(X_train.to_numpy("float32"),
                                           batch_size=AE_BATCH, verbose=0),
                           columns=cols, index=X_train.index)
    Z_test = pd.DataFrame(encoder.predict(X_test.to_numpy("float32"),
                                          batch_size=AE_BATCH, verbose=0),
                          columns=cols, index=X_test.index)

    # ---- PCA: exactly 30 components, fitted on the TRAINING latent only ----
    pca, pca_info = xpca.fit_pca_on_train(Z_train, setting=PCA_COMPONENTS,
                                          seed=SEED)
    leak = xpca.assert_pca_fitted_on_train_only(pca, Z_train, Z_test,
                                                setting=PCA_COMPONENTS, seed=SEED)
    if leak["result"] != "[OK]":
        raise cp.PaperReplicationError(
            "the PCA moved when the test rows changed, so it was not fitted on "
            "the training rows alone: %s" % leak["detail"])
    variance = xpca.pca_variance_table(pca)
    P_train = xpca.transform_with_pca(pca, Z_train, "the training latent")
    P_test = xpca.transform_with_pca(pca, Z_test, "the test latent")

    # ---- LSTM: 30 PCA comps as a 30x1 sequence (decision 5) ----------------
    X_train_seq, y_train_enc, _ = md.arrange_inputs(P_train, y_train,
                                                    meta=None, mode="feature_seq")
    n_classes = int(np.asarray(y_train_enc).max() + 1)
    if label_mapping:
        n_classes = max(n_classes, len(ev.label_names_from_mapping(label_mapping)))
    model = build_paper_lstm((X_train_seq.shape[1], X_train_seq.shape[2]),
                             n_classes)
    model.fit(X_train_seq, np.asarray(y_train_enc), epochs=CLF_EPOCHS,
              batch_size=CLF_BATCH, shuffle=True, verbose=2 if verbose else 0)
    training_seconds = time.perf_counter() - started

    # ---- evaluate with the SHARED evaluator (schema + conventions) ---------
    X_test_seq, _labels, row_index = md.arrange_inputs(P_test, y_test,
                                                       meta_test,
                                                       mode="feature_seq")
    flags = None
    if duplicate_flags is not None:
        flags = (np.asarray(duplicate_flags)[row_index]
                 if len(row_index) != len(duplicate_flags) else duplicate_flags)
    result = ev.evaluate(model, X_test_seq, y_test, label_mapping,
                         meta_test=meta_test, duplicate_flags=flags,
                         model_name=LABEL, training_seconds=training_seconds,
                         verbose=False)

    # ---- save, in the layout comparison.py reads --------------------------
    model.save(place / "model.keras")
    encoder.save(place / "encoder.keras")
    from joblib import dump
    dump(pca, place / "pca.joblib")
    variance.to_csv(place / "pca_variance.csv", index=False)
    result["per_class"].to_csv(place / "per_class_report.csv", index=False)
    ev.build_predictions_frame(result, meta_test, row_index).to_csv(
        place / "predictions.csv", index=False)

    metrics = {
        "config": NAME, "label": LABEL,
        "paper_replication": True,
        "n_input_features": int(n_features),
        "dropped_column": dropped,
        "latent_dim": int(AE_LATENT),
        "pca_components": int(pca_info["n_components"]),
        "pca": pca_info | {"variance_per_component":
                           variance["explained_variance_ratio"].tolist()},
        "model_input_shape": [int(X_train_seq.shape[1]),
                              int(X_train_seq.shape[2])],
        "parameters": int(model.count_params()),
        "ae_parameters": int(autoencoder.count_params()),
        "seed": SEED,
        "training_seconds": training_seconds,
        "hyperparameters": {
            "source": "paper (hardcoded), not shared config",
            "ae": {"input": int(n_features), "latent": AE_LATENT,
                   "hidden": AE_HIDDEN, "bottleneck_activation": AE_BOTTLENECK_ACT,
                   "output_activation": AE_OUTPUT_ACT, "loss": AE_LOSS,
                   "optimizer": "adam", "lr": AE_LR, "batch": AE_BATCH,
                   "epochs": AE_EPOCHS, "early_stopping": False},
            "pca": {"components": PCA_COMPONENTS},
            "lstm": {"units": [LSTM_UNITS_1, LSTM_UNITS_2],
                     "dropout": LSTM_DROPOUT, "activation": LSTM_ACT,
                     "softmax": "applied in evaluation.predict_proba",
                     "optimizer": "adam", "lr": CLF_LR, "batch": CLF_BATCH,
                     "epochs": CLF_EPOCHS, "early_stopping": False},
        },
        "full": result["full"], "duplicate_free": result["duplicate_free"],
        "binary": result["binary"], "inference": result["inference"],
    }
    (place / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    if verbose:
        print("  [DONE]  %s  macro-F1 %.4f in %.1f s"
              % (LABEL, result["full"]["f1_macro"], training_seconds))
    return {"metrics": metrics, "result": result, "model": model,
            "folder": str(place), "status": "written"}


def run(verbose: bool = True) -> dict[str, Any]:
    return cp.run_one_paper(NAME, train, verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
