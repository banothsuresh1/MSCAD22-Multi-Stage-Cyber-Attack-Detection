r"""
tests/step12/conftest.py -- fixtures for the eight configuration files.

NEW FILE.  Everything these tests touch is inside tmp_path: the Step 12
folders are redirected through `config`, never through pathlib, and no test
writes to C:\\MSCAD2022 or trains a real model unless it says so and uses two
epochs on a handful of synthetic rows.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp

LABELS = {0: "Normal", 1: "Port_Scan", 2: "Brute_Force"}
KEYS = [key for key, _label in config.STEP12_DISPLAY_ORDER]


@pytest.fixture(autouse=True)
def step12_dirs(tmp_path, monkeypatch):
    """Every folder the configuration files read or write, inside tmp_path."""
    for name, leaf in (("DATA_ROOT", "data"),
                       ("OUTPUTS_DIR", "outputs"),
                       ("STEP8_OUTPUT_DIR", "step8"),
                       ("STEP10_OUTPUT_DIR", "step10"),
                       ("STEP10B_OUTPUT_DIR", "step10_attention"),
                       ("STEP11_OUTPUT_DIR", "step11"),
                       ("STEP11B_OUTPUT_DIR", "step11_attention"),
                       ("STEP12_OUTPUT_DIR", "step12"),
                       ("STEP12_CONFIG_DIR", "step12"),
                       ("STEP12_B6_OUTPUT_DIR", "step12_baseline6"),
                       ("B6_OUTPUT_DIR", "step12_baseline6"),
                       ("STEP12_SANITY_OUTPUT_DIR", "step12_sanity"),
                       ("STEP12_SEEDS_OUTPUT_DIR", "step12_seeds"),
                       ("STEP12_THESIS_DIR", "step12/thesis"),
                       ("STEP13_OUTPUT_DIR", "step13"),
                       ("STEP13ALL_OUTPUT_DIR", "step13_all")):
        monkeypatch.setattr(config, name, tmp_path / leaf, raising=False)
    monkeypatch.setattr(config, "STEP12_EXPECTED_ROWS", {})
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True)
    return tmp_path


def metrics_payload(label, accuracy=0.9, f1=0.5, g_mean=0.4, auc=0.9,
                    fpr=0.02, params=1000, train=10.0, pca=None, seed=None):
    out = {
        "config": label, "label": label, "model": label,
        "n_input_features": 30, "latent_dim": 15, "parameters": params,
        "training_seconds": train, "best_epoch": 4,
        "seed": config.RANDOM_STATE if seed is None else seed,
        "full": {"accuracy": accuracy, "precision_macro": 0.6,
                 "recall_macro": 0.55, "f1_macro": f1, "f1_weighted": 0.8,
                 "g_mean": g_mean, "roc_auc": {"macro": auc}},
        "duplicate_free": {"f1_macro": f1, "g_mean": g_mean,
                           "accuracy": accuracy},
        "binary": {"f1": 0.95, "false_positive_rate": fpr},
        "inference": {"ms_per_1000_flows": 50.0},
    }
    if pca is not None:
        out["pca_components"] = pca
        out["pca"] = {"variance_kept": 0.912}
    return out


def predictions_frame(rows=12, seed=0, benign_probability=0.7):
    rng = np.random.default_rng(seed)
    y_true = np.array([0, 1, 2] * (rows // 3))
    y_pred = y_true.copy()
    y_pred[0] = 1
    attack = 1.0 - benign_probability
    frame = pd.DataFrame({
        "Orig_Row": np.arange(len(y_true)),
        "y_true": y_true, "y_pred": y_pred,
        "y_true_label": [LABELS[int(c)] for c in y_true],
        "y_pred_label": [LABELS[int(c)] for c in y_pred],
        "has_train_duplicate": np.zeros(len(y_true), dtype=bool),
        "p_Normal": benign_probability,
        "p_Port_Scan": attack / 2,
        "p_Brute_Force": attack / 2,
    })
    frame["noise"] = rng.random(len(y_true))
    return frame


def confusion_frame():
    names = [LABELS[code] for code in sorted(LABELS)]
    return pd.DataFrame(np.eye(len(names)) * 0.9 + 0.05,
                        index=names, columns=names)


def write_result(key, payload=None, predictions=None, model=True,
                 confusion=True):
    """One configuration's saved folder, as its finished run left it."""
    from pathlib import Path

    folder = Path(cmp.result_path(key)).parent
    folder.mkdir(parents=True, exist_ok=True)
    label = dict(config.STEP12_DISPLAY_ORDER)[key]
    payload = metrics_payload(label) if payload is None else payload
    (folder / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    (predictions if predictions is not None
     else predictions_frame()).to_csv(folder / "predictions.csv", index=False)
    if model:
        (folder / "model.keras").write_bytes(b"not a real model, just bytes")
    if confusion:
        confusion_frame().to_csv(folder / "confusion_matrix_rownorm.csv")
    return folder


@pytest.fixture(autouse=True)
def step8_labels(step12_dirs):
    """A small Step 8 y_train.csv, so the Step 9 split check can actually run.

    `common.protocol_checks` asks `imbalance.make_training_sets` for the split
    twice and compares the rows; without labels on disk it reports [FAIL]
    ("not readable here"), which is correct but would make every run() test
    fail for a reason that has nothing to do with the file under test.
    """
    folder = step12_dirs / "step8"
    folder.mkdir(parents=True, exist_ok=True)
    codes = np.array([0] * 60 + [1] * 25 + [2] * 15)
    pd.DataFrame({"Label_encoded": codes,
                  "Label_binary": (codes != 0).astype("int64")}).to_csv(
        folder / config.STEP8_FILES["y_train"], index=False)
    return folder


@pytest.fixture
def finished(step12_dirs):
    """Write a saved result for one or more keys, and return the writer."""
    def write(*keys, **overrides):
        folders = {}
        for i, key in enumerate(keys or KEYS):
            label = dict(config.STEP12_DISPLAY_ORDER)[key]
            payload = overrides.get(key) or metrics_payload(
                label, f1=0.50 + 0.02 * i, g_mean=0.40 + 0.02 * i,
                accuracy=0.90 + 0.001 * i, params=1000 + 10 * i,
                train=10.0 + i,
                pca=2 if key in ("baseline4_ae_pca_lstm",
                                 config.BASELINE6_NAME) else None)
            folders[key] = write_result(key, payload,
                                        predictions_frame(seed=i))
        return folders
    return write


@pytest.fixture
def all_finished(finished):
    """All eight configurations already saved -- the usual state of Step 12."""
    return finished(*KEYS)
