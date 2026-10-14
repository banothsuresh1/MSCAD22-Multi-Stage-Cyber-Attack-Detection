"""
tests/test_experiments.py -- pytest tests for STEP 12 (the model comparison).

NEW FILE.  No existing test file is modified.  Tiny synthetic data, 2-3 epochs,
CPU only, everything inside tmp_path.  Never C:\\MSCAD2022.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pytest tests -v
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import experiments as xp

REAL_STEP12_DIR = config.STEP12_OUTPUT_DIR

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")

N_ALL = 9          # "all" features
N_SELECTED = 5     # the Step 6 selection
N_ROWS = 120
CLASSES = ["Normal", "Port_Scan", "Brute_Force"]


@pytest.fixture(autouse=True)
def _tiny_and_in_tmp(tmp_path, monkeypatch):
    for step in (4, 5, 6, 7, 8, 9, 10, 11, 12):
        monkeypatch.setattr(config, "STEP%d_OUTPUT_DIR" % step,
                            tmp_path / ("step%d" % step))
    monkeypatch.setattr(config, "AE_LATENT_DIM", 3)
    monkeypatch.setattr(config, "AE_HIDDEN", [4])
    monkeypatch.setattr(config, "AE_EPOCHS", 2)
    monkeypatch.setattr(config, "AE_BATCH_SIZE", 16)
    monkeypatch.setattr(config, "AE_PATIENCE", 2)
    monkeypatch.setattr(config, "CNN_FILTERS", [8, 8])
    monkeypatch.setattr(config, "LSTM_UNITS", 8)
    monkeypatch.setattr(config, "DENSE_UNITS", 8)
    monkeypatch.setattr(config, "CLF_EPOCHS", 2)
    monkeypatch.setattr(config, "CLF_BATCH_SIZE", 16)
    monkeypatch.setattr(config, "CLF_PATIENCE", 2)
    monkeypatch.setattr(config, "CLF_VAL_SIZE", 0.2)
    monkeypatch.setattr(config, "N_SEEDS", 1)
    monkeypatch.setattr(config, "INPUT_MODE", "feature_seq")


def names(n):
    return ["f%02d" % i for i in range(n)]


def make_frames(n_rows=N_ROWS, seed=0):
    rng = np.random.default_rng(seed)
    codes = np.resize([0, 1, 2], n_rows)
    X_all = pd.DataFrame(rng.random((n_rows, N_ALL)), columns=names(N_ALL))
    y = pd.DataFrame({"Label": [CLASSES[c] for c in codes], "Label_encoded": codes,
                      "Label_binary": (codes > 0).astype("int64")})
    meta = pd.DataFrame({"Orig_Row": np.arange(n_rows),
                         "Source_Capture": "N-0",
                         "Timestamp": pd.date_range("2020-01-01", periods=n_rows,
                                                    freq="s").astype(str)})
    return X_all, y, meta


def write_inputs(X_train=None, y_train=None, X_test=None, y_test=None):
    """Lay down the Step 4/5/6/8 files Step 12 reads."""
    if X_train is None:
        X_train, y_train, meta_train = make_frames(seed=0)
    else:
        _a, _b, meta_train = make_frames(len(X_train), seed=0)
    if X_test is None:
        X_test, y_test, meta_test = make_frames(45, seed=1)
    else:
        _a, _b, meta_test = make_frames(len(X_test), seed=1)

    (config.STEP4_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (config.STEP4_OUTPUT_DIR / config.STEP4_FILES["label_mapping"]).write_text(
        json.dumps({name: i for i, name in enumerate(CLASSES)}), encoding="utf-8")

    folder5 = config.STEP5_OUTPUT_DIR
    folder5.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(folder5 / config.STEP5_FILES["X_train"], index=False)
    X_test.to_csv(folder5 / config.STEP5_FILES["X_test"], index=False)
    meta_train.to_csv(folder5 / config.STEP5_FILES["meta_train"], index=False)
    meta_test.to_csv(folder5 / config.STEP5_FILES["meta_test"], index=False)

    folder6 = config.STEP6_OUTPUT_DIR
    folder6.mkdir(parents=True, exist_ok=True)
    picked = names(N_ALL)[:N_SELECTED]
    X_train[picked].to_csv(folder6 / config.STEP6_FILES["X_train_selected"],
                           index=False)
    X_test[picked].to_csv(folder6 / config.STEP6_FILES["X_test_selected"],
                          index=False)
    (folder6 / config.STEP6_FILES["selected"]).write_text(
        json.dumps({"features": picked, "n_selected": N_SELECTED}), encoding="utf-8")

    # Step 8 holds the labels Step 12 reads, and Z_test/y_test for the fingerprint
    folder8 = config.STEP8_OUTPUT_DIR
    folder8.mkdir(parents=True, exist_ok=True)
    y_train.to_csv(folder8 / config.STEP8_FILES["y_train"], index=False)
    y_test.to_csv(folder8 / config.STEP8_FILES["y_test"], index=False)
    X_train.iloc[:, :3].to_csv(folder8 / config.STEP8_FILES["Z_train"], index=False)
    X_test.iloc[:, :3].to_csv(folder8 / config.STEP8_FILES["Z_test"], index=False)
    return X_train, y_train, X_test, y_test, meta_test


SPECS = [
    {"name": "proposed", "label": "Proposed", "feature_source": "selected30",
     "use_ae": True, "model_type": "cnn_lstm", "reuse_step7_10": False},
    {"name": "b1", "label": "B1", "feature_source": "all", "use_ae": True,
     "model_type": "cnn_lstm", "reuse_step7_10": False},
    {"name": "b2", "label": "B2", "feature_source": "all", "use_ae": True,
     "model_type": "lstm", "reuse_step7_10": False},
    {"name": "b3", "label": "B3", "feature_source": "all", "use_ae": False,
     "model_type": "cnn_lstm", "reuse_step7_10": False},
]


# ===========================================================================
# 1. the LSTM-only builder
# ===========================================================================
def test_the_lstm_only_model_has_no_conv_layers():
    model = xp.build_lstm_only((5, 1), n_classes=3)
    kinds = [type(layer).__name__ for layer in model.layers]
    assert "LSTM" in kinds
    assert not any(k.startswith("Conv") for k in kinds), "Baseline 2 has no CNN"
    assert not any("Pool" in k for k in kinds)
    assert model.output_shape[-1] == 3


def test_the_lstm_only_model_outputs_logits_like_step_10():
    model = xp.build_lstm_only((5, 1), n_classes=3)
    assert model.layers[-1].get_config().get("activation") == "linear"
    loss = model.loss
    assert isinstance(loss, keras.losses.SparseCategoricalCrossentropy)
    assert loss.get_config()["from_logits"] is True

    X = np.random.default_rng(0).random((8, 5, 1)).astype("float32") * 4
    from src.mscad import models as md
    logits = md.predict_logits(model, X)
    assert not np.allclose(logits.sum(axis=1), 1.0)
    assert np.allclose(md.predict_proba(model, X).sum(axis=1), 1.0)


def test_the_builder_lookup_rejects_an_unknown_model_type():
    with pytest.raises(xp.ExperimentError):
        xp.builder_for("transformer")


def test_lstm_only_uses_the_same_hyperparameters_as_the_cnn_lstm():
    from src.mscad import models as md
    lstm = xp.build_lstm_only((5, 1), 3)
    cnn = md.build_cnn_lstm((5, 1), 3)
    lstm_layer = [l for l in lstm.layers if type(l).__name__ == "LSTM"][0]
    cnn_lstm_layer = [l for l in cnn.layers if type(l).__name__ == "LSTM"][0]
    assert lstm_layer.get_config()["units"] == cnn_lstm_layer.get_config()["units"]
    dense = [l for l in lstm.layers if l.name == "dense"][0]
    assert dense.get_config()["units"] == config.DENSE_UNITS


# ===========================================================================
# 2. input shapes per configuration
# ===========================================================================
def test_input_shapes_are_right_for_every_configuration():
    write_inputs()
    selected = xp.load_feature_source("selected30")
    every = xp.load_feature_source("all")

    assert selected["n_features"] == N_SELECTED
    assert every["n_features"] == N_ALL

    from src.mscad import models as md
    # with the AE: (latent, 1).  Without: (n_features, 1).
    assert md.as_feature_sequence(selected["X_train"]).shape[1:] == (N_SELECTED, 1)
    assert md.as_feature_sequence(every["X_train"]).shape[1:] == (N_ALL, 1)


def test_use_ae_false_skips_the_encoder_entirely():
    X_train, y_train, X_test, y_test, meta_test = write_inputs()
    spec = dict(SPECS[3])                      # B3: use_ae False
    result = xp.run_experiment(spec, y_train, y_test, meta_test, {}, None,
                               overwrite=True, verbose=False)

    folder = xp.experiment_folder("b3")
    assert not (folder / "encoder.keras").exists(), "B3 must not train an AE"
    metrics = result["metrics"]
    assert metrics["latent_dim"] is None
    assert metrics["n_input_features"] == N_ALL
    assert list(metrics["model_input_shape"]) == [N_ALL, 1]


def test_use_ae_true_compresses_to_the_latent_dim():
    X_train, y_train, X_test, y_test, meta_test = write_inputs()
    result = xp.run_experiment(dict(SPECS[1]), y_train, y_test, meta_test, {}, None,
                               overwrite=True, verbose=False)

    folder = xp.experiment_folder("b1")
    assert (folder / "encoder.keras").exists()
    metrics = result["metrics"]
    assert metrics["latent_dim"] == config.AE_LATENT_DIM
    assert list(metrics["model_input_shape"]) == [config.AE_LATENT_DIM, 1]


# ===========================================================================
# 3. the baseline autoencoder sees the training rows only
# ===========================================================================
def test_the_baseline_autoencoder_is_trained_on_train_only(tmp_path):
    X_train, y_train, X_test, y_test, _meta = write_inputs()

    out = tmp_path / "ae_a"
    Z_train_a, Z_test_a, info_a = xp.encode_with_new_ae(X_train, X_test, y_train,
                                                        out, verbose=False)
    rng = np.random.default_rng(99)
    wrecked = pd.DataFrame(rng.normal(50, 20, X_test.shape), columns=X_test.columns)
    Z_train_b, _zt, info_b = xp.encode_with_new_ae(X_train, wrecked, y_train,
                                                   tmp_path / "ae_b", verbose=False)

    assert np.allclose(Z_train_a.to_numpy(), Z_train_b.to_numpy()), \
        "the test rows must not influence the encoder"
    assert info_a["train_mse"] == pytest.approx(info_b["train_mse"])
    assert Z_train_a.shape == (len(X_train), config.AE_LATENT_DIM)
    assert Z_test_a.shape == (len(X_test), config.AE_LATENT_DIM)


# ===========================================================================
# 4. every configuration is comparable
# ===========================================================================
def test_all_configurations_share_the_split_indices_and_class_weights():
    X_train, y_train, X_test, y_test, _meta = write_inputs()
    from src.mscad import imbalance as im

    marker = pd.DataFrame({"row": np.arange(len(y_train))})
    runs = [im.make_training_sets(marker.copy(), y_train, verbose=False)
            for _ in range(4)]

    first_fit = runs[0][0]["row"].tolist()
    first_val = runs[0][2]["row"].tolist()
    for fit, _yf, val, _yv, weights, _sw in runs:
        assert fit["row"].tolist() == first_fit
        assert val["row"].tolist() == first_val
        assert weights == runs[0][4]
    assert set(first_fit) & set(first_val) == set()
    assert len(first_fit) + len(first_val) == len(y_train)


def test_the_consistency_checks_pass_for_the_real_sources():
    X_train, y_train, X_test, y_test, _meta = write_inputs()
    sources = {"a": xp.load_feature_source("selected30"),
               "b": xp.load_feature_source("all")}
    before = xp.test_fingerprint()

    table, ok = xp.consistency_checks(sources, y_train, y_test, before,
                                      verbose=False)

    assert ok is True
    assert "[FAIL]" not in set(table["result"])
    assert any("byte-for-byte" in c for c in table["check"])


def test_the_test_files_are_unchanged_by_the_whole_comparison():
    X_train, y_train, X_test, y_test, meta_test = write_inputs()
    watched = [config.STEP5_OUTPUT_DIR / config.STEP5_FILES["X_test"],
               config.STEP8_OUTPUT_DIR / config.STEP8_FILES["y_test"]]
    before = {p.name: p.read_bytes() for p in watched}

    for spec in SPECS[:2]:
        xp.run_experiment(dict(spec), y_train, y_test, meta_test, {}, None,
                          overwrite=True, verbose=False)

    for path in watched:
        assert path.read_bytes() == before[path.name], path.name


def test_label_mapping_is_shared_by_every_configuration():
    write_inputs()
    from src.mscad import evaluation as ev
    loaded = ev.load_evaluation_inputs(verbose=False)
    names_map = ev.label_names_from_mapping(loaded["label_mapping"])
    assert names_map == {0: "Normal", 1: "Port_Scan", 2: "Brute_Force"}


# ===========================================================================
# 5. the comparison table
# ===========================================================================
def test_the_comparison_table_has_one_row_per_configuration_and_every_column():
    X_train, y_train, X_test, y_test, meta_test = write_inputs()
    runs = [xp.run_experiment(dict(spec), y_train, y_test, meta_test, {}, None,
                              overwrite=True, verbose=False)
            for spec in SPECS]

    table = xp.comparison_table(runs)

    assert len(table) == len(SPECS)
    for column in ("config", "input features", "latent dim", "params", "accuracy",
                   "precision (macro)", "recall (macro)", "F1 (macro)",
                   "F1 (weighted)", "G-mean", "ROC-AUC (macro OvR)", "binary F1",
                   "binary FPR", "F1 macro (dup-free)", "G-mean (dup-free)",
                   "train time (s)", "inference (ms/1k)"):
        assert column in table.columns, column
    assert set(table["config"]) == {s["label"] for s in SPECS}


def test_mark_best_marks_one_winner_and_skips_a_tie():
    table = pd.DataFrame({"config": ["A", "B", "C"],
                          "F1 (macro)": [0.1, 0.3, 0.2],
                          "G-mean": [0.0, 0.0, 0.0],
                          "binary FPR": [0.9, 0.1, 0.5]})
    marked = xp.mark_best(table)

    assert marked["F1 (macro)"].tolist() == ["0.1000", "0.3000*", "0.2000"]
    # lower is better for the FPR
    assert marked["binary FPR"].tolist() == ["0.9000", "0.1000*", "0.5000"]
    # a three-way tie has no winner
    assert all(not v.endswith("*") for v in marked["G-mean"])


def test_the_summary_is_written_from_the_computed_numbers():
    table = pd.DataFrame({
        "config": ["Proposed: x", "B1: y"], "input features": [30, 70],
        "latent dim": [15, 15], "params": [100, 100],
        "F1 (macro)": [0.80, 0.60], "G-mean": [0.5, 0.4],
        "F1 macro (dup-free)": [0.70, 0.50],
        "ROC-AUC (macro OvR)": [0.9, 0.8], "binary FPR": [0.1, 0.2],
        "inference (ms/1k)": [10.0, 20.0]})
    environment = {"python": "3.13", "tensorflow": "2.21", "keras": "3.15",
                   "numpy": "2", "pandas": "2", "scikit-learn": "1",
                   "devices": ["CPU"], "input_mode": "feature_seq", "n_seeds": 1,
                   "imbalance": {"strategy": "class_weight", "scheme": "balanced"}}

    summary = xp.build_summary(table, pd.DataFrame(), environment)

    assert "0.8000" in summary and "Proposed: x" in summary
    assert "+0.2000" in summary, "the margin must come from the numbers"
    assert "duplicate-free" in summary.lower()
    assert "significance" in summary.lower(), "the no-significance caveat"


def test_seed_comparison_reports_differences_only():
    per_seed = {"proposed": [0.80, 0.84], "b1": [0.70, 0.72]}
    table = xp.seed_comparison(per_seed)

    proposed = table[table["config"] == "proposed"].iloc[0]
    assert proposed["macro-F1 mean"] == pytest.approx(0.82)
    other = table[table["config"] == "b1"].iloc[0]
    assert other["proposed - this (mean)"] == pytest.approx(0.11)
    assert "p" not in [c.lower() for c in table.columns], "no p-values are claimed"


# ===========================================================================
# 6. resuming
# ===========================================================================
def test_a_finished_configuration_is_skipped_when_overwrite_is_off():
    X_train, y_train, X_test, y_test, meta_test = write_inputs()
    spec = dict(SPECS[3])

    first = xp.run_experiment(spec, y_train, y_test, meta_test, {}, None,
                              overwrite=True, verbose=False)
    assert first["status"] == "written"

    again = xp.run_experiment(spec, y_train, y_test, meta_test, {}, None,
                              overwrite=False, verbose=False)
    assert again["status"] == "loaded_existing"
    assert again["metrics"]["full"]["f1_macro"] == \
        pytest.approx(first["metrics"]["full"]["f1_macro"])


def test_experiment_is_done_reports_what_is_on_disk():
    write_inputs()
    assert xp.experiment_is_done("never_run") is False


# ===========================================================================
# 7. the Windows-path guard
# ===========================================================================
def test_step12_refuses_a_windows_path_off_windows(tmp_path, monkeypatch, capsys,
                                                   assert_no_leaked_output):
    """On Linux, Path(r"C:\\...") is RELATIVE -- nothing may be created."""
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR",
                        config.Path(r"C:\MSCAD2022\outputs\step12"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    assert xp.prepare_output_dir(config.STEP12_OUTPUT_DIR, "the Step 12 outputs") \
        is None
    assert xp.experiment_folder("proposed") is None
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("prepare_output_dir for the Step 12 folder")


def test_step12_output_dir_is_under_data_root_not_the_repo():
    assert REAL_STEP12_DIR.parent.parent == config.DATA_ROOT
    assert config.PROJECT_ROOT not in REAL_STEP12_DIR.parents
    assert REAL_STEP12_DIR.name == "step12"
