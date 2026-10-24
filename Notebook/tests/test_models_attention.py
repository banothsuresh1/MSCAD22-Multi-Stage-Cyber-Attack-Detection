r"""
tests/test_models_attention.py -- pytest tests for Approach 2
(hybrid CNN-Attention-LSTM).

NEW FILE.  No existing test file is modified, and Approach 1's code and saved
outputs are never touched.  Tiny synthetic data, 2-3 epochs, CPU only,
everything inside tmp_path.  Never C:\\MSCAD2022.

No pathlib patching: the other operating system is simulated through
`config.running_on_windows`, and Windows paths are built with PureWindowsPath.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import json
import os
from pathlib import PureWindowsPath

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp
from src.mscad import models as md
from src.mscad import models_attention as ma

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")

N_LATENT = 8
N_ROWS = 60
N_CLASSES = 3


@pytest.fixture(autouse=True)
def _tiny_and_in_tmp(tmp_path, monkeypatch):
    for step in (5, 8, 9, 10, 11, 12):
        monkeypatch.setattr(config, "STEP%d_OUTPUT_DIR" % step,
                            tmp_path / ("step%d" % step))
    monkeypatch.setattr(config, "STEP10B_OUTPUT_DIR",
                        tmp_path / "step10_attention")
    monkeypatch.setattr(config, "STEP11B_OUTPUT_DIR",
                        tmp_path / "step11_attention")
    monkeypatch.setattr(config, "CNN_FILTERS", [8, 8])
    monkeypatch.setattr(config, "LSTM_UNITS", 8)
    monkeypatch.setattr(config, "ATTN_UNITS", 8)
    monkeypatch.setattr(config, "DENSE_UNITS", 8)
    monkeypatch.setattr(config, "CLF_EPOCHS", 2)
    monkeypatch.setattr(config, "CLF_BATCH_SIZE", 16)
    monkeypatch.setattr(config, "CLF_PATIENCE", 2)
    monkeypatch.setattr(config, "CLF_VAL_SIZE", 0.2)
    monkeypatch.setattr(config, "INPUT_MODE", "feature_seq")
    monkeypatch.setattr(config, "N_SEEDS", 1)


def make_xy(n_rows=N_ROWS, n_latent=N_LATENT, seed=0):
    rng = np.random.default_rng(seed)
    codes = np.resize(list(range(N_CLASSES)), n_rows)
    values = rng.uniform(0, 1, size=(n_rows, n_latent))
    values[:, 0] = codes * 0.3 + rng.normal(0, 0.02, n_rows)
    Z = pd.DataFrame(values, columns=["z%02d" % i for i in range(n_latent)])
    y = pd.DataFrame({"Label_encoded": codes,
                      "Label_binary": (codes != 0).astype("int64")})
    return Z, y


def built_model(n_latent=N_LATENT):
    keras.utils.set_random_seed(0)
    return ma.build_cnn_attention_lstm((n_latent, 1), N_CLASSES)


# ===========================================================================
# the attention weights
# ===========================================================================
def test_attention_weights_are_a_distribution():
    model = built_model()
    X = np.random.default_rng(1).uniform(0, 1, (12, N_LATENT, 1)).astype("float32")

    alpha = ma.attention_weights(model, X)

    assert alpha.shape[0] == 12
    assert (alpha >= 0).all()
    assert alpha.sum(axis=1) == pytest.approx(np.ones(12), abs=1e-9)


def test_the_numpy_weights_match_the_layers_own_computation():
    """What the heatmap shows must be what the layer computed.

    TensorFlow rather than `keras`'s own numerical namespace, which exists
    only in Keras 3; the layer itself is written in TensorFlow ops, so this
    asks it in the language it speaks on either version.
    """
    tf = pytest.importorskip("tensorflow")

    model = built_model()
    X = np.random.default_rng(2).uniform(0, 1, (6, N_LATENT, 1)).astype("float32")
    layer = ma.attention_layer(model)
    sequence = keras.Model(model.inputs, layer.input)

    theirs = np.asarray(
        layer.scores(tf.convert_to_tensor(sequence.predict(X, verbose=0))),
        dtype="float64")
    ours = ma.attention_weights(model, X)

    assert theirs.shape == ours.shape
    assert ours == pytest.approx(theirs, abs=1e-6)


def test_the_attention_positions_are_the_pooled_feature_map():
    model = built_model(n_latent=N_LATENT)
    alpha = ma.attention_weights(
        model, np.zeros((2, N_LATENT, 1), dtype="float32"))

    assert alpha.shape[1] == N_LATENT // config.CNN_POOL
    assert ma.attention_layer(model).input.shape[1] == alpha.shape[1]


def test_the_attention_summary_reports_entropy_against_uniform():
    positions = 4
    uniform = np.full((10, positions), 1.0 / positions)
    peaked = np.zeros((10, positions))
    peaked[:, 1] = 1.0

    flat, _per_class = ma.attention_summary(uniform, np.zeros(10, dtype=int),
                                            {0: "Normal"})
    sharp, _p2 = ma.attention_summary(peaked, np.zeros(10, dtype=int),
                                      {0: "Normal"})

    assert flat.attrs["entropy"]["normalised_mean"] == pytest.approx(1.0)
    assert sharp.attrs["entropy"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert flat.attrs["entropy"]["uniform_would_be"] == pytest.approx(
        np.log(positions))


def test_the_per_class_attention_has_one_row_per_predicted_class():
    alpha = np.full((9, 3), 1.0 / 3)
    predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    _overall, per_class = ma.attention_summary(
        alpha, predicted, {0: "Normal", 1: "Port_Scan", 2: "Brute_Force"})

    assert len(per_class) == 3
    assert list(per_class["predicted"]) == ["Normal", "Port_Scan",
                                            "Brute_Force"]
    assert list(per_class["flows"]) == [3, 3, 3]


# ===========================================================================
# the model
# ===========================================================================
def test_the_model_outputs_logits_of_the_right_shape():
    model = built_model()
    X = np.random.default_rng(3).uniform(0, 1, (5, N_LATENT, 1)).astype("float32")

    logits = md.predict_logits(model, X)
    probabilities = md.predict_proba(model, X)

    assert logits.shape == (5, N_CLASSES)
    # logits, not probabilities: the rows do not sum to 1
    assert not np.allclose(logits.sum(axis=1), 1.0)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(5), abs=1e-6)
    assert model.layers[-1].get_config()["activation"] == "linear"


def test_the_architecture_differs_from_approach_1_only_by_attention():
    attention = built_model()
    keras.utils.set_random_seed(0)
    plain = md.build_cnn_lstm((N_LATENT, 1), N_CLASSES)

    kinds_a = [type(l).__name__ for l in attention.layers]
    kinds_p = [type(l).__name__ for l in plain.layers]

    assert "AdditiveAttention" in kinds_a
    assert "AdditiveAttention" not in kinds_p
    # the CNN front end is the same set of layer types
    assert [k for k in kinds_p if k in ("Conv1D", "BatchNormalization",
                                        "MaxPooling1D", "LSTM", "Dense")] == \
        [k for k in kinds_a if k in ("Conv1D", "BatchNormalization",
                                     "MaxPooling1D", "LSTM", "Dense")]
    assert attention.count_params() > plain.count_params()


def test_the_saved_model_reloads_with_identical_outputs(tmp_path):
    """The layer is registered, so no custom_objects are needed."""
    model = built_model()
    X = np.random.default_rng(4).uniform(0, 1, (7, N_LATENT, 1)).astype("float32")
    before = md.predict_proba(model, X)

    path = tmp_path / "approach2.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)

    assert md.predict_proba(reloaded, X) == pytest.approx(before, abs=1e-6)
    assert ma.attention_weights(reloaded, X) == pytest.approx(
        ma.attention_weights(model, X), abs=1e-6)


def test_the_layer_round_trips_through_its_config():
    layer = ma._attention_class()(units=11, name="additive_attention")
    clone = type(layer).from_config(layer.get_config())

    assert clone.units == 11
    assert clone.name == "additive_attention"


def test_a_model_without_attention_is_refused():
    with pytest.raises(ma.AttentionError, match="additive_attention"):
        ma.attention_layer(md.build_cnn_lstm((N_LATENT, 1), N_CLASSES))


# ===========================================================================
# the training protocol
# ===========================================================================
def test_both_approaches_get_the_same_split_weights_and_seed():
    from src.mscad import imbalance as im

    Z, y = make_xy()
    _model, _history, info = ma.train_attention(Z, y, verbose=False)

    _fit, _yf, X_val, _yv, class_weights, _sw = im.make_training_sets(
        Z, y, verbose=False)
    assert info["validation_rows"] == len(X_val)
    assert info["class_weights"] == {int(k): float(v)
                                     for k, v in class_weights.items()}
    assert info["seed"] == config.RANDOM_STATE
    assert info["model_type"] == "cnn_attention_lstm"
    assert info["input_shape"] == (N_LATENT, 1)


def test_the_protocol_checks_pass_on_a_clean_run():
    Z, y = make_xy()
    _model, _history, info = ma.train_attention(Z, y, verbose=False)

    table, ok = ma.protocol_checks(info, verbose=False)

    assert ok is True
    assert (table["result"] == "[OK]").all()
    assert any("logits" in c for c in table["check"])


def test_the_protocol_checks_catch_a_different_seed():
    Z, y = make_xy()
    _model, _history, info = ma.train_attention(Z, y, seed=1234, verbose=False)

    table, ok = ma.protocol_checks(info, verbose=False)

    assert ok is False
    failed = table[table["result"] == "[FAIL]"]["check"].tolist()
    assert any("seed" in c for c in failed)


def test_approach_1_outputs_are_untouched_by_a_run(tmp_path):
    """Its metrics.json is read, never written."""
    folder = tmp_path / "step11"
    folder.mkdir(parents=True)
    payload = {"full": {"f1_macro": 0.7170, "accuracy": 0.9896},
               "duplicate_free": {"f1_macro": 0.6}, "binary": {"f1": 0.9}}
    path = folder / config.STEP11_FILES["metrics"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()

    loaded = ma.approach1_metrics()
    Z, y = make_xy()
    ma.train_attention(Z, y, verbose=False)

    assert loaded["full"]["f1_macro"] == pytest.approx(0.7170)
    assert path.read_bytes() == before
    check = ma.assert_approach1_unchanged(loaded, verbose=False)
    assert check["result"] == "[OK]"


def test_a_drifted_approach_1_metric_is_caught(tmp_path):
    folder = tmp_path / "step11"
    folder.mkdir(parents=True)
    path = folder / config.STEP11_FILES["metrics"]
    path.write_text(json.dumps({"full": {"f1_macro": 0.7170}}),
                    encoding="utf-8")
    before = ma.approach1_metrics()
    path.write_text(json.dumps({"full": {"f1_macro": 0.5000}}),
                    encoding="utf-8")

    check = ma.assert_approach1_unchanged(before, verbose=False)

    assert check["result"] == "[FAIL]"
    assert "full/f1_macro" in check["detail"]


# ===========================================================================
# selection on validation only
# ===========================================================================
def test_the_primary_approach_is_chosen_on_validation_macro_f1():
    better = ma.select_primary_approach(0.80, approach1_validation=0.70)
    worse = ma.select_primary_approach(0.60, approach1_validation=0.70)
    tied = ma.select_primary_approach(0.70, approach1_validation=0.70)

    assert better["primary"] == "Approach 2"
    assert worse["primary"] == "Approach 1"
    assert tied["primary"] == "Approach 1"          # the incumbent keeps a tie
    for out in (better, worse, tied):
        assert out["test_set_used"] is False
        assert "validation" in out["chosen_on"]


def test_the_incumbent_is_kept_when_approach_1s_validation_score_is_missing(
        tmp_path):
    out = ma.select_primary_approach(0.99)

    assert out["primary"] == "Approach 1"
    assert out["approach1_validation_macro_f1"] is None
    assert "not on this machine" in out["reason"]


def test_the_selection_never_reads_a_test_metric(tmp_path):
    """Writing a spectacular test score must not change the choice."""
    folder = tmp_path / "step11"
    folder.mkdir(parents=True)
    (folder / config.STEP11_FILES["metrics"]).write_text(
        json.dumps({"full": {"f1_macro": 0.999}}), encoding="utf-8")

    out = ma.select_primary_approach(0.10, approach1_validation=0.20)

    assert out["primary"] == "Approach 1"
    assert out["approach1_validation_macro_f1"] == pytest.approx(0.20)


# ===========================================================================
# the Step 12 row
# ===========================================================================
def test_approach_2_is_row_2_with_and_without_its_outputs(tmp_path):
    def write(key, payload):
        folder = (tmp_path / "step11_attention" if key == config.APPROACH2_KEY
                  else tmp_path / "step12" / key)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metrics.json").write_text(json.dumps(payload),
                                             encoding="utf-8")

    payload = {"n_input_features": 30, "latent_dim": 15, "parameters": 1,
               "full": {"accuracy": 0.9, "f1_macro": 0.5, "g_mean": 0.4,
                        "precision_macro": 0.5, "recall_macro": 0.5,
                        "f1_weighted": 0.5, "roc_auc": {"macro": 0.9}},
               "duplicate_free": {"f1_macro": 0.5, "g_mean": 0.4},
               "binary": {"f1": 0.9, "false_positive_rate": 0.1},
               "inference": {"ms_per_1000_flows": 10.0},
               "training_seconds": 1.0}
    for key, _label in config.STEP12_DISPLAY_ORDER:
        if key != config.APPROACH2_KEY:
            write(key, payload)

    label = dict(config.STEP12_DISPLAY_ORDER)[config.APPROACH2_KEY]
    without = cmp.build_full_comparison(verbose=False)["table"]
    assert without.iloc[1]["config"] == label
    # the cell is the state plus the step that produces this row
    assert str(without.iloc[1]["input features"]).startswith(
        config.NOT_TRAINED_TEXT)

    write(config.APPROACH2_KEY, payload)
    with_it = cmp.build_full_comparison(verbose=False)["table"]
    assert with_it.iloc[1]["config"] == label
    assert with_it.iloc[1]["input features"] == 30
    assert list(with_it["config"]) == list(without["config"])


def test_the_approach_2_config_block_holds_the_documented_defaults():
    assert config.STEP10B_OUTPUT_DIR.name == "step10_attention"
    assert config.STEP11B_OUTPUT_DIR.name == "step11_attention"
    assert config.ATTN_UNITS >= 1
    assert isinstance(config.OVERWRITE_STEP10B, bool)
    # Approach 2 reuses Approach 1's hyperparameters rather than redefining them
    for name in ("CNN_FILTERS", "CNN_KERNEL", "CNN_POOL", "LSTM_UNITS",
                 "DROPOUT", "DENSE_UNITS", "CLF_EPOCHS", "CLF_BATCH_SIZE",
                 "CLF_LR", "CLF_PATIENCE", "EARLY_STOP_METRIC"):
        assert hasattr(config, name)
    assert config.APPROACH2_KEY == "approach2_attention"


# ===========================================================================
# the guards
# ===========================================================================
def test_the_runs_refuse_a_windows_path(tmp_path, monkeypatch, capsys,
                                        assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP10B_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step10_attention"))
    monkeypatch.setattr(config, "STEP11B_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step11_attention"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    assert ma.step10b_outputs_exist() is False
    assert ma.step11b_outputs_exist() is False
    with pytest.raises(ma.AttentionError, match="not usable here"):
        ma.load_step10b(verbose=False)
    capsys.readouterr()
    assert_no_leaked_output("the Approach 2 loaders")


def test_the_output_dirs_are_under_data_root_not_the_repo():
    for folder in (config.STEP10B_OUTPUT_DIR, config.STEP11B_OUTPUT_DIR):
        assert config.PROJECT_ROOT not in folder.parents


def test_the_layer_is_registered_before_a_saved_model_is_loaded(tmp_path):
    """The regression that only bites on the resumable path.

    The layer class is built lazily, so its `register_keras_serializable`
    decorator runs on first use.  A kernel that only LOADS a saved Approach 2
    model never builds one, and Keras would answer "Could not locate class
    'AdditiveAttention'".  `load_step10b` registers it first and passes
    `custom_objects`, and both names Keras may look it up under are covered.
    """
    objects = ma.custom_objects()

    assert set(objects) == {"AdditiveAttention", "mscad>AdditiveAttention"}
    assert objects["AdditiveAttention"] is objects["mscad>AdditiveAttention"]
    assert ma.register_attention() is objects["AdditiveAttention"]

    # a saved model loads through that path with no exception
    model = built_model()
    folder = tmp_path / "step10_attention"
    folder.mkdir(parents=True)
    model.save(folder / config.STEP10B_FILES["model"])
    (folder / config.STEP10B_FILES["model_card"]).write_text("{}",
                                                             encoding="utf-8")

    loaded = ma.load_step10b(verbose=False)
    X = np.random.default_rng(5).uniform(0, 1, (4, N_LATENT, 1)).astype("float32")
    assert md.predict_proba(loaded["model"], X) == pytest.approx(
        md.predict_proba(model, X), abs=1e-6)
