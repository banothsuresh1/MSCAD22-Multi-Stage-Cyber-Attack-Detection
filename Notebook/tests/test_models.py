"""
tests/test_models.py -- pytest tests for STEP 10 (the hybrid CNN-LSTM).

Tiny synthetic data, 2-3 epochs, CPU only, everything inside tmp_path.
No real data, and never C:\\MSCAD2022.

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
from src.mscad import models as md

REAL_STEP10_DIR = config.STEP10_OUTPUT_DIR

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")

N_LATENT = 6
CLASSES = ["Normal", "Port_Scan", "Brute_Force"]


@pytest.fixture(autouse=True)
def _tiny_and_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STEP4_OUTPUT_DIR", tmp_path / "step4")
    monkeypatch.setattr(config, "STEP5_OUTPUT_DIR", tmp_path / "step5")
    monkeypatch.setattr(config, "STEP8_OUTPUT_DIR", tmp_path / "step8")
    monkeypatch.setattr(config, "STEP9_OUTPUT_DIR", tmp_path / "step9")
    monkeypatch.setattr(config, "STEP10_OUTPUT_DIR", tmp_path / "step10")
    monkeypatch.setattr(config, "AE_LATENT_DIM", N_LATENT)
    monkeypatch.setattr(config, "CNN_FILTERS", [8, 8])
    monkeypatch.setattr(config, "LSTM_UNITS", 8)
    monkeypatch.setattr(config, "DENSE_UNITS", 8)
    monkeypatch.setattr(config, "CLF_EPOCHS", 2)
    monkeypatch.setattr(config, "CLF_BATCH_SIZE", 16)
    monkeypatch.setattr(config, "CLF_PATIENCE", 2)
    monkeypatch.setattr(config, "CLF_VAL_SIZE", 0.2)
    monkeypatch.setattr(config, "INPUT_MODE", "feature_seq")
    monkeypatch.setattr(config, "WINDOW_SIZE", 4)


def make_zy(n_rows: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    codes = np.resize([0, 1, 2], n_rows)
    Z = pd.DataFrame(rng.random((n_rows, N_LATENT)),
                     columns=["z%02d" % (i + 1) for i in range(N_LATENT)])
    y = pd.DataFrame({"Label": [CLASSES[c] for c in codes], "Label_encoded": codes,
                      "Label_binary": (codes > 0).astype("int64")})
    return Z, y


def make_meta(n_rows: int, captures=("N-0", "Scan-1"), seed: int = 0):
    """Step 5 metadata: a capture name and an increasing timestamp per row."""
    which = np.resize(list(captures), n_rows)
    stamps = pd.date_range("2020-01-01", periods=n_rows, freq="s").astype(str)
    return pd.DataFrame({"Orig_Row": np.arange(n_rows),
                         "Source_Capture": which, "Timestamp": stamps})


def write_step8(Z=None, y=None, Z_test=None, y_test=None):
    if Z is None:
        Z, y = make_zy()
    if Z_test is None:
        Z_test, y_test = make_zy(n_rows=45, seed=3)
    folder = config.STEP8_OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    names = config.STEP8_FILES
    Z.to_csv(folder / names["Z_train"], index=False)
    y.to_csv(folder / names["y_train"], index=False)
    Z_test.to_csv(folder / names["Z_test"], index=False)
    y_test.to_csv(folder / names["y_test"], index=False)
    return Z, y, Z_test, y_test


# ===========================================================================
# 1. the input arrangement
# ===========================================================================
def test_feature_seq_gives_one_channel_per_latent_feature():
    Z, y = make_zy(n_rows=20)
    X, labels, index = md.arrange_inputs(Z, y, mode="feature_seq")

    assert X.shape == (20, N_LATENT, 1)
    assert labels.tolist() == y["Label_encoded"].tolist()
    assert index.tolist() == list(range(20))
    assert np.allclose(X[:, :, 0], Z.to_numpy(dtype="float32"))


def test_time_windows_never_cross_a_capture_boundary():
    n = 40
    Z, y = make_zy(n_rows=n)
    # two captures, in alternating rows, so a careless implementation mixes them
    meta = make_meta(n, captures=("N-0", "Scan-1"))

    X, labels, ends = md.build_time_windows(Z, y, meta, window=4, verbose=False)

    capture = meta["Source_Capture"].to_numpy()
    rows = np.arange(n)
    for end in ends:
        # the window's rows are the 4 latest rows of that capture up to `end`
        same = rows[(capture == capture[end]) & (rows <= end)]
        assert len(same) >= 4
        chosen = same[-4:]
        assert chosen[-1] == end
        assert len({capture[r] for r in chosen}) == 1, "one capture per window"
    assert X.shape == (len(ends), 4, N_LATENT)


def test_time_windows_are_built_separately_on_each_side_of_the_split():
    """A window may never contain a row from the other side."""
    n_train, n_test = 40, 20
    Z_train, y_train = make_zy(n_rows=n_train, seed=0)
    Z_test, y_test = make_zy(n_rows=n_test, seed=1)
    meta_train = make_meta(n_train, captures=("N-0",))
    meta_test = make_meta(n_test, captures=("N-0",))

    X_train, _lt, ends_train = md.build_time_windows(Z_train, y_train, meta_train,
                                                     window=4, verbose=False)
    X_test, _le, ends_test = md.build_time_windows(Z_test, y_test, meta_test,
                                                   window=4, verbose=False)

    assert ends_train.max() < n_train
    assert ends_test.max() < n_test

    # Every training window is built from training values only.  Both sides are
    # cast to float32 first: build_time_windows casts, and comparing a float32
    # value against the float64 original can straddle a rounding boundary.
    train_values = {row.tobytes()
                    for row in np.ascontiguousarray(
                        Z_train.to_numpy(dtype="float32"))}
    test_values = {row.tobytes()
                   for row in np.ascontiguousarray(
                       Z_test.to_numpy(dtype="float32"))}
    for window in X_train:
        for row in np.ascontiguousarray(window):
            assert row.tobytes() in train_values
            assert row.tobytes() not in test_values, \
                "a training window must never contain a test row"
    for window in X_test:
        for row in np.ascontiguousarray(window):
            assert row.tobytes() in test_values


def test_a_window_longer_than_any_capture_is_refused():
    Z, y = make_zy(n_rows=10)
    meta = make_meta(10, captures=("N-0", "Scan-1"))
    with pytest.raises(md.ModelError) as err:
        md.build_time_windows(Z, y, meta, window=50, verbose=False)
    assert "window" in str(err.value).lower()


def test_time_window_mode_needs_the_metadata():
    Z, y = make_zy(n_rows=20)
    with pytest.raises(md.ModelError) as err:
        md.arrange_inputs(Z, y, meta=None, mode="time_window")
    assert "meta" in str(err.value)


def test_an_unknown_input_mode_is_refused():
    Z, y = make_zy(n_rows=20)
    with pytest.raises(md.ModelError) as err:
        md.arrange_inputs(Z, y, mode="transformer")
    assert "transformer" in str(err.value)


# ===========================================================================
# 2. the architecture, and logits
# ===========================================================================
@pytest.mark.parametrize("mode,shape", [("feature_seq", (N_LATENT, 1)),
                                        ("time_window", (4, N_LATENT))])
def test_model_shapes_for_both_input_modes(mode, shape):
    model = md.build_cnn_lstm(shape, n_classes=3)
    assert tuple(model.input_shape[1:]) == shape
    assert model.output_shape[-1] == 3

    X = np.random.default_rng(0).random((7,) + shape).astype("float32")
    logits = md.predict_logits(model, X)
    assert logits.shape == (7, 3)


def test_the_model_outputs_logits_not_probabilities():
    """The last layer has no activation, so rows must NOT already sum to 1."""
    model = md.build_cnn_lstm((N_LATENT, 1), n_classes=3)
    assert model.layers[-1].get_config().get("activation") == "linear"

    X = np.random.default_rng(1).random((20, N_LATENT, 1)).astype("float32") * 5
    logits = md.predict_logits(model, X)
    sums = logits.sum(axis=1)
    assert not np.allclose(sums, 1.0), "these should be logits, not probabilities"
    assert np.any(logits < 0) or np.any(logits > 1), "logits are unbounded"


def test_predict_proba_rows_sum_to_one_and_softmax_is_applied_once():
    model = md.build_cnn_lstm((N_LATENT, 1), n_classes=3)
    X = np.random.default_rng(2).random((20, N_LATENT, 1)).astype("float32") * 5

    probabilities = md.predict_proba(model, X)

    assert probabilities.shape == (20, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    # softmax of the logits, computed independently -- so it was applied ONCE
    logits = md.predict_logits(model, X)
    shifted = logits - logits.max(axis=1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    assert np.allclose(probabilities, expected)


def test_the_loss_is_configured_for_logits():
    model = md.build_cnn_lstm((N_LATENT, 1), n_classes=3)
    loss = model.loss
    assert isinstance(loss, keras.losses.SparseCategoricalCrossentropy)
    assert loss.get_config()["from_logits"] is True, \
        "the loss must apply the softmax, or it would be applied twice"


def test_a_kernel_longer_than_the_input_is_refused():
    with pytest.raises(md.ModelError) as err:
        md.build_cnn_lstm((2, 1), n_classes=3, kernel=5)
    assert "kernel" in str(err.value)


def test_the_architecture_table_totals_the_parameters():
    model = md.build_cnn_lstm((N_LATENT, 1), n_classes=3)
    table = md.architecture_table(model)
    assert table["layer"].iloc[-1] == "TOTAL"
    assert int(table["parameters"].iloc[-1]) == \
        int(table["parameters"].iloc[:-1].sum())
    assert "lstm" in set(table["layer"]) and "logits" in set(table["layer"])


# ===========================================================================
# 3. training: the weights reach the loss, validation stays unweighted
# ===========================================================================
def test_class_weights_reach_the_training_loss(monkeypatch):
    Z, y = make_zy(n_rows=120)
    captured = {}
    real_fit = keras.Model.fit

    def spy(self, *args, **kwargs):
        captured["class_weight"] = kwargs.get("class_weight")
        captured["validation_data"] = kwargs.get("validation_data")
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(keras.Model, "fit", spy)
    md.train_model(Z, y, verbose=False)

    weights = captured["class_weight"]
    assert weights is not None, "class_weight must be passed to fit()"
    assert set(weights) == set(y["Label_encoded"].unique())
    assert all(v > 0 for v in weights.values())
    # validation_data is a plain (X, y) pair -- no sample weights ride along
    assert captured["validation_data"] is not None
    assert len(captured["validation_data"]) == 2, \
        "a third element would be validation sample weights, which must not exist"


def test_the_validation_loss_is_unweighted():
    """history['val_loss'] must equal the plain unweighted loss on X_val."""
    Z, y = make_zy(n_rows=120)
    model, history, info = md.train_model(Z, y, epochs=1, verbose=False)

    from src.mscad import imbalance as im
    _zf, _yf, Z_val, y_val, weights, _sw = im.make_training_sets(Z, y, verbose=False)
    X_val, labels_val, _idx = md.arrange_inputs(Z_val, y_val, mode="feature_seq")

    logits = md.predict_logits(model, X_val)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    unweighted = float(np.mean(loss_fn(labels_val, logits).numpy()))

    assert float(history["val_loss"].iloc[0]) == pytest.approx(unweighted, rel=1e-3)

    # and it is NOT the weighted loss, which the class weights would produce
    per_row = loss_fn(labels_val, logits).numpy() if hasattr(
        loss_fn(labels_val, logits), "numpy") else None
    if per_row is not None and per_row.ndim == 1:
        sample_w = np.array([weights[int(c)] for c in labels_val])
        weighted = float(np.average(per_row, weights=sample_w))
        if abs(weighted - unweighted) > 1e-6:
            assert abs(float(history["val_loss"].iloc[0]) - weighted) > 1e-6


def test_early_stopping_watches_macro_f1():
    Z, y = make_zy(n_rows=120)
    _model, history, info = md.train_model(Z, y, epochs=3, verbose=False)

    assert config.EARLY_STOP_METRIC in history.columns
    assert (history[config.EARLY_STOP_METRIC].between(0, 1)).all()
    assert info["best_epoch"] == int(history[config.EARLY_STOP_METRIC].idxmax()) + 1


def test_training_never_reads_the_test_files():
    """Scrambling the test files must not change a single trained weight."""
    Z, y, Z_test, y_test = write_step8()
    first = md.run_step10(overwrite=True, verbose=False)
    X = np.random.default_rng(4).random((10, N_LATENT, 1)).astype("float32")
    before = md.predict_logits(first["model"], X)

    rng = np.random.default_rng(99)
    wrecked = pd.DataFrame(rng.normal(50, 20, Z_test.shape), columns=Z_test.columns)
    write_step8(Z=Z, y=y, Z_test=wrecked, y_test=y_test)
    second = md.run_step10(overwrite=True, verbose=False)
    after = md.predict_logits(second["model"], X)

    assert np.allclose(before, after), "the test set must not affect training"


def test_the_test_files_are_unchanged_by_training():
    write_step8()
    folder8 = config.STEP8_OUTPUT_DIR
    before = {p.name: p.read_bytes() for p in folder8.iterdir()}

    md.run_step10(overwrite=True, verbose=False)

    for name, content in before.items():
        assert (folder8 / name).read_bytes() == content, name


# ===========================================================================
# 4. end to end
# ===========================================================================
def test_run_step10_writes_the_model_and_a_card():
    write_step8()
    result = md.run_step10(overwrite=True, verbose=False)

    assert result["ok"] is True
    folder = config.STEP10_OUTPUT_DIR
    for key, name in config.STEP10_FILES.items():
        if key in ("loss_curve", "f1_curve"):
            continue
        assert (folder / name).exists(), name

    card = json.loads((folder / config.STEP10_FILES["model_card"])
                      .read_text(encoding="utf-8"))
    assert card["input_mode"] == "feature_seq"
    assert "NOT a time sequence" in card["input_mode_note"]
    assert "no natural order" in card["input_mode_note"]
    assert card["architecture"]["output"].startswith("logits")
    assert card["imbalance"]["strategy"] == config.IMBALANCE_STRATEGY
    assert card["imbalance"]["class_weights"]
    assert card["versions"]["keras"]
    assert card["test_set"].startswith("never read")


def test_step10_loads_instead_of_retraining_when_overwrite_is_off():
    write_step8()
    first = md.run_step10(overwrite=True, verbose=False)
    again = md.run_step10(overwrite=False, verbose=False)

    assert again["status"] == "loaded_existing"
    assert again["info"]["best_epoch"] == first["info"]["best_epoch"]
    X = np.random.default_rng(6).random((5, N_LATENT, 1)).astype("float32")
    assert np.allclose(md.predict_logits(first["model"], X),
                       md.predict_logits(again["model"], X), atol=1e-5)


# ===========================================================================
# 5. the Windows-path guard
# ===========================================================================
def test_step10_refuses_a_windows_path_off_windows(tmp_path, monkeypatch, capsys,
                                                   assert_no_leaked_output):
    """On Linux, Path(r"C:\\...") is RELATIVE -- nothing may be created."""
    monkeypatch.setattr(config, "STEP10_OUTPUT_DIR",
                        config.Path(r"C:\MSCAD2022\outputs\step10"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    assert md.prepare_output_dir(config.STEP10_OUTPUT_DIR, "the Step 10 outputs") \
        is None
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("prepare_output_dir for the Step 10 folder")


def test_step10_output_dir_is_under_data_root_not_the_repo():
    assert REAL_STEP10_DIR.parent.parent == config.DATA_ROOT
    assert config.PROJECT_ROOT not in REAL_STEP10_DIR.parents
    assert REAL_STEP10_DIR.name == "step10"
