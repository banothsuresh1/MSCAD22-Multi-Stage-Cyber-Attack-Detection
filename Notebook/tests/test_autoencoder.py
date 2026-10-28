"""
tests/test_autoencoder.py -- pytest tests for STEP 7 (train the autoencoder) and
STEP 8 (encode the training and test sets).

Everything is built from a small synthetic frame in a temporary folder: 120 rows,
8 features, 3 classes, a 3-number bottleneck and 3 epochs, so the whole module
runs in seconds on a CPU.  No real data, and never C:\\MSCAD2022.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pytest tests -v
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

# quieten the TensorFlow C++ logger before anything imports it
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import autoencoder as ae

REAL_STEP7_DIR = config.STEP7_OUTPUT_DIR          # captured before any patching
REAL_STEP8_DIR = config.STEP8_OUTPUT_DIR

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")

N_ROWS = 120
N_FEATURES = 8
CLASSES = ["Normal", "Port_Scan", "Brute_Force"]


@pytest.fixture(autouse=True)
def _tiny_and_in_tmp(tmp_path, monkeypatch):
    """Everything in tmp_path, and small enough to train in a second."""
    monkeypatch.setattr(config, "STEP6_OUTPUT_DIR", tmp_path / "step6")
    monkeypatch.setattr(config, "STEP7_OUTPUT_DIR", tmp_path / "step7")
    monkeypatch.setattr(config, "STEP8_OUTPUT_DIR", tmp_path / "step8")
    monkeypatch.setattr(config, "AE_LATENT_DIM", 3)
    monkeypatch.setattr(config, "AE_HIDDEN", [5])
    monkeypatch.setattr(config, "AE_EPOCHS", 3)
    monkeypatch.setattr(config, "AE_BATCH_SIZE", 16)
    monkeypatch.setattr(config, "AE_VAL_SIZE", 0.2)
    monkeypatch.setattr(config, "AE_PATIENCE", 2)
    monkeypatch.setattr(config, "N_SELECTED_FEATURES", N_FEATURES)


# ===========================================================================
# the synthetic Step 6 outputs
# ===========================================================================
def feature_names(n: int = N_FEATURES) -> list[str]:
    return ["feat_%02d" % i for i in range(n)]


def make_xy(n_rows: int = N_ROWS, seed: int = 0):
    """Features already in [0, 1], as Step 5's MinMax scaler leaves them."""
    rng = np.random.default_rng(seed)
    y_code = np.resize([0, 1, 2], n_rows)
    X = pd.DataFrame(rng.random((n_rows, N_FEATURES)), columns=feature_names())
    y = pd.DataFrame({"Label": [CLASSES[c] for c in y_code],
                      "Label_encoded": y_code,
                      "Label_binary": (y_code > 0).astype("int64")})
    return X, y


def write_step6(X_train=None, y_train=None, X_test=None, y_test=None):
    """Lay down the five Step 6 files Steps 7-8 read."""
    if X_train is None:
        X_train, y_train = make_xy(seed=0)
    if X_test is None:
        X_test, y_test = make_xy(n_rows=45, seed=1)
    folder = config.STEP6_OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    names = config.STEP6_FILES
    X_train.to_csv(folder / names["X_train_selected"], index=False)
    X_test.to_csv(folder / names["X_test_selected"], index=False)
    y_train.to_csv(folder / names["y_train"], index=False)
    y_test.to_csv(folder / names["y_test"], index=False)
    (folder / names["selected"]).write_text(json.dumps({
        "features": feature_names(), "n_selected": N_FEATURES}), encoding="utf-8")
    return X_train, X_test, y_train, y_test


# ===========================================================================
# 1. 7.1 -- the input checks
# ===========================================================================
def test_inputs_are_checked_before_training():
    X_train, X_test, y_train, y_test = write_step6()
    table, ok = ae.check_inputs(X_train, X_test, y_train, y_test,
                                feature_names(), verbose=False)
    assert ok is True
    assert "[FAIL]" not in set(table["result"])


def test_a_feature_frame_in_the_wrong_order_is_refused_up_front():
    X_train, X_test, y_train, y_test = write_step6()
    swapped = X_train[feature_names()[::-1]]
    with pytest.raises(ae.AutoencoderError) as err:
        ae.check_inputs(swapped, X_test, y_train, y_test, feature_names(),
                        verbose=False)
    assert "order" in str(err.value)


def test_a_label_column_among_the_features_is_refused():
    X_train, X_test, y_train, y_test = write_step6()
    leaky = X_train.copy()
    leaky["Label_encoded"] = y_train["Label_encoded"]
    with pytest.raises(ae.AutoencoderError) as err:
        ae.check_inputs(leaky, X_test, y_train, y_test,
                        feature_names() + ["Label_encoded"], verbose=False)
    assert "Label_encoded" in str(err.value)


def test_a_latent_dim_larger_than_the_input_is_refused(monkeypatch):
    """The guard refuses latent > input. Since the AE=30 sensitivity run
    (commit 846a775) a latent width EQUAL to the input is allowed on purpose
    -- the degenerate, non-compressing case the AE=30 configs use on the 30
    selected features -- so only a latent LARGER than the input is an error."""
    X_train, X_test, y_train, y_test = write_step6()
    monkeypatch.setattr(config, "AE_LATENT_DIM", N_FEATURES + 1)
    with pytest.raises(ae.AutoencoderError) as err:
        ae.check_inputs(X_train, X_test, y_train, y_test, feature_names(),
                        verbose=False)
    assert "cannot exceed the input" in str(err.value)


def test_a_latent_dim_equal_to_the_input_is_allowed(monkeypatch):
    """latent == input is the AE=30 degenerate case, allowed by design."""
    X_train, X_test, y_train, y_test = write_step6()
    monkeypatch.setattr(config, "AE_LATENT_DIM", N_FEATURES)
    table, all_ok = ae.check_inputs(X_train, X_test, y_train, y_test,
                                    feature_names(), verbose=False)
    assert all_ok is True


# ===========================================================================
# 2. 7.2 -- the architecture
# ===========================================================================
def test_the_encoder_shares_the_autoencoders_weights():
    """The encoder is the first half of the AE, not a copy of it."""
    autoencoder, encoder = ae.build_autoencoder(N_FEATURES)
    assert encoder.output_shape[-1] == config.AE_LATENT_DIM
    assert encoder.input_shape[-1] == N_FEATURES
    shared = {id(w) for w in encoder.weights}
    assert shared <= {id(w) for w in autoencoder.weights}, \
        "the encoder must reuse the trained weights, not hold its own"


def test_the_architecture_table_lists_both_halves():
    autoencoder, _encoder = ae.build_autoencoder(N_FEATURES)
    table = ae.architecture_table(autoencoder)
    assert set(table["half"]) >= {"encoder", "decoder"}
    assert table["layer"].iloc[-1] == "TOTAL"
    assert int(table["parameters"].iloc[-1]) == \
        int(table["parameters"].iloc[:-1].sum())
    assert "latent" in set(table["layer"])


def test_z_column_names_are_padded_and_sort_correctly():
    assert ae.z_columns(3) == ["z01", "z02", "z03"]
    assert ae.z_columns(15)[0] == "z01" and ae.z_columns(15)[-1] == "z15"
    assert ae.z_columns(15) == sorted(ae.z_columns(15)), "they must sort as text"


# ===========================================================================
# 3. 7.3 -- training
# ===========================================================================
def test_training_is_deterministic_with_a_fixed_seed():
    X_train, _xt, y_train, _yt = write_step6()

    _a, encoder_a, history_a, _ia = ae.train_autoencoder(X_train, y_train,
                                                         verbose=False)
    _b, encoder_b, history_b, _ib = ae.train_autoencoder(X_train, y_train,
                                                         verbose=False)

    assert np.allclose(history_a["loss"], history_b["loss"])
    Z_a = ae.make_encoder_function(encoder_a, feature_names())(X_train)
    Z_b = ae.make_encoder_function(encoder_b, feature_names())(X_train)
    assert np.array_equal(Z_a.to_numpy(), Z_b.to_numpy()), \
        "the same seed must give bit-identical codes"


def test_the_encoder_output_has_the_latent_shape():
    X_train, X_test, y_train, _yt = write_step6()
    _ae, encoder, _h, _i = ae.train_autoencoder(X_train, y_train, verbose=False)
    encode = ae.make_encoder_function(encoder, feature_names())

    Z = encode(X_train)
    assert Z.shape == (len(X_train), config.AE_LATENT_DIM)
    assert list(Z.columns) == ae.z_columns(config.AE_LATENT_DIM)
    assert np.isfinite(Z.to_numpy()).all()
    assert encode(X_test).shape == (len(X_test), config.AE_LATENT_DIM)


def test_the_autoencoder_never_sees_the_test_set():
    """Changing X_test must not move a single weight."""
    X_train, X_test, y_train, y_test = write_step6()
    first = ae.run_step7(overwrite=True, verbose=False)
    Z_first = ae.make_encoder_function(first["encoder"], feature_names())(X_train)

    rng = np.random.default_rng(99)
    wrecked = pd.DataFrame(rng.random(X_test.shape), columns=X_test.columns)
    write_step6(X_train=X_train, y_train=y_train, X_test=wrecked, y_test=y_test)
    second = ae.run_step7(overwrite=True, verbose=False)
    Z_second = ae.make_encoder_function(second["encoder"], feature_names())(X_train)

    assert np.array_equal(Z_first.to_numpy(), Z_second.to_numpy())
    assert first["info"]["train_mse"] == pytest.approx(second["info"]["train_mse"])


def test_the_history_and_best_epoch_are_recorded():
    X_train, _xt, y_train, _yt = write_step6()
    _a, _e, history, info = ae.train_autoencoder(X_train, y_train, verbose=False)

    assert list(history.columns[:1]) == ["epoch"]
    assert "loss" in history.columns and "val_loss" in history.columns
    assert len(history) == info["epochs_run"] <= config.AE_EPOCHS
    assert info["best_epoch"] == int(history["val_loss"].idxmin()) + 1
    assert info["fit_rows"] + info["validation_rows"] == len(X_train)


# ===========================================================================
# 4. encode() refuses the wrong columns
# ===========================================================================
def test_encode_rejects_a_reordered_frame():
    """A Dense layer reads positions, so a reordered frame would be silently wrong."""
    X_train, _xt, y_train, _yt = write_step6()
    _a, encoder, _h, _i = ae.train_autoencoder(X_train, y_train, verbose=False)
    encode = ae.make_encoder_function(encoder, feature_names())

    swapped = X_train[feature_names()[::-1]]
    with pytest.raises(ae.AutoencoderError) as err:
        encode(swapped)
    message = str(err.value)
    assert "WRONG ORDER" in message
    assert "position" in message


def test_encode_rejects_missing_and_unexpected_columns():
    X_train, _xt, y_train, _yt = write_step6()
    _a, encoder, _h, _i = ae.train_autoencoder(X_train, y_train, verbose=False)
    encode = ae.make_encoder_function(encoder, feature_names())

    with pytest.raises(ae.AutoencoderError) as err:
        encode(X_train.drop(columns=["feat_03"]))
    assert "missing" in str(err.value) and "feat_03" in str(err.value)

    extra = X_train.copy()
    extra["feat_99"] = 0.5
    with pytest.raises(ae.AutoencoderError) as err:
        encode(extra)
    assert "unexpected" in str(err.value) and "feat_99" in str(err.value)


def test_encode_accepts_the_right_frame():
    X_train, _xt, y_train, _yt = write_step6()
    _a, encoder, _h, _i = ae.train_autoencoder(X_train, y_train, verbose=False)
    encode = ae.make_encoder_function(encoder, feature_names())
    assert encode(X_train).shape == (len(X_train), config.AE_LATENT_DIM)
    assert encode.feature_order == feature_names()


# ===========================================================================
# 5. the saved encoder reloads and gives identical codes
# ===========================================================================
def test_the_saved_encoder_reloads_and_gives_identical_z():
    X_train, _xt, _yt, _y = write_step6()
    result = ae.run_step7(overwrite=True, verbose=False)
    assert result["ok"] is True

    before = ae.make_encoder_function(result["encoder"], feature_names())(X_train)
    encode, _encoder, feature_order = ae.load_encoder(verbose=False)
    after = encode(X_train)

    assert feature_order == feature_names()
    assert np.allclose(before.to_numpy(), after.to_numpy(), rtol=0, atol=1e-6)
    assert list(after.columns) == ae.z_columns(config.AE_LATENT_DIM)


def test_step7_writes_every_file_and_records_the_versions():
    write_step6()
    result = ae.run_step7(overwrite=True, verbose=False)

    folder = config.STEP7_OUTPUT_DIR
    for key, name in config.STEP7_FILES.items():
        if key == "loss_curve":
            continue                      # matplotlib may not be installed
        assert (folder / name).exists(), name

    saved = json.loads((folder / config.STEP7_FILES["ae_config"])
                       .read_text(encoding="utf-8"))
    assert saved["features"] == feature_names()
    assert saved["latent_dim"] == config.AE_LATENT_DIM
    assert saved["supervised"] is False
    assert saved["versions"]["framework"] == config.AE_FRAMEWORK
    assert saved["versions"]["keras"]
    assert "[FAIL]" not in set(result["saved"]["verification"]["result"])


# ===========================================================================
# 6. Step 8
# ===========================================================================
def test_step8_writes_z_files_with_the_right_shapes_and_aligned_labels():
    X_train, X_test, y_train, y_test = write_step6()
    step7 = ae.run_step7(overwrite=True, verbose=False)
    result = ae.run_step8(overwrite=True, encoder=step7["encoder"],
                          autoencoder=step7["autoencoder"],
                          feature_order=step7["feature_order"], verbose=False)

    assert result["ok"] is True
    folder = config.STEP8_OUTPUT_DIR
    Z_train = pd.read_csv(folder / config.STEP8_FILES["Z_train"])
    Z_test = pd.read_csv(folder / config.STEP8_FILES["Z_test"])
    saved_y_train = pd.read_csv(folder / config.STEP8_FILES["y_train"])
    saved_y_test = pd.read_csv(folder / config.STEP8_FILES["y_test"])

    assert Z_train.shape == (len(X_train), config.AE_LATENT_DIM)
    assert Z_test.shape == (len(X_test), config.AE_LATENT_DIM)
    assert list(Z_train.columns) == list(Z_test.columns) == \
        ae.z_columns(config.AE_LATENT_DIM)
    assert len(saved_y_train) == len(Z_train)
    assert len(saved_y_test) == len(Z_test)
    assert saved_y_train["Label"].tolist() == y_train["Label"].tolist()
    assert saved_y_test["Label"].tolist() == y_test["Label"].tolist()
    assert "[FAIL]" not in set(result["verification"]["result"])


def test_step8_uses_the_same_encoder_for_both_sets():
    X_train, X_test, _y, _yt = write_step6()
    step7 = ae.run_step7(overwrite=True, verbose=False)
    result = ae.run_step8(overwrite=True, encoder=step7["encoder"],
                          autoencoder=step7["autoencoder"],
                          feature_order=step7["feature_order"], verbose=False)

    encode = ae.make_encoder_function(step7["encoder"], feature_names())
    assert np.allclose(result["Z_train"].to_numpy(), encode(X_train).to_numpy())
    assert np.allclose(result["Z_test"].to_numpy(), encode(X_test).to_numpy())
    assert result["test_mse"] is not None and result["test_mse"] > 0


def test_step8_row_order_is_unchanged():
    X_train, _xt, y_train, _yt = write_step6()
    step7 = ae.run_step7(overwrite=True, verbose=False)
    result = ae.run_step8(overwrite=True, encoder=step7["encoder"],
                          autoencoder=step7["autoencoder"],
                          feature_order=step7["feature_order"], verbose=False)
    # row i of Z is still row i of X, so the labels still belong to it
    assert result["y_train"]["Label"].tolist() == y_train["Label"].tolist()
    assert len(result["Z_train"]) == len(X_train)


def test_step8_loads_instead_of_re_encoding_when_overwrite_is_off():
    write_step6()
    step7 = ae.run_step7(overwrite=True, verbose=False)
    first = ae.run_step8(overwrite=True, encoder=step7["encoder"],
                         autoencoder=step7["autoencoder"],
                         feature_order=step7["feature_order"], verbose=False)
    again = ae.run_step8(overwrite=False, verbose=False)

    assert again["status"] == "loaded_existing"
    assert again["Z_train"].shape == first["Z_train"].shape
    assert np.allclose(again["Z_train"].to_numpy(), first["Z_train"].to_numpy(),
                       atol=1e-6)


def test_step7_loads_instead_of_retraining_when_overwrite_is_off():
    write_step6()
    first = ae.run_step7(overwrite=True, verbose=False)
    again = ae.run_step7(overwrite=False, verbose=False)

    assert again["status"] == "loaded_existing"
    assert again["feature_order"] == first["feature_order"]
    assert again["info"]["best_epoch"] == first["info"]["best_epoch"]


def test_steps_7_and_8_do_not_touch_the_step6_outputs():
    write_step6()
    folder6 = config.STEP6_OUTPUT_DIR
    before = {p.name: p.read_bytes() for p in folder6.iterdir()}

    step7 = ae.run_step7(overwrite=True, verbose=False)
    ae.run_step8(overwrite=True, encoder=step7["encoder"],
                 autoencoder=step7["autoencoder"],
                 feature_order=step7["feature_order"], verbose=False)

    for name, content in before.items():
        assert (folder6 / name).read_bytes() == content, name


# ===========================================================================
# 7. the Windows-path guard
# ===========================================================================
def test_steps_7_and_8_refuse_a_windows_path_off_windows(tmp_path, monkeypatch,
                                                          capsys,
                                                          assert_no_leaked_output):
    """On Linux, Path(r"C:\\...") is RELATIVE -- nothing may be created."""
    monkeypatch.setattr(config, "STEP7_OUTPUT_DIR",
                        config.Path(r"C:\MSCAD2022\outputs\step7"))
    monkeypatch.setattr(config, "STEP8_OUTPUT_DIR",
                        config.Path(r"C:\MSCAD2022\outputs\step8"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    assert ae.prepare_output_dir(config.STEP7_OUTPUT_DIR, "the Step 7 outputs") is None
    assert ae.prepare_output_dir(config.STEP8_OUTPUT_DIR, "the Step 8 outputs") is None
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("prepare_output_dir for the Step 7 and 8 folders")


def test_step7_and_step8_dirs_are_under_data_root_not_the_repo():
    for folder, name in ((REAL_STEP7_DIR, "step7"), (REAL_STEP8_DIR, "step8")):
        assert folder.parent.parent == config.DATA_ROOT
        assert config.PROJECT_ROOT not in folder.parents
        assert folder.name == name
