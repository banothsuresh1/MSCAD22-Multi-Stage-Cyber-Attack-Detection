r"""
tests/test_experiments_baseline6.py -- pytest tests for Baseline 6
(AE -> PCA -> CNN-LSTM), src/mscad/experiments_baseline6.py.

NEW FILE.  The point of these tests is the one thing Baseline 6 must get
right: it reuses Baseline 4's SAVED encoder and SAVED PCA and never fits a
feature space of its own, so the B4/B6 difference is the classifier and
nothing else.

Everything runs inside tmp_path on a handful of synthetic rows.  The only
Keras object built is a two-layer encoder; nothing is trained for longer than
two epochs.  Never C:\\MSCAD2022.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import experiments_baseline6 as b6

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")

FEATURES = ["f%d" % i for i in range(1, 7)]
LATENT = 4


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR", tmp_path / "step12")
    monkeypatch.setattr(config, "STEP12_B6_OUTPUT_DIR",
                        tmp_path / "step12_baseline6", raising=False)
    monkeypatch.setattr(config, "STEP8_OUTPUT_DIR", tmp_path / "step8")


# ===========================================================================
# helpers -- Baseline 4's folder, exactly as 12.4 leaves it
# ===========================================================================
def tiny_encoder(n_features=len(FEATURES), latent=LATENT):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(latent, activation="relu", name="z"),
    ])
    return model          # not compiled: nothing here trains the encoder


def latent_frame(rows=40, latent=LATENT, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.random((rows, latent)) * np.array([10.0, 5.0, 1.0, 0.2])
    return pd.DataFrame(values, columns=["z%d" % i for i in range(1, latent + 1)])


def write_baseline4(tmp_path, components=2):
    """Baseline 4's saved encoder, settings and fitted PCA."""
    from joblib import dump
    from sklearn.decomposition import PCA

    folder = tmp_path / "step12" / config.BASELINE4["name"]
    folder.mkdir(parents=True, exist_ok=True)
    names = config.BASELINE4_FILES
    tiny_encoder().save(folder / names["encoder"])
    (folder / names["ae_config"]).write_text(
        json.dumps({"features": FEATURES, "latent_dim": LATENT}),
        encoding="utf-8")
    pca = PCA(n_components=components, random_state=config.RANDOM_STATE)
    pca.fit(latent_frame().to_numpy(dtype="float64"))
    dump(pca, folder / names["pca"])
    return folder


def digests(folder):
    names = config.BASELINE4_FILES
    return {name: hashlib.sha256((folder / names[name]).read_bytes()).hexdigest()
            for name in ("encoder", "pca")}


# ===========================================================================
# 12.6  the folder and the "already done" test
# ===========================================================================
def test_baseline6_writes_to_its_own_new_folder(tmp_path):
    place = b6.folder()

    assert place == tmp_path / "step12_baseline6"
    assert place != tmp_path / "step12" / config.BASELINE4["name"]
    assert b6.is_done() is False


def test_is_done_follows_its_own_metrics_file(tmp_path):
    place = tmp_path / "step12_baseline6"
    place.mkdir(parents=True, exist_ok=True)
    (place / config.BASELINE6_FILES["metrics"]).write_text("{}",
                                                           encoding="utf-8")

    assert b6.is_done() is True


# ===========================================================================
# 12.6a  Baseline 4's encoder and PCA, reused and never refitted
# ===========================================================================
def test_the_saved_encoder_and_pca_are_loaded_from_baseline_4s_folder(tmp_path):
    folder = write_baseline4(tmp_path, components=2)
    before = digests(folder)

    pipeline = b6.load_baseline4_pipeline(verbose=False)

    assert pipeline["order"] == FEATURES
    assert int(pipeline["pca"].n_components_) == 2
    assert Path(pipeline["encoder_path"]) == folder / "encoder.keras"
    assert Path(pipeline["pca_path"]) == folder / "pca.joblib"
    # nothing was refitted: both files are byte for byte what B4 left
    assert digests(folder) == before


def test_the_reused_encoder_produces_baseline_4s_latent_space(tmp_path):
    write_baseline4(tmp_path)
    pipeline = b6.load_baseline4_pipeline(verbose=False)
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((10, len(FEATURES))), columns=FEATURES)

    first = pipeline["encode"](X, what="test features")
    again = pipeline["encode"](X, what="test features")

    assert first.shape == (10, LATENT)
    np.testing.assert_allclose(first.to_numpy(), again.to_numpy())


def test_a_missing_baseline_4_is_named_rather_than_worked_around(tmp_path):
    folder = write_baseline4(tmp_path)
    (folder / config.BASELINE4_FILES["pca"]).unlink()

    with pytest.raises(b6.Baseline6Error) as error:
        b6.load_baseline4_pipeline(verbose=False)

    assert "Baseline 4's fitted PCA is missing" in str(error.value)
    assert "Run the Baseline 4 cells (12.4) first" in str(error.value)
    assert "never fits one of its own" in str(error.value)


def test_the_encoder_checks_its_feature_order_before_encoding(tmp_path):
    write_baseline4(tmp_path)
    pipeline = b6.load_baseline4_pipeline(verbose=False)
    rng = np.random.default_rng(2)
    wrong = pd.DataFrame(rng.random((5, len(FEATURES))),
                         columns=list(reversed(FEATURES)))

    with pytest.raises(Exception):
        pipeline["encode"](wrong, what="test features")


# ===========================================================================
# 12.6a  B6_PCA_COMPONENTS as an integer: a NEW PCA, training latents only
# ===========================================================================
def test_an_integer_setting_fits_a_new_pca_on_the_training_latents_only(
        tmp_path):
    Z_train = latent_frame(rows=40, seed=3)
    Z_other = latent_frame(rows=40, seed=99)       # different "test" rows

    pca, info = b6.fit_new_pca(Z_train, 3, seed=config.RANDOM_STATE)
    again, _info = b6.fit_new_pca(Z_train, 3, seed=config.RANDOM_STATE)

    assert int(pca.n_components_) == 3
    assert "TRAINING" in info["fitted_on"] or "training" in info["fitted_on"]
    # the fit is a pure function of the training latents: the other rows were
    # never an argument, and asking twice gives the same components
    np.testing.assert_allclose(pca.components_, again.components_)
    assert pca.components_.shape[1] == Z_train.shape[1]
    assert not np.allclose(
        pca.components_,
        b6.fit_new_pca(Z_other, 3, seed=config.RANDOM_STATE)[0].components_)


def test_baseline_4s_own_pca_is_untouched_when_a_new_one_is_fitted(tmp_path):
    folder = write_baseline4(tmp_path, components=2)
    before = digests(folder)

    b6.fit_new_pca(latent_frame(), 3, seed=config.RANDOM_STATE)

    assert digests(folder) == before


def test_asking_for_more_components_than_the_latent_space_is_refused():
    with pytest.raises(b6.Baseline6Error) as error:
        b6.fit_new_pca(latent_frame(latent=LATENT), LATENT + 1)

    assert "B6_PCA_COMPONENTS" in str(error.value)
    assert "only %d dimensions" % LATENT in str(error.value)


def test_the_default_setting_reuses_baseline_4s_components():
    assert config.B6_PCA_COMPONENTS == "same_as_B4"


# ===========================================================================
# 12.6a  the PCA length check -- what the convolution has to slide over
# ===========================================================================
def test_two_components_leave_the_convolution_almost_nothing_to_slide_over():
    table = b6.pca_length_table(2)
    row = table.iloc[0]

    assert int(row["PCA components (sequence length in)"]) == 2
    assert int(row["kernel"]) == int(config.CNN_KERNEL)
    assert int(row["positions the LSTM sees"]) == 1
    assert bool(row["kernel fits inside the sequence"]) is False
    assert row["result"] == "[WARN]"


def test_the_warning_sentence_says_what_b6_can_and_cannot_be_read_as():
    warning = b6.pca_length_warning(b6.pca_length_table(2))

    assert warning is not None
    assert "2 component(s)" in warning
    assert "1 position(s)" in warning
    assert "PCA features with an extra learned front end" in warning
    assert "deliberately NOT changed" in warning


def test_a_long_enough_sequence_passes_with_no_warning():
    table = b6.pca_length_table(15)
    row = table.iloc[0]

    assert bool(row["kernel fits inside the sequence"]) is True
    assert int(row["positions the LSTM sees"]) == 15 // int(config.CNN_POOL)
    assert row["result"] == "[OK]"
    assert b6.pca_length_warning(table) is None


def test_the_length_check_uses_the_shared_architectures_own_numbers():
    table = b6.pca_length_table(8)
    row = table.iloc[0]

    assert int(row["kernel"]) == int(config.CNN_KERNEL)
    assert int(row["pool size"]) == int(config.CNN_POOL)
    # padding="same" keeps the length, then pooling divides it
    assert int(row["length after Conv1D (padding=same)"]) == 8
    assert int(row["positions the LSTM sees"]) == 8 // int(config.CNN_POOL)


# ===========================================================================
# 12.6b  the protocol, asserted rather than assumed
# ===========================================================================
def protocol_info(seed=None, model_type="cnn_lstm"):
    from src.mscad import imbalance as im

    y_train = pd.DataFrame({
        b6.TARGET_COLUMN: np.array([0] * 40 + [1] * 20 + [2] * 20),
        "Label_binary": np.array([0] * 40 + [1] * 40)})
    marker = pd.DataFrame({"row": np.arange(len(y_train))})
    _f, _yf, _v, _yv, weights, _s = im.make_training_sets(marker, y_train,
                                                          verbose=False)
    return {"y_train": y_train, "class_weights": weights,
            "seed": config.RANDOM_STATE if seed is None else seed,
            "model_type": model_type}


def test_the_protocol_checks_pass_when_b6_ran_as_approach_1_does(tmp_path):
    write_baseline4(tmp_path)
    pipeline = b6.load_baseline4_pipeline(verbose=False)

    table = b6.protocol_checks(protocol_info(), pipeline, refit=False,
                               verbose=False)

    assert len(table)
    assert set(table["result"]) == {"[OK]"}
    checks = " ".join(table["check"])
    assert "fit/validation indices" in checks
    assert "class weights" in checks
    assert "seed" in checks
    assert "shared CNN-LSTM builder" in checks
    assert "hyperparameters" in checks
    assert "byte for byte" in checks


def test_a_different_seed_is_reported_as_a_failure(tmp_path):
    write_baseline4(tmp_path)
    pipeline = b6.load_baseline4_pipeline(verbose=False)

    table = b6.protocol_checks(protocol_info(seed=config.RANDOM_STATE + 1),
                               pipeline, refit=False, verbose=False)
    failed = table[table["result"] == "[FAIL]"]

    assert len(failed) == 1
    assert "seed is Approach 1's" in failed.iloc[0]["check"]


def test_a_different_classifier_is_reported_as_a_failure(tmp_path):
    write_baseline4(tmp_path)
    pipeline = b6.load_baseline4_pipeline(verbose=False)

    table = b6.protocol_checks(protocol_info(model_type="lstm"), pipeline,
                               refit=False, verbose=False)
    failed = table[table["result"] == "[FAIL]"]

    assert len(failed) == 1
    assert "shared CNN-LSTM builder" in failed.iloc[0]["check"]


def test_the_hyperparameter_row_names_every_shared_setting(tmp_path):
    write_baseline4(tmp_path)
    pipeline = b6.load_baseline4_pipeline(verbose=False)

    table = b6.protocol_checks(protocol_info(), pipeline, refit=False,
                               verbose=False)
    detail = table[table["check"].str.contains("hyperparameters")].iloc[0][
        "detail"]

    for number in (config.CNN_KERNEL, config.CNN_POOL, config.LSTM_UNITS,
                   config.DENSE_UNITS, config.CLF_EPOCHS,
                   config.CLF_BATCH_SIZE, config.CLF_PATIENCE):
        assert str(number) in detail
    assert config.EARLY_STOP_METRIC in detail


def test_the_reused_pca_row_carries_baseline_4s_own_digest(tmp_path):
    folder = write_baseline4(tmp_path)
    pipeline = b6.load_baseline4_pipeline(verbose=False)
    expected = hashlib.sha256(
        (folder / config.BASELINE4_FILES["pca"]).read_bytes()).hexdigest()

    table = b6.protocol_checks(protocol_info(), pipeline, refit=False,
                               verbose=False)
    row = table[table["check"].str.contains("PCA is reused")].iloc[0]

    assert expected[:16] in row["detail"]
    assert row["result"] == "[OK]"


def test_a_refit_says_so_instead_of_claiming_baseline_4s_pca(tmp_path):
    write_baseline4(tmp_path)
    pipeline = b6.load_baseline4_pipeline(verbose=False)

    table = b6.protocol_checks(protocol_info(), pipeline, refit=True,
                               verbose=False)
    checks = " ".join(table["check"])

    assert "a NEW PCA was fitted, on the TRAINING latents only" in checks
    assert "PCA is reused byte for byte" not in checks
    assert set(table["result"]) == {"[OK]"}


# ===========================================================================
# config: Baseline 6's block, and what it must not have changed
# ===========================================================================
def test_baseline6_is_the_eighth_row_of_the_display_order():
    order = list(config.STEP12_DISPLAY_ORDER)

    assert len(order) == 8
    assert order[-1] == (config.BASELINE6_NAME, config.BASELINE6_LABEL)
    assert order[-2][0] == "baseline5"
    assert config.BASELINE6_NAME in config.STEP12_ISOLATES


def test_baseline6_reuses_baseline_4s_feature_source_and_encoder():
    spec = config.BASELINE6_SPEC

    assert spec["feature_source"] == config.BASELINE4_AE_FEATURES
    assert spec["model_type"] == "cnn_lstm"
    assert spec["reuse_encoder_from"] == config.BASELINE4["name"]


def test_baseline6_has_its_own_colour_and_output_folder():
    assert config.MODEL_COLOURS[config.BASELINE6_NAME] == "#a06a2c"
    assert len(set(config.MODEL_COLOURS.values())) == 8
    assert "step12_baseline6" in str(config.STEP12_B6_OUTPUT_DIR)


def test_overwrite_baseline6_is_off_by_default():
    assert config.OVERWRITE_BASELINE6 is False
