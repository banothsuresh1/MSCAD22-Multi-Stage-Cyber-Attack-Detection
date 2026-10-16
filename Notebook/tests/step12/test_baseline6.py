r"""
tests/step12/test_baseline6.py -- Baseline 6 (AE -> PCA -> CNN-LSTM), the one
configuration in Step 12 that trains.

NEW FILE.  The claims worth checking are the ones that make B6 comparable:
it reuses Baseline 4's SAVED encoder and SAVED PCA rather than fitting a
feature space of its own, an integer component count fits a NEW PCA on the
TRAINING latents only, the sequence-length check predicts what the shared
builder will actually do, and the protocol is Approach 1's.

The only model built here is a two-layer encoder; nothing trains for more than
two epochs, and everything is inside tmp_path.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp
from src.mscad import experiments_baseline6 as b6mod
from src.mscad.experiments_step12 import baseline6 as module
from src.mscad.experiments_step12 import common

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent.parent
KEY = config.BASELINE6_NAME
FEATURES = ["f%d" % i for i in range(1, 7)]
LATENT = 6


def explode(*_args, **_kwargs):                 # pragma: no cover - must not run
    raise AssertionError("run_baseline6 was called; this file must LOAD")


def tiny_encoder(n_features=len(FEATURES), latent=LATENT):
    return keras.Sequential([keras.layers.Input(shape=(n_features,)),
                             keras.layers.Dense(latent, activation="relu",
                                                name="z")])


def latent_frame(rows=40, latent=LATENT, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.linspace(10.0, 0.2, latent)
    return pd.DataFrame(rng.random((rows, latent)) * scale,
                        columns=["z%d" % i for i in range(1, latent + 1)])


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
    return {kind: hashlib.sha256(
        (folder / names[kind]).read_bytes()).hexdigest()
        for kind in ("encoder", "pca")}


# ===========================================================================
# the shared interface, and what B6 declares it is
# ===========================================================================
def test_the_file_exposes_the_shared_interface():
    assert module.KEY == KEY
    assert module.DISPLAY_LABEL == dict(config.STEP12_DISPLAY_ORDER)[KEY]
    assert module.COMPONENTS["CNN"].startswith("yes")
    assert module.COMPONENTS["attention"] == "no"
    assert module.COMPONENTS["LSTM"] == "yes"
    assert "never refitted" in module.COMPONENTS["autoencoder"]
    assert module.COMPONENTS["what it isolates"] == config.STEP12_ISOLATES[KEY]


def test_the_pca_component_says_which_feature_space_b6_is_in(monkeypatch):
    assert module.reuses_baseline4_pca() is (
        isinstance(config.B6_PCA_COMPONENTS, str))
    monkeypatch.setattr(config, "B6_PCA_COMPONENTS", "same_as_B4")
    assert module.reuses_baseline4_pca() is True
    assert "never refitted" in module._pca_component()
    monkeypatch.setattr(config, "B6_PCA_COMPONENTS", 4)
    assert module.reuses_baseline4_pca() is False
    assert "NEW PCA(4)" in module._pca_component()
    assert "untouched" in module._pca_component()


# ===========================================================================
# Baseline 4's saved encoder and PCA, reused and never refitted
# ===========================================================================
def test_it_uses_baseline_4s_saved_encoder_and_pca_unchanged(tmp_path):
    folder = write_baseline4(tmp_path, components=4)
    before = digests(folder)

    pipeline = b6mod.load_baseline4_pipeline(verbose=False)

    assert pipeline["order"] == FEATURES
    assert int(pipeline["pca"].n_components_) == 4
    assert digests(folder) == before             # nothing was refitted


def test_a_missing_baseline_4_is_named_rather_than_worked_around(tmp_path):
    folder = write_baseline4(tmp_path)
    (folder / config.BASELINE4_FILES["pca"]).unlink()

    with pytest.raises(b6mod.Baseline6Error) as error:
        b6mod.load_baseline4_pipeline(verbose=False)

    assert "Baseline 4's fitted PCA is missing" in str(error.value)
    assert "never fits one of its own" in str(error.value)


def test_an_integer_count_fits_a_new_pca_on_the_training_latents_only(
        tmp_path):
    folder = write_baseline4(tmp_path, components=2)
    before = digests(folder)
    Z_train = latent_frame(rows=40, seed=3)
    Z_other = latent_frame(rows=40, seed=99)

    pca, info = b6mod.fit_new_pca(Z_train, 3, seed=config.RANDOM_STATE)
    again, _info = b6mod.fit_new_pca(Z_train, 3, seed=config.RANDOM_STATE)

    assert int(pca.n_components_) == 3
    assert "training" in info["fitted_on"].lower()
    np.testing.assert_allclose(pca.components_, again.components_)
    assert not np.allclose(
        pca.components_,
        b6mod.fit_new_pca(Z_other, 3, seed=config.RANDOM_STATE)[0].components_)
    assert digests(folder) == before             # B4's PCA is untouched


def test_more_components_than_the_latent_space_is_refused():
    with pytest.raises(b6mod.Baseline6Error) as error:
        b6mod.fit_new_pca(latent_frame(latent=LATENT), LATENT + 1)

    assert "B6_PCA_COMPONENTS" in str(error.value)


# ===========================================================================
# the PCA length check
# ===========================================================================
def test_a_sequence_shorter_than_the_kernel_cannot_be_built():
    check = module.pca_length_check(int(config.CNN_KERNEL) - 1, verbose=False)
    row = check["table"].iloc[0]

    assert bool(row["the shared CNN-LSTM can be built"]) is False
    assert row["result"] == "[FAIL]"
    assert "REFUSES" in check["warning"]
    assert "B6_PCA_COMPONENTS" in check["warning"]
    assert "deliberately NOT lowered" in check["warning"]


def test_a_buildable_but_short_sequence_warns_and_says_why():
    check = module.pca_length_check(int(config.CNN_KERNEL), verbose=False)
    row = check["table"].iloc[0]

    assert bool(row["the shared CNN-LSTM can be built"]) is True
    assert row["result"] == "[WARN]"
    assert "extra learned front end" in check["warning"]
    assert "deliberately NOT changed" in check["warning"]


def test_a_long_enough_sequence_needs_no_warning():
    check = module.pca_length_check(12, verbose=False)

    assert check["table"].iloc[0]["result"] == "[OK]"
    assert check["warning"] is None


def test_the_length_check_predicts_what_the_builder_actually_does():
    """The check is only worth having if it matches models.build_cnn_lstm."""
    from src.mscad import models as md

    for components in (int(config.CNN_KERNEL) - 1, int(config.CNN_KERNEL),
                       int(config.CNN_POOL) * 2, 12):
        row = module.pca_length_check(components, verbose=False)[
            "table"].iloc[0]
        try:
            model = md.build_cnn_lstm((components, 1), 3, compile_model=False)
        except Exception:
            assert bool(row["the shared CNN-LSTM can be built"]) is False, \
                components
            continue
        assert bool(row["the shared CNN-LSTM can be built"]) is True, components
        seen = int(model.get_layer("lstm").input.shape[1])
        assert seen == int(row["positions the LSTM sees"]), (components, seen)


def test_the_component_count_in_config_is_buildable():
    setting = config.B6_PCA_COMPONENTS
    if isinstance(setting, str):
        assert setting == "same_as_B4"
        return
    assert int(setting) >= int(config.CNN_KERNEL)
    assert module.pca_length_check(int(setting), verbose=False)[
        "table"].iloc[0]["the shared CNN-LSTM can be built"]


# ===========================================================================
# loading, and the lock
# ===========================================================================
def test_a_finished_baseline_6_is_loaded_not_retrained(finished, monkeypatch):
    finished(KEY)
    monkeypatch.setattr(b6mod, "run_baseline6", explode)

    out = module.run(verbose=False)

    assert out["status"] == common.LOADED
    assert out["status_detail"] == "loaded (locked)"
    assert set(out["unchanged"]["result"]) == {"[OK]"}


def test_a_missing_baseline_6_with_the_lock_on_stops(step12_dirs, monkeypatch):
    monkeypatch.setattr(b6mod, "run_baseline6", explode)

    with pytest.raises(common.Step12ConfigError) as error:
        module.run(verbose=False)

    assert "set LOCK_TRAINED_MODELS = False in config.py" in str(error.value)


def test_the_trainer_is_the_existing_run_baseline6(step12_dirs, monkeypatch):
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", False)
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return {"metrics": {"full": {"f1_macro": 0.5, "accuracy": 0.9,
                                     "g_mean": 0.4},
                            "duplicate_free": {}, "binary": {},
                            "inference": {}, "pca_components": 3},
                "protocol": pd.DataFrame([{"check": "x", "result": "[OK]",
                                           "detail": "y"}])}

    monkeypatch.setattr(b6mod, "run_baseline6", fake)

    out = module.run(verbose=False)

    assert len(calls) == 1
    assert calls[0]["overwrite"] == config.OVERWRITE_B6
    assert out["status"] == common.TRAINED
    assert list(out["protocol"]["check"]) == ["x"]


def test_the_module_source_names_no_training_code_of_its_own():
    """Checked on the parsed code, not the text: the docstring names
    models.build_cnn_lstm on purpose, to say where the training lives."""
    import ast

    source = Path(module.__file__).read_text(encoding="utf-8")
    called = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (function.attr if isinstance(function, ast.Attribute)
                else getattr(function, "id", ""))
        called.add(name)

    assert "build_cnn_lstm" not in called
    assert "fit" not in called
    assert "train_classifier" not in called
    assert "run_baseline6" in called            # it calls the existing one


# ===========================================================================
# where B6 writes
# ===========================================================================
def test_baseline_6_writes_to_its_own_folder(finished, tmp_path, monkeypatch):
    finished(KEY)
    monkeypatch.setattr(b6mod, "run_baseline6", explode)

    out = module.run(verbose=False)

    assert Path(out["metrics_path"]).parent == tmp_path / "step12_baseline6"
    assert Path(out["manifest"]["folder"]) == tmp_path / "step12" / "baseline6"
    assert config.B6_OUTPUT_DIR == config.STEP12_B6_OUTPUT_DIR


def test_the_provenance_lists_the_two_baseline_4_files_it_reuses(finished,
                                                                 tmp_path,
                                                                 monkeypatch):
    finished(KEY)
    write_baseline4(tmp_path, components=4)
    monkeypatch.setattr(b6mod, "run_baseline6", explode)

    table = module.run(verbose=False)["provenance"]
    reused = table[table["file"].str.startswith("reused:")]

    assert len(reused) == 2
    assert set(reused["exists"]) == {True}
    for _i, row in reused.iterrows():
        assert len(row["sha256"]) == 64


# ===========================================================================
# runnable on its own
# ===========================================================================
def test_the_module_runs_as_a_script():
    finished = subprocess.run(
        [sys.executable, "-m", "src.mscad.experiments_step12.baseline6",
         "--components"],
        cwd=str(NOTEBOOK_DIR), capture_output=True, text=True, timeout=600)

    assert finished.returncode == 0, finished.stderr[-2000:]
    assert module.DISPLAY_LABEL in finished.stdout
