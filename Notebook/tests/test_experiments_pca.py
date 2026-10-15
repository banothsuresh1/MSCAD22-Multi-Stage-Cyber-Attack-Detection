r"""
tests/test_experiments_pca.py -- pytest tests for Baseline 4 (AE -> PCA -> LSTM).

NEW FILE.  No existing test file is modified, and tests/test_experiments.py is
left exactly as Step 12 wrote it.  Tiny synthetic data, 2-3 epochs, CPU only,
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
from src.mscad import experiments as xp
from src.mscad import experiments_pca as xpca

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")

N_ALL = 9
N_SELECTED = 5
N_ROWS = 120
CLASSES = ["Normal", "Port_Scan", "Brute_Force"]


@pytest.fixture(autouse=True)
def _tiny_and_in_tmp(tmp_path, monkeypatch):
    for step in (4, 5, 6, 7, 8, 9, 10, 11, 12):
        monkeypatch.setattr(config, "STEP%d_OUTPUT_DIR" % step,
                            tmp_path / ("step%d" % step))
    monkeypatch.setattr(config, "AE_LATENT_DIM", 6)
    monkeypatch.setattr(config, "AE_HIDDEN", [8])
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


def latent_frame(n_rows=N_ROWS, n_latent=6, seed=0):
    """A latent matrix with a deliberately uneven variance profile."""
    rng = np.random.default_rng(seed)
    scales = np.linspace(4.0, 0.05, n_latent)
    values = rng.normal(0.0, 1.0, size=(n_rows, n_latent)) * scales
    return pd.DataFrame(values, columns=xpca.pca_columns(n_latent))


# ===========================================================================
# PCA_N_COMPONENTS
# ===========================================================================
def test_a_float_setting_is_a_variance_target():
    assert xpca.pca_components_setting(15, 0.95) == pytest.approx(0.95)
    assert xpca.pca_components_setting(15, 0.5) == pytest.approx(0.5)


def test_an_int_setting_is_a_component_count():
    assert xpca.pca_components_setting(15, 8) == 8
    assert xpca.pca_components_setting(15, 15) == 15


def test_more_components_than_the_latent_space_has_raises_a_clear_error():
    with pytest.raises(xpca.PCAExperimentError) as err:
        xpca.pca_components_setting(15, 20)
    message = str(err.value)
    assert "20" in message and "15" in message
    assert "AE_LATENT_DIM" in message       # it names the two ways out


def test_a_float_outside_the_unit_interval_and_other_nonsense_are_refused():
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(xpca.PCAExperimentError, match="between 0"):
            xpca.pca_components_setting(15, bad)
    with pytest.raises(xpca.PCAExperimentError, match="at least 1"):
        xpca.pca_components_setting(15, 0)
    # None is not nonsense: it means "use config.PCA_N_COMPONENTS"
    for bad in ("0.95", True, [4]):
        with pytest.raises(xpca.PCAExperimentError, match="PCA_N_COMPONENTS"):
            xpca.pca_components_setting(15, bad)


def test_the_setting_defaults_to_the_config_value(monkeypatch):
    monkeypatch.setattr(config, "PCA_N_COMPONENTS", 4)
    assert xpca.pca_components_setting(15) == 4


# ===========================================================================
# fitting and transforming
# ===========================================================================
def test_a_float_setting_keeps_at_least_that_much_variance():
    Z_train = latent_frame(seed=1)

    pca, info = xpca.fit_pca_on_train(Z_train, setting=0.90)
    table = xpca.pca_variance_table(pca)

    assert info["variance_kept"] >= 0.90
    assert float(table["cumulative_variance_ratio"].iloc[-1]) >= 0.90
    assert info["n_components"] <= Z_train.shape[1]
    # and it does not keep more than it needs to
    assert float(table["cumulative_variance_ratio"].iloc[-2]) < 0.90


def test_an_int_setting_keeps_exactly_that_many_components():
    pca, info = xpca.fit_pca_on_train(latent_frame(seed=2), setting=3)

    assert info["n_components"] == 3
    assert info["requested_kind"] == "fixed components"
    assert len(xpca.pca_variance_table(pca)) == 3


def test_the_variance_table_is_sorted_and_cumulative():
    pca, _info = xpca.fit_pca_on_train(latent_frame(seed=3), setting=5)
    table = xpca.pca_variance_table(pca)

    ratio = table["explained_variance_ratio"].to_numpy()
    assert list(ratio) == sorted(ratio, reverse=True)
    assert table["cumulative_variance_ratio"].to_numpy() == \
        pytest.approx(np.cumsum(ratio), abs=1e-9)
    assert list(table["component"]) == ["pc_00", "pc_01", "pc_02", "pc_03",
                                        "pc_04"]


def test_transform_produces_the_promised_columns_and_row_count():
    Z_train = latent_frame(seed=4)
    Z_test = latent_frame(n_rows=30, seed=5)
    pca, info = xpca.fit_pca_on_train(Z_train, setting=4)

    P_test = xpca.transform_with_pca(pca, Z_test)

    assert list(P_test.columns) == ["pc_00", "pc_01", "pc_02", "pc_03"]
    assert len(P_test) == len(Z_test)
    assert info["n_components"] == 4


def test_transforming_the_wrong_width_is_refused():
    pca, _info = xpca.fit_pca_on_train(latent_frame(seed=6), setting=3)
    with pytest.raises(xpca.PCAExperimentError, match="columns but the PCA"):
        xpca.transform_with_pca(pca, latent_frame(n_rows=5, n_latent=4, seed=7))


# ===========================================================================
# the leakage guarantee
# ===========================================================================
def test_changing_the_test_set_does_not_change_the_pca():
    """The whole point: the test rows never reach .fit()."""
    Z_train = latent_frame(seed=8)
    first, _i1 = xpca.fit_pca_on_train(Z_train, setting=4)

    # a completely different "test set" is irrelevant to the fit
    for seed in (9, 10, 11):
        Z_test = latent_frame(n_rows=40, seed=seed) * 100.0
        second, _i2 = xpca.fit_pca_on_train(Z_train, setting=4)
        assert second.components_ == pytest.approx(first.components_)
        assert second.mean_ == pytest.approx(first.mean_)
        # transforming it changes nothing about the fitted object
        xpca.transform_with_pca(second, Z_test)
        assert second.mean_ == pytest.approx(first.mean_)


def test_the_leak_check_passes_on_a_clean_fit_and_has_teeth():
    Z_train = latent_frame(seed=12)
    Z_test = latent_frame(n_rows=40, seed=13) + 50.0
    pca, _info = xpca.fit_pca_on_train(Z_train, setting=4)

    check = xpca.assert_pca_fitted_on_train_only(pca, Z_train, Z_test,
                                                 setting=4)

    assert check["result"] == "[OK]"
    assert "never reached .fit()" in check["detail"]

    # and it FAILS when the PCA really was fitted on train + test
    mixed = pd.concat([Z_train, Z_test], ignore_index=True)
    leaky, _i = xpca.fit_pca_on_train(mixed, setting=4)
    assert xpca.assert_pca_fitted_on_train_only(leaky, Z_train, Z_test,
                                                setting=4)["result"] == "[FAIL]"


# ===========================================================================
# the LSTM-only model and the input shape
# ===========================================================================
def test_the_baseline4_model_is_lstm_only_with_no_convolutions():
    model = xp.build_lstm_only((4, 1), 3)

    kinds = [type(layer).__name__ for layer in model.layers]
    assert "LSTM" in kinds
    assert not any(k.startswith("Conv") for k in kinds)
    assert not any("Pool" in k for k in kinds)
    assert tuple(model.input_shape[1:]) == (4, 1)


def test_the_classifier_input_is_n_components_by_one():
    """PCA output -> arrange_inputs -> (n_components, 1), not (latent, 1)."""
    from src.mscad import models as md

    pca, info = xpca.fit_pca_on_train(latent_frame(seed=14), setting=4)
    P = xpca.transform_with_pca(pca, latent_frame(n_rows=20, seed=15))
    y = pd.DataFrame({"Label_encoded": np.resize([0, 1, 2], 20),
                      "Label_binary": np.resize([0, 1, 1], 20)})

    arranged, _labels, _index = md.arrange_inputs(P, y, None, verbose=False)

    assert arranged.shape == (20, info["n_components"], 1)
    assert info["n_components"] == 4


def test_the_architecture_table_lists_five_configurations():
    table = xpca.architecture_table()

    assert len(table) == len(config.EXPERIMENTS_WITH_PCA) == 5
    assert list(table.columns) == ["config", "feature selection", "AE", "PCA",
                                   "CNN", "LSTM", "input shape"]
    row = table[table["config"] == config.BASELINE4["label"]].iloc[0]
    assert row["PCA"].startswith("yes")
    assert row["CNN"] == "no" and row["LSTM"] == "yes"
    assert row["input shape"] == "(n_components, 1)"
    # exactly one configuration uses PCA
    assert int((table["PCA"].str.startswith("yes")).sum()) == 1


# ===========================================================================
# the configuration list
# ===========================================================================
def test_step_12s_own_experiment_list_is_untouched():
    """Steps 13 and 14 address these four by name, so they must not move."""
    names_only = [e["name"] for e in config.EXPERIMENTS]

    assert len(config.EXPERIMENTS) == 4
    assert names_only == ["proposed", "baseline1_ae_cnn_lstm",
                          "baseline2_ae_lstm", "baseline3_cnn_lstm"]
    assert all(name in names_only for name in config.FUSION_MODELS)
    assert not any(e.get("use_pca") for e in config.EXPERIMENTS)


def test_the_five_configuration_list_adds_baseline4_and_nothing_else():
    assert config.EXPERIMENTS_WITH_PCA[:4] == config.EXPERIMENTS
    assert config.EXPERIMENTS_WITH_PCA[-1] is config.BASELINE4
    assert config.BASELINE4["use_pca"] is True
    assert config.BASELINE4["model_type"] == "lstm"
    assert config.BASELINE4["name"] not in [e["name"] for e in config.EXPERIMENTS]


def test_the_baseline4_config_block_holds_the_documented_defaults():
    assert config.BASELINE4_AE_FEATURES in ("all", "selected30")
    assert isinstance(config.PCA_N_COMPONENTS, (int, float))
    assert isinstance(config.OVERWRITE_BASELINE4, bool)
    for key in ("comparison", "per_class_full", "summary", "report",
                "chart_metrics", "chart_confusion"):
        assert key in config.STEP12B_FILES
    # the addendum must never write over a Step 12 file
    assert not (set(config.STEP12B_FILES.values())
                & set(config.STEP12_FILES.values()))


# ===========================================================================
# the comparison table
# ===========================================================================
def fake_run(name, label, pca_components=None, latent=6, f1=0.5):
    metrics = {
        "config": name, "label": label, "n_input_features": 9,
        "latent_dim": latent, "parameters": 1234,
        "full": {"accuracy": 0.8, "precision_macro": 0.7, "recall_macro": 0.6,
                 "f1_macro": f1, "f1_weighted": 0.75, "g_mean": 0.5,
                 "roc_auc": {"macro": 0.9}},
        "duplicate_free": {"f1_macro": f1 - 0.1, "g_mean": 0.4},
        "binary": {"f1": 0.9, "false_positive_rate": 0.02},
        "inference": {"ms_per_1000_flows": 12.0},
        "training_seconds": 3.0,
    }
    if pca_components is not None:
        metrics["pca_components"] = pca_components
    return {"metrics": metrics, "folder": "/nowhere"}


def test_the_comparison_table_has_one_row_per_configuration_and_a_pca_column():
    runs = [fake_run("proposed", "Proposed", f1=0.50),
            fake_run("baseline1_ae_cnn_lstm", "B1", f1=0.40),
            fake_run("baseline2_ae_lstm", "B2", f1=0.30),
            fake_run("baseline3_cnn_lstm", "B3", latent=None, f1=0.20),
            fake_run("baseline4_ae_pca_lstm", "B4", pca_components=4, f1=0.60)]

    table = xpca.combined_comparison(runs)

    assert len(table) == 5
    assert "PCA components" in table.columns
    assert list(table.columns).index("PCA components") == \
        list(table.columns).index("latent dim") + 1
    assert list(table["PCA components"]) == ["-", "-", "-", "-", "4 of 6"]
    for column in ("accuracy", "precision (macro)", "recall (macro)",
                   "F1 (macro)", "F1 (weighted)", "G-mean",
                   "ROC-AUC (macro OvR)", "binary F1", "binary FPR",
                   "F1 macro (dup-free)", "G-mean (dup-free)",
                   "train time (s)", "inference (ms/1k)", "params"):
        assert column in table.columns


def test_no_configuration_is_called_best_on_the_pca_column():
    runs = [fake_run("a", "A", f1=0.5), fake_run("b", "B", pca_components=4,
                                                 f1=0.6)]
    marked = xpca.mark_best(xpca.combined_comparison(runs))

    assert "*" not in "".join(str(v) for v in marked["PCA components"])
    assert "0.6000*" in str(marked["F1 (macro)"].iloc[1])


# ===========================================================================
# the summary
# ===========================================================================
def test_the_summary_measures_pca_against_baseline_2_only():
    runs = [fake_run("baseline2_ae_lstm", "B2: AE -> LSTM (no CNN)", f1=0.30),
            fake_run("baseline4_ae_pca_lstm", "B4: AE -> PCA -> LSTM",
                     pca_components=4, f1=0.45)]
    baseline4 = {"metrics": runs[-1]["metrics"] | {
        "pca": {"n_components": 4, "latent_dim": 6, "requested": 0.95,
                "requested_kind": "variance fraction", "variance_kept": 0.96,
                "variance_per_component": [0.5, 0.3, 0.1, 0.06]},
        "pipeline_inference": {"encoder_ms_per_1000_flows": 5.0,
                               "pca_transform_ms_per_1000_flows": 1.0,
                               "classifier_ms_per_1000_flows": 12.0,
                               "end_to_end_ms_per_1000_flows": 18.0}}}
    table = xpca.combined_comparison(runs)

    text = xpca.build_summary(table, pd.DataFrame(), baseline4, runs)

    assert "+0.1500" in text                      # 0.45 - 0.30, stated as a delta
    assert "above" in text
    assert "4 of 6" in text or "**4 of 6**" in text or "4** of **6" in text \
        or "kept **4 of 6**" in text
    assert "95" in text or "96" in text           # the variance kept
    assert "identical apart from the PCA step" in text
    assert "no significance is claimed" in text


def test_the_summary_reports_a_loss_as_a_loss():
    runs = [fake_run("baseline2_ae_lstm", "B2: AE -> LSTM (no CNN)", f1=0.60),
            fake_run("baseline4_ae_pca_lstm", "B4: AE -> PCA -> LSTM",
                     pca_components=4, f1=0.40)]
    baseline4 = {"metrics": runs[-1]["metrics"]}
    text = xpca.build_summary(xpca.combined_comparison(runs), pd.DataFrame(),
                              baseline4, runs)

    assert "below" in text
    assert "-0.2000" in text


def test_a_tie_in_the_summary_is_called_a_tie():
    runs = [fake_run("a", "A", f1=0.5), fake_run("b", "B", f1=0.5)]
    text = xpca.build_summary(xpca.combined_comparison(runs), pd.DataFrame(),
                              {"metrics": {}}, runs)
    assert "(a tie)" in text


# ===========================================================================
# skipping finished work
# ===========================================================================
def test_a_finished_baseline4_is_skipped_when_overwrite_is_off(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    folder = tmp_path / "step12" / config.BASELINE4["name"]
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(json.dumps(
        {"full": {"f1_macro": 0.4242}, "label": "B4"}), encoding="utf-8")
    monkeypatch.setattr(config, "OVERWRITE_BASELINE4", False)

    assert xpca.baseline4_is_done() is True
    out = xpca.run_baseline4(pd.DataFrame(), pd.DataFrame(), None, {},
                             verbose=True)

    assert out["status"] == "loaded_existing"
    assert out["metrics"]["full"]["f1_macro"] == pytest.approx(0.4242)
    assert "[SKIP]" in capsys.readouterr().out


def test_a_missing_step_12_configuration_is_named_rather_than_guessed(tmp_path,
                                                                      monkeypatch):
    (tmp_path / "step12").mkdir(parents=True)
    with pytest.raises(xpca.PCAExperimentError) as err:
        xpca.load_step12_runs(verbose=False)

    message = str(err.value)
    assert "proposed" in message and "metrics.json" in message
    assert "Run Step 12 first" in message


# ===========================================================================
# the encoder is reused, never retrained
# ===========================================================================
def test_baseline4_reuses_the_baseline1_encoder_weights(tmp_path, monkeypatch,
                                                        capsys):
    """Same weights, loaded from Baseline 1's folder -- not a new autoencoder."""
    monkeypatch.setattr(config, "BASELINE4_AE_FEATURES", "all")
    folder1 = tmp_path / "step12" / "baseline1_ae_cnn_lstm"
    folder1.mkdir(parents=True)

    keras.utils.set_random_seed(0)
    encoder = keras.Sequential([keras.Input(shape=(N_ALL,)),
                                keras.layers.Dense(6, activation="sigmoid")])
    encoder.predict(np.zeros((1, N_ALL), dtype="float32"), verbose=0)
    encoder.save(folder1 / "encoder.keras")
    (folder1 / "ae_config.json").write_text(
        json.dumps({"features": names(N_ALL)}), encoding="utf-8")
    original = [w.copy() for w in encoder.get_weights()]

    X = pd.DataFrame(np.random.default_rng(0).uniform(0, 1, (10, N_ALL)),
                     columns=names(N_ALL))
    encode, loaded, order, provenance = xpca.reused_encoder(
        X, X, pd.DataFrame(), tmp_path / "step12" / "baseline4", verbose=True)

    assert provenance["trained_here"] is False
    assert provenance["source"] == "baseline1_ae_cnn_lstm"
    assert order == names(N_ALL)
    for before, after in zip(original, loaded.get_weights()):
        assert after == pytest.approx(before)
    assert "reused from Baseline 1" in capsys.readouterr().out
    assert encode(X, what="x").shape == (10, 6)


def test_an_unknown_ae_feature_source_is_refused(monkeypatch):
    monkeypatch.setattr(config, "BASELINE4_AE_FEATURES", "top10")
    with pytest.raises(xpca.PCAExperimentError, match="BASELINE4_AE_FEATURES"):
        xpca.reused_encoder(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                            "/nowhere", verbose=False)


def test_a_missing_step7_encoder_is_named_rather_than_retrained(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(config, "BASELINE4_AE_FEATURES", "selected30")
    with pytest.raises(xpca.PCAExperimentError) as err:
        xpca.reused_encoder(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                            tmp_path, verbose=False)
    assert "Run Step 7 first" in str(err.value)
    assert "never trains its own" in str(err.value)


# ===========================================================================
# the Windows-path guard
# ===========================================================================
def test_save_refuses_a_windows_path_rather_than_making_junk(
        tmp_path, monkeypatch, capsys, assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step12"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    saved = xpca.save_step12b({"comparison": pd.DataFrame()}, verbose=True)

    assert saved["status"] == "skipped_not_windows"
    assert saved["ok"] is False
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("save_step12b")


def test_the_loaders_refuse_a_windows_path(tmp_path, monkeypatch,
                                           assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step12"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    assert xpca.step12b_outputs_exist() is False
    assert xpca.baseline1_folder() is None
    with pytest.raises(xpca.PCAExperimentError, match="not usable here"):
        xpca.load_step12b(verbose=False)
    with pytest.raises(xpca.PCAExperimentError, match="not usable here"):
        xpca.load_step12_runs(verbose=False)
    assert_no_leaked_output("the Baseline 4 loaders")
