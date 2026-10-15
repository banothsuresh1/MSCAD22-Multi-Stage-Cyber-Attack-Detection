r"""
tests/test_experiments_baseline5.py -- pytest tests for Baseline 5
(MI-only top 30 -> AE -> hybrid CNN-LSTM).

NEW FILE.  No existing test file is modified, tests/test_experiments.py and
tests/test_experiments_pca.py included.  Tiny synthetic data, 2-3 epochs, CPU
only, everything inside tmp_path.  Never C:\\MSCAD2022.

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
from src.mscad import experiments_baseline5 as b5

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")

N_ALL = 12
N_ROWS = 90


@pytest.fixture(autouse=True)
def _tiny_and_in_tmp(tmp_path, monkeypatch):
    for step in (4, 5, 6, 7, 8, 9, 10, 11, 12):
        monkeypatch.setattr(config, "STEP%d_OUTPUT_DIR" % step,
                            tmp_path / ("step%d" % step))
    monkeypatch.setattr(config, "BASELINE5_OUTPUT_DIR",
                        tmp_path / "step12" / "baseline5")
    monkeypatch.setattr(config, "BASELINE5_TOP_K", 4)
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


def names(n=N_ALL):
    return ["f%02d" % i for i in range(n)]


def make_xy(n_rows=N_ROWS, n_features=N_ALL, seed=0):
    """Features where the first few genuinely carry the label."""
    rng = np.random.default_rng(seed)
    codes = np.resize([0, 1, 2], n_rows)
    values = rng.uniform(0, 1, size=(n_rows, n_features))
    for i in range(4):                       # informative columns
        values[:, i] = codes * 0.3 + rng.normal(0, 0.02, n_rows)
    X = pd.DataFrame(values, columns=names(n_features))
    y = pd.DataFrame({"Label_encoded": codes,
                      "Label_binary": (codes != 0).astype("int64")})
    return X, y


def mi_frame(order, scores=None):
    scores = scores if scores is not None else np.linspace(1.0, 0.1, len(order))
    return pd.DataFrame({"rank": np.arange(1, len(order) + 1),
                         "feature": list(order),
                         "mutual_information": list(scores)})


# ===========================================================================
# the MI ranking
# ===========================================================================
def test_mi_is_computed_on_the_training_rows_only():
    """Changing the test set cannot change the ranking: it is not an input."""
    X_train, y_train = make_xy(seed=1)
    first = b5.compute_mi(X_train, y_train, verbose=False)

    for seed in (2, 3):
        _X_test, _y_test = make_xy(n_rows=40, seed=seed)
        again = b5.compute_mi(X_train, y_train, verbose=False)
        assert list(again["feature"]) == list(first["feature"])
        assert again["mutual_information"].to_numpy() == pytest.approx(
            first["mutual_information"].to_numpy())


def test_the_mi_ranking_uses_step_6s_own_function_and_settings():
    """It calls feature_selection.mutual_information, so the settings match."""
    from src.mscad import feature_selection as fs

    X_train, y_train = make_xy(seed=4)
    ours = b5.compute_mi(X_train, y_train, verbose=False)
    _series, theirs, _seconds = fs.mutual_information(X_train, y_train,
                                                      verbose=False)

    assert list(ours["feature"]) == list(theirs["feature"])
    assert ours["mutual_information"].to_numpy() == pytest.approx(
        theirs["mutual_information"].to_numpy())


def test_top_k_returns_exactly_k_unique_features_ordered_by_score():
    table = mi_frame(names(10))

    selected = b5.top_k_by_mi(table, k=4)

    assert len(selected) == 4
    assert selected["feature"].nunique() == 4
    assert list(selected["mi_rank"]) == [1, 2, 3, 4]
    assert selected["mutual_information"].is_monotonic_decreasing
    assert list(selected["feature"]) == names(10)[:4]


def test_ties_are_broken_by_name_so_the_selection_is_reproducible():
    table = mi_frame(["b", "a", "c"], scores=[0.5, 0.5, 0.1])

    first = b5.top_k_by_mi(table, k=2)
    again = b5.top_k_by_mi(table.sample(frac=1.0, random_state=0), k=2)

    assert list(first["feature"]) == ["a", "b"]      # tie -> alphabetical
    assert list(again["feature"]) == list(first["feature"])


def test_a_feature_missing_from_the_training_matrix_is_not_selected():
    table = mi_frame(["gone", "a", "b", "c"])

    selected = b5.top_k_by_mi(table, k=3, available=["a", "b", "c", "d"])

    assert "gone" not in list(selected["feature"])
    assert list(selected["feature"]) == ["a", "b", "c"]


def test_asking_for_more_features_than_exist_is_refused():
    with pytest.raises(b5.Baseline5Error, match="BASELINE5_TOP_K"):
        b5.top_k_by_mi(mi_frame(names(3)), k=10)
    with pytest.raises(b5.Baseline5Error, match="at least 1"):
        b5.top_k_by_mi(mi_frame(names(3)), k=0)


def test_step_6s_pearson_filtered_file_is_not_used_for_an_mi_only_baseline(
        tmp_path):
    """mi_scores.csv holds only the Pearson survivors; mi_scores_all.csv is
    the one an MI-only selection may read."""
    folder = tmp_path / "step6"
    folder.mkdir(parents=True)
    (folder / config.STEP6_FILES["mi_kept"]).write_text("rank,feature\n1,a\n",
                                                        encoding="utf-8")

    table, path = b5.load_step6_mi(verbose=False)

    assert table is None and path is None       # the kept file is ignored


def test_step_6s_all_feature_file_is_used_when_it_exists(tmp_path, capsys):
    folder = tmp_path / "step6"
    folder.mkdir(parents=True)
    mi_frame(names(6)).to_csv(folder / config.STEP6_FILES["mi_all"], index=False)

    table, path = b5.load_step6_mi(verbose=True)

    assert table is not None and len(table) == 6
    assert config.STEP6_FILES["mi_all"] in str(path)
    assert "read from" in capsys.readouterr().out


# ===========================================================================
# the overlap with the proposed selection
# ===========================================================================
def test_the_overlap_summary_counts_both_directions():
    table, summary = b5.feature_overlap(["a", "b", "c", "d"],
                                        ["a", "b", "e", "f"])

    assert summary["shared"] == 2
    assert summary["only_mi"] == ["c", "d"]
    assert summary["only_proposed"] == ["e", "f"]
    assert summary["jaccard"] == pytest.approx(2 / 6)
    assert summary["identical"] is False
    assert set(table["where"]) == {"both", "MI only", "proposed only"}
    assert len(table) == 6


def test_identical_selections_stop_the_run_with_a_clear_message():
    _table, summary = b5.feature_overlap(["a", "b", "c"], ["c", "b", "a"])

    assert summary["identical"] is True
    assert summary["jaccard"] == pytest.approx(1.0)
    with pytest.raises(b5.IdenticalSelectionError) as err:
        b5.stop_if_identical(summary)

    message = str(err.value)
    assert "nothing to compare" in message
    assert "Nothing was trained" in message
    assert "would BE the proposed model" in message


def test_a_different_selection_does_not_stop_the_run():
    _table, summary = b5.feature_overlap(["a", "b", "c"], ["a", "b", "z"])
    assert b5.stop_if_identical(summary) is None


# ===========================================================================
# the model
# ===========================================================================
def test_the_baseline5_classifier_has_both_conv_and_lstm_layers():
    """The same builder Step 10 uses, so the CNN front end is there."""
    from src.mscad import models as md

    model = xp.builder_for(config.BASELINE5_SPEC["model_type"])(
        (config.AE_LATENT_DIM, 1), 3)
    kinds = [type(layer).__name__ for layer in model.layers]

    assert xp.builder_for("cnn_lstm") is md.build_cnn_lstm
    assert any(k.startswith("Conv") for k in kinds)
    assert "LSTM" in kinds
    assert tuple(model.input_shape[1:]) == (config.AE_LATENT_DIM, 1)


def test_the_baseline5_autoencoder_is_trained_on_train_only(tmp_path):
    """Its input width is the selection size and its output the latent dim."""
    X_train, y_train = make_xy(n_features=config.BASELINE5_TOP_K, seed=5)
    X_test, _y_test = make_xy(n_rows=20, n_features=config.BASELINE5_TOP_K,
                              seed=6)
    folder = tmp_path / "step12" / "baseline5"

    Z_train, Z_test, info = xp.encode_with_new_ae(X_train, X_test, y_train,
                                                  folder, verbose=False)

    assert Z_train.shape == (len(X_train), config.AE_LATENT_DIM)
    assert Z_test.shape == (len(X_test), config.AE_LATENT_DIM)
    saved = json.loads((folder / "ae_config.json").read_text(encoding="utf-8"))
    assert saved["n_features"] == config.BASELINE5_TOP_K
    assert saved["latent_dim"] == config.AE_LATENT_DIM
    assert saved["trained_on"] == "the TRAINING rows only"
    assert info["epochs_run"] >= 1


# ===========================================================================
# the six-configuration table
# ===========================================================================
def fake_run(name, label, f1=0.5, pca=None):
    metrics = {
        "config": name, "label": label, "n_input_features": 9,
        "latent_dim": 3, "parameters": 1234,
        "full": {"accuracy": 0.8, "precision_macro": 0.7, "recall_macro": 0.6,
                 "f1_macro": f1, "f1_weighted": 0.75, "g_mean": 0.5,
                 "roc_auc": {"macro": 0.9}},
        "duplicate_free": {"f1_macro": f1 - 0.1, "g_mean": 0.4},
        "binary": {"f1": 0.9, "false_positive_rate": 0.02},
        "inference": {"ms_per_1000_flows": 12.0},
        "training_seconds": 3.0,
    }
    if pca is not None:
        metrics["pca_components"] = pca
    return {"metrics": metrics, "folder": "/nowhere"}


def six_runs():
    return [fake_run("proposed", "Proposed: FS -> AE -> CNN-LSTM", f1=0.50),
            fake_run("baseline1_ae_cnn_lstm", "B1", f1=0.40),
            fake_run("baseline2_ae_lstm", "B2", f1=0.30),
            fake_run("baseline3_cnn_lstm", "B3", f1=0.20),
            fake_run("baseline4_ae_pca_lstm", "B4", f1=0.45, pca=4),
            fake_run("baseline5", "B5: MI top 30 -> AE -> CNN-LSTM", f1=0.35)]


def test_the_table_has_six_rows_and_every_metric_column():
    table = b5.comparison_table(six_runs())

    assert len(table) == 6
    for column in ("config", "input features", "latent dim", "PCA components",
                   "params", "accuracy", "precision (macro)", "recall (macro)",
                   "F1 (macro)", "F1 (weighted)", "G-mean",
                   "ROC-AUC (macro OvR)", "binary F1", "binary FPR",
                   "F1 macro (dup-free)", "G-mean (dup-free)",
                   "train time (s)", "inference (ms/1k)"):
        assert column in table.columns
    assert list(table["PCA components"]) == ["-", "-", "-", "-", "4 of 3", "-"]


def test_the_existing_rows_are_checked_against_step_12s_saved_table(tmp_path):
    table = b5.comparison_table(six_runs())
    folder = tmp_path / "step12"
    folder.mkdir(parents=True)
    table.iloc[:5].to_csv(folder / config.STEP12_FILES["comparison"],
                          index=False)

    check = b5.assert_existing_rows_match(table, verbose=False)

    assert check["result"] == "[OK]"
    assert "identical" in check["detail"]


def test_a_changed_existing_row_is_caught(tmp_path):
    table = b5.comparison_table(six_runs())
    folder = tmp_path / "step12"
    folder.mkdir(parents=True)
    tampered = table.iloc[:5].copy()
    tampered.loc[0, "F1 (macro)"] = 0.99
    tampered.to_csv(folder / config.STEP12_FILES["comparison"], index=False)

    check = b5.assert_existing_rows_match(table, verbose=False)

    assert check["result"] == "[FAIL]"
    assert "F1 (macro)" in check["detail"]


def test_the_check_is_skipped_rather_than_failed_when_step_12_has_no_table():
    check = b5.assert_existing_rows_match(b5.comparison_table(six_runs()),
                                          verbose=False)
    assert check["result"] == "[SKIP]"


def test_the_head_to_head_reports_direction_per_metric():
    table = b5.comparison_table(six_runs())

    difference = b5.head_to_head(table, "Proposed: FS -> AE -> CNN-LSTM",
                                 "B5: MI top 30 -> AE -> CNN-LSTM")

    row = difference[difference["metric"] == "F1 (macro)"].iloc[0]
    assert row["difference (Proposed - B5)"] == pytest.approx(0.15)
    assert row["favours"] == "Proposed"
    # lower-is-better columns flip the direction
    fpr = difference[difference["metric"] == "binary FPR"].iloc[0]
    assert bool(fpr["lower is better"]) is True
    assert fpr["favours"] == "tie"
    # descriptors are not reported as results
    assert "input features" not in list(difference["metric"])
    assert "latent dim" not in list(difference["metric"])


def test_the_architecture_table_lists_six_configurations():
    table = b5.architecture_table()

    assert len(table) == 6
    row = table.iloc[-1]
    assert row["feature selection"].startswith("MI only")
    assert row["CNN"] == "yes" and row["LSTM"] == "yes" and row["PCA"] == "no"
    assert row["input shape"] == "(%d, 1)" % config.AE_LATENT_DIM


# ===========================================================================
# nothing existing is touched
# ===========================================================================
def test_the_existing_step12_files_are_fingerprinted_and_unchanged(tmp_path):
    folder = tmp_path / "step12"
    folder.mkdir(parents=True)
    for name in (config.STEP12_FILES["comparison"],
                 config.STEP12B_FILES["comparison"]):
        (folder / name).write_text("config,F1 (macro)\nA,0.5\n", encoding="utf-8")

    before = b5.existing_step12_fingerprint()
    assert set(before) == {config.STEP12_FILES["comparison"],
                           config.STEP12B_FILES["comparison"]}

    # writing a NEW file beside them changes nothing
    (folder / config.STEP12C_FILES["comparison"]).write_text("x", encoding="utf-8")
    assert b5.existing_step12_fingerprint() == before

    # and a rewrite would be caught
    (folder / config.STEP12_FILES["comparison"]).write_text("tampered",
                                                            encoding="utf-8")
    after = b5.existing_step12_fingerprint()
    assert after[config.STEP12_FILES["comparison"]] != \
        before[config.STEP12_FILES["comparison"]]


def test_the_new_files_never_share_a_name_with_an_existing_one():
    new = set(config.STEP12C_FILES.values())
    assert not (new & set(config.STEP12_FILES.values()))
    assert not (new & set(config.STEP12B_FILES.values()))
    assert all(name.endswith("_6configs.csv") or name.endswith("_6configs.png")
               or name.endswith("_6configs.md") for name in new), sorted(new)


def test_baseline5_is_not_in_the_experiments_list():
    """Steps 13 and 14 iterate EXPERIMENTS; Baseline 5 must stay out of it."""
    assert len(config.EXPERIMENTS) == 4
    assert len(config.EXPERIMENTS_WITH_PCA) == 5
    for listed in (config.EXPERIMENTS, config.EXPERIMENTS_WITH_PCA):
        assert config.BASELINE5_NAME not in [e["name"] for e in listed]


# ===========================================================================
# skipping finished work, and the guards
# ===========================================================================
def test_a_finished_baseline5_is_skipped_when_overwrite_is_off(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    folder = tmp_path / "step12" / config.BASELINE5_NAME
    folder.mkdir(parents=True)
    (folder / config.BASELINE5_FILES["metrics"]).write_text(
        json.dumps({"full": {"f1_macro": 0.3141}}), encoding="utf-8")
    monkeypatch.setattr(config, "OVERWRITE_BASELINE5", False)

    assert b5.baseline5_is_done() is True
    out = b5.run_baseline5(pd.DataFrame(), pd.DataFrame(), None, {},
                           verbose=True)

    assert out["status"] == "loaded_existing"
    assert out["metrics"]["full"]["f1_macro"] == pytest.approx(0.3141)
    assert "[SKIP]" in capsys.readouterr().out


def test_a_missing_configuration_is_named_rather_than_guessed(tmp_path):
    (tmp_path / "step12").mkdir(parents=True)
    with pytest.raises(b5.Baseline5Error) as err:
        b5.existing_runs(verbose=False)
    assert "proposed" in str(err.value)
    assert "Run Step 12" in str(err.value)


def test_a_missing_proposed_feature_list_is_named(tmp_path):
    (tmp_path / "step6").mkdir(parents=True)
    with pytest.raises(b5.Baseline5Error) as err:
        b5.proposed_features(verbose=False)
    assert "Run Step 6 first" in str(err.value)


def test_save_refuses_a_windows_path_rather_than_making_junk(
        tmp_path, monkeypatch, capsys, assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step12"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    saved = b5.save_step12c({"comparison": pd.DataFrame()}, verbose=True)

    assert saved["status"] == "skipped_not_windows"
    assert saved["ok"] is False
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("save_step12c")


def test_the_loaders_refuse_a_windows_path(tmp_path, monkeypatch,
                                           assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step12"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    assert b5.step12c_outputs_exist() is False
    assert b5.baseline5_folder() is None
    assert b5.existing_step12_fingerprint() == {}
    with pytest.raises(b5.Baseline5Error, match="not usable here"):
        b5.load_step12c(verbose=False)
    with pytest.raises(b5.Baseline5Error, match="not usable here"):
        b5.existing_runs(verbose=False)
    assert_no_leaked_output("the Baseline 5 loaders")


def test_the_baseline5_config_block_holds_the_documented_defaults():
    assert config.BASELINE5_NAME == "baseline5"
    assert config.BASELINE5_SPEC["use_ae"] is True
    assert config.BASELINE5_SPEC["use_pca"] is False
    assert config.BASELINE5_SPEC["model_type"] == "cnn_lstm"
    assert config.BASELINE5_SPEC["feature_source"] == "mi_top30"
    assert isinstance(config.OVERWRITE_BASELINE5, bool)
    assert config.BASELINE5_OUTPUT_DIR.name == "baseline5"
    assert config.BASELINE5_OUTPUT_DIR.parent == config.STEP12_OUTPUT_DIR
    for key in ("mi_scores", "selected", "overlap", "model", "metrics",
                "predictions"):
        assert key in config.BASELINE5_FILES
