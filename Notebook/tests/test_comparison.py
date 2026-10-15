r"""
tests/test_comparison.py -- pytest tests for the Step 12 full comparison.

NEW FILE.  No existing test file is modified.  Everything runs on small
hand-made metrics.json files inside tmp_path; no model is built and nothing is
trained.  Never C:\\MSCAD2022.

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

LABELS = {0: "Normal", 1: "Port_Scan", 2: "Brute_Force"}

# captured before any fixture empties it, so the defaults test sees the real
# block from config.py rather than the per-test override
EXPECTED_ROWS_AS_SHIPPED = dict(config.STEP12_EXPECTED_ROWS)


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    for step in (10, 11, 12):
        monkeypatch.setattr(config, "STEP%d_OUTPUT_DIR" % step,
                            tmp_path / ("step%d" % step))
    monkeypatch.setattr(config, "STEP11B_OUTPUT_DIR",
                        tmp_path / "step11_attention", raising=False)
    monkeypatch.setattr(config, "STEP12_EXPECTED_ROWS", {})


def metrics(label, accuracy=0.9, f1=0.5, g_mean=0.4, auc=0.9, fpr=0.02,
            params=1000, train=10.0, inference=50.0, latent=15, pca=None,
            features=30, free_f1=None):
    out = {
        "config": label, "label": label, "n_input_features": features,
        "latent_dim": latent, "parameters": params, "training_seconds": train,
        "full": {"accuracy": accuracy, "precision_macro": 0.6,
                 "recall_macro": 0.55, "f1_macro": f1, "f1_weighted": 0.8,
                 "g_mean": g_mean, "roc_auc": {"macro": auc}},
        "duplicate_free": {"f1_macro": f1 if free_f1 is None else free_f1,
                           "g_mean": g_mean},
        "binary": {"f1": 0.95, "false_positive_rate": fpr},
        "inference": {"ms_per_1000_flows": inference},
    }
    if pca is not None:
        out["pca_components"] = pca
    return out


def write_result(tmp_path, key, payload, predictions=None):
    folder = (tmp_path / "step11_attention" if key == config.APPROACH2_KEY
              else tmp_path / "step12" / key)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    if predictions is not None:
        predictions.to_csv(folder / "predictions.csv", index=False)
    return folder


def predictions_frame(y_true, y_pred, duplicates=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return pd.DataFrame({
        "y_true": y_true, "y_pred": y_pred,
        "y_true_label": [LABELS[int(c)] for c in y_true],
        "y_pred_label": [LABELS[int(c)] for c in y_pred],
        "has_train_duplicate": (np.zeros(len(y_true), dtype=bool)
                                if duplicates is None else duplicates)})


def write_all(tmp_path, include_approach2=False, **overrides):
    for i, (key, label) in enumerate(config.STEP12_DISPLAY_ORDER):
        if key == config.APPROACH2_KEY and not include_approach2:
            continue
        payload = overrides.get(key) or metrics(label, f1=0.5 + 0.01 * i)
        write_result(tmp_path, key, payload)


# ===========================================================================
# build_full_comparison
# ===========================================================================
def test_every_row_is_present_in_the_display_order(tmp_path):
    write_all(tmp_path, include_approach2=True)

    built = cmp.build_full_comparison(verbose=False)

    assert list(built["table"]["config"]) == [l for _k, l
                                              in config.STEP12_DISPLAY_ORDER]
    assert len(built["table"]) == 7
    assert built["missing"] == []
    for column in cmp.COLUMNS:
        assert column in built["table"].columns


def test_approach_2_is_always_row_2(tmp_path):
    """With or without its result, the order is the same."""
    write_all(tmp_path, include_approach2=False)
    without = cmp.build_full_comparison(verbose=False)["table"]
    write_all(tmp_path, include_approach2=True)
    with_it = cmp.build_full_comparison(verbose=False)["table"]

    label = dict(config.STEP12_DISPLAY_ORDER)[config.APPROACH2_KEY]
    assert without.iloc[1]["config"] == label
    assert with_it.iloc[1]["config"] == label
    assert list(without["config"]) == list(with_it["config"])


def test_an_untrained_configuration_is_blank_and_named(tmp_path, capsys):
    write_all(tmp_path, include_approach2=False)

    built = cmp.build_full_comparison(verbose=True)

    assert built["missing"] == [config.APPROACH2_KEY]
    row = built["table"].iloc[1]
    assert row["input features"] == config.NOT_TRAINED_TEXT
    assert pd.isna(row["F1 (macro)"])
    assert "10.2" in cmp.which_step_produces(config.APPROACH2_KEY)


def test_building_the_table_never_trains(tmp_path, monkeypatch):
    """There is no training entry point here at all: only reads."""
    write_all(tmp_path, include_approach2=True)

    def explode(*args, **kwargs):           # any model build would hit this
        raise AssertionError("the comparison must not build or train a model")

    monkeypatch.setattr("src.mscad.models.build_cnn_lstm", explode)
    monkeypatch.setattr("src.mscad.experiments.train_classifier", explode)

    built = cmp.build_full_comparison(verbose=False)
    assert len(built["metrics"]) == 7


def test_a_missing_result_names_the_step_that_produces_it():
    assert "Step 12" in cmp.which_step_produces("baseline1_ae_cnn_lstm")
    assert "12.3" in cmp.which_step_produces("baseline4_ae_pca_lstm")
    assert "12.4" in cmp.which_step_produces("baseline5")
    assert "10.2" in cmp.which_step_produces(config.APPROACH2_KEY)
    assert "Step 10" in cmp.which_step_produces("proposed")


# ===========================================================================
# the values must not move
# ===========================================================================
def test_the_rebuilt_rows_equal_the_saved_metrics(tmp_path):
    write_all(tmp_path, include_approach2=True)

    built = cmp.build_full_comparison(verbose=False)
    frame, ok = cmp.check_rows_unchanged(built, verbose=False)

    assert ok is True
    assert set(frame["result"]) <= {"[OK]", "[SKIP]"}
    exact = frame[frame["check"].str.startswith("exact")]
    assert len(exact) == 7
    assert (exact["result"] == "[OK]").all()


def test_the_published_figures_are_checked_and_can_fail(tmp_path, monkeypatch):
    write_all(tmp_path, include_approach2=True,
              proposed=metrics("Approach 1: FS -> AE -> CNN-LSTM",
                               accuracy=0.9896, f1=0.7170, g_mean=0.7505))
    monkeypatch.setattr(config, "STEP12_EXPECTED_ROWS", {
        "proposed": {"accuracy": 0.9896, "f1_macro": 0.7170,
                     "g_mean": 0.7505}})

    built = cmp.build_full_comparison(verbose=False)
    frame, ok = cmp.check_rows_unchanged(built, verbose=False)
    assert ok is True

    # a drifted number is caught
    monkeypatch.setattr(config, "STEP12_EXPECTED_ROWS", {
        "proposed": {"accuracy": 0.9896, "f1_macro": 0.6000,
                     "g_mean": 0.7505}})
    frame, ok = cmp.check_rows_unchanged(built, verbose=False)
    assert ok is False
    assert "[FAIL]" in set(frame["result"])


def test_a_configuration_that_was_never_run_is_skipped_not_passed(tmp_path,
                                                                  monkeypatch):
    write_all(tmp_path, include_approach2=False)
    monkeypatch.setattr(config, "STEP12_EXPECTED_ROWS",
                        {config.APPROACH2_KEY: {"accuracy": 0.99}})

    built = cmp.build_full_comparison(verbose=False)
    frame, ok = cmp.check_rows_unchanged(built, verbose=False)

    skipped = frame[frame["result"] == "[SKIP]"]
    assert len(skipped) >= 1
    assert ok is True                      # skipped, never silently passed


# ===========================================================================
# labels and internal keys
# ===========================================================================
def test_display_labels_map_to_unchanged_internal_keys():
    keys = [k for k, _l in config.STEP12_DISPLAY_ORDER]

    for spec in config.EXPERIMENTS:
        assert spec["name"] in keys
    assert keys[0] == "proposed"
    assert keys[1] == config.APPROACH2_KEY
    assert "baseline5" in keys and "baseline4_ae_pca_lstm" in keys
    # the display labels are new; the internal keys are not
    assert dict(config.STEP12_DISPLAY_ORDER)["proposed"].startswith("Approach 1")


def test_fusion_models_are_still_valid_after_the_relabelling():
    known = [e["name"] for e in config.EXPERIMENTS]
    assert all(name in known for name in config.FUSION_MODELS)
    assert config.FUSION_MODELS == ["proposed", "baseline2_ae_lstm",
                                    "baseline3_cnn_lstm"]
    assert len(config.EXPERIMENTS) == 4


def test_the_configuration_table_covers_every_row():
    table = cmp.configuration_table()

    assert len(table) == len(config.STEP12_DISPLAY_ORDER)
    assert list(table["config"]) == [l for _k, l in config.STEP12_DISPLAY_ORDER]
    assert list(table.columns) == ["config", "feature selection", "AE", "PCA",
                                   "CNN", "attention", "LSTM", "input shape",
                                   "what it isolates"]
    assert int((table["attention"] == "yes").sum()) == 1
    assert int((table["PCA"].str.startswith("yes")).sum()) == 1
    assert table["what it isolates"].str.len().min() > 0


# ===========================================================================
# best-value marking
# ===========================================================================
def test_the_best_value_is_marked_and_direction_is_respected(tmp_path):
    write_all(tmp_path, include_approach2=True,
              proposed=metrics("Approach 1: FS -> AE -> CNN-LSTM", f1=0.90,
                               fpr=0.50, params=9000),
              baseline2_ae_lstm=metrics("B2: AE -> LSTM (no CNN)", f1=0.10,
                                        fpr=0.01, params=100))

    marked = cmp.mark_best(cmp.build_full_comparison(verbose=False)["table"])
    row = dict(zip(marked["config"], marked["F1 (macro)"]))
    fpr = dict(zip(marked["config"], marked["binary FPR"]))
    params = dict(zip(marked["config"], marked["params"]))

    assert row["Approach 1: FS -> AE -> CNN-LSTM"].endswith("*")   # highest F1
    assert fpr["B2: AE -> LSTM (no CNN)"].endswith("*")            # lowest FPR
    assert params["B2: AE -> LSTM (no CNN)"].endswith("*")         # fewest
    assert not fpr["Approach 1: FS -> AE -> CNN-LSTM"].endswith("*")


def test_a_tie_is_marked_on_every_tied_row(tmp_path):
    write_all(tmp_path, include_approach2=True,
              proposed=metrics("Approach 1: FS -> AE -> CNN-LSTM", f1=0.75),
              baseline1_ae_cnn_lstm=metrics("B1: AE -> CNN-LSTM (no selection)",
                                            f1=0.75))

    marked = cmp.mark_best(cmp.build_full_comparison(verbose=False)["table"])
    starred = [c for c, v in zip(marked["config"], marked["F1 (macro)"])
               if str(v).endswith("*")]

    assert "Approach 1: FS -> AE -> CNN-LSTM" in starred
    assert "B1: AE -> CNN-LSTM (no selection)" in starred


def test_the_untrained_row_takes_no_part_in_the_marking(tmp_path):
    write_all(tmp_path, include_approach2=False)

    table = cmp.build_full_comparison(verbose=False)["table"]
    marked = cmp.mark_best(table)
    row = marked.iloc[1]

    assert row["F1 (macro)"] == config.NOT_TRAINED_TEXT
    assert "*" not in str(row["F1 (macro)"])
    starred = sum(1 for v in marked["F1 (macro)"] if str(v).endswith("*"))
    assert starred >= 1                     # a trained row still wins


# ===========================================================================
# the zero-recall diagnostic
# ===========================================================================
def test_zero_recall_names_the_classes_behind_a_zero_g_mean(tmp_path):
    # class 2 is never predicted, so its recall is 0 and the G-mean must be 0
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 0]
    write_result(tmp_path, "proposed",
                 metrics("Approach 1: FS -> AE -> CNN-LSTM", g_mean=0.0),
                 predictions_frame(y_true, y_pred))

    table = cmp.zero_recall_diagnostic()
    row = table[table["scope"] == "full test set"].iloc[0]

    assert row["G-mean is zero"] is True or row["G-mean is zero"] == True
    assert row["classes with recall 0"] == 1
    assert "Brute_Force" in row["which"]
    assert "(2 rows)" in row["which"]


def test_zero_recall_reports_none_when_every_class_is_found(tmp_path):
    write_result(tmp_path, "proposed", metrics("Approach 1"),
                 predictions_frame([0, 1, 2], [0, 1, 2]))

    table = cmp.zero_recall_diagnostic()
    assert list(table["which"])[0] == "none"
    assert not bool(table.iloc[0]["G-mean is zero"])


def test_zero_recall_covers_the_duplicate_free_subset_too(tmp_path):
    duplicates = np.array([True, False, False, False])
    write_result(tmp_path, "proposed", metrics("Approach 1"),
                 predictions_frame([0, 0, 1, 2], [0, 0, 1, 1], duplicates))

    table = cmp.zero_recall_diagnostic()
    assert set(table["scope"]) == {"full test set", "duplicate-free subset"}
    assert len(table) == 2


# ===========================================================================
# per-class and head-to-head
# ===========================================================================
def test_the_per_class_table_is_in_the_display_order(tmp_path):
    frame = predictions_frame([0, 1, 2, 0], [0, 1, 2, 0])
    write_result(tmp_path, "proposed", metrics("Approach 1: FS -> AE -> CNN-LSTM"),
                 frame)
    write_result(tmp_path, "baseline5",
                 metrics("B5: MI-only top 30 -> AE -> CNN-LSTM"), frame)

    table = cmp.per_class_table()

    assert list(table.columns)[:2] == ["Label", "test rows"]
    assert list(table.columns)[2:] == ["Approach 1: FS -> AE -> CNN-LSTM",
                                       "B5: MI-only top 30 -> AE -> CNN-LSTM"]


def test_the_head_to_head_pairs_are_the_attributable_ones(tmp_path):
    write_all(tmp_path, include_approach2=True)

    table = cmp.build_full_comparison(verbose=False)["table"]
    head = cmp.all_head_to_heads(table)

    pairs = set(head["pair"])
    assert any("selection method" in p for p in pairs)
    assert any("feature selection" in p for p in pairs)
    assert any("architecture" in p for p in pairs)
    assert len(pairs) == 3


def test_a_head_to_head_is_skipped_when_one_side_is_missing(tmp_path):
    write_all(tmp_path, include_approach2=False)

    head = cmp.all_head_to_heads(
        cmp.build_full_comparison(verbose=False)["table"])

    assert not any("architecture" in p for p in set(head["pair"]))
    assert len(set(head["pair"])) == 2


def test_head_to_head_direction_accounts_for_lower_is_better(tmp_path):
    write_all(tmp_path, include_approach2=True,
              proposed=metrics("Approach 1: FS -> AE -> CNN-LSTM", f1=0.8,
                               fpr=0.10),
              baseline5=metrics("B5: MI-only top 30 -> AE -> CNN-LSTM", f1=0.6,
                                fpr=0.01))

    table = cmp.build_full_comparison(verbose=False)["table"]
    head = cmp.all_head_to_heads(table)
    part = head[head["pair"].str.startswith("selection method")]

    f1 = part[part["metric"] == "F1 (macro)"].iloc[0]
    fpr = part[part["metric"] == "binary FPR"].iloc[0]
    assert f1["favours"] == "Approach 1: FS -> AE -> CNN-LSTM"
    assert fpr["favours"] == "B5: MI-only top 30 -> AE -> CNN-LSTM"


# ===========================================================================
# the guards
# ===========================================================================
def test_the_full_run_refuses_a_windows_path(tmp_path, monkeypatch, capsys,
                                             assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step12"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    out = cmp.run_step12_full(verbose=True)

    assert out["status"] == "skipped_not_windows"
    assert out["ok"] is False
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("run_step12_full")


def test_the_new_files_never_share_a_name_with_an_existing_one():
    new = set(config.STEP12_FULL_FILES.values())
    for existing in (config.STEP12_FILES, config.STEP12B_FILES,
                     config.STEP12C_FILES):
        assert not (new & set(existing.values())), sorted(new & set(existing.values()))


def test_the_display_order_block_holds_the_documented_defaults():
    assert len(config.STEP12_DISPLAY_ORDER) == 7
    assert config.NOT_TRAINED_TEXT == "not yet trained - run 10.2"
    assert set(config.STEP12_ISOLATES) == {k for k, _l
                                           in config.STEP12_DISPLAY_ORDER}
    assert config.STEP12_EXACT_TOLERANCE == 1e-9
    assert config.STEP12_EXPECTED_TOLERANCE == 5e-5
    for key in ("proposed", "baseline1_ae_cnn_lstm", "baseline2_ae_lstm",
                "baseline3_cnn_lstm", "baseline4_ae_pca_lstm"):
        assert set(EXPECTED_ROWS_AS_SHIPPED[key]) == {"accuracy", "f1_macro",
                                                      "g_mean"}
    # Baseline 5 has no published figure yet: it is new, so there is nothing
    # to hold it to, and inventing one would be worse than leaving it out
    assert "baseline5" not in EXPECTED_ROWS_AS_SHIPPED
    assert config.APPROACH2_KEY not in EXPECTED_ROWS_AS_SHIPPED
