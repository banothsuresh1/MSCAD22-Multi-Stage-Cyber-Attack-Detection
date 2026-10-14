"""
tests/test_evaluation.py -- pytest tests for STEP 11 (evaluation).

Most of these need no model at all: the metrics are checked against hand-worked
examples, which is the only way to know they are right.  Everything inside
tmp_path; no real data, and never C:\\MSCAD2022.

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
from src.mscad import evaluation as ev

REAL_STEP11_DIR = config.STEP11_OUTPUT_DIR


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STEP11_OUTPUT_DIR", tmp_path / "step11")


# ===========================================================================
# 1. G-mean
# ===========================================================================
def test_g_mean_on_a_hand_worked_example():
    """Two classes, recalls 1.0 and 0.25 -> sqrt(1.0 * 0.25) = 0.5."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1])     # class 1: 1 of 4 recalled

    assert ev.geometric_mean_recall(y_true, y_pred) == pytest.approx(0.5)


def test_g_mean_is_zero_when_any_class_is_never_recalled():
    """The whole point of G-mean: a forgotten class cannot be averaged away."""
    y_true = np.array([0] * 99 + [1])
    y_pred = np.array([0] * 100)                     # class 1 never predicted

    assert ev.geometric_mean_recall(y_true, y_pred) == 0.0
    # while accuracy stays flattering
    from sklearn.metrics import accuracy_score
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.99)


def test_g_mean_of_a_perfect_classifier_is_one():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert ev.geometric_mean_recall(y, y) == pytest.approx(1.0)


def test_g_mean_cross_check_reports_agreement_or_absence():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    check = ev.cross_check_gmean(y_true, y_pred)

    assert check["ours"] == pytest.approx(ev.geometric_mean_recall(y_true, y_pred))
    if check["imblearn"] is None:
        assert "not installed" in check["note"]
    else:
        assert check["agree"] is True, "the two computations must match"


# ===========================================================================
# 2. the headline metrics
# ===========================================================================
def test_macro_f1_on_a_hand_worked_example():
    """Class 0: P=1.0 R=0.5 F1=2/3.  Class 1: P=2/3 R=1.0 F1=0.8."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = ev.headline_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["f1_macro"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["recall_macro"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["precision_macro"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics["rows"] == 4


def test_weighted_differs_from_macro_when_the_classes_are_imbalanced():
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.array([0] * 95 + [1] * 5)
    metrics = ev.headline_metrics(y_true, y_pred)
    assert metrics["f1_weighted"] > metrics["f1_macro"], \
        "the weighted average must favour the majority class"


# ===========================================================================
# 3. ROC-AUC with an absent class
# ===========================================================================
def test_roc_auc_skips_a_class_with_no_test_rows_and_names_it():
    # 3 classes exist, but class 2 never appears in y_true
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1],
                              [0.2, 0.7, 0.1], [0.3, 0.6, 0.1]])

    result = ev.roc_auc_ovr(y_true, probabilities, n_classes=3)

    assert result["skipped_absent"] == [2]
    assert result["macro"] is not None
    assert 0.0 <= result["macro"] <= 1.0
    assert result["macro"] == pytest.approx(1.0), "these predictions are perfect"


def test_roc_auc_is_undefined_with_only_one_class_present():
    y_true = np.array([0, 0, 0])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    result = ev.roc_auc_ovr(y_true, probabilities, n_classes=2)
    assert result["macro"] is None
    assert "undefined" in result["note"]


# ===========================================================================
# 4. the binary view
# ===========================================================================
def test_binary_view_on_a_hand_worked_example():
    """Class 0 is benign; P(attack) = 1 - P(class 0)."""
    y_true = np.array([0, 0, 1, 2])
    probabilities = np.array([
        [0.9, 0.05, 0.05],     # benign, called benign     -> true negative
        [0.3, 0.4, 0.3],       # benign, called attack     -> FALSE POSITIVE
        [0.1, 0.8, 0.1],       # attack, called attack     -> true positive
        [0.2, 0.3, 0.5],       # attack, called attack     -> true positive
    ])

    view = ev.binary_view(y_true, probabilities, benign_codes=[0], n_classes=3)

    assert view["true_negatives"] == 1
    assert view["false_positives"] == 1
    assert view["true_positives"] == 2
    assert view["false_negatives"] == 0
    assert view["accuracy"] == pytest.approx(0.75)
    assert view["recall"] == pytest.approx(1.0)        # both attacks caught
    assert view["precision"] == pytest.approx(2 / 3)   # one false alarm
    assert view["false_positive_rate"] == pytest.approx(0.5)   # 1 of 2 benign


def test_binary_attack_probability_is_one_minus_the_benign_classes():
    y_true = np.array([0, 1])
    probabilities = np.array([[0.6, 0.25, 0.15], [0.1, 0.5, 0.4]])
    view = ev.binary_view(y_true, probabilities, benign_codes=[0], n_classes=3)
    # row 0: p(attack) = 1 - 0.6 = 0.4 -> benign; row 1: 1 - 0.1 = 0.9 -> attack
    assert view["true_negatives"] == 1 and view["true_positives"] == 1


def test_benign_codes_are_read_from_label_binary():
    y_test = pd.DataFrame({"Label": ["Normal", "Port_Scan", "Benign"],
                           "Label_encoded": [3, 4, 0],
                           "Label_binary": [0, 1, 0]})
    assert ev.benign_codes_from(y_test, {}) == [0, 3]


# ===========================================================================
# 5. duplicate detection
# ===========================================================================
def test_duplicate_mask_on_a_constructed_example():
    columns = ["a", "b"]
    X_train = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=columns)
    X_test = pd.DataFrame([[1.0, 2.0], [9.0, 9.0], [3.0, 4.0], [5.0, 6.0]],
                          columns=columns)

    mask = ev.duplicate_mask(X_train, X_test)

    assert mask.tolist() == [True, False, True, False]
    assert mask.sum() == 2


def test_a_near_miss_is_not_a_duplicate():
    X_train = pd.DataFrame([[1.0, 2.0]], columns=["a", "b"])
    X_test = pd.DataFrame([[1.0, 2.0000001]], columns=["a", "b"])
    assert ev.duplicate_mask(X_train, X_test).tolist() == [False]


def test_the_duplicate_check_needs_matching_columns():
    X_train = pd.DataFrame({"a": [1.0], "b": [2.0]})
    X_test = pd.DataFrame({"b": [2.0], "a": [1.0]})
    with pytest.raises(ev.EvaluationError):
        ev.duplicate_mask(X_train, X_test)


def test_the_duplicate_free_score_uses_the_same_predictions():
    """Nothing is retrained: only the row set changes."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    probabilities = np.tile([[0.6, 0.4]], (4, 1))
    mask = np.array([True, True, False, False])       # the first two are twins

    summary, counts, full, free = ev.duplicate_comparison(
        y_true, y_pred, probabilities, mask, {0: "Normal", 1: "Attack"},
        n_classes=2)

    assert full["rows"] == 4
    assert free["rows"] == 2
    # the duplicate-free subset is rows 2 and 3, both class 1, both predicted 1
    assert free["accuracy"] == pytest.approx(1.0)
    assert set(summary["metric"]) >= {"rows", "accuracy", "f1_macro", "g_mean"}
    row = counts.set_index("Label")
    assert int(row.loc["Normal", "with a train duplicate"]) == 2
    assert int(row.loc["Attack", "duplicate-free rows"]) == 2


def test_every_row_duplicated_is_reported_rather_than_crashing():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    probabilities = np.tile([[0.5, 0.5]], (2, 1))
    mask = np.array([True, True])

    _summary, _counts, full, free = ev.duplicate_comparison(
        y_true, y_pred, probabilities, mask, None, n_classes=2)

    assert full["rows"] == 2
    assert free["rows"] == 0
    assert "no duplicate-free subset" in free["note"]


# ===========================================================================
# 6. the supporting tables
# ===========================================================================
def test_per_class_table_lists_every_class_with_its_support():
    y_true = np.array([0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2])
    table = ev.per_class_table(y_true, y_pred,
                               {0: "Normal", 1: "Scan", 2: "Flood"}, n_classes=3)

    assert set(table["Label"]) == {"Normal", "Scan", "Flood"}
    assert int(table.loc[table["Label"] == "Normal", "support"].iloc[0]) == 3
    assert table["support"].sum() == 5
    assert table["support"].is_monotonic_decreasing


def test_confusion_tables_rows_are_truth_and_normalise_to_one():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    counts, normalised = ev.confusion_tables(y_true, y_pred,
                                             {0: "Normal", 1: "Attack"}, 2)

    assert counts.loc["Normal", "Normal"] == 1
    assert counts.loc["Normal", "Attack"] == 1
    assert counts.loc["Attack", "Attack"] == 2
    assert np.allclose(normalised.sum(axis=1), 1.0)
    assert normalised.loc["Normal", "Normal"] == pytest.approx(0.5)


def test_a_class_with_no_rows_leaves_a_zero_row_rather_than_nan():
    y_true = np.array([0, 0])
    y_pred = np.array([0, 0])
    _counts, normalised = ev.confusion_tables(y_true, y_pred, None, n_classes=3)
    assert not normalised.isna().to_numpy().any()
    assert normalised.iloc[2].sum() == 0.0


def test_label_names_are_read_either_way_round():
    assert ev.label_names_from_mapping({"Normal": 3})[3] == "Normal"
    assert ev.label_names_from_mapping({"3": "Normal"})[3] == "Normal"
    assert ev.label_names_from_mapping({}) == {}


def test_seed_aggregation_reports_mean_and_std():
    runs = [{"full": {"f1_macro": 0.80, "accuracy": 0.9},
             "duplicate_free": {"f1_macro": 0.70, "accuracy": 0.8}},
            {"full": {"f1_macro": 0.90, "accuracy": 0.9},
             "duplicate_free": {"f1_macro": 0.60, "accuracy": 0.8}}]

    table = ev.aggregate_seeds(runs)
    row = table[(table["scope"] == "full") & (table["metric"] == "f1_macro")].iloc[0]
    assert row["mean"] == pytest.approx(0.85)
    assert row["std"] == pytest.approx(np.std([0.8, 0.9], ddof=1), abs=1e-6)
    assert row["seeds"] == 2


# ===========================================================================
# 7. the Windows-path guard
# ===========================================================================
def test_step11_refuses_a_windows_path_off_windows(tmp_path, monkeypatch, capsys,
                                                   assert_no_leaked_output):
    """On Linux, Path(r"C:\\...") is RELATIVE -- nothing may be created."""
    monkeypatch.setattr(config, "STEP11_OUTPUT_DIR",
                        config.Path(r"C:\MSCAD2022\outputs\step11"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    assert ev.prepare_output_dir(config.STEP11_OUTPUT_DIR, "the Step 11 outputs") \
        is None
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("prepare_output_dir for the Step 11 folder")


def test_step11_output_dir_is_under_data_root_not_the_repo():
    assert REAL_STEP11_DIR.parent.parent == config.DATA_ROOT
    assert config.PROJECT_ROOT not in REAL_STEP11_DIR.parents
    assert REAL_STEP11_DIR.name == "step11"
