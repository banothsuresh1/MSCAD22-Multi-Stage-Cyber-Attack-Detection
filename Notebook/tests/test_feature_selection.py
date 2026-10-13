"""
tests/test_feature_selection.py -- pytest tests for STEP 6 (feature selection).

Everything is built from a small synthetic frame in a temporary folder: 300
rows, 12 features, 3 classes, no real data and never C:\\MSCAD2022.  An autouse
fixture redirects config.STEP5_OUTPUT_DIR and config.STEP6_OUTPUT_DIR to
tmp_path, so no test can write to a configured folder on any operating system.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pytest tests -v
"""

import json

import numpy as np
import pandas as pd
import pytest

import config
from src.mscad import feature_selection as fs

REAL_STEP6_DIR = config.STEP6_OUTPUT_DIR          # captured before any patching

N_ROWS = 300
CLASSES = ["Normal", "Port_Scan", "Brute_Force"]


@pytest.fixture(autouse=True)
def _never_write_outside_tmp(tmp_path, monkeypatch):
    """Step 5 input and Step 6 output both live in tmp_path."""
    monkeypatch.setattr(config, "STEP5_OUTPUT_DIR", tmp_path / "step5")
    monkeypatch.setattr(config, "STEP6_OUTPUT_DIR", tmp_path / "step6")
    # keep the tests quick: the ranking forest does not need 200 trees
    monkeypatch.setattr(config, "RF_PARAMS",
                        dict(n_estimators=12, class_weight="balanced",
                             n_jobs=1, random_state=config.RANDOM_STATE))
    monkeypatch.setattr(config, "PI_N_REPEATS", 2)
    monkeypatch.setattr(config, "N_SELECTED_FEATURES", 6)


# ===========================================================================
# the synthetic dataset
# ===========================================================================
def make_xy(n_rows: int = N_ROWS, seed: int = 0):
    """12 features: 3 informative, 1 an exact copy, 1 constant, the rest noise."""
    rng = np.random.default_rng(seed)
    y_code = np.repeat([0, 1, 2], n_rows // 3)
    names = np.array([CLASSES[c] for c in y_code])

    data = {}
    # three features that genuinely separate the classes, by different amounts
    data["strong"] = y_code + rng.normal(0, 0.10, n_rows)
    data["medium"] = y_code + rng.normal(0, 0.45, n_rows)
    data["weak"] = y_code + rng.normal(0, 1.10, n_rows)
    # an EXACT copy of "strong" -- Pearson must drop one of the two
    data["strong_copy"] = data["strong"].copy()
    data["constant"] = np.zeros(n_rows)                  # dropped before 6.2
    for i in range(5):
        data["noise_%d" % i] = rng.normal(0, 1, n_rows)
    data[config.ZERO_DURATION_COLUMN] = (rng.random(n_rows) < 0.3).astype("int64")
    data["Protocol_TCP"] = (rng.random(n_rows) < 0.8).astype("int64")

    X = pd.DataFrame(data)
    y = pd.DataFrame({"Label": names, "Label_encoded": y_code,
                      "Label_binary": (y_code > 0).astype("int64")})
    return X, y


def write_step5(X_train=None, y_train=None, X_test=None, y_test=None):
    """Lay down the four Step 5 files Step 6 reads."""
    if X_train is None:
        X_train, y_train = make_xy(seed=0)
    if X_test is None:
        X_test, y_test = make_xy(n_rows=90, seed=1)
    folder = config.STEP5_OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    names = config.STEP5_FILES
    X_train.to_csv(folder / names["X_train"], index=False)
    X_test.to_csv(folder / names["X_test"], index=False)
    y_train.to_csv(folder / names["y_train"], index=False)
    y_test.to_csv(folder / names["y_test"], index=False)
    return X_train, X_test, y_train, y_test


# ===========================================================================
# 1. 6.1 -- the input checks
# ===========================================================================
def test_inputs_are_checked_before_selection():
    X_train, X_test, y_train, y_test = write_step5()
    table, ok = fs.check_inputs(X_train, X_test, y_train, y_test, verbose=False)
    assert ok is True
    assert "[FAIL]" not in set(table["result"])


def test_a_label_column_among_the_features_is_refused():
    X_train, X_test, y_train, y_test = write_step5()
    leaky = X_train.copy()
    leaky["Label_encoded"] = y_train["Label_encoded"]
    with pytest.raises(fs.SelectionError) as err:
        fs.check_inputs(leaky, X_test, y_train, y_test, verbose=False)
    assert "Label_encoded" in str(err.value)


def test_an_identifier_column_among_the_features_is_refused():
    X_train, X_test, y_train, y_test = write_step5()
    leaky = X_train.copy()
    leaky["Src IP"] = "192.168.1.1"
    with pytest.raises(fs.SelectionError) as err:
        fs.check_inputs(leaky, X_test, y_train, y_test, verbose=False)
    assert "Src IP" in str(err.value)
    assert "leak" in str(err.value)


def test_mismatched_columns_between_train_and_test_are_refused():
    X_train, X_test, y_train, y_test = write_step5()
    with pytest.raises(fs.SelectionError) as err:
        fs.check_inputs(X_train, X_test.drop(columns=["weak"]), y_train, y_test,
                        verbose=False)
    assert "same columns" in str(err.value)


def test_the_constant_column_is_dropped():
    X_train, X_test, _y, _yt = write_step5()
    kept_train, kept_test, constant = fs.drop_constant_columns(
        X_train, X_test, verbose=False)
    assert constant == ["constant"]
    assert "constant" not in kept_train.columns
    assert "constant" not in kept_test.columns
    assert list(kept_train.columns) == list(kept_test.columns)


# ===========================================================================
# 2. 6.2 -- mutual information, computed ONCE
# ===========================================================================
def test_mutual_information_is_computed_exactly_once_per_run(monkeypatch):
    """MI is stochastic, so a second call would silently disagree with the first."""
    write_step5()
    from sklearn import feature_selection as skfs

    calls = {"n": 0}
    real = skfs.mutual_info_classif

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(skfs, "mutual_info_classif", counting)
    result = fs.run_step6(overwrite=True, verbose=False)

    assert calls["n"] == 1, "mutual_info_classif ran %d times" % calls["n"]
    # and the one result is what both later stages used
    ranking = result["ranking"].set_index("feature")
    for feature in result["pearson_kept"]:
        assert ranking.loc[feature, "MI"] == pytest.approx(
            float(result["mi"][feature])), feature


def test_the_informative_features_score_above_the_noise():
    write_step5()
    X_train = pd.read_csv(config.STEP5_OUTPUT_DIR / config.STEP5_FILES["X_train"])
    y_train = pd.read_csv(config.STEP5_OUTPUT_DIR / config.STEP5_FILES["y_train"])
    X_train = X_train.drop(columns=["constant"])

    mi, table, seconds = fs.mutual_information(X_train, y_train, verbose=False)

    assert seconds >= 0
    assert mi["strong"] > mi["weak"], "a cleaner signal must score higher"
    for i in range(5):
        assert mi["strong"] > mi["noise_%d" % i]
    assert list(table.columns) == ["rank", "feature", "mutual_information"]
    assert table["mutual_information"].is_monotonic_decreasing


# ===========================================================================
# 3. 6.3 -- the Pearson redundancy filter
# ===========================================================================
def test_a_duplicated_feature_is_dropped_and_the_higher_mi_one_kept():
    write_step5()
    X = pd.read_csv(config.STEP5_OUTPUT_DIR / config.STEP5_FILES["X_train"])
    X = X.drop(columns=["constant"])
    # "strong" and "strong_copy" are identical, so |r| == 1
    mi = pd.Series({c: 0.1 for c in X.columns})
    mi["strong"] = 0.9          # the clear winner of the pair
    mi["strong_copy"] = 0.4

    kept, dropped, corr, remaining = fs.pearson_filter(X, mi, verbose=False)

    assert "strong" in kept
    assert "strong_copy" not in kept
    assert len(dropped) == 1
    row = dropped.iloc[0]
    assert row["kept"] == "strong" and row["dropped"] == "strong_copy"
    assert row["abs_r"] == pytest.approx(1.0)
    assert row["MI kept"] >= row["MI dropped"]


def test_an_mi_tie_is_broken_by_feature_order_so_the_result_is_deterministic():
    write_step5()
    X = pd.read_csv(config.STEP5_OUTPUT_DIR / config.STEP5_FILES["X_train"])
    X = X.drop(columns=["constant"])
    mi = pd.Series({c: 0.5 for c in X.columns})      # every score identical

    first = fs.pearson_filter(X, mi, verbose=False)[0]
    for _ in range(4):
        assert fs.pearson_filter(X, mi, verbose=False)[0] == first
    # "strong" comes before "strong_copy" in column order, so it is the one kept
    assert "strong" in first and "strong_copy" not in first


def test_pairs_below_the_threshold_are_reported_and_kept():
    write_step5()
    X = pd.read_csv(config.STEP5_OUTPUT_DIR / config.STEP5_FILES["X_train"])
    X = X.drop(columns=["constant", "strong_copy"])
    rng = np.random.default_rng(3)
    X["partner"] = X["strong"] * 0.9 + rng.normal(0, 0.45, len(X))
    mi = pd.Series({c: 0.5 for c in X.columns})

    kept, dropped, corr, remaining = fs.pearson_filter(
        X, mi, threshold=0.99, report_lower=0.50, verbose=False)

    assert "partner" in kept and "strong" in kept, "neither may be dropped"
    pairs = {frozenset([a, b]) for a, b in
             zip(remaining["feature A"], remaining["feature B"])}
    assert frozenset(["strong", "partner"]) in pairs
    assert (remaining["abs_r"] < 0.99).all() and (remaining["abs_r"] >= 0.50).all()


# ===========================================================================
# 4. 6.4 -- permutation importance, and the duplicate-overlap count
# ===========================================================================
def test_duplicate_overlap_count_is_correct_on_a_constructed_example():
    fit = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=["a", "b"])
    # rows 0 and 2 are exact copies of fit rows; rows 1 and 3 are new
    val = pd.DataFrame([[1.0, 2.0], [9.0, 9.0], [5.0, 6.0], [7.0, 8.0]],
                       columns=["a", "b"])
    labels = pd.Series(["Port_Scan", "Port_Scan", "Normal", "Normal"])

    table, overall, mask = fs.duplicate_overlap(fit, val, labels, verbose=False)

    assert overall["validation rows"] == 4
    assert overall["also in fit"] == 2
    assert overall["% of validation"] == 50.0
    assert mask.tolist() == [True, False, True, False]
    per_label = table.set_index(config.LABEL_COLUMN)
    assert per_label.loc["Port_Scan", "also in fit"] == 1
    assert per_label.loc["Normal", "also in fit"] == 1
    assert per_label.loc["TOTAL", "also in fit"] == 2


def test_permutation_importance_ranks_signal_above_noise():
    write_step5()
    X = pd.read_csv(config.STEP5_OUTPUT_DIR / config.STEP5_FILES["X_train"])
    y = pd.read_csv(config.STEP5_OUTPUT_DIR / config.STEP5_FILES["y_train"])
    features = ["strong", "medium", "weak", "noise_0", "noise_1"]

    pi_table, forest, info = fs.permutation_scores(X, y, features, verbose=False)

    assert list(pi_table.columns) == ["feature", "pi_mean", "pi_std"]
    assert pi_table["pi_mean"].is_monotonic_decreasing
    assert set(pi_table["feature"]) == set(features)
    scores = dict(zip(pi_table["feature"], pi_table["pi_mean"]))
    assert scores["strong"] > scores["noise_0"]
    assert 0.0 <= info["baseline_macro_f1"] <= 1.0
    assert info["fit_rows"] + info["validation_rows"] == len(X)
    assert len(info["per_class_f1"]) >= 1


# ===========================================================================
# 5. 6.5 -- combining the two scores
# ===========================================================================
def test_combined_score_is_the_sum_of_two_normalised_scores():
    mi = pd.Series({"a": 1.0, "b": 0.5, "c": 0.0})
    pi = pd.DataFrame({"feature": ["a", "b", "c"], "pi_mean": [0.0, 0.2, 0.4],
                       "pi_std": [0.0, 0.0, 0.0]})

    table, info = fs.combine_scores(mi, pi, features=["a", "b", "c"], verbose=False)

    assert table["MI_norm"].between(0, 1).all()
    assert table["PI_norm"].between(0, 1).all()
    assert np.allclose(table["CombinedScore"], table["MI_norm"] + table["PI_norm"])
    row = table.set_index("feature")
    assert row.loc["a", "MI_norm"] == pytest.approx(1.0)
    assert row.loc["c", "MI_norm"] == pytest.approx(0.0)
    assert row.loc["c", "PI_norm"] == pytest.approx(1.0)
    # a: 1.0 + 0.0, b: 0.5 + 0.5, c: 0.0 + 1.0 -> all exactly 1.0, a three-way tie
    assert list(table["feature"]) == ["a", "b", "c"], "ties fall back to MI' then name"
    assert table["rank"].tolist() == [1, 2, 3]


def test_sort_order_is_combined_then_mi_then_name():
    mi = pd.Series({"zebra": 0.0, "alpha": 0.0, "beta": 1.0})
    pi = pd.DataFrame({"feature": ["zebra", "alpha", "beta"],
                       "pi_mean": [1.0, 1.0, 0.0], "pi_std": [0.0, 0.0, 0.0]})

    table, _info = fs.combine_scores(mi, pi, features=["zebra", "alpha", "beta"],
                                     verbose=False)
    # all three score 1.0; MI' separates beta (1.0) from the other two (0.0),
    # and the remaining tie is broken alphabetically
    assert list(table["feature"]) == ["beta", "alpha", "zebra"]


def test_negative_permutation_importance_is_clipped():
    mi = pd.Series({"a": 1.0, "b": 0.0})
    pi = pd.DataFrame({"feature": ["a", "b"], "pi_mean": [-0.3, 0.5],
                       "pi_std": [0.0, 0.0]})

    clipped, info = fs.combine_scores(mi, pi, features=["a", "b"],
                                      clip_negative=True, verbose=False)
    assert info["n_negative_pi"] == 1
    assert (clipped["PI"] >= 0).all()
    assert clipped.set_index("feature").loc["a", "PI"] == pytest.approx(0.0)

    kept, info2 = fs.combine_scores(mi, pi, features=["a", "b"],
                                    clip_negative=False, verbose=False)
    assert kept.set_index("feature").loc["a", "PI"] == pytest.approx(-0.3)


def test_a_flat_score_normalises_to_zero_instead_of_dividing_by_zero():
    mi = pd.Series({"a": 0.4, "b": 0.4, "c": 0.4})          # max == min
    pi = pd.DataFrame({"feature": ["a", "b", "c"], "pi_mean": [0.0, 0.5, 1.0],
                       "pi_std": [0.0, 0.0, 0.0]})

    table, info = fs.combine_scores(mi, pi, features=["a", "b", "c"], verbose=False)

    assert info["mi_flat"] is True and info["pi_flat"] is False
    assert (table["MI_norm"] == 0).all(), "a flat score contributes nothing"
    assert np.isfinite(table["CombinedScore"]).all()
    assert list(table["feature"]) == ["c", "b", "a"], "PI alone decides the order"


# ===========================================================================
# 6. 6.6 -- selection
# ===========================================================================
def test_exactly_n_features_are_selected_in_rank_order():
    write_step5()
    result = fs.run_step6(overwrite=True, verbose=False)

    n = config.N_SELECTED_FEATURES
    assert len(result["selected"]) == n
    assert result["short"] is False
    assert result["selected"] == list(result["ranking"]["feature"].head(n))
    assert list(result["X_train_selected"].columns) == result["selected"]
    assert list(result["X_test_selected"].columns) == result["selected"]


def test_fewer_features_than_requested_selects_all_of_them_and_warns(monkeypatch,
                                                                     capsys):
    ranking = pd.DataFrame({
        "rank": [1, 2, 3], "feature": ["a", "b", "c"],
        "MI": [1.0, 0.5, 0.0], "PI": [1.0, 0.5, 0.0],
        "MI_norm": [1.0, 0.5, 0.0], "PI_norm": [1.0, 0.5, 0.0],
        "CombinedScore": [2.0, 1.0, 0.0]})
    monkeypatch.setattr(config, "N_SELECTED_FEATURES", 30)

    features, marked, short = fs.select_top(ranking, verbose=True)

    assert short is True
    assert features == ["a", "b", "c"], "all of them, not 30"
    assert marked["selected"].all()
    printed = capsys.readouterr().out
    assert "WARN" in printed and "only 3" in printed


def test_x_test_is_never_used_for_any_selection_decision():
    """Changing X_test must not move MI, Pearson, PI or the ranking."""
    X_train, X_test, y_train, y_test = write_step5()
    first = fs.run_step6(overwrite=True, verbose=False)

    # scramble X_test beyond recognition, keeping its columns
    rng = np.random.default_rng(99)
    wrecked = pd.DataFrame(rng.normal(50, 20, X_test.shape), columns=X_test.columns)
    wrecked["constant"] = 0.0
    write_step5(X_train=X_train, y_train=y_train, X_test=wrecked, y_test=y_test)
    second = fs.run_step6(overwrite=True, verbose=False)

    assert second["selected"] == first["selected"]
    assert second["pearson_kept"] == first["pearson_kept"]
    pd.testing.assert_frame_equal(first["ranking"], second["ranking"])
    # and the ONLY thing that changed is the values taken from X_test
    assert list(second["X_test_selected"].columns) == first["selected"]


# ===========================================================================
# 7. end to end: the saved files
# ===========================================================================
def test_run_step6_writes_and_verifies_every_file():
    write_step5()
    result = fs.run_step6(overwrite=True, verbose=False)

    assert result["ok"] is True
    folder = config.STEP6_OUTPUT_DIR
    for key, name in config.STEP6_FILES.items():
        if key in ("heatmap", "ranking_chart", "funnel", "rf_baseline"):
            continue              # matplotlib / joblib may not be installed
        assert (folder / name).exists(), name
    assert "[FAIL]" not in set(result["saved"]["verification"]["result"])


def test_saved_files_have_the_right_shapes_and_order():
    X_train, X_test, _y, _yt = write_step5()
    result = fs.run_step6(overwrite=True, verbose=False)
    folder = config.STEP6_OUTPUT_DIR
    names = config.STEP6_FILES

    picked = json.loads((folder / names["selected"]).read_text(encoding="utf-8"))
    assert picked["features"] == result["selected"]
    assert picked["n_selected"] == config.N_SELECTED_FEATURES
    assert picked["config"]["PEARSON_THRESHOLD"] == config.PEARSON_THRESHOLD
    assert picked["config"]["RF_PARAMS"]["n_estimators"] == \
        config.RF_PARAMS["n_estimators"]

    saved_train = pd.read_csv(folder / names["X_train_selected"], low_memory=False)
    saved_test = pd.read_csv(folder / names["X_test_selected"], low_memory=False)
    assert list(saved_train.columns) == picked["features"]
    assert list(saved_test.columns) == picked["features"]
    assert len(saved_train) == len(X_train)
    assert len(saved_test) == len(X_test)

    y_train = pd.read_csv(folder / names["y_train"])
    assert list(y_train.columns) == ["Label", "Label_encoded", "Label_binary"]
    assert len(y_train) == len(X_train)

    ranking = pd.read_csv(folder / names["ranking"], low_memory=False)
    assert ranking["selected"].sum() == config.N_SELECTED_FEATURES
    assert list(ranking.loc[ranking["selected"], "feature"]) == picked["features"]


def test_step6_does_not_touch_the_step5_outputs():
    write_step5()
    folder5 = config.STEP5_OUTPUT_DIR
    before = {p.name: p.read_bytes() for p in folder5.iterdir()}

    fs.run_step6(overwrite=True, verbose=False)

    for name, content in before.items():
        assert (folder5 / name).read_bytes() == content, name


def test_outputs_are_loaded_instead_of_reselected_when_overwrite_is_off():
    write_step5()
    first = fs.run_step6(overwrite=True, verbose=False)
    again = fs.run_step6(overwrite=False, verbose=False)

    assert again["status"] == "loaded_existing"
    assert again["selected"] == first["selected"]
    assert again["X_train_selected"].shape == first["X_train_selected"].shape


def test_the_report_records_every_stage():
    write_step5()
    result = fs.run_step6(overwrite=True, verbose=False)
    stages = dict((name, count) for name, count in result["stages"])

    assert stages["initial"] == 12
    assert stages["after the constant drop"] == 11
    assert stages["after Pearson"] < stages["after the constant drop"], \
        "strong_copy must have been dropped"
    assert stages["selected"] == config.N_SELECTED_FEATURES
    assert "strong_copy" not in result["selected"]


# ===========================================================================
# 8. the Windows-path guard
# ===========================================================================
def test_step6_refuses_a_windows_path_off_windows(tmp_path, monkeypatch, capsys,
                                                  assert_no_leaked_output):
    """On Linux, Path(r"C:\\...") is RELATIVE -- nothing may be created."""
    monkeypatch.setattr(config, "STEP6_OUTPUT_DIR",
                        config.Path(r"C:\MSCAD2022\outputs\step6"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    folder = fs.prepare_output_dir(config.STEP6_OUTPUT_DIR, "the Step 6 outputs")

    assert folder is None
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("prepare_output_dir for the Step 6 folder")


def test_step6_output_dir_is_under_data_root_not_the_repo():
    assert REAL_STEP6_DIR.parent.parent == config.DATA_ROOT
    assert config.PROJECT_ROOT not in REAL_STEP6_DIR.parents
    assert REAL_STEP6_DIR.name == "step6"
