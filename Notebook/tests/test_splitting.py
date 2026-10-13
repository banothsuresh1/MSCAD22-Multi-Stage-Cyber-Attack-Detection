"""
tests/test_splitting.py -- pytest tests for STEP 5 (the train/test split).

Everything is built from small synthetic frames in a temporary folder: no real
captures, no merged dataset, and never C:\\MSCAD2022.  An autouse fixture
redirects BOTH config.STEP4_OUTPUT_DIR and config.STEP5_OUTPUT_DIR to tmp_path,
so no test can write to a configured folder on any operating system.

    cd /d C:\\MSCAD2022\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pytest tests -v
"""

import json

import numpy as np
import pandas as pd
import pytest

import config
from src.mscad import preprocessing as pp
from src.mscad import splitting as sp

REAL_STEP5_DIR = config.STEP5_OUTPUT_DIR          # captured before any patching


@pytest.fixture(autouse=True)
def _never_write_outside_tmp(tmp_path, monkeypatch):
    """Step 4 and Step 5 output always go to tmp_path, never anywhere else."""
    monkeypatch.setattr(config, "STEP4_OUTPUT_DIR", tmp_path / "step4")
    monkeypatch.setattr(config, "STEP5_OUTPUT_DIR", tmp_path / "step5")


# ===========================================================================
# the synthetic dataset
# ===========================================================================
# A rare class of 5 rows, which is what makes stratification matter: an
# unstratified 20% split would lose it from one side about a third of the time.
LABEL_PLAN = (["Brute_Force"] * 100 + ["Port_Scan"] * 50 + ["DoS slowloris"] * 25
              + ["ICMP_Flood"] * 12 + ["Normal"] * 8 + ["Web_Crwling"] * 5)


def make_merged(labels=None, seed: int = 0) -> pd.DataFrame:
    """A miniature merged dataset in the Step 3 layout."""
    labels = list(LABEL_PLAN if labels is None else labels)
    n_rows = len(labels)
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}

    for col in config.KAGGLE_FEATURES:
        data[col] = rng.uniform(1, 100, n_rows).round(3)

    data["Flow Duration"] = [0.0, 0.0] + list(rng.uniform(10, 5000, n_rows - 2))
    for col in ("Flow Byts/s", "Flow Pkts/s"):
        values = list(rng.uniform(1, 500, n_rows))
        values[0] = values[1] = np.inf          # zero duration -> inf, as Step 1 does
        data[col] = values
    data["Bwd PSH Flags"] = [0.0] * n_rows      # constant -> Step 4 drops it

    data["Src IP"] = ["192.168.159.%d" % (i % 7 + 1) for i in range(n_rows)]
    data["Dst IP"] = ["192.168.159.20"] * n_rows
    data["Src Port"] = list(range(40000, 40000 + n_rows))
    data["Dst Port"] = [80] * n_rows
    data["Protocol"] = [(6, 17, 1, 2)[i % 4] for i in range(n_rows)]
    data["Timestamp"] = ["2020-01-01 %02d:%02d:%02d.000000"
                         % (18 + i // 3600, (i // 60) % 60, i % 60)
                         for i in range(n_rows)]
    data[config.LABEL_COLUMN] = labels

    df = pd.DataFrame(data)[config.FLOW_COLUMNS]
    df["Source_Capture"] = ["W-B-02"] * n_rows
    df["Orig_Row"] = list(range(n_rows))
    df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"])
    return df


def run_step4(df=None):
    """Write the Step 4 outputs Step 5 reads, into tmp_path."""
    return pp.run_step4(make_merged() if df is None else df, overwrite=True,
                        verbose=False)


# ===========================================================================
# 1. the input checks
# ===========================================================================
def test_inputs_are_checked_before_the_split():
    run_step4()
    loaded = sp.load_step4_outputs(verbose=False)
    table, ok = sp.check_inputs(loaded["features"], loaded["labels"],
                                loaded["metadata"], verbose=False)
    assert ok is True
    assert "[FAIL]" not in set(table["result"])
    assert len(loaded["features"]) == len(LABEL_PLAN)


def test_a_row_count_mismatch_stops_the_split():
    run_step4()
    loaded = sp.load_step4_outputs(verbose=False)
    with pytest.raises(sp.SplittingError) as err:
        sp.check_inputs(loaded["features"], loaded["labels"].iloc[:-3],
                        loaded["metadata"], verbose=False)
    assert "row count" in str(err.value)
    assert "OVERWRITE_STEP4" in str(err.value)


def test_a_class_with_one_row_gives_a_clear_error():
    """stratify cannot put a 1-row class on both sides of the split."""
    df = make_merged()
    # one row, and one row only, of a class nothing else shares
    df.loc[df.index[0], config.LABEL_COLUMN] = "SQL_Injection"
    run_step4(df)
    loaded = sp.load_step4_outputs(verbose=False)
    assert (loaded["labels"][config.LABEL_COLUMN] == "SQL_Injection").sum() == 1

    with pytest.raises(sp.SplittingError) as err:
        sp.check_inputs(loaded["features"], loaded["labels"], loaded["metadata"],
                        verbose=False)
    message = str(err.value)
    assert "at least 2 rows per class" in message
    assert "'SQL_Injection': 1" in message, "name the class and its row count"
    # and the advice is actionable, not just a complaint
    assert "merge" in message and "drop" in message


def test_the_expected_row_count_is_a_note_not_a_failure(monkeypatch):
    """A different row count must not block the split -- it only warns."""
    monkeypatch.setattr(config, "STEP5_EXPECTED_ROWS", 105_387)
    run_step4()
    loaded = sp.load_step4_outputs(verbose=False)
    table, ok = sp.check_inputs(loaded["features"], loaded["labels"],
                                loaded["metadata"], verbose=False)
    assert ok is True
    row = table[table["check"].str.contains("expected")]
    assert row["result"].iloc[0] == "[NOTE]"
    assert "105,387" in row["detail"].iloc[0]


# ===========================================================================
# 2. the split itself
# ===========================================================================
def test_split_is_80_20_with_no_overlap():
    run_step4()
    loaded = sp.load_step4_outputs(verbose=False)
    train_idx, test_idx = sp.split_indices(loaded["labels"], verbose=False)

    n = len(loaded["labels"])
    assert len(train_idx) + len(test_idx) == n
    assert len(test_idx) == round(n * config.TEST_SIZE)
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert np.union1d(train_idx, test_idx).size == n


def test_the_same_random_state_gives_the_same_split():
    run_step4()
    labels = sp.load_step4_outputs(verbose=False)["labels"]

    a_train, a_test = sp.split_indices(labels, random_state=42, verbose=False)
    b_train, b_test = sp.split_indices(labels, random_state=42, verbose=False)
    assert a_train.tolist() == b_train.tolist()
    assert a_test.tolist() == b_test.tolist()

    c_train, c_test = sp.split_indices(labels, random_state=7, verbose=False)
    assert c_test.tolist() != a_test.tolist(), "a different seed must split differently"


def test_stratified_split_keeps_class_proportions_and_fills_every_class():
    run_step4()
    labels = sp.load_step4_outputs(verbose=False)["labels"]
    train_idx, test_idx = sp.split_indices(labels, verbose=False)
    table, ok, info = sp.class_balance_table(labels, train_idx, test_idx,
                                             verbose=False)

    assert ok is True
    assert info["classes_without_test_rows"] == []
    rows = table[table[config.STRATIFY_COLUMN].astype(str) != "TOTAL"]
    assert (rows["test"] >= 1).all(), "every class needs at least one test row"
    # the rare 5-row class survives on both sides
    rare = rows[rows[config.STRATIFY_COLUMN] == "Web_Crwling"].iloc[0]
    assert rare["total"] == 5 and rare["train"] == 4 and rare["test"] == 1
    # and the proportions hold to within one row's worth of share
    assert info["worst_drift_pp"] < info["tolerance_pp"]


def test_the_binary_target_is_balanced_too():
    run_step4()
    labels = sp.load_step4_outputs(verbose=False)["labels"]
    train_idx, test_idx = sp.split_indices(labels, verbose=False)
    table, ok, _info = sp.class_balance_table(labels, train_idx, test_idx,
                                              column="Label_binary", verbose=False)
    assert ok is True
    rows = table[table["Label_binary"].astype(str) != "TOTAL"]
    assert len(rows) == 2, "benign and attack"


def test_features_labels_and_metadata_stay_aligned():
    """The whole reason the split is on indices rather than on the frames."""
    run_step4()
    loaded = sp.load_step4_outputs(verbose=False)
    train_idx, test_idx = sp.split_indices(loaded["labels"], verbose=False)

    for indices in (train_idx, test_idx):
        X = sp.take(loaded["features"], indices)
        y = sp.take(loaded["labels"], indices)
        meta = sp.take(loaded["metadata"], indices)

        assert len(X) == len(y) == len(meta) == len(indices)
        # row i of each frame is the same original row
        assert meta["Orig_Row"].tolist() == \
            loaded["metadata"]["Orig_Row"].iloc[indices].tolist()
        assert y[config.LABEL_COLUMN].tolist() == \
            loaded["labels"][config.LABEL_COLUMN].iloc[indices].tolist()
        assert X["Flow Duration"].tolist() == \
            loaded["features"]["Flow Duration"].iloc[indices].tolist()


# ===========================================================================
# 3. the scaler, refitted on train only
# ===========================================================================
def test_the_scaler_is_fitted_on_train_only():
    """Changing the TEST rows must not move the fit."""
    run_step4()
    loaded = sp.load_step4_outputs(verbose=False)
    train_idx, test_idx = sp.split_indices(loaded["labels"], verbose=False)
    X_train = sp.take(loaded["features"], train_idx)
    X_test = sp.take(loaded["features"], test_idx)

    _a, _b, scaler_a, _ia = sp.refit_scaler_on_train(X_train, X_test, verbose=False)

    tampered = X_test.copy()
    tampered["Flow Duration"] = tampered["Flow Duration"] * 1000 + 1e6
    _c, _d, scaler_b, _ib = sp.refit_scaler_on_train(X_train, tampered, verbose=False)

    assert np.allclose(scaler_a.data_min_, scaler_b.data_min_)
    assert np.allclose(scaler_a.data_max_, scaler_b.data_max_)


def test_train_is_inside_0_1_and_test_may_exceed_it():
    run_step4()
    loaded = sp.load_step4_outputs(verbose=False)
    train_idx, test_idx = sp.split_indices(loaded["labels"], verbose=False)
    X_train = sp.take(loaded["features"], train_idx)
    X_test = sp.take(loaded["features"], test_idx)

    scaled_train, scaled_test, _s, info = sp.refit_scaler_on_train(
        X_train, X_test, verbose=False)

    assert info["scaler_name"] == "minmax"
    assert info["train_min"] >= -1e-9 and info["train_max"] <= 1 + 1e-9
    # out-of-range test values are counted, not hidden, and not clipped
    assert "test_cells_outside_0_1" in info
    assert info["clipped"] is False
    block = scaled_test[info["columns"]].to_numpy(dtype="float64")
    counted = int((block < 0).sum() + (block > 1).sum())
    assert counted == info["test_cells_outside_0_1"]


def test_one_hot_and_binary_columns_are_not_scaled():
    run_step4()
    loaded = sp.load_step4_outputs(verbose=False)
    train_idx, test_idx = sp.split_indices(loaded["labels"], verbose=False)
    X_train = sp.take(loaded["features"], train_idx)
    X_test = sp.take(loaded["features"], test_idx)

    scaled_train, _st, _s, info = sp.refit_scaler_on_train(
        X_train, X_test, scaled_columns=loaded["scaled_columns"], verbose=False)

    untouched = [c for c in X_train.columns if c not in set(info["columns"])]
    assert set(config.COMBINED_DROP_FEATURE_COLUMNS) <= set(untouched)
    for col in untouched:
        assert scaled_train[col].tolist() == X_train[col].tolist(), col


def test_clipping_is_off_by_default_but_can_be_turned_on(monkeypatch):
    run_step4()
    loaded = sp.load_step4_outputs(verbose=False)
    train_idx, test_idx = sp.split_indices(loaded["labels"], verbose=False)
    X_train = sp.take(loaded["features"], train_idx)
    X_test = sp.take(loaded["features"], test_idx)

    assert config.CLIP_TEST_TO_TRAIN_RANGE is False
    _a, test_clipped, _s, info = sp.refit_scaler_on_train(
        X_train, X_test, clip=True, verbose=False)
    block = test_clipped[info["columns"]].to_numpy(dtype="float64")
    assert block.min() >= -1e-9 and block.max() <= 1 + 1e-9


# ===========================================================================
# 4. the leakage check
# ===========================================================================
def test_leakage_count_is_correct_on_a_constructed_example():
    columns = ["a", "b"]
    X_train = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=columns)
    # rows 0 and 2 are exact copies of training rows; rows 1 and 3 are not
    X_test = pd.DataFrame([[1.0, 2.0], [9.0, 9.0], [5.0, 6.0], [7.0, 8.0]],
                          columns=columns)
    y_test = pd.DataFrame({config.LABEL_COLUMN:
                           ["Port_Scan", "Port_Scan", "Normal", "Normal"]})

    table, overall, mask = sp.leakage_report(X_train, X_test, y_test, verbose=False)

    assert overall["test rows"] == 4
    assert overall["also in train"] == 2
    assert overall["% of test"] == 50.0
    assert mask.tolist() == [True, False, True, False]

    per_label = table.set_index(config.LABEL_COLUMN)
    assert per_label.loc["Port_Scan", "also in train"] == 1
    assert per_label.loc["Normal", "also in train"] == 1
    assert per_label.loc["Port_Scan", "% of this class's test rows"] == 50.0
    assert per_label.loc["TOTAL", "also in train"] == 2


def test_leakage_needs_the_same_columns_in_the_same_order():
    X_train = pd.DataFrame({"a": [1.0], "b": [2.0]})
    X_test = pd.DataFrame({"b": [2.0], "a": [1.0]})
    y_test = pd.DataFrame({config.LABEL_COLUMN: ["Normal"]})
    with pytest.raises(sp.SplittingError):
        sp.leakage_report(X_train, X_test, y_test, verbose=False)


def test_a_near_miss_is_not_counted_as_leakage():
    X_train = pd.DataFrame([[1.0, 2.0]], columns=["a", "b"])
    X_test = pd.DataFrame([[1.0, 2.0000001]], columns=["a", "b"])
    y_test = pd.DataFrame({config.LABEL_COLUMN: ["Normal"]})
    _t, overall, _m = sp.leakage_report(X_train, X_test, y_test, verbose=False)
    assert overall["also in train"] == 0, "the match must be exact, not approximate"


# ===========================================================================
# 5. end to end: run_step5, the saved files, and the reload path
# ===========================================================================
def test_run_step5_writes_and_verifies_every_file():
    run_step4()
    result = sp.run_step5(overwrite=True, verbose=False)

    assert result["ok"] is True
    folder = config.STEP5_OUTPUT_DIR
    for key, name in config.STEP5_FILES.items():
        if key == "chart":
            continue                      # matplotlib may not be installed
        assert (folder / name).exists(), name

    verification = result["saved"]["verification"]
    assert "[FAIL]" not in set(verification["result"])
    assert len(result["X_train"]) + len(result["X_test"]) == len(LABEL_PLAN)


def test_saved_indices_reproduce_the_split():
    run_step4()
    result = sp.run_step5(overwrite=True, verbose=False)
    folder = config.STEP5_OUTPUT_DIR

    train_pos = pd.read_csv(folder / config.STEP5_FILES["train_indices"])["row_position"]
    test_pos = pd.read_csv(folder / config.STEP5_FILES["test_indices"])["row_position"]

    assert np.intersect1d(train_pos, test_pos).size == 0
    assert sorted(train_pos.tolist() + test_pos.tolist()) == list(range(len(LABEL_PLAN)))

    loaded = sp.load_step4_outputs(verbose=False)
    y_test = pd.read_csv(folder / config.STEP5_FILES["y_test"])
    assert y_test[config.LABEL_COLUMN].tolist() == \
        loaded["labels"][config.LABEL_COLUMN].iloc[test_pos].tolist()


def test_x_train_and_x_test_have_identical_columns():
    run_step4()
    sp.run_step5(overwrite=True, verbose=False)
    folder = config.STEP5_OUTPUT_DIR

    X_train = pd.read_csv(folder / config.STEP5_FILES["X_train"], low_memory=False)
    X_test = pd.read_csv(folder / config.STEP5_FILES["X_test"], low_memory=False)
    assert list(X_train.columns) == list(X_test.columns)

    listed = json.loads((config.STEP4_OUTPUT_DIR / config.STEP4_FILES["feature_list"])
                        .read_text(encoding="utf-8"))["features"]
    assert list(X_train.columns) == listed
    for col in config.ID_COLUMNS_NOT_FOR_TRAINING:
        assert col not in X_train.columns, "%s must never be a model feature" % col


def test_the_view_files_use_the_preprocessing_csv_layout():
    run_step4()
    sp.run_step5(overwrite=True, verbose=False)
    folder = config.STEP5_OUTPUT_DIR

    for key in ("train_view", "test_view"):
        view = pd.read_csv(folder / config.STEP5_FILES[key], low_memory=False)
        assert list(view.columns[:6]) == list(config.COMBINED_ID_COLUMNS)
        assert list(view.columns[-1:]) == [config.LABEL_COLUMN]
        for col in config.COMBINED_DROP_FEATURE_COLUMNS:
            assert col not in view.columns, col
        assert set(view["Protocol"]) <= set(config.PROTOCOL_TEXT_VALUES)


def test_unscaled_files_are_the_untouched_step4_features():
    run_step4()
    result = sp.run_step5(overwrite=True, verbose=False)
    folder = config.STEP5_OUTPUT_DIR
    clean = pd.read_csv(config.STEP4_OUTPUT_DIR / config.STEP4_FILES["features_clean"],
                        low_memory=False)

    unscaled = pd.read_csv(folder / config.STEP5_FILES["X_train_unscaled"],
                           low_memory=False)
    assert np.allclose(unscaled.to_numpy(dtype="float64"),
                       clean.iloc[result["train_idx"]].to_numpy(dtype="float64"),
                       rtol=1e-9, equal_nan=True)


def test_outputs_are_loaded_instead_of_resplit_when_overwrite_is_off():
    run_step4()
    first = sp.run_step5(overwrite=True, verbose=False)
    again = sp.run_step5(overwrite=False, verbose=False)

    assert again["status"] == "loaded_existing"
    assert again["X_train"].shape == first["X_train"].shape
    assert again["train_idx"].tolist() == first["train_idx"].tolist()


def test_step5_does_not_touch_the_step4_outputs():
    run_step4()
    folder4 = config.STEP4_OUTPUT_DIR
    before = {name: (folder4 / name).read_bytes()
              for name in config.STEP4_FILES.values() if (folder4 / name).exists()}

    sp.run_step5(overwrite=True, verbose=False)

    for name, content in before.items():
        assert (folder4 / name).read_bytes() == content, name


def test_step5_does_no_resampling_or_class_weighting():
    """Every input row lands on exactly one side, unchanged and uncopied."""
    run_step4()
    result = sp.run_step5(overwrite=True, verbose=False)

    counts = pd.Series(LABEL_PLAN).value_counts()
    for name, total in counts.items():
        n_train = int((result["y_train"][config.LABEL_COLUMN] == name).sum())
        n_test = int((result["y_test"][config.LABEL_COLUMN] == name).sum())
        assert n_train + n_test == total, name
    # the imbalance is carried through untouched, not evened out
    train_counts = result["y_train"][config.LABEL_COLUMN].value_counts()
    assert train_counts.max() / train_counts.min() > 10


# ===========================================================================
# 6. the Windows-path guard
# ===========================================================================
def test_step5_refuses_a_windows_path_off_windows(tmp_path, monkeypatch, capsys,
                                                  assert_no_leaked_output):
    """On Linux, Path(r"C:\\...") is RELATIVE -- nothing may be created."""
    monkeypatch.setattr(config, "STEP5_OUTPUT_DIR", config.Path(r"C:\MSCAD2022\outputs\step5"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    folder = sp.prepare_output_dir(config.STEP5_OUTPUT_DIR, "the Step 5 outputs")

    assert folder is None
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("prepare_output_dir for the Step 5 folder")


def test_step5_output_dir_is_under_data_root_not_the_repo():
    assert REAL_STEP5_DIR.parent.parent == config.DATA_ROOT
    assert config.PROJECT_ROOT not in REAL_STEP5_DIR.parents
    assert REAL_STEP5_DIR.name == "step5"
