"""
tests/test_preprocessing.py -- pytest tests for STEP 4 (clean, encode, scale).

Everything is built from small synthetic frames in a temporary folder: no real
captures, no merged dataset, and never C:\\MSCAD2022.  An autouse fixture
redirects config.STEP4_OUTPUT_DIR to tmp_path for every test, so no test can
write to a configured folder on any operating system.

    cd /d C:\\MSCAD2022\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pytest tests -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.mscad import preprocessing as pp

REAL_STEP4_DIR = config.STEP4_OUTPUT_DIR          # captured before any patching


@pytest.fixture(autouse=True)
def _never_write_outside_tmp(tmp_path, monkeypatch):
    """Step 4 output always goes to tmp_path, never to the configured folder."""
    monkeypatch.setattr(config, "STEP4_OUTPUT_DIR", tmp_path / "step4")


# ===========================================================================
# the synthetic dataset: inf, NaN, a constant column, duplicates, odd protocol
# ===========================================================================
def make_merged(n_rows: int = 12) -> pd.DataFrame:
    """A miniature merged dataset with every awkward case Step 4 must handle."""
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}

    for col in config.KAGGLE_FEATURES:
        data[col] = rng.uniform(1, 100, n_rows).round(3)

    # zero-duration flows -> inf in the rate columns, exactly as Step 1 produces
    data["Flow Duration"] = [0.0, 0.0, 0.0] + list(rng.uniform(10, 5000, n_rows - 3))
    for col in ("Flow Byts/s", "Flow Pkts/s", "Fwd Pkts/s"):
        values = list(rng.uniform(1, 500, n_rows))
        values[0] = values[1] = values[2] = np.inf
        data[col] = values

    data["Bwd PSH Flags"] = [0.0] * n_rows        # constant -> must be dropped
    data["CWE Flag Count"] = [0.0] * n_rows       # constant -> must be dropped
    data["Pkt Len Mean"] = [np.nan] + list(rng.uniform(40, 1500, n_rows - 1))  # a NaN

    data["Src IP"] = ["192.168.159.%d" % (i % 5 + 1) for i in range(n_rows)]
    data["Dst IP"] = ["192.168.159.20"] * n_rows
    data["Src Port"] = list(range(40000, 40000 + n_rows))
    data["Dst Port"] = [80] * n_rows
    # 6 TCP, 17 UDP, 1 ICMP and 2 (IGMP) -- the unknown one must land in _other
    data["Protocol"] = ([6] * (n_rows - 3)) + [17, 1, 2]
    data["Timestamp"] = ["2020-01-01 18:%02d:%02d.000000" % (i // 60, i % 60)
                         for i in range(n_rows)]

    labels = (["Normal"] * 3 + ["Benign"] * 2 + ["Port_Scan"] * 3
              + ["Brute_Force"] * 2 + ["Web_Crwling"] * 2)
    data[config.LABEL_COLUMN] = labels[:n_rows]

    df = pd.DataFrame(data)[config.FLOW_COLUMNS]
    df["Source_Capture"] = (["N-0"] * 3 + ["Scan-1"] * 3 + ["W-B-02"] * 3
                            + ["W-B-01"] * 3)[:n_rows]
    df["Orig_Row"] = list(range(n_rows))
    df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"])

    # Two rows sharing identical feature values -> duplicates that must be KEPT.
    # Rows 5 and 6 are both Port_Scan on protocol 6, so they stay identical after
    # the Protocol one-hot as well (rows 10 and 11 would not: ICMP vs IGMP).
    feature_cols = [c for c in config.KAGGLE_FEATURES if c in df.columns]
    df.loc[df.index[6], feature_cols] = df.loc[df.index[5], feature_cols].to_numpy()
    return df


# ===========================================================================
# 1. inf
# ===========================================================================
def test_cap_replaces_inf_with_the_column_maximum():
    df = make_merged()
    before = int(np.isinf(df["Flow Byts/s"].to_numpy(dtype="float64")).sum())
    expected_cap = pp.finite_max(df["Flow Byts/s"])

    features, _labels, _meta, info = pp.clean_dataset(df, verbose=False)

    assert before == 3
    assert int(np.isinf(features["Flow Byts/s"].to_numpy(dtype="float64")).sum()) == 0
    assert features["Flow Byts/s"].max() == pytest.approx(expected_cap)
    assert info["inf"]["strategy"] == "cap"
    assert info["inf"]["inf_before_total"] == 9        # 3 columns x 3 rows


def test_zero_duration_column_marks_the_right_rows():
    df = make_merged()
    features, _l, _m, info = pp.clean_dataset(df, verbose=False)

    marker = features[config.ZERO_DURATION_COLUMN]
    assert marker.tolist()[:3] == [1, 1, 1]
    assert marker.sum() == 3
    assert info["inf"]["zero_duration_rows"] == 3
    # and it matches Flow Duration == 0 exactly
    assert (marker.to_numpy() == (df["Flow Duration"].to_numpy() == 0)).all()


def test_nan_median_strategy_leaves_no_inf_and_adds_no_marker():
    df = make_merged()
    features, report, info = pp.handle_infinities(
        df[[c for c in config.KAGGLE_FEATURES]], strategy="nan_median", verbose=False)

    assert int(np.isinf(features.to_numpy(dtype="float64")).sum()) == 0
    assert config.ZERO_DURATION_COLUMN not in features.columns
    assert info["strategy"] == "nan_median"
    assert len(report) == 3


def test_no_inf_or_nan_left_in_either_output(tmp_path):
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)
    for key in ("features_clean", "features_scaled"):
        arr = result[key].to_numpy(dtype="float64")
        assert int(np.isnan(arr).sum()) == 0, key
        assert int(np.isinf(arr).sum()) == 0, key


# ===========================================================================
# 2. NaN
# ===========================================================================
def test_nan_is_median_imputed_and_reported():
    df = make_merged()
    assert df["Pkt Len Mean"].isna().sum() == 1
    expected = float(df["Pkt Len Mean"].median(skipna=True))

    features, _l, _m, info = pp.clean_dataset(df, verbose=False)

    assert features["Pkt Len Mean"].isna().sum() == 0
    assert features["Pkt Len Mean"].iloc[0] == pytest.approx(expected)
    assert "Pkt Len Mean" in info["nan"]["columns_imputed"]


# ===========================================================================
# 3. constant columns
# ===========================================================================
def test_constant_columns_are_dropped_and_listed():
    df = make_merged()
    features, _l, _m, info = pp.clean_dataset(df, verbose=False)

    assert "Bwd PSH Flags" not in features.columns
    assert "CWE Flag Count" not in features.columns
    assert set(["Bwd PSH Flags", "CWE Flag Count"]) <= set(info["constant_dropped"])
    # a varying column of the same family is NOT dropped
    assert "Flow Duration" in features.columns


def test_a_column_that_varies_is_kept():
    df = make_merged()
    df["Bwd PSH Flags"] = list(range(len(df)))       # no longer constant
    _f, _l, _m, info = pp.clean_dataset(df, verbose=False)
    assert "Bwd PSH Flags" not in info["constant_dropped"]


# ===========================================================================
# 4. metadata and row order
# ===========================================================================
def test_metadata_columns_are_not_in_the_features():
    df = make_merged()
    features, labels, metadata, _info = pp.clean_dataset(df, verbose=False)

    for col in config.METADATA_COLUMNS:
        assert col not in features.columns, col
        assert col in metadata.columns, col
    assert config.LABEL_COLUMN not in features.columns
    assert config.LABEL_COLUMN in labels.columns
    # Protocol is a FEATURE, not metadata
    assert "Protocol" not in metadata.columns


def test_row_count_and_row_order_are_preserved():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)

    n = len(df)
    for key in ("features_clean", "features_scaled", "labels", "metadata"):
        assert len(result[key]) == n, key
    # row i still refers to the same flow
    assert result["metadata"]["Orig_Row"].tolist() == df["Orig_Row"].tolist()
    assert result["metadata"]["Src IP"].tolist() == df["Src IP"].tolist()
    assert result["labels"][config.LABEL_COLUMN].tolist() == df[config.LABEL_COLUMN].tolist()


def test_duplicate_rows_are_kept():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)

    features = result["features_clean"]
    assert len(features) == len(df), "no row may be dropped"
    assert features.duplicated().sum() >= 1, "the duplicate pair must survive"
    dup_report = result["clean_info"]["reports"]["duplicates"]
    assert int(dup_report["rows"].sum()) == len(df)


# ===========================================================================
# 5. labels
# ===========================================================================
def test_label_encoding_and_binary_target():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)
    labels = result["labels"]

    assert {"Label", "Label_encoded", "Label_binary"} <= set(labels.columns)
    assert labels["Label"].tolist() == df[config.LABEL_COLUMN].tolist()   # original kept

    benign = labels[labels["Label"].isin(["Benign", "Normal"])]
    attack = labels[~labels["Label"].isin(["Benign", "Normal"])]
    assert set(benign["Label_binary"]) == {0}
    assert set(attack["Label_binary"]) == {1}

    # Benign and Normal stay SEPARATE multi-class labels
    mapping = result["encode_info"]["label_mapping"]
    assert mapping["Benign"] != mapping["Normal"]


def test_label_mapping_is_saved_and_consistent(tmp_path):
    import json
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)

    path = config.STEP4_OUTPUT_DIR / config.STEP4_FILES["label_mapping"]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == result["encode_info"]["label_mapping"]
    for name, code in saved.items():
        rows = result["labels"][result["labels"]["Label"] == name]
        assert set(rows["Label_encoded"]) == {code}, name


def test_label_map_is_not_applied_when_empty():
    df = make_merged()
    assert config.LABEL_MAP == {}, "the default must rename nothing"
    labels_out, report, applied = pp.apply_label_map(
        df[[config.LABEL_COLUMN]], verbose=False)

    assert applied == {}
    assert labels_out[config.LABEL_COLUMN].tolist() == df[config.LABEL_COLUMN].tolist()
    assert "Web_Crwling" in set(labels_out[config.LABEL_COLUMN])   # dataset spelling kept
    assert (report["renamed_to"] == "").all()


def test_label_map_is_applied_when_set():
    df = make_merged()
    labels_out, report, applied = pp.apply_label_map(
        df[[config.LABEL_COLUMN]], label_map={"Web_Crwling": "Web_Crawling"},
        verbose=False)

    assert applied == {"Web_Crwling": "Web_Crawling"}
    assert "Web_Crawling" in set(labels_out[config.LABEL_COLUMN])
    assert "Web_Crwling" not in set(labels_out[config.LABEL_COLUMN])
    assert (report.set_index("Label").loc["Web_Crwling", "renamed_to"] == "Web_Crawling")
    # every other label untouched
    assert (set(labels_out[config.LABEL_COLUMN]) - {"Web_Crawling"}
            == set(df[config.LABEL_COLUMN]) - {"Web_Crwling"})


# ===========================================================================
# 6. protocol one-hot
# ===========================================================================
def test_protocol_is_one_hot_encoded_with_an_other_bucket():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)
    features = result["features_clean"]

    for col in ("Protocol_ICMP", "Protocol_TCP", "Protocol_UDP", "Protocol_other"):
        assert col in features.columns, col
    assert "Protocol" not in features.columns

    # protocol 2 (IGMP) is unknown -> Protocol_other
    assert features["Protocol_other"].sum() == 1
    assert features["Protocol_ICMP"].sum() == 1
    assert features["Protocol_UDP"].sum() == 1
    # exactly one of the four is set on every row
    four = features[["Protocol_ICMP", "Protocol_TCP", "Protocol_UDP", "Protocol_other"]]
    assert set(four.sum(axis=1)) == {1}


# ===========================================================================
# 7. scaling
# ===========================================================================
def test_minmax_scaled_values_are_within_0_and_1():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)

    scaled_cols = result["artifacts"]["scaled_columns"]
    block = result["features_scaled"][scaled_cols].to_numpy(dtype="float64")
    assert block.min() >= -1e-9
    assert block.max() <= 1 + 1e-9


def test_one_hot_and_binary_columns_are_not_scaled():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)

    untouched = ["Protocol_ICMP", "Protocol_TCP", "Protocol_UDP", "Protocol_other",
                 config.ZERO_DURATION_COLUMN]
    for col in untouched:
        assert col not in result["artifacts"]["scaled_columns"], col
        pd.testing.assert_series_equal(result["features_scaled"][col],
                                       result["features_clean"][col])


def test_standard_scaler_can_be_selected():
    df = make_merged()
    features, labels, _m, _i = pp.clean_dataset(df, verbose=False)
    features, labels, enc = pp.encode_dataset(features, labels, verbose=False)
    scaled, scaler, cols = pp.scale_features(
        features, scaler_name="standard",
        protocol_columns=enc["protocol_columns"], verbose=False)

    assert type(scaler).__name__ == "StandardScaler"
    means = scaled[cols].to_numpy(dtype="float64").mean(axis=0)
    assert np.allclose(means, 0, atol=1e-9)


def test_unscaled_features_are_saved_too(tmp_path):
    """So a later step can refit the scaler on a training split only."""
    df = make_merged()
    pp.run_step4(df, overwrite=True, verbose=False)

    clean = pd.read_csv(config.STEP4_OUTPUT_DIR / config.STEP4_FILES["features_clean"])
    scaled = pd.read_csv(config.STEP4_OUTPUT_DIR / config.STEP4_FILES["features_scaled"])
    assert list(clean.columns) == list(scaled.columns)
    assert len(clean) == len(scaled) == len(df)
    assert clean["Flow Duration"].max() > 1.0, "the clean file must NOT be scaled"


# ===========================================================================
# 8. saving, re-reading, overwrite
# ===========================================================================
def test_every_output_file_is_written_and_re_read(tmp_path):
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)

    folder = config.STEP4_OUTPUT_DIR
    for key, name in config.STEP4_FILES.items():
        assert (folder / name).exists(), name

    verification = result["saved"]["verification"]
    assert set(verification["result"]) == {"[OK]"}
    for _, row in verification.iterrows():
        if row["rows_on_disk"] != "-":
            assert row["rows_on_disk"] == row["rows_in_memory"] == len(df)
    assert result["saved"]["ok"] is True
    assert result["ok"] is True


def test_outputs_are_loaded_instead_of_recomputed_when_overwrite_is_off(tmp_path):
    df = make_merged()
    pp.run_step4(df, overwrite=True, verbose=False)
    assert pp.step4_outputs_exist() is True

    again = pp.run_step4(df, overwrite=False, verbose=False)
    assert again["status"] == "loaded_existing"
    assert len(again["features_clean"]) == len(df)
    assert "label_mapping" in again


def test_overwrite_true_recomputes():
    df = make_merged()
    pp.run_step4(df, overwrite=True, verbose=False)
    smaller = make_merged(8)
    again = pp.run_step4(smaller, overwrite=True, verbose=False)
    assert again["status"] == "written"
    assert len(again["features_clean"]) == 8


def test_checks_all_pass_on_clean_output():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)
    checks = result["checks"]
    assert set(checks["result"]) == {"[OK]"}, checks.to_string()


def test_report_lists_every_action():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)
    report = result["report"]
    stages = set(report["stage"])
    for expected in ("4.1 clean", "4.1a inf", "4.1b nan", "4.1c constant",
                     "4.1d metadata", "4.1e duplicates", "4.1f labels",
                     "4.2 encode", "4.3 scale", "4.3 checks"):
        assert expected in stages, expected


# ===========================================================================
# 9. scope: Step 4 must NOT split or resample
# ===========================================================================
def test_step4_does_not_split_or_resample():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)

    assert len(result["features_clean"]) == len(df), "no split, no resampling"
    before = df[config.LABEL_COLUMN].value_counts().to_dict()
    after = result["labels"]["Label"].value_counts().to_dict()
    assert before == after, "the class balance must be untouched"
    assert "train" not in result and "test" not in result


# ===========================================================================
# 10. path safety
# ===========================================================================
def test_step4_refuses_a_windows_path_off_windows(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "STEP4_OUTPUT_DIR", Path(r"C:\MSCAD2022\outputs\step4"))
    monkeypatch.setattr(config.os, "name", "posix")
    monkeypatch.chdir(tmp_path)

    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=True)
    capsys.readouterr()

    assert result["saved"]["status"] == "skipped_not_windows"
    assert list(tmp_path.iterdir()) == [], "no 'C:\\\\MSCAD2022' folder may be created"


def test_step4_output_dir_is_under_data_root_not_the_repo():
    assert str(config.DATA_ROOT) in str(REAL_STEP4_DIR)
    assert str(config.PROJECT_ROOT) not in str(REAL_STEP4_DIR)
    assert REAL_STEP4_DIR == config.OUTPUTS_DIR / "step4"


# ===========================================================================
# 11. the combined file -- preprocessing.csv
# ===========================================================================
def test_combined_file_is_written_with_features_then_labels():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)

    path = config.STEP4_OUTPUT_DIR / config.STEP4_FILES["combined"]
    assert path.exists() and path.name == "preprocessing.csv"

    on_disk = pd.read_csv(path, low_memory=False)
    features = result["features_scaled"]

    assert len(on_disk) == len(df)
    assert on_disk.shape[1] == features.shape[1] + 3
    # features first, in feature order; the three label columns last
    assert list(on_disk.columns[:features.shape[1]]) == list(features.columns)
    assert list(on_disk.columns[-3:]) == ["Label", "Label_encoded", "Label_binary"]
    assert result["combined"]["ok"] is True


def test_combined_row_order_matches_the_separate_files():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)
    folder = config.STEP4_OUTPUT_DIR

    combined = pd.read_csv(folder / config.STEP4_FILES["combined"], low_memory=False)
    labels = pd.read_csv(folder / config.STEP4_FILES["labels"], low_memory=False)
    scaled = pd.read_csv(folder / config.STEP4_FILES["features_scaled"], low_memory=False)

    assert combined["Label"].tolist() == labels["Label"].tolist()
    assert combined["Label_encoded"].tolist() == labels["Label_encoded"].tolist()
    assert combined["Label_binary"].tolist() == labels["Label_binary"].tolist()
    # row i is the same flow, and it is the ORIGINAL order
    assert combined["Label"].tolist() == df[config.LABEL_COLUMN].tolist()
    for col in scaled.columns:
        assert np.allclose(combined[col].to_numpy(dtype="float64"),
                           scaled[col].to_numpy(dtype="float64"),
                           rtol=1e-9, equal_nan=True), col


def test_combined_uses_scaled_features_by_default():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)
    combined = pd.read_csv(config.STEP4_OUTPUT_DIR / config.STEP4_FILES["combined"])

    scaled_cols = result["artifacts"]["scaled_columns"]
    block = combined[scaled_cols].to_numpy(dtype="float64")
    assert block.min() >= -1e-9 and block.max() <= 1 + 1e-9, "must be the SCALED features"


def test_combined_use_scaled_false_takes_the_unscaled_features(monkeypatch):
    df = make_merged()
    monkeypatch.setattr(config, "COMBINED_USE_SCALED", False)
    result = pp.run_step4(df, overwrite=True, verbose=False)

    combined = pd.read_csv(config.STEP4_OUTPUT_DIR / config.STEP4_FILES["combined"])
    clean = result["features_clean"]
    assert combined["Flow Duration"].max() == pytest.approx(clean["Flow Duration"].max())
    assert combined["Flow Duration"].max() > 1.0, "unscaled values are not in [0,1]"


def test_combined_include_metadata_puts_metadata_first(monkeypatch):
    df = make_merged()
    monkeypatch.setattr(config, "COMBINED_INCLUDE_METADATA", True)
    result = pp.run_step4(df, overwrite=True, verbose=False)

    combined = pd.read_csv(config.STEP4_OUTPUT_DIR / config.STEP4_FILES["combined"],
                           low_memory=False)
    meta_cols = list(result["metadata"].columns)
    n_features = result["features_scaled"].shape[1]

    assert list(combined.columns[:len(meta_cols)]) == meta_cols
    assert combined.shape[1] == len(meta_cols) + n_features + 3
    assert list(combined.columns[-3:]) == ["Label", "Label_encoded", "Label_binary"]
    assert combined["Orig_Row"].tolist() == df["Orig_Row"].tolist()
    assert result["combined"]["ok"] is True


def test_combined_defaults_exclude_metadata():
    assert config.COMBINED_INCLUDE_METADATA is False
    assert config.COMBINED_USE_SCALED is True
    df = make_merged()
    pp.run_step4(df, overwrite=True, verbose=False)
    combined = pd.read_csv(config.STEP4_OUTPUT_DIR / config.STEP4_FILES["combined"],
                           low_memory=False)
    for col in config.METADATA_COLUMNS:
        assert col not in combined.columns, col


def test_building_from_saved_files_gives_the_same_result():
    """The skip path (OVERWRITE_STEP4 False) must produce an identical file."""
    df = make_merged()
    pp.run_step4(df, overwrite=True, verbose=False)
    path = config.STEP4_OUTPUT_DIR / config.STEP4_FILES["combined"]
    from_run = pd.read_csv(path, low_memory=False)

    path.unlink()                                  # force a rebuild from the saved files
    rebuilt = pp.build_combined_from_saved(verbose=False)

    assert rebuilt["ok"] is True
    from_saved = pd.read_csv(path, low_memory=False)
    pd.testing.assert_frame_equal(from_run, from_saved)


def test_skip_path_rebuilds_the_combined_file():
    df = make_merged()
    pp.run_step4(df, overwrite=True, verbose=False)
    again = pp.run_step4(df, overwrite=False, verbose=False)

    assert again["status"] == "loaded_existing"
    assert again["combined"]["ok"] is True
    assert (config.STEP4_OUTPUT_DIR / config.STEP4_FILES["combined"]).exists()


def test_mismatched_row_counts_fail_instead_of_padding_with_nan():
    df = make_merged()
    features, labels, metadata, _i = pp.clean_dataset(df, verbose=False)
    features, labels, _e = pp.encode_dataset(features, labels, verbose=False)

    with pytest.raises(pp.PreprocessingError) as err:
        pp.build_combined(features, labels.iloc[:5], verbose=False)
    assert "different row counts" in str(err.value)
    assert "NaN" in str(err.value)


def test_combined_verification_detects_a_tampered_file():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)
    path = config.STEP4_OUTPUT_DIR / config.STEP4_FILES["combined"]

    # drop a row on disk: the re-read check must catch it
    tampered = pd.read_csv(path, low_memory=False).iloc[:-1]
    tampered.to_csv(path, index=False)

    table, ok, _on_disk = pp.verify_combined(
        result["features_scaled"], result["features_scaled"], result["labels"],
        path=path, verbose=False)
    assert ok is False
    assert "[FAIL]" in set(table["result"])


def test_combined_has_no_nan_or_inf():
    df = make_merged()
    result = pp.run_step4(df, overwrite=True, verbose=False)
    combined = pd.read_csv(config.STEP4_OUTPUT_DIR / config.STEP4_FILES["combined"],
                           low_memory=False)
    features = [c for c in combined.columns if c not in pp.LABEL_OUTPUT_COLUMNS]
    block = combined[features].to_numpy(dtype="float64")
    assert int(np.isnan(block).sum()) == 0
    assert int(np.isinf(block).sum()) == 0


def test_other_step4_files_are_unchanged_by_the_combined_file():
    """Adding preprocessing.csv must not alter the eight existing outputs."""
    df = make_merged()
    pp.run_step4(df, overwrite=True, verbose=False)
    folder = config.STEP4_OUTPUT_DIR
    before = {name: (folder / name).read_bytes()
              for key, name in config.STEP4_FILES.items() if key != "combined"}

    (folder / config.STEP4_FILES["combined"]).unlink()
    pp.build_combined_from_saved(verbose=False)

    for name, content in before.items():
        assert (folder / name).read_bytes() == content, name
