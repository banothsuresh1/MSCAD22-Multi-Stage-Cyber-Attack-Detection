"""
tests/test_data_loading.py -- pytest tests for src/mscad/data_loading.py (Step 2).

Everything here is built from tiny fake CSVs in a temporary folder: no real
captures, no Wireshark, no scapy, and the real C:\\MSCAD2022 data is never read
or written.

    cd /d C:\\MSCAD2022\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pytest tests -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.mscad import data_loading as dl


# ===========================================================================
# helpers
# ===========================================================================
def fake_flow_frame(n_rows: int = 3, label: str = "Normal",
                    src_ip: str = "192.168.159.1",
                    start_second: int = 0, **overrides) -> pd.DataFrame:
    """A DataFrame with EXACTLY config.FLOW_COLUMNS, in order."""
    data = {}
    for col in config.FLOW_COLUMNS:
        data[col] = [1.0] * n_rows
    data["Src IP"] = [src_ip] * n_rows
    data["Dst IP"] = ["192.168.159.20"] * n_rows
    data["Src Port"] = [40000 + i for i in range(n_rows)]
    data["Dst Port"] = [80] * n_rows
    data["Protocol"] = [6] * n_rows
    data["Timestamp"] = ["2020-09-13 12:%02d:%02d.000000"
                         % ((start_second + i) // 60, (start_second + i) % 60)
                         for i in range(n_rows)]
    data[config.LABEL_COLUMN] = [label] * n_rows
    data.update(overrides)
    return pd.DataFrame(data)[config.FLOW_COLUMNS]


def write_flow_csv(folder: Path, capture: str, **kwargs) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("%s_flows.csv" % capture)
    fake_flow_frame(**kwargs).to_csv(path, index=False)
    return path


def six_captures(folder: Path) -> Path:
    """One small CSV per documented capture, with distinct labels and times."""
    plan = [("N-0", "Normal", 4, 0), ("Scan-1", "Port_Scan", 5, 10),
            ("App-01", "DoS slowloris", 3, 20), ("App-02", "ICMP_Flood", 2, 30),
            ("W-B-01", "Web_Crwling", 6, 40), ("W-B-02", "Brute_Force", 7, 50)]
    for capture, label, rows, start in plan:
        write_flow_csv(folder, capture, n_rows=rows, label=label, start_second=start)
    return folder


# ===========================================================================
# 1. discovery picks only *_flows.csv
# ===========================================================================
def test_find_flow_csvs_picks_only_the_flow_pattern(tmp_path):
    six_captures(tmp_path)
    # decoys that must all be ignored
    (tmp_path / "raw").mkdir()
    fake_flow_frame().to_csv(tmp_path / "raw" / "N-0_raw.csv", index=False)
    fake_flow_frame().to_csv(tmp_path / "MSCAD.csv", index=False)
    (tmp_path / "notes.txt").write_text("not a csv")
    (tmp_path / "N-0_flows.csv.bak").write_text("backup")

    files, folder = dl.find_flow_csvs(tmp_path, verbose=False)

    assert folder == tmp_path
    assert len(files) == 6
    names = [f.path.name for f in files]
    assert all(n.endswith("_flows.csv") for n in names)
    assert "N-0_raw.csv" not in names and "MSCAD.csv" not in names


def test_find_flow_csvs_does_not_recurse_into_raw(tmp_path):
    """glob() is deliberately non-recursive, so CSVs\\raw\\ is never touched."""
    six_captures(tmp_path)
    (tmp_path / "raw").mkdir(exist_ok=True)
    fake_flow_frame().to_csv(tmp_path / "raw" / "Scan-1_flows.csv", index=False)

    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)

    assert len(files) == 6
    assert all(f.path.parent == tmp_path for f in files)


def test_files_come_back_in_capture_order(tmp_path):
    six_captures(tmp_path)
    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    assert [f.capture for f in files] == config.CAPTURE_ORDER


def test_missing_capture_raises_instead_of_merging_a_partial_dataset(tmp_path):
    six_captures(tmp_path)
    (tmp_path / "App-02_flows.csv").unlink()

    with pytest.raises(dl.MissingCapturesError) as err:
        dl.find_flow_csvs(tmp_path, verbose=False)
    assert "App-02" in str(err.value)

    # ...but it can be allowed explicitly
    files, _ = dl.find_flow_csvs(tmp_path, require_all=False, verbose=False)
    assert len(files) == 5


def test_empty_folder_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(dl.MissingCapturesError):
        dl.find_flow_csvs(tmp_path / "empty", verbose=False)


def test_row_count_and_size_are_reported(tmp_path):
    write_flow_csv(tmp_path, "N-0", n_rows=42)
    files, _ = dl.find_flow_csvs(tmp_path, require_all=False, verbose=False)
    assert files[0].rows == 42                 # header excluded
    assert files[0].size_mb >= 0


# ===========================================================================
# 2. Source_Capture comes from the file name
# ===========================================================================
def test_capture_name_is_taken_from_the_file_name(tmp_path):
    for capture in ("N-0", "W-B-02", "App-01"):
        p = write_flow_csv(tmp_path, capture)
        assert dl.FlowCsv(p, 0, 0).capture == capture


def test_source_capture_column_matches_the_file_it_came_from(tmp_path):
    six_captures(tmp_path)
    merged, infos = dl.merge_flow_csvs(files=dl.find_flow_csvs(tmp_path, verbose=False)[0],
                                       verbose=False)
    assert set(merged["Source_Capture"]) == set(config.CAPTURE_ORDER)
    for capture, expected in [("N-0", 4), ("W-B-02", 7)]:
        assert (merged["Source_Capture"] == capture).sum() == expected


def test_orig_row_restarts_at_zero_for_each_capture(tmp_path):
    six_captures(tmp_path)
    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    merged, _ = dl.merge_flow_csvs(files=files, verbose=False)
    for capture, rows in [("N-0", 4), ("W-B-02", 7)]:
        got = sorted(merged.loc[merged["Source_Capture"] == capture, "Orig_Row"])
        assert got == list(range(rows))


# ===========================================================================
# 3. column mismatch raises a clear error
# ===========================================================================
def test_missing_column_raises_and_names_it(tmp_path):
    df = fake_flow_frame().drop(columns=["Idle Max"])
    with pytest.raises(dl.ColumnMismatchError) as err:
        dl.validate_columns(df, source="broken.csv", verbose=False)
    msg = str(err.value)
    assert "broken.csv" in msg and "Idle Max" in msg and "missing" in msg


def test_extra_column_raises_and_names_it(tmp_path):
    df = fake_flow_frame()
    df["Surprise"] = 1
    with pytest.raises(dl.ColumnMismatchError) as err:
        dl.validate_columns(df, verbose=False)
    assert "Surprise" in str(err.value) and "extra" in str(err.value)


def test_wrong_order_raises_and_shows_the_positions():
    df = fake_flow_frame()
    cols = list(df.columns)
    cols[1], cols[2] = cols[2], cols[1]        # swap Dst IP and Src Port
    with pytest.raises(dl.ColumnMismatchError) as err:
        dl.validate_columns(df[cols], verbose=False)
    msg = str(err.value)
    assert "WRONG ORDER" in msg and "position  2" in msg


def test_a_correct_header_passes_silently():
    dl.validate_columns(fake_flow_frame(), verbose=False)       # must not raise


def test_timestamp_dt_does_not_break_validation():
    df = fake_flow_frame()
    df[dl.TIMESTAMP_DT_COLUMN] = pd.to_datetime(df["Timestamp"])
    dl.validate_columns(df, verbose=False)                      # must not raise


# ===========================================================================
# 4. merged rows == sum of inputs
# ===========================================================================
def test_merged_rows_equal_the_sum_of_the_inputs(tmp_path):
    six_captures(tmp_path)
    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    expected = sum(f.rows for f in files)
    merged, infos = dl.merge_flow_csvs(files=files, verbose=False)
    assert len(merged) == expected == 4 + 5 + 3 + 2 + 6 + 7


def test_merged_column_layout(tmp_path):
    six_captures(tmp_path)
    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    merged, _ = dl.merge_flow_csvs(files=files, verbose=False)
    assert list(merged.columns) == config.MERGED_COLUMNS
    assert list(merged.columns[:73]) == config.FLOW_COLUMNS
    assert list(merged.columns[73:]) == ["Source_Capture", "Orig_Row", "Timestamp_dt"]
    assert len(merged.columns) == 76


def test_check_merge_all_pass_on_a_good_merge(tmp_path):
    six_captures(tmp_path)
    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    merged, infos = dl.merge_flow_csvs(files=files, verbose=False)
    checks, ok = dl.check_merge(merged, infos, verbose=False)
    assert ok is True
    assert all(c["result"] == "PASS" for c in checks)
    assert any(c["check"].startswith("merged rows") for c in checks)


def test_check_merge_reports_fail_when_rows_are_lost(tmp_path):
    """A dropped row must be caught by the checks, not slip through."""
    six_captures(tmp_path)
    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    merged, infos = dl.merge_flow_csvs(files=files, verbose=False)
    broken = merged.iloc[:-1]                               # lose one row
    checks, ok = dl.check_merge(broken, infos, verbose=False)
    assert ok is False
    failed = [c for c in checks if c["result"] == "FAIL"]
    assert any("merged rows" in c["check"] for c in failed)


# ===========================================================================
# 5. Label counts preserved
# ===========================================================================
def test_label_counts_are_preserved_by_the_merge(tmp_path):
    six_captures(tmp_path)
    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    per_file = {}
    for f in files:
        df, _ = dl.load_one_csv(f.path, verbose=False)
        for label, n in df[config.LABEL_COLUMN].value_counts().items():
            per_file[str(label)] = per_file.get(str(label), 0) + int(n)

    merged, _ = dl.merge_flow_csvs(files=files, verbose=False)
    after = {str(k): int(v) for k, v in merged[config.LABEL_COLUMN].value_counts().items()}
    assert after == per_file
    assert after == {"Brute_Force": 7, "Web_Crwling": 6, "Port_Scan": 5,
                     "Normal": 4, "DoS slowloris": 3, "ICMP_Flood": 2}


def test_label_spelling_is_not_touched(tmp_path):
    write_flow_csv(tmp_path, "W-B-01", label="Web_Crwling")
    df, _ = dl.load_one_csv(tmp_path / "W-B-01_flows.csv", verbose=False)
    assert set(df[config.LABEL_COLUMN]) == {"Web_Crwling"}     # the dataset's own spelling


# ===========================================================================
# 6. inf / NaN preserved
# ===========================================================================
def test_inf_and_nan_survive_loading(tmp_path):
    df = fake_flow_frame(4)
    df.loc[0, "Flow Byts/s"] = np.inf
    df.loc[1, "Flow Pkts/s"] = -np.inf
    df.loc[2, "Pkt Len Mean"] = np.nan
    (tmp_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path / "N-0_flows.csv", index=False)

    loaded, info = dl.load_one_csv(tmp_path / "N-0_flows.csv", verbose=False)

    assert np.isinf(loaded["Flow Byts/s"].iloc[0])
    assert np.isneginf(loaded["Flow Pkts/s"].iloc[1])
    assert pd.isna(loaded["Pkt Len Mean"].iloc[2])
    assert info["inf_cells"] == 2
    assert info["nan_cells"] == 1


def test_inf_and_nan_survive_the_merge_and_the_disk_round_trip(tmp_path, monkeypatch):
    six_captures(tmp_path)
    df = fake_flow_frame(3, label="Normal")
    df.loc[0, "Flow Byts/s"] = np.inf
    df.loc[1, "Flow Pkts/s"] = -np.inf
    df.loc[2, "Active Mean"] = np.nan
    df.to_csv(tmp_path / "N-0_flows.csv", index=False)

    merged_dir = tmp_path / "out" / "merged"
    monkeypatch.setattr(config, "MERGED_DIR", merged_dir)
    monkeypatch.setattr(config, "MERGED_CSV", merged_dir / "MSCAD_merged.csv")
    monkeypatch.setattr(config, "MERGED_PARQUET", merged_dir / "MSCAD_merged.parquet")
    monkeypatch.setattr(config, "STEP2_OUTPUT_DIR", tmp_path / "out" / "step2")

    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    merged, infos = dl.merge_flow_csvs(files=files, verbose=False)
    checks, ok = dl.check_merge(merged, infos, verbose=False)
    assert ok is True, [c for c in checks if c["result"] == "FAIL"]

    dl.save_merged(merged, checks=checks, infos=infos, overwrite=True, verbose=False)
    back = dl.load_merged(verbose=False)

    feats = [c for c in config.KAGGLE_FEATURES if c in back.columns]
    arr = back[feats].to_numpy(dtype="float64")
    assert int(np.isinf(arr).sum()) == 2, "inf must survive the CSV round trip"
    assert int(np.isnan(arr).sum()) == 1, "NaN must survive the CSV round trip"
    assert len(back) == len(merged)


def test_duplicates_are_not_dropped(tmp_path):
    """Three identical rows must still be three rows after merging."""
    folder = tmp_path
    six_captures(folder)
    dup = fake_flow_frame(3, label="Normal")
    dup.loc[1] = dup.loc[0]
    dup.loc[2] = dup.loc[0]
    dup.to_csv(folder / "N-0_flows.csv", index=False)

    files, _ = dl.find_flow_csvs(folder, verbose=False)
    merged, _ = dl.merge_flow_csvs(files=files, verbose=False)
    assert (merged["Source_Capture"] == "N-0").sum() == 3


# ===========================================================================
# 7. stable timestamp sort
# ===========================================================================
def test_merged_rows_are_sorted_by_timestamp(tmp_path):
    six_captures(tmp_path)
    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    merged, _ = dl.merge_flow_csvs(files=files, verbose=False)
    ts = merged[dl.TIMESTAMP_DT_COLUMN].dropna()
    assert ts.is_monotonic_increasing


def test_sort_is_stable_for_identical_timestamps(tmp_path):
    """Equal timestamps must keep CAPTURE_ORDER, then Orig_Row -- reproducibly."""
    # every row in both files carries the SAME timestamp
    same = "2020-09-13 12:00:00.000000"
    for capture, label, rows in [("N-0", "Normal", 3), ("Scan-1", "Port_Scan", 3)]:
        df = fake_flow_frame(rows, label=label)
        df["Timestamp"] = same
        df.to_csv(tmp_path / ("%s_flows.csv" % capture), index=False)
    for capture, label in [("App-01", "DoS slowloris"), ("App-02", "ICMP_Flood"),
                           ("W-B-01", "Web_Crwling"), ("W-B-02", "Brute_Force")]:
        write_flow_csv(tmp_path, capture, n_rows=1, label=label, start_second=59)

    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    merged, _ = dl.merge_flow_csvs(files=files, verbose=False)

    tied = merged[merged[dl.TIMESTAMP_DT_COLUMN] == pd.Timestamp(same)]
    assert list(tied["Source_Capture"]) == ["N-0"] * 3 + ["Scan-1"] * 3
    assert list(tied["Orig_Row"]) == [0, 1, 2, 0, 1, 2]

    # and it is deterministic: merging again gives the identical order
    merged2, _ = dl.merge_flow_csvs(files=files, verbose=False)
    pd.testing.assert_frame_equal(merged, merged2)


def test_unparseable_timestamps_become_nat_and_sort_last(tmp_path):
    six_captures(tmp_path)
    df = fake_flow_frame(3, label="Normal")
    df.loc[1, "Timestamp"] = "not a timestamp"
    df.to_csv(tmp_path / "N-0_flows.csv", index=False)

    loaded, info = dl.load_one_csv(tmp_path / "N-0_flows.csv", verbose=False)
    assert info["timestamp_unparsed"] == 1
    assert pd.isna(loaded[dl.TIMESTAMP_DT_COLUMN].iloc[1])
    # the ORIGINAL text column is kept untouched
    assert loaded["Timestamp"].iloc[1] == "not a timestamp"

    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    merged, _ = dl.merge_flow_csvs(files=files, verbose=False)
    assert pd.isna(merged[dl.TIMESTAMP_DT_COLUMN].iloc[-1])     # NaT last


# ===========================================================================
# 8. timestamp format detection
# ===========================================================================
def test_iso_timestamps_are_detected_as_unambiguous():
    info = dl.detect_timestamp_format(["2020-09-13 12:26:40.000000"])
    assert info["ambiguous"] is False
    assert info["format"] == dl.ISO_FORMAT
    assert "year comes first" in info["reason"]


def test_day_over_12_resolves_the_order():
    info = dl.detect_timestamp_format(["26/04/2017 11:11:17"])
    assert info["ambiguous"] is False and info["dayfirst"] is True


def test_month_over_12_resolves_the_order():
    info = dl.detect_timestamp_format(["04/26/2017 11:11:17"])
    assert info["ambiguous"] is False and info["dayfirst"] is False


def test_truly_ambiguous_dates_are_flagged_with_examples():
    samples = ["01/02/2017 11:11:17", "03/04/2017 11:11:18", "05/06/2017 11:11:19"]
    info = dl.detect_timestamp_format(samples)
    assert info["ambiguous"] is True
    assert info["format"] is None
    assert len(info["examples"]) == 3


def test_ambiguous_timestamps_stop_the_load_until_a_human_decides(tmp_path):
    df = fake_flow_frame(3)
    df["Timestamp"] = ["01/02/2017 11:11:17", "03/04/2017 11:11:18",
                       "05/06/2017 11:11:19"]
    df.to_csv(tmp_path / "N-0_flows.csv", index=False)

    with pytest.raises(dl.AmbiguousTimestampError) as err:
        dl.load_one_csv(tmp_path / "N-0_flows.csv", verbose=False)
    assert "dayfirst" in str(err.value)

    # an explicit answer lets it through
    loaded, _ = dl.load_one_csv(tmp_path / "N-0_flows.csv", dayfirst=True, verbose=False)
    assert loaded[dl.TIMESTAMP_DT_COLUMN].iloc[0] == pd.Timestamp("2017-02-01 11:11:17")


# ===========================================================================
# 9. dtypes and the kept original Timestamp
# ===========================================================================
def test_dtypes_are_as_specified(tmp_path):
    write_flow_csv(tmp_path, "N-0", n_rows=3)
    df, _ = dl.load_one_csv(tmp_path / "N-0_flows.csv", verbose=False)
    assert str(df["Src IP"].dtype) == "string"
    assert str(df[config.LABEL_COLUMN].dtype) == "string"
    for c in ("Src Port", "Dst Port", "Protocol"):
        assert str(df[c].dtype) == "int64", c
    for c in ("Flow Duration", "Flow Byts/s", "Idle Min"):
        assert str(df[c].dtype) == "float64", c


def test_original_timestamp_text_is_kept_beside_the_parsed_one(tmp_path):
    write_flow_csv(tmp_path, "N-0", n_rows=2)
    df, _ = dl.load_one_csv(tmp_path / "N-0_flows.csv", verbose=False)
    assert "Timestamp" in df.columns and dl.TIMESTAMP_DT_COLUMN in df.columns
    assert df["Timestamp"].iloc[0] == "2020-09-13 12:00:00.000000"
    assert df[dl.TIMESTAMP_DT_COLUMN].iloc[0] == pd.Timestamp("2020-09-13 12:00:00")


# ===========================================================================
# 10. saving
# ===========================================================================
def test_save_skips_an_existing_file_unless_overwrite(tmp_path, monkeypatch):
    merged_dir = tmp_path / "merged"
    monkeypatch.setattr(config, "MERGED_DIR", merged_dir)
    monkeypatch.setattr(config, "MERGED_CSV", merged_dir / "MSCAD_merged.csv")
    monkeypatch.setattr(config, "MERGED_PARQUET", merged_dir / "MSCAD_merged.parquet")
    monkeypatch.setattr(config, "STEP2_OUTPUT_DIR", tmp_path / "step2")

    six_captures(tmp_path / "csvs")
    files, _ = dl.find_flow_csvs(tmp_path / "csvs", verbose=False)
    merged, infos = dl.merge_flow_csvs(files=files, verbose=False)

    first = dl.save_merged(merged, infos=infos, overwrite=True, verbose=False)
    assert first["csv"].endswith("MSCAD_merged.csv")

    second = dl.save_merged(merged, infos=infos, overwrite=False, verbose=False)
    assert "skipped" in second["csv"]


def test_merge_report_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MERGED_DIR", tmp_path / "merged")
    monkeypatch.setattr(config, "MERGED_CSV", tmp_path / "merged" / "m.csv")
    monkeypatch.setattr(config, "MERGED_PARQUET", tmp_path / "merged" / "m.parquet")
    monkeypatch.setattr(config, "STEP2_OUTPUT_DIR", tmp_path / "step2")

    six_captures(tmp_path / "csvs")
    files, _ = dl.find_flow_csvs(tmp_path / "csvs", verbose=False)
    merged, infos = dl.merge_flow_csvs(files=files, verbose=False)
    checks, _ = dl.check_merge(merged, infos, verbose=False)
    out = dl.save_merged(merged, checks=checks, infos=infos, overwrite=True, verbose=False)

    report = pd.read_csv(out["report"])
    assert set(report["kind"]) == {"file", "check"}
    assert (report["kind"] == "file").sum() == 6
    assert ((report["kind"] == "check") & (report["result"] == "PASS")).sum() == len(checks)


def test_load_merged_raises_a_clear_error_when_nothing_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MERGED_CSV", tmp_path / "nope.csv")
    monkeypatch.setattr(config, "MERGED_PARQUET", tmp_path / "nope.parquet")
    with pytest.raises(dl.DataLoadingError) as err:
        dl.load_merged(verbose=False)
    assert "save_merged" in str(err.value)
