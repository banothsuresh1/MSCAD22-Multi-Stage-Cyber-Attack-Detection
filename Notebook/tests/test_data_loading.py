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


def redirect_outputs(monkeypatch, root: Path) -> Path:
    """Point every Step 2/3 output at a temp folder -- never C:\\MSCAD2022."""
    step2, step3 = root / "step2", root / "step3"
    monkeypatch.setattr(config, "OUTPUTS_DIR", root)
    monkeypatch.setattr(config, "STEP2_OUTPUT_DIR", step2)
    monkeypatch.setattr(config, "STEP3_OUTPUT_DIR", step3)
    monkeypatch.setattr(config, "MERGED_DIR", step3)
    monkeypatch.setattr(config, "MERGED_CSV", step3 / "MSCAD_merged.csv")
    monkeypatch.setattr(config, "MERGED_PARQUET", step3 / "MSCAD_merged.parquet")
    return step3


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

    redirect_outputs(monkeypatch, tmp_path / "out")

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
    redirect_outputs(monkeypatch, tmp_path / "out")

    six_captures(tmp_path / "csvs")
    files, _ = dl.find_flow_csvs(tmp_path / "csvs", verbose=False)
    merged, infos = dl.merge_flow_csvs(files=files, verbose=False)

    first = dl.save_merged(merged, infos=infos, overwrite=True, verbose=False)
    assert first["status"] == "written"
    assert first["ok"] is True

    second = dl.save_merged(merged, infos=infos, overwrite=False, verbose=False)
    assert second["status"] == "kept_existing"
    assert second["ok"] is True          # the existing file is still verified


def test_merge_report_is_written(tmp_path, monkeypatch):
    redirect_outputs(monkeypatch, tmp_path / "out")

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


# ===========================================================================
# 11. presentation helpers (display layer, no data changes)
# ===========================================================================
def test_kv_table_builds_a_two_column_frame():
    t = dl.kv_table([("rows", 10), ("columns", 76)])
    assert list(t.columns) == ["metric", "value"]
    assert t["value"].tolist() == [10, 76]


def test_show_helpers_fall_back_to_plain_text_outside_jupyter(capsys):
    """pytest is not IPython, so these must print rather than call display()."""
    assert dl.in_notebook() is False
    dl.show_title("1. Shape and memory")
    dl.show_table(pd.DataFrame({"metric": ["rows"], "value": [983]}), note="a note")
    printed = capsys.readouterr().out
    assert "1. Shape and memory" in printed
    assert "983" in printed and "a note" in printed


def test_show_table_does_not_change_global_pandas_options():
    before = (pd.get_option("display.max_columns"), pd.get_option("display.width"))
    dl.show_table(fake_flow_frame(2), all_columns=True)
    assert (pd.get_option("display.max_columns"), pd.get_option("display.width")) == before


def test_summarize_dataset_returns_tables_and_changes_nothing(tmp_path, capsys):
    six_captures(tmp_path)
    files, _ = dl.find_flow_csvs(tmp_path, verbose=False)
    merged, _ = dl.merge_flow_csvs(files=files, verbose=False)
    merged.loc[merged.index[0], "Flow Byts/s"] = np.inf
    merged.loc[merged.index[1], "Pkt Len Mean"] = np.nan
    before = merged.copy(deep=True)

    out = dl.summarize_dataset(merged, make_plot=False, compare_kaggle=False)
    capsys.readouterr()

    for key in ("shape", "info_table", "label_counts", "crosstab", "time_ranges",
                "protocol_counts", "nan_inf_table", "describe", "duplicate_rows"):
        assert key in out, key
    assert isinstance(out["info_table"], pd.DataFrame)
    assert list(out["info_table"].columns) == ["column", "dtype", "non_null", "null"]
    assert len(out["info_table"]) == len(merged.columns)
    # the dataset itself is untouched
    pd.testing.assert_frame_equal(merged, before)
    assert np.isinf(merged["Flow Byts/s"].iloc[0])


# ===========================================================================
# 12. describe() without the inf warnings
# ===========================================================================
def test_describe_table_hides_inf_from_the_stats_but_counts_it():
    df = pd.DataFrame({"Flow Byts/s": [np.inf, 10.0, 20.0, 30.0],
                       "Tot Fwd Pkts": [1.0, 2.0, 3.0, 4.0]})
    out = dl.describe_table(df, columns=["Flow Byts/s", "Tot Fwd Pkts"])

    assert out.loc["Flow Byts/s", "inf_count"] == 1
    assert out.loc["Tot Fwd Pkts", "inf_count"] == 0
    assert out.loc["Flow Byts/s", "count"] == 3          # the inf row is excluded
    assert out.loc["Flow Byts/s", "mean"] == 20.0        # finite, not inf
    assert np.isfinite(out.loc["Flow Byts/s", "std"])
    assert list(out.columns)[0] == "inf_count"


def test_describe_table_does_not_modify_the_input():
    df = pd.DataFrame({"Flow Byts/s": [np.inf, 1.0]})
    dl.describe_table(df, columns=["Flow Byts/s"])
    assert np.isinf(df["Flow Byts/s"].iloc[0]), "the original must keep its inf"


def test_describe_emits_no_runtime_warnings_on_inf():
    import warnings
    df = pd.DataFrame({"Flow Byts/s": [np.inf, np.inf, 1.0, 2.0, 3.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = dl.describe_table(df, columns=["Flow Byts/s"])
    assert out.loc["Flow Byts/s", "inf_count"] == 2


# ===========================================================================
# 13. the Kaggle reference comparison
# ===========================================================================
def write_kaggle(folder: Path, name: str, labels: dict) -> Path:
    """A minimal Kaggle-like file: one feature column plus Label."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, n in labels.items():
        rows.extend([{"Flow Duration": 1.0, "Label": label}] * n)
    path = folder / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_config_has_the_kaggle_csv_dir():
    assert str(config.KAGGLE_CSV_DIR).endswith("CSV_Original_kaggle")
    assert config.KAGGLE_CSV_DIR in config.READ_ONLY_DIRS
    assert config.KAGGLE_CSV_DIR not in config.DIRS_TO_CREATE


def test_normalise_label_matches_spacing_case_and_underscores():
    assert dl.normalise_label("Web_Crwling") == dl.normalise_label("Web Crwling")
    assert dl.normalise_label(" web crwling ") == dl.normalise_label("Web_Crwling")
    assert dl.normalise_label("DoS slowloris") == dl.normalise_label("dos_slowloris")
    assert dl.normalise_label("Port_Scan") != dl.normalise_label("PortScan")


def test_find_kaggle_csv_prefers_a_csv_named_mscad(tmp_path):
    write_kaggle(tmp_path, "other.csv", {"Normal": 1})
    write_kaggle(tmp_path, "MSCAD.csv", {"Normal": 1})
    (tmp_path / "MSCAD.xlsx").write_bytes(b"not really a spreadsheet")
    chosen, candidates = config.find_kaggle_csv(tmp_path)
    assert chosen.name == "MSCAD.csv"
    assert len(candidates) == 3


def test_find_kaggle_csv_falls_back_to_any_csv(tmp_path):
    write_kaggle(tmp_path, "labels_export.csv", {"Normal": 1})
    chosen, _ = config.find_kaggle_csv(tmp_path)
    assert chosen.name == "labels_export.csv"


def test_comparison_matches_labels_that_differ_only_in_spelling(tmp_path, capsys):
    df = pd.DataFrame({config.LABEL_COLUMN: ["Web_Crwling"] * 23 + ["Normal"] * 10})
    write_kaggle(tmp_path, "MSCAD.csv", {"Web Crwling": 20, "normal": 10})

    result = dl.compare_labels_with_kaggle(df, folder=tmp_path, verbose=True)
    capsys.readouterr()

    table = result["table"].set_index("Label")
    assert table.loc["Web_Crwling", "ours"] == 23
    assert table.loc["Web_Crwling", "kaggle"] == 20
    assert table.loc["Web_Crwling", "difference"] == 3
    assert table.loc["Web_Crwling", "% difference"] == pytest.approx(15.0)
    assert table.loc["TOTAL (matched)", "ours"] == 33
    assert table.loc["TOTAL (matched)", "kaggle"] == 30
    # nothing was renamed: both spellings are reported side by side
    spelling = result["spelling"]
    assert {"Web_Crwling", "Normal"} == set(spelling["our spelling"])
    assert {"Web Crwling", "normal"} == set(spelling["Kaggle spelling"])
    assert len(result["one_sided"]) == 0


def test_comparison_lists_labels_present_on_only_one_side(tmp_path, capsys):
    df = pd.DataFrame({config.LABEL_COLUMN: ["Normal"] * 5 + ["Benign"] * 3})
    write_kaggle(tmp_path, "MSCAD.csv", {"Normal": 5, "Infiltration": 7})

    result = dl.compare_labels_with_kaggle(df, folder=tmp_path, verbose=True)
    capsys.readouterr()

    one_sided = result["one_sided"].set_index("Label")
    assert one_sided.loc["Benign", "side"] == "ours only"
    assert one_sided.loc["Benign", "flows"] == 3
    assert one_sided.loc["Infiltration", "side"] == "Kaggle only"
    assert one_sided.loc["Infiltration", "flows"] == 7
    assert set(result["table"]["Label"]) == {"Normal", "TOTAL (matched)"}


def test_comparison_handles_a_missing_folder_without_raising(tmp_path, capsys):
    df = pd.DataFrame({config.LABEL_COLUMN: ["Normal"] * 3})
    result = dl.compare_labels_with_kaggle(df, folder=tmp_path / "nope", verbose=True)
    printed = capsys.readouterr().out
    assert result is None
    assert "skipped" in printed and "folder does not exist" in printed


def test_comparison_handles_an_empty_folder_without_raising(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    df = pd.DataFrame({config.LABEL_COLUMN: ["Normal"] * 3})
    result = dl.compare_labels_with_kaggle(df, folder=tmp_path / "empty", verbose=True)
    printed = capsys.readouterr().out
    assert result is None
    assert "no .csv or .xlsx" in printed


def test_comparison_handles_a_file_without_a_label_column(tmp_path, capsys):
    pd.DataFrame({"Flow Duration": [1.0, 2.0]}).to_csv(tmp_path / "MSCAD.csv", index=False)
    df = pd.DataFrame({config.LABEL_COLUMN: ["Normal"] * 3})
    result = dl.compare_labels_with_kaggle(df, folder=tmp_path, verbose=True)
    printed = capsys.readouterr().out
    assert result is None
    assert "no Label column" in printed


def test_comparison_reports_which_file_it_picked(tmp_path, capsys):
    write_kaggle(tmp_path, "MSCAD.csv", {"Normal": 5})
    write_kaggle(tmp_path, "zz_other.csv", {"Normal": 99})
    df = pd.DataFrame({config.LABEL_COLUMN: ["Normal"] * 5})

    result = dl.compare_labels_with_kaggle(df, folder=tmp_path, verbose=True)
    printed = capsys.readouterr().out

    assert Path(result["info"]["path"]).name == "MSCAD.csv"
    assert "MSCAD.csv" in printed and "zz_other.csv" in printed
    assert result["table"].set_index("Label").loc["Normal", "kaggle"] == 5


# ===========================================================================
# 14. STEP 2 -- data acquisition
# ===========================================================================
def test_acquire_loads_every_capture_from_a_folder(tmp_path, monkeypatch):
    redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    for capture, label, rows in [("N-0", "Normal", 4), ("Scan-1", "Port_Scan", 5),
                                 ("App-01", "DoS slowloris", 3)]:
        write_flow_csv(folder, capture, n_rows=rows, label=label)

    frames, report, failures = dl.acquire_captures(folder, verbose=False)

    assert set(frames) == {"N-0", "Scan-1", "App-01"}
    assert failures == []
    assert len(report) == 3
    assert set(report["status"]) == {"loaded"}
    assert report.set_index("capture").loc["Scan-1", "rows"] == 5
    assert list(report.columns) == dl.ACQUISITION_FIELDS


def test_acquire_tags_source_capture_and_orig_row(tmp_path, monkeypatch):
    redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    write_flow_csv(folder, "N-0", n_rows=4, label="Normal")
    write_flow_csv(folder, "W-B-02", n_rows=3, label="Brute_Force")

    frames, _report, _f = dl.acquire_captures(folder, verbose=False)

    assert set(frames["N-0"]["Source_Capture"]) == {"N-0"}
    assert frames["N-0"]["Orig_Row"].tolist() == [0, 1, 2, 3]
    assert frames["W-B-02"]["Orig_Row"].tolist() == [0, 1, 2]


def test_acquire_reports_rows_columns_size_labels_and_timestamps(tmp_path, monkeypatch):
    redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    write_flow_csv(folder, "Scan-1", n_rows=5, label="Port_Scan", start_second=10)

    _frames, report, _f = dl.acquire_captures(folder, verbose=False)
    row = report.iloc[0]

    assert row["file_name"] == "Scan-1_flows.csv"
    assert row["rows"] == 5
    # 73 flow columns + Timestamp_dt (added on load) + Source_Capture + Orig_Row
    assert row["columns"] == 76
    assert row["size_mb"] >= 0
    assert '"Port_Scan": 5' in row["label_counts"]
    assert pd.notna(row["first_timestamp"]) and pd.notna(row["last_timestamp"])
    assert row["first_timestamp"] <= row["last_timestamp"]


def test_acquire_stops_with_a_clear_message_naming_a_missing_folder(tmp_path):
    missing = tmp_path / "not_there"
    with pytest.raises(dl.MissingCapturesError) as err:
        dl.acquire_captures(missing, verbose=False)
    assert str(missing) in str(err.value)
    assert "does not exist" in str(err.value)


def test_acquire_stops_with_a_clear_message_on_an_empty_folder(tmp_path):
    empty = tmp_path / "CSVs"
    empty.mkdir()
    (empty / "readme.txt").write_text("no csvs here")
    with pytest.raises(dl.MissingCapturesError) as err:
        dl.acquire_captures(empty, verbose=False)
    assert str(empty) in str(err.value)
    assert "_flows.csv" in str(err.value)


def test_acquire_reports_one_bad_file_and_keeps_the_rest(tmp_path, monkeypatch, capsys):
    redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    write_flow_csv(folder, "N-0", n_rows=4, label="Normal")
    write_flow_csv(folder, "Scan-1", n_rows=5, label="Port_Scan")
    # a file with the right name but a broken header
    fake_flow_frame(3).drop(columns=["Idle Max"]).to_csv(
        folder / "App-01_flows.csv", index=False)

    frames, report, failures = dl.acquire_captures(folder, verbose=True)
    capsys.readouterr()

    assert set(frames) == {"N-0", "Scan-1"}, "the good files must still load"
    assert len(failures) == 1 and failures[0]["file"] == "App-01_flows.csv"
    assert "Idle Max" in failures[0]["error"]
    assert report.set_index("capture").loc["App-01", "status"] == "FAILED"


def test_acquire_writes_its_report_to_step2_output_dir(tmp_path, monkeypatch):
    step3 = redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    write_flow_csv(folder, "N-0", n_rows=2, label="Normal")

    dl.acquire_captures(folder, verbose=False)

    written = config.STEP2_OUTPUT_DIR / "acquisition_report.csv"
    assert written.exists()
    assert pd.read_csv(written).iloc[0]["capture"] == "N-0"
    assert step3.parent == config.OUTPUTS_DIR


# ===========================================================================
# 15. STEP 3 -- column consistency, merging, saving, verifying
# ===========================================================================
def test_column_consistency_passes_when_every_capture_matches(tmp_path, monkeypatch):
    redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    six_captures(folder)
    frames, _r, _f = dl.acquire_captures(folder, verbose=False)

    table, ok = dl.check_column_consistency(frames, verbose=False)

    assert ok is True
    assert set(table["status"]) == {"ok"}
    assert len(table) == 6


def test_column_mismatch_is_reported_in_a_table_not_silently_filled(tmp_path):
    good = fake_flow_frame(3)
    good["Source_Capture"], good["Orig_Row"] = "N-0", range(3)
    odd = fake_flow_frame(3).drop(columns=["Idle Max"])
    odd["Surprise"] = 1
    odd["Source_Capture"], odd["Orig_Row"] = "Scan-1", range(3)

    table, ok = dl.check_column_consistency({"N-0": good, "Scan-1": odd}, verbose=False)

    assert ok is False
    row = table.set_index("capture").loc["Scan-1"]
    assert row["status"] == "MISMATCH"
    assert "Idle Max" in row["missing_vs_reference"]
    assert "Surprise" in row["extra_vs_reference"]
    assert table.set_index("capture").loc["N-0", "status"] == "ok"


def test_merged_file_is_actually_written_and_matches_on_disk(tmp_path, monkeypatch):
    step3 = redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    six_captures(folder)
    frames, _r, _f = dl.acquire_captures(folder, verbose=False)
    merged, _ = dl.merge_flow_csvs(frames=frames, verbose=False)

    result = dl.save_merged(merged, overwrite=True, verbose=False)

    target = step3 / "MSCAD_merged.csv"
    assert target.exists(), "the merged CSV must really be on disk"
    assert result["status"] == "written"
    assert result["ok"] is True
    v = result["verification"]
    assert v["exists"] is True
    assert Path(v["path"]).is_absolute()
    assert v["size_bytes"] > 0
    assert v["rows_on_disk"] == v["rows_in_memory"] == len(merged)
    assert v["cols_on_disk"] == v["cols_in_memory"] == merged.shape[1]
    # and re-reading it independently gives the same shape
    assert pd.read_csv(target, low_memory=False).shape == merged.shape


def test_verify_reports_fail_when_the_file_is_absent(tmp_path, monkeypatch):
    redirect_outputs(monkeypatch, tmp_path / "out")
    merged = fake_flow_frame(3)
    result = dl.verify_merged_file(merged, tmp_path / "nothing.csv", verbose=False)
    assert result["exists"] is False
    assert result["ok"] is False
    assert "does not exist" in result["reason"]


def test_overwrite_false_keeps_the_existing_file(tmp_path, monkeypatch):
    step3 = redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    six_captures(folder)
    frames, _r, _f = dl.acquire_captures(folder, verbose=False)
    merged, _ = dl.merge_flow_csvs(frames=frames, verbose=False)

    dl.save_merged(merged, overwrite=True, verbose=False)
    target = step3 / "MSCAD_merged.csv"
    stamp = target.stat().st_mtime_ns
    original = target.read_bytes()

    smaller = merged.iloc[:2]
    result = dl.save_merged(smaller, overwrite=False, verbose=False)

    assert result["status"] == "kept_existing"
    assert target.read_bytes() == original, "the existing file must not be rewritten"
    assert target.stat().st_mtime_ns == stamp
    # the report still describes the real file, not the smaller frame
    assert result["verification"]["rows_on_disk"] == len(merged)


def test_overwrite_true_replaces_the_file(tmp_path, monkeypatch):
    step3 = redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    six_captures(folder)
    frames, _r, _f = dl.acquire_captures(folder, verbose=False)
    merged, _ = dl.merge_flow_csvs(frames=frames, verbose=False)

    dl.save_merged(merged, overwrite=True, verbose=False)
    half = merged.iloc[: len(merged) // 2]
    result = dl.save_merged(half, overwrite=True, verbose=False)

    assert result["status"] == "written"
    assert result["verification"]["rows_on_disk"] == len(half)


def test_save_creates_the_output_folder_if_it_is_missing(tmp_path, monkeypatch):
    step3 = redirect_outputs(monkeypatch, tmp_path / "deep" / "nested" / "out")
    assert not step3.exists()
    folder = tmp_path / "CSVs"
    six_captures(folder)
    frames, _r, _f = dl.acquire_captures(folder, verbose=False)
    merged, _ = dl.merge_flow_csvs(frames=frames, verbose=False)

    result = dl.save_merged(merged, overwrite=True, verbose=False)

    assert step3.is_dir()
    assert (step3 / "MSCAD_merged.csv").exists()
    assert result["ok"] is True


def test_step3_reports_land_in_step3_output_dir(tmp_path, monkeypatch):
    step3 = redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    six_captures(folder)
    frames, _r, _f = dl.acquire_captures(folder, verbose=False)
    merged, _ = dl.merge_flow_csvs(frames=frames, verbose=False)

    written = dl.save_step3_reports(merged, verbose=False)

    for key in ("label_counts", "crosstab", "summary"):
        assert Path(written[key]).exists(), key
        assert Path(written[key]).parent == step3
    assert "Nothing was cleaned" in Path(written["summary"]).read_text()
    counts = pd.read_csv(written["label_counts"])
    assert int(counts["flows"].sum()) == len(merged)


def test_step2_and_step3_paths_are_not_under_project_root():
    """The bug that lost the merged file: outputs hanging off the code folder."""
    for name in ("CSV_INPUT_DIR", "OUTPUTS_DIR", "STEP2_OUTPUT_DIR",
                 "STEP3_OUTPUT_DIR", "MERGED_CSV", "MERGED_PARQUET"):
        value = str(getattr(config, name))
        assert str(config.PROJECT_ROOT) not in value, (
            "%s is under PROJECT_ROOT, so a second copy of the code would move it" % name)
        assert str(config.DATA_ROOT) in value, "%s should hang off DATA_ROOT" % name


def test_merged_csv_sits_in_step3_output_dir():
    assert config.MERGED_CSV.parent == config.STEP3_OUTPUT_DIR
    assert config.MERGED_CSV.name == "MSCAD_merged.csv"
    assert config.STEP3_OUTPUT_DIR == config.OUTPUTS_DIR / "step3"


def test_keeping_an_existing_file_is_not_reported_as_a_failure(tmp_path, monkeypatch, capsys):
    """OVERWRITE_MERGED=False is the user's choice, not a verification failure."""
    step3 = redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    six_captures(folder)
    frames, _r, _f = dl.acquire_captures(folder, verbose=False)
    merged, _ = dl.merge_flow_csvs(frames=frames, verbose=False)
    dl.save_merged(merged, overwrite=True, verbose=False)

    # a DIFFERENT frame, with the file kept
    result = dl.save_merged(merged.iloc[:3], overwrite=False, verbose=True)
    printed = capsys.readouterr().out

    assert result["status"] == "kept_existing"
    assert result["ok"] is True, "a kept file that is readable is not a failure"
    assert result["stale"] is True, "but the difference must be flagged"
    assert "[FAIL]" not in printed
    assert "DIFFERS from what was just built" in printed
    assert "OVERWRITE_MERGED" in printed


def test_keeping_an_identical_file_reports_a_clean_match(tmp_path, monkeypatch, capsys):
    redirect_outputs(monkeypatch, tmp_path / "out")
    folder = tmp_path / "CSVs"
    six_captures(folder)
    frames, _r, _f = dl.acquire_captures(folder, verbose=False)
    merged, _ = dl.merge_flow_csvs(frames=frames, verbose=False)
    dl.save_merged(merged, overwrite=True, verbose=False)

    result = dl.save_merged(merged, overwrite=False, verbose=True)
    printed = capsys.readouterr().out

    assert result["status"] == "kept_existing"
    assert result["ok"] is True and result["stale"] is False
    assert "matches what was just built" in printed
