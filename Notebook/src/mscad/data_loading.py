"""
src/mscad/data_loading.py -- STEP 2: load the per-capture flow CSVs and merge them.

What this module DOES
---------------------
    find_flow_csvs()     locate the <stem>_flows.csv files (never raw\\*_raw.csv)
    load_one_csv()       read one file with the right dtypes, parse the timestamp
    validate_columns()   header must be EXACTLY config.FLOW_COLUMNS
    merge_flow_csvs()    add Source_Capture / Orig_Row, concat, stable time sort
    check_merge()        PASS/FAIL checks with numbers, never a bare "looks fine"
    save_merged()        write the merged CSV (+ Parquet) and merge_report.csv
    load_merged()        read it back FROM DISK
    summarize_dataset()  the full overview of the reloaded dataset

What this module deliberately does NOT do
-----------------------------------------
No cleaning. inf, -inf, NaN and duplicate rows are all carried through
untouched, and so is the class imbalance. No encoding, scaling, splitting,
feature selection, windowing or training -- those are Step 3 onward.

Beginner note on paths
----------------------
Every path comes from config.py, as a pathlib.Path built from raw strings, so
folder names containing spaces work with no quoting anywhere.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# config.py lives two folders up (src/mscad/ -> src/ -> PROJECT_ROOT).
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402


# ===========================================================================
# errors
# ===========================================================================
class DataLoadingError(Exception):
    """Base class for the errors this module raises on purpose."""


class ColumnMismatchError(DataLoadingError):
    """A file's header is not exactly config.FLOW_COLUMNS.

    Fatal by design: silently reordering or renaming columns here would corrupt
    every later stage, and guessing which column is which is worse than stopping.
    """


class MissingCapturesError(DataLoadingError):
    """The files found do not match config.CAPTURE_ORDER."""


class AmbiguousTimestampError(DataLoadingError):
    """The Timestamp text could be day/month or month/day -- a human must decide."""


# ===========================================================================
# chart colours (single-series bar chart; see summarize_dataset)
# ===========================================================================
_BAR = "#2a78d6"        # one hue: this is ONE measure across categories
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SOFT = "#52514e"
_GRID = "#dcdcd8"


# ===========================================================================
# 1. find the files
# ===========================================================================
@dataclass
class FlowCsv:
    """One Step 1 output file."""

    path: Path
    rows: int
    size_bytes: int

    @property
    def capture(self) -> str:
        """The capture name, taken from the file NAME: 'N-0_flows.csv' -> 'N-0'."""
        name = self.path.name
        suffix = config.FLOW_CSV_PATTERN.replace("*", "")      # "_flows.csv"
        return name[: -len(suffix)] if name.endswith(suffix) else self.path.stem

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 2)


def count_csv_rows(path: Path) -> int:
    """Data rows in a CSV (header excluded), counted without loading it."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def find_flow_csvs(folder: Path | str | None = None,
                   pattern: str | None = None,
                   require_all: bool = True,
                   verbose: bool = True) -> tuple[list[FlowCsv], Path]:
    """Find the flow CSVs and return (files in CAPTURE_ORDER, the folder used).

    Only files matching config.FLOW_CSV_PATTERN are picked up, so
    CSVs\\raw\\<stem>_raw.csv and anything else in the folder are ignored --
    glob() does not recurse, so the raw\\ subfolder is never even looked at.

    With no folder given, config.CSV_CANDIDATE_DIRS are tried in order and the
    one that actually contains matching files wins.  The folder used is printed
    and returned, so the choice is never silent.
    """
    pattern = pattern or config.FLOW_CSV_PATTERN
    candidates = [Path(folder)] if folder is not None else list(config.CSV_CANDIDATE_DIRS)

    chosen: Path | None = None
    checked: list[str] = []
    for cand in candidates:
        cand = Path(cand)
        if not cand.is_dir():
            checked.append("%s  (no such folder)" % cand)
            continue
        if sorted(cand.glob(pattern)):
            chosen = cand
            break
        checked.append("%s  (exists, but no %s in it)" % (cand, pattern))

    if chosen is None:
        if verbose:
            print("  looked in:")
            for line in checked:
                print("     ", line)
        raise MissingCapturesError(
            "no files matching %r found in any of: %s"
            % (pattern, [str(c) for c in candidates]))

    files = [FlowCsv(path=p, rows=count_csv_rows(p), size_bytes=p.stat().st_size)
             for p in sorted(chosen.glob(pattern))]

    # Put them in the documented scenario order; anything unexpected goes last.
    order = {name: i for i, name in enumerate(config.CAPTURE_ORDER)}
    files.sort(key=lambda f: (order.get(f.capture, len(order)), f.capture))

    if verbose:
        if folder is None and chosen != Path(config.CSV_OUTPUT_DIR):
            print("  NOTE: config.CSV_OUTPUT_DIR held no %s, so a fallback folder was used."
                  % pattern)
            for line in checked:
                print("        tried", line)
        print("  folder : %s" % chosen)
        print("  pattern: %s   (raw\\*_raw.csv is NOT matched)" % pattern)
        print()
        print("  %-10s %-26s %10s %12s" % ("CAPTURE", "FILE", "SIZE (MB)", "ROWS"))
        print("  " + "-" * 62)
        for f in files:
            print("  %-10s %-26s %10.2f %12s" % (f.capture, f.path.name, f.size_mb,
                                                 f"{f.rows:,}"))
        print("  " + "-" * 62)
        print("  %-10s %-26s %10.2f %12s" % ("TOTAL", "%d files" % len(files),
                                             sum(f.size_mb for f in files),
                                             f"{sum(f.rows for f in files):,}"))

    found = [f.capture for f in files]
    missing = [c for c in config.CAPTURE_ORDER if c not in found]
    extra = [c for c in found if c not in config.CAPTURE_ORDER]
    if verbose:
        print()
        if missing:
            print("  [WARN]  expected captures MISSING: %s" % missing)
        if extra:
            print("  [WARN]  unexpected captures found : %s" % extra)
        if not missing and not extra:
            print("  [OK]    all %d expected captures present, in CAPTURE_ORDER"
                  % len(config.CAPTURE_ORDER))

    if require_all and (missing or extra):
        raise MissingCapturesError(
            "expected exactly %s but found %s (missing %s, unexpected %s). "
            "Stopping rather than merging an incomplete dataset."
            % (config.CAPTURE_ORDER, found, missing, extra))
    return files, chosen


# ===========================================================================
# 2. load one file
# ===========================================================================
STR_COLUMNS = ["Src IP", "Dst IP", config.LABEL_COLUMN]
INT_COLUMNS = ["Src Port", "Dst Port", "Protocol"]
TIMESTAMP_COLUMN = "Timestamp"
TIMESTAMP_DT_COLUMN = "Timestamp_dt"

# Step 1 writes "2020-09-13 12:26:40.000000" -- year first, so unambiguous.
ISO_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def build_dtypes() -> dict[str, Any]:
    """The dtype for every column: str identifiers, int ports, float64 features.

    float64 is what keeps inf and -inf as inf rather than turning them into
    strings or NaN.
    """
    dtypes: dict[str, Any] = {}
    for c in STR_COLUMNS:
        dtypes[c] = "string"
    for c in INT_COLUMNS:
        dtypes[c] = "int64"
    dtypes[TIMESTAMP_COLUMN] = "string"        # the ORIGINAL text, kept as-is
    for c in config.KAGGLE_FEATURES:
        dtypes[c] = "float64"
    return dtypes


def detect_timestamp_format(samples: Sequence[str]) -> dict[str, Any]:
    """Work out how the Timestamp text is laid out.

    Returns {"format", "ambiguous", "dayfirst", "reason", "examples"}.
    'ambiguous' is True only when the first two numbers are BOTH <= 12, so
    01/02/2020 could be 1 February or 2 January and only a human can say which.
    """
    examples = [str(s) for s in samples if s is not None and str(s) != "<NA>"][:5]
    if not examples:
        return {"format": None, "ambiguous": False, "dayfirst": False,
                "reason": "no values to inspect", "examples": []}

    first = examples[0]
    head = first.split(" ")[0]

    # ISO-like: 4-digit year first -> no possible day/month confusion.
    if len(head) >= 10 and head[:4].isdigit() and head[4] in "-/":
        fmt = ISO_FORMAT if "." in first else "%Y-%m-%d %H:%M:%S"
        return {"format": fmt, "ambiguous": False, "dayfirst": False,
                "reason": "year comes first (ISO 8601), so day/month cannot be confused",
                "examples": examples}

    parts = head.replace("-", "/").split("/")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        a, b = int(parts[0]), int(parts[1])
        if a > 12:
            return {"format": "%d/%m/%Y %H:%M:%S", "ambiguous": False, "dayfirst": True,
                    "reason": "first number is %d (> 12), so it must be the day" % a,
                    "examples": examples}
        if b > 12:
            return {"format": "%m/%d/%Y %H:%M:%S", "ambiguous": False, "dayfirst": False,
                    "reason": "second number is %d (> 12), so it must be the day" % b,
                    "examples": examples}
        return {"format": None, "ambiguous": True, "dayfirst": None,
                "reason": ("both leading numbers are <= 12 (%d and %d), so day/month "
                           "order cannot be determined from the data" % (a, b)),
                "examples": examples}

    return {"format": None, "ambiguous": False, "dayfirst": False,
            "reason": "unrecognised layout; pandas will be asked to infer it",
            "examples": examples}


def load_one_csv(path: Path | str, capture: str | None = None,
                 timestamp_format: str | None = None,
                 dayfirst: bool | None = None,
                 verbose: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read ONE flow CSV.  Returns (DataFrame, info dict).

    * identifiers and Label as text, ports and Protocol as int, features float64
    * inf / -inf / NaN are kept exactly as they are -- nothing is filled or dropped
    * the original Timestamp text column is KEPT, and Timestamp_dt is added
      beside it as a real datetime
    """
    path = Path(path)
    capture = capture or FlowCsv(path, 0, 0).capture

    try:
        df = pd.read_csv(path, dtype=build_dtypes(), low_memory=False)
    except ValueError as exc:
        # A dtype clash is worth reporting precisely rather than silently relaxing.
        if verbose:
            print("  [WARN] %s: strict dtypes failed (%s)" % (path.name, exc))
            print("         re-reading with inferred dtypes; the column report below")
            print("         will show which columns did not match.")
        df = pd.read_csv(path, low_memory=False)

    ts_info = detect_timestamp_format(df[TIMESTAMP_COLUMN].head(5).tolist()
                                      if TIMESTAMP_COLUMN in df.columns else [])
    fmt = timestamp_format if timestamp_format is not None else ts_info["format"]

    if ts_info["ambiguous"] and timestamp_format is None and dayfirst is None:
        raise AmbiguousTimestampError(
            "%s: the Timestamp layout is ambiguous. %s\nExamples:\n  %s\n"
            "Re-run with dayfirst=True (01/02 = 1 February) or dayfirst=False "
            "(01/02 = 2 January), or pass timestamp_format explicitly."
            % (path.name, ts_info["reason"], "\n  ".join(ts_info["examples"])))

    if fmt:
        parsed = pd.to_datetime(df[TIMESTAMP_COLUMN], format=fmt, errors="coerce")
    else:
        parsed = pd.to_datetime(df[TIMESTAMP_COLUMN], dayfirst=bool(dayfirst),
                                errors="coerce")
    df[TIMESTAMP_DT_COLUMN] = parsed

    feature_cols = [c for c in config.KAGGLE_FEATURES if c in df.columns]
    numeric = df[feature_cols]
    info = {
        "capture": capture,
        "path": str(path),
        "rows": len(df),
        "columns": len(df.columns),
        "label_counts": df[config.LABEL_COLUMN].value_counts().to_dict()
                        if config.LABEL_COLUMN in df.columns else {},
        "nan_cells": int(numeric.isna().to_numpy().sum()),
        "inf_cells": int(np.isinf(numeric.to_numpy(dtype="float64")).sum()),
        "timestamp_format": fmt,
        "timestamp_reason": ts_info["reason"],
        "timestamp_unparsed": int(parsed.isna().sum()),
        "first_timestamp": parsed.min(),
        "last_timestamp": parsed.max(),
    }

    if verbose:
        print("  %-8s rows %9s | NaN %6d | inf %6d | labels %s"
              % (capture, f"{info['rows']:,}", info["nan_cells"], info["inf_cells"],
                 info["label_counts"]))
        if info["timestamp_unparsed"]:
            print("           [WARN] %d Timestamp value(s) could not be parsed -> NaT"
                  % info["timestamp_unparsed"])
    return df, info


# ===========================================================================
# 3. validate the header
# ===========================================================================
def validate_columns(df: pd.DataFrame, source: str = "<dataframe>",
                     expected: Sequence[str] | None = None,
                     verbose: bool = True) -> None:
    """The header must equal config.FLOW_COLUMNS -- same names, same ORDER.

    Raises ColumnMismatchError with a readable diff.  Timestamp_dt is ignored
    here, because load_one_csv() adds it after the file was read.
    """
    expected = list(expected if expected is not None else config.FLOW_COLUMNS)
    actual = [c for c in df.columns if c != TIMESTAMP_DT_COLUMN]

    if actual == expected:
        if verbose:
            print("  [OK]    %s: header matches config.FLOW_COLUMNS (%d columns)"
                  % (source, len(expected)))
        return

    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    lines = ["%s: header does NOT match config.FLOW_COLUMNS" % source,
             "  expected %d columns, got %d" % (len(expected), len(actual))]
    if missing:
        lines.append("  missing (%d): %s" % (len(missing), missing))
    if extra:
        lines.append("  extra   (%d): %s" % (len(extra), extra))
    if not missing and not extra:
        wrong = [(i + 1, e, a) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
        lines.append("  same names but WRONG ORDER at %d position(s):" % len(wrong))
        for pos, e, a in wrong[:20]:
            lines.append("    position %2d: expected |%s|  got |%s|" % (pos, e, a))
    raise ColumnMismatchError("\n".join(lines))


# ===========================================================================
# STEP 2 -- DATA ACQUISITION
# ===========================================================================
ACQUISITION_FIELDS = ["capture", "file_name", "rows", "columns", "size_mb",
                      "first_timestamp", "last_timestamp", "label_counts",
                      "nan_cells", "inf_cells", "status", "error"]


def acquire_captures(folder: Path | str | None = None, verbose: bool = True,
                     save_report: bool = True, **load_kwargs):
    """STEP 2: load one DataFrame per capture CSV, tagged and reported.

    Reads config.CSV_INPUT_DIR (C:\\MSCAD2022\\CSVs) -- an absolute path, never
    anything relative to the working directory.

    Each frame is tagged exactly as before:
      Source_Capture -- the file stem
      Orig_Row       -- the row's position inside its own file

    Stops with one clear message NAMING THE FOLDER when the folder is missing
    or holds no matching CSVs.  If a single file fails to read, that file is
    reported and the others still load.

    Returns (frames, report DataFrame, failures).
    """
    folder = Path(folder) if folder is not None else config.CSV_INPUT_DIR
    if not folder.is_dir():
        raise MissingCapturesError(
            "Step 2 cannot start: the flow-CSV folder does not exist:\n    %s\n"
            "Set config.CSV_INPUT_DIR to the folder holding the Step 1 "
            "'<capture>_flows.csv' files." % folder)

    matches = sorted(folder.glob(config.FLOW_CSV_PATTERN))
    if not matches:
        raise MissingCapturesError(
            "Step 2 cannot start: no files matching %r in:\n    %s\n"
            "Run Step 1 first, or point config.CSV_INPUT_DIR at the right folder."
            % (config.FLOW_CSV_PATTERN, folder))

    files, _used = find_flow_csvs(folder, require_all=False, verbose=False)

    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for f in files:
        row: dict[str, Any] = {k: "" for k in ACQUISITION_FIELDS}
        row.update({"capture": f.capture, "file_name": f.path.name,
                    "size_mb": f.size_mb})
        try:
            df, info = load_one_csv(f.path, capture=f.capture, verbose=False,
                                    **load_kwargs)
            validate_columns(df, source=f.path.name, verbose=False)
            # tag here, in Step 2, so the per-capture frames already carry it
            df["Source_Capture"] = f.capture
            df["Orig_Row"] = np.arange(len(df), dtype="int64")
            frames[f.capture] = df
            row.update({
                "rows": info["rows"], "columns": int(df.shape[1]),
                "first_timestamp": info["first_timestamp"],
                "last_timestamp": info["last_timestamp"],
                "label_counts": json.dumps({str(k): int(v)
                                            for k, v in info["label_counts"].items()}),
                "nan_cells": info["nan_cells"], "inf_cells": info["inf_cells"],
                "status": "loaded",
            })
        except Exception as exc:                     # one bad file must not stop the rest
            row.update({"status": "FAILED", "error": "%s: %s" % (type(exc).__name__, exc)})
            failures.append({"file": f.path.name, "error": row["error"]})
        rows.append(row)

    report = pd.DataFrame(rows, columns=ACQUISITION_FIELDS)

    if save_report:
        config.STEP2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report.to_csv(config.STEP2_OUTPUT_DIR / "acquisition_report.csv", index=False)

    if verbose:
        show_title("Step 2.1 -- files loaded from %s" % folder)
        show_table(report.drop(columns=["error"] if not failures else []))
        loaded = sum(1 for r in rows if r["status"] == "loaded")
        total_rows = sum(int(r["rows"] or 0) for r in rows if r["status"] == "loaded")
        show_table(kv_table([("folder", str(folder)),
                             ("files found", len(files)),
                             ("loaded", loaded),
                             ("failed", len(failures)),
                             ("total rows", f"{total_rows:,}")]))
        if failures:
            show_note("These files could NOT be read and are excluded from Step 3:")
            show_table(pd.DataFrame(failures))
        if save_report:
            show_note("Report saved to %s"
                      % (config.STEP2_OUTPUT_DIR / "acquisition_report.csv"))

    if not frames:
        raise MissingCapturesError(
            "Step 2 loaded no usable capture at all from:\n    %s" % folder)
    return frames, report, failures


# ===========================================================================
# STEP 3 -- column consistency across the captures
# ===========================================================================
def check_column_consistency(frames: dict[str, pd.DataFrame],
                             verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """Every capture must have the same columns, in the same order.

    Differences are REPORTED in a table rather than silently filled: pandas
    would happily concatenate mismatched frames and quietly insert NaN, which
    would look like missing data later instead of like the bug it is.

    Returns (table, all_consistent).
    """
    reference_name = next(iter(frames))
    reference = list(frames[reference_name].columns)

    rows = []
    for capture, df in frames.items():
        cols = list(df.columns)
        missing = [c for c in reference if c not in cols]
        extra = [c for c in cols if c not in reference]
        same_order = cols == reference
        rows.append({
            "capture": capture, "columns": len(cols),
            "missing_vs_reference": ", ".join(missing) if missing else "",
            "extra_vs_reference": ", ".join(extra) if extra else "",
            "same_order": same_order,
            "status": "ok" if (not missing and not extra and same_order) else "MISMATCH",
        })
    table = pd.DataFrame(rows)
    consistent = bool((table["status"] == "ok").all())

    if verbose:
        show_title("Step 3.1 -- column consistency across captures")
        show_table(table,
                   note="Reference capture: **%s** (%d columns). A mismatch is reported, "
                        "never silently filled -- concatenating mismatched frames would "
                        "insert NaN that looks like missing data."
                        % (reference_name, len(reference)))
        if not consistent:
            show_note("**[FAIL]** the captures do not share one column set -- fix Step 1 "
                      "before merging.")
    return table, consistent


# ===========================================================================
# 4. merge
# ===========================================================================
def merge_flow_csvs(frames: dict[str, pd.DataFrame] | None = None,
                    files: Sequence[FlowCsv] | None = None,
                    capture_order: Sequence[str] | None = None,
                    verbose: bool = True,
                    **load_kwargs) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Merge the per-capture frames into one DataFrame.

    Steps, in this order:
      1 Source_Capture  -- which capture each row came from (from the file name)
      2 Orig_Row        -- the row's index inside its OWN file, so any merged row
                           can be traced straight back to its source CSV
      3 pd.concat in config.CAPTURE_ORDER with ignore_index=True
      4 a STABLE sort by Timestamp_dt (kind="mergesort"): rows with the same
        timestamp keep their original relative order, so the result is
        reproducible instead of depending on pandas internals
      5 columns reordered to config.MERGED_COLUMNS
        (FLOW_COLUMNS + Source_Capture + Orig_Row + Timestamp_dt)

    Values are never altered: inf, NaN and duplicate rows all survive.
    """
    order = list(capture_order if capture_order is not None else config.CAPTURE_ORDER)
    infos: list[dict[str, Any]] = []

    if frames is None:
        if files is None:
            files, _folder = find_flow_csvs(verbose=verbose)
        frames = {}
        for f in files:
            df, info = load_one_csv(f.path, capture=f.capture, verbose=verbose,
                                    **load_kwargs)
            validate_columns(df, source=f.path.name, verbose=verbose)
            frames[f.capture] = df
            infos.append(info)

    pieces: list[pd.DataFrame] = []
    for capture in order:
        if capture not in frames:
            continue
        part = frames[capture].copy()
        part["Source_Capture"] = capture
        # a plain 0..n-1 position inside this file, independent of the merge
        part["Orig_Row"] = np.arange(len(part), dtype="int64")
        pieces.append(part)
    for capture, part in frames.items():          # anything not in CAPTURE_ORDER
        if capture not in order:
            part = part.copy()
            part["Source_Capture"] = capture
            part["Orig_Row"] = np.arange(len(part), dtype="int64")
            pieces.append(part)

    if not pieces:
        raise DataLoadingError("nothing to merge: no frames were provided")

    merged = pd.concat(pieces, ignore_index=True)
    before = len(merged)

    # stable sort -> equal timestamps keep CAPTURE_ORDER / Orig_Row order
    merged = merged.sort_values(TIMESTAMP_DT_COLUMN, kind="mergesort",
                                na_position="last").reset_index(drop=True)
    assert len(merged) == before, "sorting must not change the row count"

    present = [c for c in config.MERGED_COLUMNS if c in merged.columns]
    leftover = [c for c in merged.columns if c not in config.MERGED_COLUMNS]
    merged = merged[present + leftover]

    if verbose:
        print()
        print("  merged shape      : %s" % (merged.shape,))
        print("  column order      : FLOW_COLUMNS + %s" % config.MERGE_EXTRA_COLUMNS)
        print("  sorted by         : %s (stable mergesort)" % TIMESTAMP_DT_COLUMN)
        if leftover:
            print("  [WARN] unexpected extra columns kept at the end: %s" % leftover)
    return merged, infos


# ===========================================================================
# 5. merge checks
# ===========================================================================
def check_merge(merged: pd.DataFrame, infos: Sequence[dict[str, Any]],
                verbose: bool = True) -> tuple[list[dict[str, Any]], bool]:
    """Compare the merged frame against the per-file numbers.  PASS/FAIL each.

    Returns (checks, all_passed).  Nothing is asserted without a number next
    to it.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, expected: Any, actual: Any) -> None:
        checks.append({"check": name, "expected": expected, "actual": actual,
                       "result": "PASS" if expected == actual else "FAIL"})

    # 1. row count
    add("merged rows == sum of file rows",
        sum(i["rows"] for i in infos), len(merged))

    # 2. per-capture row counts
    per_file = {i["capture"]: i["rows"] for i in infos}
    per_merged = merged["Source_Capture"].value_counts().to_dict()
    for capture in sorted(per_file):
        add("rows kept for %s" % capture, per_file[capture], per_merged.get(capture, 0))

    # 3. label counts
    expected_labels: dict[str, int] = {}
    for i in infos:
        for label, n in i["label_counts"].items():
            expected_labels[str(label)] = expected_labels.get(str(label), 0) + int(n)
    actual_labels = {str(k): int(v) for k, v in
                     merged[config.LABEL_COLUMN].value_counts().items()}
    for label in sorted(set(expected_labels) | set(actual_labels)):
        add("Label count %r" % label, expected_labels.get(label, 0),
            actual_labels.get(label, 0))

    # 4. columns
    add("column count", len(config.MERGED_COLUMNS), len(merged.columns))
    add("column names and order", list(config.MERGED_COLUMNS), list(merged.columns))

    # 5. NaN and inf totals, over the 66 features only
    feats = [c for c in config.KAGGLE_FEATURES if c in merged.columns]
    arr = merged[feats].to_numpy(dtype="float64")
    add("NaN cells in the 66 features", sum(i["nan_cells"] for i in infos),
        int(np.isnan(arr).sum()))
    add("inf cells in the 66 features", sum(i["inf_cells"] for i in infos),
        int(np.isinf(arr).sum()))

    all_passed = all(c["result"] == "PASS" for c in checks)
    if verbose:
        print("  %-42s %-28s %-28s %s" % ("CHECK", "EXPECTED", "ACTUAL", "RESULT"))
        print("  " + "-" * 112)
        for c in checks:
            exp, act = str(c["expected"]), str(c["actual"])
            if len(exp) > 26:                       # the long column-name lists
                exp, act = exp[:23] + "...", act[:23] + "..."
            print("  %-42s %-28s %-28s %s" % (c["check"][:42], exp, act, c["result"]))
        print("  " + "-" * 112)
        n_fail = sum(1 for c in checks if c["result"] == "FAIL")
        print("  %d checks, %d PASS, %d FAIL" % (len(checks), len(checks) - n_fail, n_fail))
        if n_fail:
            print()
            print("  [FAIL] the merge is NOT correct -- do not use it:")
            for c in checks:
                if c["result"] == "FAIL":
                    print("     %s: expected %s, got %s"
                          % (c["check"], c["expected"], c["actual"]))
    return checks, all_passed


# ===========================================================================
# 6. save
# ===========================================================================
def verify_merged_file(merged: pd.DataFrame, path: Path | None = None,
                       verbose: bool = True, expect_match: bool = True) -> dict[str, Any]:
    """Prove the merged dataset really is on disk, and say how it compares.

    Checks, in order: the file exists; its absolute path; its size; and its row
    and column counts RE-READ FROM DISK against the in-memory frame.  Prints
    [OK] or a clear [FAIL] -- never "saved" without having looked.

    expect_match=False is used when an existing file was deliberately KEPT
    (OVERWRITE_MERGED is off).  A row-count difference is then the expected
    consequence of that choice, not a failure, so the file is judged on being
    readable and non-empty -- but the difference is still reported loudly, so a
    stale file cannot pass unnoticed.
    """
    path = Path(path) if path is not None else config.MERGED_CSV
    result: dict[str, Any] = {"path": str(path.resolve()) if path.exists() else str(path),
                              "exists": path.exists()}

    if not path.exists():
        result.update({"ok": False, "reason": "the file does not exist"})
        if verbose:
            show_table(kv_table([("expected path", str(path)),
                                 ("exists", False),
                                 ("result", "[FAIL] nothing was written")]))
        return result

    size = path.stat().st_size
    on_disk = load_merged(path, verbose=False)
    result.update({
        "size_bytes": size,
        "size_mb": round(size / 1024 / 1024, 2),
        "rows_in_memory": int(merged.shape[0]), "rows_on_disk": int(on_disk.shape[0]),
        "cols_in_memory": int(merged.shape[1]), "cols_on_disk": int(on_disk.shape[1]),
    })
    rows_ok = result["rows_in_memory"] == result["rows_on_disk"]
    cols_ok = result["cols_in_memory"] == result["cols_on_disk"]
    result["matches_memory"] = bool(rows_ok and cols_ok)
    result["stale"] = bool(not result["matches_memory"] and not expect_match)
    result["ok"] = (bool(rows_ok and cols_ok and size > 0) if expect_match
                    else bool(size > 0))
    result["dataframe"] = on_disk

    if expect_match:
        verdict = ("[OK] the merged dataset is on disk and matches"
                   if result["ok"] else "[FAIL] the file on disk does NOT match")
    elif result["matches_memory"]:
        verdict = "[OK] the existing file was kept, and it matches what was just built"
    else:
        verdict = ("[OK] file is valid, but it DIFFERS from what was just built "
                   "-- set config.OVERWRITE_MERGED = True to replace it")

    if verbose:
        show_table(kv_table([
            ("absolute path", result["path"]),
            ("exists", True),
            ("size (MB)", result["size_mb"]),
            ("rows: in memory -> on disk",
             "%s -> %s  %s" % (f"{result['rows_in_memory']:,}",
                               f"{result['rows_on_disk']:,}",
                               "same" if rows_ok else "DIFFERENT")),
            ("columns: in memory -> on disk",
             "%s -> %s  %s" % (result["cols_in_memory"], result["cols_on_disk"],
                               "same" if cols_ok else "DIFFERENT")),
            ("result", verdict)]))
        if result["stale"]:
            show_note("**The file on disk is NOT the dataset you just built.** It has "
                      "%s rows; the merge produced %s. Everything downstream will read "
                      "the file, so set `config.OVERWRITE_MERGED = True` and re-run this "
                      "cell if Step 1 or Step 2 changed."
                      % (f"{result['rows_on_disk']:,}", f"{result['rows_in_memory']:,}"))
    return result


def save_merged(merged: pd.DataFrame, checks: Sequence[dict[str, Any]] | None = None,
                infos: Sequence[dict[str, Any]] | None = None,
                overwrite: bool | None = None,
                verbose: bool = True) -> dict[str, Any]:
    """STEP 3: write the merged dataset to config.MERGED_CSV, then VERIFY it.

    config.MERGED_CSV is absolute and hangs off DATA_ROOT, so the destination
    does not depend on where the notebook was started or where the code lives.
    The folder is created if missing.

    If the file already exists and overwrite is off, it is NOT rewritten: that
    is said plainly and the existing file is LOADED instead, so the returned
    report always describes a real file on disk.
    """
    overwrite = config.OVERWRITE_MERGED if overwrite is None else overwrite
    config.STEP3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"target": str(config.MERGED_CSV),
                           "folder": str(config.STEP3_OUTPUT_DIR)}

    if config.MERGED_CSV.exists() and not overwrite:
        out["status"] = "kept_existing"
        if verbose:
            show_note("`%s` already exists and `config.OVERWRITE_MERGED` is **False**, "
                      "so it was NOT rewritten -- the existing file is loaded instead. "
                      "Set `OVERWRITE_MERGED = True` to replace it."
                      % config.MERGED_CSV)
    else:
        merged.to_csv(config.MERGED_CSV, index=False)
        out["status"] = "written"
        if verbose:
            show_note("Wrote **%s**" % config.MERGED_CSV)

    # --- Parquet (optional, same folder) ---------------------------------
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        out["parquet"] = "skipped (pyarrow not installed)"
        if verbose:
            show_note("Parquet skipped: pyarrow is not installed "
                      "(`\"%s\" -m pip install pyarrow`). The CSV is unaffected."
                      % sys.executable)
    else:
        if config.MERGED_PARQUET.exists() and not overwrite:
            out["parquet"] = "kept existing"
        else:
            merged.to_parquet(config.MERGED_PARQUET, index=False)
            out["parquet"] = str(config.MERGED_PARQUET)

    # --- the merge report -------------------------------------------------
    report_rows: list[dict[str, Any]] = []
    for i in (infos or []):
        report_rows.append({
            "kind": "file", "name": i["capture"], "expected": "", "actual": "",
            "result": "", "rows": i["rows"], "nan_cells": i["nan_cells"],
            "inf_cells": i["inf_cells"],
            "label_counts": json.dumps({str(k): int(v) for k, v in i["label_counts"].items()}),
            "timestamp_format": i["timestamp_format"] or "",
            "first_timestamp": str(i["first_timestamp"]),
            "last_timestamp": str(i["last_timestamp"]),
        })
    for c in (checks or []):
        report_rows.append({
            "kind": "check", "name": c["check"], "expected": str(c["expected"])[:300],
            "actual": str(c["actual"])[:300], "result": c["result"], "rows": "",
            "nan_cells": "", "inf_cells": "", "label_counts": "",
            "timestamp_format": "", "first_timestamp": "", "last_timestamp": "",
        })
    report_path = config.STEP3_OUTPUT_DIR / "merge_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    out["report"] = str(report_path)

    # --- and prove it ------------------------------------------------------
    out["verification"] = verify_merged_file(
        merged, config.MERGED_CSV, verbose=verbose,
        expect_match=(out["status"] == "written"))
    out["stale"] = bool(out["verification"].get("stale"))
    out["ok"] = bool(out["verification"].get("ok"))
    return out


# ===========================================================================
# 7. load it back from disk
# ===========================================================================
def load_merged(path: Path | str | None = None, prefer_parquet: bool = False,
                verbose: bool = True) -> pd.DataFrame:
    """Read the merged dataset back FROM DISK.

    Reading the CSV back is the stricter check of the two: it proves the file on
    disk really round-trips, rather than just re-using the frame in memory.
    """
    if path is None:
        path = (config.MERGED_PARQUET
                if (prefer_parquet and config.MERGED_PARQUET.exists())
                else config.MERGED_CSV)
    path = Path(path)
    if not path.exists():
        raise DataLoadingError("%s does not exist -- run save_merged() first" % path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        dtypes = build_dtypes()
        dtypes["Source_Capture"] = "string"
        dtypes["Orig_Row"] = "int64"
        df = pd.read_csv(path, dtype=dtypes, low_memory=False,
                         parse_dates=[TIMESTAMP_DT_COLUMN])
    if verbose:
        print("  loaded %s from %s" % ((df.shape,), path))
    return df


# ===========================================================================
# 8. the overview
# ===========================================================================
DESCRIBE_COLUMNS = ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
                    "Flow Byts/s", "Flow Pkts/s", "Pkt Len Mean",
                    "SYN Flag Cnt", "Init Bwd Win Byts"]

PROTOCOL_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP"}


# ---------------------------------------------------------------------------
# display helpers -- rendered tables in Jupyter, plain text everywhere else
# ---------------------------------------------------------------------------
# Every section below builds a DataFrame and hands it to show_table().  Inside
# Jupyter that renders as a real HTML table; in a plain script or under pytest
# there is no IPython, so it falls back to printing.  Nothing here changes a
# value -- it is presentation only.
def _ipython():
    """The live IPython shell, or None when not running inside one."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    try:
        return get_ipython()
    except Exception:
        return None


def in_notebook() -> bool:
    """True when there is an IPython shell to render into."""
    return _ipython() is not None


def show_title(text: str, level: int = 3) -> None:
    """A section heading: markdown in Jupyter, a banner on the console."""
    if in_notebook():
        from IPython.display import Markdown, display
        display(Markdown("%s %s" % ("#" * level, text)))
    else:
        print()
        print("=" * 78)
        print(text)
        print("=" * 78)


def show_note(text: str) -> None:
    """A small italic note under a table."""
    if in_notebook():
        from IPython.display import Markdown, display
        display(Markdown("_%s_" % text))
    else:
        print("  %s" % text)


def show_table(frame: pd.DataFrame, note: str | None = None,
               all_columns: bool = False) -> None:
    """Render a DataFrame as a table, or print it when there is no IPython.

    all_columns=True widens pandas' display just for this one call, using
    option_context so the global options are left exactly as they were.
    """
    if all_columns:
        # scroll horizontally rather than collapsing columns into "..."
        with pd.option_context("display.max_columns", None, "display.width", None):
            if in_notebook():
                from IPython.display import display
                display(frame)
            else:
                print(frame.to_string())
    elif in_notebook():
        from IPython.display import display
        display(frame)
    else:
        with pd.option_context("display.max_columns", None, "display.width", None):
            print(frame.to_string())
    if note:
        show_note(note)


def kv_table(pairs: Sequence[tuple[str, Any]]) -> pd.DataFrame:
    """A plain two-column metric/value table."""
    return pd.DataFrame({"metric": [k for k, _ in pairs],
                         "value": [v for _, v in pairs]})


def normalise_label(name: Any) -> str:
    """Fold a label for MATCHING only -- the original spelling is never changed.

    Trims, lower-cases, and treats spaces and underscores as the same, so
    "Web_Crwling", "web crwling" and " Web Crwling " all match each other.
    Both original spellings are still shown in the comparison table.
    """
    import re
    return re.sub(r"[\s_]+", "_", str(name).strip().lower())


def plot_label_counts(df: pd.DataFrame, path: Path | None = None,
                      verbose: bool = True) -> Path | None:
    """Bar chart of the Label counts on a log scale -> PNG.

    One measure across categories, so the bars are ONE colour and there is no
    legend (the title names the measure).  The log scale is needed because the
    classes differ by orders of magnitude -- and because a log axis makes exact
    values hard to read, every bar carries its count as a direct label.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")                      # no GUI needed, just a file
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("  [SKIP]  matplotlib is not installed, so no chart was drawn.")
            print('             "%s" -m pip install matplotlib' % sys.executable)
        return None

    path = Path(path) if path is not None else config.STEP3_OUTPUT_DIR / "label_counts.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    counts = df[config.LABEL_COLUMN].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    bars = ax.bar(counts.index.astype(str), counts.to_numpy(),
                  color=_BAR, width=0.68)          # width < 1 leaves a gap
    ax.set_yscale("log")
    ax.set_ylabel("flows (log scale)", color=_INK_SOFT, fontsize=10)
    ax.set_title("Flows per class in the merged MSCAD dataset (%s flows, %d classes)"
                 % (f"{len(df):,}", len(counts)), color=_INK, fontsize=12, pad=14)

    # recessive grid, y only; no top/right spines
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_SOFT, labelsize=9)

    top = counts.max()
    ax.set_ylim(0.8, top * 3)
    for bar, value in zip(bars, counts.to_numpy()):
        ax.annotate(f"{int(value):,}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, color=_INK)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def read_kaggle_labels(folder: Path | None = None, verbose: bool = True):
    """Read the Label column of the Kaggle MSCAD file.  Never raises.

    Returns (counts dict keyed by the ORIGINAL label spelling, info dict).
    counts is None when the folder or the file is missing, or the file cannot
    be read -- the caller prints one line and carries on, because nothing in
    Step 2 depends on this comparison.
    """
    info: dict[str, Any] = {"path": None, "candidates": [], "problem": None}
    base = Path(folder) if folder is not None else config.KAGGLE_CSV_DIR
    info["folder"] = str(base)

    if not base.is_dir():
        info["problem"] = "folder does not exist: %s" % base
        if verbose:
            print("  Kaggle reference skipped -- %s" % info["problem"])
        return None, info

    path, candidates = config.find_kaggle_csv(base)
    info["candidates"] = [p.name for p in candidates]
    if path is None:
        info["problem"] = "no .csv or .xlsx in %s" % base
        if verbose:
            print("  Kaggle reference skipped -- %s" % info["problem"])
        return None, info
    info["path"] = str(path)

    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            try:
                import openpyxl  # noqa: F401
            except ImportError:
                info["problem"] = ("%s is a spreadsheet and openpyxl is not installed "
                                   "(pip install openpyxl)" % path.name)
                if verbose:
                    print("  Kaggle reference skipped -- %s" % info["problem"])
                return None, info
            kag = pd.read_excel(path)
        else:
            kag = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        info["problem"] = "could not read %s: %r" % (path.name, exc)
        if verbose:
            print("  Kaggle reference skipped -- %s" % info["problem"])
        return None, info

    label_col = next((c for c in kag.columns
                      if str(c).strip().lower() == config.LABEL_COLUMN.lower()), None)
    if label_col is None:
        info["problem"] = "%s has no Label column" % path.name
        if verbose:
            print("  Kaggle reference skipped -- %s" % info["problem"])
        return None, info

    info["rows"] = len(kag)
    info["label_column"] = str(label_col)
    return {str(k): int(v) for k, v in kag[label_col].value_counts().items()}, info


def compare_labels_with_kaggle(df: pd.DataFrame, folder: Path | None = None,
                               verbose: bool = True):
    """Reference-only comparison of our Label counts with the Kaggle file's.

    Labels are matched on a NORMALISED form (trimmed, lower-cased, spaces and
    underscores equivalent) but both original spellings are shown -- nothing is
    renamed.  Labels present on only one side get their own small table.

    Returns a dict with the tables, or None when there is no Kaggle file.
    """
    ours = {str(k): int(v) for k, v in df[config.LABEL_COLUMN].value_counts().items()}
    theirs, info = read_kaggle_labels(folder, verbose=verbose)

    if theirs is None:
        if verbose:
            show_note("Nothing to compare against, so only our own counts are shown. "
                      "This is not an error: Step 2 does not depend on the Kaggle file.")
            show_table(pd.DataFrame({"Label": list(ours),
                                     "ours": [ours[k] for k in ours]})
                       .sort_values("ours", ascending=False).reset_index(drop=True))
        return None

    if verbose:
        if len(info["candidates"]) > 1:
            show_note("%d files in %s: %s -- using **%s**"
                      % (len(info["candidates"]), info["folder"],
                         ", ".join(info["candidates"]), Path(info["path"]).name))
        else:
            show_note("Kaggle file: %s (%s rows)"
                      % (info["path"], f"{info.get('rows', 0):,}"))

    ours_norm = {normalise_label(k): k for k in ours}
    theirs_norm = {normalise_label(k): k for k in theirs}
    shared = [k for k in ours_norm if k in theirs_norm]

    rows = []
    for key in sorted(shared, key=lambda k: -ours[ours_norm[k]]):
        our_name, their_name = ours_norm[key], theirs_norm[key]
        o, t = ours[our_name], theirs[their_name]
        rows.append({"Label": our_name, "ours": o, "kaggle": t,
                     "difference": o - t,
                     "% difference": (100.0 * (o - t) / t) if t else float("nan")})
    table = pd.DataFrame(rows)
    if len(table):
        total_o, total_t = int(table["ours"].sum()), int(table["kaggle"].sum())
        table = pd.concat([table, pd.DataFrame([{
            "Label": "TOTAL (matched)", "ours": total_o, "kaggle": total_t,
            "difference": total_o - total_t,
            "% difference": (100.0 * (total_o - total_t) / total_t) if total_t else float("nan"),
        }])], ignore_index=True)
        table["% difference"] = table["% difference"].round(2)

    one_sided = pd.DataFrame(
        [{"Label": ours_norm[k], "side": "ours only", "flows": ours[ours_norm[k]]}
         for k in ours_norm if k not in theirs_norm]
        + [{"Label": theirs_norm[k], "side": "Kaggle only", "flows": theirs[theirs_norm[k]]}
           for k in theirs_norm if k not in ours_norm])

    spelling = pd.DataFrame(
        [{"our spelling": ours_norm[k], "Kaggle spelling": theirs_norm[k]}
         for k in shared if ours_norm[k] != theirs_norm[k]])

    if verbose:
        if len(table):
            show_table(table,
                       note=("'% difference' is (ours - kaggle) / kaggle x 100. "
                             "REFERENCE ONLY: the Kaggle file was produced by the Java "
                             "CICFlowMeter with its own settings, so these are not "
                             "expected to match and nothing here is adjusted to make "
                             "them match."))
        else:
            show_note("No label matched between the two files.")
        if len(one_sided):
            show_note("Labels present on only one side (nothing was renamed):")
            show_table(one_sided)
        if len(spelling):
            show_note("Matched on the normalised form, but spelled differently:")
            show_table(spelling)

    return {"table": table, "one_sided": one_sided, "spelling": spelling,
            "ours": ours, "kaggle": theirs, "info": info}


def describe_table(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """describe() for the key features, computed WITHOUT inf so it is readable.

    inf makes mean, std and the quantiles undefined and makes numpy emit
    "invalid value encountered in subtract" warnings.  For the DISPLAY only we
    replace inf with NaN on a COPY, and add an inf_count column so the inf
    values are still visible as a number.  df itself is never modified.
    """
    cols = [c for c in (columns or DESCRIBE_COLUMNS) if c in df.columns]
    block = df[cols]
    inf_counts = {c: int(np.isinf(block[c].to_numpy(dtype="float64")).sum()) for c in cols}
    finite = block.replace([np.inf, -np.inf], np.nan)
    out = finite.describe().T
    out.insert(0, "inf_count", [inf_counts[c] for c in out.index])
    return out


def summarize_dataset(df: pd.DataFrame, make_plot: bool = True,
                      compare_kaggle: bool = True) -> dict[str, Any]:
    """The full overview of the reloaded merged dataset, as rendered tables.

    Counts and describes only: nothing is dropped, filled or changed anywhere.
    """
    out: dict[str, Any] = {}
    feats = [c for c in config.KAGGLE_FEATURES if c in df.columns]

    # --- 1. shape and memory --------------------------------------------
    show_title("1. Shape and memory")
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    out["shape"], out["memory_mb"] = df.shape, round(mem_mb, 1)
    verdict = ("over 2 GB -- Step 3 could downcast float64 to float32 and the "
               "identifier columns to category (typically 50-70% smaller). NOT "
               "applied here: Step 2 does not change values."
               if mem_mb > 2048 else
               "comfortably under 2 GB, so no downcasting is suggested")
    show_table(kv_table([("rows", f"{df.shape[0]:,}"),
                         ("columns", df.shape[1]),
                         ("memory, deep (MB)", round(mem_mb, 1)),
                         ("verdict", verdict)]))

    # --- 2 & 3. head and tail -------------------------------------------
    show_title("2. head(10)")
    show_table(df.head(10), all_columns=True,
               note="All %d columns; scroll sideways to see them." % df.shape[1])
    show_title("3. tail(5)")
    show_table(df.tail(5), all_columns=True)

    # --- 4. info ---------------------------------------------------------
    show_title("4. Columns, dtypes and null counts")
    info_tbl = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_null": [int(df[c].notna().sum()) for c in df.columns],
        "null": [int(df[c].isna().sum()) for c in df.columns],
    })
    out["info_table"] = info_tbl
    show_table(info_tbl)

    # --- 5. labels -------------------------------------------------------
    show_title("5. Label counts")
    counts = df[config.LABEL_COLUMN].value_counts()
    out["label_counts"] = counts.to_dict()
    label_tbl = pd.DataFrame({"Label": counts.index.astype(str),
                              "flows": counts.to_numpy(),
                              "share_%": (100.0 * counts / len(df)).round(3).to_numpy()})
    label_tbl = pd.concat([label_tbl, pd.DataFrame([{
        "Label": "TOTAL", "flows": len(df), "share_%": 100.0}])], ignore_index=True)
    imbalance = counts.max() / max(1, counts.min())
    out["imbalance_ratio"] = round(float(imbalance), 1)
    show_table(label_tbl,
               note="Largest class / smallest class = **%.1f : 1** -- Step 3 has to "
                    "deal with this." % imbalance)

    # --- 6. capture x label ---------------------------------------------
    show_title("6. Source_Capture x Label")
    crosstab = pd.crosstab(df["Source_Capture"], df[config.LABEL_COLUMN], margins=True)
    out["crosstab"] = crosstab
    show_table(crosstab)

    # --- 7. time ---------------------------------------------------------
    show_title("7. First and last Timestamp per capture")
    times = df.groupby("Source_Capture")[TIMESTAMP_DT_COLUMN].agg(["min", "max", "count"])
    times["duration"] = times["max"] - times["min"]
    out["time_ranges"] = times
    n_nat = int(df[TIMESTAMP_DT_COLUMN].isna().sum())
    is_sorted = bool(df[TIMESTAMP_DT_COLUMN].dropna().is_monotonic_increasing)
    out["sorted_by_time"] = is_sorted
    show_table(times.reset_index(),
               note="Unparsed timestamps (NaT): **%d**. Rows in ascending Timestamp "
                    "order: **%s**." % (n_nat, is_sorted))

    # --- 8. identifiers and protocols ------------------------------------
    show_title("8. Identifiers and protocols")
    out["unique_src_ip"] = int(df["Src IP"].nunique())
    out["unique_dst_ip"] = int(df["Dst IP"].nunique())
    show_table(kv_table([("unique Src IP", f"{out['unique_src_ip']:,}"),
                         ("unique Dst IP", f"{out['unique_dst_ip']:,}"),
                         ("unique Src Port", f"{df['Src Port'].nunique():,}"),
                         ("unique Dst Port", f"{df['Dst Port'].nunique():,}")]))
    proto = df["Protocol"].value_counts().sort_index()
    out["protocol_counts"] = proto.to_dict()
    show_table(pd.DataFrame({
        "Protocol": proto.index.astype(int),
        "name": [PROTOCOL_NAMES.get(int(n), "other") for n in proto.index],
        "flows": proto.to_numpy(),
        "share_%": (100.0 * proto / len(df)).round(3).to_numpy()}))

    # --- 9. NaN and inf ---------------------------------------------------
    show_title("9. NaN and inf per column")
    nan_by_col = df[feats].isna().sum()
    inf_by_col = pd.Series({c: int(np.isinf(df[c].to_numpy(dtype="float64")).sum())
                            for c in feats})
    out["nan_total"], out["inf_total"] = int(nan_by_col.sum()), int(inf_by_col.sum())
    nz = pd.DataFrame({"column": feats,
                       "NaN": nan_by_col.reindex(feats).to_numpy(),
                       "inf": inf_by_col.reindex(feats).to_numpy()})
    nz = nz[(nz["NaN"] > 0) | (nz["inf"] > 0)].reset_index(drop=True)
    out["nan_inf_table"] = nz
    if len(nz):
        show_table(nz,
                   note="Totals across the 66 features: **%s NaN**, **%s inf**. Both are "
                        "KEPT -- inf in a rate column is the correct answer for a "
                        "zero-duration flow, and Step 3 decides what to do about it."
                        % (f"{out['nan_total']:,}", f"{out['inf_total']:,}"))
    else:
        show_table(pd.DataFrame([{"column": "(none)", "NaN": 0, "inf": 0}]),
                   note="No column has any NaN or inf.")

    # --- 10. duplicates ---------------------------------------------------
    show_title("10. Duplicates (counted, never dropped)")
    exact = int(df.duplicated().sum())
    same_features = int(df.duplicated(subset=feats).sum())
    same_flow = int(df.duplicated(subset=config.ID_COLUMNS).sum())
    out["duplicate_rows"], out["identical_feature_rows"] = exact, same_features
    show_table(kv_table([
        ("exact duplicate rows (all %d columns)" % df.shape[1], f"{exact:,}"),
        ("rows with identical 66-feature values", f"{same_features:,}"),
        ("  as a share of the data", "%.2f%%" % (100.0 * same_features / len(df))),
        ("rows with a repeated 5-tuple + Timestamp", f"{same_flow:,}")]),
        note="Expected in floods, port scans and brute force, where thousands of flows "
             "genuinely look the same. Nothing is dropped here.")

    # --- 11. describe -----------------------------------------------------
    show_title("11. describe() for the key features")
    desc = describe_table(df)
    out["describe"] = desc
    show_table(desc,
               note="Computed on a copy with inf replaced by NaN, so the statistics are "
                    "readable and numpy emits no 'invalid value' warnings; the inf_count "
                    "column shows how many inf values each feature really has. The "
                    "dataset itself still holds its inf values.")

    # --- 12. chart --------------------------------------------------------
    if make_plot:
        show_title("12. Label bar chart (log scale)")
        out["plot"] = plot_label_counts(df)

    # --- 13. Kaggle reference --------------------------------------------
    if compare_kaggle:
        show_title("13. Our Label counts vs the Kaggle MSCAD file (reference only)")
        out["kaggle_comparison"] = compare_labels_with_kaggle(df)

    return out


# ===========================================================================
# STEP 3 -- report files
# ===========================================================================
def save_step3_reports(merged: pd.DataFrame, summary: dict[str, Any] | None = None,
                       verbose: bool = True) -> dict[str, str]:
    """Write the Step 3 reports beside the merged dataset, in STEP3_OUTPUT_DIR.

    label_counts.csv, source_capture_x_label.csv, label_counts.png and
    summary.txt -- all under C:\\MSCAD2022\\outputs\\step3, an absolute path.
    """
    config.STEP3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    counts = merged[config.LABEL_COLUMN].value_counts()
    label_tbl = pd.DataFrame({"Label": counts.index.astype(str),
                              "flows": counts.to_numpy(),
                              "share_%": (100.0 * counts / len(merged)).round(4).to_numpy()})
    path = config.STEP3_OUTPUT_DIR / "label_counts.csv"
    label_tbl.to_csv(path, index=False)
    written["label_counts"] = str(path)

    crosstab = pd.crosstab(merged["Source_Capture"], merged[config.LABEL_COLUMN],
                           margins=True)
    path = config.STEP3_OUTPUT_DIR / "source_capture_x_label.csv"
    crosstab.to_csv(path)
    written["crosstab"] = str(path)

    chart = plot_label_counts(merged, verbose=False)
    if chart is not None:
        written["chart"] = str(chart)

    feats = [c for c in config.KAGGLE_FEATURES if c in merged.columns]
    arr = merged[feats].to_numpy(dtype="float64")
    lines = [
        "MSCAD Step 3 -- merged dataset summary",
        "=" * 60,
        "merged file      : %s" % config.MERGED_CSV,
        "rows             : %d" % merged.shape[0],
        "columns          : %d" % merged.shape[1],
        "captures         : %s" % ", ".join(sorted(merged["Source_Capture"].unique())),
        "first timestamp  : %s" % merged[TIMESTAMP_DT_COLUMN].min(),
        "last timestamp   : %s" % merged[TIMESTAMP_DT_COLUMN].max(),
        "NaN cells (66 features) : %d" % int(np.isnan(arr).sum()),
        "inf cells (66 features) : %d" % int(np.isinf(arr).sum()),
        "exact duplicate rows    : %d" % int(merged.duplicated().sum()),
        "identical-feature rows  : %d" % int(merged.duplicated(subset=feats).sum()),
        "",
        "label counts:",
    ]
    for label, n in counts.items():
        lines.append("  %-18s %10d  (%.3f%%)" % (label, n, 100.0 * n / len(merged)))
    lines += ["", "Nothing was cleaned: inf, NaN and duplicate rows are all kept."]
    path = config.STEP3_OUTPUT_DIR / "summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written["summary"] = str(path)

    if verbose:
        show_title("Step 3 report files")
        show_table(pd.DataFrame({"report": list(written), "path": list(written.values())}))
    return written
