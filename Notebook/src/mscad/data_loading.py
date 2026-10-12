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
def save_merged(merged: pd.DataFrame, checks: Sequence[dict[str, Any]] | None = None,
                infos: Sequence[dict[str, Any]] | None = None,
                overwrite: bool | None = None,
                verbose: bool = True) -> dict[str, Any]:
    """Write MERGED_CSV, MERGED_PARQUET (if pyarrow is there) and merge_report.csv.

    An existing file is kept unless config.OVERWRITE_MERGED is True.
    """
    overwrite = config.OVERWRITE_MERGED if overwrite is None else overwrite
    config.MERGED_DIR.mkdir(parents=True, exist_ok=True)
    config.STEP2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}

    # --- CSV -------------------------------------------------------------
    if config.MERGED_CSV.exists() and not overwrite:
        out["csv"] = "skipped (exists, config.OVERWRITE_MERGED is False)"
        if verbose:
            print("  [SKIP]  %s already exists; set config.OVERWRITE_MERGED = True to redo it"
                  % config.MERGED_CSV.name)
    else:
        merged.to_csv(config.MERGED_CSV, index=False)
        out["csv"] = str(config.MERGED_CSV)
        if verbose:
            print("  [SAVE]  %s  (%.1f MB)"
                  % (config.MERGED_CSV, config.MERGED_CSV.stat().st_size / 1024 / 1024))

    # --- Parquet ---------------------------------------------------------
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        out["parquet"] = "skipped (pyarrow not installed)"
        if verbose:
            print("  [SKIP]  Parquet: pyarrow is not installed.  To enable it:")
            print('             "%s" -m pip install pyarrow' % sys.executable)
            print("          Continuing with the CSV only -- nothing is lost.")
    else:
        if config.MERGED_PARQUET.exists() and not overwrite:
            out["parquet"] = "skipped (exists, config.OVERWRITE_MERGED is False)"
            if verbose:
                print("  [SKIP]  %s already exists" % config.MERGED_PARQUET.name)
        else:
            merged.to_parquet(config.MERGED_PARQUET, index=False)
            out["parquet"] = str(config.MERGED_PARQUET)
            if verbose:
                print("  [SAVE]  %s  (%.1f MB)"
                      % (config.MERGED_PARQUET,
                         config.MERGED_PARQUET.stat().st_size / 1024 / 1024))

    # --- merge report ----------------------------------------------------
    report_rows: list[dict[str, Any]] = []
    for i in (infos or []):
        report_rows.append({
            "kind": "file", "name": i["capture"], "expected": "",
            "actual": "", "result": "",
            "rows": i["rows"], "nan_cells": i["nan_cells"], "inf_cells": i["inf_cells"],
            "label_counts": json.dumps({str(k): int(v)
                                        for k, v in i["label_counts"].items()}),
            "timestamp_format": i["timestamp_format"] or "",
            "first_timestamp": str(i["first_timestamp"]),
            "last_timestamp": str(i["last_timestamp"]),
        })
    for c in (checks or []):
        report_rows.append({
            "kind": "check", "name": c["check"],
            "expected": str(c["expected"])[:300], "actual": str(c["actual"])[:300],
            "result": c["result"], "rows": "", "nan_cells": "", "inf_cells": "",
            "label_counts": "", "timestamp_format": "",
            "first_timestamp": "", "last_timestamp": "",
        })
    report = pd.DataFrame(report_rows)
    report_path = config.STEP2_OUTPUT_DIR / "merge_report.csv"
    report.to_csv(report_path, index=False)
    out["report"] = str(report_path)
    if verbose:
        print("  [SAVE]  %s" % report_path)
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

    path = Path(path) if path is not None else config.STEP2_OUTPUT_DIR / "label_counts.png"
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


def compare_labels_with_kaggle(df: pd.DataFrame, verbose: bool = True):
    """Reference only: our Label counts beside the Kaggle MSCAD.csv counts.

    Returns None when there is no Kaggle file -- that is not an error, and
    nothing in Step 2 depends on it.
    """
    ours = {str(k): int(v) for k, v in df[config.LABEL_COLUMN].value_counts().items()}
    path = config.find_kaggle_file()
    if path is None:
        if verbose:
            print("  no Kaggle file in %s, so there is nothing to compare against."
                  % config.KAGGLE_DIR)
            print("  our own label counts:")
            for label, n in sorted(ours.items(), key=lambda kv: -kv[1]):
                print("     %-18s %12s" % (label, f"{n:,}"))
        return None

    kag = (pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls")
           else pd.read_csv(path, low_memory=False))
    kag.columns = [str(c).strip() for c in kag.columns]
    theirs = {str(k): int(v) for k, v in kag[config.LABEL_COLUMN].value_counts().items()}

    if verbose:
        print("  %-18s %14s %14s   %s" % ("LABEL", "OURS", "KAGGLE", "NOTE"))
        print("  " + "-" * 66)
        for label in sorted(set(ours) | set(theirs), key=lambda k: -ours.get(k, 0)):
            note = ""
            if label not in theirs:
                note = "only ours (we create Benign)"
            elif label not in ours:
                note = "only Kaggle"
            print("  %-18s %14s %14s   %s"
                  % (label, f"{ours.get(label, 0):,}", f"{theirs.get(label, 0):,}", note))
        print("  " + "-" * 66)
        print("  %-18s %14s %14s" % ("TOTAL", f"{sum(ours.values()):,}",
                                     f"{sum(theirs.values()):,}"))
        print()
        print("  REFERENCE ONLY. The Kaggle file was produced by the Java CICFlowMeter")
        print("  with its own settings, so these counts are not expected to be equal,")
        print("  and nothing here is adjusted to make them match.")
    return {"ours": ours, "kaggle": theirs}


def summarize_dataset(df: pd.DataFrame, make_plot: bool = True,
                      compare_kaggle: bool = True) -> dict[str, Any]:
    """Print the full overview of the reloaded merged dataset.

    Counts only -- nothing is dropped, filled or changed anywhere in here.
    """
    out: dict[str, Any] = {}

    print("=" * 78)
    print("1. shape and memory")
    print("=" * 78)
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    out["shape"] = df.shape
    out["memory_mb"] = round(mem_mb, 1)
    print("  shape          : %s" % (df.shape,))
    print("  memory (deep)  : %.1f MB" % mem_mb)
    if mem_mb > 2048:
        print("  [NOTE] over 2 GB in memory. Step 3 could downcast float64 -> float32")
        print("         and the identifier columns -> category, which typically saves")
        print("         50-70%. NOT applied here: Step 2 does not change values.")
    else:
        print("  comfortably under the 2 GB mark, so no downcasting is suggested.")

    print()
    print("=" * 78)
    print("2. head(10)")
    print("=" * 78)
    print(df.head(10).to_string())

    print()
    print("=" * 78)
    print("3. tail(5)")
    print("=" * 78)
    print(df.tail(5).to_string())

    print()
    print("=" * 78)
    print("4. info()")
    print("=" * 78)
    df.info(verbose=True, show_counts=True)

    print()
    print("=" * 78)
    print("5. Label counts and percentages")
    print("=" * 78)
    counts = df[config.LABEL_COLUMN].value_counts()
    out["label_counts"] = counts.to_dict()
    print("  %-18s %12s %9s" % ("LABEL", "FLOWS", "SHARE"))
    print("  " + "-" * 42)
    for label, n in counts.items():
        print("  %-18s %12s %8.3f%%" % (label, f"{n:,}", 100.0 * n / len(df)))
    print("  " + "-" * 42)
    print("  %-18s %12s %8.3f%%" % ("TOTAL", f"{len(df):,}", 100.0))
    imbalance = counts.max() / max(1, counts.min())
    out["imbalance_ratio"] = round(float(imbalance), 1)
    print("  largest class / smallest class = %.1f : 1   <- Step 3 has to deal with this"
          % imbalance)

    print()
    print("=" * 78)
    print("6. Source_Capture x Label")
    print("=" * 78)
    crosstab = pd.crosstab(df["Source_Capture"], df[config.LABEL_COLUMN], margins=True)
    out["crosstab"] = crosstab
    print(crosstab.to_string())

    print()
    print("=" * 78)
    print("7. first and last Timestamp per capture")
    print("=" * 78)
    times = df.groupby("Source_Capture")[TIMESTAMP_DT_COLUMN].agg(["min", "max", "count"])
    times["duration"] = times["max"] - times["min"]
    out["time_ranges"] = times
    print(times.to_string())
    n_nat = int(df[TIMESTAMP_DT_COLUMN].isna().sum())
    print()
    print("  unparsed timestamps (NaT): %d" % n_nat)
    is_sorted = df[TIMESTAMP_DT_COLUMN].dropna().is_monotonic_increasing
    out["sorted_by_time"] = bool(is_sorted)
    print("  rows are in ascending Timestamp order: %s" % is_sorted)

    print()
    print("=" * 78)
    print("8. identifiers and protocols")
    print("=" * 78)
    out["unique_src_ip"] = int(df["Src IP"].nunique())
    out["unique_dst_ip"] = int(df["Dst IP"].nunique())
    print("  unique Src IP : %s" % f"{out['unique_src_ip']:,}")
    print("  unique Dst IP : %s" % f"{out['unique_dst_ip']:,}")
    print("  unique Src Port / Dst Port : %s / %s"
          % (f"{df['Src Port'].nunique():,}", f"{df['Dst Port'].nunique():,}"))
    print()
    proto = df["Protocol"].value_counts().sort_index()
    out["protocol_counts"] = proto.to_dict()
    print("  %-10s %-6s %12s %9s" % ("PROTOCOL", "NAME", "FLOWS", "SHARE"))
    print("  " + "-" * 42)
    for num, n in proto.items():
        print("  %-10s %-6s %12s %8.3f%%"
              % (num, PROTOCOL_NAMES.get(int(num), "other"), f"{n:,}",
                 100.0 * n / len(df)))

    print()
    print("=" * 78)
    print("9. NaN and inf, per column (only columns with a non-zero count)")
    print("=" * 78)
    feats = [c for c in config.KAGGLE_FEATURES if c in df.columns]
    nan_by_col = df[feats].isna().sum()
    inf_by_col = pd.Series(
        {c: int(np.isinf(df[c].to_numpy(dtype="float64")).sum()) for c in feats})
    out["nan_total"] = int(nan_by_col.sum())
    out["inf_total"] = int(inf_by_col.sum())
    shown = False
    for c in feats:
        if nan_by_col[c] or inf_by_col[c]:
            shown = True
            print("  %-22s NaN %8s   inf %8s"
                  % (c, f"{int(nan_by_col[c]):,}", f"{int(inf_by_col[c]):,}"))
    if not shown:
        print("  no column has any NaN or inf.")
    print()
    print("  totals: NaN %s cells, inf %s cells (across the 66 features)"
          % (f"{out['nan_total']:,}", f"{out['inf_total']:,}"))
    print("  Both are KEPT. inf in Flow Byts/s and Flow Pkts/s is the correct answer")
    print("  for a zero-duration flow; Step 3 decides what to do about it.")

    print()
    print("=" * 78)
    print("10. duplicates (counted, NOT dropped)")
    print("=" * 78)
    exact = int(df.duplicated().sum())
    same_features = int(df.duplicated(subset=feats).sum())
    same_flow = int(df.duplicated(subset=config.ID_COLUMNS).sum())
    out["duplicate_rows"] = exact
    out["identical_feature_rows"] = same_features
    print("  exact duplicate rows (all %d columns) : %s"
          % (len(df.columns), f"{exact:,}"))
    print("  rows with identical 66-feature values : %s   (%.2f%% of the data)"
          % (f"{same_features:,}", 100.0 * same_features / len(df)))
    print("  rows with a repeated 5-tuple+Timestamp: %s" % f"{same_flow:,}")
    print("  Expected in floods, port scans and brute force, where thousands of")
    print("  flows genuinely look the same. Nothing is dropped here.")

    print()
    print("=" * 78)
    print("11. describe() for the key features")
    print("=" * 78)
    cols = [c for c in DESCRIBE_COLUMNS if c in df.columns]
    desc = df[cols].describe().T
    out["describe"] = desc
    print(desc.to_string())
    print()
    print("  NOTE: inf values make mean/std inf for the rate columns. That is")
    print("  arithmetic, not a bug -- see section 9.")

    if make_plot:
        print()
        print("=" * 78)
        print("12. Label bar chart (log scale)")
        print("=" * 78)
        out["plot"] = plot_label_counts(df)

    if compare_kaggle:
        print()
        print("=" * 78)
        print("13. our Label counts vs the Kaggle MSCAD.csv (reference only)")
        print("=" * 78)
        out["kaggle_comparison"] = compare_labels_with_kaggle(df)

    return out
