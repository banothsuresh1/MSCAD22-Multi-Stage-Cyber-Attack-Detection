"""
src/mscad/preprocessing.py -- STEP 4: clean, encode and scale the merged dataset.

Input   C:\\MSCAD2022\\outputs\\step3\\MSCAD_merged.csv   (Step 3's output)
Output  C:\\MSCAD2022\\outputs\\step4\\                   (created if missing)

What Step 4 DOES
----------------
    4.1 clean    inf, NaN, constant columns, split off the non-feature columns,
                 report duplicates and labels -- row ORDER is never changed
    4.2 encode   Label -> integer + binary target; Protocol -> one-hot
    4.3 scale    numeric features only, with the fitted scaler saved

What Step 4 deliberately does NOT do
------------------------------------
No train/test split. No class-imbalance handling (no SMOTE, no resampling, no
class weights). No feature selection. No modelling. Those are later steps.

Duplicate rows are COUNTED, never dropped: the attack-sequence work in a later
step depends on the original row order, and 54% of the rows share identical
feature values because thousands of scan and brute-force flows genuinely look
alike.

Why both features_clean.csv and features_scaled.csv are saved
-------------------------------------------------------------
The scaler here is fitted on the WHOLE dataset, which is fine for inspection
but would leak information from the test set once a split exists. Keeping the
unscaled cleaned features means a later step can refit the scaler on the
training split alone.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from . import data_loading as dl  # noqa: E402

# the display helpers from Step 2/3, so Step 4 renders the same way
show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir


class PreprocessingError(Exception):
    """Raised when Step 4 cannot continue without guessing."""


# ===========================================================================
# small helpers
# ===========================================================================
def _numeric_block(df: pd.DataFrame, columns: Sequence[str] | None = None) -> np.ndarray:
    cols = list(columns) if columns is not None else list(df.columns)
    return df[cols].to_numpy(dtype="float64")


def count_inf(df: pd.DataFrame, columns: Sequence[str] | None = None) -> dict[str, int]:
    """inf count per column.  isinf is an elementwise test, so it never warns."""
    cols = [c for c in (columns or df.columns) if pd.api.types.is_numeric_dtype(df[c])]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return {c: int(np.isinf(df[c].to_numpy(dtype="float64")).sum()) for c in cols}


def count_nan(df: pd.DataFrame, columns: Sequence[str] | None = None) -> dict[str, int]:
    cols = [c for c in (columns or df.columns) if pd.api.types.is_numeric_dtype(df[c])]
    return {c: int(df[c].isna().sum()) for c in cols}


def finite_max(series: pd.Series) -> float:
    """The largest value that is not inf.  NaN when the column has none."""
    values = series.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else float("nan")


# ===========================================================================
# 4.1  CLEAN
# ===========================================================================
def split_metadata(df: pd.DataFrame, verbose: bool = True):
    """Separate the frame into (features, labels, metadata), keeping row order.

    Row i of all three refers to the same flow: nothing is sorted, filtered or
    reindexed here.
    """
    meta_cols = [c for c in config.METADATA_COLUMNS if c in df.columns]
    missing = [c for c in config.METADATA_COLUMNS if c not in df.columns]
    if config.LABEL_COLUMN not in df.columns:
        raise PreprocessingError(
            "the merged dataset has no %r column, so Step 4 cannot build a target"
            % config.LABEL_COLUMN)

    metadata = df[meta_cols].copy()
    labels = df[[config.LABEL_COLUMN]].copy()
    # Columns in METADATA_SHARED_COLUMNS are COPIED into metadata but stay in the
    # features: Protocol is an identifier AND the input to the one-hot encoding,
    # so dropping it here would silently remove the Protocol_* columns.
    shared = [c for c in config.METADATA_SHARED_COLUMNS if c in meta_cols]
    drop_cols = [c for c in meta_cols if c not in shared]
    features = df.drop(columns=drop_cols + [config.LABEL_COLUMN]).copy()

    if verbose:
        show_table(kv_table([
            ("rows", f"{len(df):,}"),
            ("columns in", df.shape[1]),
            ("metadata columns kept aside", len(meta_cols)),
            ("label column", config.LABEL_COLUMN),
            ("feature columns", features.shape[1]),
        ]), note="Metadata columns: %s%s%s" % (
            ", ".join(meta_cols),
            "" if not missing else "  (not present: %s)" % ", ".join(missing),
            ("  -- %s is also kept as a feature, for the one-hot encoding"
             % ", ".join(shared)) if shared else ""))
    return features, labels, metadata


def handle_infinities(features: pd.DataFrame, strategy: str | None = None,
                      verbose: bool = True):
    """Deal with inf.  Returns (features, report DataFrame, info).

    "cap"        +inf -> the column's largest FINITE value, -inf -> smallest,
                 plus a binary 'zero_duration' column marking Flow Duration == 0
                 (the flows that produced the inf in the first place).
    "nan_median" inf -> NaN, so the median imputation below fills it.
    """
    strategy = (strategy or config.INF_STRATEGY).lower()
    if strategy not in ("cap", "nan_median"):
        raise PreprocessingError(
            "INF_STRATEGY must be 'cap' or 'nan_median', not %r" % strategy)

    out = features.copy()
    before = count_inf(out)
    affected = [c for c, n in before.items() if n > 0]

    info: dict[str, Any] = {"strategy": strategy, "columns_affected": affected,
                            "inf_before_total": int(sum(before.values()))}

    if strategy == "cap":
        # the marker column first, while the inf values still identify the rows
        if "Flow Duration" in out.columns:
            out[config.ZERO_DURATION_COLUMN] = (
                out["Flow Duration"].to_numpy(dtype="float64") == 0).astype("int64")
            info["zero_duration_rows"] = int(out[config.ZERO_DURATION_COLUMN].sum())
        caps: dict[str, float] = {}
        for col in affected:
            values = out[col].to_numpy(dtype="float64")
            hi = finite_max(out[col])
            finite = values[np.isfinite(values)]
            lo = float(finite.min()) if finite.size else float("nan")
            if not np.isfinite(hi):
                raise PreprocessingError(
                    "%s is entirely inf, so there is no finite value to cap to. "
                    "Use INF_STRATEGY='nan_median' instead." % col)
            values = np.where(np.isposinf(values), hi, values)
            values = np.where(np.isneginf(values), lo, values)
            out[col] = values
            caps[col] = hi
        info["caps"] = caps
    else:
        out = out.replace([np.inf, -np.inf], np.nan)

    after = count_inf(out)
    report = pd.DataFrame([{
        "column": c,
        "inf_before": before[c],
        "inf_after": after.get(c, 0),
        "action": ("capped at %.6g" % info["caps"][c]) if strategy == "cap"
                  else "inf -> NaN (median imputed below)",
    } for c in affected])

    if verbose:
        show_title("4.1a  Infinite values")
        if len(report):
            show_table(report, note="Strategy **%s**. %s" % (
                strategy,
                ("A binary `%s` column was added, marking the %s zero-duration flows "
                 "that produced the inf." % (config.ZERO_DURATION_COLUMN,
                                             f"{info.get('zero_duration_rows', 0):,}"))
                if strategy == "cap" else
                "The inf values became NaN and are filled by the median imputation below."))
        else:
            show_table(pd.DataFrame([{"column": "(none)", "inf_before": 0,
                                      "inf_after": 0, "action": "nothing to do"}]))
    return out, report, info


def impute_missing(features: pd.DataFrame, verbose: bool = True):
    """Median-impute any NaN.  Reports per column, even when there are none."""
    before = count_nan(features)
    affected = [c for c, n in before.items() if n > 0]
    out = features.copy()
    medians: dict[str, float] = {}
    for col in affected:
        median = float(out[col].median(skipna=True))
        if not np.isfinite(median):
            median = 0.0            # an all-NaN column has no median
        out[col] = out[col].fillna(median)
        medians[col] = median

    after = count_nan(out)
    report = pd.DataFrame([{"column": c, "nan_before": before[c],
                            "nan_after": after.get(c, 0),
                            "filled_with_median": medians[c]} for c in affected])
    if verbose:
        show_title("4.1b  Missing values")
        if len(report):
            show_table(report, note="Filled with each column's own median. "
                                    "Never a blanket zero, which would invent a "
                                    "zero-rate flow.")
        else:
            show_table(pd.DataFrame([{"column": "(none)", "nan_before": 0,
                                      "nan_after": 0, "filled_with_median": "-"}]),
                       note="No NaN anywhere in the features.")
    return out, report, {"columns_imputed": affected, "medians": medians}


def drop_constant_columns(features: pd.DataFrame, verbose: bool = True):
    """Drop zero-variance columns -- they carry no information for any model.

    Which columns these are is MEASURED, never assumed.
    """
    constant = []
    for col in features.columns:
        series = features[col]
        n_unique = series.nunique(dropna=False)
        if n_unique <= 1:
            constant.append({"column": col, "unique_values": int(n_unique),
                             "the_value": series.iloc[0] if len(series) else None})
    report = pd.DataFrame(constant)
    out = features.drop(columns=[c["column"] for c in constant]) if constant else features.copy()

    if verbose:
        show_title("4.1c  Constant (zero-variance) columns")
        if len(report):
            show_table(report, note="Dropped: every row holds the same value, so they "
                                    "cannot separate any class.")
        else:
            show_table(pd.DataFrame([{"column": "(none)", "unique_values": "-",
                                      "the_value": "-"}]),
                       note="No constant column found.")
        show_table(kv_table([("feature columns before", features.shape[1]),
                             ("dropped", len(constant)),
                             ("feature columns after", out.shape[1])]))
    return out, report, [c["column"] for c in constant]


def report_duplicates(features: pd.DataFrame, labels: pd.DataFrame,
                      verbose: bool = True) -> pd.DataFrame:
    """Count duplicate feature rows per label.  NOTHING is dropped."""
    dup_mask = features.duplicated(keep=False)
    joined = labels[[config.LABEL_COLUMN]].copy()
    joined["is_duplicate"] = dup_mask.to_numpy()
    rows = []
    for label, block in joined.groupby(config.LABEL_COLUMN, sort=False):
        rows.append({"Label": str(label), "rows": int(len(block)),
                     "rows_sharing_features": int(block["is_duplicate"].sum()),
                     "share_%": round(100.0 * block["is_duplicate"].sum() / max(1, len(block)), 2)})
    report = pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)

    if verbose:
        show_title("4.1e  Duplicate feature rows (counted, NEVER dropped)")
        show_table(report,
                   note="Kept on purpose: the sequence work in a later step depends on "
                        "the original row order, and thousands of scan and brute-force "
                        "flows genuinely look identical.")
    return report


def apply_label_map(labels: pd.DataFrame, label_map: dict[str, str] | None = None,
                    verbose: bool = True):
    """Report the exact label strings, and apply config.LABEL_MAP only if set."""
    label_map = config.LABEL_MAP if label_map is None else label_map
    out = labels.copy()
    counts = out[config.LABEL_COLUMN].value_counts()

    report = pd.DataFrame({"Label": counts.index.astype(str),
                           "rows": counts.to_numpy()})
    report["renamed_to"] = [label_map.get(str(v), "") for v in counts.index]

    if label_map:
        out[config.LABEL_COLUMN] = out[config.LABEL_COLUMN].map(
            lambda v: label_map.get(str(v), str(v)))

    if verbose:
        show_title("4.1f  Labels")
        show_table(report,
                   note=("`config.LABEL_MAP` is **%s**, so the dataset's own spelling is "
                         "kept -- nothing was renamed." % label_map) if not label_map
                        else "`config.LABEL_MAP` applied: %s" % label_map)
    return out, report, dict(label_map)


def clean_dataset(df: pd.DataFrame, verbose: bool = True):
    """STEP 4.1 -- the whole clean, in order, with a before/after table."""
    if verbose:
        show_title("Step 4.1 -- Clean")
    rows_in, cols_in = df.shape

    if verbose:
        show_title("4.1d  Non-feature columns")
    features, labels, metadata = split_metadata(df, verbose=verbose)
    features, inf_report, inf_info = handle_infinities(features, verbose=verbose)
    features, nan_report, nan_info = impute_missing(features, verbose=verbose)
    features, const_report, dropped = drop_constant_columns(features, verbose=verbose)
    dup_report = report_duplicates(features, labels, verbose=verbose)
    labels, label_report, applied_map = apply_label_map(labels, verbose=verbose)

    if not (len(features) == len(labels) == len(metadata) == rows_in):
        raise PreprocessingError(
            "row counts drifted during cleaning: features %d, labels %d, metadata %d, "
            "input %d" % (len(features), len(labels), len(metadata), rows_in))

    before_after = pd.DataFrame([
        {"stage": "input (merged dataset)", "rows": rows_in, "columns": cols_in,
         "note": "Step 3 output"},
        {"stage": "after removing non-feature columns", "rows": len(features),
         "columns": len(config.METADATA_COLUMNS) and features.shape[1],
         "note": "%d metadata columns + Label kept aside" % len(metadata.columns)},
        {"stage": "after cleaning", "rows": len(features), "columns": features.shape[1],
         "note": "%d inf column(s) handled, %d imputed, %d constant dropped"
                 % (len(inf_report), len(nan_report), len(dropped))},
    ])
    if verbose:
        show_title("4.1  Before / after")
        show_table(before_after,
                   note="Row count is unchanged at every stage, and no row was reordered: "
                        "row i of the features, labels and metadata is the same flow.")

    info = {"rows_in": rows_in, "cols_in": cols_in, "inf": inf_info, "nan": nan_info,
            "constant_dropped": dropped, "label_map_applied": applied_map,
            "reports": {"inf": inf_report, "nan": nan_report, "constant": const_report,
                        "duplicates": dup_report, "labels": label_report,
                        "before_after": before_after}}
    return features, labels, metadata, info


# ===========================================================================
# 4.2  ENCODE
# ===========================================================================
def encode_labels(labels: pd.DataFrame, verbose: bool = True):
    """Label -> Label_encoded (LabelEncoder) + Label_binary (0 benign, 1 attack).

    The original Label string column is KEPT.
    """
    from sklearn.preprocessing import LabelEncoder

    out = labels.copy()
    encoder = LabelEncoder()
    out["Label_encoded"] = encoder.fit_transform(out[config.LABEL_COLUMN].astype(str))

    benign = {str(b) for b in config.BENIGN_LABELS}
    out["Label_binary"] = (~out[config.LABEL_COLUMN].astype(str).isin(benign)).astype("int64")

    mapping = {str(name): int(code) for name, code in
               zip(encoder.classes_, encoder.transform(encoder.classes_))}
    table = pd.DataFrame({
        "Label": list(mapping),
        "Label_encoded": [mapping[k] for k in mapping],
        "Label_binary": [0 if k in benign else 1 for k in mapping],
        "rows": [int((out[config.LABEL_COLUMN].astype(str) == k).sum()) for k in mapping],
    })

    if verbose:
        show_title("4.2a  Label encoding")
        show_table(table,
                   note="`Label_binary` is 0 for %s and 1 for everything else. They stay "
                        "SEPARATE classes in `Label_encoded` -- the binary target only "
                        "groups them." % " and ".join(sorted(benign)))
    return out, encoder, mapping, table


def one_hot_protocol(features: pd.DataFrame, verbose: bool = True):
    """Protocol -> Protocol_ICMP / Protocol_TCP / Protocol_UDP / Protocol_other.

    An unexpected protocol number (IGMP = 2, say) lands in Protocol_other, so a
    row can never silently lose its protocol.
    """
    if "Protocol" not in features.columns:
        if verbose:
            show_note("No `Protocol` column to encode.")
        return features.copy(), pd.DataFrame(), []

    out = features.copy()
    values = out["Protocol"].astype("int64")
    created = []
    for number, name in config.PROTOCOL_ONE_HOT.items():
        out[name] = (values == number).astype("int64")
        created.append(name)
    known = set(config.PROTOCOL_ONE_HOT)
    out[config.PROTOCOL_OTHER_COLUMN] = (~values.isin(known)).astype("int64")
    created.append(config.PROTOCOL_OTHER_COLUMN)
    out = out.drop(columns=["Protocol"])

    table = pd.DataFrame([{"column": name,
                           "protocol": next((n for n, m in config.PROTOCOL_ONE_HOT.items()
                                             if m == name), "anything else"),
                           "rows": int(out[name].sum())} for name in created])
    if verbose:
        show_title("4.2b  Protocol one-hot encoding")
        show_table(table, note="`Protocol` itself is dropped; these four replace it. "
                               "Every row lands in exactly one of them.")
    return out, table, created


def encode_dataset(features: pd.DataFrame, labels: pd.DataFrame, verbose: bool = True):
    """STEP 4.2 -- labels and Protocol."""
    if verbose:
        show_title("Step 4.2 -- Encode")
    labels_out, encoder, mapping, label_table = encode_labels(labels, verbose=verbose)
    features_out, proto_table, proto_cols = one_hot_protocol(features, verbose=verbose)
    info = {"label_encoder": encoder, "label_mapping": mapping,
            "protocol_columns": proto_cols,
            "reports": {"labels": label_table, "protocol": proto_table}}
    return features_out, labels_out, info


# ===========================================================================
# 4.3  SCALE
# ===========================================================================
def numeric_feature_columns(features: pd.DataFrame,
                            protocol_columns: Sequence[str] | None = None) -> list[str]:
    """The columns to scale: everything except the one-hot and binary markers."""
    skip = set(protocol_columns or [])
    skip |= set(config.PROTOCOL_ONE_HOT.values()) | {config.PROTOCOL_OTHER_COLUMN}
    skip.add(config.ZERO_DURATION_COLUMN)
    return [c for c in features.columns if c not in skip]


def scale_features(features: pd.DataFrame, scaler_name: str | None = None,
                   protocol_columns: Sequence[str] | None = None,
                   verbose: bool = True):
    """STEP 4.3 -- scale the numeric features only, and return the fitted scaler."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    scaler_name = (scaler_name or config.SCALER).lower()
    if scaler_name not in ("minmax", "standard"):
        raise PreprocessingError("SCALER must be 'minmax' or 'standard', not %r" % scaler_name)

    if verbose:
        show_title("Step 4.3 -- Scale")

    to_scale = numeric_feature_columns(features, protocol_columns)
    untouched = [c for c in features.columns if c not in to_scale]

    scaler = MinMaxScaler() if scaler_name == "minmax" else StandardScaler()
    scaled = features.copy()
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scaled[to_scale] = scaler.fit_transform(features[to_scale])

    if verbose:
        show_table(kv_table([("scaler", scaler_name),
                             ("columns scaled", len(to_scale)),
                             ("columns left as-is", len(untouched)),
                             ("left as-is", ", ".join(untouched) or "(none)")]),
                   note="One-hot and binary columns are already 0/1, so scaling them "
                        "would only blur their meaning.")
    return scaled, scaler, to_scale


# ===========================================================================
# checks
# ===========================================================================
def run_checks(features_clean: pd.DataFrame, features_scaled: pd.DataFrame,
               labels: pd.DataFrame, metadata: pd.DataFrame, rows_in: int,
               scaled_columns: Sequence[str], scaler_name: str | None = None,
               verbose: bool = True):
    """The [OK]/[FAIL] table: no NaN/inf, minmax range, row count unchanged."""
    scaler_name = (scaler_name or config.SCALER).lower()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    for name, frame in (("cleaned", features_clean), ("scaled", features_scaled)):
        arr = _numeric_block(frame)
        n_nan, n_inf = int(np.isnan(arr).sum()), int(np.isinf(arr).sum())
        add("no NaN in the %s features" % name, n_nan == 0, "%d NaN cells" % n_nan)
        add("no inf in the %s features" % name, n_inf == 0, "%d inf cells" % n_inf)

    if scaler_name == "minmax" and scaled_columns:
        block = features_scaled[list(scaled_columns)].to_numpy(dtype="float64")
        lo, hi = float(np.nanmin(block)), float(np.nanmax(block))
        add("minmax features within [0, 1]", (lo >= -1e-9) and (hi <= 1 + 1e-9),
            "min %.6g, max %.6g" % (lo, hi))
    else:
        add("scaler is %r, so no [0,1] range check" % scaler_name, True, "skipped")

    same = (len(features_clean) == len(features_scaled) == len(labels)
            == len(metadata) == rows_in)
    add("row count unchanged", same,
        "input %d -> clean %d, scaled %d, labels %d, metadata %d"
        % (rows_in, len(features_clean), len(features_scaled), len(labels), len(metadata)))

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_title("4.3  Checks")
        show_table(table)
        if not all_ok:
            show_note("**[FAIL]** -- do not use these outputs until this is resolved.")
    return table, all_ok


# ===========================================================================
# save / load
# ===========================================================================
def save_step4(features_clean: pd.DataFrame, features_scaled: pd.DataFrame,
               labels: pd.DataFrame, metadata: pd.DataFrame,
               artifacts: dict[str, Any], report: pd.DataFrame,
               verbose: bool = True) -> dict[str, Any]:
    """Write every Step 4 output, then RE-READ each file and check the rows."""
    out: dict[str, Any] = {"folder": str(config.STEP4_OUTPUT_DIR), "files": {}}
    folder = prepare_output_dir(config.STEP4_OUTPUT_DIR, "the Step 4 outputs",
                                verbose=verbose)
    if folder is None:
        out["status"] = "skipped_not_windows"
        out["ok"] = False
        return out

    names = config.STEP4_FILES
    features_clean.to_csv(folder / names["features_clean"], index=False)
    features_scaled.to_csv(folder / names["features_scaled"], index=False)
    labels.to_csv(folder / names["labels"], index=False)
    metadata.to_csv(folder / names["metadata"], index=False)
    report.to_csv(folder / names["report"], index=False)

    (folder / names["label_mapping"]).write_text(
        json.dumps(artifacts["label_mapping"], indent=2), encoding="utf-8")
    (folder / names["feature_list"]).write_text(
        json.dumps({"features": list(features_clean.columns),
                    "scaled_columns": list(artifacts["scaled_columns"]),
                    "n_features": int(features_clean.shape[1])}, indent=2),
        encoding="utf-8")

    try:
        import joblib
        joblib.dump({"scaler": artifacts["scaler"],
                     "scaler_name": artifacts["scaler_name"],
                     "label_encoder": artifacts["label_encoder"],
                     "label_mapping": artifacts["label_mapping"],
                     "feature_list": list(features_clean.columns),
                     "scaled_columns": list(artifacts["scaled_columns"]),
                     "inf_strategy": artifacts["inf_strategy"],
                     "constant_dropped": artifacts["constant_dropped"]},
                    folder / names["artifacts"])
        out["files"]["artifacts"] = str(folder / names["artifacts"])
    except ImportError:
        if verbose:
            show_note("joblib is not installed, so the fitted scaler/encoder were not "
                      "saved (`\"%s\" -m pip install joblib`). The CSVs are unaffected."
                      % sys.executable)

    # --- re-read every file and confirm the row counts --------------------
    rows_expected = len(features_clean)
    verify_rows = []
    for key, expected in (("features_clean", rows_expected),
                          ("features_scaled", len(features_scaled)),
                          ("labels", len(labels)),
                          ("metadata", len(metadata))):
        path = folder / names[key]
        on_disk = pd.read_csv(path, low_memory=False)
        verify_rows.append({
            "file": names[key], "exists": path.exists(),
            "size_MB": round(path.stat().st_size / 1024 / 1024, 3),
            "rows_in_memory": expected, "rows_on_disk": len(on_disk),
            "columns_on_disk": on_disk.shape[1],
            "result": "[OK]" if len(on_disk) == expected else "[FAIL]",
        })
        out["files"][key] = str(path)
    for key in ("label_mapping", "feature_list", "report"):
        path = folder / names[key]
        verify_rows.append({"file": names[key], "exists": path.exists(),
                            "size_MB": round(path.stat().st_size / 1024 / 1024, 3),
                            "rows_in_memory": "-", "rows_on_disk": "-",
                            "columns_on_disk": "-",
                            "result": "[OK]" if path.exists() else "[FAIL]"})
        out["files"][key] = str(path)

    verification = pd.DataFrame(verify_rows)
    out["verification"] = verification
    out["ok"] = bool((verification["result"] == "[OK]").all())
    out["status"] = "written"

    if verbose:
        show_title("Step 4 -- files written, then re-read from disk")
        show_table(verification,
                   note="Every CSV was read back and its row count compared with the "
                        "frame in memory. Folder: %s" % folder)
    return out


def step4_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP4_OUTPUT_DIR)
    if folder is None:
        return False
    return all((folder / config.STEP4_FILES[k]).exists()
               for k in ("features_clean", "features_scaled", "labels", "metadata"))


def load_step4(verbose: bool = True) -> dict[str, Any]:
    """Read the Step 4 outputs back from disk."""
    folder = config.usable_output_dir(config.STEP4_OUTPUT_DIR)
    if folder is None:
        raise PreprocessingError(config.why_unusable(config.STEP4_OUTPUT_DIR))
    names = config.STEP4_FILES
    loaded = {
        "features_clean": pd.read_csv(folder / names["features_clean"], low_memory=False),
        "features_scaled": pd.read_csv(folder / names["features_scaled"], low_memory=False),
        "labels": pd.read_csv(folder / names["labels"], low_memory=False),
        "metadata": pd.read_csv(folder / names["metadata"], low_memory=False),
    }
    mapping_path = folder / names["label_mapping"]
    if mapping_path.exists():
        loaded["label_mapping"] = json.loads(mapping_path.read_text(encoding="utf-8"))
    if verbose:
        show_title("Step 4 -- existing outputs loaded from disk")
        show_table(pd.DataFrame([{"file": names[k], "rows": len(v),
                                  "columns": v.shape[1]}
                                 for k, v in loaded.items()
                                 if isinstance(v, pd.DataFrame)]),
                   note="`config.OVERWRITE_STEP4` is False and the outputs already "
                        "exist, so they were loaded instead of recomputed.")
    return loaded


# ===========================================================================
# the whole step
# ===========================================================================
def build_report(clean_info: dict[str, Any], encode_info: dict[str, Any],
                 checks: pd.DataFrame, features_clean: pd.DataFrame,
                 scaler_name: str) -> pd.DataFrame:
    """One tidy table summarising every action Step 4 took."""
    rows: list[dict[str, Any]] = [
        {"stage": "4.1 clean", "action": "rows in", "detail": clean_info["rows_in"]},
        {"stage": "4.1 clean", "action": "columns in", "detail": clean_info["cols_in"]},
        {"stage": "4.1a inf", "action": "strategy", "detail": clean_info["inf"]["strategy"]},
        {"stage": "4.1a inf", "action": "columns affected",
         "detail": ", ".join(clean_info["inf"]["columns_affected"]) or "(none)"},
        {"stage": "4.1a inf", "action": "inf values handled",
         "detail": clean_info["inf"]["inf_before_total"]},
        {"stage": "4.1a inf", "action": "zero_duration rows",
         "detail": clean_info["inf"].get("zero_duration_rows", "-")},
        {"stage": "4.1b nan", "action": "columns imputed",
         "detail": ", ".join(clean_info["nan"]["columns_imputed"]) or "(none)"},
        {"stage": "4.1c constant", "action": "columns dropped",
         "detail": ", ".join(clean_info["constant_dropped"]) or "(none)"},
        {"stage": "4.1d metadata", "action": "columns kept aside",
         "detail": ", ".join(config.METADATA_COLUMNS)},
        {"stage": "4.1e duplicates", "action": "policy", "detail": "counted, never dropped"},
        {"stage": "4.1f labels", "action": "LABEL_MAP applied",
         "detail": clean_info["label_map_applied"] or "(empty - nothing renamed)"},
        {"stage": "4.2 encode", "action": "label classes",
         "detail": len(encode_info["label_mapping"])},
        {"stage": "4.2 encode", "action": "protocol columns",
         "detail": ", ".join(encode_info["protocol_columns"]) or "(none)"},
        {"stage": "4.3 scale", "action": "scaler", "detail": scaler_name},
        {"stage": "4.3 scale", "action": "final feature count",
         "detail": features_clean.shape[1]},
    ]
    for _, row in checks.iterrows():
        rows.append({"stage": "4.3 checks", "action": row["check"],
                     "detail": "%s %s" % (row["result"], row["detail"])})
    return pd.DataFrame(rows)


def run_step4(df: pd.DataFrame | None = None, overwrite: bool | None = None,
              verbose: bool = True) -> dict[str, Any]:
    """STEP 4 end to end: clean -> encode -> scale -> save -> verify.

    With OVERWRITE_STEP4 False and the outputs already present, they are loaded
    instead of recomputed.
    """
    overwrite = config.OVERWRITE_STEP4 if overwrite is None else overwrite
    if not overwrite and step4_outputs_exist():
        loaded = load_step4(verbose=verbose)
        loaded["status"] = "loaded_existing"
        # the combined file is rebuilt from the SAVED files, so it always
        # matches them even when Step 4 itself was skipped
        loaded["combined"] = build_combined_from_saved(verbose=verbose)
        return loaded

    if df is None:
        df = dl.load_merged(verbose=verbose)

    features, labels, metadata, clean_info = clean_dataset(df, verbose=verbose)
    features, labels, encode_info = encode_dataset(features, labels, verbose=verbose)
    scaled, scaler, scaled_columns = scale_features(
        features, protocol_columns=encode_info["protocol_columns"], verbose=verbose)
    checks, all_ok = run_checks(features, scaled, labels, metadata,
                                clean_info["rows_in"], scaled_columns, verbose=verbose)

    report = build_report(clean_info, encode_info, checks, features,
                          (config.SCALER or "minmax").lower())
    artifacts = {"scaler": scaler, "scaler_name": (config.SCALER or "minmax").lower(),
                 "label_encoder": encode_info["label_encoder"],
                 "label_mapping": encode_info["label_mapping"],
                 "scaled_columns": scaled_columns,
                 "inf_strategy": clean_info["inf"]["strategy"],
                 "constant_dropped": clean_info["constant_dropped"]}
    saved = save_step4(features, scaled, labels, metadata, artifacts, report,
                       verbose=verbose)

    if verbose:
        show_title("Step 4 -- summary")
        show_table(kv_table([
            ("rows", f"{len(features):,}"),
            ("final feature count", features.shape[1]),
            ("constant columns dropped",
             ", ".join(clean_info["constant_dropped"]) or "(none)"),
            ("columns added", ", ".join(
                ([config.ZERO_DURATION_COLUMN]
                 if config.ZERO_DURATION_COLUMN in features.columns else [])
                + list(encode_info["protocol_columns"]))),
            ("label classes", len(encode_info["label_mapping"])),
            ("scaler", artifacts["scaler_name"]),
            ("all checks passed", all_ok),
            ("output folder", saved.get("folder")),
        ]))

    combined = save_combined(
        scaled if config.COMBINED_USE_SCALED else features, labels, metadata,
        feature_order=list(features.columns), verbose=verbose)

    return {"combined": combined,
            "status": saved.get("status", "written"), "features_clean": features,
            "features_scaled": scaled, "labels": labels, "metadata": metadata,
            "checks": checks, "report": report, "artifacts": artifacts,
            "saved": saved, "ok": bool(all_ok and saved.get("ok")
                                       and combined.get("ok")),
            "clean_info": clean_info, "encode_info": encode_info}


# ===========================================================================
# 4.4  THE COMBINED FILE -- preprocessing.csv
# ===========================================================================
# One CSV holding the final preprocessed dataset, laid out for READING and for
# the sequence / attack-chain work:
#
#   Src IP | Dst IP | Src Port | Dst Port | Protocol | Timestamp
#          | <scaled features, without the four Protocol_* columns> | Label
#
# It is built by COLUMN-JOINING the frames Step 4 already produced -- nothing is
# recomputed and nothing is re-scaled -- so it can never drift from
# features_scaled.csv / labels.csv.  The other eight Step 4 outputs are
# untouched: they keep the numeric Protocol_* one-hot columns and all three
# label columns, because that is what a model is trained on.
LABEL_OUTPUT_COLUMNS = ["Label", "Label_encoded", "Label_binary"]


def protocol_text_from_one_hot(features: pd.DataFrame, verbose: bool = True):
    """Protocol_ICMP / _TCP / _UDP / _other  ->  one "ICMP"/"TCP"/"UDP"/"OTHER".

    Exactly one of the four must be 1 in every row.  A row with none set, or
    with two set, is a real defect (the one-hot encoding is broken), so it is
    reported as [FAIL] with the row count rather than silently filled in.

    Returns (series, table, ok, bad).  `bad` holds the offending row positions.
    """
    mapping = dict(config.PROTOCOL_TEXT_FROM_ONE_HOT)
    absent = [c for c in mapping if c not in features.columns]
    if absent:
        raise PreprocessingError(
            "cannot build the readable `%s` column: the feature frame has no %s. "
            "Re-run Step 4 with OVERWRITE_STEP4 = True so the Protocol one-hot "
            "columns are rebuilt." % (config.COMBINED_PROTOCOL_COLUMN, absent))

    columns = list(mapping)
    block = features[columns].to_numpy(dtype="int64")
    hits = block.sum(axis=1)
    bad = np.flatnonzero(hits != 1)

    # idxmax picks the single column that is 1.  For a bad row it would pick the
    # first column, so those rows are blanked instead of being given a wrong
    # protocol -- and the check below fails loudly.
    picked = np.asarray(columns, dtype=object)[block.argmax(axis=1)]
    series = pd.Series(picked, index=features.index,
                       name=config.COMBINED_PROTOCOL_COLUMN).map(mapping)
    if bad.size:
        series.iloc[bad] = pd.NA

    rows = []
    for column, text in mapping.items():
        rows.append({"one-hot column": column, "Protocol": text,
                     "one-hot sum": int(features[column].sum()),
                     "text rows": int((series == text).sum())})
    table = pd.DataFrame(rows)
    ok = bool(bad.size == 0)

    if verbose:
        show_table(table, note="The four one-hot columns collapse into one readable "
                               "`%s` column for preprocessing.csv. `one-hot sum` and "
                               "`text rows` must match exactly."
                               % config.COMBINED_PROTOCOL_COLUMN)
        if ok:
            show_note("[OK] every row has exactly one of the four one-hot columns set.")
        else:
            none_set = int((hits == 0).sum())
            many_set = int((hits > 1).sum())
            show_note("**[FAIL]** %d rows do not have exactly one Protocol one-hot "
                      "column set: %d row(s) with none set and %d row(s) with two or "
                      "more. First offending row positions: %s"
                      % (bad.size, none_set, many_set, list(bad[:10])))
    return series, table, ok, bad


def build_combined(features: pd.DataFrame, labels: pd.DataFrame,
                   metadata: pd.DataFrame | None = None,
                   id_columns: Sequence[str] | None = None,
                   feature_order: Sequence[str] | None = None,
                   allow_missing_identifiers: bool | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Join identifiers, features and the Label side by side, in row order.

    The row counts must match exactly.  They are checked BEFORE the join,
    because pd.concat would otherwise pad the shorter frame with NaN and the
    mismatch would only surface much later as mysterious missing values.

    Column order:
        Src IP, Dst IP, Src Port, Dst Port, Protocol, Timestamp,
        <features minus the four Protocol_* one-hot columns>, Label

    Protocol is DERIVED from the Protocol_* one-hot feature columns, not copied
    from metadata, so this works with a metadata.csv written before Protocol was
    added to METADATA_COLUMNS.
    """
    id_columns = (list(config.COMBINED_ID_COLUMNS) if id_columns is None
                  else list(id_columns))
    # The official dataset may carry none of the identifier columns -- it is a
    # flow feature table, not a capture.  There, an identifier that is absent
    # is a FACT about the dataset: it is reported and skipped.  In own mode the
    # same absence means Step 4's metadata was written wrong, which must still
    # raise, so the default follows DATASET_SOURCE and "own" is unchanged.
    if allow_missing_identifiers is None:
        allow_missing_identifiers = (
            getattr(config, "DATASET_SOURCE", "own") == "official")
    proto_name = config.COMBINED_PROTOCOL_COLUMN
    wants_protocol = proto_name in id_columns
    # The readable Protocol column is DERIVED from the Protocol_* one-hot
    # features, so it can only be built when those exist.  The official dataset
    # may have no Protocol column at all, and then there is nothing to derive
    # from: that is a fact about the dataset, reported and skipped.  In own mode
    # the same absence means Step 4's one-hot encoding is broken, so the
    # existing [FAIL] inside protocol_text_from_one_hot() still fires.
    if wants_protocol and allow_missing_identifiers:
        one_hot = list(config.PROTOCOL_TEXT_FROM_ONE_HOT)
        missing_one_hot = [c for c in one_hot if c not in features.columns]
        if missing_one_hot:
            if verbose:
                show_note("**No readable `%s` column in `%s`:** the %s dataset "
                          "has no Protocol column, so the %s one-hot features "
                          "it would be derived from do not exist either. The "
                          "flows are unchanged; there is simply no protocol to "
                          "report."
                          % (proto_name, config.COMBINED_CSV.name,
                             getattr(config, "DATASET_SOURCE", "own"),
                             ", ".join(one_hot)))
            wants_protocol = False
            id_columns = [c for c in id_columns if c != proto_name]
    from_metadata = [c for c in id_columns if c != proto_name]

    parts = {"features": len(features), "labels": len(labels)}
    if from_metadata:
        if metadata is None:
            raise PreprocessingError(
                "COMBINED_ID_COLUMNS asks for %s but no metadata frame was given"
                % from_metadata)
        absent = [c for c in from_metadata if c not in metadata.columns]
        if absent and allow_missing_identifiers:
            if verbose:
                show_note("**Identifier columns unavailable:** %s. The %s "
                          "dataset does not carry them, so `%s` is written "
                          "without those columns, and every later part that "
                          "needs them is skipped with a note rather than "
                          "given an invented value."
                          % (", ".join(absent),
                             getattr(config, "DATASET_SOURCE", "own"),
                             config.COMBINED_CSV.name))
            from_metadata = [c for c in from_metadata if c not in absent]
            id_columns = [c for c in id_columns if c not in absent]
            absent = []
        if absent:
            raise PreprocessingError(
                "the metadata frame has no %s, so those identifier columns cannot be "
                "added to preprocessing.csv. Either re-run Step 4 with "
                "OVERWRITE_STEP4 = True, or let build_combined_from_saved() fill them "
                "in from %s." % (absent, config.MERGED_CSV))
        if from_metadata:
            parts["metadata"] = len(metadata)
    if len(set(parts.values())) != 1:
        raise PreprocessingError(
            "[FAIL] the parts have different row counts and must not be joined: %s. "
            "Joining them would pad the shorter one with NaN." % parts)

    label_columns = list(config.COMBINED_LABEL_COLUMNS)
    missing_labels = [c for c in label_columns if c not in labels.columns]
    if missing_labels:
        raise PreprocessingError("labels is missing %s" % missing_labels)

    order = list(feature_order) if feature_order is not None else list(features.columns)
    unknown = [c for c in order if c not in features.columns]
    if unknown:
        raise PreprocessingError(
            "feature_list names columns that are not in the features frame: %s" % unknown)
    # The four one-hot columns come out only because the single readable Protocol
    # replaces them.  Without that column they stay, so protocol information can
    # never vanish from the file unnoticed.
    drop = set(config.COMBINED_DROP_FEATURE_COLUMNS) if wants_protocol else set()
    dropped_one_hot = [c for c in order if c in drop]
    order = [c for c in order if c not in drop]

    protocol = None
    protocol_ok = True
    if wants_protocol:
        protocol, _, protocol_ok, bad = protocol_text_from_one_hot(features,
                                                                   verbose=verbose)
        if not protocol_ok:
            raise PreprocessingError(
                "[FAIL] %d of %d rows do not have exactly one Protocol one-hot column "
                "set, so the readable `%s` column cannot be built. First offending row "
                "positions: %s" % (bad.size, len(features), proto_name, list(bad[:10])))

    # reset_index so the join is POSITIONAL: row i of each part is the same flow,
    # and a stray index would otherwise realign the rows silently.
    ids = pd.DataFrame(index=range(len(features)))
    if from_metadata:
        ids = metadata[from_metadata].reset_index(drop=True).copy()
        # ports stay integers, never floats or scaled values
        for col in config.COMBINED_INT_COLUMNS:
            if col in ids.columns:
                ids[col] = ids[col].astype("int64")
    if protocol is not None:
        ids[proto_name] = protocol.reset_index(drop=True).astype("string")
    if id_columns:
        ids = ids[id_columns]

    pieces = []
    if id_columns:
        pieces.append(ids)
    pieces.append(features[order].reset_index(drop=True))
    pieces.append(labels[label_columns].reset_index(drop=True))

    combined = pd.concat(pieces, axis=1)

    if verbose:
        show_table(kv_table([
            ("rows", f"{len(combined):,}"),
            ("identifier columns", len(id_columns)),
            ("feature columns", len(order)),
            ("Protocol_* one-hot columns left out", len(dropped_one_hot)),
            ("label columns", ", ".join(label_columns)),
            ("total columns", combined.shape[1]),
        ]), note="Built by column-joining the frames Step 4 already produced -- nothing "
                 "recomputed, nothing re-scaled. The identifiers keep their ORIGINAL "
                 "values and are joined by row POSITION, never by merging on values. "
                 "`Protocol` is derived from the one-hot columns, which are themselves "
                 "left out here but KEPT in features_scaled.csv.")
    return combined


def verify_combined(combined: pd.DataFrame, features: pd.DataFrame,
                    labels: pd.DataFrame, metadata: pd.DataFrame | None = None,
                    expected_rows: int | None = None,
                    id_columns: Sequence[str] | None = None,
                    feature_order: Sequence[str] | None = None,
                    path: Path | None = None, verbose: bool = True):
    """Re-read the combined file and check it against its parts.  [OK]/[FAIL].

    `feature_order` is the order the builder wrote, which is `feature_list.json`
    when Step 4 has one.  It is taken rather than assumed so that the column
    check is about the feature list itself and not about the two orders
    happening to agree.

    A dataset with NO identifier columns is a normal case, not a failure: with
    `COMBINED_ID_COLUMNS` empty the file is the features followed by `Label`,
    the column count is `len(feature_list) + 1`, and the identifier checks are
    reported as `[SKIP]` rather than failed.
    """
    id_columns = (list(config.COMBINED_ID_COLUMNS) if id_columns is None
                  else list(id_columns))
    proto_name = config.COMBINED_PROTOCOL_COLUMN
    from_metadata = [c for c in id_columns if c != proto_name]
    label_columns = list(config.COMBINED_LABEL_COLUMNS)
    path = Path(path) if path is not None else config.COMBINED_CSV
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    def skip(name: str, detail: str) -> None:
        """A check this dataset cannot have, reported rather than failed."""
        checks.append({"check": name, "result": "[SKIP]", "detail": detail})

    exists = path.exists()
    add("file exists", exists, str(path))
    if not exists:
        table = pd.DataFrame(checks)
        if verbose:
            show_table(table)
        return table, False, None

    size_mb = round(path.stat().st_size / 1024 / 1024, 3)
    add("size on disk", size_mb > 0, "%.3f MB" % size_mb)

    on_disk = pd.read_csv(path, low_memory=False)

    rows_expected = expected_rows if expected_rows is not None else len(features)
    add("rows match the other Step 4 files", len(on_disk) == rows_expected,
        "%s on disk vs %s expected" % (f"{len(on_disk):,}", f"{rows_expected:,}"))

    # the feature block: everything Step 4 produced except the four one-hot
    # columns, which the single readable Protocol replaces
    drop = (set(config.COMBINED_DROP_FEATURE_COLUMNS)
            if proto_name in id_columns else set())
    written_order = (list(feature_order) if feature_order is not None
                     else list(features.columns))
    source_features = [c for c in written_order if c not in drop]
    n_ids = len(id_columns)
    expected_order = id_columns + source_features + label_columns
    cols_expected = len(expected_order)
    if n_ids:
        count_label = ("columns == %d identifiers + %d features + %d label"
                       % (n_ids, len(source_features), len(label_columns)))
        order_label = "columns are in the expected order"
        order_detail = ("identifiers, then features, then %s"
                        % ", ".join(label_columns))
    else:
        # No identifier columns: the official MSCAD dataset has none, so the
        # file is the feature list followed by Label and nothing else.
        count_label = ("columns == %d features + %d label"
                       % (len(source_features), len(label_columns)))
        order_label = "columns are in the expected feature-list order"
        order_detail = ("the %d feature_list columns, then %s"
                        % (len(source_features), ", ".join(label_columns)))
    add(count_label, on_disk.shape[1] == cols_expected,
        "%d on disk vs %d expected" % (on_disk.shape[1], cols_expected))
    add(order_label, list(on_disk.columns) == expected_order,
        order_detail if list(on_disk.columns) == expected_order
        else "got %s ... %s" % (list(on_disk.columns[:7]), list(on_disk.columns[-3:])))

    present_one_hot = [c for c in config.COMBINED_DROP_FEATURE_COLUMNS
                       if c in drop and c in on_disk.columns]
    add("no Protocol_* one-hot columns", not present_one_hot,
        "they stay in features_scaled.csv only" if not present_one_hot
        else "found %s" % present_one_hot)
    extra_labels = [c for c in LABEL_OUTPUT_COLUMNS
                    if c not in label_columns and c in on_disk.columns]
    add("no Label_encoded / Label_binary columns", not extra_labels,
        "they stay in labels.csv only" if not extra_labels
        else "found %s" % extra_labels)

    if id_columns:
        add("the first %d columns are the identifiers, in order" % n_ids,
            list(on_disk.columns[:n_ids]) == id_columns,
            "got %s" % list(on_disk.columns[:n_ids]))
        if metadata is not None and len(on_disk) == len(metadata):
            compared = [c for c in from_metadata if c in metadata.columns]
            same_ids = all(
                on_disk[c].astype(str).tolist() == metadata[c].astype(str).tolist()
                for c in compared)
            add("identifier values match metadata row by row", same_ids,
                "%d columns over %s rows" % (len(compared), f"{len(on_disk):,}"))
        else:
            add("identifier values match metadata row by row", False,
                "no metadata frame, or the row counts differ")
        int_ok = all(pd.api.types.is_integer_dtype(on_disk[c])
                     for c in config.COMBINED_INT_COLUMNS if c in on_disk.columns)
        add("ports are integers", int_ok,
            ", ".join("%s=%s" % (c, on_disk[c].dtype)
                      for c in config.COMBINED_INT_COLUMNS if c in on_disk.columns))
        id_nan = int(on_disk[[c for c in id_columns if c in on_disk.columns]]
                     .isna().to_numpy().sum())
        add("no NaN in the identifier columns", id_nan == 0, "%d NaN cells" % id_nan)
        add("identifiers are NOT in feature_list.json",
            not set(id_columns) & set(source_features),
            "they are excluded from the model features")
    else:
        skip("identifier values match metadata row by row",
             "the official dataset has no identifier columns")

    if proto_name in on_disk.columns and proto_name in id_columns:
        seen = sorted(set(on_disk[proto_name].dropna().astype(str)))
        allowed = set(config.PROTOCOL_TEXT_VALUES)
        add("Protocol holds only %s" % "/".join(config.PROTOCOL_TEXT_VALUES),
            set(seen) <= allowed,
            "found %s" % seen if set(seen) <= allowed
            else "unexpected values: %s" % sorted(set(seen) - allowed))
        rows = []
        mismatch = []
        for column, text in config.PROTOCOL_TEXT_FROM_ONE_HOT.items():
            n_text = int((on_disk[proto_name].astype(str) == text).sum())
            n_hot = (int(features[column].sum()) if column in features.columns
                     else None)
            rows.append("%s %s" % (text, f"{n_text:,}"))
            if n_hot is not None and n_hot != n_text:
                mismatch.append("%s: %d text vs %d one-hot" % (text, n_text, n_hot))
        add("Protocol counts match the one-hot sums", not mismatch,
            ", ".join(rows) if not mismatch else "; ".join(mismatch))

    feature_cols = [c for c in source_features if c in on_disk.columns]
    block = on_disk[feature_cols].to_numpy(dtype="float64")
    n_nan, n_inf = int(np.isnan(block).sum()), int(np.isinf(block).sum())
    add("no NaN in the feature columns", n_nan == 0, "%d NaN cells" % n_nan)
    add("no inf in the feature columns", n_inf == 0, "%d inf cells" % n_inf)

    have_labels = all(c in on_disk.columns for c in label_columns)
    add("label column present", have_labels, ", ".join(label_columns))
    if have_labels and len(on_disk) == len(labels):
        same = all(
            on_disk[c].astype(str).tolist() == labels[c].astype(str).tolist()
            for c in label_columns)
        add("labels match labels.csv row by row", same,
            "all %s rows" % f"{len(on_disk):,}" if same else "a row differs")
    else:
        add("labels match labels.csv row by row", False, "row counts differ")

    if len(on_disk) == len(features) and feature_cols:
        same_features = np.allclose(
            on_disk[feature_cols].to_numpy(dtype="float64"),
            features[feature_cols].to_numpy(dtype="float64"),
            rtol=1e-9, atol=1e-12, equal_nan=True)
        add("feature values match the source frame", same_features,
            "%d columns compared" % len(feature_cols))
    else:
        add("feature values match the source frame", False, "shapes differ")

    table = pd.DataFrame(checks)
    # A [SKIP] row is a check this dataset cannot have, not a check that failed:
    # only a [FAIL] makes the file unusable.
    all_ok = bool(not (table["result"] == "[FAIL]").any())
    if verbose:
        show_table(table, note="Read back from **%s**" % path)
        if not all_ok:
            show_note("**[FAIL]** -- preprocessing.csv does not agree with the other "
                      "Step 4 outputs; do not use it.")
    return table, all_ok, on_disk


def save_combined(features: pd.DataFrame, labels: pd.DataFrame,
                  metadata: pd.DataFrame | None = None,
                  use_scaled: bool | None = None,
                  id_columns: Sequence[str] | None = None,
                  feature_order: Sequence[str] | None = None,
                  verbose: bool = True) -> dict[str, Any]:
    """Write preprocessing.csv and verify it from disk."""
    out: dict[str, Any] = {"path": str(config.COMBINED_CSV),
                           "use_scaled": (config.COMBINED_USE_SCALED
                                          if use_scaled is None else use_scaled)}
    folder = prepare_output_dir(config.STEP4_OUTPUT_DIR,
                                "the combined preprocessing.csv", verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    if verbose:
        show_title("4.4  Final preprocessed dataset (preprocessing.csv)")

    combined = build_combined(features, labels, metadata, id_columns=id_columns,
                              feature_order=feature_order, verbose=verbose)
    path = folder / config.STEP4_FILES["combined"]
    combined.to_csv(path, index=False)

    table, ok, on_disk = verify_combined(
        combined, features, labels, metadata, expected_rows=len(features),
        id_columns=id_columns, feature_order=feature_order, path=path,
        verbose=verbose)

    out.update({"status": "written", "ok": ok, "combined": combined,
                "on_disk": on_disk, "verification": table,
                "rows": len(combined), "columns": combined.shape[1],
                "size_mb": round(path.stat().st_size / 1024 / 1024, 3)})
    if verbose:
        show_table(kv_table([("absolute path", str(path)),
                             ("size (MB)", out["size_mb"]),
                             ("rows", f"{out['rows']:,}"),
                             ("columns", out["columns"]),
                             ("identifier columns",
                              ", ".join(id_columns if id_columns is not None
                                        else config.COMBINED_ID_COLUMNS) or "(none)"),
                             ("features from", "features_scaled" if out["use_scaled"]
                              else "features_clean"),
                             ("result", "[OK] written and verified" if ok
                              else "[FAIL] see the table above")]))
    return out


def _ids_from_merged(metadata: pd.DataFrame, wanted: Sequence[str],
                     verbose: bool = True) -> pd.DataFrame:
    """Fill identifier columns a saved metadata.csv is missing from Step 3.

    MSCAD_merged.csv has every identifier, and metadata.csv was written from it
    in the SAME row order with no row dropped, so the two line up row by row.
    When metadata still carries Source_Capture and Orig_Row those two are used
    as an exact key and the alignment is checked rather than assumed; otherwise
    the join is by row position after a row-count check.
    """
    wanted = list(wanted)
    merged_path = config.MERGED_CSV
    if not merged_path.exists():
        raise PreprocessingError(
            "the saved metadata.csv has no %s and the Step 3 merged file is not on "
            "disk either (%s), so the identifier columns cannot be recovered.\n"
            "Set OVERWRITE_STEP4 = True in config.py and re-run Step 4."
            % (wanted, merged_path))

    key = ["Source_Capture", "Orig_Row"]
    have_key = all(c in metadata.columns for c in key)
    usecols = sorted(set(wanted) | (set(key) if have_key else set()))
    merged = pd.read_csv(merged_path, usecols=lambda c: c in set(usecols),
                         low_memory=False)

    still_absent = [c for c in wanted if c not in merged.columns]
    if still_absent:
        raise PreprocessingError(
            "%s is missing %s as well, so the identifier columns cannot be "
            "recovered.\nSet OVERWRITE_STEP4 = True in config.py and re-run Step 4."
            % (merged_path, still_absent))

    if have_key and all(c in merged.columns for c in key):
        merged = merged.set_index(key)
        wanted_index = pd.MultiIndex.from_frame(metadata[key])
        unknown = (~wanted_index.isin(merged.index)).sum()
        if unknown:
            raise PreprocessingError(
                "%d metadata rows have a (Source_Capture, Orig_Row) key that is not in "
                "%s, so the identifiers cannot be matched.\nSet OVERWRITE_STEP4 = True "
                "in config.py and re-run Step 4." % (int(unknown), merged_path))
        out = merged.loc[wanted_index, wanted].reset_index(drop=True)
        how = "matched on (Source_Capture, Orig_Row)"
    else:
        if len(merged) != len(metadata):
            raise PreprocessingError(
                "%s has %s rows but the saved metadata.csv has %s, and metadata has no "
                "(Source_Capture, Orig_Row) key to match on, so the identifiers cannot "
                "be recovered safely.\nSet OVERWRITE_STEP4 = True in config.py and "
                "re-run Step 4."
                % (merged_path, f"{len(merged):,}", f"{len(metadata):,}"))
        out = merged[wanted].reset_index(drop=True)
        how = "joined by row position (metadata has no Source_Capture/Orig_Row key)"

    if verbose:
        show_note("`%s` is missing from the saved **metadata.csv** -- read from "
                  "**%s** instead, %s."
                  % ("`, `".join(wanted), merged_path.name, how))
    return out


def build_combined_from_saved(verbose: bool = True) -> dict[str, Any]:
    """Build preprocessing.csv from the SAVED Step 4 files.

    Used when Step 4 itself was skipped (OVERWRITE_STEP4 is False and the
    outputs already exist), so the combined file still matches them exactly.

    An OLD metadata.csv -- one written before Protocol joined METADATA_COLUMNS --
    is fine: Protocol in preprocessing.csv is derived from the Protocol_* one-hot
    FEATURE columns, not read from metadata.  Any other identifier column that is
    genuinely missing is recovered from the Step 3 merged file, and where each
    column came from is printed.
    """
    folder = config.usable_output_dir(config.STEP4_OUTPUT_DIR)
    if folder is None:
        if verbose:
            print("  [SKIP]  %s" % config.why_unusable(config.STEP4_OUTPUT_DIR))
        return {"status": "skipped_not_windows", "ok": False}

    names = config.STEP4_FILES
    use_scaled = config.COMBINED_USE_SCALED
    features_name = names["features_scaled"] if use_scaled else names["features_clean"]
    features = pd.read_csv(folder / features_name, low_memory=False)
    labels = pd.read_csv(folder / names["labels"], low_memory=False)

    proto_name = config.COMBINED_PROTOCOL_COLUMN
    from_metadata = [c for c in config.COMBINED_ID_COLUMNS if c != proto_name]
    metadata = None
    sources: list[dict[str, str]] = []
    if proto_name in config.COMBINED_ID_COLUMNS:
        sources.append({"column": proto_name,
                        "read from": "%s (derived from the Protocol_* one-hot "
                                     "columns)" % features_name})
    if from_metadata:
        metadata = pd.read_csv(folder / names["metadata"], low_memory=False)
        absent = [c for c in from_metadata if c not in metadata.columns]
        # An identifier the ACTIVE dataset simply does not have cannot be
        # recovered from the Step 3 file either, because it is not in that file
        # to recover.  In official mode it is dropped here, named in the table
        # below; in own mode nothing changes and the recovery path runs.
        if absent and getattr(config, "DATASET_SOURCE", "own") == "official":
            merged_columns: set[str] = set()
            merged_path = config.MERGED_CSV
            if merged_path.exists():
                merged_columns = set(pd.read_csv(merged_path, nrows=0).columns)
            unavailable = [c for c in absent if c not in merged_columns]
            if unavailable:
                if verbose:
                    show_note("**Identifier columns unavailable:** %s -- "
                              "neither `metadata.csv` nor the Step 3 merged "
                              "file carries them, because the official "
                              "dataset does not. They are left out of `%s`."
                              % (", ".join(unavailable),
                                 config.COMBINED_CSV.name))
                from_metadata = [c for c in from_metadata
                                 if c not in unavailable]
                absent = [c for c in absent if c not in unavailable]
        for column in from_metadata:
            sources.append({"column": column,
                            "read from": (names["metadata"] if column not in absent
                                          else config.MERGED_CSV.name)})
        if absent:
            recovered = _ids_from_merged(metadata, absent, verbose=verbose)
            if len(recovered) != len(metadata):
                raise PreprocessingError(
                    "recovered %s identifier rows but metadata has %s"
                    % (f"{len(recovered):,}", f"{len(metadata):,}"))
            metadata = metadata.reset_index(drop=True)
            for column in absent:
                metadata[column] = recovered[column].to_numpy()

    if verbose and sources:
        show_table(pd.DataFrame(sources),
                   note="Where each identifier column came from. Nothing is matched on "
                        "values: metadata.csv is row-aligned with the Step 4 outputs, "
                        "and a recovered column is keyed on (Source_Capture, Orig_Row) "
                        "or row position.")

    order = None
    feature_list_path = folder / names["feature_list"]
    if feature_list_path.exists():
        order = json.loads(feature_list_path.read_text(encoding="utf-8")).get("features")
        order = [c for c in (order or []) if c in features.columns] or None

    return save_combined(features, labels, metadata, use_scaled=use_scaled,
                         feature_order=order, verbose=verbose)


def get_feature_matrix(df: pd.DataFrame,
                       feature_list: Sequence[str] | None = None) -> pd.DataFrame:
    """Only the model features: no identifiers, no labels.

    preprocessing.csv deliberately carries Src IP, Dst IP, the ports, Protocol
    and Timestamp so a row can be traced back to its flow and so the sequence /
    attack-state-graph work has what it needs.  They must NEVER be fed to a
    model: the attacker IPs decided the labels in Step 1, so training on them is
    label leakage and would score near-perfectly while learning nothing.

    Pass the SCALED feature frame, which still has the numeric Protocol_* one-hot
    columns a model needs:

        import pandas as pd, config
        from src.mscad import preprocessing
        features = pd.read_csv(config.STEP4_OUTPUT_DIR / "features_scaled.csv")
        labels = pd.read_csv(config.STEP4_OUTPUT_DIR / "labels.csv")
        X = preprocessing.get_feature_matrix(features)
        y = labels["Label_encoded"]

    preprocessing.csv is the readable view, not the training matrix: it replaces
    the four Protocol_* columns with one text column and keeps only `Label`, so
    it is NOT a valid input here.
    """
    if feature_list is None:
        folder = config.usable_output_dir(config.STEP4_OUTPUT_DIR)
        path = (folder / config.STEP4_FILES["feature_list"]) if folder else None
        if path is not None and path.exists():
            feature_list = json.loads(path.read_text(encoding="utf-8"))["features"]
        else:
            # fall back to "everything that is not an identifier or a label"
            excluded = set(config.ID_COLUMNS_NOT_FOR_TRAINING) | set(LABEL_OUTPUT_COLUMNS)
            excluded |= set(config.METADATA_COLUMNS)
            feature_list = [c for c in df.columns if c not in excluded]

    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        hint = ""
        if set(missing) <= set(config.COMBINED_DROP_FEATURE_COLUMNS):
            hint = ("\nThis looks like a frame read from preprocessing.csv, which "
                    "carries one readable `%s` text column instead of the four "
                    "Protocol_* one-hot columns. Train on features_scaled.csv "
                    "instead." % config.COMBINED_PROTOCOL_COLUMN)
        raise PreprocessingError(
            "these feature_list columns are not in the frame: %s%s" % (missing, hint))
    return df[list(feature_list)]
