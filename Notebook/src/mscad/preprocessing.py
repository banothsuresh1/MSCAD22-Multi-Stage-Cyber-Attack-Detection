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
    features = df.drop(columns=meta_cols + [config.LABEL_COLUMN]).copy()

    if verbose:
        show_table(kv_table([
            ("rows", f"{len(df):,}"),
            ("columns in", df.shape[1]),
            ("metadata columns kept aside", len(meta_cols)),
            ("label column", config.LABEL_COLUMN),
            ("feature columns", features.shape[1]),
        ]), note="Metadata columns: %s%s" % (
            ", ".join(meta_cols),
            "" if not missing else "  (not present: %s)" % ", ".join(missing)))
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

    return {"status": saved.get("status", "written"), "features_clean": features,
            "features_scaled": scaled, "labels": labels, "metadata": metadata,
            "checks": checks, "report": report, "artifacts": artifacts,
            "saved": saved, "ok": bool(all_ok and saved.get("ok")),
            "clean_info": clean_info, "encode_info": encode_info}
