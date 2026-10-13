"""
src/mscad/splitting.py -- STEP 5: the stratified train / test validation split.

Input   C:\\MSCAD2022\\outputs\\step4\\   (Step 4's outputs)
            features_clean.csv   cleaned + encoded, UNSCALED
            labels.csv           Label, Label_encoded, Label_binary
            metadata.csv         Src IP, Dst IP, ports, Protocol, Timestamp, ...
            feature_list.json, label_mapping.json
Output  C:\\MSCAD2022\\outputs\\step5\\   (created if missing)

What Step 5 DOES
----------------
    5.1 split     a stratified 80/20 split on ROW INDICES, so features, labels
                  and metadata are selected with the same positions and can
                  never drift out of alignment
    5.2 scale     the SAME scaler type as Step 4, refitted on the TRAINING rows
                  only, then applied to both sides
    5.3 leakage   count the test rows whose exact feature vector also appears in
                  training -- reported, never dropped

What Step 5 deliberately does NOT do
------------------------------------
No class-imbalance handling: no SMOTE, no resampling, no class weights.  No
feature selection.  No modelling.  Those are later steps, and they must happen
on the TRAINING side of this split only.

Why the split is on indices
---------------------------
train_test_split is given np.arange(n_rows), not the frames.  The three frames
are then selected with .iloc[train_idx] / .iloc[test_idx], so row i of X_train,
y_train and meta_train is guaranteed to be the same flow.  Splitting each frame
separately would be three independent shuffles and the labels would no longer
belong to their features.

Why the scaler is refitted
--------------------------
Step 4's scaler saw every row, test rows included, so its min/max (or mean/std)
carry information from the test set.  Using features_scaled.csv across a split
leaks that into training and flatters every score.  Step 5 therefore starts from
features_clean.csv (UNSCALED) and fits a fresh scaler on X_train alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from . import data_loading as dl  # noqa: E402
from . import preprocessing as pp  # noqa: E402

# the display helpers from Step 2/3, so Step 5 renders the same way
show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

LABEL_COLUMNS = ["Label", "Label_encoded", "Label_binary"]


class SplittingError(Exception):
    """Raised when Step 5 cannot continue without guessing."""


# ===========================================================================
# 5.0  INPUT
# ===========================================================================
def load_step4_outputs(verbose: bool = True) -> dict[str, Any]:
    """Read the three Step 4 CSVs plus feature_list.json and label_mapping.json.

    features_clean.csv is the UNSCALED frame on purpose -- see the module
    docstring.  features_scaled.csv is never read here.
    """
    folder = config.usable_output_dir(config.STEP4_OUTPUT_DIR)
    if folder is None:
        raise SplittingError(
            "the Step 4 output folder is not usable here -- %s"
            % config.why_unusable(config.STEP4_OUTPUT_DIR))

    names = config.STEP4_FILES
    missing = [names[k] for k in ("features_clean", "labels", "metadata")
               if not (folder / names[k]).exists()]
    if missing:
        raise SplittingError(
            "Step 5 needs %s in %s. Run Step 4 first (set OVERWRITE_STEP4 = True "
            "if its outputs are stale)." % (missing, folder))

    out: dict[str, Any] = {"folder": str(folder)}
    out["features"] = pd.read_csv(folder / names["features_clean"], low_memory=False)
    out["labels"] = pd.read_csv(folder / names["labels"], low_memory=False)
    out["metadata"] = pd.read_csv(folder / names["metadata"], low_memory=False)

    feature_list_path = folder / names["feature_list"]
    if feature_list_path.exists():
        listed = json.loads(feature_list_path.read_text(encoding="utf-8"))
        out["feature_list"] = listed.get("features") or list(out["features"].columns)
        out["scaled_columns"] = listed.get("scaled_columns")
    else:
        out["feature_list"] = list(out["features"].columns)
        out["scaled_columns"] = None

    mapping_path = folder / names["label_mapping"]
    out["label_mapping"] = (json.loads(mapping_path.read_text(encoding="utf-8"))
                            if mapping_path.exists() else {})

    if verbose:
        show_table(kv_table([
            ("features_clean.csv", "%s rows x %d columns"
             % (f"{len(out['features']):,}", out["features"].shape[1])),
            ("labels.csv", "%s rows x %d columns"
             % (f"{len(out['labels']):,}", out["labels"].shape[1])),
            ("metadata.csv", "%s rows x %d columns"
             % (f"{len(out['metadata']):,}", out["metadata"].shape[1])),
            ("features in feature_list.json", len(out["feature_list"])),
            ("label classes", len(out["label_mapping"]) or "(no label_mapping.json)"),
            ("read from", str(folder)),
        ]), note="The features are the **unscaled** ones. Step 4's scaler saw every "
                 "row, so reusing features_scaled.csv across a split would leak test "
                 "statistics into training.")
    return out


def check_inputs(features: pd.DataFrame, labels: pd.DataFrame,
                 metadata: pd.DataFrame, stratify_column: str | None = None,
                 verbose: bool = True):
    """[OK]/[FAIL] on everything that must hold before a split is attempted.

    Raises SplittingError on anything that makes the split impossible: a row
    count mismatch, a missing stratify column, or a class with fewer than 2 rows
    (train_test_split cannot put such a class on both sides).
    """
    stratify_column = stratify_column or config.STRATIFY_COLUMN
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    counts = {"features_clean.csv": len(features), "labels.csv": len(labels),
              "metadata.csv": len(metadata)}
    same_rows = len(set(counts.values())) == 1
    add("the three input files have the same row count", same_rows,
        ", ".join("%s=%s" % (k, f"{v:,}") for k, v in counts.items()))

    if config.STEP5_EXPECTED_ROWS is not None:
        matches = len(features) == config.STEP5_EXPECTED_ROWS
        checks.append({
            "check": "row count vs the expected %s"
                     % f"{config.STEP5_EXPECTED_ROWS:,}",
            "result": "[OK]" if matches else "[NOTE]",
            "detail": ("%s rows" % f"{len(features):,}") if matches else
                      ("%s rows -- the split still works, but check that Step 3/4 "
                       "are not stale (config.STEP5_EXPECTED_ROWS says %s)"
                       % (f"{len(features):,}", f"{config.STEP5_EXPECTED_ROWS:,}"))})

    has_column = stratify_column in labels.columns
    add("the stratify column `%s` is present" % stratify_column, has_column,
        ", ".join(labels.columns))

    missing_labels = [c for c in LABEL_COLUMNS if c not in labels.columns]
    add("labels.csv carries %s" % ", ".join(LABEL_COLUMNS), not missing_labels,
        "all three present" if not missing_labels else "missing %s" % missing_labels)

    rare: dict[str, int] = {}
    if has_column:
        class_counts = labels[stratify_column].value_counts()
        rare = {str(k): int(v) for k, v in class_counts.items() if v < 2}
        add("every class has at least 2 rows (stratify needs both sides)",
            not rare,
            "smallest class: %s with %s rows"
            % (class_counts.idxmin(), f"{int(class_counts.min()):,}")
            if not rare else "too few rows: %s" % rare)

        n_test = int(np.floor(len(labels) * config.TEST_SIZE))
        enough = n_test >= len(class_counts)
        add("the test set is big enough for one row per class", enough,
            "test size %s >= %d classes" % (f"{n_test:,}", len(class_counts)))

    no_nan = not bool(labels[LABEL_COLUMNS].isna().to_numpy().any()) \
        if not missing_labels else False
    add("no missing values in the label columns", no_nan,
        "0 NaN cells" if no_nan else "found NaN, or a label column is missing")

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] != "[FAIL]").all())
    if verbose:
        show_table(table, note="A `[NOTE]` row is informational and does not block "
                               "the split; a `[FAIL]` row does.")

    if not same_rows:
        raise SplittingError(
            "[FAIL] the Step 4 files disagree on the row count (%s), so rows cannot be "
            "selected by position. Re-run Step 4 with OVERWRITE_STEP4 = True." % counts)
    if not has_column:
        raise SplittingError(
            "[FAIL] labels.csv has no `%s` column, so the split cannot be stratified. "
            "Columns present: %s" % (stratify_column, list(labels.columns)))
    if rare:
        raise SplittingError(
            "[FAIL] a stratified split needs at least 2 rows per class, so that each "
            "class can appear in BOTH train and test. These classes have fewer: %s.\n"
            "Either merge them into a larger class, or drop them before Step 5 -- but "
            "say so in the results, because a class with 1 example cannot be learned "
            "or evaluated." % rare)
    return table, all_ok


# ===========================================================================
# 5.1  SPLIT
# ===========================================================================
def split_indices(labels: pd.DataFrame, test_size: float | None = None,
                  random_state: int | None = None,
                  stratify_column: str | None = None, verbose: bool = True):
    """Stratified train/test split on ROW POSITIONS.

    Returns (train_idx, test_idx) as int64 arrays of positions into the Step 4
    files.  Everything else in Step 5 selects with these two arrays, so the
    three frames stay aligned by construction.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SplittingError(
            'scikit-learn is needed for the split.  Install it with:\n'
            '    "%s" -m pip install -r requirements.txt' % sys.executable) from exc

    test_size = config.TEST_SIZE if test_size is None else test_size
    random_state = config.RANDOM_STATE if random_state is None else random_state
    stratify_column = stratify_column or config.STRATIFY_COLUMN

    positions = np.arange(len(labels), dtype="int64")
    y = labels[stratify_column].to_numpy()

    train_idx, test_idx = train_test_split(
        positions, test_size=test_size, random_state=random_state,
        shuffle=True, stratify=y)
    train_idx = np.sort(np.asarray(train_idx, dtype="int64"))
    test_idx = np.sort(np.asarray(test_idx, dtype="int64"))

    if verbose:
        show_table(kv_table([
            ("rows in", f"{len(labels):,}"),
            ("test_size", test_size),
            ("random_state", random_state),
            ("stratify on", stratify_column),
            ("train rows", "%s  (%.2f%%)"
             % (f"{len(train_idx):,}", 100.0 * len(train_idx) / len(labels))),
            ("test rows", "%s  (%.2f%%)"
             % (f"{len(test_idx):,}", 100.0 * len(test_idx) / len(labels))),
        ]), note="The split is on row POSITIONS, then features, labels and metadata "
                 "are all selected with the same positions -- so row i of X_train, "
                 "y_train and meta_train is the same flow. The indices are sorted, so "
                 "each side keeps the original chronological row order.")
    return train_idx, test_idx


def take(frame: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    """One side of the split, with a clean 0..n-1 index."""
    return frame.iloc[indices].reset_index(drop=True)


def class_balance_table(labels: pd.DataFrame, train_idx: np.ndarray,
                        test_idx: np.ndarray, column: str | None = None,
                        verbose: bool = True):
    """Per-class counts and shares in full / train / test, with [OK]/[FAIL].

    [OK] when every class's share of its own side differs by less than
    config.SHARE_TOLERANCE_PP percentage points between train and test, AND
    every class has at least one test row.
    """
    column = column or config.STRATIFY_COLUMN
    full = labels[column]
    train = full.iloc[train_idx]
    test = full.iloc[test_idx]

    rows = []
    for name in full.value_counts().index:
        total = int((full == name).sum())
        n_train = int((train == name).sum())
        n_test = int((test == name).sum())
        rows.append({
            column: name,
            "total": total,
            "train": n_train,
            "test": n_test,
            "train % of class": round(100.0 * n_train / total, 3) if total else 0.0,
            "test % of class": round(100.0 * n_test / total, 3) if total else 0.0,
            "share of full %": round(100.0 * total / len(full), 4),
            "share of train %": round(100.0 * n_train / len(train), 4),
            "share of test %": round(100.0 * n_test / len(test), 4),
        })
    table = pd.DataFrame(rows)
    table["share drift pp"] = (table["share of train %"]
                               - table["share of test %"]).abs().round(4)

    totals = {column: "TOTAL", "total": len(full), "train": len(train),
              "test": len(test),
              "train % of class": round(100.0 * len(train) / len(full), 3),
              "test % of class": round(100.0 * len(test) / len(full), 3),
              "share of full %": 100.0, "share of train %": 100.0,
              "share of test %": 100.0, "share drift pp": ""}
    shown = pd.concat([table, pd.DataFrame([totals])], ignore_index=True)

    worst = float(table["share drift pp"].max()) if len(table) else 0.0
    missing = table.loc[table["test"] < 1, column].astype(str).tolist()

    # A stratified split can only place WHOLE rows, so moving the single row that
    # rounding forces already shifts a share by 100/n_test percentage points.
    # Below that figure no split can do better, so the tolerance is the larger of
    # the configured one and this granularity -- otherwise a small dataset would
    # be reported as "not stratified" for arithmetic that cannot be avoided.  On
    # the full dataset the granularity is tiny and the configured 0.1 pp governs.
    granularity = 100.0 / len(test) if len(test) else 0.0
    tolerance = max(config.SHARE_TOLERANCE_PP, granularity)
    balanced = worst < tolerance
    ok = bool(balanced and not missing)

    if verbose:
        show_table(shown, note="`share of ... %` is the class's share of all rows on "
                               "that side; `share drift pp` is how far train and test "
                               "disagree, in percentage points.")
        print("[OK]   every class's share drifts < %.4f pp (worst %.4f pp)"
              % (tolerance, worst) if balanced
              else "[FAIL] a class share drifts by %.4f pp, over the %.4f pp tolerance"
                   % (worst, tolerance))
        if tolerance > config.SHARE_TOLERANCE_PP:
            show_note("The tolerance here is **%.4f pp**, not the configured %.1f pp: "
                      "with only %s test rows one row is worth %.4f pp, so no split "
                      "can land closer. On the full dataset the configured %.1f pp is "
                      "what applies."
                      % (tolerance, config.SHARE_TOLERANCE_PP, f"{len(test):,}",
                         granularity, config.SHARE_TOLERANCE_PP))
        print("[OK]   every class has at least 1 test row" if not missing
              else "[FAIL] no test rows for: %s" % ", ".join(missing))
    return shown, ok, {"worst_drift_pp": worst, "tolerance_pp": tolerance,
                       "granularity_pp": granularity,
                       "classes_without_test_rows": missing}


# ===========================================================================
# 5.2  SCALE -- refit on TRAIN only
# ===========================================================================
def scaled_column_names(features: pd.DataFrame,
                        scaled_columns: Sequence[str] | None = None) -> list[str]:
    """The columns to scale: the same ones Step 4 scaled.

    When feature_list.json recorded them they are used verbatim (minus anything
    no longer present); otherwise they are recomputed with Step 4's own rule, so
    the one-hot and binary markers are left alone either way.
    """
    if scaled_columns:
        return [c for c in scaled_columns if c in features.columns]
    return pp.numeric_feature_columns(features)


def refit_scaler_on_train(X_train: pd.DataFrame, X_test: pd.DataFrame,
                          scaled_columns: Sequence[str] | None = None,
                          scaler_name: str | None = None,
                          clip: bool | None = None, verbose: bool = True):
    """Fit a fresh scaler on X_train ONLY, then transform both sides.

    Returns (X_train_scaled, X_test_scaled, scaler, info).
    """
    scaler_name = (scaler_name or config.SCALER or "minmax").lower()
    clip = config.CLIP_TEST_TO_TRAIN_RANGE if clip is None else clip
    columns = scaled_column_names(X_train, scaled_columns)

    try:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
    except ImportError as exc:
        raise SplittingError(
            'scikit-learn is needed to scale.  Install it with:\n'
            '    "%s" -m pip install -r requirements.txt' % sys.executable) from exc

    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:
        raise SplittingError(
            "config.SCALER is %r; it must be \"minmax\" or \"standard\"." % scaler_name)

    train_out = X_train.copy()
    test_out = X_test.copy()
    if columns:
        # fit on the TRAINING rows only -- this is the whole point of Step 5.2
        scaler.fit(X_train[columns].to_numpy(dtype="float64"))
        train_out[columns] = scaler.transform(X_train[columns].to_numpy(dtype="float64"))
        test_out[columns] = scaler.transform(X_test[columns].to_numpy(dtype="float64"))

    info: dict[str, Any] = {"scaler_name": scaler_name, "columns": list(columns),
                            "n_columns": len(columns), "clipped": bool(clip)}

    # --- what the fit produced, and what it means for the test side -------
    out_of_range = pd.DataFrame(columns=["column", "test values < 0", "test values > 1",
                                         "test min", "test max"])
    if columns:
        train_block = train_out[columns].to_numpy(dtype="float64")
        test_block = test_out[columns].to_numpy(dtype="float64")
        info["train_min"] = float(np.nanmin(train_block))
        info["train_max"] = float(np.nanmax(train_block))
        info["test_min"] = float(np.nanmin(test_block))
        info["test_max"] = float(np.nanmax(test_block))

        if scaler_name == "minmax":
            below = (test_block < 0).sum(axis=0)
            above = (test_block > 1).sum(axis=0)
            rows = [{"column": c,
                     "test values < 0": int(below[i]),
                     "test values > 1": int(above[i]),
                     "test min": round(float(test_block[:, i].min()), 6),
                     "test max": round(float(test_block[:, i].max()), 6)}
                    for i, c in enumerate(columns) if below[i] or above[i]]
            out_of_range = pd.DataFrame(rows, columns=out_of_range.columns)
            info["test_cells_outside_0_1"] = int(below.sum() + above.sum())
            info["test_columns_outside_0_1"] = len(out_of_range)

    if clip and columns and scaler_name == "minmax":
        test_out[columns] = test_out[columns].clip(lower=0.0, upper=1.0)
        info["test_clipped_to_0_1"] = True

    info["out_of_range"] = out_of_range

    if verbose:
        pairs = [("scaler", scaler_name),
                 ("fitted on", "X_train only (%s rows)" % f"{len(X_train):,}"),
                 ("columns scaled", "%d of %d" % (len(columns), X_train.shape[1])),
                 ("columns left alone",
                  ", ".join(c for c in X_train.columns if c not in set(columns))
                  or "(none)")]
        if columns:
            pairs += [("train range", "[%.6f, %.6f]"
                       % (info["train_min"], info["train_max"])),
                      ("test range", "[%.6f, %.6f]"
                       % (info["test_min"], info["test_max"]))]
        if scaler_name == "minmax" and columns:
            pairs.append(("test cells outside [0, 1]",
                          "%s in %d column(s)"
                          % (f"{info['test_cells_outside_0_1']:,}",
                             info["test_columns_outside_0_1"])))
        pairs.append(("clipped to [0, 1]", "yes" if clip else "no (expected, kept)"))
        show_table(kv_table(pairs))

        if scaler_name == "minmax" and columns:
            in_band = (info["train_min"] >= -1e-9 and info["train_max"] <= 1 + 1e-9)
            print("[OK]   train is inside [0, 1]: [%.6f, %.6f]"
                  % (info["train_min"], info["train_max"]) if in_band
                  else "[FAIL] train is outside [0, 1]: [%.6f, %.6f]"
                       % (info["train_min"], info["train_max"]))
            if len(out_of_range):
                show_table(out_of_range.head(20),
                           note="Test values outside [0, 1] are EXPECTED: the scaler "
                                "never saw these rows, so a test flow can exceed every "
                                "training value. `CLIP_TEST_TO_TRAIN_RANGE = False` "
                                "keeps them, because clipping would quietly destroy "
                                "real differences between large flows.")
            else:
                show_note("No test value falls outside [0, 1].")
    return train_out, test_out, scaler, info


# ===========================================================================
# 5.3  LEAKAGE CHECK -- report only, nothing is dropped
# ===========================================================================
def _row_keys(frame: pd.DataFrame) -> np.ndarray:
    """One exact key per row: the raw bytes of its float64 feature vector.

    Bytes, not a hash, so two rows match only if every feature is bit-identical.
    """
    block = np.ascontiguousarray(frame.to_numpy(dtype="float64"))
    return np.array([row.tobytes() for row in block], dtype=object)


def leakage_report(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_test: pd.DataFrame, label_column: str | None = None,
                   verbose: bool = True):
    """Count test rows whose exact feature vector also occurs in training.

    Nothing is dropped.  These are genuine flows: thousands of scan, flood and
    brute-force flows really are identical once the identifiers are removed, and
    deleting them would change the dataset.  But they mean a model can score a
    test row it has effectively already seen, so the number belongs in the
    results.
    """
    label_column = label_column or config.LABEL_COLUMN
    if list(X_train.columns) != list(X_test.columns):
        raise SplittingError(
            "X_train and X_test must have the same columns in the same order "
            "before the leakage check")

    train_keys = set(_row_keys(X_train).tolist())
    test_keys = _row_keys(X_test)
    duplicated = np.array([k in train_keys for k in test_keys], dtype=bool)

    n_test = len(X_test)
    overall = {"test rows": n_test,
               "also in train": int(duplicated.sum()),
               "% of test": round(100.0 * float(duplicated.sum()) / n_test, 3)
               if n_test else 0.0}

    rows = []
    labels = y_test[label_column].to_numpy()
    for name in pd.Series(labels).value_counts().index:
        mask = labels == name
        n = int(mask.sum())
        hit = int(duplicated[mask].sum())
        rows.append({label_column: name, "test rows": n, "also in train": hit,
                     "% of this class's test rows":
                         round(100.0 * hit / n, 3) if n else 0.0})
    table = pd.DataFrame(rows)
    table = pd.concat([table, pd.DataFrame([{
        label_column: "TOTAL", "test rows": n_test,
        "also in train": overall["also in train"],
        "% of this class's test rows": overall["% of test"]}])], ignore_index=True)

    if verbose:
        show_table(table, note="An exact match on the full feature vector, computed on "
                               "the UNSCALED features. Nothing is dropped -- these are "
                               "real flows.")
        show_note("**%s of %s test rows (%.2f%%) have a feature-identical twin in "
                  "training.** Report this alongside any accuracy figure: those rows "
                  "are effectively memorised, not generalised, so the headline score "
                  "is optimistic. Per-class recall on the classes with a LOW share "
                  "here is the honest signal."
                  % (f"{overall['also in train']:,}", f"{n_test:,}",
                     overall["% of test"]))
    return table, overall, duplicated


# ===========================================================================
# 5.4  THE VIEW FILES -- the preprocessing.csv layout, one per side
# ===========================================================================
def build_view(features_scaled: pd.DataFrame, labels: pd.DataFrame,
               metadata: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """One side of the split in the preprocessing.csv layout, for READING.

        Src IP | Dst IP | Src Port | Dst Port | Protocol | Timestamp
               | <features without the Protocol_* one-hot columns> | Label

    It reuses Step 4's own builder, so the layout cannot drift from
    preprocessing.csv.  Like that file it is a readable / sequence-work view and
    NOT a training matrix: the identifiers leak the labels and Protocol is text.
    """
    return pp.build_combined(features_scaled, labels, metadata,
                             feature_order=list(features_scaled.columns),
                             verbose=verbose)


# ===========================================================================
# the class-count chart
# ===========================================================================
# Two series, so two hues and a legend -- identity is never colour alone, and
# only the TEST bars are labelled: the question this chart answers is "does
# every class still have test rows?", and a number on all 20+ bars would be
# noise.  Log scale because the classes differ by four orders of magnitude.
_TRAIN_COLOUR = "#2a78d6"      # categorical slot 1
_TEST_COLOUR = "#eb6834"       # categorical slot 2


def plot_split_class_counts(balance: pd.DataFrame, path: Path | None = None,
                            column: str | None = None, verbose: bool = True):
    """Grouped bar chart of the per-class train and test counts -> PNG."""
    column = column or config.STRATIFY_COLUMN
    try:
        import matplotlib
        matplotlib.use("Agg")                      # no GUI needed, just a file
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("  [SKIP]  matplotlib is not installed, so no chart was drawn.")
            print('             "%s" -m pip install matplotlib' % sys.executable)
        return None

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        folder = prepare_output_dir(config.STEP5_OUTPUT_DIR, "the split chart",
                                    verbose=verbose)
        if folder is None:
            return None
        path = folder / config.STEP5_FILES["chart"]

    rows = balance[balance[column].astype(str) != "TOTAL"].copy()
    rows = rows.sort_values("total", ascending=False)
    names = rows[column].astype(str).to_numpy()
    train = rows["train"].to_numpy(dtype="float64")
    test = rows["test"].to_numpy(dtype="float64")

    positions = np.arange(len(names), dtype="float64")
    width = 0.34                                   # < 0.5 leaves a gap between pairs
    fig, ax = plt.subplots(figsize=(max(9.0, 1.15 * len(names)), 5.2), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    # a hair of surface between the two bars of a pair, per the mark spec
    gap = 0.035
    ax.bar(positions - width / 2 - gap, np.maximum(train, 0.8), width,
           label="train", color=_TRAIN_COLOUR)
    bars_test = ax.bar(positions + width / 2 + gap, np.maximum(test, 0.8), width,
                       label="test", color=_TEST_COLOUR)

    ax.set_yscale("log")
    ax.set_ylabel("flows (log scale)", color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Flows per class after the %d/%d stratified split "
                 "(%s train, %s test)"
                 % (round(100 * (1 - config.TEST_SIZE)),
                    round(100 * config.TEST_SIZE),
                    f"{int(rows['train'].sum()):,}", f"{int(rows['test'].sum()):,}"),
                 color=dl._INK, fontsize=12, pad=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=20, ha="right")

    ax.yaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(dl._GRID)
    ax.tick_params(colors=dl._INK_SOFT, labelsize=9)
    ax.set_ylim(0.8, max(train.max(), test.max(), 1.0) * 4)

    # only the test bars carry a number: that is the column being checked
    for bar, value in zip(bars_test, test):
        ax.annotate(f"{int(value):,}",
                    (bar.get_x() + bar.get_width() / 2, max(value, 0.8)),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color=dl._INK)

    legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


# ===========================================================================
# SAVE and VERIFY
# ===========================================================================
def build_split_summary(train_idx: np.ndarray, test_idx: np.ndarray,
                        balance: pd.DataFrame, scale_info: dict[str, Any],
                        leakage: dict[str, Any], n_rows: int) -> pd.DataFrame:
    """The one-row-per-metric summary saved as split_summary.csv."""
    return kv_table([
        ("rows in", n_rows),
        ("train rows", len(train_idx)),
        ("test rows", len(test_idx)),
        ("test_size", config.TEST_SIZE),
        ("random_state", config.RANDOM_STATE),
        ("stratify column", config.STRATIFY_COLUMN),
        ("classes", int((balance[config.STRATIFY_COLUMN].astype(str) != "TOTAL").sum())),
        ("scaler", scale_info.get("scaler_name")),
        ("scaler fitted on", "train only"),
        ("columns scaled", scale_info.get("n_columns")),
        ("test cells outside [0,1]", scale_info.get("test_cells_outside_0_1", "-")),
        ("test clipped to [0,1]", scale_info.get("clipped")),
        ("test rows also in train", leakage.get("also in train")),
        ("test rows also in train %", leakage.get("% of test")),
    ])


def save_step5(pieces: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Write every Step 5 output, then re-read each one and check it."""
    out: dict[str, Any] = {"folder": str(config.STEP5_OUTPUT_DIR), "files": {}}
    folder = prepare_output_dir(config.STEP5_OUTPUT_DIR, "the Step 5 outputs",
                               verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP5_FILES
    frames = {
        "X_train": pieces["X_train"], "X_test": pieces["X_test"],
        "X_train_unscaled": pieces["X_train_unscaled"],
        "X_test_unscaled": pieces["X_test_unscaled"],
        "y_train": pieces["y_train"], "y_test": pieces["y_test"],
        "meta_train": pieces["meta_train"], "meta_test": pieces["meta_test"],
        "train_view": pieces["train_view"], "test_view": pieces["test_view"],
        "class_balance": pieces["balance"], "binary_balance": pieces["binary_balance"],
        "leakage": pieces["leakage_table"],
        "split_summary": pieces["split_summary"], "report": pieces["report"],
    }
    for key, frame in frames.items():
        frame.to_csv(folder / names[key], index=False)
        out["files"][key] = str(folder / names[key])

    # the row positions in the Step 4 files, so any split can be reproduced
    for key, indices in (("train_indices", pieces["train_idx"]),
                         ("test_indices", pieces["test_idx"])):
        pd.DataFrame({"row_position": indices}).to_csv(folder / names[key], index=False)
        out["files"][key] = str(folder / names[key])

    try:
        import joblib
        joblib.dump({"scaler": pieces["scaler"],
                     "scaler_name": pieces["scale_info"]["scaler_name"],
                     "scaled_columns": pieces["scale_info"]["columns"],
                     "feature_list": list(pieces["X_train"].columns),
                     "fitted_on": "train split only",
                     "test_size": config.TEST_SIZE,
                     "random_state": config.RANDOM_STATE,
                     "stratify_column": config.STRATIFY_COLUMN},
                    folder / names["scaler"])
        out["files"]["scaler"] = str(folder / names["scaler"])
    except ImportError:
        if verbose:
            show_note("joblib is not installed, so the train-fitted scaler was not "
                      "saved (`\"%s\" -m pip install joblib`). The CSVs are unaffected."
                      % sys.executable)

    chart = plot_split_class_counts(pieces["balance"],
                                    path=folder / names["chart"], verbose=verbose)
    if chart is not None:
        out["files"]["chart"] = str(chart)

    out["status"] = "written"
    verification, ok = verify_step5(pieces, folder=folder, verbose=verbose)
    out["verification"] = verification
    out["ok"] = ok
    return out


def verify_step5(pieces: dict[str, Any], folder: Path | None = None,
                 verbose: bool = True):
    """Re-read every Step 5 file from disk and check it.  [OK]/[FAIL]."""
    folder = Path(folder) if folder is not None else config.STEP5_OUTPUT_DIR
    names = config.STEP5_FILES
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    present = {key: (folder / names[key]).exists() for key in names}
    add("every expected file is on disk", all(present.values()),
        "%d of %d" % (sum(present.values()), len(present))
        if all(present.values())
        else "missing %s" % [names[k] for k, v in present.items() if not v])

    read: dict[str, pd.DataFrame] = {}
    for key in ("X_train", "X_test", "X_train_unscaled", "X_test_unscaled",
                "y_train", "y_test", "meta_train", "meta_test",
                "train_view", "test_view", "train_indices", "test_indices"):
        path = folder / names[key]
        if path.exists():
            read[key] = pd.read_csv(path, low_memory=False)

    n_rows = pieces["n_rows"]
    n_train = len(read.get("X_train", pieces["X_train"]))
    n_test = len(read.get("X_test", pieces["X_test"]))
    add("train + test rows == the Step 4 row count", n_train + n_test == n_rows,
        "%s + %s = %s, expected %s" % (f"{n_train:,}", f"{n_test:,}",
                                       f"{n_train + n_test:,}", f"{n_rows:,}"))

    train_pos = read["train_indices"]["row_position"].to_numpy() \
        if "train_indices" in read else pieces["train_idx"]
    test_pos = read["test_indices"]["row_position"].to_numpy() \
        if "test_indices" in read else pieces["test_idx"]
    overlap = np.intersect1d(train_pos, test_pos)
    add("no row position is in both train and test", overlap.size == 0,
        "0 overlapping rows" if overlap.size == 0
        else "%d overlapping: %s" % (overlap.size, overlap[:5].tolist()))
    union = np.union1d(train_pos, test_pos)
    add("the two index files cover every row exactly once",
        union.size == n_rows and train_pos.size + test_pos.size == n_rows,
        "%s unique positions over %s rows" % (f"{union.size:,}", f"{n_rows:,}"))

    if "X_train" in read and "X_test" in read:
        same_cols = list(read["X_train"].columns) == list(read["X_test"].columns)
        add("X_train and X_test have identical columns, in order", same_cols,
            "%d columns" % read["X_train"].shape[1] if same_cols
            else "X_train %d vs X_test %d columns"
                 % (read["X_train"].shape[1], read["X_test"].shape[1]))
        in_memory = list(pieces["X_train"].columns)
        add("the columns on disk match the frame in memory",
            list(read["X_train"].columns) == in_memory,
            "%d columns" % len(in_memory))

    for side in ("train", "test"):
        xk, yk, mk = "X_%s" % side, "y_%s" % side, "meta_%s" % side
        if all(k in read for k in (xk, yk, mk)):
            n = len(read[xk])
            aligned = len(read[yk]) == n and len(read[mk]) == n
            add("%s: features, labels and metadata have the same rows" % side, aligned,
                "%s rows in all three" % f"{n:,}")
            if "Orig_Row" in read[mk].columns:
                source_meta = pieces["metadata"]
                want = source_meta["Orig_Row"].to_numpy()[
                    pieces["%s_idx" % side]] if "Orig_Row" in source_meta.columns else None
                if want is not None:
                    same = read[mk]["Orig_Row"].to_numpy().tolist() == want.tolist()
                    add("%s: metadata Orig_Row matches the split positions" % side,
                        same, "%s rows compared" % f"{n:,}")
            labels_match = (read[yk][config.LABEL_COLUMN].astype(str).tolist()
                            == pieces[yk][config.LABEL_COLUMN].astype(str).tolist())
            add("%s: labels on disk match the frame in memory" % side, labels_match,
                "%s rows compared" % f"{n:,}")

    for side in ("train", "test"):
        vk = "%s_view" % side
        if vk in read:
            view = read[vk]
            first = list(view.columns[:len(config.COMBINED_ID_COLUMNS)])
            add("%s_view has the preprocessing.csv layout" % side,
                first == list(config.COMBINED_ID_COLUMNS)
                and list(view.columns[-1:]) == [config.LABEL_COLUMN]
                and not [c for c in config.COMBINED_DROP_FEATURE_COLUMNS
                         if c in view.columns],
                "identifiers first, Label last, no Protocol_* columns")

    if "X_train" in read:
        block = read["X_train"].to_numpy(dtype="float64")
        n_nan, n_inf = int(np.isnan(block).sum()), int(np.isinf(block).sum())
        add("no NaN or inf in X_train", n_nan == 0 and n_inf == 0,
            "%d NaN, %d inf" % (n_nan, n_inf))
    if "X_test" in read:
        block = read["X_test"].to_numpy(dtype="float64")
        n_nan, n_inf = int(np.isnan(block).sum()), int(np.isinf(block).sum())
        add("no NaN or inf in X_test", n_nan == 0 and n_inf == 0,
            "%d NaN, %d inf" % (n_nan, n_inf))

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_title("Step 5 -- files written, then re-read from disk")
        show_table(table, note="Folder: %s" % folder)
        if not all_ok:
            show_note("**[FAIL]** -- the Step 5 outputs do not agree with each other; "
                      "do not train on them.")
    return table, all_ok


def build_report(check_table: pd.DataFrame, balance: pd.DataFrame,
                 scale_info: dict[str, Any], leakage_table: pd.DataFrame,
                 leakage: dict[str, Any]) -> pd.DataFrame:
    """step5_report.csv -- one row per action, in the order they happened."""
    rows = [
        {"step": "5.0 input", "action": "read features_clean.csv, labels.csv, "
                                        "metadata.csv",
         "detail": "unscaled features on purpose; Step 4's scaler saw every row"},
        {"step": "5.1 split", "action": "stratified train_test_split on row positions",
         "detail": "test_size=%s, random_state=%s, stratify=%s, shuffle=True"
                   % (config.TEST_SIZE, config.RANDOM_STATE, config.STRATIFY_COLUMN)},
        {"step": "5.1 split", "action": "class balance checked",
         "detail": "worst share drift %s pp, tolerance %s pp"
                   % (balance["share drift pp"].replace("", np.nan).max(),
                      config.SHARE_TOLERANCE_PP)},
        {"step": "5.2 scale", "action": "refit %s on the training rows only"
                                        % scale_info.get("scaler_name"),
         "detail": "%d columns scaled; one-hot and binary markers left alone"
                   % scale_info.get("n_columns", 0)},
        {"step": "5.2 scale", "action": "test values outside [0, 1] reported, not clipped",
         "detail": "%s cells in %s column(s); CLIP_TEST_TO_TRAIN_RANGE=%s"
                   % (scale_info.get("test_cells_outside_0_1", "-"),
                      scale_info.get("test_columns_outside_0_1", "-"),
                      config.CLIP_TEST_TO_TRAIN_RANGE)},
        {"step": "5.3 leakage", "action": "test rows with a feature-identical twin in "
                                          "train counted",
         "detail": "%s of %s test rows (%s%%) -- reported, nothing dropped"
                   % (leakage.get("also in train"), leakage.get("test rows"),
                      leakage.get("% of test"))},
        {"step": "5.x not done", "action": "class imbalance NOT handled",
         "detail": "no SMOTE, no resampling, no class weights -- a later step, and on "
                   "the TRAINING side only"},
        {"step": "5.x not done", "action": "no feature selection, no modelling",
         "detail": "later steps"},
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# load / exist / run
# ===========================================================================
def step5_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP5_OUTPUT_DIR)
    if folder is None:
        return False
    keys = ("X_train", "X_test", "X_train_unscaled", "X_test_unscaled",
            "y_train", "y_test", "meta_train", "meta_test",
            "train_indices", "test_indices")
    return all((folder / config.STEP5_FILES[k]).exists() for k in keys)


def load_step5(verbose: bool = True) -> dict[str, Any]:
    """Read the saved Step 5 outputs back instead of resplitting."""
    folder = config.usable_output_dir(config.STEP5_OUTPUT_DIR)
    if folder is None:
        raise SplittingError("the Step 5 folder is not usable here -- %s"
                             % config.why_unusable(config.STEP5_OUTPUT_DIR))
    names = config.STEP5_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    for key in ("X_train", "X_test", "X_train_unscaled", "X_test_unscaled",
                "y_train", "y_test", "meta_train", "meta_test",
                "train_view", "test_view", "split_summary", "report"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    # the tables Step 5 computed, so the notebook shows real numbers on this
    # path too instead of "not available"
    for key, name in (("balance", "class_balance"),
                      ("binary_balance", "binary_balance"),
                      ("leakage_table", "leakage")):
        path = folder / names[name]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    out["train_idx"] = pd.read_csv(folder / names["train_indices"])["row_position"] \
        .to_numpy(dtype="int64")
    out["test_idx"] = pd.read_csv(folder / names["test_indices"])["row_position"] \
        .to_numpy(dtype="int64")
    out["n_rows"] = len(out["train_idx"]) + len(out["test_idx"])

    scaler_path = folder / names["scaler"]
    out["scaler"] = None
    if scaler_path.exists():
        try:
            import joblib
            out["scaler"] = joblib.load(scaler_path)
        except ImportError:
            pass

    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP5 = False)"),
            ("folder", str(folder)),
            ("X_train", "%s rows x %d columns"
             % (f"{len(out['X_train']):,}", out["X_train"].shape[1])),
            ("X_test", "%s rows x %d columns"
             % (f"{len(out['X_test']):,}", out["X_test"].shape[1])),
            ("train + test", f"{out['n_rows']:,}"),
        ]), note="Set `OVERWRITE_STEP5 = True` in config.py to split again.")
    return out


def run_step5(overwrite: bool | None = None, verbose: bool = True) -> dict[str, Any]:
    """STEP 5 end to end: check -> split -> refit the scaler -> report -> save."""
    overwrite = config.OVERWRITE_STEP5 if overwrite is None else overwrite
    if not overwrite and step5_outputs_exist():
        return load_step5(verbose=verbose)

    if verbose:
        show_title("Step 5.0 -- input from Step 4", level=3)
    loaded = load_step4_outputs(verbose=verbose)
    features, labels, metadata = (loaded["features"], loaded["labels"],
                                  loaded["metadata"])
    check_table, _checks_ok = check_inputs(features, labels, metadata,
                                           verbose=verbose)

    if verbose:
        show_title("Step 5.1 -- stratified split", level=3)
    train_idx, test_idx = split_indices(labels, verbose=verbose)

    X_train_unscaled, X_test_unscaled = take(features, train_idx), take(features, test_idx)
    y_train, y_test = take(labels, train_idx), take(labels, test_idx)
    meta_train, meta_test = take(metadata, train_idx), take(metadata, test_idx)

    balance, balance_ok, balance_info = class_balance_table(
        labels, train_idx, test_idx, verbose=verbose)
    if verbose:
        show_title("Binary target (benign vs attack)", level=4)
    binary_balance, binary_ok, _bi = class_balance_table(
        labels, train_idx, test_idx, column="Label_binary", verbose=verbose)

    if verbose:
        show_title("Step 5.2 -- scale, fitted on the training rows only", level=3)
    X_train, X_test, scaler, scale_info = refit_scaler_on_train(
        X_train_unscaled, X_test_unscaled,
        scaled_columns=loaded.get("scaled_columns"), verbose=verbose)

    if verbose:
        show_title("Step 5.3 -- leakage check (reported, nothing dropped)", level=3)
    leakage_table, leakage, _mask = leakage_report(
        X_train_unscaled, X_test_unscaled, y_test, verbose=verbose)

    train_view = build_view(X_train, y_train, meta_train)
    test_view = build_view(X_test, y_test, meta_test)

    summary = build_split_summary(train_idx, test_idx, balance, scale_info,
                                  leakage, len(features))
    report = build_report(check_table, balance, scale_info, leakage_table, leakage)

    pieces = {
        "X_train": X_train, "X_test": X_test,
        "X_train_unscaled": X_train_unscaled, "X_test_unscaled": X_test_unscaled,
        "y_train": y_train, "y_test": y_test,
        "meta_train": meta_train, "meta_test": meta_test,
        "train_view": train_view, "test_view": test_view,
        "train_idx": train_idx, "test_idx": test_idx,
        "metadata": metadata, "n_rows": len(features),
        "split_summary": summary, "report": report,
        "balance": balance, "binary_balance": binary_balance,
        "scaler": scaler, "scale_info": scale_info,
        "leakage": leakage, "leakage_table": leakage_table,
        "check_table": check_table,
    }

    if verbose:
        show_title("Step 5.4 -- saved files", level=3)
    saved = save_step5(pieces, verbose=verbose)

    if verbose:
        show_title("Step 5 -- summary", level=3)
        show_table(kv_table([
            ("X_train", "%s rows x %d columns"
             % (f"{X_train.shape[0]:,}", X_train.shape[1])),
            ("X_test", "%s rows x %d columns"
             % (f"{X_test.shape[0]:,}", X_test.shape[1])),
            ("classes with no test rows",
             ", ".join(balance_info["classes_without_test_rows"]) or "(none)"),
            ("worst class share drift", "%.4f pp (tolerance %.4f pp)"
             % (balance_info["worst_drift_pp"], balance_info["tolerance_pp"])),
            ("scaler", "%s, fitted on train only" % scale_info["scaler_name"]),
            ("test rows also in train", "%s (%.2f%%)"
             % (f"{leakage['also in train']:,}", leakage["% of test"])),
            ("stratified split OK", balance_ok and binary_ok),
            ("files verified", saved.get("ok")),
            ("output folder", saved.get("folder")),
        ]), note="Class imbalance is **untouched** on purpose -- no SMOTE, no "
                 "resampling, no class weights. Handle it in a later step, fitted on "
                 "the TRAINING side of this split only.")

    out = dict(pieces)
    out.update({"status": saved.get("status", "written"), "saved": saved,
                "ok": bool(balance_ok and binary_ok and saved.get("ok"))})
    return out
