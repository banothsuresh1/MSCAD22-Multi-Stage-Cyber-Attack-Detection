"""
src/mscad/feature_selection.py -- STEP 6: feature selection, 66+ -> 30.

Input   C:\\MSCAD2022\\outputs\\step5\\   (Step 5's outputs)
            X_train.csv, X_test.csv     scaled with the TRAIN-fitted scaler
            y_train.csv, y_test.csv     Label, Label_encoded, Label_binary
Output  C:\\MSCAD2022\\outputs\\step6\\   (created if missing)

The three scores, and why all three
-----------------------------------
    6.2 MI       mutual information between each feature and Label_encoded.
                 A statistical dependency measure: how much knowing the feature
                 tells you about the class.  It looks at one feature at a time,
                 so two identical features both score highly.
    6.3 Pearson  |r| between features -- no labels involved at all.  This is the
                 part that removes REDUNDANCY: of two features carrying the same
                 information, one is enough.
    6.4 PI       permutation importance: how far the forest's macro-F1 falls
                 when one feature's column is shuffled.  This measures what a
                 model actually uses, in the presence of every other feature,
                 which is exactly what MI cannot see.
    6.5 combine  both scores min-max normalised to [0, 1] and added.

MI and PI disagree in a specific, known way, and that is the point of adding
them.  PI understates a feature whose partner is still in the set at |r| around
0.8-0.95: shuffle one and the forest simply leans on the other, so neither looks
important.  MI never sees other features, so it is immune to that.  Conversely MI
rewards a feature that merely correlates with the label without adding anything
beyond what other features already give; PI catches those.

What Step 6 deliberately does NOT do
------------------------------------
No autoencoder, no CNN-LSTM, no modelling of any kind.  No class-imbalance
handling: no SMOTE, no resampling.  The RandomForest in 6.4 exists ONLY to rank
features -- its score is not a result, for the reason given in `baseline_report`.

Nothing here touches the test set
---------------------------------
Every decision -- MI, the correlation matrix, the inner fit/validation split, the
permutation importance -- is computed on X_train alone.  X_test is read once, at
the very end, to take the same 30 columns in the same order.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from . import data_loading as dl  # noqa: E402
from . import splitting as sp  # noqa: E402

# the display helpers from Step 2/3, so Step 6 renders the same way
show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

LABEL_COLUMNS = ["Label", "Label_encoded", "Label_binary"]
TARGET_COLUMN = "Label_encoded"

# Columns that are already 0/1.  mutual_info_classif needs to be told, or it
# treats them as continuous and its k-nearest-neighbour estimate is wrong.
def discrete_feature_names() -> list[str]:
    return list(config.COMBINED_DROP_FEATURE_COLUMNS) + [config.ZERO_DURATION_COLUMN]


class SelectionError(Exception):
    """Raised when Step 6 cannot continue without guessing."""


def _require_sklearn(what: str):
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise SelectionError(
            'scikit-learn is needed for %s.  Install it with:\n'
            '    "%s" -m pip install -r requirements.txt' % (what, sys.executable)
        ) from exc


# ===========================================================================
# 6.1  INPUT
# ===========================================================================
def load_step5_outputs(verbose: bool = True) -> dict[str, Any]:
    """Read X_train / X_test / y_train / y_test from the Step 5 folder."""
    folder = config.usable_output_dir(config.STEP5_OUTPUT_DIR)
    if folder is None:
        raise SelectionError("the Step 5 output folder is not usable here -- %s"
                             % config.why_unusable(config.STEP5_OUTPUT_DIR))

    names = config.STEP5_FILES
    wanted = ("X_train", "X_test", "y_train", "y_test")
    missing = [names[k] for k in wanted if not (folder / names[k]).exists()]
    if missing:
        raise SelectionError(
            "Step 6 needs %s in %s. Run Step 5 first (set OVERWRITE_STEP5 = True "
            "if its outputs are stale)." % (missing, folder))

    out: dict[str, Any] = {"folder": str(folder)}
    for key in wanted:
        out[key] = pd.read_csv(folder / names[key], low_memory=False)
    return out


def check_inputs(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.DataFrame, y_test: pd.DataFrame,
                 verbose: bool = True):
    """[OK]/[FAIL] on everything that must hold before selection starts."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    same_cols = list(X_train.columns) == list(X_test.columns)
    add("X_train and X_test have identical columns, in order", same_cols,
        "%d columns" % X_train.shape[1] if same_cols
        else "X_train %d vs X_test %d" % (X_train.shape[1], X_test.shape[1]))

    rows_ok = len(X_train) == len(y_train) and len(X_test) == len(y_test)
    add("features and labels have matching row counts", rows_ok,
        "train %s/%s, test %s/%s" % (f"{len(X_train):,}", f"{len(y_train):,}",
                                     f"{len(X_test):,}", f"{len(y_test):,}"))

    leaked_labels = [c for c in LABEL_COLUMNS if c in X_train.columns]
    add("no label column is among the features", not leaked_labels,
        "Label, Label_encoded and Label_binary are in y_train only"
        if not leaked_labels else "found %s" % leaked_labels)

    identifiers = list(config.ID_COLUMNS_NOT_FOR_TRAINING) + \
        ["Source_Capture", "Orig_Row", "Timestamp_dt"]
    leaked_ids = [c for c in identifiers if c in X_train.columns]
    add("no identifier column is among the features", not leaked_ids,
        "the attacker IPs decided the labels, so they must never be features"
        if not leaked_ids else "found %s" % leaked_ids)

    has_target = TARGET_COLUMN in y_train.columns
    add("the target column `%s` is present" % TARGET_COLUMN, has_target,
        ", ".join(y_train.columns))

    # Only the numeric columns: a non-numeric one is already a [FAIL] above (it
    # can only be a leaked identifier), and coercing the whole frame here would
    # raise a ValueError about parsing an IP address instead of saying that.
    numeric = X_train.select_dtypes(include=[np.number])
    non_numeric = [c for c in X_train.columns if c not in numeric.columns]
    add("every feature column is numeric", not non_numeric,
        "%d columns" % X_train.shape[1] if not non_numeric
        else "not numeric: %s" % non_numeric)
    block = numeric.to_numpy(dtype="float64")
    n_nan, n_inf = int(np.isnan(block).sum()), int(np.isinf(block).sum())
    add("no NaN or inf in X_train", n_nan == 0 and n_inf == 0,
        "%d NaN, %d inf" % (n_nan, n_inf))

    rare: dict[str, int] = {}
    if has_target:
        counts = y_train[TARGET_COLUMN].value_counts()
        rare = {str(k): int(v) for k, v in counts.items() if v < 2}
        add("every class has at least 2 training rows", not rare,
            "%d classes, smallest %s rows"
            % (len(counts), f"{int(counts.min()):,}") if not rare
            else "too few rows: %s" % rare)

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table)

    # Ordered by how specific the diagnosis is, not by the order of the checks.
    # A leaked column also makes the two frames disagree on their columns, and
    # "Src IP is in the features" sends you to the right place; "the columns
    # differ" sends you hunting.
    if leaked_labels:
        raise SelectionError(
            "[FAIL] %s is in the feature frame. Selecting features against a target "
            "that is itself a feature would rank it first and mean nothing."
            % leaked_labels)
    if leaked_ids:
        raise SelectionError(
            "[FAIL] %s is in the feature frame. The attacker IPs decided the labels "
            "in Step 1, so an identifier would be selected immediately and leak the "
            "answer into every later model." % leaked_ids)
    if non_numeric:
        raise SelectionError(
            "[FAIL] these feature columns are not numeric, so they cannot be scored: "
            "%s. Step 5 writes only numeric features; re-run it with "
            "OVERWRITE_STEP5 = True." % non_numeric)
    if not same_cols:
        raise SelectionError(
            "[FAIL] X_train and X_test do not have the same columns in the same "
            "order, so the same features cannot be taken from both. Re-run Step 5 "
            "with OVERWRITE_STEP5 = True.")
    if not has_target:
        raise SelectionError(
            "[FAIL] y_train has no `%s` column; got %s"
            % (TARGET_COLUMN, list(y_train.columns)))
    if rare:
        raise SelectionError(
            "[FAIL] these classes have fewer than 2 training rows, so the inner "
            "fit/validation split in 6.4 cannot be stratified: %s" % rare)
    return table, all_ok


def class_counts_table(y_train: pd.DataFrame,
                       verbose: bool = True) -> pd.DataFrame:
    """The training class distribution Step 6 is selecting against."""
    counts = y_train[config.LABEL_COLUMN].value_counts()
    table = pd.DataFrame({
        config.LABEL_COLUMN: counts.index.astype(str),
        "train rows": counts.to_numpy(),
        "% of train": (100.0 * counts / len(y_train)).round(3).to_numpy(),
    })
    table = pd.concat([table, pd.DataFrame([{
        config.LABEL_COLUMN: "TOTAL", "train rows": len(y_train),
        "% of train": 100.0}])], ignore_index=True)
    if verbose:
        show_table(table, note="Selection is scored with **macro**-F1, which weights "
                               "every class equally, so a feature only the smallest "
                               "classes need still shows up.")
    return table


def drop_constant_columns(X_train: pd.DataFrame, X_test: pd.DataFrame,
                          verbose: bool = True):
    """Remove columns with no variance in TRAIN.  Pearson is undefined for them.

    A constant column also carries no information about the label, so nothing is
    lost -- but it would produce a NaN row and column in the correlation matrix
    and quietly poison the redundancy filter.
    """
    nunique = X_train.nunique(dropna=False)
    constant = [c for c in X_train.columns if nunique[c] <= 1]
    kept = [c for c in X_train.columns if c not in constant]

    if verbose:
        if constant:
            show_table(pd.DataFrame({
                "dropped (constant in train)": constant,
                "its single value": [X_train[c].iloc[0] for c in constant]}),
                note="Pearson correlation is undefined for a constant column, and a "
                     "constant tells a model nothing, so these go before 6.2.")
        else:
            show_note("No constant column in X_train -- nothing dropped here.")
    return X_train[kept].copy(), X_test[kept].copy(), constant


# ===========================================================================
# 6.2  MUTUAL INFORMATION -- computed ONCE, reused in 6.3 and 6.5
# ===========================================================================
def mutual_information(X_train: pd.DataFrame, y_train: pd.DataFrame,
                       verbose: bool = True):
    """MI between every feature and Label_encoded.  Train only.

    Called exactly once per Step 6 run.  The returned Series is threaded through
    the Pearson tie-break (6.3) and the combined score (6.5) rather than being
    recomputed: mutual_info_classif is stochastic (it jitters continuous inputs),
    so a second call would give slightly different numbers and the tie-break and
    the ranking would silently disagree.
    """
    _require_sklearn("mutual information")
    from sklearn.feature_selection import mutual_info_classif

    discrete = [c in set(discrete_feature_names()) for c in X_train.columns]
    started = time.perf_counter()
    scores = mutual_info_classif(
        X_train.to_numpy(dtype="float64"),
        y_train[TARGET_COLUMN].to_numpy(),
        discrete_features=np.asarray(discrete, dtype=bool),
        n_neighbors=config.MI_N_NEIGHBORS,
        random_state=config.MI_RANDOM_STATE)
    elapsed = time.perf_counter() - started

    mi = pd.Series(scores, index=list(X_train.columns), name="MI")
    table = (mi.sort_values(ascending=False).rename_axis("feature")
             .reset_index().rename(columns={"MI": "mutual_information"}))
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    if verbose:
        n_discrete = int(np.sum(discrete))
        show_table(table.head(20), note="Top 20 of %d features by mutual information "
                                        "with `%s`. Computed on the TRAINING rows "
                                        "only, once, in %.1f s. %d column(s) were "
                                        "declared discrete (the one-hot and binary "
                                        "markers); treating them as continuous would "
                                        "give a wrong k-nearest-neighbour estimate."
                                        % (len(table), TARGET_COLUMN, elapsed,
                                           n_discrete))
    return mi, table, elapsed


# ===========================================================================
# 6.3  PEARSON REDUNDANCY FILTER -- no labels involved
# ===========================================================================
def pearson_filter(X_train: pd.DataFrame, mi: pd.Series,
                   threshold: float | None = None,
                   report_lower: float | None = None, verbose: bool = True):
    """Drop one of every pair of features with |r| >= threshold.

    Deterministic by construction: the pairs are walked from the highest |r|
    downward, a pair is skipped when either side has already gone, and of the two
    the one with the LOWER MI is dropped.  MI ties are broken by feature_list
    order, so the result never depends on dict or set iteration order.
    """
    threshold = config.PEARSON_THRESHOLD if threshold is None else threshold
    report_lower = (config.PEARSON_REPORT_LOWER if report_lower is None
                    else report_lower)

    corr = X_train.corr(method="pearson").abs()
    order = {name: i for i, name in enumerate(X_train.columns)}

    # the upper triangle, as (|r|, a, b) rows -- each pair exactly once
    values = corr.to_numpy()
    iu = np.triu_indices_from(values, k=1)
    names = list(X_train.columns)
    pairs = [(float(values[i, j]), names[i], names[j])
             for i, j in zip(*iu) if np.isfinite(values[i, j])]
    # highest |r| first; ties resolved by column order so the walk is stable
    pairs.sort(key=lambda t: (-t[0], order[t[1]], order[t[2]]))

    dropped: dict[str, dict[str, Any]] = {}
    rows = []
    for r, a, b in pairs:
        if r < threshold:
            break
        if a in dropped or b in dropped:
            continue
        # keep the higher MI; on a tie keep whichever comes first in feature order
        mi_a, mi_b = float(mi.get(a, 0.0)), float(mi.get(b, 0.0))
        if mi_a > mi_b or (mi_a == mi_b and order[a] < order[b]):
            keep, drop = a, b
        else:
            keep, drop = b, a
        dropped[drop] = {"kept": keep, "r": r}
        rows.append({"kept": keep, "dropped": drop, "abs_r": round(r, 6),
                     "MI kept": round(float(mi.get(keep, 0.0)), 6),
                     "MI dropped": round(float(mi.get(drop, 0.0)), 6)})

    kept_features = [c for c in X_train.columns if c not in dropped]
    dropped_table = pd.DataFrame(
        rows, columns=["kept", "dropped", "abs_r", "MI kept", "MI dropped"])

    # pairs that survive and are still worth knowing about
    remaining = [{"feature A": a, "feature B": b, "abs_r": round(r, 6)}
                 for r, a, b in pairs
                 if report_lower <= r < threshold
                 and a in set(kept_features) and b in set(kept_features)]
    remaining_table = pd.DataFrame(
        remaining, columns=["feature A", "feature B", "abs_r"])

    if verbose:
        show_table(kv_table([
            ("threshold |r|", ">= %.2f" % threshold),
            ("features in", X_train.shape[1]),
            ("pairs over the threshold", len(rows)),
            ("features dropped", len(dropped)),
            ("features out", len(kept_features)),
        ]), note="No labels are used here. This step removes REDUNDANCY between "
                 "features; 6.2 and 6.4 are what measure usefulness.")
        if len(dropped_table):
            show_table(dropped_table,
                       note="Of each pair the higher-MI feature was kept. `MI kept` "
                            "is always >= `MI dropped`.")
        else:
            show_note("No pair reached |r| >= %.2f -- nothing dropped." % threshold)
        if len(remaining_table):
            show_table(remaining_table.head(25),
                       note="**Kept, but worth knowing:** these pairs sit in "
                            "[%.2f, %.2f). They are not redundant enough to drop, but "
                            "permutation importance UNDERSTATES both halves -- shuffle "
                            "one and the forest leans on the other. That is why the "
                            "MI score is added to PI rather than replaced by it."
                            % (report_lower, threshold))
        else:
            show_note("No remaining pair in [%.2f, %.2f)." % (report_lower, threshold))
    return kept_features, dropped_table, corr, remaining_table


# ===========================================================================
# 6.4  PERMUTATION IMPORTANCE -- an INNER split of the training set
# ===========================================================================
def duplicate_overlap(fit_block: pd.DataFrame, val_block: pd.DataFrame,
                      val_labels: pd.Series, verbose: bool = True):
    """Validation rows whose exact feature vector also appears in the fit part.

    The same measurement Step 5.3 makes across the real split, repeated here for
    the INNER split, because it is what makes the baseline macro-F1 below
    unusable as a model result.
    """
    keys_fit = set(sp._row_keys(fit_block).tolist())
    keys_val = sp._row_keys(val_block)
    mask = np.array([k in keys_fit for k in keys_val], dtype=bool)

    n = len(val_block)
    overall = {"validation rows": n, "also in fit": int(mask.sum()),
               "% of validation": round(100.0 * float(mask.sum()) / n, 3) if n else 0.0}

    rows = []
    labels = np.asarray(val_labels)
    for name in pd.Series(labels).value_counts().index:
        sel = labels == name
        total = int(sel.sum())
        hit = int(mask[sel].sum())
        rows.append({config.LABEL_COLUMN: name, "validation rows": total,
                     "also in fit": hit,
                     "% of this class": round(100.0 * hit / total, 3) if total else 0.0})
    table = pd.DataFrame(rows)
    table = pd.concat([table, pd.DataFrame([{
        config.LABEL_COLUMN: "TOTAL", "validation rows": n,
        "also in fit": overall["also in fit"],
        "% of this class": overall["% of validation"]}])], ignore_index=True)

    if verbose:
        show_table(table, note="Exact matches on the full feature vector, inside the "
                               "INNER split. Nothing is dropped.")
    return table, overall, mask


def permutation_scores(X_train: pd.DataFrame, y_train: pd.DataFrame,
                       features: Sequence[str], verbose: bool = True):
    """Fit a forest on part of TRAIN, permute on the rest.  Train only.

    Returns (pi_table, forest, info).  The Step 5 test set is never touched.
    """
    _require_sklearn("permutation importance")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    features = list(features)
    X = X_train[features]
    y = y_train[TARGET_COLUMN].to_numpy()
    names = y_train[config.LABEL_COLUMN].to_numpy()

    counts = pd.Series(y).value_counts()
    if int(counts.min()) < 2:
        raise SelectionError(
            "the inner fit/validation split cannot be stratified: these classes have "
            "fewer than 2 training rows: %s"
            % {str(k): int(v) for k, v in counts.items() if v < 2})

    positions = np.arange(len(X), dtype="int64")
    fit_idx, val_idx = train_test_split(
        positions, test_size=config.PI_VALIDATION_SIZE,
        random_state=config.RANDOM_STATE, shuffle=True, stratify=y)

    X_fit, X_val = X.iloc[fit_idx], X.iloc[val_idx]
    y_fit, y_val = y[fit_idx], y[val_idx]

    # a class that lands in validation with fewer than 2 rows gives a macro-F1
    # term computed from one example -- usable for ranking, but say so
    val_counts = pd.Series(names[val_idx]).value_counts()
    thin = {str(k): int(v) for k, v in val_counts.items() if v < 2}
    missing = [str(n) for n in pd.unique(names) if n not in set(val_counts.index)]

    forest = RandomForestClassifier(**config.RF_PARAMS)
    started = time.perf_counter()
    forest.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started

    predicted = forest.predict(X_val)
    baseline = float(f1_score(y_val, predicted, average="macro"))

    started = time.perf_counter()
    result = permutation_importance(
        forest, X_val, y_val, scoring=config.PI_SCORING,
        n_repeats=config.PI_N_REPEATS, random_state=config.RANDOM_STATE,
        n_jobs=config.RF_PARAMS.get("n_jobs", None))
    pi_seconds = time.perf_counter() - started

    pi_table = pd.DataFrame({
        "feature": features,
        "pi_mean": result.importances_mean,
        "pi_std": result.importances_std,
    }).sort_values(["pi_mean", "feature"], ascending=[False, True]).reset_index(drop=True)

    # per-class F1 on the validation part, by NAME rather than encoded number
    per_class = f1_score(y_val, predicted, average=None,
                         labels=np.unique(y_val))
    code_to_name = {int(code): str(name) for code, name in zip(y, names)}
    per_class_table = pd.DataFrame({
        config.LABEL_COLUMN: [code_to_name.get(int(c), str(c))
                              for c in np.unique(y_val)],
        "validation rows": [int((y_val == c).sum()) for c in np.unique(y_val)],
        "F1": np.round(per_class, 6),
    }).sort_values("validation rows", ascending=False).reset_index(drop=True)

    overlap_table, overlap, _mask = duplicate_overlap(
        X_fit, X_val, names[val_idx], verbose=False)

    info = {"baseline_macro_f1": baseline, "fit_rows": len(fit_idx),
            "validation_rows": len(val_idx), "fit_seconds": fit_seconds,
            "pi_seconds": pi_seconds, "thin_validation_classes": thin,
            "classes_missing_from_validation": missing,
            "per_class_f1": per_class_table, "overlap": overlap,
            "overlap_table": overlap_table}

    if verbose:
        show_table(kv_table([
            ("features ranked", len(features)),
            ("inner fit rows", f"{len(fit_idx):,}"),
            ("inner validation rows", f"{len(val_idx):,}"),
            ("forest", "RandomForest(n_estimators=%s, class_weight=%r)"
             % (config.RF_PARAMS.get("n_estimators"),
                config.RF_PARAMS.get("class_weight"))),
            ("fit time", "%.1f s" % fit_seconds),
            ("scoring", config.PI_SCORING),
            ("n_repeats", config.PI_N_REPEATS),
            ("permutation time", "%.1f s" % pi_seconds),
            ("baseline macro-F1", "%.4f" % baseline),
        ]))
        show_table(per_class_table, note="Per-class F1 on the inner validation part.")
        if missing:
            show_note("**[WARN]** these classes have NO validation rows, so they "
                      "contribute nothing to the ranking: %s" % ", ".join(missing))
        if thin:
            show_note("**[WARN]** these classes have fewer than 2 validation rows, so "
                      "their F1 term comes from a single example: %s"
                      % ", ".join("%s=%s" % kv for kv in thin.items()))
        show_table(overlap_table,
                   note="Validation rows whose exact feature vector also appears in "
                        "the fit part.")
        show_note("**The baseline macro-F1 above is a ranking tool, not a model "
                  "result.** %s of %s validation rows (%.2f%%) are feature-identical "
                  "to a fit row, so the forest is scored partly on rows it has "
                  "already seen. Quote Step 7's numbers, never this one."
                  % (f"{overlap['also in fit']:,}", f"{overlap['validation rows']:,}",
                     overlap["% of validation"]))
        show_table(pi_table.head(20),
                   note="Top 20 by permutation importance: the drop in macro-F1 when "
                        "that one column is shuffled. A feature whose partner is still "
                        "in the set (|r| 0.80-0.95) is UNDERSTATED here -- the forest "
                        "substitutes the partner -- which is why MI is added to it.")
    return pi_table, forest, info


# ===========================================================================
# 6.5  COMBINE
# ===========================================================================
def _min_max(values: np.ndarray, label: str, verbose: bool):
    """Scale to [0, 1].  A flat vector becomes all zeros, with a warning."""
    low, high = float(np.min(values)), float(np.max(values))
    if high == low:
        if verbose:
            show_note("**[WARN]** every %s score is %.6f, so there is nothing to "
                      "separate the features by. The normalised %s is 0 for all of "
                      "them and the ranking rests entirely on the other score."
                      % (label, low, label))
        return np.zeros_like(values, dtype="float64"), True
    return (values - low) / (high - low), False


def combine_scores(mi: pd.Series, pi_table: pd.DataFrame,
                   features: Sequence[str] | None = None,
                   clip_negative: bool | None = None, verbose: bool = True):
    """CombinedScore = MI' + PI', both min-max normalised over the SAME features.

    Sorted by CombinedScore descending; ties broken by MI' and then by feature
    name, so the order is fully determined and never depends on row order.
    """
    clip_negative = (config.CLIP_NEGATIVE_PI if clip_negative is None
                     else clip_negative)
    features = list(pi_table["feature"] if features is None else features)

    pi_lookup = dict(zip(pi_table["feature"], pi_table["pi_mean"]))
    mi_values = np.array([float(mi.get(f, 0.0)) for f in features], dtype="float64")
    pi_values = np.array([float(pi_lookup.get(f, 0.0)) for f in features],
                         dtype="float64")

    n_negative = int((pi_values < 0).sum())
    if clip_negative:
        pi_values = np.clip(pi_values, 0.0, None)

    mi_norm, mi_flat = _min_max(mi_values, "MI", verbose)
    pi_norm, pi_flat = _min_max(pi_values, "PI", verbose)
    combined = mi_norm + pi_norm

    table = pd.DataFrame({
        "feature": features, "MI": mi_values, "PI": pi_values,
        "MI_norm": mi_norm, "PI_norm": pi_norm, "CombinedScore": combined,
    })
    table = table.sort_values(["CombinedScore", "MI_norm", "feature"],
                              ascending=[False, False, True]).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    if verbose:
        if n_negative:
            show_note("%d feature(s) had a NEGATIVE permutation importance -- "
                      "shuffling them made the score slightly BETTER, which is noise "
                      "around zero, not evidence against them. %s"
                      % (n_negative,
                         "They were clipped to 0 before normalising, so that noise "
                         "cannot stretch the PI scale (CLIP_NEGATIVE_PI = True)."
                         if clip_negative else
                         "They were kept as-is (CLIP_NEGATIVE_PI = False)."))
        show_table(table.head(35),
                   note="Both scores are min-max normalised to [0, 1] over these %d "
                        "features and then ADDED, so each contributes at most 1.0 and "
                        "neither can dominate by living on a larger scale."
                        % len(features))
    return table, {"n_negative_pi": n_negative, "mi_flat": mi_flat,
                   "pi_flat": pi_flat, "clipped": bool(clip_negative)}


# ===========================================================================
# 6.6  SELECT
# ===========================================================================
def select_top(ranking: pd.DataFrame, n: int | None = None, verbose: bool = True):
    """The first n features of the ranking, or all of them if fewer remain."""
    n = config.N_SELECTED_FEATURES if n is None else n
    available = len(ranking)
    short = available < n
    taken = ranking.head(min(n, available))
    features = list(taken["feature"])

    out = ranking.copy()
    out["selected"] = out["feature"].isin(set(features))

    if verbose:
        if short:
            show_note("**[WARN] only %d features survived the Pearson filter, fewer "
                      "than the %d requested, so ALL %d were selected.** The filter "
                      "removed more than expected -- check "
                      "pearson_dropped_pairs.csv before using this set, and consider "
                      "raising PEARSON_THRESHOLD above %.2f."
                      % (available, n, available, config.PEARSON_THRESHOLD))
        else:
            cut = float(taken["CombinedScore"].iloc[-1])
            nxt = float(ranking["CombinedScore"].iloc[n]) if available > n else None
            show_table(kv_table([
                ("requested", n),
                ("available after Pearson", available),
                ("selected", len(features)),
                ("lowest selected CombinedScore", "%.4f (rank %d)" % (cut, len(features))),
                ("highest rejected CombinedScore",
                 "%.4f (rank %d)" % (nxt, n + 1) if nxt is not None else "-"),
                ("gap at the cut", "%.4f" % (cut - nxt) if nxt is not None else "-"),
            ]), note="A small gap at the cut means rank %d and rank %d are close to "
                     "interchangeable; it is not a sharp boundary." % (n, n + 1))
    return features, out, short


# ===========================================================================
# CHARTS
# ===========================================================================
# One hue per job.  |r| and the funnel counts are MAGNITUDE, so they use a single
# blue ramp light -> dark; MI' and PI' are two SERIES, so they get the first two
# categorical slots and a legend.  Checked with the palette validator.
_MI_COLOUR = "#2a78d6"          # categorical slot 1
_PI_COLOUR = "#eb6834"          # categorical slot 2
_SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
             "#0d366b"]         # blue 100 -> 700, for |r|
# Ordinal steps for the funnel, blue 250 -> 700.  The ramp is SAMPLED to the
# number of stages rather than cycled: a cycled ramp would give the last stage
# the lightest colour again and destroy the light -> dark reading.  Step 250 is
# the lightest allowed on a light surface (2.06:1).
_ORDINAL_STEPS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                  "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def ordinal_ramp(n: int) -> list[str]:
    """n steps of one hue, light -> dark, evenly spaced over _ORDINAL_STEPS."""
    if n <= 1:
        return [_ORDINAL_STEPS[-1]]
    last = len(_ORDINAL_STEPS) - 1
    return [_ORDINAL_STEPS[round(i * last / (n - 1))] for i in range(n)]


def _pyplot(verbose: bool):
    try:
        import matplotlib
        matplotlib.use("Agg")                      # no GUI needed, just a file
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        if verbose:
            print("  [SKIP]  matplotlib is not installed, so no chart was drawn.")
            print('             "%s" -m pip install matplotlib' % sys.executable)
        return None


def _chart_path(path, key: str, verbose: bool):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    folder = prepare_output_dir(config.STEP6_OUTPUT_DIR, "the Step 6 charts",
                               verbose=verbose)
    return None if folder is None else folder / config.STEP6_FILES[key]


def plot_pearson_heatmap(corr: pd.DataFrame, path=None, verbose: bool = True):
    """|r| between every pair of features.  Sequential: one hue, light -> dark."""
    plt = _pyplot(verbose)
    if plt is None:
        return None
    path = _chart_path(path, "heatmap", verbose)
    if path is None:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("mscad_blue", _SEQ_RAMP)
    n = corr.shape[0]
    size = max(7.0, 0.22 * n + 3.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.92), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    image = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=0.0, vmax=1.0,
                      interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    tick = 7 if n > 40 else 8
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=tick, color=dl._INK_SOFT)
    ax.set_yticklabels(corr.index, fontsize=tick, color=dl._INK_SOFT)
    ax.set_title("Absolute Pearson correlation between features (training rows, "
                 "%d features)" % n, color=dl._INK, fontsize=12, pad=14)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)

    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("|r|  (0 = unrelated, 1 = the same information)",
                  color=dl._INK_SOFT, fontsize=9)
    bar.ax.tick_params(colors=dl._INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def plot_combined_ranking(ranking: pd.DataFrame, n: int | None = None,
                          path=None, verbose: bool = True):
    """Top-n CombinedScore as MI' + PI', stacked.  Two series, so a legend."""
    plt = _pyplot(verbose)
    if plt is None:
        return None
    path = _chart_path(path, "ranking_chart", verbose)
    if path is None:
        return None

    n = config.N_SELECTED_FEATURES if n is None else n
    top = ranking.head(min(n, len(ranking))).iloc[::-1]       # best at the top
    labels = top["feature"].astype(str).to_numpy()
    mi = top["MI_norm"].to_numpy(dtype="float64")
    pi = top["PI_norm"].to_numpy(dtype="float64")
    positions = np.arange(len(labels), dtype="float64")

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.30 * len(labels) + 1.6)), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    height = 0.62
    ax.barh(positions, mi, height, label="MI' (dependency on the label)",
            color=_MI_COLOUR)
    # a sliver of surface between the two segments, per the mark spec
    gap = 0.004 * max(float((mi + pi).max()), 1e-9)
    ax.barh(positions, pi, height, left=mi + gap,
            label="PI' (drop in macro-F1 when shuffled)", color=_PI_COLOUR)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5, color=dl._INK_SOFT)
    ax.set_xlabel("CombinedScore = MI' + PI'   (each part 0-1, so 0-2 in total)",
                  color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Top %d features by combined score (training rows only)"
                 % len(labels), color=dl._INK, fontsize=12, pad=14)

    ax.xaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(dl._GRID)
    ax.tick_params(colors=dl._INK_SOFT, labelsize=9)
    ax.set_xlim(0, max(2.02, float((mi + pi).max()) * 1.12))

    for y, total in zip(positions, mi + pi):
        ax.annotate("%.2f" % total, (total, y), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=8, color=dl._INK)

    legend = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def plot_feature_funnel(counts: Sequence[tuple[str, int]], path=None,
                        verbose: bool = True):
    """How many features survive each stage.  One measure, so one hue, ordinal."""
    plt = _pyplot(verbose)
    if plt is None:
        return None
    path = _chart_path(path, "funnel", verbose)
    if path is None:
        return None

    stages = [str(name) for name, _ in counts][::-1]
    values = [int(value) for _, value in counts][::-1]
    # darkest at the END of the funnel, so the ramp reads with the narrowing
    colours = ordinal_ramp(len(values))[::-1]
    positions = np.arange(len(values), dtype="float64")

    fig, ax = plt.subplots(figsize=(9.5, 0.70 * len(values) + 1.8), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    ax.barh(positions, values, 0.6, color=colours)
    ax.set_yticks(positions)
    ax.set_yticklabels(stages, fontsize=10, color=dl._INK_SOFT)
    ax.set_xlabel("features", color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Features remaining at each stage of Step 6",
                 color=dl._INK, fontsize=12, pad=14)

    ax.xaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(dl._GRID)
    ax.tick_params(colors=dl._INK_SOFT, labelsize=9)
    ax.set_xlim(0, max(values) * 1.16 if values else 1)

    for y, value in zip(positions, values):
        ax.annotate("%d" % value, (value, y), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=10, color=dl._INK)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


# ===========================================================================
# SAVE and VERIFY
# ===========================================================================
def build_report(stages: Sequence[tuple[str, int]], mi_seconds: float,
                 pi_info: dict[str, Any], combine_info: dict[str, Any],
                 dropped_pairs: pd.DataFrame, short: bool) -> pd.DataFrame:
    """step6_report.csv -- one row per action, in the order they happened."""
    rows = [{"stage": name, "features": count} for name, count in stages]
    extra = [
        ("6.2 mutual information", "computed once on the training rows",
         "%.1f s, n_neighbors=%s, random_state=%s"
         % (mi_seconds, config.MI_N_NEIGHBORS, config.MI_RANDOM_STATE)),
        ("6.3 Pearson filter", "redundant features dropped",
         "%d pair(s) at |r| >= %.2f" % (len(dropped_pairs), config.PEARSON_THRESHOLD)),
        ("6.4 permutation importance", "RandomForest on an inner split of TRAIN",
         "fit %s rows, validate %s rows, scoring=%s, n_repeats=%d, %.1f s"
         % (f"{pi_info['fit_rows']:,}", f"{pi_info['validation_rows']:,}",
            config.PI_SCORING, config.PI_N_REPEATS, pi_info["pi_seconds"])),
        ("6.4 permutation importance", "baseline macro-F1 (a RANKING tool, not a result)",
         "%.4f -- %.2f%% of validation rows are feature-identical to a fit row"
         % (pi_info["baseline_macro_f1"], pi_info["overlap"]["% of validation"])),
        ("6.5 combine", "MI and PI min-max normalised, then added",
         "%d negative PI value(s) %s"
         % (combine_info["n_negative_pi"],
            "clipped to 0" if combine_info["clipped"] else "kept")),
        ("6.6 select", "top %d by CombinedScore" % config.N_SELECTED_FEATURES,
         "fewer available, so all were selected" if short else "exactly as requested"),
        ("6.x not done", "no class-imbalance handling, no modelling",
         "no SMOTE, no resampling, no class weights beyond the ranking forest"),
    ]
    table = pd.DataFrame(rows)
    table["action"] = ""
    table["detail"] = ""
    more = pd.DataFrame(extra, columns=["stage", "action", "detail"])
    more["features"] = ""
    return pd.concat([table[["stage", "features", "action", "detail"]],
                      more[["stage", "features", "action", "detail"]]],
                     ignore_index=True)


def save_step6(pieces: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Write every Step 6 output, then re-read each one and check it."""
    out: dict[str, Any] = {"folder": str(config.STEP6_OUTPUT_DIR), "files": {}}
    folder = prepare_output_dir(config.STEP6_OUTPUT_DIR, "the Step 6 outputs",
                               verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP6_FILES
    frames = {
        "mi_all": pieces["mi_table_all"], "mi_kept": pieces["mi_table_kept"],
        "corr_matrix": pieces["corr"], "dropped_pairs": pieces["dropped_pairs"],
        "remaining_pairs": pieces["remaining_pairs"],
        "pi_scores": pieces["pi_table"], "per_class_f1": pieces["per_class_f1"],
        "pi_overlap": pieces["overlap_table"], "ranking": pieces["ranking"],
        "X_train_selected": pieces["X_train_selected"],
        "X_test_selected": pieces["X_test_selected"],
        "y_train": pieces["y_train"], "y_test": pieces["y_test"],
        "report": pieces["report"],
    }
    for key, frame in frames.items():
        # the correlation matrix is the only frame whose index carries meaning
        frame.to_csv(folder / names[key], index=(key == "corr_matrix"))
        out["files"][key] = str(folder / names[key])

    (folder / names["selected"]).write_text(json.dumps({
        "features": list(pieces["selected"]),
        "n_selected": len(pieces["selected"]),
        "config": {
            "PEARSON_THRESHOLD": config.PEARSON_THRESHOLD,
            "PEARSON_REPORT_LOWER": config.PEARSON_REPORT_LOWER,
            "N_SELECTED_FEATURES": config.N_SELECTED_FEATURES,
            "RF_PARAMS": dict(config.RF_PARAMS),
            "PI_SCORING": config.PI_SCORING,
            "PI_N_REPEATS": config.PI_N_REPEATS,
            "PI_VALIDATION_SIZE": config.PI_VALIDATION_SIZE,
            "MI_N_NEIGHBORS": config.MI_N_NEIGHBORS,
            "MI_RANDOM_STATE": config.MI_RANDOM_STATE,
            "RANDOM_STATE": config.RANDOM_STATE,
            "CLIP_NEGATIVE_PI": config.CLIP_NEGATIVE_PI,
        },
        "selected_on": "the TRAINING rows only; X_test was used solely to take "
                       "the same columns",
    }, indent=2), encoding="utf-8")
    out["files"]["selected"] = str(folder / names["selected"])

    try:
        import joblib
        joblib.dump({"forest": pieces["forest"],
                     "rf_params": dict(config.RF_PARAMS),
                     "features": list(pieces["pearson_kept"]),
                     "baseline_macro_f1": pieces["pi_info"]["baseline_macro_f1"],
                     "scoring": config.PI_SCORING,
                     "note": "fitted on an inner split of TRAIN to RANK features. "
                             "Not a trained model for reporting."},
                    folder / names["rf_baseline"])
        out["files"]["rf_baseline"] = str(folder / names["rf_baseline"])
    except ImportError:
        if verbose:
            show_note("joblib is not installed, so the ranking forest was not saved "
                      "(`\"%s\" -m pip install joblib`). The CSVs are unaffected."
                      % sys.executable)

    for key, drawn in (("heatmap", plot_pearson_heatmap(pieces["corr"],
                                                        path=folder / names["heatmap"],
                                                        verbose=verbose)),
                       ("ranking_chart", plot_combined_ranking(
                           pieces["ranking"], path=folder / names["ranking_chart"],
                           verbose=verbose)),
                       ("funnel", plot_feature_funnel(
                           pieces["stages"], path=folder / names["funnel"],
                           verbose=verbose))):
        if drawn is not None:
            out["files"][key] = str(drawn)

    out["status"] = "written"
    verification, ok = verify_step6(pieces, folder=folder, verbose=verbose)
    out["verification"] = verification
    out["ok"] = ok
    return out


def verify_step6(pieces: dict[str, Any], folder: Path | None = None,
                 verbose: bool = True):
    """Re-read every Step 6 file from disk and check it.  [OK]/[FAIL]."""
    folder = Path(folder) if folder is not None else config.STEP6_OUTPUT_DIR
    names = config.STEP6_FILES
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    present = {key: (folder / names[key]).exists() for key in names
               if key not in ("heatmap", "ranking_chart", "funnel")}
    add("every expected file is on disk", all(present.values()),
        "%d of %d" % (sum(present.values()), len(present)) if all(present.values())
        else "missing %s" % [names[k] for k, v in present.items() if not v])

    selected = json.loads((folder / names["selected"]).read_text(encoding="utf-8"))
    wanted = list(selected["features"])
    X_train = pd.read_csv(folder / names["X_train_selected"], low_memory=False)
    X_test = pd.read_csv(folder / names["X_test_selected"], low_memory=False)
    y_train = pd.read_csv(folder / names["y_train"], low_memory=False)
    y_test = pd.read_csv(folder / names["y_test"], low_memory=False)

    expected_n = min(config.N_SELECTED_FEATURES, len(pieces["pearson_kept"]))
    add("exactly %d features were selected" % expected_n, len(wanted) == expected_n,
        "%d in selected_features.json" % len(wanted))
    add("X_train_selected has those columns, in that order",
        list(X_train.columns) == wanted, "%d columns" % X_train.shape[1])
    add("X_test_selected has the SAME columns, in the same order",
        list(X_test.columns) == wanted and list(X_test.columns) == list(X_train.columns),
        "%d columns" % X_test.shape[1])

    add("row counts still match Step 5",
        len(X_train) == pieces["n_train_rows"] and len(X_test) == pieces["n_test_rows"],
        "train %s, test %s" % (f"{len(X_train):,}", f"{len(X_test):,}"))
    add("labels are still aligned with the features",
        len(y_train) == len(X_train) and len(y_test) == len(X_test),
        "y_train %s, y_test %s" % (f"{len(y_train):,}", f"{len(y_test):,}"))
    add("the label columns were copied unchanged",
        list(y_train.columns) == LABEL_COLUMNS,
        ", ".join(y_train.columns))

    leaked = [c for c in LABEL_COLUMNS + list(config.ID_COLUMNS_NOT_FOR_TRAINING)
              if c in X_train.columns]
    add("no label or identifier column was selected", not leaked,
        "none" if not leaked else "found %s" % leaked)

    for name, frame in (("X_train_selected", X_train), ("X_test_selected", X_test)):
        block = frame.to_numpy(dtype="float64")
        n_nan, n_inf = int(np.isnan(block).sum()), int(np.isinf(block).sum())
        add("no NaN or inf in %s" % name, n_nan == 0 and n_inf == 0,
            "%d NaN, %d inf" % (n_nan, n_inf))

    ranking = pd.read_csv(folder / names["ranking"], low_memory=False)
    add("combined_ranking.csv marks exactly the selected features",
        sorted(ranking.loc[ranking["selected"], "feature"]) == sorted(wanted),
        "%d marked selected of %d ranked"
        % (int(ranking["selected"].sum()), len(ranking)))
    recomputed = (ranking["MI_norm"] + ranking["PI_norm"]).to_numpy()
    add("CombinedScore == MI_norm + PI_norm in the saved file",
        np.allclose(recomputed, ranking["CombinedScore"].to_numpy(), atol=1e-9),
        "%d rows checked" % len(ranking))
    in_unit = ((ranking["MI_norm"].between(-1e-9, 1 + 1e-9)).all()
               and (ranking["PI_norm"].between(-1e-9, 1 + 1e-9)).all())
    add("MI_norm and PI_norm are inside [0, 1]", bool(in_unit),
        "MI %.4f-%.4f, PI %.4f-%.4f"
        % (ranking["MI_norm"].min(), ranking["MI_norm"].max(),
           ranking["PI_norm"].min(), ranking["PI_norm"].max()))

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_title("Step 6 -- files written, then re-read from disk")
        show_table(table, note="Folder: %s" % folder)
        if not all_ok:
            show_note("**[FAIL]** -- the Step 6 outputs do not agree with each other; "
                      "do not train on them.")
    return table, all_ok


# ===========================================================================
# load / exist / run
# ===========================================================================
def step6_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP6_OUTPUT_DIR)
    if folder is None:
        return False
    keys = ("selected", "X_train_selected", "X_test_selected", "y_train", "y_test",
            "ranking", "mi_all", "pi_scores")
    return all((folder / config.STEP6_FILES[k]).exists() for k in keys)


def load_step6(verbose: bool = True) -> dict[str, Any]:
    """Read the saved Step 6 outputs back instead of reselecting."""
    folder = config.usable_output_dir(config.STEP6_OUTPUT_DIR)
    if folder is None:
        raise SelectionError("the Step 6 folder is not usable here -- %s"
                             % config.why_unusable(config.STEP6_OUTPUT_DIR))
    names = config.STEP6_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    for key in ("mi_all", "mi_kept", "dropped_pairs", "remaining_pairs", "pi_scores",
                "per_class_f1", "pi_overlap", "ranking", "X_train_selected",
                "X_test_selected", "y_train", "y_test", "report"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    corr_path = folder / names["corr_matrix"]
    if corr_path.exists():
        out["corr"] = pd.read_csv(corr_path, index_col=0, low_memory=False)
    picked = json.loads((folder / names["selected"]).read_text(encoding="utf-8"))
    out["selected"] = list(picked["features"])
    out["selected_config"] = picked.get("config", {})

    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP6 = False)"),
            ("folder", str(folder)),
            ("features selected", len(out["selected"])),
            ("X_train_selected", "%s rows x %d columns"
             % (f"{len(out['X_train_selected']):,}",
                out["X_train_selected"].shape[1])),
            ("X_test_selected", "%s rows x %d columns"
             % (f"{len(out['X_test_selected']):,}",
                out["X_test_selected"].shape[1])),
        ]), note="Set `OVERWRITE_STEP6 = True` in config.py to select again. "
                 "Permutation importance is the slow part, so this path is worth "
                 "keeping.")
    return out


def run_step6(overwrite: bool | None = None, verbose: bool = True) -> dict[str, Any]:
    """STEP 6 end to end: check -> MI -> Pearson -> PI -> combine -> top N."""
    overwrite = config.OVERWRITE_STEP6 if overwrite is None else overwrite
    if not overwrite and step6_outputs_exist():
        return load_step6(verbose=verbose)

    if verbose:
        show_title("Step 6.1 -- training data overview", level=3)
    loaded = load_step5_outputs(verbose=verbose)
    X_train, X_test = loaded["X_train"], loaded["X_test"]
    y_train, y_test = loaded["y_train"], loaded["y_test"]
    n_initial = X_train.shape[1]

    if verbose:
        show_table(kv_table([
            ("X_train", "%s rows x %d columns"
             % (f"{len(X_train):,}", X_train.shape[1])),
            ("X_test", "%s rows x %d columns"
             % (f"{len(X_test):,}", X_test.shape[1])),
            ("input features", n_initial),
            ("target", "%s (multiclass)" % TARGET_COLUMN),
            ("read from", loaded["folder"]),
        ]), note="X_test is read here and then left alone until 6.6. Every selection "
                 "decision is made on X_train.")
    check_table, _ok = check_inputs(X_train, X_test, y_train, y_test, verbose=verbose)
    classes = class_counts_table(y_train, verbose=verbose)
    X_train, X_test, constant = drop_constant_columns(X_train, X_test, verbose=verbose)
    n_after_constant = X_train.shape[1]

    if verbose:
        show_title("Step 6.2 -- mutual information (training rows only)", level=3)
    mi, mi_table_all, mi_seconds = mutual_information(X_train, y_train, verbose=verbose)

    if verbose:
        show_title("Step 6.3 -- Pearson redundancy filter (no labels used)", level=3)
    kept, dropped_pairs, corr, remaining_pairs = pearson_filter(
        X_train, mi, verbose=verbose)
    n_after_pearson = len(kept)
    mi_table_kept = (mi[kept].sort_values(ascending=False).rename_axis("feature")
                     .reset_index().rename(columns={"MI": "mutual_information"}))
    mi_table_kept.insert(0, "rank", np.arange(1, len(mi_table_kept) + 1))

    if verbose:
        show_title("Step 6.4 -- permutation importance (inner split of TRAIN)", level=3)
    pi_table, forest, pi_info = permutation_scores(X_train, y_train, kept,
                                                   verbose=verbose)

    if verbose:
        show_title("Step 6.5 -- combine MI and PI", level=3)
    ranking, combine_info = combine_scores(mi, pi_table, features=kept, verbose=verbose)

    if verbose:
        show_title("Step 6.6 -- select the top %d" % config.N_SELECTED_FEATURES,
                   level=3)
    selected, ranking, short = select_top(ranking, verbose=verbose)

    X_train_selected = X_train[selected].copy()
    X_test_selected = X_test[selected].copy()        # the ONLY use of X_test

    stages = [("initial", n_initial),
              ("after the constant drop", n_after_constant),
              ("after Pearson", n_after_pearson),
              ("selected", len(selected))]
    report = build_report(stages, mi_seconds, pi_info, combine_info,
                          dropped_pairs, short)

    pieces = {
        "mi": mi, "mi_table_all": mi_table_all, "mi_table_kept": mi_table_kept,
        "corr": corr, "dropped_pairs": dropped_pairs,
        "remaining_pairs": remaining_pairs, "pearson_kept": kept,
        "pi_table": pi_table, "forest": forest, "pi_info": pi_info,
        "per_class_f1": pi_info["per_class_f1"],
        "overlap_table": pi_info["overlap_table"],
        "ranking": ranking, "selected": selected, "short": short,
        "X_train_selected": X_train_selected, "X_test_selected": X_test_selected,
        "y_train": y_train, "y_test": y_test,
        "n_train_rows": len(X_train), "n_test_rows": len(X_test),
        "stages": stages, "report": report, "classes": classes,
        "constant_dropped": constant, "check_table": check_table,
        "combine_info": combine_info, "mi_seconds": mi_seconds,
    }

    saved = save_step6(pieces, verbose=verbose)

    if verbose:
        show_title("Step 6 -- summary", level=3)
        show_table(kv_table([
            ("features in", n_initial),
            ("constant, dropped", len(constant)),
            ("redundant, dropped by Pearson", n_after_constant - n_after_pearson),
            ("ranked", n_after_pearson),
            ("selected", len(selected)),
            ("X_train_selected", "%s rows x %d columns"
             % (f"{len(X_train_selected):,}", X_train_selected.shape[1])),
            ("X_test_selected", "%s rows x %d columns"
             % (f"{len(X_test_selected):,}", X_test_selected.shape[1])),
            ("MI time", "%.1f s" % mi_seconds),
            ("PI time", "%.1f s" % pi_info["pi_seconds"]),
            ("RF baseline macro-F1 (ranking tool only)",
             "%.4f" % pi_info["baseline_macro_f1"]),
            ("files verified", saved.get("ok")),
            ("output folder", saved.get("folder")),
        ]), note="The class imbalance is still untouched -- no SMOTE, no resampling. "
                 "Handle it in Step 7, fitted on the training side only.")

    out = dict(pieces)
    out.update({"status": saved.get("status", "written"), "saved": saved,
                "ok": bool(saved.get("ok"))})
    return out
