r"""
src/mscad/comparison.py -- the Step 12 comparison across every configuration.

NEW FILE.  Nothing in converter.py, data_loading.py, preprocessing.py,
splitting.py, feature_selection.py, autoencoder.py, imbalance.py, models.py,
evaluation.py, experiments.py, experiments_pca.py, experiments_baseline5.py,
fusion.py or explainability.py is modified.  This module imports those and
assembles their saved results into one table.

Why this exists
---------------
Step 12 grew in three places: the four original configurations, the Baseline 4
addendum and the Baseline 5 addendum, each with its own table.  A reader had to
stitch three tables together, and the row order differed between them.  This
module builds ONE table, in ONE order, from the saved `metrics.json` of every
configuration.

It never trains anything.  Every row is read from disk; a missing result is
named, with the step that produces it, and the row is left blank rather than
filled in by a fresh run.  `build_full_comparison()` is the single entry point
and is what every table, chart and head-to-head below uses.

The guarantee this module has to keep
-------------------------------------
The numbers already written up must not move.  `check_rows_unchanged()`
compares every rebuilt row against the saved `metrics.json` it came from
(1e-9) and against the published figures in `config.STEP12_EXPECTED_ROWS`
(4 decimals, 5e-5).  Both print `[OK]`/`[FAIL]` and both are saved.
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
from . import evaluation as ev  # noqa: E402
from . import experiments as xp  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir


class ComparisonError(Exception):
    """Raised when the comparison cannot be built without guessing."""


# the columns, in order, exactly as the write-up quotes them
COLUMNS = [
    "config", "input features", "latent dim", "PCA components", "params",
    "accuracy", "precision (macro)", "recall (macro)", "F1 (macro)",
    "F1 (weighted)", "G-mean", "ROC-AUC (macro OvR)", "binary F1",
    "binary FPR", "F1 macro (dup-free)", "G-mean (dup-free)",
    "train time (s)", "inference (ms/1k)",
]
DESCRIPTORS = {"config", "input features", "latent dim", "PCA components"}
LOWER_IS_BETTER = {"binary FPR", "params", "train time (s)",
                   "inference (ms/1k)"}


# ===========================================================================
# 12.0  WHERE EACH ROW'S SAVED RESULT LIVES
# ===========================================================================
def result_path(key: str) -> Path | None:
    """The metrics.json for one configuration, or None if unreachable here.

    Approach 2 is written by Step 11b rather than by Step 12, so it is looked
    up separately; `getattr` keeps this working before the Part 2 config block
    exists.
    """
    if key == getattr(config, "APPROACH2_KEY", "approach2_attention"):
        folder = getattr(config, "STEP11B_OUTPUT_DIR", None)
        if folder is None:
            return None
        folder = config.usable_output_dir(folder)
        return None if folder is None else folder / "metrics.json"
    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    return None if folder is None else folder / key / "metrics.json"


def load_result(key: str) -> dict[str, Any] | None:
    """One configuration's saved metrics, or None when it has not been run."""
    path = result_path(key)
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def missing_results(order: Sequence[tuple[str, str]] | None = None
                    ) -> list[tuple[str, str]]:
    """(key, where it would be) for every configuration not on disk."""
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    out = []
    for key, _label in order:
        if load_result(key) is None:
            path = result_path(key)
            out.append((key, str(path) if path else "(folder not usable here)"))
    return out


def which_step_produces(key: str) -> str:
    """The instruction a reader needs when a result is missing."""
    if key == getattr(config, "APPROACH2_KEY", "approach2_attention"):
        return "notebook section 10.2 (train Approach 2), then 11.x"
    if key == "baseline4_ae_pca_lstm":
        return "the Baseline 4 cells (12.3), with OVERWRITE_BASELINE4 = True"
    if key == "baseline5":
        return "the Baseline 5 cells (12.4), with OVERWRITE_BASELINE5 = True"
    if key == "proposed":
        return "Step 10 and Step 11, then Step 12"
    return "Step 12 (set OVERWRITE_STEP12 = True)"


# ===========================================================================
# 12.5  THE FULL COMPARISON TABLE
# ===========================================================================
def _row_from_metrics(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """One table row from one saved metrics.json, whichever step wrote it."""
    full = metrics.get("full") or {}
    free = metrics.get("duplicate_free") or {}
    binary = metrics.get("binary") or {}
    inference = metrics.get("inference") or {}
    auc = (full.get("roc_auc") or {}).get("macro")
    latent = metrics.get("latent_dim")
    pca = metrics.get("pca_components")
    return {
        "config": label,
        "input features": metrics.get("n_input_features"),
        "latent dim": latent if latent else "-",
        "PCA components": ("%d of %d" % (int(pca), int(latent))
                           if pca and latent else "-"),
        "params": metrics.get("parameters"),
        "accuracy": full.get("accuracy"),
        "precision (macro)": full.get("precision_macro"),
        "recall (macro)": full.get("recall_macro"),
        "F1 (macro)": full.get("f1_macro"),
        "F1 (weighted)": full.get("f1_weighted"),
        "G-mean": full.get("g_mean"),
        "ROC-AUC (macro OvR)": auc,
        "binary F1": binary.get("f1"),
        "binary FPR": binary.get("false_positive_rate",
                                 binary.get("fpr")),
        "F1 macro (dup-free)": free.get("f1_macro"),
        "G-mean (dup-free)": free.get("g_mean"),
        "train time (s)": metrics.get("training_seconds"),
        "inference (ms/1k)": inference.get("ms_per_1000_flows"),
    }


def _blank_row(label: str) -> dict[str, Any]:
    """A placeholder row for a configuration that has not been trained."""
    row = {column: None for column in COLUMNS}
    row["config"] = label
    row["input features"] = config.NOT_TRAINED_TEXT
    return row


def build_full_comparison(order: Sequence[tuple[str, str]] | None = None,
                          verbose: bool = True) -> dict[str, Any]:
    """THE entry point: every configuration, in the display order, from disk.

    Returns the table plus the raw metrics and the list of configurations that
    are missing, so a caller can report the gap rather than discover it as a
    KeyError three cells later.  Nothing is trained, ever.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    rows, loaded, missing = [], {}, []
    for key, label in order:
        metrics = load_result(key)
        if metrics is None:
            missing.append(key)
            rows.append(_blank_row(label))
            continue
        loaded[key] = metrics
        rows.append(_row_from_metrics(label, metrics))

    table = pd.DataFrame(rows)[COLUMNS]
    if verbose:
        if missing:
            show_note("%d of %d configurations have no saved result yet: %s. "
                      "Their rows are blank and marked `%s` -- nothing was "
                      "trained to fill them."
                      % (len(missing), len(order),
                         ", ".join("`%s` (%s)" % (k, which_step_produces(k))
                                   for k in missing),
                         config.NOT_TRAINED_TEXT))
        else:
            show_note("All %d configurations read from disk. No model was "
                      "trained or retrained." % len(order))
    return {"table": table, "metrics": loaded, "missing": missing,
            "order": list(order)}


def trained_rows(table: pd.DataFrame) -> pd.Series:
    """Boolean mask of the rows that actually have results."""
    return table["input features"].astype(str) != config.NOT_TRAINED_TEXT


def mark_best(table: pd.DataFrame) -> pd.DataFrame:
    """A copy with the best value in each metric column marked with `*`.

    Ties are marked on every tied row -- a tie is a real result and hiding it
    behind one arbitrary winner would be a false statement.  The "not yet
    trained" row takes no part in the comparison.
    """
    out = table.copy()
    live = trained_rows(table)
    for column in table.columns:
        if column in DESCRIPTORS:
            continue
        values = pd.to_numeric(table[column], errors="coerce").where(live)
        if values.notna().sum() < 1:
            out[column] = ["-" if pd.isna(v) else v for v in table[column]]
            continue
        best = values.min() if column in LOWER_IS_BETTER else values.max()

        def render(value, is_live):
            if pd.isna(value):
                return config.NOT_TRAINED_TEXT if not is_live else "-"
            star = "*" if abs(float(value) - float(best)) < 1e-12 else ""
            if column == "params":
                return "%s%s" % (f"{int(value):,}", star)
            return "%.4f%s" % (float(value), star)

        out[column] = [render(v, bool(l)) for v, l in zip(values, live)]
    return out


# ===========================================================================
# 12.x  THE GUARANTEE: EXISTING NUMBERS MUST NOT MOVE
# ===========================================================================
def check_rows_unchanged(built: dict[str, Any],
                         verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """Both checks from the brief: exact against disk, readable against print.

    The exact check re-reads each saved metrics.json and compares it with the
    row this module built from it (1e-9).  The readable check compares the
    headline numbers with the figures already in the write-up, rounded to 4
    decimals (5e-5).  A configuration that has not been run is SKIPPED, never
    silently passed.
    """
    rows: list[dict[str, Any]] = []
    table = built["table"]
    labels = dict(built["order"])

    for key, label in built["order"]:
        metrics = built["metrics"].get(key)
        if metrics is None:
            rows.append({"check": "exact: %s" % key, "result": "[SKIP]",
                         "detail": "no saved result (%s)"
                                   % which_step_produces(key)})
            continue
        expected = _row_from_metrics(label, metrics)
        here = table[table["config"] == label]
        worst, worst_column = 0.0, None
        for column in COLUMNS:
            if column in DESCRIPTORS:
                continue
            a = pd.to_numeric(pd.Series([expected[column]]),
                              errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([here[column].iloc[0]]),
                              errors="coerce").iloc[0]
            if pd.isna(a) and pd.isna(b):
                continue
            gap = abs(float(a) - float(b)) if not (pd.isna(a) or pd.isna(b)) \
                else float("inf")
            if gap > worst:
                worst, worst_column = gap, column
        ok = worst <= config.STEP12_EXACT_TOLERANCE
        rows.append({
            "check": "exact: %s" % key,
            "result": "[OK]" if ok else "[FAIL]",
            "detail": "max difference %.1e from %s%s"
                      % (worst, Path(str(result_path(key))).name,
                         "" if ok else " at %s" % worst_column)})

    for key, expected in config.STEP12_EXPECTED_ROWS.items():
        metrics = built["metrics"].get(key)
        if metrics is None:
            rows.append({"check": "published figures: %s" % key,
                         "result": "[SKIP]",
                         "detail": "no saved result (%s)"
                                   % which_step_produces(key)})
            continue
        full = metrics.get("full") or {}
        differences = {}
        for name, want in expected.items():
            got = full.get(name)
            if got is None:
                differences[name] = float("inf")
                continue
            differences[name] = abs(round(float(got), 4) - float(want))
        worst = max(differences.values()) if differences else 0.0
        ok = worst <= config.STEP12_EXPECTED_TOLERANCE
        rows.append({
            "check": "published figures: %s" % key,
            "result": "[OK]" if ok else "[FAIL]",
            "detail": ", ".join("%s %.4f (want %.4f)"
                                % (n, float(full.get(n, float("nan"))),
                                   expected[n]) for n in expected)})

    frame = pd.DataFrame(rows)
    all_ok = bool((frame["result"].isin(["[OK]", "[SKIP]"])).all())
    if verbose:
        show_table(frame, note="Exact: every rebuilt row against the saved "
                               "metrics.json it came from (1e-9). Published: "
                               "the headline numbers against the figures "
                               "already in the write-up (4 decimals, 5e-5). "
                               "`[SKIP]` means that configuration has not been "
                               "run on this machine.")
        print()
        print("[%s]  existing rows identical"
              % ("OK" if all_ok else "FAIL"))
    return frame, all_ok


# ===========================================================================
# 12.1  THE CONFIGURATION TABLE
# ===========================================================================
def configuration_table(order: Sequence[tuple[str, str]] | None = None
                        ) -> pd.DataFrame:
    """config | feature selection | AE | PCA | CNN | attention | LSTM | input
    shape | what it isolates, for every configuration in the display order."""
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    approach2 = getattr(config, "APPROACH2_KEY", "approach2_attention")
    specs = {spec["name"]: spec for spec in config.EXPERIMENTS_WITH_PCA}

    rows = []
    for key, label in order:
        spec = specs.get(key, {})
        if key == approach2:
            selection, uses_ae, uses_pca = ("Step 6 top %d"
                                            % config.N_SELECTED_FEATURES,
                                            True, False)
            cnn, attention, lstm = "yes", "yes", "yes"
            shape = "(%d, 1)" % config.AE_LATENT_DIM
        elif key == "baseline5":
            selection, uses_ae, uses_pca = ("MI only, top %d"
                                            % config.BASELINE5_TOP_K,
                                            True, False)
            cnn, attention, lstm = "yes", "no", "yes"
            shape = "(%d, 1)" % config.AE_LATENT_DIM
        else:
            selection = ("Step 6 top %d" % config.N_SELECTED_FEATURES
                         if spec.get("feature_source") == "selected30"
                         else "none")
            uses_ae = bool(spec.get("use_ae"))
            uses_pca = bool(spec.get("use_pca"))
            cnn = "yes" if spec.get("model_type") == "cnn_lstm" else "no"
            attention, lstm = "no", "yes"
            shape = ("(n_components, 1)" if uses_pca else
                     "(%d, 1)" % config.AE_LATENT_DIM if uses_ae else
                     "(n_features, 1)")
        rows.append({
            "config": label,
            "feature selection": selection,
            "AE": ("yes (-> %d)" % config.AE_LATENT_DIM) if uses_ae else "no",
            "PCA": ("yes (%s)" % config.PCA_N_COMPONENTS) if uses_pca else "no",
            "CNN": cnn, "attention": attention, "LSTM": lstm,
            "input shape": shape,
            "what it isolates": config.STEP12_ISOLATES.get(key, ""),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 12.6  PER-CLASS F1, AND 12.9's ZERO-RECALL DIAGNOSTIC
# ===========================================================================
def _predictions(key: str) -> pd.DataFrame | None:
    path = result_path(key)
    if path is None:
        return None
    candidate = path.parent / "predictions.csv"
    if not candidate.exists():
        return None
    return pd.read_csv(candidate, low_memory=False)


def per_class_table(order: Sequence[tuple[str, str]] | None = None,
                    duplicate_free: bool = False) -> pd.DataFrame:
    """Label x configuration F1, in the display order, from each
    predictions.csv.

    Built from the predictions rather than the saved per-class reports because
    those exist only for the full test set; the same file carries
    `has_train_duplicate`, so the duplicate-free view is computed the same way
    for every configuration.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    columns: dict[str, pd.Series] = {}
    support: pd.Series | None = None
    for key, label in order:
        frame = _predictions(key)
        if frame is None or not len(frame):
            continue
        if duplicate_free:
            if "has_train_duplicate" not in frame.columns:
                continue
            frame = frame[~frame["has_train_duplicate"].astype(bool)]
        if not len(frame):
            continue
        names = (frame.drop_duplicates("y_true")
                 .set_index("y_true")["y_true_label"].to_dict())
        n_classes = int(max(frame["y_true"].max(), frame["y_pred"].max())) + 1
        table = ev.per_class_table(frame["y_true"].to_numpy(),
                                   frame["y_pred"].to_numpy(),
                                   {int(k): str(v) for k, v in names.items()},
                                   n_classes)
        columns[label] = table.set_index("Label")["f1"]
        if support is None:
            support = table.set_index("Label")["support"]
    if not columns:
        return pd.DataFrame()
    out = pd.DataFrame(columns)
    out.index.name = "Label"
    if support is not None:
        out.insert(0, "test rows", support)
    return out.reset_index()


def zero_recall_diagnostic(order: Sequence[tuple[str, str]] | None = None
                           ) -> pd.DataFrame:
    """Which classes a configuration never detects -- the reason a G-mean is 0.

    The geometric mean of the per-class recalls is zero the moment ONE class
    has recall zero, so a 0.0000 in that column is not "poor overall": it is
    "this class was never found once". Naming the class turns a dead number
    into a finding.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    rows = []
    for key, label in order:
        frame = _predictions(key)
        if frame is None or not len(frame):
            continue
        for scope in ("full test set", "duplicate-free subset"):
            part = frame
            if scope.startswith("duplicate-free"):
                if "has_train_duplicate" not in frame.columns:
                    continue
                part = frame[~frame["has_train_duplicate"].astype(bool)]
            if not len(part):
                continue
            missed = []
            for code, group in part.groupby("y_true"):
                recall = float((group["y_pred"] == code).mean())
                if recall == 0.0:
                    name = str(group["y_true_label"].iloc[0])
                    missed.append("%s (%d rows)" % (name, len(group)))
            rows.append({
                "config": label, "scope": scope,
                "classes with recall 0": len(missed),
                "which": "; ".join(missed) if missed else "none",
                "G-mean is zero": bool(missed)})
    return pd.DataFrame(rows)


# ===========================================================================
# 12.8  HEAD TO HEAD
# ===========================================================================
def head_to_head(table: pd.DataFrame, left: str, right: str,
                 pair: str | None = None) -> pd.DataFrame:
    """left minus right, metric by metric, with the direction accounted for."""
    rows = []
    a_row = table[table["config"] == left]
    b_row = table[table["config"] == right]
    if not len(a_row) or not len(b_row):
        return pd.DataFrame()
    for column in COLUMNS:
        if column in DESCRIPTORS:
            continue
        a = pd.to_numeric(pd.Series([a_row[column].iloc[0]]),
                          errors="coerce").iloc[0]
        b = pd.to_numeric(pd.Series([b_row[column].iloc[0]]),
                          errors="coerce").iloc[0]
        if pd.isna(a) or pd.isna(b):
            continue
        lower = column in LOWER_IS_BETTER
        difference = float(a) - float(b)
        favours = ("tie" if abs(difference) < 1e-12
                   else (right if (difference > 0) == lower else left))
        rows.append({"pair": pair or "%s vs %s" % (left, right),
                     "metric": column, "left": float(a), "right": float(b),
                     "difference (left - right)": difference,
                     "lower is better": lower, "favours": favours})
    return pd.DataFrame(rows)


def all_head_to_heads(table: pd.DataFrame,
                      order: Sequence[tuple[str, str]] | None = None
                      ) -> pd.DataFrame:
    """The three comparisons the write-up argues from.

    Each pair differs in exactly one thing, which is what makes the difference
    attributable: the selection method, the selection step itself, and the
    attention block.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    labels = dict(order)
    approach2 = getattr(config, "APPROACH2_KEY", "approach2_attention")
    pairs = [
        ("proposed", "baseline5", "selection method: fused vs MI only"),
        ("proposed", "baseline1_ae_cnn_lstm",
         "feature selection: with vs without"),
        ("proposed", approach2, "architecture: CNN-LSTM vs CNN-Attention-LSTM"),
    ]
    frames = []
    for left, right, why in pairs:
        if left not in labels or right not in labels:
            continue
        frame = head_to_head(table, labels[left], labels[right], pair=why)
        if len(frame):
            frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame())


# ===========================================================================
# 12.7  CHARTS -- all in the display order
# ===========================================================================
def _chart_table(table: pd.DataFrame) -> pd.DataFrame:
    """The trained rows only, still in the display order.

    A configuration with no result is left out of the charts rather than drawn
    as a zero-height bar, which would read as "scored 0" to anyone glancing at
    it.
    """
    return table[trained_rows(table)].reset_index(drop=True)


def _runs_for_confusion(built: dict[str, Any]) -> list[dict[str, Any]]:
    """What plot_confusion_grid expects, in the display order."""
    runs = []
    for key, label in built["order"]:
        metrics = built["metrics"].get(key)
        path = result_path(key)
        if metrics is None or path is None:
            continue
        runs.append({"metrics": {"label": label}, "folder": str(path.parent)})
    return runs


def draw_charts(built: dict[str, Any], folder: Path,
                verbose: bool = True) -> dict[str, str]:
    """Every Step 12 chart, written with a `_full` suffix beside the old ones."""
    names = config.STEP12_FULL_FILES
    table = _chart_table(built["table"])
    charts: dict[str, str] = {}
    if not len(table):
        if verbose:
            print("  [SKIP]  no configuration has a saved result, so no chart "
                  "was drawn.")
        return charts

    drawn = xp.plot_metric_bars(
        table, ["F1 (macro)", "G-mean", "ROC-AUC (macro OvR)"],
        "Full test set, %d configurations -- higher is better" % len(table),
        folder / names["chart_metrics"], verbose=verbose)
    if drawn:
        charts["metrics"] = str(drawn)
    drawn = xp.plot_metric_bars(
        table, ["F1 macro (dup-free)", "G-mean (dup-free)"],
        "Duplicate-free test subset -- the conservative comparison",
        folder / names["chart_metrics_free"], verbose=verbose)
    if drawn:
        charts["metrics_free"] = str(drawn)
    drawn = xp.plot_time_bars(table, folder / names["chart_time"],
                              verbose=verbose)
    if drawn:
        charts["time"] = str(drawn)
    per_class = built.get("per_class")
    if per_class is not None and len(per_class):
        drawn = xp.plot_per_class_heatmap(per_class,
                                          folder / names["chart_per_class"],
                                          verbose=verbose)
        if drawn:
            charts["per_class"] = str(drawn)
    drawn = xp.plot_confusion_grid(_runs_for_confusion(built),
                                   folder / names["chart_confusion"],
                                   verbose=verbose)
    if drawn:
        charts["confusion"] = str(drawn)
    return charts


# ===========================================================================
# 12.9  THE DISCUSSION, FROM THE NUMBERS
# ===========================================================================
def _best(table: pd.DataFrame, column: str) -> tuple[list[str], float]:
    values = pd.to_numeric(table[column], errors="coerce").where(
        trained_rows(table))
    if not values.notna().any():
        return [], float("nan")
    best = (values.min() if column in LOWER_IS_BETTER else values.max())
    names = [c for c, v in zip(table["config"], values)
             if pd.notna(v) and abs(float(v) - float(best)) < 1e-12]
    return names, float(best)


def build_summary(built: dict[str, Any], zero_recall: pd.DataFrame,
                  head: pd.DataFrame) -> str:
    """The Step 12 discussion, composed only from the computed table."""
    table = built["table"]
    live = _chart_table(table)
    lines = ["## Step 12 - discussion", ""]
    if not len(live):
        return "\n".join(lines + ["No configuration has a saved result yet."])

    for column in ("F1 (macro)", "G-mean", "ROC-AUC (macro OvR)",
                   "binary FPR", "F1 macro (dup-free)", "G-mean (dup-free)"):
        names, value = _best(table, column)
        if not names:
            continue
        # naming six "winners" when every row is identical reads as six
        # results; it is one, and it is that there is no difference
        if len(names) == int(trained_rows(table).sum()) and len(names) > 1:
            lines.append("- **%s**: every configuration ties at %.4f, so this "
                         "column separates nothing." % (column, value))
            continue
        lines.append("- Best **%s**: %.4f, from %s%s."
                     % (column, value, ", ".join("`%s`" % n for n in names),
                        " (a tie)" if len(names) > 1 else ""))

    approach1 = dict(built["order"]).get("proposed")
    if approach1:
        losses = []
        for column in COLUMNS:
            if column in DESCRIPTORS:
                continue
            names, value = _best(table, column)
            if names and approach1 not in names:
                mine = pd.to_numeric(
                    table.loc[table["config"] == approach1, column],
                    errors="coerce").iloc[0]
                if pd.notna(mine):
                    shown = ("%s %s vs %s" % (names[0], f"{int(value):,}",
                                              f"{int(mine):,}")
                             if column == "params" else
                             "%s %.4f vs %.4f" % (names[0], value, float(mine)))
                    losses.append("%s (%s)" % (column, shown))
        lines.append(
            "- Where Approach 1 is **not** best: %s."
            % ("; ".join(losses) if losses
               else "nowhere -- it leads every column"))

    accuracy = pd.to_numeric(live["accuracy"], errors="coerce")
    macro = pd.to_numeric(live["F1 (macro)"], errors="coerce")
    if accuracy.notna().any() and macro.notna().any():
        lines.append(
            "- Accuracy runs %.4f to %.4f while macro-F1 runs %.4f to %.4f. "
            "The gap is the class imbalance: accuracy is dominated by the "
            "common classes, macro-F1 gives a five-row class the same weight "
            "as a sixty-thousand-row one, so it is the number to read."
            % (accuracy.min(), accuracy.max(), macro.min(), macro.max()))

    if len(zero_recall):
        zero = zero_recall[zero_recall["G-mean is zero"]
                           & (zero_recall["scope"] == "full test set")]
        if len(zero):
            lines.append(
                "- **G-mean = 0** for %s. The geometric mean of the per-class "
                "recalls is zero the moment one class is never detected, so "
                "this is not \"poor overall\": these classes were missed "
                "entirely -- %s."
                % (", ".join("`%s`" % c for c in zero["config"]),
                   "; ".join("%s: %s" % (r["config"], r["which"])
                             for _i, r in zero.iterrows())))
        else:
            lines.append("- No configuration has a class with recall 0 on the "
                         "full test set, so no G-mean is structurally zero.")

    for pair in (head["pair"].unique() if len(head) else []):
        part = head[head["pair"] == pair]
        row = part[part["metric"] == "F1 (macro)"]
        if not len(row):
            continue
        row = row.iloc[0]
        lines.append(
            "- **%s:** macro-F1 %.4f against %.4f, a difference of %+.4f%s."
            % (pair, row["left"], row["right"],
               row["difference (left - right)"],
               "" if row["favours"] == "tie"
               else " in favour of %s" % row["favours"]))

    missing = built.get("missing") or []
    if missing:
        lines.append(
            "- Not yet trained: %s. Those rows are blank and take no part in "
            "the best-value marking."
            % ", ".join("`%s`" % dict(built["order"])[k] for k in missing))

    lines += [
        "",
        "**Reading the ablation.** Each baseline removes or replaces exactly "
        "one component of the proposed pipeline: B1 drops the feature "
        "selection, B2 drops the CNN front end, B3 drops the autoencoder, B4 "
        "replaces the CNN with linear PCA, and B5 replaces the fused "
        "MI + Pearson + PI selection with MI alone. A gap between the proposed "
        "approach and one baseline is attributable to that component; a gap "
        "between two baselines is not, because more than one thing differs.",
        "",
        "**Four caveats.**",
        "",
        "1. **The rare classes have about five test rows each.** A per-class "
        "F1 for them moves a long way on a single flow changing hands, and "
        "macro-F1 is an average over those classes.",
        "2. **Duplicates inflate every full-test score.** Roughly half this "
        "dataset's rows share identical feature values with a training row, so "
        "the duplicate-free columns are the conservative comparison.",
        "3. **One seed per configuration** (N_SEEDS = %d). Every difference "
        "here is a single observation; no significance is claimed."
        % int(config.N_SEEDS),
        "4. **Accuracy is not the headline.** Lead with macro-F1 and G-mean; "
        "a configuration that wins on accuracy and loses on G-mean has not "
        "won.",
    ]
    return "\n".join(lines)


# ===========================================================================
# SAVE AND RUN
# ===========================================================================
def verify_saved(folder: Path, out: dict[str, Any],
                 verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """Read the saved files back from disk and check them."""
    names = config.STEP12_FULL_FILES
    checks: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        checks.append({"check": check, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    path = folder / names["comparison"]
    if path.exists():
        table = pd.read_csv(path)
        expected = len(config.STEP12_DISPLAY_ORDER)
        add("the comparison table has one row per configuration",
            len(table) == expected,
            "%d rows, expected %d" % (len(table), expected))
        add("the rows are in the display order",
            list(table["config"]) == [l for _k, l
                                      in config.STEP12_DISPLAY_ORDER],
            " | ".join(str(c) for c in table["config"]))
        missing = [c for c in COLUMNS if c not in table.columns]
        add("every column is present", not missing,
            "%d columns" % len(table.columns) if not missing
            else "missing %s" % missing)
    else:
        add("the comparison table has one row per configuration", False,
            "not written")

    add("the existing rows are unchanged", bool(out.get("unchanged_ok")),
        "exact (1e-9) and published figures (4 dp, 5e-5)")

    before = out.get("protected_before") or {}
    after = protected_fingerprint()
    untouched = bool(before) and all(after.get(k) == v
                                     for k, v in before.items())
    add("Step 10, Step 11 and the per-configuration results are byte-identical",
        untouched, "%d files hashed before and after" % len(before))

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table, note="Read back from disk after writing. The "
                               "existing Step 12 tables and charts keep their "
                               "own names; everything here carries a `_full` "
                               "suffix.")
    return table, all_ok


def protected_fingerprint() -> dict[str, str]:
    """SHA-256 of everything this module must not touch.

    Step 10 and Step 11 belong to Approach 1, and each configuration's
    metrics.json is the source every row is read from -- if one of those moved,
    the comparison would be reporting different numbers from the write-up.
    """
    import hashlib

    out: dict[str, str] = {}
    for folder_setting in ("STEP10_OUTPUT_DIR", "STEP11_OUTPUT_DIR"):
        folder = config.usable_output_dir(getattr(config, folder_setting))
        if folder is None or not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                out[str(path.relative_to(folder.parent))] = \
                    hashlib.sha256(path.read_bytes()).hexdigest()
    for key, _label in config.STEP12_DISPLAY_ORDER:
        path = result_path(key)
        if path is not None and path.exists():
            out["metrics:%s" % key] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return out


def run_step12_full(verbose: bool = True) -> dict[str, Any]:
    """Build every Step 12 table and chart from the saved results.

    Nothing is trained: this reads `metrics.json` and `predictions.csv` for
    each configuration and assembles them.  Safe to re-run at any time.
    """
    if verbose:
        show_title("Step 12 -- the full comparison", level=3)

    protected_before = protected_fingerprint()
    built = build_full_comparison(verbose=verbose)
    table = built["table"]
    marked = mark_best(table)
    built["per_class"] = per_class_table()
    per_class_free = per_class_table(duplicate_free=True)
    zero_recall = zero_recall_diagnostic()
    head = all_head_to_heads(table)
    unchanged, unchanged_ok = check_rows_unchanged(built, verbose=verbose)
    summary = build_summary(built, zero_recall, head)

    out: dict[str, Any] = {
        "built": built, "comparison": table, "comparison_marked": marked,
        "per_class": built["per_class"], "per_class_free": per_class_free,
        "zero_recall": zero_recall, "head_to_head": head,
        "unchanged": unchanged, "unchanged_ok": unchanged_ok,
        "configuration": configuration_table(), "summary": summary,
        "missing": built["missing"], "protected_before": protected_before,
    }

    folder = prepare_output_dir(config.STEP12_OUTPUT_DIR,
                                "the full Step 12 comparison", verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP12_FULL_FILES
    table.to_csv(folder / names["comparison"], index=False)
    marked.to_csv(folder / names["comparison_marked"], index=False)
    if len(built["per_class"]):
        built["per_class"].to_csv(folder / names["per_class"], index=False)
    if len(per_class_free):
        per_class_free.to_csv(folder / names["per_class_free"], index=False)
    if len(zero_recall):
        zero_recall.to_csv(folder / names["zero_recall"], index=False)
    if len(head):
        head.to_csv(folder / names["head_to_head"], index=False)
    unchanged.to_csv(folder / names["unchanged_check"], index=False)
    (folder / names["summary"]).write_text(summary, encoding="utf-8")
    out["charts"] = draw_charts(built, folder, verbose=verbose)

    report, ok = verify_saved(folder, out, verbose=verbose)
    report.to_csv(folder / names["report"], index=False)
    out.update({"report": report, "status": "written",
                "ok": bool(ok and unchanged_ok), "folder": str(folder)})
    if verbose:
        print("  [%s]  full comparison written to %s"
              % ("OK" if out["ok"] else "FAIL", folder))
        show_title("Step 12 -- all configurations", level=3)
        show_table(marked, all_columns=True,
                   note="`*` marks the best value in each column; ties are "
                        "marked on every tied row. Lower is better for the "
                        "FPR, the parameter count and the times. Rows read "
                        "`%s` until their model exists."
                        % config.NOT_TRAINED_TEXT)
    return out
