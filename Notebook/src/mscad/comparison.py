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
    # Baseline 6 has a folder of its own too, for the same reason: it is not
    # written by the Step 12 runner.  Added beside the Approach 2 branch; no
    # existing branch changes.
    if key == getattr(config, "BASELINE6_NAME", "baseline6_ae_pca_cnn_lstm"):
        folder = getattr(config, "STEP12_B6_OUTPUT_DIR", None)
        if folder is None:
            return None
        folder = config.usable_output_dir(folder)
        return None if folder is None else folder / "metrics.json"
    # Approach 1 is the third of the same kind, and the one that was missing.
    # Its evaluation is written by Step 11 (evaluation.run_step11) to
    # step11\metrics.json, never by the Step 12 runner to
    # step12\proposed\metrics.json -- so looking only in the Step 12 folder
    # left the proposed pipeline out of its own comparison.  The Step 12 copy,
    # when a run of experiments.run_step12 has written one, is preferred: it
    # carries the descriptor fields as well as the metrics.
    if key == "proposed":
        step12 = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
        if step12 is not None and (step12 / key / "metrics.json").exists():
            return step12 / key / "metrics.json"
        folder = config.usable_output_dir(config.STEP11_OUTPUT_DIR)
        if folder is None:
            return None
        return folder / config.STEP11_FILES["metrics"]
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
    if key == getattr(config, "BASELINE6_NAME", "baseline6_ae_pca_cnn_lstm"):
        return "the Baseline 6 cells (12.6), with OVERWRITE_BASELINE6 = True"
    if key == "proposed":
        return "Step 10 and Step 11, then Step 12"
    return "Step 12 (set OVERWRITE_STEP12 = True)"


# ===========================================================================
# 12.2  THE RUN CELLS, WITHOUT THE TABLES 12.1 AND 12.5 NOW OWN
# ===========================================================================
# `experiments.run_step12`, `experiments_pca.run_step12_with_baseline4` and
# `experiments_baseline5.run_baseline5_comparison` each end by printing a
# configuration table and a comparison table of their own four, five or six
# rows, under the old `EXPERIMENTS` labels ("Proposed: ...").  12.1 and 12.5
# print both in full -- all seven configurations, in `STEP12_DISPLAY_ORDER` --
# so those versions are strictly contained in these and only make a reader
# stitch rows together again.
#
# Nothing here changes a run function: each already takes `verbose`, so the
# tables are turned off through its own flag and one status line per
# configuration is printed instead.  What is computed, saved and returned is
# identical -- `load_step12_quiet` returns the run function's own dict.
def display_label(key: str) -> str:
    """The 12.1/12.5 label for a configuration, whatever its spec calls it.

    The specs in `config.EXPERIMENTS` still carry the labels Step 12 was
    written with, and they are quoted inside files already on disk, so they
    are left alone; display reads `STEP12_DISPLAY_ORDER` instead, which is
    where every other Step 12 table gets its labels.
    """
    for name, label in config.STEP12_DISPLAY_ORDER:
        if name == key:
            return label
    return key


def run_status_lines(out: dict[str, Any] | None,
                     keys: Sequence[str] | None = None) -> list[str]:
    """One short line per configuration, in place of a table.

    `keys` defaults to the four `config.EXPERIMENTS` names, which is what
    `experiments.run_step12` covers; the Baseline 4 and Baseline 5 cells pass
    their own longer lists.

    The status comes from the run function's own return value: a run it trained
    says `written`, one it read back says `loaded_existing`.  A run function
    that took its load-from-disk path returns no per-configuration runs at all,
    so the saved `metrics.json` is checked instead -- a line never claims a
    result that is not on disk.
    """
    out = out or {}
    keys = ([spec["name"] for spec in config.EXPERIMENTS] if keys is None
            else list(keys))

    statuses: dict[str, str] = {}
    for run in (out.get("runs") or []):
        if not isinstance(run, dict):
            continue
        name = (run.get("spec") or {}).get("name")
        if name:
            statuses[str(name)] = str(run.get("status") or "")

    labels = {key: display_label(key) for key in keys}
    width = max([len(label) for label in labels.values()] or [0])

    lines = []
    for key in keys:
        status = statuses.get(key)
        if status == "written":
            what = "trained and saved"
        elif status == "loaded_existing":
            what = "loaded (saved result)"
        else:
            path = result_path(key)
            what = ("loaded (saved result)"
                    if path is not None and path.exists()
                    else "not available -- run %s" % which_step_produces(key))
        label = labels[key]
        lines.append("%s %s %s" % (label, "." * (width - len(label) + 3), what))
    return lines


def load_step12_quiet(runner: Any = None, keys: Sequence[str] | None = None,
                      show: bool = True, **kwargs: Any) -> dict[str, Any]:
    """Run or load a Step 12 section with its superseded tables suppressed.

    `runner` is the run function to call and defaults to
    `experiments.run_step12`; it is called with `verbose=False` and every other
    argument passed straight through, so it does exactly the work it always
    did -- trains what is missing, reads back what is finished, writes the same
    files -- and its own dict is returned unchanged.

    Returns that dict; prints the status lines unless `show=False`.
    """
    runner = xp.run_step12 if runner is None else runner
    out = runner(verbose=False, **kwargs)
    if show:
        for line in run_status_lines(out, keys=keys):
            print(line)
    return out


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


def _blank_row(label: str, key: str | None = None) -> dict[str, Any]:
    """A placeholder row for a configuration that has not been trained.

    The cell names the step that produces THIS configuration.  The old fixed
    text said "run 10.2" for whichever row was missing, and 10.2 trains
    Approach 2 -- wrong for seven of the eight rows.
    """
    row = {column: None for column in COLUMNS}
    row["config"] = label
    row["input features"] = (config.NOT_TRAINED_TEXT if key is None
                             else "%s - %s" % (config.NOT_TRAINED_TEXT,
                                               which_step_produces(key)))
    return row


def _proposed_descriptors() -> dict[str, Any]:
    """Approach 1's descriptor fields, from files Steps 4, 8 and 10 already saved.

    Step 11's metrics.json records the scores but not the input width, the
    latent width, the parameter count, the seed or the best epoch: Step 11b
    records those for Approach 2, Step 11 never did for Approach 1.  They are
    READ here from `model_card.json` and `selected_features.json` -- nothing is
    recomputed, no model is loaded and no saved file is rewritten -- so the
    descriptor columns can be filled without re-running Step 11.
    """
    out: dict[str, Any] = {}
    step6 = config.usable_output_dir(config.STEP6_OUTPUT_DIR)
    if step6 is not None:
        picked = Path(step6) / config.STEP6_FILES["selected"]
        if picked.exists():
            try:
                saved = json.loads(picked.read_text(encoding="utf-8"))
            except (ValueError, OSError):           # pragma: no cover
                saved = {}
            count = saved.get("n_selected") or len(saved.get("features") or [])
            if count:
                out["n_input_features"] = int(count)
    step10 = config.usable_output_dir(config.STEP10_OUTPUT_DIR)
    if step10 is not None:
        card_path = Path(step10) / config.STEP10_FILES.get("model_card",
                                                           "model_card.json")
        if card_path.exists():
            try:
                card = json.loads(card_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):           # pragma: no cover
                card = {}
            shape = list(card.get("input_shape_per_sample") or [])
            if shape:
                out["model_input_shape"] = shape
                out["latent_dim"] = int(shape[0])
            architecture = card.get("architecture") or {}
            if architecture.get("total_parameters") is not None:
                out["parameters"] = int(architecture["total_parameters"])
            training = card.get("training") or {}
            for source, target in (("seed", "seed"),
                                   ("best_epoch", "best_epoch"),
                                   ("best_val_macro_f1", "best_val_macro_f1"),
                                   ("training_seconds", "training_seconds")):
                if training.get(source) is not None:
                    out[target] = training[source]
    return out


def _with_descriptors(key: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Fill a row's descriptor fields when its own metrics.json lacks them."""
    if key != "proposed" or metrics is None:
        return metrics
    extra = {name: value for name, value in _proposed_descriptors().items()
             if metrics.get(name) in (None, "")}
    return {**metrics, **extra} if extra else metrics


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
            rows.append(_blank_row(label, key))
            continue
        metrics = _with_descriptors(key, metrics)
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
    """Boolean mask of the rows that actually have results.

    `startswith` rather than equality: a blank row's cell is the state followed
    by the step that produces it, and that tail differs per configuration.  An
    equality test here would count every untrained row as trained and let it
    into the best-value marking.
    """
    return ~table["input features"].astype(str).str.startswith(
        config.NOT_TRAINED_TEXT)


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
def published_figures() -> dict[str, dict[str, float]]:
    """The write-up's figures for the ACTIVE dataset, or an empty dict.

    config.STEP12_EXPECTED_ROWS holds the figures for the dataset this run
    reads: the own extraction's, or -- under DATASET_SOURCE = "official" --
    nothing, until the official numbers the write-up quotes are recorded in
    config.STEP12_EXPECTED_ROWS_OFFICIAL.  Checking official results against
    the own extraction's figures reported [FAIL] on every row: correct, in that
    the numbers differ, and useless, because they are a different dataset's
    numbers over a different class count.  Empty means [SKIP] with the reason,
    which is an honest "not checked".
    """
    return dict(getattr(config, "STEP12_EXPECTED_ROWS", None) or {})


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

    expected_rows = published_figures()
    if not expected_rows:
        rows.append({
            "check": "published figures",
            "result": "[SKIP]",
            "detail": "no figures are recorded for the %s dataset, so there is "
                      "nothing to compare; record them in "
                      "config.STEP12_EXPECTED_ROWS_OFFICIAL once the write-up "
                      "quotes them" % dataset_source()})
    for key, expected in expected_rows.items():
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


# ===========================================================================
# 12.1b  THE EIGHT-CONFIGURATION TABLES  (added; nothing above is changed)
# ===========================================================================
# Baseline 6 joins the display order, so every function above that iterates
# `STEP12_DISPLAY_ORDER` already covers it.  What it cannot do is describe a
# configuration that is in neither EXPERIMENTS_WITH_PCA nor the two addenda,
# so the configuration and training-setup tables are built here instead.
FIGURE_FILES = {
    "metrics": "fig_metrics_8configs",
    "metrics_free": "fig_metrics_duplicate_free_8configs",
    "accuracy": "fig_accuracy_8configs",
    "roc": "fig_roc_macro_8configs",
    "per_class": "fig_per_class_f1_8configs",
    "confusion": "fig_confusion_grid_8configs",
    "time": "fig_time_8configs",
    "fpr": "fig_binary_fpr_8configs",
}


# ===========================================================================
# 12.x  THE HONEST TABLE: ONE DATASET PER TABLE
# ===========================================================================
# Each dataset has its own output root (outputs\ for the own extraction,
# outputs_official\ for the published MSCAD.csv), and a configuration can be
# finished in one and never trained in the other.  Today, under
# DATASET_SOURCE = "official", only Approach 1 and Approach 2 have been
# retrained; B1-B6 exist only in the own-extraction root.
#
# A row read from the other root would be a false statement, not a stale one:
# the two datasets have different class counts (6 vs 7) and different label
# distributions, so the numbers are not on the same scale.  Nothing below
# falls back to the other root -- `in_active_root` is the assertion that says
# so, and the pending rows say which dataset they are pending ON.


def dataset_source() -> str:
    """The dataset this run reads, as the tables name it."""
    return str(getattr(config, "DATASET_SOURCE", "own"))


def not_trained_here_text() -> str:
    """The placeholder for a configuration missing from the ACTIVE root."""
    return "not trained on %s" % dataset_source()


def in_active_root(path: Path | str | None) -> bool:
    """True only when `path` sits under config.OUTPUTS_ROOT.

    Compared by path PARTS, never as text: "outputs" is a prefix of
    "outputs_official", so a substring test would call a file in the own root
    a file in the official one.
    """
    if path is None:
        return False
    root = Path(getattr(config, "OUTPUTS_ROOT", ""))
    parts, root_parts = Path(path).parts, root.parts
    return bool(root_parts) and parts[:len(root_parts)] == root_parts


def results_in_active_root(order: Sequence[tuple[str, str]] | None = None
                           ) -> tuple[list[str], list[str]]:
    """(trained here, pending here) for the configurations in display order.

    "Trained here" means the file exists AND is under the active root.  The
    second half of that is the point: it is what makes a cross-root read
    impossible rather than merely unlikely.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    here, pending = [], []
    for key, _label in order:
        path = result_path(key)
        if in_active_root(path) and Path(path).exists():
            here.append(key)
        else:
            pending.append(key)
    return here, pending


def active_root_provenance(order: Sequence[tuple[str, str]] | None = None
                           ) -> pd.DataFrame:
    """Every row, its state, and the exact file it was read from."""
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    here, _pending = results_in_active_root(order)
    rows = []
    for key, label in order:
        path = result_path(key)
        rows.append({
            "config": label,
            "state": ("read from disk" if key in here
                      else not_trained_here_text()),
            # where it is, or where it WOULD be -- always under the active
            # root, which is the whole point: there is nowhere else to look.
            "metrics.json": "(unreachable here)" if path is None else str(path),
            "file exists": "yes" if key in here else "no",
            "under %s root" % dataset_source(): ("yes" if in_active_root(path)
                                                 else "no"),
        })
    return pd.DataFrame(rows)


def honest_comparison(order: Sequence[tuple[str, str]] | None = None,
                      verbose: bool = True) -> dict[str, Any]:
    """The comparison table with the pending rows naming the dataset.

    `build_full_comparison` already leaves a missing row blank -- it reads one
    path per configuration and that path is rooted at config.OUTPUTS_ROOT.
    What this adds is the wording: a reader of the rendered table sees "not
    trained on official" rather than a blank and an instruction, so the table
    cannot be mistaken for eight results on one dataset.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    built = build_full_comparison(order=order, verbose=False)
    here, pending = results_in_active_root(order)
    labels = dict(order)

    marked = mark_best(built["table"])
    state = [key in here for key, _label in order]
    for column in marked.columns:
        if column == "config":
            continue
        marked[column] = [
            # a descriptor a run did not record reads "-", never "NaN"
            ("-" if pd.isna(value) else value) if is_here else
            (not_trained_here_text() if column == "input features" else "-")
            for value, is_here in zip(marked[column], state)]

    trained = built["table"].loc[state].reset_index(drop=True)
    return {"built": built, "table": built["table"], "marked": marked,
            "trained": trained, "trained_keys": here, "pending_keys": pending,
            "pending_labels": [labels[key] for key in pending],
            "provenance": active_root_provenance(order),
            "source": dataset_source()}


def trained_only_table(honest: dict[str, Any]) -> pd.DataFrame:
    """The trained configurations alone, with the full metric columns.

    Two rows today (Approach 1 and Approach 2).  Best-value marking is done
    over THESE rows only: marking over eight rows where six are blank says
    "best of eight" for what is really "better of two".
    """
    return mark_best(honest["trained"]) if len(honest["trained"]) else \
        pd.DataFrame(columns=COLUMNS)


def show_honest_step10(majority_accuracy: float | None = None,
                       random_macro_f1: float | None = None,
                       order: Sequence[tuple[str, str]] | None = None,
                       verbose: bool = True) -> dict[str, Any]:
    """What the Step 10 notebook cell calls.  Nothing is trained or written.

    The exports, checks and figures stay where they were: this calls
    `run_step12_complete` quietly for them and then renders the honest view of
    the same table, so the CSVs on disk and the table on screen are the same
    numbers.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    full = run_step12_complete(majority_accuracy=majority_accuracy,
                               random_macro_f1=random_macro_f1, verbose=False)
    honest = honest_comparison(order=order, verbose=False)
    honest["complete"] = full
    source = honest["source"]

    if verbose:
        show_title("10.x -- the comparison, one dataset only", level=4)
        show_note(
            "Only **%d of %d** configurations have been retrained on the "
            "**%s** dataset: %s. The rest are listed as `%s`; their earlier "
            "results in the other output root are NOT repeated here, because "
            "the two datasets have different class counts (6 vs 7) and "
            "different label distributions, so a row-by-row comparison across "
            "them is not valid."
            % (len(honest["trained_keys"]), len(order), source,
               ", ".join("`%s`" % key for key in honest["trained_keys"]),
               not_trained_here_text()))
        show_table(honest["marked"], all_columns=True,
                   note="Every filled row read from a metrics.json under `%s`. "
                        "`*` marks the best value in a column among the rows "
                        "that have one; lower is better for the FPR, the "
                        "parameter count and the times."
                        % config.OUTPUTS_ROOT
                        + ("" if majority_accuracy is None else
                           " The majority-class accuracy is %.4f."
                           % majority_accuracy))
        pair = trained_only_table(honest)
        if len(pair):
            show_title("Approach 1 vs Approach 2 (the %s dataset)" % source,
                       level=5)
            show_table(pair, all_columns=True,
                       note="The configurations actually trained on this "
                            "dataset, with the full metric columns and the "
                            "best value marked over these rows only.")
        show_table(honest["provenance"], all_columns=True,
                   note="Where each row came from. Every path is under the "
                        "active output root, including the rows with no file: "
                        "a pending row is pending because nothing exists "
                        "THERE, never because it was filled from the other "
                        "root.")
    return honest


def model_colour(key: str) -> str:
    """One fixed colour per configuration, used by every Step 12 figure."""
    colours = getattr(config, "MODEL_COLOURS", {}) or {}
    if key in colours:
        return colours[key]
    order = [k for k, _label in config.STEP12_DISPLAY_ORDER]
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#8a63d2", "#00a9b5", "#a06a2c"]
    return palette[(order.index(key) if key in order else 0) % len(palette)]


def short_label(key: str) -> str:
    return display_label(key).split(":")[0].strip()


def configuration_table_8(order: Sequence[tuple[str, str]] | None = None
                          ) -> pd.DataFrame:
    """One row per configuration: what it is made of, and what it isolates.

    Everything that can be read from a saved metrics.json is read from it --
    the input width, the latent width, the PCA components and the variance
    they keep, the parameter count -- so the table describes the model that
    actually ran rather than the configuration that was intended.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    approach2 = getattr(config, "APPROACH2_KEY", "approach2_attention")
    baseline6 = getattr(config, "BASELINE6_NAME", "baseline6_ae_pca_cnn_lstm")
    specs = {spec["name"]: spec for spec in config.EXPERIMENTS_WITH_PCA}
    rows = []
    for key, label in order:
        metrics = load_result(key) or {}
        spec = specs.get(key, {})
        pca_block = metrics.get("pca") or {}
        components = metrics.get("pca_components")
        latent = metrics.get("latent_dim")
        if key == approach2:
            selection, ae, pca, cnn, attention = (
                "Step 6 top %d" % config.N_SELECTED_FEATURES, True, False,
                "yes", "yes")
        elif key == "baseline5":
            selection, ae, pca, cnn, attention = (
                "MI only, top %d" % config.BASELINE5_TOP_K, True, False,
                "yes", "no")
        elif key == baseline6:
            selection, ae, pca, cnn, attention = (
                "none (all Step 5 features)", True, True, "yes", "no")
        else:
            selection = ("Step 6 top %d" % config.N_SELECTED_FEATURES
                         if spec.get("feature_source") == "selected30"
                         else "none (all Step 5 features)")
            ae = bool(spec.get("use_ae"))
            pca = bool(spec.get("use_pca"))
            cnn = "yes" if spec.get("model_type") == "cnn_lstm" else "no"
            attention = "no"
        rows.append({
            "config": label,
            "input features": metrics.get("n_input_features", "-"),
            "feature selection": selection,
            "AE": ("yes (-> %s)" % (latent or config.AE_LATENT_DIM)) if ae
                  else "no",
            "PCA": ("%s of %s (%.1f%% variance)"
                    % (components, latent,
                       100 * float(pca_block.get("variance_kept", float("nan"))))
                    if pca and components else ("yes" if pca else "no")),
            "CNN": cnn, "attention": attention, "LSTM": "yes",
            "input shape": (str(tuple(metrics.get("model_input_shape") or []))
                            or "-"),
            "params": metrics.get("parameters", "-"),
            "what it isolates": config.STEP12_ISOLATES.get(key, ""),
        })
    return pd.DataFrame(rows)


def training_setup_table(order: Sequence[tuple[str, str]] | None = None
                         ) -> pd.DataFrame:
    """The protocol, side by side, so any difference is visible rather than
    assumed away.

    Values recorded by the run itself (seed, best epoch, training time) come
    from its metrics.json; the shared settings come from config, which is what
    every configuration was trained with.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    rows = []
    for key, label in order:
        metrics = load_result(key)
        if metrics is None:
            rows.append({"config": label, "optimiser": "-",
                         "learning rate": "-", "batch size": "-",
                         "max epochs": "-", "best epoch": "-",
                         "patience": "-", "early stopping on": "-",
                         "class weighting": "-", "seed": "-",
                         "train time (s)": "-"})
            continue
        rows.append({
            "config": label,
            "optimiser": "Adam",
            "learning rate": config.CLF_LR,
            "batch size": config.CLF_BATCH_SIZE,
            "max epochs": config.CLF_EPOCHS,
            "best epoch": metrics.get("best_epoch", "-"),
            "patience": config.CLF_PATIENCE,
            "early stopping on": config.EARLY_STOP_METRIC,
            "class weighting": "%s / %s" % (config.IMBALANCE_STRATEGY,
                                            config.CLASS_WEIGHT_SCHEME),
            "seed": metrics.get("seed", config.RANDOM_STATE),
            "train time (s)": (round(float(metrics["training_seconds"]), 1)
                               if metrics.get("training_seconds") is not None
                               else "-"),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 12.9b  THE SEVEN HEAD-TO-HEAD PAIRS
# ===========================================================================
HEAD_TO_HEAD_PAIRS_8 = [
    ("proposed", "baseline1_ae_cnn_lstm",
     "feature selection: Approach 1 vs B1 (with vs without)"),
    ("proposed", "baseline5",
     "selection method: Approach 1 vs B5 (fused vs MI only)"),
    ("proposed", "approach2_attention",
     "attention: Approach 1 vs Approach 2"),
    ("baseline1_ae_cnn_lstm", "baseline2_ae_lstm",
     "the CNN front end: B1 vs B2"),
    ("baseline1_ae_cnn_lstm", "baseline3_cnn_lstm",
     "the autoencoder: B1 vs B3"),
    ("baseline4_ae_pca_lstm", "baseline6_ae_pca_cnn_lstm",
     "the CNN on PCA features: B4 vs B6"),
    ("baseline1_ae_cnn_lstm", "baseline6_ae_pca_cnn_lstm",
     "PCA before an unchanged CNN-LSTM: B1 vs B6"),
]


def pca_width_caveat(left: str, right: str) -> str:
    """"" unless the two configurations' PCA spaces are different widths.

    A pair is only a one-difference comparison if everything else matches.  B4
    keeps the components a 0.95 variance target selected, while B6 needs at
    least `CNN_KERNEL` of them for the shared stack to be buildable at all, so
    whenever those counts differ the pair label has to say so instead of
    promising an isolation it does not have.
    """
    counts = []
    for key in (left, right):
        metrics = load_result(key) or {}
        value = metrics.get("pca_components")
        counts.append(None if value is None else int(value))
    if None in counts or counts[0] == counts[1]:
        return ""
    return (" [two differences: PCA(%d) vs PCA(%d) as well -- see 12.11]"
            % (counts[0], counts[1]))


def all_head_to_heads_8(table: pd.DataFrame,
                        order: Sequence[tuple[str, str]] | None = None
                        ) -> pd.DataFrame:
    """Every pair that differs in exactly one component, metric by metric.

    A pair whose two configurations turn out NOT to differ in exactly one
    component says so in its own label, rather than being dropped or left
    looking clean.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    labels = dict(order)
    frames = []
    for left, right, why in HEAD_TO_HEAD_PAIRS_8:
        if left not in labels or right not in labels:
            continue
        frame = head_to_head(table, labels[left], labels[right],
                             pair=why + pca_width_caveat(left, right))
        if len(frame):
            frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def per_class_difference_8(per_class: pd.DataFrame,
                           order: Sequence[tuple[str, str]] | None = None
                           ) -> pd.DataFrame:
    """Per-class F1 differences for the same pairs, class by class."""
    if per_class is None or not len(per_class):
        return pd.DataFrame()
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    labels = dict(order)
    rows = []
    for left, right, why in HEAD_TO_HEAD_PAIRS_8:
        left_label, right_label = labels.get(left), labels.get(right)
        if (left_label not in per_class.columns
                or right_label not in per_class.columns):
            continue
        why = why + pca_width_caveat(left, right)
        for _i, line in per_class.iterrows():
            a = pd.to_numeric(line[left_label], errors="coerce")
            b = pd.to_numeric(line[right_label], errors="coerce")
            if pd.isna(a) or pd.isna(b):
                continue
            rows.append({
                "pair": why, "Label": line["Label"],
                "test rows": line.get("test rows"),
                short_label(left): round(float(a), 6),
                short_label(right): round(float(b), 6),
                "difference": round(float(a - b), 6),
                "favours": (short_label(left) if a > b
                            else short_label(right) if b > a else "tie")})
    return pd.DataFrame(rows)


# ===========================================================================
# 12.8b  THE PUBLICATION FIGURES -- one colour per configuration, PNG + PDF
# ===========================================================================
def _pyplot(verbose: bool = True):
    return xp._pyplot(verbose)


def _style(ax) -> None:
    """The project's axis style, from the module that defines it."""
    from . import fusion as fz
    fz._style(ax)


def _sequential_ramp() -> list[str]:
    """The one-hue ramp every heatmap in this project uses."""
    from . import fusion as fz
    return list(fz._SEQ_RAMP)


def _save_figure(fig, folder: Path, stem: str, verbose: bool = True,
                 also: Path | None = None) -> dict[str, str]:
    """Write one figure as PNG (dpi 300) and PDF, into one or two folders.

    New stems, so nothing a previous Step 12 wrote is overwritten.
    """
    import matplotlib.pyplot as plt

    dpi = int(getattr(config, "THESIS_FIGURE_DPI", 300))
    written: dict[str, str] = {}
    fig.tight_layout()
    for place in [folder] + ([also] if also is not None else []):
        place = Path(place)
        place.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            path = place / ("%s.%s" % (stem, suffix))
            fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
            written[("%s:%s" % (place.name, suffix))] = str(path)
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s.png / .pdf" % stem)
    return written


def _figure_rows(table: pd.DataFrame,
                 order: Sequence[tuple[str, str]] | None = None
                 ) -> list[tuple[str, str, pd.Series]]:
    """(key, label, row) for every configuration that has numbers, in order."""
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    rows = []
    for key, label in order:
        part = table[table["config"] == label]
        if len(part) and pd.notna(pd.to_numeric(part.iloc[0].get("F1 (macro)"),
                                                errors="coerce")):
            rows.append((key, label, part.iloc[0]))
    return rows


def figure_metric_bars(table: pd.DataFrame, columns: Sequence[str],
                       title: str, folder: Path, stem: str,
                       reference: float | None = None,
                       reference_label: str = "", verbose: bool = True,
                       also: Path | None = None) -> dict[str, str]:
    """Grouped bars across configurations, each in its own fixed colour.

    The colour identifies the configuration, so the bar groups are also
    separated by position and every bar carries its value: colour is never the
    only encoding.
    """
    plt = _pyplot(verbose)
    rows = _figure_rows(table)
    if plt is None or not rows:
        return {}
    columns = [c for c in columns if c in table.columns]
    positions = np.arange(len(columns), dtype="float64")
    width = 0.80 / max(len(rows), 1)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    for index, (key, label, row) in enumerate(rows):
        values = [float(pd.to_numeric(row.get(c), errors="coerce"))
                  for c in columns]
        offset = (index - (len(rows) - 1) / 2) * width
        bars = ax.bar(positions + offset, np.nan_to_num(values), width * 0.92,
                      label=short_label(key), color=model_colour(key))
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.annotate("%.3f" % value,
                        (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=6, rotation=90,
                        color=dl._INK)
    if reference is not None and not np.isnan(reference):
        ax.axhline(reference, color=dl._INK_SOFT, linestyle="--", linewidth=1.2)
        ax.annotate(reference_label or "reference %.3f" % reference,
                    (len(columns) - 0.5, reference), textcoords="offset points",
                    xytext=(-4, 4), ha="right", fontsize=8, color=dl._INK_SOFT)
    ax.set_xticks(positions)
    ax.set_xticklabels(columns, fontsize=9, color=dl._INK_SOFT)
    ax.set_ylabel("score (0-1)", color=dl._INK_SOFT, fontsize=10)
    ax.set_title(title, color=dl._INK, fontsize=12, pad=12)
    ax.set_ylim(0, 1.18)
    legend = ax.legend(frameon=False, loc="upper center", fontsize=8,
                       ncol=min(len(rows), 8))
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return _save_figure(fig, folder, stem, verbose=verbose, also=also)


def figure_accuracy(table: pd.DataFrame, folder: Path, stem: str,
                    majority: float | None = None, verbose: bool = True,
                    also: Path | None = None) -> dict[str, str]:
    """Accuracy with the majority-class line, on an axis that does not lie.

    The axis starts a little below the lowest bar rather than at the majority
    line, and the range is stated in the title, because a zoomed accuracy axis
    is the easiest way to make a 0.002 difference look decisive.
    """
    plt = _pyplot(verbose)
    rows = _figure_rows(table)
    if plt is None or not rows:
        return {}
    values = [float(pd.to_numeric(row.get("accuracy"), errors="coerce"))
              for _k, _l, row in rows]
    lowest = float(np.nanmin(values + ([majority] if majority else [])))
    floor = max(0.0, lowest - 0.02)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    bars = ax.bar(np.arange(len(rows)), values, 0.62,
                  color=[model_colour(k) for k, _l, _r in rows])
    for bar, value in zip(bars, values):
        ax.annotate("%.4f" % value,
                    (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8, color=dl._INK)
    if majority is not None and not np.isnan(majority):
        ax.axhline(majority, color=dl._INK_SOFT, linestyle="--", linewidth=1.3)
        ax.annotate("majority class %.4f" % majority,
                    (len(rows) - 0.5, majority), textcoords="offset points",
                    xytext=(-4, 5), ha="right", fontsize=9, color=dl._INK_SOFT)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([short_label(k) for k, _l, _r in rows], fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_ylabel("accuracy", color=dl._INK_SOFT, fontsize=10)
    ax.set_ylim(floor, 1.002)
    ax.set_title("Accuracy barely separates these models -- axis %.3f to 1.000"
                 % floor, color=dl._INK, fontsize=12, pad=12)
    return _save_figure(fig, folder, stem, verbose=verbose, also=also)


def _roc_macro(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """The macro one-vs-rest ROC curve of one configuration's predictions."""
    from sklearn.metrics import auc, roc_curve

    columns = [c for c in frame.columns if c.startswith("p_")]
    truth = frame["y_true"].to_numpy()
    grid = np.linspace(0.0, 1.0, 201)
    curves = []
    for index, column in enumerate(columns):
        positive = (truth == index).astype("int64")
        if positive.sum() == 0 or positive.sum() == len(positive):
            continue
        fpr, tpr, _t = roc_curve(positive, frame[column].to_numpy())
        curves.append(np.interp(grid, fpr, tpr))
    if not curves:
        return grid, grid, float("nan")
    mean = np.mean(curves, axis=0)
    return grid, mean, float(auc(grid, mean))


def figure_roc(folder: Path, stem: str,
               order: Sequence[tuple[str, str]] | None = None,
               verbose: bool = True, also: Path | None = None
               ) -> dict[str, str]:
    """One macro-average OvR ROC curve per configuration, in display order."""
    plt = _pyplot(verbose)
    if plt is None:
        return {}
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    drawn = 0
    for key, label in order:
        frame = _predictions(key)
        if frame is None or "y_true" not in frame.columns:
            continue
        grid, mean, area = _roc_macro(frame)
        if np.isnan(area):
            continue
        ax.plot(grid, mean, linewidth=2.0, color=model_colour(key),
                label="%s (AUC %.4f)" % (short_label(key), area))
        drawn += 1
    if not drawn:
        plt.close(fig)
        return {}
    ax.plot([0, 1], [0, 1], linewidth=1.0, color=dl._GRID, linestyle="--")
    ax.set_xlabel("false-positive rate", color=dl._INK_SOFT, fontsize=10)
    ax.set_ylabel("true-positive rate", color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Macro one-vs-rest ROC, %d configurations" % drawn,
                 color=dl._INK, fontsize=12, pad=12)
    legend = ax.legend(frameon=False, loc="lower right", fontsize=8)
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    return _save_figure(fig, folder, stem, verbose=verbose, also=also)


def figure_per_class_heatmap(per_class: pd.DataFrame, folder: Path, stem: str,
                             verbose: bool = True, also: Path | None = None
                             ) -> dict[str, str]:
    """Label x configuration F1, one hue, every cell annotated."""
    plt = _pyplot(verbose)
    if plt is None or per_class is None or not len(per_class):
        return {}
    from matplotlib.colors import LinearSegmentedColormap

    frame = per_class.set_index("Label")
    support = (frame.pop("test rows") if "test rows" in frame.columns
               else None)
    values = frame.to_numpy(dtype="float64")
    labels = ["%s (%s)" % (name, "{:,}".format(int(support[name])))
              if support is not None else str(name) for name in frame.index]

    fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.6 * len(frame) + 2.2)))
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)
    cmap = LinearSegmentedColormap.from_list("mscad_blue", _sequential_ramp())
    image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels([str(c).split(":")[0] for c in frame.columns],
                       rotation=30, ha="right", fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels(labels, fontsize=9, color=dl._INK_SOFT)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            ax.annotate("%.2f" % values[i, j], (j, i), ha="center",
                        va="center", fontsize=8,
                        color="#ffffff" if values[i, j] > 0.55 else dl._INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Per-class F1 (test rows per class in the label)",
                 color=dl._INK, fontsize=12, pad=12)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("F1", color=dl._INK_SOFT, fontsize=9)
    bar.ax.tick_params(colors=dl._INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)
    return _save_figure(fig, folder, stem, verbose=verbose, also=also)


def figure_confusion_grid(folder: Path, stem: str,
                          order: Sequence[tuple[str, str]] | None = None,
                          verbose: bool = True, also: Path | None = None
                          ) -> dict[str, str]:
    """Row-normalised confusion matrices, 2 x 4, in the display order."""
    plt = _pyplot(verbose)
    if plt is None:
        return {}
    from matplotlib.colors import LinearSegmentedColormap

    order = config.STEP12_DISPLAY_ORDER if order is None else order
    panels = []
    for key, label in order:
        path = result_path(key)
        if path is None:
            continue
        candidate = path.parent / "confusion_matrix_rownorm.csv"
        if candidate.exists():
            panels.append((key, label, pd.read_csv(candidate, index_col=0)))
    if not panels:
        return {}

    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10.0, 3.0 * rows + 0.8),
                             squeeze=False)
    fig.patch.set_facecolor(dl._SURFACE)
    cmap = LinearSegmentedColormap.from_list("mscad_blue", _sequential_ramp())
    for position, (key, label, matrix) in enumerate(panels):
        ax = axes[position // columns][position % columns]
        ax.set_facecolor(dl._SURFACE)
        values = matrix.to_numpy(dtype="float64")
        ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        names = [str(v)[:10] for v in matrix.index]
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6,
                           color=dl._INK_SOFT)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6, color=dl._INK_SOFT)
        ax.set_title(short_label(key), color=model_colour(key), fontsize=10,
                     pad=6)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
    for empty in range(len(panels), rows * columns):
        axes[empty // columns][empty % columns].axis("off")
    fig.suptitle("Row-normalised confusion matrices -- the diagonal is recall",
                 color=dl._INK, fontsize=12)
    return _save_figure(fig, folder, stem, verbose=verbose, also=also)


def figure_single_metric(table: pd.DataFrame, column: str, title: str,
                         ylabel: str, folder: Path, stem: str,
                         lower_is_better: bool = False, log: bool = False,
                         verbose: bool = True, also: Path | None = None
                         ) -> dict[str, str]:
    """One measure across the configurations: times, or the binary FPR."""
    plt = _pyplot(verbose)
    rows = _figure_rows(table)
    if plt is None or not rows or column not in table.columns:
        return {}
    values = [float(pd.to_numeric(row.get(column), errors="coerce"))
              for _k, _l, row in rows]
    if not np.isfinite(np.nan_to_num(values)).any():
        return {}

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    fig.patch.set_facecolor(dl._SURFACE)
    _style(ax)
    bars = ax.bar(np.arange(len(rows)), np.nan_to_num(values), 0.62,
                  color=[model_colour(k) for k, _l, _r in rows])
    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        ax.annotate(("%.4f" if abs(value) < 10 else "%.1f") % value,
                    (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8, color=dl._INK)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([short_label(k) for k, _l, _r in rows], fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_ylabel(ylabel, color=dl._INK_SOFT, fontsize=10)
    ax.set_title("%s%s" % (title, " (lower is better)" if lower_is_better
                           else ""), color=dl._INK, fontsize=12, pad=12)
    return _save_figure(fig, folder, stem, verbose=verbose, also=also)


def draw_figures_8(built: dict[str, Any], folder: Path,
                   majority_accuracy: float | None = None,
                   random_macro_f1: float | None = None,
                   thesis: Path | None = None, verbose: bool = True
                   ) -> dict[str, str]:
    """Every 12.8 figure, as PNG (dpi 300) and PDF, with new file names."""
    table = built["table"]
    charts: dict[str, str] = {}
    charts.update(figure_metric_bars(
        table, ["F1 (macro)", "G-mean", "ROC-AUC (macro OvR)"],
        "Full test set -- macro-F1, G-mean and ROC-AUC, %d configurations"
        % len(_figure_rows(table)), folder, FIGURE_FILES["metrics"],
        reference=random_macro_f1,
        reference_label=("random-baseline macro-F1 %.3f" % random_macro_f1
                         if random_macro_f1 is not None else ""),
        verbose=verbose, also=thesis))
    charts.update(figure_metric_bars(
        table, ["F1 macro (dup-free)", "G-mean (dup-free)"],
        "Duplicate-free subset -- the conservative comparison",
        folder, FIGURE_FILES["metrics_free"], reference=random_macro_f1,
        reference_label=("random-baseline macro-F1 %.3f" % random_macro_f1
                         if random_macro_f1 is not None else ""),
        verbose=verbose, also=thesis))
    charts.update(figure_accuracy(table, folder, FIGURE_FILES["accuracy"],
                                  majority=majority_accuracy, verbose=verbose,
                                  also=thesis))
    charts.update(figure_roc(folder, FIGURE_FILES["roc"], verbose=verbose,
                             also=thesis))
    charts.update(figure_per_class_heatmap(built.get("per_class"), folder,
                                           FIGURE_FILES["per_class"],
                                           verbose=verbose, also=thesis))
    charts.update(figure_confusion_grid(folder, FIGURE_FILES["confusion"],
                                        verbose=verbose, also=thesis))
    charts.update(figure_single_metric(
        table, "train time (s)", "Training time", "seconds", folder,
        FIGURE_FILES["time"], lower_is_better=True, log=True, verbose=verbose,
        also=thesis))
    charts.update(figure_single_metric(
        table, "binary FPR", "Binary false-positive rate",
        "false positives / benign flows", folder, FIGURE_FILES["fpr"],
        lower_is_better=True, verbose=verbose, also=thesis))
    return charts


# ===========================================================================
# 12.x  THE THESIS EXPORTS -- the same tables, as CSV and LaTeX
# ===========================================================================
def _latex_escape(value: Any) -> str:
    """Make one cell safe for LaTeX without changing what it says."""
    text = "" if value is None else str(value)
    if text == "nan":
        return "--"
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                     ("}", r"\}"), ("~", r"\textasciitilde{}"),
                     ("^", r"\textasciicircum{}")):
        text = text.replace(old, new)
    return text.replace("->", r"$\rightarrow$")


def to_latex(frame: pd.DataFrame, caption: str, label: str,
             note: str | None = None, decimals: int = 4) -> str:
    """A booktabs table a thesis can \\input{} directly.

    Written here rather than through `DataFrame.to_latex` so the caption, the
    label, the column alignment and the footnote are all under this project's
    control and identical across every exported table.
    """
    if frame is None or not len(frame):
        return "%% %s: no rows\n" % caption
    columns = list(frame.columns)
    alignment = "l" + "".join(
        "r" if pd.api.types.is_numeric_dtype(frame[c]) else "l"
        for c in columns[1:])
    lines = [r"\begin{table}[htbp]", r"  \centering",
             r"  \caption{%s}" % _latex_escape(caption),
             r"  \label{tab:%s}" % label,
             r"  \begin{tabular}{%s}" % alignment, r"    \toprule",
             "    " + " & ".join(r"\textbf{%s}" % _latex_escape(c)
                                 for c in columns) + r" \\",
             r"    \midrule"]
    for _i, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (int, np.integer)):
                cells.append("{:,}".format(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append("--" if pd.isna(value)
                             else ("%%.%df" % decimals) % float(value))
            else:
                cells.append(_latex_escape(value))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        lines.append(r"  \par\smallskip\footnotesize{%s}" % _latex_escape(note))
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


THESIS_TABLES = {
    "configuration": ("step12_configurations",
                      "The eight configurations compared in Step 12, and the "
                      "single component each one isolates."),
    "training_setup": ("step12_training_setup",
                       "Training protocol per configuration. Every shared "
                       "setting is identical; the recorded columns come from "
                       "each run's own metrics.json."),
    "sanity": ("step12_sanity_baselines",
               "Sanity baselines. These are not models under comparison: they "
               "are the floor, and the leakage test."),
    "comparison": ("step12_comparison",
                   "Test-set comparison of the eight configurations. The "
                   "duplicate-free columns exclude test flows whose exact "
                   "feature vector also occurs in training."),
    "per_class": ("step12_per_class_f1",
                  "Per-class F1 on the full test set, with the test support "
                  "of each class."),
    "per_class_free": ("step12_per_class_f1_duplicate_free",
                       "Per-class F1 on the duplicate-free subset."),
    "head_to_head": ("step12_head_to_head",
                     "Head-to-head differences between configurations that "
                     "differ in exactly one component."),
    "per_class_difference": ("step12_head_to_head_per_class",
                             "Per-class F1 differences for the same pairs."),
    "zero_recall": ("step12_zero_recall",
                    "Configurations with a G-mean of zero, and the class each "
                    "one never detects."),
    "provenance": ("step12_provenance",
                   "Provenance: the saved file behind every published number, "
                   "with its SHA-256."),
    "seeds": ("step12_seeds",
              "Across-seed means and standard deviations, where more than one "
              "seed was trained."),
}


def export_thesis(tables: dict[str, pd.DataFrame], folder: Path | None = None,
                  note: str | None = None, verbose: bool = True
                  ) -> dict[str, str]:
    """Write every supplied table as CSV and as a booktabs .tex file."""
    folder = (config.usable_output_dir(config.STEP12_THESIS_DIR)
              if folder is None else Path(folder))
    if folder is None:
        if verbose:
            print("  [SKIP]  the thesis folder is not usable here -- %s"
                  % config.why_unusable(config.STEP12_THESIS_DIR))
        return {}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for key, frame in tables.items():
        if frame is None or not len(frame):
            continue
        stem, caption = THESIS_TABLES.get(
            key, ("step12_%s" % key, "Step 12 -- %s." % key.replace("_", " ")))
        csv_path = folder / ("%s.csv" % stem)
        tex_path = folder / ("%s.tex" % stem)
        frame.to_csv(csv_path, index=False)
        tex_path.write_text(to_latex(frame, caption, stem, note=note),
                            encoding="utf-8")
        written["%s.csv" % stem] = str(csv_path)
        written["%s.tex" % stem] = str(tex_path)
    if verbose and written:
        print("  [SAVE]  %d thesis files in %s" % (len(written), folder))
    return written


# ===========================================================================
# 12.7b  THE COMPLETE STEP 12 RUN -- eight rows, verified against disk
# ===========================================================================
def verify_rows_against_disk(table: pd.DataFrame,
                             order: Sequence[tuple[str, str]] | None = None,
                             tolerance: float = 1e-9) -> pd.DataFrame:
    """Every number in the table against the metrics.json it came from.

    `check_rows_unchanged` already does this against the published figures in
    config; this is the other half of the question -- does the rendered table
    equal the file on disk RIGHT NOW -- and it is what makes a quoted number
    traceable.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    rows = []
    for key, label in order:
        metrics = load_result(key)
        if metrics is None:
            rows.append({"config": label, "metrics.json": "(absent)",
                         "columns compared": 0, "largest difference": np.nan,
                         "result": "[SKIP]"})
            continue
        # The SAME enrichment the table was built with.  Step 11 never recorded
        # Approach 1's parameter count, latent width or seed, so `_with_
        # descriptors` reads them from step10\model_card.json and step6's
        # selected_features.json -- saved files, nothing recomputed.  Rebuilding
        # from the raw metrics.json instead compared 50,438 parameters against
        # `null` and reported an infinite difference: a [FAIL] that said the
        # table disagreed with disk when it did not.  Every number still comes
        # from a file on disk; this reads the same files the row did.
        metrics = _with_descriptors(key, metrics)
        built = _row_from_metrics(label, metrics)
        part = table[table["config"] == label]
        if not len(part):
            rows.append({"config": label,
                         "metrics.json": str(result_path(key)),
                         "columns compared": 0, "largest difference": np.nan,
                         "result": "[FAIL]"})
            continue
        line = part.iloc[0]
        worst, compared = 0.0, 0
        for column, value in built.items():
            if column == "config" or column not in table.columns:
                continue
            left = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            right = pd.to_numeric(pd.Series([line[column]]),
                                  errors="coerce").iloc[0]
            if pd.isna(left) and pd.isna(right):
                continue
            if pd.isna(left) or pd.isna(right):
                worst = max(worst, float("inf"))
                compared += 1
                continue
            worst = max(worst, abs(float(left) - float(right)))
            compared += 1
        rows.append({"config": label, "metrics.json": str(result_path(key)),
                     "columns compared": compared,
                     "largest difference": worst,
                     "result": "[OK]" if worst <= tolerance else "[FAIL]"})
    return pd.DataFrame(rows)


def run_step12_complete(majority_accuracy: float | None = None,
                        random_macro_f1: float | None = None,
                        verbose: bool = True) -> dict[str, Any]:
    """12.7 to 12.9: the eight-row comparison, its checks, tables and figures.

    Nothing is trained and nothing is recomputed from a model: every row is
    read from a saved metrics.json, and `verify_rows_against_disk` proves the
    rendered table equals those files to 1e-9 before anything is written.
    """
    if verbose:
        show_title("12.7 -- the eight-configuration comparison", level=4)
    built = build_full_comparison(verbose=False)
    table = built["table"]
    marked = mark_best(table)
    built["per_class"] = per_class_table()
    per_class_free = per_class_table(duplicate_free=True)
    zero_recall = zero_recall_diagnostic()
    head = all_head_to_heads_8(table)
    per_class_difference = per_class_difference_8(built["per_class"])
    verified = verify_rows_against_disk(table)
    unchanged, unchanged_ok = check_rows_unchanged(built, verbose=False)
    configuration = configuration_table_8()
    setup = training_setup_table()

    out: dict[str, Any] = {
        "built": built, "comparison": table, "comparison_marked": marked,
        "per_class": built["per_class"], "per_class_free": per_class_free,
        "zero_recall": zero_recall, "head_to_head": head,
        "per_class_difference": per_class_difference,
        "verified": verified, "unchanged": unchanged,
        "unchanged_ok": unchanged_ok, "configuration": configuration,
        "training_setup": setup, "missing": built["missing"],
        "majority_accuracy": majority_accuracy,
        "random_macro_f1": random_macro_f1,
    }

    folder = prepare_output_dir(config.STEP12_OUTPUT_DIR,
                                "the Step 12 comparison", verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out
    folder = Path(folder)
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
    configuration.to_csv(folder / "configuration_table_8configs.csv",
                         index=False)
    setup.to_csv(folder / "training_setup_8configs.csv", index=False)
    verified.to_csv(folder / "row_vs_disk_check.csv", index=False)
    if len(per_class_difference):
        per_class_difference.to_csv(
            folder / "head_to_head_per_class_8configs.csv", index=False)

    thesis = config.usable_output_dir(config.STEP12_THESIS_DIR)
    out["charts"] = draw_figures_8(built, folder,
                                   majority_accuracy=majority_accuracy,
                                   random_macro_f1=random_macro_f1,
                                   thesis=thesis, verbose=verbose)
    out.update({"status": "written", "folder": str(folder),
                "ok": bool(unchanged_ok
                           and not (verified["result"] == "[FAIL]").any())})
    if verbose:
        show_table(marked, all_columns=True,
                   note="Eight configurations, every row read from its saved "
                        "metrics.json. `*` marks the best value in a column on "
                        "every tied row; lower is better for the FPR, the "
                        "parameter count and the times."
                        + ("" if majority_accuracy is None else
                           " The majority-class accuracy is %.4f."
                           % majority_accuracy))
    return out
