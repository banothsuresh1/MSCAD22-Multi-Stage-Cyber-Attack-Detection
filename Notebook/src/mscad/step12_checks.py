r"""
src/mscad/step12_checks.py -- STEP 12: provenance, the retraining lock and the
sanity baselines.

NEW FILE.  Nothing existing is modified: every function here imports and calls
the finished Steps 1-11 code.  Three jobs, in the order the notebook runs them:

12.0  PROVENANCE.  Which saved file is each published number from?  Every
      configuration's model, metrics.json and predictions.csv are fingerprinted
      (path, size, modified time, SHA-256), and three consistency questions are
      asked and ANSWERED rather than assumed:
        a) are the predictions Step 13 and 13.8 fused the same ones that
           configuration's predictions.csv holds today?
        b) does Approach 1's Step 11 metrics.json agree with its Step 12 row,
           and with any outputs_backup folder on this machine?
        c) was a model file written AFTER the metrics that describe it?
      A failure is printed with what differs and what the reader has to decide.
      Nothing is repaired automatically: a provenance check that quietly fixes
      its own evidence is not a check.

12.0b RETRAINING LOCK.  `LOCK_TRAINED_MODELS = True` makes any training call
      for a configuration whose metrics.json already exists refuse, whatever
      its OVERWRITE_* flag says.  This is a wrapper, not an edit: the training
      functions are untouched and still do exactly what they did.

12.2  SANITY BASELINES.  A majority-class predictor, a stratified random
      predictor and a LABEL-SHUFFLE model -- Approach 1's architecture and
      protocol, trained on permuted fit labels and scored against the real test
      labels.  They are reported in their own table and never mixed into the
      model ranking: they exist to show the floor, and to catch a leak if one
      of them scores far above it.

Nothing here is Keras-3 only; the one model it trains goes through the same
builder and callbacks Step 12 uses, on the Keras the other models were saved
with.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from . import comparison as cmp  # noqa: E402
from . import data_loading as dl  # noqa: E402
from . import evaluation as ev  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = "Label_encoded"
OK, FAIL, SKIP = "[OK]", "[FAIL]", "[SKIP]"


class Step12CheckError(Exception):
    """Raised when a check cannot even be attempted without guessing."""


# ===========================================================================
# shared: the checks table every section adds to
# ===========================================================================
class Checks:
    """One table of [OK]/[FAIL]/[SKIP], with the section each row belongs to."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, section: str, check: str, ok: bool | None, detail: str,
            action: str = "") -> None:
        self.rows.append({"section": section, "check": check,
                          "result": SKIP if ok is None else (OK if ok else FAIL),
                          "detail": detail, "what to do": action})

    def frame(self, section: str | None = None) -> pd.DataFrame:
        table = pd.DataFrame(self.rows)
        if section is not None and len(table):
            table = table[table["section"] == section].reset_index(drop=True)
        return table

    @property
    def failures(self) -> pd.DataFrame:
        table = self.frame()
        return (table[table["result"] == FAIL].reset_index(drop=True)
                if len(table) else table)

    @property
    def ok(self) -> bool:
        table = self.frame()
        return bool(len(table)) and not len(self.failures)


def display_label(key: str) -> str:
    return cmp.display_label(key)


def short_label(key: str) -> str:
    return display_label(key).split(":")[0].strip()


def model_keys() -> list[str]:
    return [key for key, _label in config.STEP12_DISPLAY_ORDER]


# ===========================================================================
# 12.0  PROVENANCE
# ===========================================================================
def _digest(path: Path, limit_mb: float = 512.0) -> str:
    """SHA-256 of a file, read in chunks so a large model does not load twice."""
    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb > limit_mb:                       # pragma: no cover - huge file
        return "(not hashed: %.0f MB)" % size_mb
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime,
                                  tz=timezone.utc).strftime(
                                      "%Y-%m-%d %H:%M:%S UTC")


def result_folder(key: str) -> Path | None:
    """Where this configuration's saved result lives, whichever step wrote it."""
    path = cmp.result_path(key)
    return None if path is None else path.parent


def file_record(key: str, kind: str, path: Path | None) -> dict[str, Any]:
    """One row of provenance.csv: what the file is, and exactly which bytes."""
    label = display_label(key)
    if path is None or not path.exists():
        return {"model": label, "key": key, "file": kind,
                "path": "" if path is None else str(path), "exists": False,
                "size (bytes)": 0, "modified": "", "sha256": ""}
    return {"model": label, "key": key, "file": kind, "path": str(path),
            "exists": True, "size (bytes)": int(path.stat().st_size),
            "modified": _stamp(path), "sha256": _digest(path)}


def provenance_table(keys: Sequence[str] | None = None) -> pd.DataFrame:
    """Model, metrics and predictions for every configuration, fingerprinted.

    This is the table a reader checks a quoted number against: every row in
    the comparison comes from one of these files, and the SHA-256 says which
    bytes produced it.
    """
    keys = model_keys() if keys is None else list(keys)
    rows = []
    for key in keys:
        folder = result_folder(key)
        if folder is None:
            rows.append({"model": display_label(key), "key": key,
                         "file": "(folder not usable here)", "path": "",
                         "exists": False, "size (bytes)": 0, "modified": "",
                         "sha256": ""})
            continue
        candidates = {"metrics": folder / "metrics.json",
                      "predictions": folder / "predictions.csv"}
        # the model is not always beside its metrics: Approach 1 keeps the
        # published model in step10\, and Approach 2 in step10_attention\
        searched = [folder]
        if key == "proposed":
            step10 = config.usable_output_dir(config.STEP10_OUTPUT_DIR)
            if step10 is not None:
                searched.append(Path(step10))
        if key == getattr(config, "APPROACH2_KEY", "approach2_attention"):
            step10b = config.usable_output_dir(
                getattr(config, "STEP10B_OUTPUT_DIR", config.STEP10_OUTPUT_DIR))
            if step10b is not None:
                searched.append(Path(step10b))
        wanted = ["model.keras", config.STEP10_FILES["model"],
                  getattr(config, "STEP10B_FILES", {}).get("model", "")]
        model = None
        for place in searched:
            for name in wanted:
                if name and (place / name).exists():
                    model = place / name
                    break
            if model is not None:
                break
        candidates["model"] = model if model is not None else folder / "model.keras"
        for kind in ("model", "metrics", "predictions"):
            rows.append(file_record(key, kind, candidates[kind]))
    return pd.DataFrame(rows)


def _saved_predictions(key: str) -> pd.DataFrame | None:
    folder = result_folder(key)
    if folder is None:
        return None
    path = folder / "predictions.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def _fused_predictions(which: str) -> pd.DataFrame | None:
    """Step 13's or 13.8's saved per-flow file, or None when it has not run."""
    if which == "step13":
        folder = config.usable_output_dir(config.STEP13_OUTPUT_DIR)
        name = config.STEP13_FILES["fused_predictions"]
    else:
        folder = config.usable_output_dir(
            getattr(config, "STEP13ALL_OUTPUT_DIR", config.STEP13_OUTPUT_DIR))
        name = getattr(config, "STEP13ALL_FILES", {}).get(
            "fused_predictions", "fused_predictions_all.csv")
    if folder is None or not (Path(folder) / name).exists():
        return None
    return pd.read_csv(Path(folder) / name, low_memory=False)


def check_fusion_inputs(checks: Checks, verbose: bool = True) -> pd.DataFrame:
    """(a) Did Step 13 / 13.8 fuse the predictions these models hold TODAY?

    The fused files record each model's attack risk r_m per flow.  Recomputing
    r_m from the configuration's current predictions.csv and comparing per
    Orig_Row answers the question that matters after a retrain: is the fusion
    in the write-up still the fusion these models produce?
    """
    section = "12.0 provenance"
    rows = []
    for which, models in (("step13", list(config.FUSION_MODELS)),
                          ("step13_all", [k for k in model_keys()])):
        fused = _fused_predictions(which)
        if fused is None:
            checks.add(section, "%s fused the current predictions" % which,
                       None, "%s has not been run on this machine" % which,
                       "run it, or ignore this row")
            continue
        for key in models:
            label = display_label(key)
            column = ("r_%s" % key if which == "step13"
                      else "r_%s" % short_label(key))
            if column not in fused.columns:
                continue
            saved = _saved_predictions(key)
            if saved is None or "Orig_Row" not in saved.columns:
                rows.append({"fusion": which, "model": label,
                             "compared rows": 0,
                             "max |r difference|": float("nan"),
                             "class disagreements": -1, "result": SKIP,
                             "detail": "no predictions.csv to compare against"})
                continue
            probability_columns = [c for c in saved.columns
                                   if c.startswith("p_")]
            if not probability_columns:
                continue
            merged = fused[["Orig_Row", column]].merge(
                saved[["Orig_Row", "y_pred"] + probability_columns],
                on="Orig_Row", how="inner")
            if not len(merged):
                rows.append({"fusion": which, "model": label,
                             "compared rows": 0,
                             "max |r difference|": float("nan"),
                             "class disagreements": -1, "result": FAIL,
                             "detail": "no Orig_Row in common"})
                continue
            probabilities = merged[probability_columns].to_numpy(dtype="float64")
            benign_index = [i for i, name in enumerate(probability_columns)
                            if _is_benign_column(name)]
            if not benign_index:
                rows.append({"fusion": which, "model": label,
                             "compared rows": int(len(merged)),
                             "max |r difference|": float("nan"),
                             "class disagreements": -1, "result": SKIP,
                             "detail": "no benign class among %s, so the "
                                       "attack risk cannot be recomputed"
                                       % ", ".join(probability_columns)})
                continue
            risk_now = 1.0 - probabilities[:, benign_index].sum(axis=1)
            difference = float(np.abs(merged[column].to_numpy(dtype="float64")
                                      - risk_now).max())
            ok = difference <= 1e-6
            rows.append({"fusion": which, "model": label,
                         "compared rows": int(len(merged)),
                         "max |r difference|": difference,
                         "class disagreements": 0 if ok else -1,
                         "result": OK if ok else FAIL,
                         "detail": ("the fused file and predictions.csv agree"
                                    if ok else
                                    "the fused risk differs by %.3e, so %s was "
                                    "retrained or re-evaluated after %s ran"
                                    % (difference, label, which))})
    table = pd.DataFrame(rows)
    if len(table):
        failed = table[table["result"] == FAIL]
        checks.add(section, "Step 13 / 13.8 fused the predictions these models "
                            "hold now", not len(failed),
                   "%d of %d model-fusion pairs agree to 1e-6"
                   % (int((table["result"] == OK).sum()), len(table)),
                   "" if not len(failed) else
                   "re-run Step 13 and 13.8 against the current models "
                   "(OVERWRITE_STEP13 / OVERWRITE_STEP13ALL = True), or "
                   "restore the models these fusions were computed from")
    if verbose and len(table):
        show_table(table, all_columns=True,
                   note="Each fused file records r_m per flow. This recomputes "
                        "r_m from the configuration's CURRENT predictions.csv "
                        "and compares per Orig_Row.")
    return table


def _is_benign_column(column: str) -> bool:
    """Is this `p_<label>` column one of the benign classes?

    By NAME, because a saved predictions.csv carries the class labels but not
    `Label_binary`; `config.BENIGN_LABELS` is the same list every other step
    decides this with.
    """
    name = str(column[2:]).strip().lower()
    return name in {str(value).strip().lower()
                    for value in config.BENIGN_LABELS}


def backup_folders() -> list[Path]:
    """Any outputs_backup* folder under the data root, newest first."""
    root = config.usable_output_dir(config.DATA_ROOT)
    if root is None or not Path(root).exists():
        return []
    prefix = getattr(config, "BACKUP_PREFIX", "outputs_backup")
    folders = [p for p in Path(root).iterdir()
               if p.is_dir() and p.name.startswith(prefix)]
    return sorted(folders, key=lambda p: p.stat().st_mtime, reverse=True)


def check_approach1_consistency(checks: Checks, verbose: bool = True
                                ) -> pd.DataFrame:
    """(b) Approach 1's Step 11 metrics, its Step 12 row, and any backup.

    Side by side, with the differences named.  No value is chosen here: which
    of them is the right one is a decision about what was actually run, and
    only the person who ran it can make it.
    """
    section = "12.0 provenance"
    wanted = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
              "f1_weighted", "g_mean"]
    sources: dict[str, dict[str, Any]] = {}

    folder11 = config.usable_output_dir(config.STEP11_OUTPUT_DIR)
    step11_path = None
    if folder11 is not None and (Path(folder11) / "metrics.json").exists():
        step11_path = Path(folder11) / "metrics.json"
        sources["Step 11 (current)"] = json.loads(
            step11_path.read_text(encoding="utf-8"))
    # The Step 12 row counts as a SECOND source only when it is a different
    # file.  `comparison.result_path("proposed")` falls back to Step 11's
    # metrics.json when the Step 12 runner has not written its own copy, and
    # comparing that file with itself would report a vacuous [OK] for a
    # consistency check that has nothing to compare.
    current = cmp.load_result("proposed")
    current_path = cmp.result_path("proposed")
    same_file = (step11_path is not None and current_path is not None
                 and Path(current_path) == step11_path)
    if current is not None and not same_file:
        sources["Step 12 row (current)"] = current
    for backup in backup_folders():
        for candidate in (backup / "step11" / "metrics.json",
                          backup / "outputs" / "step11" / "metrics.json"):
            if candidate.exists():
                sources["backup: %s" % backup.name] = json.loads(
                    candidate.read_text(encoding="utf-8"))
                break

    rows = []
    for name, metrics in sources.items():
        full = metrics.get("full") or {}
        row = {"source": name}
        for metric in wanted:
            row[metric] = full.get(metric)
        row["training_seconds"] = metrics.get("training_seconds")
        row["best_epoch"] = metrics.get("best_epoch")
        rows.append(row)
    table = pd.DataFrame(rows)

    if len(table) < 2:
        checks.add(section, "Approach 1's metrics agree across the places that "
                            "hold them", None,
                   "only %d source found (%s)"
                   % (len(table), ", ".join(sources) or "none"),
                   "nothing to compare; this is normal if no backup exists")
    else:
        numbers = table[wanted].astype("float64")
        spread = float(np.nanmax(numbers.to_numpy(), axis=0).max()
                       - np.nanmin(numbers.to_numpy(), axis=0).min())
        differing = [metric for metric in wanted
                     if float(np.nanmax(numbers[metric]) -
                              np.nanmin(numbers[metric])) > 1e-9]
        checks.add(section,
                   "Approach 1's metrics agree across the places that hold them",
                   not differing,
                   "%d sources compared: %s" % (len(table), ", ".join(sources))
                   + ("" if not differing else
                      "; they differ on %s (largest spread %.4f)"
                      % (", ".join(differing), spread)),
                   "" if not differing else
                   "decide which run the write-up quotes: either restore the "
                   "backup whose numbers the text uses, or re-run Steps 13-14 "
                   "against the CURRENT Approach 1 so every table agrees")
    if verbose and len(table):
        show_table(table, all_columns=True,
                   note="Approach 1's headline metrics from every place this "
                        "machine holds them. Nothing is chosen automatically.")
    return table


def check_model_timestamps(checks: Checks, provenance: pd.DataFrame,
                           verbose: bool = True) -> pd.DataFrame:
    """(c) A model file written AFTER the metrics that describe it.

    That ordering is the signature of a silent retrain: the weights moved, the
    numbers beside them did not.
    """
    section = "12.0 provenance"
    rows = []
    for key in model_keys():
        part = provenance[(provenance["key"] == key)
                          & (provenance["exists"])]
        files = dict(zip(part["file"], part["modified"]))
        if "model" not in files or "metrics" not in files:
            continue
        card = _model_card(key)
        rows.append({
            "model": display_label(key),
            "model file written": files["model"],
            "metrics written": files["metrics"],
            "model newer than metrics": files["model"] > files["metrics"],
            "epochs in history": card.get("epochs_run"),
            "training seconds recorded": card.get("training_seconds"),
            "result": FAIL if files["model"] > files["metrics"] else OK,
        })
    table = pd.DataFrame(rows)
    if len(table):
        failed = table[table["result"] == FAIL]
        checks.add(section, "no model file is newer than the metrics that "
                            "describe it", not len(failed),
                   "%d configurations checked" % len(table)
                   + ("" if not len(failed) else "; newer model: %s"
                      % ", ".join(failed["model"])),
                   "" if not len(failed) else
                   "that model was rewritten after its metrics were saved; "
                   "re-evaluate it (Step 11/12) or restore the model the "
                   "metrics came from")
    if verbose and len(table):
        show_table(table, all_columns=True,
                   note="A model written after its own metrics means the saved "
                        "numbers describe different weights.")
    return table


def _model_card(key: str) -> dict[str, Any]:
    """Whatever the configuration recorded about its training run."""
    metrics = cmp.load_result(key) or {}
    folder = result_folder(key)
    card: dict[str, Any] = {}
    if folder is not None:
        for name in ("model_card.json", "ae_config.json"):
            path = Path(folder) / name
            if path.exists():
                card.update(json.loads(path.read_text(encoding="utf-8")))
    for field in ("epochs_run", "best_epoch", "training_seconds", "seed",
                  "best_val_macro_f1", "parameters"):
        if field in metrics and field not in card:
            card[field] = metrics[field]
    return card


def run_provenance(keys: Sequence[str] | None = None, verbose: bool = True
                   ) -> dict[str, Any]:
    """12.0 end to end: fingerprint everything, then ask the three questions."""
    checks = Checks()
    if verbose:
        show_title("12.0 -- provenance: which saved file is each number from?",
                   level=4)
    provenance = provenance_table(keys)
    present = provenance[provenance["exists"]]
    checks.add("12.0 provenance", "every configuration has a saved result",
               bool(len(present[present["file"] == "metrics"])
                    == len(model_keys())),
               "%d of %d metrics.json files found"
               % (len(present[present["file"] == "metrics"]),
                  len(model_keys())),
               "train the missing ones; the comparison leaves their row blank")
    if verbose:
        show_table(provenance[["model", "file", "exists", "size (bytes)",
                               "modified", "sha256"]].assign(
            sha256=lambda f: f["sha256"].str.slice(0, 16)),
            all_columns=True,
            note="Every published number comes from one of these files. The "
                 "SHA-256 is truncated here; the full digest is in "
                 "provenance.csv.")

    fusion = check_fusion_inputs(checks, verbose=verbose)
    approach1 = check_approach1_consistency(checks, verbose=verbose)
    timestamps = check_model_timestamps(checks, provenance, verbose=verbose)

    out = {"provenance": provenance, "fusion_inputs": fusion,
           "approach1": approach1, "timestamps": timestamps,
           "checks": checks.frame(), "failures": checks.failures,
           "ok": checks.ok}
    out["report"] = provenance_report(out)
    saved = save_provenance(out, verbose=verbose)
    out["saved"] = saved
    out["folder"] = saved.get("folder")
    if verbose:
        show_table(checks.frame()[["check", "result", "detail", "what to do"]],
                   all_columns=True,
                   note="12.0's own verdict. A [FAIL] here is not fixed "
                        "automatically: the row says what differs and what the "
                        "decision is.")
        print("[%s]  provenance: %d checks, %d failed"
              % ("OK" if checks.ok else "FAIL", len(checks.frame()),
                 len(checks.failures)))
    return out


def provenance_report(out: dict[str, Any]) -> str:
    """The markdown a supervisor reads before any table."""
    lines = ["# Step 12.0 -- provenance report", "",
             "Generated from the files on this machine. Every number in Step "
             "12 comes from a file listed in `provenance.csv`; the SHA-256 "
             "there identifies the exact bytes.", ""]
    provenance = out["provenance"]
    found = provenance[provenance["exists"]]
    lines.append("- **%d files fingerprinted** across %d configurations."
                 % (len(found), provenance["key"].nunique()))
    missing = provenance[(~provenance["exists"])
                         & (provenance["file"] == "metrics")]
    if len(missing):
        lines.append("- **Missing results:** %s -- those rows stay blank in "
                     "the comparison." % ", ".join(missing["model"]))
    failures = out["failures"]
    lines.append("")
    if not len(failures):
        lines.append("**Every consistency check passed.** The fused files, the "
                     "saved metrics and the model timestamps agree.")
    else:
        lines.append("**%d check(s) failed. Nothing was repaired "
                     "automatically.**" % len(failures))
        for _i, row in failures.iterrows():
            lines.append("")
            lines.append("### %s" % row["check"])
            lines.append("- What differs: %s" % row["detail"])
            if row["what to do"]:
                lines.append("- Decision: %s" % row["what to do"])
    return "\n".join(lines)


def save_provenance(out: dict[str, Any], verbose: bool = True
                    ) -> dict[str, Any]:
    """Write 12.0's evidence beside the Step 12 outputs."""
    names = config.STEP12_PROVENANCE_FILES
    folder = prepare_output_dir(config.STEP12_OUTPUT_DIR,
                                "the Step 12.0 provenance files",
                                verbose=verbose)
    if folder is None:
        return {"status": "skipped_not_windows", "ok": False}
    folder = Path(folder)
    out["provenance"].to_csv(folder / names["provenance"], index=False)
    out["checks"].to_csv(folder / names["consistency"], index=False)
    if len(out["approach1"]):
        out["approach1"].to_csv(folder / names["approach1"], index=False)
    (folder / names["report"]).write_text(out["report"], encoding="utf-8")
    if verbose:
        print("  [SAVE]  %s" % (folder / names["provenance"]))
    return {"status": "written", "folder": str(folder),
            "ok": bool(out["ok"])}


# ===========================================================================
# 12.0b  THE RETRAINING LOCK
# ===========================================================================
def is_trained(key: str) -> bool:
    """True when this configuration already has a saved metrics.json."""
    path = cmp.result_path(key)
    return bool(path is not None and path.exists())


def locked(key: str) -> bool:
    """Would a training call for this configuration be refused right now?"""
    return bool(getattr(config, "LOCK_TRAINED_MODELS", False)
                and is_trained(key))


def lock_message(key: str) -> str:
    path = cmp.result_path(key)
    return (
        "LOCK_TRAINED_MODELS is True and %s already has a saved result:\n"
        "    %s\n"
        "Training it again would change the model the published tables, the "
        "Step 13 fusion and the Step 14 explanations were all computed from, "
        "and the numbers beside it would no longer describe it.\n"
        "To retrain on purpose: set LOCK_TRAINED_MODELS = False in config.py, "
        "re-run this cell, and then re-run Steps 11-14 so every table agrees "
        "again." % (display_label(key), path))


def train_guard(key: str, trainer: Callable[..., Any], *args: Any,
                verbose: bool = True, **kwargs: Any) -> Any:
    """Call `trainer` only if the lock allows it.

    A wrapper, never an edit: the training functions are untouched and still
    honour their own OVERWRITE_* flags.  This sits in front of them so that
    "the flag was left on" cannot quietly cost a published model.
    """
    if locked(key):
        raise Step12CheckError(lock_message(key))
    if verbose:
        print("  [LOCK]  %s: %s"
              % (display_label(key),
                 "no saved result yet, training is allowed"
                 if not is_trained(key) else
                 "LOCK_TRAINED_MODELS is False, so a retrain is allowed"))
    return trainer(*args, **kwargs)


def lock_table(keys: Sequence[str] | None = None) -> pd.DataFrame:
    """What the lock would do for every configuration, right now."""
    keys = model_keys() if keys is None else list(keys)
    rows = []
    for key in keys:
        rows.append({
            "model": display_label(key),
            "has a saved result": is_trained(key),
            "training now would": ("be REFUSED" if locked(key)
                                   else "be allowed"),
            "why": ("LOCK_TRAINED_MODELS = True and metrics.json exists"
                    if locked(key) else
                    "no saved metrics.json yet" if not is_trained(key)
                    else "LOCK_TRAINED_MODELS = False")})
    return pd.DataFrame(rows)


# ===========================================================================
# 12.2  SANITY BASELINES -- the floor the models have to clear
# ===========================================================================
def _scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
            label_names: dict[int, str]) -> dict[str, float]:
    """Accuracy, macro-F1 and G-mean through Step 11's own evaluator."""
    headline = ev.headline_metrics(y_true, y_pred, None, n_classes)
    return {"accuracy": float(headline.get("accuracy") or 0.0),
            "F1 (macro)": float(headline.get("f1_macro") or 0.0),
            "G-mean": float(headline.get("g_mean") or 0.0)}


def majority_class_predictions(y_train: pd.DataFrame, n_rows: int) -> np.ndarray:
    """Always the most frequent TRAINING class -- the accuracy floor."""
    code = int(pd.Series(y_train[TARGET_COLUMN]).value_counts().idxmax())
    return np.full(int(n_rows), code, dtype="int64")


def stratified_predictions(y_train: pd.DataFrame, n_rows: int,
                           seed: int | None = None) -> np.ndarray:
    """sklearn's stratified dummy: draws from the TRAINING class shares."""
    from sklearn.dummy import DummyClassifier

    seed = config.RANDOM_STATE if seed is None else seed
    labels = np.asarray(y_train[TARGET_COLUMN])
    dummy = DummyClassifier(strategy="stratified", random_state=seed)
    dummy.fit(np.zeros((len(labels), 1)), labels)
    return np.asarray(dummy.predict(np.zeros((int(n_rows), 1))),
                      dtype="int64")


def class_shares(y_test: pd.DataFrame, label_names: dict[int, str]
                 ) -> pd.DataFrame:
    """How the test set is split between classes -- why accuracy says little."""
    counts = pd.Series(y_test[TARGET_COLUMN]).value_counts().sort_values(
        ascending=False)
    total = int(counts.sum())
    return pd.DataFrame([{
        "class": label_names.get(int(code), str(code)),
        "test rows": int(count),
        "share (%)": round(100.0 * count / total, 2)}
        for code, count in counts.items()])


def shuffle_fit_labels(y_train: pd.DataFrame, seed: int | None = None
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permute the labels of the Step 9 FIT rows, leaving validation alone.

    The split itself is taken from the REAL labels first, so the fit and
    validation rows are the ones every other configuration trained on; only
    the labels attached to the fit rows are permuted.  Returns (y_train with
    the fit labels permuted, a record of the permutation).
    """
    from . import imbalance as im

    seed = config.RANDOM_STATE if seed is None else seed
    marker = pd.DataFrame({"row": np.arange(len(y_train))})
    fit_rows, _y_fit, val_rows, _y_val, _w, _s = im.make_training_sets(
        marker, y_train, verbose=False)
    positions = np.asarray(fit_rows["row"])

    rng = np.random.default_rng(seed)
    permuted = y_train.copy()
    original = np.asarray(y_train[TARGET_COLUMN])[positions]
    shuffled = rng.permutation(original)
    values = np.asarray(y_train[TARGET_COLUMN]).copy()
    values[positions] = shuffled
    permuted[TARGET_COLUMN] = values
    if "Label_binary" in permuted.columns:
        binary = np.asarray(y_train["Label_binary"]).copy()
        binary[positions] = (shuffled != _benign_code(y_train)).astype("int64")
        permuted["Label_binary"] = binary

    record = pd.DataFrame({
        "fit row": positions,
        "true label": original,
        "shuffled label": shuffled,
        "unchanged": original == shuffled})
    return permuted, record


def _benign_code(y_train: pd.DataFrame) -> int:
    if "Label_binary" in y_train.columns:
        benign = y_train.loc[y_train["Label_binary"] == 0, TARGET_COLUMN]
        if len(benign):
            return int(benign.iloc[0])
    return 0


def train_label_shuffle(X_train: pd.DataFrame, y_train: pd.DataFrame,
                        seed: int | None = None, epochs: int | None = None,
                        verbose: bool = True) -> dict[str, Any]:
    """Approach 1's architecture and protocol, on permuted FIT labels.

    Everything is the Step 12 protocol -- `experiments.train_classifier`'s own
    pieces, the same builder, the same class weighting, the same unweighted
    validation and the same early stopping -- with exactly one change: the fit
    labels are a permutation of themselves.  A model that still scores well has
    found the label somewhere other than the features, which is what this is
    for.

    The split is computed from the REAL labels before the permutation, so the
    rows are the ones every other configuration fitted on.
    """
    from . import experiments as xp
    from . import imbalance as im
    from . import models as md

    keras = md._keras()
    seed = config.RANDOM_STATE if seed is None else seed
    epochs = config.CLF_EPOCHS if epochs is None else epochs
    md.set_all_seeds(seed)

    X_fit, y_fit, X_val, y_val, class_weights, _sw = im.make_training_sets(
        X_train, y_train, verbose=False)
    X_fit_arr, labels_fit, _i1 = md.arrange_inputs(X_fit, y_fit, None,
                                                   verbose=False)
    X_val_arr, labels_val, _i2 = md.arrange_inputs(X_val, y_val, None,
                                                   verbose=False)

    rng = np.random.default_rng(seed)
    shuffled_fit = rng.permutation(np.asarray(labels_fit))
    record = pd.DataFrame({"fit position": np.arange(len(labels_fit)),
                           "true label": np.asarray(labels_fit),
                           "shuffled label": shuffled_fit})

    n_classes = int(max(int(np.max(labels_fit)), int(np.max(labels_val))) + 1)
    model = xp.builder_for("cnn_lstm")(X_fit_arr.shape[1:], n_classes)
    macro_f1 = md._macro_f1_callback_class()(X_val_arr, labels_val,
                                             config.CLF_BATCH_SIZE)
    stopper = keras.callbacks.EarlyStopping(
        monitor=config.EARLY_STOP_METRIC, mode="max",
        patience=config.CLF_PATIENCE, restore_best_weights=True, verbose=0)

    started = time.perf_counter()
    history = model.fit(
        X_fit_arr, shuffled_fit,
        validation_data=(X_val_arr, labels_val),          # real, unweighted
        epochs=epochs, batch_size=config.CLF_BATCH_SIZE, shuffle=True,
        class_weight=im.keras_class_weight(class_weights),
        callbacks=[macro_f1, stopper], verbose=0)
    seconds = time.perf_counter() - started

    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    if verbose:
        print("  label-shuffle model trained: %d epochs, %.1f s (fit labels "
              "permuted, validation and test labels untouched)"
              % (len(frame), seconds))
    return {"model": model, "history": frame, "permutation": record,
            "seconds": seconds, "seed": seed, "n_classes": n_classes,
            "fit_rows": int(len(labels_fit)),
            "validation_rows": int(len(labels_val))}


def run_sanity(checks: Checks | None = None, overwrite: bool | None = None,
               epochs: int | None = None, verbose: bool = True
               ) -> dict[str, Any]:
    """12.2 end to end: the two dummies, the label-shuffle model, the verdict."""
    from . import experiments as xp
    from . import models as md

    checks = Checks() if checks is None else checks
    section = "12.2 sanity"
    if verbose:
        show_title("12.2 -- sanity baselines: what the floor looks like",
                   level=4)

    loaded = ev.load_evaluation_inputs(verbose=False)
    y_test, meta_test = loaded["y_test"], loaded.get("meta_test")
    label_names = ev.label_names_from_mapping(loaded["label_mapping"])
    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder8 is None:
        raise Step12CheckError("the Step 8 outputs are not readable here -- %s"
                               % config.why_unusable(config.STEP8_OUTPUT_DIR))
    y_train = pd.read_csv(Path(folder8) / config.STEP8_FILES["y_train"],
                          low_memory=False)
    n_classes = int(max(int(y_train[TARGET_COLUMN].max()),
                        int(y_test[TARGET_COLUMN].max())) + 1)
    duplicate_flags = np.zeros(len(y_test), dtype=bool)
    if loaded["X_train"] is not None and loaded["X_test"] is not None:
        duplicate_flags = ev.duplicate_mask(loaded["X_train"],
                                            loaded["X_test"])
    clean = ~np.asarray(duplicate_flags, dtype=bool)
    truth = np.asarray(y_test[TARGET_COLUMN])

    rows = []

    def add_row(name: str, predictions: np.ndarray, note: str) -> dict:
        full = _scores(truth, predictions, n_classes, label_names)
        free = (_scores(truth[clean], predictions[clean], n_classes,
                        label_names) if clean.any() else
                {"accuracy": float("nan"), "F1 (macro)": float("nan"),
                 "G-mean": float("nan")})
        row = {"baseline": name,
               "accuracy": round(full["accuracy"], 4),
               "F1 (macro)": round(full["F1 (macro)"], 4),
               "G-mean": round(full["G-mean"], 4),
               "accuracy (dup-free)": round(free["accuracy"], 4),
               "F1 macro (dup-free)": round(free["F1 (macro)"], 4),
               "G-mean (dup-free)": round(free["G-mean"], 4),
               "what it is": note}
        rows.append(row)
        return row

    majority = add_row("majority class (most frequent training class)",
                       majority_class_predictions(y_train, len(y_test)),
                       "predicts one class for every flow")
    random_row = add_row("stratified random (seed %d)" % config.RANDOM_STATE,
                         stratified_predictions(y_train, len(y_test)),
                         "draws from the training class shares")

    # --- the label-shuffle model -----------------------------------------
    shuffle_row = None
    folder = config.usable_output_dir(config.STEP12_SANITY_OUTPUT_DIR)
    names = config.SANITY_FILES
    overwrite = (getattr(config, "OVERWRITE_STEP12_SANITY", True)
                 if overwrite is None else overwrite)
    saved_metrics = None
    if folder is not None and (Path(folder) / names["metrics"]).exists() \
            and not overwrite:
        saved_metrics = json.loads(
            (Path(folder) / names["metrics"]).read_text(encoding="utf-8"))

    if saved_metrics is not None:
        full = saved_metrics.get("full") or {}
        free = saved_metrics.get("duplicate_free") or {}
        shuffle_row = {"baseline": "label-shuffle model (loaded)",
                       "accuracy": round(float(full.get("accuracy") or 0), 4),
                       "F1 (macro)": round(float(full.get("f1_macro") or 0), 4),
                       "G-mean": round(float(full.get("g_mean") or 0), 4),
                       "accuracy (dup-free)": round(
                           float(free.get("accuracy") or float("nan")), 4),
                       "F1 macro (dup-free)": round(
                           float(free.get("f1_macro") or float("nan")), 4),
                       "G-mean (dup-free)": round(
                           float(free.get("g_mean") or float("nan")), 4),
                       "what it is": "Approach 1's architecture on permuted "
                                     "fit labels"}
        rows.append(shuffle_row)
        trained = {"loaded": True}
    else:
        source = xp.load_feature_source("selected30", verbose=False)
        from . import autoencoder as ae
        encode, _encoder, order = ae.load_encoder(verbose=False)
        Z_train = encode(source["X_train"][order], what="training features")
        Z_test = encode(source["X_test"][order], what="test features")
        trained = train_label_shuffle(Z_train, y_train, epochs=epochs,
                                      verbose=verbose)
        X_test_arr, _labels, row_index = md.arrange_inputs(Z_test, y_test,
                                                           meta_test,
                                                           verbose=False)
        result = ev.evaluate(trained["model"], X_test_arr, y_test,
                             loaded["label_mapping"], meta_test=meta_test,
                             duplicate_flags=duplicate_flags,
                             model_name="label-shuffle sanity model",
                             training_seconds=trained["seconds"], verbose=False)
        full, free = result["full"], result.get("duplicate_free") or {}
        shuffle_row = {"baseline": "label-shuffle model (trained here)",
                       "accuracy": round(float(full.get("accuracy") or 0), 4),
                       "F1 (macro)": round(float(full.get("f1_macro") or 0), 4),
                       "G-mean": round(float(full.get("g_mean") or 0), 4),
                       "accuracy (dup-free)": round(
                           float(free.get("accuracy") or float("nan")), 4),
                       "F1 macro (dup-free)": round(
                           float(free.get("f1_macro") or float("nan")), 4),
                       "G-mean (dup-free)": round(
                           float(free.get("g_mean") or float("nan")), 4),
                       "what it is": "Approach 1's architecture and protocol, "
                                     "fit labels permuted"}
        rows.append(shuffle_row)
        trained["result"] = result
        trained["row_index"] = row_index

    table = pd.DataFrame(rows)

    # --- the verdict ------------------------------------------------------
    margin = float(getattr(config, "SANITY_LEAKAGE_MARGIN", 0.10))
    reference = float(max(random_row["F1 (macro)"], majority["F1 (macro)"]))
    leak = float(shuffle_row["F1 (macro)"]) > reference + margin
    checks.add(section, "the label-shuffle model stays near the random floor",
               not leak,
               "label-shuffle macro-F1 %.4f against a random floor of %.4f "
               "(allowed margin %.2f)"
               % (shuffle_row["F1 (macro)"], reference, margin),
               "" if not leak else
               "POSSIBLE LEAKAGE: a model trained on permuted labels should "
               "not beat chance. Check that no identifier column reaches the "
               "features and that the scaler, the selection and the "
               "autoencoder were all fitted on training rows only")
    checks.add(section, "accuracy alone cannot separate these models", True,
               "the majority class alone scores %.4f accuracy on the test set"
               % majority["accuracy"],
               "quote macro-F1 and G-mean, not accuracy")

    shares = class_shares(y_test, label_names)
    out = {"table": table, "class_shares": shares, "checks": checks.frame(),
           "failures": checks.failures, "leak": leak,
           "majority_accuracy": float(majority["accuracy"]),
           "random_macro_f1": float(random_row["F1 (macro)"]),
           "shuffle_macro_f1": float(shuffle_row["F1 (macro)"]),
           "trained": trained, "n_classes": n_classes,
           "label_names": label_names}
    out["report"] = sanity_report(out)
    out["saved"] = save_sanity(out, verbose=verbose)
    if verbose:
        show_table(table, all_columns=True,
                   note="Sanity baselines ONLY. These are never part of the "
                        "model ranking: they are the floor the models have to "
                        "clear, and the leakage test.")
        show_table(shares, note="Why accuracy says little here.")
        print("[%s]  sanity: label-shuffle macro-F1 %.4f vs random floor %.4f"
              % ("OK" if not leak else "FAIL", out["shuffle_macro_f1"],
                 out["random_macro_f1"]))
    return out


def sanity_report(out: dict[str, Any]) -> str:
    """The markdown under the sanity table, from the computed numbers."""
    shares = out["class_shares"]
    biggest = shares.head(2)
    lines = ["### Why accuracy is similar for every model", ""]
    lines.append("The test set is dominated by its largest classes: **%s** "
                 "together are %.1f%% of the test rows."
                 % (" and ".join(biggest["class"]),
                    float(biggest["share (%)"].sum())))
    lines.append("")
    lines.append("A predictor that answers with the single most frequent "
                 "training class and nothing else already reaches **%.4f "
                 "accuracy**, with a macro-F1 of %.4f and a G-mean of %.4f -- "
                 "it never finds a single attack class."
                 % (out["majority_accuracy"],
                    float(out["table"].iloc[0]["F1 (macro)"]),
                    float(out["table"].iloc[0]["G-mean"])))
    lines.append("")
    lines.append("That is why **macro-F1 and G-mean are the comparison "
                 "metrics** in Step 12: they weight every class equally, so a "
                 "model that misses a rare attack cannot hide behind the "
                 "majority class. Accuracy is reported for completeness and is "
                 "read against the %.4f reference line."
                 % out["majority_accuracy"])
    lines.append("")
    lines.append("### The leakage test")
    lines.append("")
    lines.append("The label-shuffle model is Approach 1's architecture and "
                 "protocol with the fit labels permuted. It scores **%.4f "
                 "macro-F1** against a random floor of %.4f."
                 % (out["shuffle_macro_f1"], out["random_macro_f1"]))
    if out["leak"]:
        lines.append("")
        lines.append("**[FAIL] That is far above the floor.** A model trained "
                     "on permuted labels cannot learn the task; scoring well "
                     "means the label is reachable from somewhere other than "
                     "the features. Check the identifier columns, the scaler, "
                     "the feature selection and the autoencoder for anything "
                     "fitted on more than the training rows.")
    else:
        lines.append("")
        lines.append("It stays at the floor, which is what it should do: the "
                     "features carry no copy of the label.")
    return "\n".join(lines)


def save_sanity(out: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Write 12.2's table, model and evidence into step12_sanity\\."""
    names = config.SANITY_FILES
    folder = prepare_output_dir(config.STEP12_SANITY_OUTPUT_DIR,
                                "the Step 12.2 sanity outputs", verbose=verbose)
    if folder is None:
        return {"status": "skipped_not_windows", "ok": False}
    folder = Path(folder)
    out["table"].to_csv(folder / names["table"], index=False)
    (folder / names["report"]).write_text(out["report"], encoding="utf-8")
    trained = out.get("trained") or {}
    if trained.get("model") is not None:
        trained["model"].save(folder / names["model"])
        trained["history"].to_csv(folder / names["history"], index=False)
        trained["permutation"].to_csv(folder / names["permutation"],
                                      index=False)
        result = trained.get("result") or {}
        if result:
            (folder / names["metrics"]).write_text(json.dumps({
                "name": "label-shuffle sanity model",
                "architecture": "Approach 1 CNN-LSTM, identical protocol",
                "what was permuted": "the Step 9 fit labels only",
                "seed": trained.get("seed"), "fit_rows": trained.get("fit_rows"),
                "validation_rows": trained.get("validation_rows"),
                "training_seconds": trained.get("seconds"),
                "full": result.get("full"),
                "duplicate_free": result.get("duplicate_free"),
                "binary": result.get("binary"),
            }, indent=2, default=float), encoding="utf-8")
            ev.build_predictions_frame(result, None,
                                       trained.get("row_index")).to_csv(
                folder / names["predictions"], index=False)
    if verbose:
        print("  [SAVE]  %s" % (folder / names["table"]))
    return {"status": "written", "folder": str(folder), "ok": True}


# ===========================================================================
# 12.10  EXTRA SEEDS -- only when they were asked for
# ===========================================================================
def seed_folder(key: str, seed: int) -> Path | None:
    place = config.usable_output_dir(config.STEP12_SEEDS_OUTPUT_DIR)
    return None if place is None else Path(place) / key / ("seed%d" % seed)


def _feature_space(key: str) -> dict[str, Any] | None:
    """(X_train, X_test, model_type) in one configuration's classifier space.

    Built from the loaders the other steps already use, so a re-seeded run
    starts from exactly the features its seed-0 run started from.  Approach 2
    is deliberately absent: its classifier is not one of the shared builders,
    so re-seeding it belongs to section 10.2 rather than here.
    """
    from . import experiments_baseline6 as b6
    from . import experiments_pca as xpca
    from . import fusion_all as fa

    approach2 = getattr(config, "APPROACH2_KEY", "approach2_attention")
    baseline6 = getattr(config, "BASELINE6_NAME", "baseline6_ae_pca_cnn_lstm")
    if key == approach2:
        return None
    if key == baseline6:
        pipeline = b6.load_baseline4_pipeline(verbose=False)
        from . import experiments as xp
        source = xp.load_feature_source(
            config.BASELINE6_SPEC["feature_source"], verbose=False)
        order = pipeline["order"]
        Z_train = pipeline["encode"](source["X_train"][order], what="train")
        Z_test = pipeline["encode"](source["X_test"][order], what="test")
        return {"X_train": xpca.transform_with_pca(pipeline["pca"], Z_train,
                                                   "the training latents"),
                "X_test": xpca.transform_with_pca(pipeline["pca"], Z_test,
                                                  "the test latents"),
                "model_type": "cnn_lstm"}
    bundle = fa.load_bundle(key, verbose=False)
    spec = bundle.get("spec") or {}
    return {"X_train": bundle["X_train"], "X_test": bundle["X_test"],
            "model_type": spec.get("model_type", "cnn_lstm")}


def run_seeds(seeds: Sequence[int] | None = None, epochs: int | None = None,
              verbose: bool = True) -> dict[str, Any]:
    """Retrain each configuration on extra seeds, into folders of their own.

    Only runs when `N_SEEDS >= 3`, because two runs cannot say anything about
    spread.  Seed 0's folders are never touched: every re-seeded run is written
    under `step12_seeds\\<key>\\seed<N>\\`.
    """
    from . import experiments as xp
    from . import models as md

    n_seeds = int(getattr(config, "N_SEEDS", 1))
    if seeds is None:
        seeds = [config.RANDOM_STATE + i for i in range(1, max(n_seeds, 1))]
    if n_seeds < 3:
        if verbose:
            show_note("`N_SEEDS = %d`, so no extra seed is trained. Every "
                      "number in Step 12 is a **single run**, and a difference "
                      "of about 0.01 in macro-F1 may not be meaningful."
                      % n_seeds)
        return {"status": "skipped", "n_seeds": n_seeds, "table": pd.DataFrame(),
                "summary": pd.DataFrame(), "skipped": []}

    loaded = ev.load_evaluation_inputs(verbose=False)
    y_test, meta_test = loaded["y_test"], loaded.get("meta_test")
    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    y_train = pd.read_csv(Path(folder8) / config.STEP8_FILES["y_train"],
                          low_memory=False)
    duplicate_flags = None
    if loaded["X_train"] is not None and loaded["X_test"] is not None:
        duplicate_flags = ev.duplicate_mask(loaded["X_train"],
                                            loaded["X_test"])

    rows, skipped = [], []
    for key in model_keys():
        try:
            space = _feature_space(key)
        except Exception as problem:
            skipped.append((key, str(problem).split("\n")[0]))
            continue
        if space is None:
            skipped.append((key, "its classifier is not one of the shared "
                                 "builders; re-seed it in section 10.2"))
            continue
        base = cmp.load_result(key)
        if base is not None:
            rows.append({"model": display_label(key), "seed": config.RANDOM_STATE,
                         "F1 (macro)": float(base["full"]["f1_macro"]),
                         "G-mean": float(base["full"].get("g_mean") or np.nan),
                         "binary FPR": float((base.get("binary") or {})
                                             .get("false_positive_rate", np.nan)),
                         "source": "seed 0 (the published run)"})
        for seed in seeds:
            place = seed_folder(key, int(seed))
            if place is None:
                continue
            place.mkdir(parents=True, exist_ok=True)
            metrics_path = place / "metrics.json"
            if metrics_path.exists():
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                model, history, info = xp.train_classifier(
                    space["X_train"], y_train, model_type=space["model_type"],
                    seed=int(seed), epochs=epochs, verbose=False)
                X_test_arr, _labels, row_index = md.arrange_inputs(
                    space["X_test"], y_test, meta_test, verbose=False)
                flags = (np.asarray(duplicate_flags)[row_index]
                         if duplicate_flags is not None
                         and len(row_index) != len(duplicate_flags)
                         else duplicate_flags)
                result = ev.evaluate(model, X_test_arr, y_test,
                                     loaded["label_mapping"],
                                     meta_test=meta_test, duplicate_flags=flags,
                                     model_name="%s seed %d"
                                                % (display_label(key), seed),
                                     training_seconds=info.get("seconds"),
                                     verbose=False)
                saved = {"config": key, "seed": int(seed),
                         "full": result["full"],
                         "duplicate_free": result["duplicate_free"],
                         "binary": result["binary"],
                         "training_seconds": info.get("seconds")}
                metrics_path.write_text(json.dumps(saved, indent=2,
                                                   default=float),
                                        encoding="utf-8")
                history.to_csv(place / "history.csv", index=False)
                if verbose:
                    print("  trained %-28s seed %d: macro-F1 %.4f"
                          % (short_label(key), seed,
                             result["full"]["f1_macro"]))
            rows.append({"model": display_label(key), "seed": int(seed),
                         "F1 (macro)": float(saved["full"]["f1_macro"]),
                         "G-mean": float(saved["full"].get("g_mean") or np.nan),
                         "binary FPR": float((saved.get("binary") or {})
                                             .get("false_positive_rate", np.nan)),
                         "source": str(metrics_path)})

    table = pd.DataFrame(rows)
    summary = pd.DataFrame()
    if len(table):
        grouped = table.groupby("model")[["F1 (macro)", "G-mean",
                                          "binary FPR"]]
        summary = pd.concat([grouped.mean().add_suffix(" mean"),
                             grouped.std(ddof=0).add_suffix(" std"),
                             grouped.size().rename("seeds")], axis=1
                            ).reset_index()
    return {"status": "written", "n_seeds": n_seeds, "seeds": list(seeds),
            "table": table, "summary": summary, "skipped": skipped}


def indistinguishable_pairs(summary: pd.DataFrame,
                            column: str = "F1 (macro)") -> pd.DataFrame:
    """Pairs whose means are closer together than one standard deviation.

    Not a significance test -- with three seeds there is no basis for one.  It
    is the weakest honest statement the numbers support: these two cannot be
    told apart by this evidence.
    """
    if summary is None or not len(summary):
        return pd.DataFrame()
    mean_column, std_column = "%s mean" % column, "%s std" % column
    rows = []
    models = list(summary["model"])
    for i, left in enumerate(models):
        for right in models[i + 1:]:
            a = summary[summary["model"] == left].iloc[0]
            b = summary[summary["model"] == right].iloc[0]
            gap = abs(float(a[mean_column]) - float(b[mean_column]))
            spread = max(float(a[std_column]), float(b[std_column]))
            rows.append({"pair": "%s vs %s" % (left.split(":")[0],
                                               right.split(":")[0]),
                         "%s gap" % column: round(gap, 6),
                         "largest std": round(spread, 6),
                         "verdict": ("not distinguishable" if gap < spread
                                     else "gap exceeds one std")})
    return pd.DataFrame(rows)


# ===========================================================================
# 12.11  DISCUSSION AND LIMITATIONS -- every sentence from a computed number
# ===========================================================================
def _ranking(table: pd.DataFrame, column: str, lower_is_better: bool = False
             ) -> list[tuple[str, float]]:
    values = pd.to_numeric(table[column], errors="coerce")
    pairs = [(str(name), float(value)) for name, value
             in zip(table["config"], values) if pd.notna(value)]
    return sorted(pairs, key=lambda pair: pair[1],
                  reverse=not lower_is_better)


def _place_of(ranking: Sequence[tuple[str, float]], prefix: str) -> int | None:
    for position, (name, _value) in enumerate(ranking, start=1):
        if str(name).startswith(prefix):
            return position
    return None


def build_discussion(step12: dict[str, Any], sanity: dict[str, Any] | None,
                     provenance: dict[str, Any] | None,
                     seeds: dict[str, Any] | None,
                     baseline6: dict[str, Any] | None) -> str:
    """12.11, generated from the tables above and from nothing else."""
    table = step12["comparison"]
    lines = ["## Step 12.11 -- discussion and limitations", ""]

    # --- the ranking ------------------------------------------------------
    lines.append("### Which configuration wins, and on what")
    lines.append("")
    for column, lower, name in (("F1 (macro)", False, "macro-F1"),
                                ("G-mean", False, "G-mean"),
                                ("binary FPR", True, "binary FPR"),
                                ("F1 macro (dup-free)", False,
                                 "macro-F1 on the duplicate-free subset"),
                                ("G-mean (dup-free)", False,
                                 "G-mean on the duplicate-free subset")):
        ranking = _ranking(table, column, lower)
        if not ranking:
            continue
        best, value = ranking[0]
        leaders = [n for n, v in ranking if abs(v - value) <= 1e-12]
        if len(leaders) == len(ranking):
            lines.append("- **%s** (%s is better): every configuration scores "
                         "%.4f, so this metric separates none of them here."
                         % (name, "lower" if lower else "higher", value))
            continue
        lines.append("- **%s** (%s is better): %s lead%s with %.4f. Order: %s."
                     % (name, "lower" if lower else "higher",
                        " and ".join(str(n).split(":")[0] for n in leaders),
                        "" if len(leaders) > 1 else "s", value,
                        ", then ".join("%s %.4f" % (str(n).split(":")[0], v)
                                       for n, v in ranking[:5])))
        for prefix in ("Approach 1", "Approach 2"):
            place = _place_of(ranking, prefix)
            if place is None:
                continue
            own = dict((str(n), v) for n, v in ranking)
            mine = [v for n, v in ranking if str(n).startswith(prefix)]
            if mine and abs(mine[0] - value) > 1e-12:
                lines.append("  - %s is **not** best here: %.4f against "
                             "%.4f, %d of %d."
                             % (prefix, mine[0], value, place, len(ranking)))
    lines.append("")

    # --- what each component contributes ---------------------------------
    head = step12.get("head_to_head")
    if head is not None and len(head):
        lines.append("### What each component contributes")
        lines.append("")
        for pair in head["pair"].unique():
            part = head[head["pair"] == pair]
            macro = part[part["metric"] == "F1 (macro)"]
            if not len(macro):
                continue
            row = macro.iloc[0]
            difference = float(row.get("difference (left - right)",
                                       float("nan")))
            lines.append("- **%s:** macro-F1 differs by %+.4f (%s)."
                         % (pair, difference, row.get("favours", "")))
        lines.append("")

    # --- accuracy, and why it says little --------------------------------
    if sanity is not None:
        lines.append("### Why accuracy is similar for every model")
        lines.append("")
        lines.append("A predictor that always answers with the most frequent "
                     "training class reaches **%.4f accuracy** while finding "
                     "no attack class at all (macro-F1 %.4f). Accuracy here "
                     "is a measure of the class imbalance, not of detection; "
                     "macro-F1 and G-mean are the comparison metrics."
                     % (sanity["majority_accuracy"],
                        float(sanity["table"].iloc[0]["F1 (macro)"])))
        lines.append("")

    # --- the checks --------------------------------------------------------
    lines.append("### What the checks found")
    lines.append("")
    if provenance is not None:
        failures = provenance.get("failures")
        if failures is None or not len(failures):
            lines.append("- **Provenance (12.0): every check passed.** Each "
                         "row of the comparison is traceable to the saved "
                         "file listed in `provenance.csv`, and the fused "
                         "files in Steps 13 and 13.8 were computed from the "
                         "predictions these models still hold.")
        else:
            lines.append("- **Provenance (12.0): %d check(s) FAILED.**"
                         % len(failures))
            for _i, row in failures.iterrows():
                lines.append("  - %s -- %s" % (row["check"], row["detail"]))
                if row.get("what to do"):
                    lines.append("    - Decision needed: %s"
                                 % row["what to do"])
    if sanity is not None:
        lines.append("- **Sanity (12.2):** the label-shuffle model scores "
                     "%.4f macro-F1 against a random floor of %.4f -- %s."
                     % (sanity["shuffle_macro_f1"], sanity["random_macro_f1"],
                        "**[FAIL] possible leakage**" if sanity["leak"]
                        else "no sign of leakage"))
    verified = step12.get("verified")
    if verified is not None and len(verified):
        failed = verified[verified["result"] == FAIL]
        lines.append("- **Table vs disk (12.7):** %d of %d rows match their "
                     "metrics.json to 1e-9%s."
                     % (int((verified["result"] == OK).sum()), len(verified),
                        "" if not len(failed) else
                        "; **%s do not**" % ", ".join(failed["config"])))
    lines.append("")

    # --- limitations -------------------------------------------------------
    lines.append("### Limitations")
    lines.append("")
    free = pd.to_numeric(table.get("F1 macro (dup-free)"), errors="coerce")
    full = pd.to_numeric(table.get("F1 (macro)"), errors="coerce")
    if free is not None and full is not None and free.notna().any():
        drop = float((full - free).max())
        lines.append("- **Duplicate rows.** About 54%% of the dataset's rows "
                     "are feature-identical to another row, so a test flow can "
                     "be an exact copy of a training flow. The duplicate-free "
                     "columns exclude those: the largest macro-F1 fall between "
                     "the two views is %.4f, and the duplicate-free number is "
                     "the conservative one to quote." % drop)
    lines.append("- **Random flow-level split.** The split is over flows, not "
                 "over captures or sessions, so flows from one attack session "
                 "can appear in both training and test. Every number here is "
                 "therefore optimistic relative to an unseen capture.")
    per_class = step12.get("per_class")
    if per_class is not None and len(per_class) and "test rows" in per_class:
        rare = per_class[pd.to_numeric(per_class["test rows"],
                                       errors="coerce") <= 10]
        if len(rare):
            lines.append("- **Rare classes.** %s have %s test rows "
                         "respectively, so their per-class F1 moves in large "
                         "steps and a difference there is not evidence."
                         % (", ".join(str(v) for v in rare["Label"]),
                            ", ".join(str(int(v)) for v in rare["test rows"])))
    if seeds is None or seeds.get("status") != "written":
        lines.append("- **One seed.** `N_SEEDS = %d`: every figure is a single "
                     "run. Differences of about 0.01 in macro-F1 may not be "
                     "meaningful, and none of the gaps above is a significance "
                     "claim." % int(getattr(config, "N_SEEDS", 1)))
    else:
        lines.append("- **Spread across %d seeds** is reported in 12.10; pairs "
                     "closer together than one standard deviation are marked "
                     "*not distinguishable*, which is the strongest statement "
                     "this evidence supports." % seeds.get("n_seeds"))
    if baseline6 is not None and baseline6.get("length_warning"):
        lines.append("- **Baseline 6's sequence length.** %s"
                     % baseline6["length_warning"])
    b4_metrics = cmp.load_result("baseline4_ae_pca_lstm") or {}
    b6_metrics = (baseline6 or {}).get("metrics") or {}
    b4_components = b4_metrics.get("pca_components")
    b6_components = b6_metrics.get("pca_components")
    if (b4_components is not None and b6_components is not None
            and int(b4_components) != int(b6_components)):
        lines.append(
            "- **B4 vs B6 differs in two things, not one.** Baseline 4's PCA "
            "keeps %d component(s) and Baseline 6's keeps %d, because the "
            "shared Conv1D kernel of %d cannot be built on a shorter "
            "sequence and the architecture was not changed to fit. So that "
            "pair mixes the classifier (LSTM vs CNN-LSTM) with the width of "
            "the PCA space, and the difference between them cannot be "
            "attributed to the classifier alone. **B1 vs B6 is the clean "
            "single-difference pair for Baseline 6**: the same AE latents and "
            "the same CNN-LSTM, with PCA(%d) inserted and nothing else "
            "changed."
            % (int(b4_components), int(b6_components), int(config.CNN_KERNEL),
               int(b6_components)))
    lines.append("- **Provenance is per machine.** The SHA-256 values in "
                 "`provenance.csv` identify the files on the machine that ran "
                 "this notebook; a rerun elsewhere produces different model "
                 "bytes and must be re-fingerprinted.")
    return "\n".join(lines)
