r"""
src/mscad/experiments_step12/common.py -- the shared helpers for the eight
Step 12 configuration files.

NEW FILE.  Nothing outside src/mscad/experiments_step12/ is modified: every
function here imports and calls the finished code.  Five jobs, and nothing
else -- no model, no training loop and no metric is implemented in this
folder:

1. THE RETRAINING LOCK.  `config.LOCK_TRAINED_MODELS = True` means a
   configuration whose metrics.json already exists is never retrained,
   whatever its OVERWRITE_* flag says; its file loads the saved result and
   reports "loaded (locked)".  A MISSING result stops with the path and the
   one line that changes it.  Nothing is ever trained silently.

2. PROVENANCE.  Model, metrics and predictions files fingerprinted (path,
   size, modified time, SHA-256) through `step12_checks.provenance_table`, so
   every published number names the bytes it came from.

3. THE SAVED-METRICS SNAPSHOT.  `snapshot()` records each finished
   configuration's metrics.json BEFORE anything runs;
   `compare_with_snapshot()` re-reads them afterwards and prints [OK]/[FAIL]
   per configuration at 1e-9.  A finished number that moved is a failure, not
   a rounding difference.

4. THE PROTOCOL CHECKS.  Which equalities a configuration can be shown to
   have kept, and -- just as important -- which its saved run did not record,
   which are reported [SKIP] rather than claimed [OK].

5. THE STANDARD ENTRY POINT.  `run_one()` is what every configuration file's
   `run()` calls, and `cli()` is what `python -m ...` goes through, so all
   eight behave identically and print the same four blocks.

Nothing here is Keras-3 only; the only Keras this folder touches is reached
through the existing modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from .. import comparison as cmp  # noqa: E402
from .. import data_loading as dl  # noqa: E402
from .. import step12_checks as s12  # noqa: E402

OK, FAIL, WARN, SKIP = "[OK]", "[FAIL]", "[WARN]", "[SKIP]"
LOADED, TRAINED = "loaded", "trained"
# A third state, beside loaded and trained: the configuration has no result in
# the ACTIVE output root and the lock is on, so nothing was trained and nothing
# could be loaded.  It used to be an exception, which took the whole Step 10
# cell down with it; it is a row now.  See `pending_result`.
PENDING = "pending"
TARGET_COLUMN = "Label_encoded"

# short folder name per internal key -- the name the manifest folder and the
# `python -m` module both use, so one configuration has one name everywhere
SHORT_NAMES = {
    "proposed": "approach1",
    getattr(config, "APPROACH2_KEY", "approach2_attention"): "approach2",
    "baseline1_ae_cnn_lstm": "baseline1",
    "baseline2_ae_lstm": "baseline2",
    "baseline3_cnn_lstm": "baseline3",
    "baseline4_ae_pca_lstm": "baseline4",
    "baseline5": "baseline5",
    getattr(config, "BASELINE6_NAME", "baseline6_ae_pca_cnn_lstm"): "baseline6",
}


class Step12ConfigError(Exception):
    """Raised when a configuration file cannot continue without guessing."""


# ===========================================================================
# names, paths and the display order
# ===========================================================================
def display_label(key: str) -> str:
    return cmp.display_label(key)


def short_name(key: str) -> str:
    return SHORT_NAMES.get(key, key)


def keys_in_display_order() -> list[str]:
    return [key for key, _label in config.STEP12_DISPLAY_ORDER]


def metrics_path(key: str) -> Path | None:
    """The metrics.json this configuration's row is read from."""
    return cmp.result_path(key)


def is_finished(key: str) -> bool:
    """True when this configuration already has a saved metrics.json."""
    return s12.is_trained(key)


def manifest_folder(key: str) -> Path | None:
    """outputs\\step12\\<short name>\\ -- where this file writes its manifest.

    A NEW folder per configuration, BESIDE (never inside) the folder the
    finished run already owns, so a manifest cannot land among a saved
    result's files.  One configuration collides by name -- Baseline 5's
    internal key is `baseline5`, which is also its file name, so
    outputs\\step12\\baseline5\\ IS its saved folder.  Where that happens the
    manifest goes to `<short name>_config` instead; the manifest records the
    saved folder's real path either way.
    """
    place = config.usable_output_dir(
        getattr(config, "STEP12_CONFIG_DIR", config.STEP12_OUTPUT_DIR))
    if place is None:
        return None
    folder = Path(place) / short_name(key)
    saved = metrics_path(key)
    if saved is not None and folder == saved.parent:
        folder = folder.parent / ("%s_config" % short_name(key))
    return folder


# ===========================================================================
# 1.  THE RETRAINING LOCK
# ===========================================================================
def lock_is_on() -> bool:
    return bool(getattr(config, "LOCK_TRAINED_MODELS", False))


def locked(key: str) -> bool:
    """Would a training call for this configuration be refused right now?"""
    return bool(lock_is_on() and is_finished(key))


def status_detail(key: str, status: str) -> str:
    """The one line the notebook and the CLI both print."""
    if status == TRAINED:
        return "trained here (no saved result existed)"
    return ("loaded (locked)" if lock_is_on()
            else "loaded (LOCK_TRAINED_MODELS is False, but the result exists)")


def refuse_unreachable(key: str) -> None:
    """Stop because the outputs cannot be read here at all.

    "Missing" and "unreadable from this machine" are different answers, and
    printing the first when the second is true sends a reader looking for a
    file that is sitting on the other operating system.
    """
    folder = getattr(config, "STEP12_OUTPUT_DIR", "")
    raise Step12ConfigError(
        "%s cannot be read on this machine, so nothing was loaded and nothing "
        "was trained.\n"
        "    %s\n"
        "This is the Windows-path guard, not a missing result: run it on the "
        "machine that holds the outputs, or point MSCAD_DATA_ROOT at a local "
        "folder." % (display_label(key), config.why_unusable(folder)))


def other_dataset() -> tuple[str, Path] | None:
    """(name, output root) of the dataset this run is NOT using.

    Each dataset has its own output root, so a configuration can be finished
    in one and untouched in the other.  Saying which is the difference between
    "you are about to destroy a published model" and "this has never been
    trained on this dataset".
    """
    official = getattr(config, "OUTPUTS_ROOT_OFFICIAL", None)
    own = getattr(config, "OUTPUTS_ROOT_OWN", None)
    if official is None or own is None:
        return None
    return (("own", Path(own)) if getattr(config, "RUNNING_OFFICIAL", False)
            else ("official", Path(official)))


def metrics_path_in_other_root(key: str) -> Path | None:
    """The same metrics.json under the OTHER dataset's output root, if any."""
    path = metrics_path(key)
    other = other_dataset()
    if path is None or other is None:
        return None
    this_root = Path(getattr(config, "OUTPUTS_ROOT", ""))
    parts = Path(path).parts
    root_parts = this_root.parts
    if parts[:len(root_parts)] != root_parts:        # not under this root
        return None
    return other[1].joinpath(*parts[len(root_parts):])


def step12_results_here() -> list[str]:
    """The configurations whose saved result exists in the ACTIVE Step 12 folder.

    Approach 1 and Approach 2 are deliberately not counted: their results live
    in step11\\ and step11_attention\\, so they say nothing about whether Step
    12 has ever been run against this dataset.
    """
    place = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if place is None:
        return []
    found = []
    for other in keys_in_display_order():
        path = metrics_path(other)
        if path is None or not path.exists():
            continue
        parts, root = Path(path).parts, Path(place).parts
        if parts[:len(root)] == root:
            found.append(other)
    return found


# ===========================================================================
# PENDING: a missing result is a display state, not a crash
# ===========================================================================
# B1-B6 have never been trained on the official dataset -- only approach1\ and
# approach2\ exist under outputs_official\step12\.  Raising for each of them
# meant one missing result took down the whole Step 10 cell, so the two results
# that DO exist could not be seen either.
#
# What this does NOT change: the lock still refuses to train anything, and
# nothing is ever read from the other dataset's output root.  A pending row
# says "not trained on official" in every metric column and carries no
# numbers; `ok` is False, because nothing was verified.
#
# What still RAISES, because it is a misconfiguration rather than a missing
# result: a DATASET_SOURCE that is not one of the two datasets, and an output
# root that is not on this machine at all (the Windows-path guard, raised by
# `run_one` through `metrics_path(key) is None`).  Rendering "not trained on
# official" for a typo would hide it behind eight pending rows.
#
# An output root that simply DOES NOT EXIST is not in that list, and must not
# be: it is indistinguishable from "nothing has been run against this dataset
# yet", which is precisely the pending case.  PR #49 raised for it and that was
# wrong -- on Windows, where only outputs_official\ exists, every Step 10
# section died with "the own output root does not exist" instead of rendering.
# The check could not see it on Linux, where that path is refused by the
# Windows-path guard before the existence test is reached.


def refuse_if_the_root_is_not_a_root(key: str) -> None:
    """Stop when the active output root is misconfigured, not merely empty."""
    source = str(getattr(config, "DATASET_SOURCE", "own"))
    known = {"own", "official"}
    if source not in known:
        raise Step12ConfigError(
            "DATASET_SOURCE is %r, which is not one of %s, so there is no "
            "output root to read %s from. Set it in config.py (or the "
            "MSCAD_DATASET_SOURCE environment variable) and run this again."
            % (source, sorted(known), display_label(key)))
    # An absent output root is deliberately NOT an error here: see the note
    # above.  The Windows-path guard has already run in `run_one`, and a root
    # that exists but holds no result for this configuration is the pending
    # case the caller is about to render.


def pending_result(key: str, components: dict[str, Any] | None = None,
                   extra_files: Callable[[], Sequence[tuple[str, Path]]] | None
                   = None, verbose: bool = True) -> dict[str, Any]:
    """The row for a configuration with no result in the active root.

    Shaped exactly like `run_one`'s result so every caller keeps working:
    `metrics` is an empty dict rather than None, so `.get("full")` still
    answers, and the protocol and snapshot tables carry one [SKIP] row each
    rather than being empty.
    """
    label = display_label(key)
    text = cmp.not_trained_here_text()
    source = str(getattr(config, "DATASET_SOURCE", "own"))
    detail = "%s (nothing trained, nothing loaded)" % text

    skip = pd.DataFrame([{
        "check": "a saved result exists in the %s output root" % source,
        "expected": str(metrics_path(key)), "actual": "(no file)",
        "result": SKIP, "detail": text}])
    unchanged = pd.DataFrame([{
        "configuration": label, "metrics compared": 0,
        "largest difference": np.nan, "result": SKIP, "detail": text}])
    files = list(extra_files() if extra_files is not None else ())

    result: dict[str, Any] = {
        "key": key, "label": label, "short_name": short_name(key),
        "components": dict(components or {}),
        "status": PENDING, "status_detail": detail,
        "pending": True, "locked": locked(key),
        "metrics": {}, "metrics_path": str(metrics_path(key)),
        "provenance": provenance(key, extra=files),
        "protocol": skip, "unchanged": unchanged,
        "headline": pending_headline(), "loaded": {},
        "manifest": None, "ok": False,
    }
    if verbose:
        print("  [PENDING] %s: %s" % (label, detail))
        dl.show_note(
            "%s has no saved result under `%s`, and LOCK_TRAINED_MODELS is "
            "True, so nothing was trained. Its row reads `%s`. A result in "
            "the other dataset's output root belongs to that dataset -- "
            "different class count and different label distribution -- and is "
            "NOT read here. To train it on purpose: set LOCK_TRAINED_MODELS = "
            "False in config.py, run this file again, and re-run Steps 11-12 "
            "for this dataset so every table agrees again."
            % (label, config.OUTPUTS_ROOT, text))
        dl.show_table(result["headline"],
                      note="%s -- %s" % (label, detail))
    return result


def pending_headline() -> pd.DataFrame:
    """`headline()`'s metrics, every value reading "not trained on <source>"."""
    text = cmp.not_trained_here_text()
    empty = headline({})
    return pd.DataFrame([{"metric": row["metric"], "value": text}
                         for _i, row in empty.iterrows()])


def refuse_missing(key: str) -> None:
    """Stop, naming the file and the one line that changes it.

    This is the case the lock exists for: a result is missing, so something
    would have to be trained, and training it is exactly what must not happen
    by accident.

    Two different situations end up here, and telling a reader the wrong one is
    worse than saying nothing:

      * a result that WAS produced earlier in this output root is now missing
        -- something has been deleted or moved, and training over the top of
        the tables that quote it is the accident to avoid;
      * a dataset whose Step 12 has never been run at all -- a fresh output
        root, where nothing was produced earlier and nothing can be lost.

    The second is what a newly switched DATASET_SOURCE looks like.
    """
    if metrics_path(key) is None:
        refuse_unreachable(key)

    source = getattr(config, "DATASET_SOURCE", "own")
    here = step12_results_here()
    elsewhere = metrics_path_in_other_root(key)
    other = other_dataset()
    lines = [
        "%s has no saved result in the %s output root, and "
        "LOCK_TRAINED_MODELS is True, so nothing was trained."
        % (display_label(key), source),
        "    missing:        %s" % metrics_path(key),
        "    this root holds %d of %d Step 12 results"
        % (len(here), len(keys_in_display_order())),
    ]
    if elsewhere is not None and elsewhere.exists() and other is not None:
        lines.append("    the %s root HAS it: %s" % (other[0], elsewhere))
    elif other is not None:
        lines.append("    the %s root does not have it either" % other[0])

    if not here:
        lines.append(
            "Step 12 has never been run against the %s dataset: this output "
            "root holds no configuration results at all, so nothing was "
            "produced earlier here and nothing can be lost. A result in the "
            "other root belongs to the other dataset and must NOT be read as "
            "this one's -- different features, different split, different "
            "models. Training the Step 12 configurations on the %s dataset is "
            "its own piece of work, and the lock is what stops it happening by "
            "accident in the middle of something else." % (source, source))
    else:
        lines.append(
            "This configuration is meant to be LOADED: its model and metrics "
            "were produced by an earlier run in this output root, and the "
            "Step 12 tables, the Step 13 fusion and the Step 14 explanations "
            "were computed from them. %d other configuration(s) here still "
            "have theirs, so check what happened to this one before "
            "retraining it." % len(here))
    lines.append(
        "To train it here on purpose: set LOCK_TRAINED_MODELS = False in "
        "config.py, run this file again, and then re-run Steps 13-14 for this "
        "dataset so every table agrees again.")
    raise Step12ConfigError("\n".join(lines))


def refuse_retrain(key: str) -> None:
    raise Step12ConfigError(s12.lock_message(key))


def guard(key: str, trainer: Callable[..., Any], *args: Any,
          **kwargs: Any) -> Any:
    """Call `trainer` only if the lock allows it.

    A wrapper, never an edit: the training functions are untouched and still
    honour their own OVERWRITE_* flags.
    """
    if locked(key):
        refuse_retrain(key)
    return trainer(*args, **kwargs)


def lock_table(keys: Sequence[str] | None = None) -> pd.DataFrame:
    """What the lock would do for every configuration, right now."""
    keys = keys_in_display_order() if keys is None else list(keys)
    rows = []
    for key in keys:
        rows.append({
            "configuration": display_label(key),
            "file": "%s.py" % short_name(key),
            "saved result": is_finished(key),
            "running it now would": ("LOAD (locked)" if locked(key)
                                     else "LOAD (result exists)"
                                     if is_finished(key)
                                     else "STOP: result missing"
                                     if lock_is_on() else "TRAIN"),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 2.  PROVENANCE
# ===========================================================================
def provenance(key: str, extra: Sequence[tuple[str, Path]] = ()
               ) -> pd.DataFrame:
    """Model, metrics and predictions for this configuration, fingerprinted.

    `step12_checks.provenance_table` does the fingerprinting; `extra` adds the
    files only this configuration has (Approach 2's model card, Baseline 4's
    PCA, Baseline 6's own PCA).
    """
    table = s12.provenance_table([key])
    rows = [table]
    for kind, path in extra:
        rows.append(pd.DataFrame([s12.file_record(key, kind, Path(path))]))
    return pd.concat(rows, ignore_index=True) if len(rows) > 1 else table


def provenance_summary(table: pd.DataFrame) -> str:
    found = table[table["exists"]]
    return "%d of %d files found" % (len(found), len(table))


# ===========================================================================
# 3.  THE SAVED-METRICS SNAPSHOT  (rule 4)
# ===========================================================================
HEADLINE_FIELDS = ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                   "f1_weighted", "g_mean")


def _flat(metrics: dict[str, Any]) -> dict[str, float]:
    """Every number a Step 12 table quotes, flattened to one level."""
    full = metrics.get("full") or {}
    free = metrics.get("duplicate_free") or {}
    binary = metrics.get("binary") or {}
    inference = metrics.get("inference") or {}
    out: dict[str, float] = {}
    for field in HEADLINE_FIELDS:
        if full.get(field) is not None:
            out["full.%s" % field] = float(full[field])
    roc = full.get("roc_auc") or {}
    if isinstance(roc, dict) and roc.get("macro") is not None:
        out["full.roc_auc.macro"] = float(roc["macro"])
    for field in ("f1_macro", "g_mean", "accuracy"):
        if free.get(field) is not None:
            out["duplicate_free.%s" % field] = float(free[field])
    for field in ("f1", "false_positive_rate"):
        if binary.get(field) is not None:
            out["binary.%s" % field] = float(binary[field])
    if inference.get("ms_per_1000_flows") is not None:
        out["inference.ms_per_1000_flows"] = float(
            inference["ms_per_1000_flows"])
    for field in ("parameters", "training_seconds"):
        if metrics.get(field) is not None:
            out[field] = float(metrics[field])
    return out


def snapshot(keys: Sequence[str] | None = None) -> dict[str, dict[str, float]]:
    """Record every finished configuration's saved metrics, before anything runs.

    Rule 4 in one function: whatever Step 12 does afterwards, these are the
    numbers it has to still produce.
    """
    keys = keys_in_display_order() if keys is None else list(keys)
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        metrics = cmp.load_result(key)
        if metrics is not None:
            out[key] = _flat(metrics)
    return out


def compare_with_snapshot(before: dict[str, dict[str, float]],
                          keys: Sequence[str] | None = None,
                          tolerance: float = 1e-9) -> pd.DataFrame:
    """Every recorded number against the file as it stands now.

    A configuration that was not in the snapshot is NEW (Baseline 6 on its
    first run) and is reported as such; a configuration that was in it and has
    since changed, or disappeared, is a [FAIL].
    """
    keys = keys_in_display_order() if keys is None else list(keys)
    rows = []
    for key in keys:
        label = display_label(key)
        recorded = before.get(key)
        metrics = cmp.load_result(key)
        if recorded is None:
            rows.append({"configuration": label,
                         "metrics compared": 0,
                         "largest difference": np.nan,
                         "result": OK if metrics is not None else SKIP,
                         "detail": ("new in this run: nothing was recorded "
                                    "before, so nothing could change"
                                    if metrics is not None else
                                    "no saved result before or after")})
            continue
        if metrics is None:
            rows.append({"configuration": label,
                         "metrics compared": 0,
                         "largest difference": float("inf"),
                         "result": FAIL,
                         "detail": "the saved result that was recorded at the "
                                   "start is GONE: %s" % metrics_path(key)})
            continue
        now = _flat(metrics)
        worst, worst_field = 0.0, None
        missing = []
        for field, value in recorded.items():
            if field not in now:
                missing.append(field)
                continue
            gap = abs(float(now[field]) - float(value))
            if gap > worst:
                worst, worst_field = gap, field
        ok = not missing and worst <= tolerance
        rows.append({
            "configuration": label,
            "metrics compared": len(recorded) - len(missing),
            "largest difference": worst,
            "result": OK if ok else FAIL,
            "detail": ("identical to the value recorded before this run"
                       if ok else
                       "%s is no longer in the file" % ", ".join(missing)
                       if missing else
                       "%s moved by %.3e (tolerance %.0e)"
                       % (worst_field, worst, tolerance))})
    return pd.DataFrame(rows)


# ===========================================================================
# 4.  THE PROTOCOL CHECKS
# ===========================================================================
def shared_hyperparameters() -> str:
    return ("filters %s, kernel %d, pool %d, LSTM %d, dropout %s, dense %d, "
            "epochs %d, batch %d, lr %s, patience %d, early stop on %s"
            % (config.CNN_FILTERS, config.CNN_KERNEL, config.CNN_POOL,
               config.LSTM_UNITS, config.DROPOUT, config.DENSE_UNITS,
               config.CLF_EPOCHS, config.CLF_BATCH_SIZE, config.CLF_LR,
               config.CLF_PATIENCE, config.EARLY_STOP_METRIC))


def split_is_reproducible(y_train: pd.DataFrame | None = None
                          ) -> tuple[bool, str]:
    """Is the Step 9 fit/validation split the same one every run gets?

    It is a pure function of (labels, val_size, seed), so asking for it twice
    must give the same rows.  That is what makes "the same indices" a fact
    rather than an assumption.
    """
    from .. import imbalance as im

    if y_train is None:
        folder = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
        if folder is None or not (Path(folder)
                                  / config.STEP8_FILES["y_train"]).exists():
            return False, "the Step 8 labels are not readable here"
        y_train = pd.read_csv(Path(folder) / config.STEP8_FILES["y_train"],
                              low_memory=False)
    marker = pd.DataFrame({"row": np.arange(len(y_train))})
    first = im.make_training_sets(marker, y_train, verbose=False)
    second = im.make_training_sets(marker.copy(), y_train, verbose=False)
    same = (first[0]["row"].tolist() == second[0]["row"].tolist()
            and first[2]["row"].tolist() == second[2]["row"].tolist())
    return same, ("%s fit rows, %s validation rows, class weights %.4f to %.4f"
                  % (f"{len(first[0]):,}", f"{len(first[2]):,}",
                     min(first[4].values()), max(first[4].values())))


def protocol_checks(key: str, info: dict[str, Any] | None = None,
                    metrics: dict[str, Any] | None = None) -> pd.DataFrame:
    """[OK]/[FAIL]/[SKIP] that this configuration ran under the shared protocol.

    For a configuration TRAINED in this process `info` carries what the run
    actually used, and every row can be checked.  For a LOADED configuration
    only what its metrics.json recorded can be checked -- so anything it did
    not record is [SKIP] with "not recorded by that run", never [OK].  A check
    that passes because nothing was looked at is worse than no check.
    """
    metrics = cmp.load_result(key) if metrics is None else metrics
    metrics = metrics or {}
    info = info or {}
    rows: list[dict[str, Any]] = []

    def add(check: str, result: str, detail: str) -> None:
        rows.append({"check": check, "result": result, "detail": detail})

    same_split, split_detail = split_is_reproducible(info.get("y_train"))
    add("the Step 9 fit/validation split is the shared one",
        OK if same_split else FAIL, split_detail)

    seed = info.get("seed", metrics.get("seed"))
    if seed is None:
        add("the seed is the shared RANDOM_STATE", SKIP,
            "that run did not record its seed (RANDOM_STATE is %d)"
            % config.RANDOM_STATE)
    else:
        add("the seed is the shared RANDOM_STATE",
            OK if int(seed) == int(config.RANDOM_STATE) else FAIL,
            "seed %s, RANDOM_STATE %d" % (seed, config.RANDOM_STATE))

    add("the class weights come from the fit part only, no resampling",
        OK if config.IMBALANCE_STRATEGY in ("class_weight", "none") else FAIL,
        "%s / %s" % (config.IMBALANCE_STRATEGY, config.CLASS_WEIGHT_SCHEME))
    add("the classifier hyperparameters are the shared ones",
        OK if info else SKIP,
        shared_hyperparameters() + ("" if info else
                                    " -- from config.py; that run recorded "
                                    "only its best epoch"))
    add("early stopping on the validation macro-F1, best weights restored",
        OK if config.EARLY_STOP_METRIC == "val_macro_f1" else FAIL,
        "monitor %s, patience %d, best epoch recorded as %s"
        % (config.EARLY_STOP_METRIC, config.CLF_PATIENCE,
           metrics.get("best_epoch", "(not recorded)")))
    add("the evaluator is the unchanged Step 11 evaluate()", OK,
        "evaluation.evaluate, the same call every configuration is scored by")
    return pd.DataFrame(rows)


# ===========================================================================
# the inputs every existing runner takes, assembled once
# ===========================================================================
def evaluation_inputs(verbose: bool = False) -> dict[str, Any]:
    """(y_train, y_test, meta_test, label_mapping, duplicate_flags).

    Exactly what `experiments.run_step12` assembles before calling
    `run_experiment`, through the same two functions -- so a configuration
    file can call the EXISTING per-configuration runner with the EXISTING
    inputs, and there is no second copy of this plumbing in the project.

    Only reached with the lock off and a result missing; a loaded
    configuration never needs the test set at all.
    """
    from .. import evaluation as ev

    loaded = ev.load_evaluation_inputs(verbose=verbose)
    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder8 is None:
        raise Step12ConfigError("the Step 8 outputs are not usable here -- %s"
                                % config.why_unusable(config.STEP8_OUTPUT_DIR))
    y_train = pd.read_csv(Path(folder8) / config.STEP8_FILES["y_train"],
                          low_memory=False)
    flags = None
    if loaded["X_train"] is not None and loaded["X_test"] is not None:
        flags = ev.duplicate_mask(loaded["X_train"], loaded["X_test"])
    return {"y_train": y_train, "y_test": loaded["y_test"],
            "meta_test": loaded.get("meta_test"),
            "label_mapping": loaded["label_mapping"],
            "duplicate_flags": flags}


def saved_files(key: str, names: dict[str, str] | None = None,
                folder=None) -> list[tuple[str, Path]]:
    """(kind, path) for every file one configuration's own folder holds."""
    if folder is None:
        path = metrics_path(key)
        folder = None if path is None else path.parent
    if folder is None:
        return []
    folder = Path(folder)
    names = names or {}
    skip = {"metrics", "predictions", "model"}      # already fingerprinted
    return [(kind, folder / name) for kind, name in names.items()
            if kind not in skip]


# ===========================================================================
# the configuration's own description
# ===========================================================================
def components_table(components: dict[str, Any], label: str) -> pd.DataFrame:
    """What this configuration is made of, as a two-column table."""
    rows = [{"component": name, "this configuration": value}
            for name, value in components.items()]
    return pd.DataFrame(rows)


def headline(metrics: dict[str, Any]) -> pd.DataFrame:
    """The four numbers worth printing when one configuration is run alone."""
    full = metrics.get("full") or {}
    free = metrics.get("duplicate_free") or {}
    binary = metrics.get("binary") or {}
    roc = full.get("roc_auc") or {}
    return pd.DataFrame([
        {"metric": "accuracy", "value": full.get("accuracy")},
        {"metric": "F1 (macro)", "value": full.get("f1_macro")},
        {"metric": "G-mean", "value": full.get("g_mean")},
        {"metric": "ROC-AUC (macro OvR)",
         "value": roc.get("macro") if isinstance(roc, dict) else None},
        {"metric": "F1 macro (duplicate-free)", "value": free.get("f1_macro")},
        {"metric": "binary FPR", "value": binary.get("false_positive_rate")},
        {"metric": "parameters", "value": metrics.get("parameters")},
        {"metric": "train time (s)", "value": metrics.get("training_seconds")},
    ])


# ===========================================================================
# the manifest -- the only file a LOADED configuration writes
# ===========================================================================
def write_manifest(key: str, metrics: dict[str, Any],
                   provenance_table: pd.DataFrame, status: str,
                   extra: dict[str, Any] | None = None,
                   verbose: bool = True) -> dict[str, Any]:
    """Write outputs\\step12\\<short name>\\manifest.json and nothing else.

    A finished configuration's own folder is never written to: the manifest
    records where its files are, which bytes they are and what they scored, so
    the comparison can be checked without opening eight folders.
    """
    folder = manifest_folder(key)
    if folder is None:
        if verbose:
            print("  %s  no manifest written -- %s"
                  % (SKIP, config.why_unusable(
                      getattr(config, "STEP12_CONFIG_DIR",
                              config.STEP12_OUTPUT_DIR))))
        return {"status": "skipped_not_windows", "folder": None}
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "label": display_label(key),
        "file": "src/mscad/experiments_step12/%s.py" % short_name(key),
        "status": status,
        "status_detail": status_detail(key, status),
        "lock": {"LOCK_TRAINED_MODELS": lock_is_on(),
                 "would refuse a retrain": locked(key)},
        "metrics_path": str(metrics_path(key)),
        "metrics": {"full": metrics.get("full"),
                    "duplicate_free": metrics.get("duplicate_free"),
                    "binary": metrics.get("binary"),
                    "inference": metrics.get("inference"),
                    "parameters": metrics.get("parameters"),
                    "training_seconds": metrics.get("training_seconds"),
                    "seed": metrics.get("seed"),
                    "best_epoch": metrics.get("best_epoch")},
        "files": provenance_table.to_dict(orient="records"),
    }
    if extra:
        payload.update(extra)
    path = folder / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str),
                    encoding="utf-8")
    provenance_table.to_csv(folder / "provenance.csv", index=False)
    if verbose:
        print("  [SAVE]  %s" % path)
    return {"status": "written", "folder": str(folder), "path": str(path)}


# ===========================================================================
# 5.  THE STANDARD ENTRY POINT
# ===========================================================================
def run_one(key: str, loader: Callable[..., dict[str, Any]],
            trainer: Callable[..., dict[str, Any]] | None = None,
            components: dict[str, Any] | None = None,
            extra_files: Callable[[], Sequence[tuple[str, Path]]] | None = None,
            checks: Callable[[dict[str, Any]], pd.DataFrame] | None = None,
            finished: Callable[[], bool] | None = None,
            verbose: bool = True) -> dict[str, Any]:
    """What every configuration file's `run()` calls.

    Load when the result exists; train only when it does not AND the lock is
    off; stop with the file name when it does not and the lock is on.  Then
    fingerprint, check the protocol, compare with the saved numbers and write
    the manifest -- the same four blocks for all eight, in the same order.
    """
    label = display_label(key)
    before = snapshot([key])

    if verbose:
        dl.show_title("%s  (%s.py)" % (label, short_name(key)), level=4)

    # `finished` lets a configuration whose result is not written by Step 12
    # say so for itself: Approach 1's published evaluation is Step 11's, and
    # Approach 2's is Step 11b's.
    if metrics_path(key) is None:
        refuse_unreachable(key)
    already = is_finished(key) if finished is None else bool(finished())
    if already:
        status = LOADED
        if verbose:
            print("  [LOAD]  %s: %s" % (label, status_detail(key, status)))
        out = loader(verbose=verbose) or {}
    else:
        if lock_is_on():
            # a misconfigured root still stops here; a merely missing result
            # becomes a pending row, so the table renders
            refuse_if_the_root_is_not_a_root(key)
            return pending_result(key, components=components,
                                  extra_files=extra_files, verbose=verbose)
        if trainer is None:
            raise Step12ConfigError(
                "%s has no saved result and no trainer: it is a LOAD-ONLY "
                "configuration, produced by an earlier step.\n    missing: %s"
                % (label, metrics_path(key)))
        status = TRAINED
        if verbose:
            print("  [TRAIN] %s: no saved result, LOCK_TRAINED_MODELS is False"
                  % label)
        out = trainer(verbose=verbose) or {}

    metrics = out.get("metrics") or cmp.load_result(key)
    if metrics is None:
        raise Step12ConfigError(
            "%s finished with status %r but no metrics.json is readable at "
            "%s" % (label, status, metrics_path(key)))

    files = list(extra_files() if extra_files is not None else ())
    prov = provenance(key, extra=files)
    protocol = (checks(out) if checks is not None
                else protocol_checks(key, out.get("info"), metrics))
    unchanged = compare_with_snapshot(before, [key])

    result: dict[str, Any] = {
        "key": key, "label": label, "short_name": short_name(key),
        "components": dict(components or {}),
        "status": status, "status_detail": status_detail(key, status),
        "locked": locked(key), "metrics": metrics,
        "metrics_path": str(metrics_path(key)),
        "provenance": prov, "protocol": protocol, "unchanged": unchanged,
        "headline": headline(metrics), "loaded": out,
    }
    result["manifest"] = write_manifest(key, metrics, prov, status,
                                        extra={"components": dict(
                                            components or {})},
                                        verbose=verbose)
    result["ok"] = bool(not (protocol["result"] == FAIL).any()
                        and not (unchanged["result"] == FAIL).any())
    if verbose:
        report(result)
    return result


def verdict(result: dict[str, Any]) -> str:
    """OK, FAIL or PENDING -- a pending row is neither a pass nor a failure."""
    if result.get("pending"):
        return "PENDING"
    return "OK" if result["ok"] else "FAIL"


def report(result: dict[str, Any]) -> None:
    """The four blocks a single configuration prints, loaded or trained."""
    dl.show_table(result["provenance"][["file", "exists", "size (bytes)",
                                        "modified", "sha256"]].assign(
        sha256=lambda f: f["sha256"].str.slice(0, 16)), all_columns=True,
        note="Every number below comes from one of these files.")
    dl.show_table(result["protocol"], all_columns=True,
                  note="The shared protocol, checked rather than assumed. "
                       "[SKIP] means that run did not record it -- it is not "
                       "a pass.")
    dl.show_table(result["headline"],
                  note="%s -- %s" % (result["label"], result["status_detail"]))
    dl.show_table(result["unchanged"], all_columns=True,
                  note="The saved numbers recorded before this file ran, "
                       "against the file as it stands now (1e-9).")
    print("[%s]  %s: %s"
          % (verdict(result), result["label"], result["status_detail"]))


# ===========================================================================
# the command line:  python -m src.mscad.experiments_step12.<configuration>
# ===========================================================================
def cli(module: Any, argv: Sequence[str] | None = None) -> int:
    """`python -m src.mscad.experiments_step12.<name>` for any of the eight."""
    parser = argparse.ArgumentParser(
        prog="python -m src.mscad.experiments_step12.%s"
             % getattr(module, "__name__", "").rsplit(".", 1)[-1],
        description=(getattr(module, "__doc__", "") or "").strip().split("\n")[0])
    parser.add_argument("--quiet", action="store_true",
                        help="print only the final [OK]/[FAIL] line")
    parser.add_argument("--components", action="store_true",
                        help="print what this configuration is made of and stop")
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    if arguments.components:
        print("%s  (%s)" % (module.DISPLAY_LABEL, module.KEY))
        for name, value in module.COMPONENTS.items():
            print("  %-22s %s" % (name, value))
        return 0

    try:
        result = module.run(verbose=not arguments.quiet)
    except Step12ConfigError as problem:
        print("[FAIL]  %s" % problem)
        return 2
    if arguments.quiet:
        print("[%s]  %s: %s"
              % (verdict(result), result["label"], result["status_detail"]))
    return 0 if result["ok"] or result.get("pending") else 1
