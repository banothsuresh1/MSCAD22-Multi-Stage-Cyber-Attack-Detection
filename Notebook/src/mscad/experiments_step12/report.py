r"""
src/mscad/experiments_step12/report.py -- the Step 12 sections that are about
all eight configurations at once.

NEW FILE, in the new folder, because rule 1 forbids touching comparison.py --
and nothing here needs to: every table, figure and export already exists as a
function in `comparison.py` or `step12_checks.py`, and this module only
assembles them and adds what the eight separate files make possible, which is
a cross-check that each file's own COMPONENTS agrees with the row the
comparison table prints for it.

    12.0   provenance_section()    -- SHA-256, dates, the lock, the three
                                      consistency questions
    12.9   configuration_tables()  -- the configuration and training-setup
                                      tables, plus the files-vs-table check
    12.10  the notebook calls comparison.run_step12_complete() itself
    12.11  per_class_sections()
    12.12  figure_table()
    12.13  head_to_head_highlights()
    12.14  discussion() and export_tables()

Nothing here trains anything, and nothing here writes into a folder a
finished run owns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from .. import comparison as cmp  # noqa: E402
from .. import data_loading as dl  # noqa: E402
from .. import step12_checks as s12  # noqa: E402
from . import common  # noqa: E402

OK, FAIL, SKIP = common.OK, common.FAIL, common.SKIP


# ===========================================================================
# 12.0  PROVENANCE AND LOCK STATUS
# ===========================================================================
def provenance_section(verbose: bool = True) -> dict[str, Any]:
    """Which saved file is each number from, and what would the lock do now?

    `step12_checks.run_provenance` is unchanged and answers all three of the
    brief's questions: it fingerprints every file, recomputes each model's
    attack risk from its CURRENT predictions.csv and compares it with what
    Step 13 and 13.8 fused, compares Approach 1's Step 11 metrics with its
    Step 12 row and any backup, and flags a model file written after the
    metrics that describe it.
    """
    out = s12.run_provenance(verbose=verbose)
    out["lock"] = common.lock_table()
    out["snapshot"] = common.snapshot()
    if verbose:
        dl.show_table(out["lock"], all_columns=True,
                      note="LOCK_TRAINED_MODELS = %s. A configuration with a "
                           "saved result is LOADED, never retrained; a missing "
                           "one STOPS with its file name."
                           % config.LOCK_TRAINED_MODELS)
        print("  recorded %d saved results before anything ran (rule 4)"
              % len(out["snapshot"]))
    return out


# ===========================================================================
# running the eight files
# ===========================================================================
def run_all(verbose: bool = True, names: Sequence[str] | None = None
            ) -> dict[str, dict[str, Any]]:
    """Every configuration file's run(), in display order.

    The notebook runs them one section at a time; this is the same thing in
    one call, for a test or a command line.
    """
    from . import CONFIG_MODULES, module_for

    names = CONFIG_MODULES if names is None else list(names)
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        module = module_for(name)
        out[name] = module.run(verbose=verbose)
    return out


def status_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per configuration: loaded or trained, and whether it checks out."""
    rows = []
    for name, result in results.items():
        rows.append({
            "configuration": result["label"],
            "file": "%s.py" % name,
            "status": result["status"],
            "detail": result["status_detail"],
            "macro-F1": (result["metrics"].get("full") or {}).get("f1_macro"),
            "protocol": ("%d checks, %d failed"
                         % (len(result["protocol"]),
                            int((result["protocol"]["result"] == FAIL).sum()))),
            "numbers unchanged": result["unchanged"].iloc[0]["result"],
            "result": OK if result["ok"] else FAIL,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 12.9  THE CONFIGURATION AND TRAINING-SETUP TABLES
# ===========================================================================
def components_table() -> pd.DataFrame:
    """What each of the eight FILES says it is made of.

    Read from the files themselves rather than from config, so the table a
    reader sees is the one the code that ran actually declares.
    """
    from . import CONFIG_MODULES, module_for

    rows = []
    for name in CONFIG_MODULES:
        module = module_for(name)
        row = {"file": "%s.py" % name, "config": module.DISPLAY_LABEL}
        row.update({key: value for key, value in module.COMPONENTS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def files_agree_with_table(table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Does each file's COMPONENTS agree with its row in the comparison?

    The eight files are the new description of Step 12 and
    `comparison.configuration_table_8()` is the published one; if they ever
    disagree about whether a configuration has a CNN, one of them is wrong and
    a reader cannot tell which.  So it is checked rather than assumed.
    """
    table = cmp.configuration_table_8() if table is None else table
    from . import CONFIG_MODULES, module_for

    published = table.set_index("config")
    rows = []
    for name in CONFIG_MODULES:
        module = module_for(name)
        label = module.DISPLAY_LABEL
        if label not in published.index:
            rows.append({"file": "%s.py" % name, "config": label,
                         "result": FAIL,
                         "detail": "no row in the comparison table"})
            continue
        line = published.loc[label]
        problems = []
        for component, column in (("CNN", "CNN"), ("attention", "attention"),
                                  ("LSTM", "LSTM")):
            mine = str(module.COMPONENTS.get(component, "")).lower()
            theirs = str(line.get(column, "")).lower()
            mine_yes = mine.startswith("yes")
            theirs_yes = theirs.startswith("yes")
            if mine_yes != theirs_yes:
                problems.append("%s: the file says %r, the table says %r"
                                % (component, mine, theirs))
        for component, column in (("autoencoder", "AE"), ("PCA", "PCA")):
            mine_yes = str(module.COMPONENTS.get(component, "")).lower(
                ).startswith("yes")
            theirs_yes = str(line.get(column, "")).lower().startswith("yes") \
                or str(line.get(column, ""))[:1].isdigit()
            if mine_yes != theirs_yes:
                problems.append("%s: the file says %r, the table says %r"
                                % (component, module.COMPONENTS.get(component),
                                   line.get(column)))
        rows.append({"file": "%s.py" % name, "config": label,
                     "result": OK if not problems else FAIL,
                     "detail": ("the file and the table describe the same "
                                "configuration" if not problems
                                else "; ".join(problems))})
    return pd.DataFrame(rows)


def configuration_tables(verbose: bool = True) -> dict[str, Any]:
    """12.9: what the eight configurations are, and how each one was trained."""
    components = components_table()
    setup = cmp.training_setup_table()
    agreement = files_agree_with_table()
    if verbose:
        dl.show_table(components, all_columns=True,
                      note="Declared by the eight configuration files "
                           "themselves (their COMPONENTS dict).")
        dl.show_table(agreement, all_columns=True,
                      note="Each file's own description against its row in "
                           "the published configuration table. A [FAIL] here "
                           "means the two disagree and a reader cannot tell "
                           "which is right.")
    return {"components": components, "training_setup": setup,
            "agreement": agreement,
            "ok": bool(not (agreement["result"] == FAIL).any())}


# ===========================================================================
# the majority-class reference the tables and the discussion quote
# ===========================================================================
def majority_accuracy(verbose: bool = True) -> dict[str, Any]:
    """What a predictor that answers with one class already scores.

    Not a model and never in the ranking: it is the reference line the
    accuracy column is read against, and the reason macro-F1 and G-mean are
    the comparison metrics.  Computed through the existing helpers, with
    nothing trained.
    """
    from .. import evaluation as ev

    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder8 is None:
        return {"accuracy": None, "shares": pd.DataFrame(),
                "detail": config.why_unusable(config.STEP8_OUTPUT_DIR)}
    loaded = ev.load_evaluation_inputs(verbose=False)
    y_test = loaded["y_test"]
    label_names = ev.label_names_from_mapping(loaded["label_mapping"])
    y_train = pd.read_csv(Path(folder8) / config.STEP8_FILES["y_train"],
                          low_memory=False)
    predictions = s12.majority_class_predictions(y_train, len(y_test))
    truth = np.asarray(y_test[s12.TARGET_COLUMN])
    n_classes = int(max(int(truth.max()),
                        int(y_train[s12.TARGET_COLUMN].max())) + 1)
    headline = ev.headline_metrics(truth, predictions, None, n_classes)
    shares = s12.class_shares(y_test, label_names)
    out = {"accuracy": float(headline.get("accuracy") or 0.0),
           "f1_macro": float(headline.get("f1_macro") or 0.0),
           "g_mean": float(headline.get("g_mean") or 0.0),
           "shares": shares,
           "detail": "predicts the most frequent TRAINING class for every flow"}
    if verbose:
        dl.show_table(shares,
                      note="The test set by class. A single-class predictor "
                           "already reaches %.4f accuracy with a macro-F1 of "
                           "%.4f, which is why accuracy is not the comparison "
                           "metric." % (out["accuracy"], out["f1_macro"]))
    return out


# ===========================================================================
# 12.11  PER-CLASS F1
# ===========================================================================
def per_class_sections(step12: dict[str, Any], verbose: bool = True
                       ) -> dict[str, pd.DataFrame]:
    """Per-class F1, full and duplicate-free, as run_step12_complete built them."""
    out = {"per_class": step12.get("per_class"),
           "per_class_free": step12.get("per_class_free"),
           "zero_recall": step12.get("zero_recall")}
    if verbose:
        for key, scope in (("per_class", "full test set"),
                           ("per_class_free", "duplicate-free subset")):
            frame = out.get(key)
            if frame is not None and len(frame):
                dl.show_table(frame, all_columns=True,
                              note="Per-class F1 on the %s, with each class's "
                                   "test support." % scope)
        zero = out.get("zero_recall")
        if zero is not None and len(zero):
            dl.show_table(zero, all_columns=True,
                          note="A G-mean of zero means at least one class is "
                               "never predicted; this names which.")
    return out


# ===========================================================================
# 12.12  THE FIGURES
# ===========================================================================
def figure_table(charts: dict[str, str] | None) -> pd.DataFrame:
    """Every figure that was written, as PNG and PDF, with its folder."""
    if not charts:
        return pd.DataFrame()
    rows = []
    for name, path in sorted(charts.items()):
        where, suffix = (name.split(":") + [""])[:2]
        rows.append({"folder": where, "format": suffix,
                     "file": Path(path).name, "path": path,
                     "exists": Path(path).exists()})
    return pd.DataFrame(rows)


def colour_table() -> pd.DataFrame:
    """The fixed colour per configuration, and where it comes from."""
    rows = []
    for key, label in config.STEP12_DISPLAY_ORDER:
        rows.append({"config": label,
                     "file": "%s.py" % common.short_name(key),
                     "colour": cmp.model_colour(key),
                     "from CONFIG_COLOURS": config.CONFIG_COLOURS.get(key),
                     "result": OK if (cmp.model_colour(key)
                                      == config.CONFIG_COLOURS.get(key))
                     else FAIL})
    return pd.DataFrame(rows)


# ===========================================================================
# 12.13  HEAD-TO-HEAD
# ===========================================================================
def head_to_head_highlights(head: pd.DataFrame | None,
                            metric: str = "F1 (macro)") -> pd.DataFrame:
    """One line per pair: the macro-F1 difference and which side it favours."""
    if head is None or not len(head):
        return pd.DataFrame()
    rows = []
    for pair in dict.fromkeys(head["pair"]):
        part = head[(head["pair"] == pair) & (head["metric"] == metric)]
        if not len(part):
            continue
        line = part.iloc[0]
        rows.append({"pair": pair, "metric": metric,
                     "left": line["left"], "right": line["right"],
                     "difference (left - right)":
                         line["difference (left - right)"],
                     "favours": line["favours"]})
    return pd.DataFrame(rows)


# ===========================================================================
# 12.14  DISCUSSION AND THE THESIS EXPORTS
# ===========================================================================
def discussion(step12: dict[str, Any], provenance: dict[str, Any] | None,
               baseline6: dict[str, Any] | None,
               majority: dict[str, Any] | None = None) -> str:
    """12.14, generated from the computed tables through build_discussion().

    `step12_checks.build_discussion` is unchanged: it ranks the configurations
    on each metric, says where Approach 1 and Approach 2 are NOT best, and
    lists the limitations.  The majority-class paragraph is added here because
    this Step 12 has no sanity section of its own to carry it.
    """
    text = s12.build_discussion(step12, None, provenance, None, baseline6)
    if majority and majority.get("accuracy") is not None:
        shares = majority.get("shares")
        biggest = ("" if shares is None or not len(shares) else
                   " The two largest test classes are %s, %.1f%% of the test "
                   "rows between them."
                   % (" and ".join(shares.head(2)["class"]),
                      float(shares.head(2)["share (%)"].sum())))
        text += ("\n- **Why accuracy is similar for every configuration.** A "
                 "predictor that answers with the single most frequent "
                 "training class and nothing else already reaches **%.4f "
                 "accuracy**, with a macro-F1 of %.4f and a G-mean of %.4f -- "
                 "it never finds one attack class.%s So accuracy is reported "
                 "for completeness and read against that line; macro-F1 and "
                 "G-mean are the comparison metrics."
                 % (majority["accuracy"], majority["f1_macro"],
                    majority["g_mean"], biggest))
    return text


def export_tables(step12: dict[str, Any], provenance: dict[str, Any] | None,
                  configuration: dict[str, Any] | None = None,
                  verbose: bool = True) -> dict[str, str]:
    """Every Step 12 table as CSV and as a booktabs .tex, via export_thesis()."""
    tables = {
        "configuration": cmp.configuration_table_8(),
        "training_setup": (configuration or {}).get(
            "training_setup", cmp.training_setup_table()),
        "comparison": step12.get("comparison"),
        "per_class": step12.get("per_class"),
        "per_class_free": step12.get("per_class_free"),
        "head_to_head": step12.get("head_to_head"),
        "per_class_difference": step12.get("per_class_difference"),
        "zero_recall": step12.get("zero_recall"),
    }
    if provenance is not None:
        tables["provenance"] = provenance.get("provenance")
    return cmp.export_thesis({name: frame for name, frame in tables.items()
                              if frame is not None and len(frame)},
                             verbose=verbose)


# ===========================================================================
# rule 4, at the end: every finished number still equals the saved value
# ===========================================================================
def verify_unchanged(before: dict[str, dict[str, float]],
                     verbose: bool = True) -> pd.DataFrame:
    """The snapshot taken in 12.0 against the files as they stand now."""
    table = common.compare_with_snapshot(before)
    if verbose:
        dl.show_table(table, all_columns=True,
                      note="Rule 4: every finished configuration's saved "
                           "metrics, recorded in 12.0 before anything ran, "
                           "against the same files now (1e-9).")
        failed = int((table["result"] == FAIL).sum())
        print("[%s]  %d of %d configurations unchanged"
              % ("OK" if not failed else "FAIL",
                 int((table["result"] == OK).sum()), len(table)))
    return table
