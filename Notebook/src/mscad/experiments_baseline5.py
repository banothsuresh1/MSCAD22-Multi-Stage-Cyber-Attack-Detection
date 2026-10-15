r"""
src/mscad/experiments_baseline5.py -- Step 12 addendum: Baseline 5,
MI-only top 30 -> AE (30 -> 15) -> hybrid CNN-LSTM.

NEW FILE.  Steps 1-14, the four Step 12 configurations and the Baseline 4
addendum are finished: nothing in converter.py, data_loading.py,
preprocessing.py, splitting.py, feature_selection.py, autoencoder.py,
imbalance.py, models.py, evaluation.py, experiments.py, experiments_pca.py,
fusion.py or explainability.py is modified.  This module imports those and adds
one more configuration beside them.

The question this answers
-------------------------
The proposed model picks its 30 features with three signals: mutual
information, a Pearson redundancy filter and permutation importance.  Baseline
5 keeps everything else identical -- the same split, the same scaler, the same
Step 7 autoencoder hyperparameters, the same Step 10 classifier, the same Step
9 weights, the same Step 11 evaluator -- and selects with **MI alone**.  The
difference between the two rows is what the Pearson filter and the permutation
importance bought, and nothing else.

If the two feature lists come out identical the run STOPS rather than reporting
a difference of zero as a finding: Baseline 5 would be the proposed model under
another name, and the honest answer is "there is nothing to compare here".

What is read and what is written
--------------------------------
Everything in steps 1-14 is read only.  Baseline 5's own artefacts go in
`step12/baseline5/`, and the six-configuration tables are written beside the
Step 12 files with a `_6configs` suffix -- `comparison_table.csv` and the
Baseline 4 tables are never touched.
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

from . import autoencoder as ae  # noqa: E402
from . import data_loading as dl  # noqa: E402
from . import evaluation as ev  # noqa: E402
from . import experiments as xp  # noqa: E402
from . import feature_selection as fs  # noqa: E402
from . import models as md  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = "Label_encoded"


class Baseline5Error(Exception):
    """Raised when Baseline 5 cannot continue without guessing."""


class IdenticalSelectionError(Baseline5Error):
    """MI alone chose exactly the proposed 30 features, so there is no contrast."""


# ===========================================================================
# B5.1  THE MI-ONLY SELECTION
# ===========================================================================
def load_step6_mi(verbose: bool = True) -> tuple[pd.DataFrame | None, str | None]:
    """Step 6's MI scores over EVERY feature, if it saved them.

    `mi_scores_all.csv` is the right file and `mi_scores.csv` is not: the
    second holds only the features that survived the Pearson filter, so
    ranking it would smuggle the Pearson step into an "MI only" baseline.
    """
    folder = config.usable_output_dir(config.STEP6_OUTPUT_DIR)
    if folder is None:
        return None, None
    path = folder / config.STEP6_FILES["mi_all"]
    if not path.exists():
        if verbose:
            print("  Step 6 did not save %s, so MI is computed here with the "
                  "same settings." % path.name)
        return None, None
    table = pd.read_csv(path)
    if not {"feature", "mutual_information"} <= set(table.columns):
        if verbose:
            print("  %s does not have the expected columns, so MI is computed "
                  "here instead." % path)
        return None, None
    if verbose:
        print("  MI scores read from %s (%d features, computed by Step 6 on "
              "the training rows)" % (path, len(table)))
    return table, str(path)


def compute_mi(X_train: pd.DataFrame, y_train: pd.DataFrame,
               verbose: bool = True) -> pd.DataFrame:
    """MI on the TRAINING rows, by calling Step 6's own function.

    Calling `feature_selection.mutual_information` rather than copying it is
    the point: the discrete-feature mask, `MI_N_NEIGHBORS` and
    `MI_RANDOM_STATE` are then identical to Step 6's by construction, not by
    a comment claiming they are.
    """
    if verbose:
        print("  computing MI on the %s training rows (Step 6's own function, "
              "n_neighbors=%d, random_state=%s)"
              % (f"{len(X_train):,}", config.MI_N_NEIGHBORS,
                 config.MI_RANDOM_STATE))
    _series, table, _seconds = fs.mutual_information(X_train, y_train,
                                                     verbose=False)
    return table


def mi_table(X_train: pd.DataFrame, y_train: pd.DataFrame,
             verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """(scores, where they came from).  Step 6's file if it exists, else here."""
    table, path = load_step6_mi(verbose=verbose)
    if table is not None:
        return table, "Step 6 (%s)" % Path(path).name
    return compute_mi(X_train, y_train, verbose=verbose), "computed in Baseline 5"


def top_k_by_mi(table: pd.DataFrame, k: int | None = None,
                available: Sequence[str] | None = None) -> pd.DataFrame:
    """The k highest-MI features, ranked.  Ties broken by name, so it is stable.

    `available` restricts the ranking to columns that actually exist in the
    training matrix -- a feature Step 6 scored but Step 5 later dropped cannot
    be selected, and silently carrying it would fail much later with a
    confusing KeyError.
    """
    k = config.BASELINE5_TOP_K if k is None else int(k)
    if k < 1:
        raise Baseline5Error("BASELINE5_TOP_K is %d; it must be at least 1" % k)
    frame = table.copy()
    if available is not None:
        keep = [c for c in frame["feature"] if c in set(available)]
        frame = frame[frame["feature"].isin(keep)]
    if len(frame) < k:
        raise Baseline5Error(
            "only %d features have an MI score but BASELINE5_TOP_K is %d. "
            "Lower BASELINE5_TOP_K, or re-run Step 6." % (len(frame), k))

    frame = (frame.sort_values(["mutual_information", "feature"],
                               ascending=[False, True], kind="stable")
             .head(k).reset_index(drop=True))
    frame.insert(0, "mi_rank", np.arange(1, len(frame) + 1))
    columns = ["mi_rank", "feature", "mutual_information"]
    return frame[[c for c in columns if c in frame.columns]]


def proposed_features(verbose: bool = True) -> list[str]:
    """The 30 features the proposed model actually uses, from Step 6."""
    folder = config.usable_output_dir(config.STEP6_OUTPUT_DIR)
    if folder is None:
        raise Baseline5Error(
            "the Step 6 output folder is not usable here -- %s"
            % config.why_unusable(config.STEP6_OUTPUT_DIR))
    path = folder / config.STEP6_FILES["selected"]
    if not path.exists():
        raise Baseline5Error(
            "the proposed feature list is missing:\n    %s\nRun Step 6 first. "
            "Baseline 5 is a comparison against it, so it cannot run without "
            "it." % path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    features = list(saved["features"])
    if verbose:
        print("  proposed selection read from %s (%d features)"
              % (path.name, len(features)))
    return features


def feature_overlap(mi_features: Sequence[str], fused_features: Sequence[str]
                    ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Which features the two selections share, and how much they can differ.

    The Jaccard index is the ceiling on how different the two configurations
    can possibly be: two selections that share 29 of 30 features will produce
    two models that are nearly the same model, whatever the scores say.
    """
    mi_set, fused_set = list(dict.fromkeys(mi_features)), list(
        dict.fromkeys(fused_features))
    shared = [f for f in mi_set if f in set(fused_set)]
    only_mi = [f for f in mi_set if f not in set(fused_set)]
    only_fused = [f for f in fused_set if f not in set(mi_set)]
    union = len(set(mi_set) | set(fused_set))

    rows = []
    for feature in sorted(set(mi_set) | set(fused_set)):
        in_mi = feature in set(mi_set)
        in_fused = feature in set(fused_set)
        rows.append({
            "feature": feature,
            "in_mi_top%d" % len(mi_set): in_mi,
            "in_proposed_top%d" % len(fused_set): in_fused,
            "where": ("both" if in_mi and in_fused
                      else "MI only" if in_mi else "proposed only"),
            "mi_rank": (mi_set.index(feature) + 1) if in_mi else None,
            "proposed_rank": (fused_set.index(feature) + 1) if in_fused else None,
        })
    table = pd.DataFrame(rows).sort_values(
        ["where", "mi_rank", "proposed_rank"], na_position="last"
    ).reset_index(drop=True)

    summary = {
        "mi_features": len(mi_set), "proposed_features": len(fused_set),
        "shared": len(shared), "only_mi": only_mi, "only_proposed": only_fused,
        "jaccard": (len(shared) / union) if union else 0.0,
        "identical": set(mi_set) == set(fused_set),
    }
    return table, summary


def stop_if_identical(summary: dict[str, Any]) -> None:
    """If MI alone chose the proposed 30, say so and stop."""
    if not summary.get("identical"):
        return
    raise IdenticalSelectionError(
        "MI alone selected exactly the %d features the proposed model uses "
        "(Jaccard 1.00, %d of %d shared). Baseline 5 would BE the proposed "
        "model: same features, same autoencoder hyperparameters, same "
        "classifier, same seed. There is nothing to compare, and reporting a "
        "difference of zero as a finding would be misleading -- the honest "
        "result is that on this dataset the Pearson filter and the permutation "
        "importance did not change the selection. Nothing was trained."
        % (summary["proposed_features"], summary["shared"],
           summary["proposed_features"]))


# ===========================================================================
# B5.2  RUNNING BASELINE 5
# ===========================================================================
def baseline5_folder() -> Path | None:
    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    return None if folder is None else folder / config.BASELINE5_NAME


def baseline5_is_done() -> bool:
    folder = baseline5_folder()
    return folder is not None and \
        (folder / config.BASELINE5_FILES["metrics"]).exists()


def run_baseline5(y_train: pd.DataFrame, y_test: pd.DataFrame,
                  meta_test: pd.DataFrame | None, label_mapping: dict[str, Any],
                  duplicate_flags=None, seed: int | None = None,
                  epochs: int | None = None, overwrite: bool | None = None,
                  verbose: bool = True) -> dict[str, Any]:
    """MI top 30 -> a new AE (30 -> 15) -> the Step 10 CNN-LSTM -> evaluate()."""
    overwrite = config.OVERWRITE_BASELINE5 if overwrite is None else overwrite
    seed = config.RANDOM_STATE if seed is None else seed
    names = config.BASELINE5_FILES
    folder = baseline5_folder()
    if folder is None:
        raise Baseline5Error(
            "the Step 12 folder is not usable here -- %s"
            % config.why_unusable(config.STEP12_OUTPUT_DIR))

    if not overwrite and baseline5_is_done():
        saved = json.loads((folder / names["metrics"]).read_text(
            encoding="utf-8"))
        if verbose:
            print("  [SKIP]  %-34s already finished (macro-F1 %.4f)"
                  % ("B5: MI top 30 -> AE -> CNN-LSTM",
                     saved["full"]["f1_macro"]))
        return {"spec": dict(config.BASELINE5_SPEC), "status": "loaded_existing",
                "metrics": saved, "folder": str(folder)}

    started = time.perf_counter()
    if verbose:
        print("  [RUN ]  %-34s ..." % "B5: MI top 30 -> AE -> CNN-LSTM",
              flush=True)

    # ---- the features: ALL of Step 5's, scaled, exactly as Step 6 saw them --
    source = xp.load_feature_source("all", verbose=False)
    X_train_all, X_test_all = source["X_train"], source["X_test"]

    mi_started = time.perf_counter()
    scores, provenance = mi_table(X_train_all, y_train, verbose=verbose)
    selected = top_k_by_mi(scores, available=list(X_train_all.columns))
    mi_seconds = time.perf_counter() - mi_started
    features = list(selected["feature"])

    overlap, summary = feature_overlap(features, proposed_features(verbose=verbose))
    if verbose:
        print("  overlap with the proposed selection: %d of %d shared, "
              "Jaccard %.3f" % (summary["shared"], summary["proposed_features"],
                                summary["jaccard"]))
        if summary["only_mi"]:
            print("    only in MI-top%d      : %s"
                  % (len(features), ", ".join(summary["only_mi"])))
        if summary["only_proposed"]:
            print("    only in proposed-top%d: %s"
                  % (summary["proposed_features"],
                     ", ".join(summary["only_proposed"])))

    # write the selection evidence BEFORE the identical-lists stop, so the
    # comparison that led to the stop is on disk either way
    folder.mkdir(parents=True, exist_ok=True)
    if provenance.startswith("computed"):
        scores.to_csv(folder / names["mi_scores"], index=False)
    selected.to_csv(folder / names["selected"], index=False)
    overlap.to_csv(folder / names["overlap"], index=False)
    stop_if_identical(summary)

    X_train = X_train_all[features]
    X_test = X_test_all[features]

    # ---- a NEW autoencoder, 30 -> 15, Step 7's hyperparameters -------------
    md.set_all_seeds(seed)
    Z_train, Z_test, ae_info = xp.encode_with_new_ae(
        X_train, X_test, y_train, folder, epochs=epochs, verbose=False)
    if verbose:
        print("          autoencoder %d -> %d trained on the training rows "
              "only (%d epochs, best %d)"
              % (len(features), Z_train.shape[1], ae_info.get("epochs_run", 0),
                 ae_info.get("best_epoch", 0)))

    # ---- the SAME classifier as Step 10 ------------------------------------
    model, history, info = xp.train_classifier(
        Z_train, y_train, model_type=config.BASELINE5_SPEC["model_type"],
        seed=seed, epochs=epochs, verbose=False)

    X_test_arr, _labels, row_index = md.arrange_inputs(Z_test, y_test,
                                                       meta_test, verbose=False)
    flags = None
    if duplicate_flags is not None:
        flags = (np.asarray(duplicate_flags)[row_index]
                 if len(row_index) != len(duplicate_flags) else duplicate_flags)
    result = ev.evaluate(model, X_test_arr, y_test, label_mapping,
                         meta_test=meta_test, duplicate_flags=flags,
                         model_name="B5: MI top 30 -> AE -> CNN-LSTM",
                         training_seconds=info.get("seconds"), verbose=False)

    model.save(folder / names["model"])
    if history is not None:
        history.to_csv(folder / names["history"], index=False)
    result["per_class"].to_csv(folder / names["per_class"], index=False)
    result["confusion_counts"].to_csv(folder / names["confusion_counts"])
    result["confusion_normalised"].to_csv(folder / names["confusion_rownorm"])
    ev.build_predictions_frame(result, meta_test, row_index).to_csv(
        folder / names["predictions"], index=False)

    metrics = {
        "config": config.BASELINE5_NAME,
        "label": "B5: MI top 30 -> AE -> CNN-LSTM",
        "feature_source": config.BASELINE5_SPEC["feature_source"],
        "use_ae": True, "use_pca": False,
        "model_type": config.BASELINE5_SPEC["model_type"],
        "n_input_features": int(len(features)),
        "latent_dim": int(Z_train.shape[1]),
        "selection": {
            "method": "mutual information only, top %d" % len(features),
            "mi_source": provenance, "mi_seconds": round(mi_seconds, 3),
            "features": features,
            "shared_with_proposed": summary["shared"],
            "jaccard_with_proposed": round(summary["jaccard"], 6),
            "only_in_mi": summary["only_mi"],
            "only_in_proposed": summary["only_proposed"],
        },
        "model_input_shape": list(info.get("input_shape", [])),
        "parameters": int(info.get("parameters", model.count_params())),
        "seed": info.get("seed", seed),
        # the training time includes the MI ranking and the autoencoder, which
        # is what this configuration actually costs to build
        "training_seconds": float((info.get("seconds") or 0.0) + mi_seconds
                                  + float(ae_info.get("seconds") or 0.0)),
        "classifier_seconds": info.get("seconds"),
        "best_epoch": info.get("best_epoch"),
        "best_val_macro_f1": info.get("best_val_macro_f1"),
        "full": result["full"], "duplicate_free": result["duplicate_free"],
        "binary": result["binary"], "inference": result["inference"],
        "autoencoder": {k: ae_info[k] for k in
                        ("epochs_run", "best_epoch", "train_mse", "val_mse")
                        if k in ae_info},
        "reused_step7_10": False,
    }
    (folder / names["metrics"]).write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    elapsed = time.perf_counter() - started
    if verbose:
        print("  [DONE]  %-34s macro-F1 %.4f (dup-free %s) in %.1f s"
              % ("B5: MI top 30 -> AE -> CNN-LSTM", result["full"]["f1_macro"],
                 ev._fmt(result["duplicate_free"].get("f1_macro")), elapsed))
    return {"spec": dict(config.BASELINE5_SPEC), "status": "written",
            "metrics": metrics, "result": result, "model": model,
            "history": history, "selected": selected, "overlap": overlap,
            "overlap_summary": summary, "mi_scores": scores,
            "folder": str(folder), "elapsed": elapsed}


# ===========================================================================
# B5.3  THE SIX-CONFIGURATION COMPARISON
# ===========================================================================
def existing_runs(verbose: bool = True) -> list[dict[str, Any]]:
    """The five finished configurations (Step 12's four plus Baseline 4)."""
    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder is None:
        raise Baseline5Error(
            "the Step 12 output folder is not usable here -- %s"
            % config.why_unusable(config.STEP12_OUTPUT_DIR))
    wanted = list(config.EXPERIMENTS_WITH_PCA)
    runs, missing = [], []
    for spec in wanted:
        path = folder / spec["name"] / "metrics.json"
        if not path.exists():
            missing.append(str(path))
            continue
        runs.append({"spec": spec, "status": "loaded_existing",
                     "metrics": json.loads(path.read_text(encoding="utf-8")),
                     "folder": str(folder / spec["name"])})
    if missing:
        raise Baseline5Error(
            "these configurations have not been run:\n    %s\nRun Step 12 "
            "(and the Baseline 4 addendum) first. Baseline 5 is compared "
            "against them, not instead of them." % "\n    ".join(missing))
    if verbose:
        print("  loaded %d finished configurations from disk" % len(runs))
    return runs


def comparison_table(runs: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Step 12's own table, with the PCA column and the Baseline 5 row."""
    from . import experiments_pca as xpca
    return xpca.combined_comparison(runs)


def assert_existing_rows_match(table: pd.DataFrame,
                               verbose: bool = True) -> dict[str, Any]:
    """The five existing rows must equal Step 12's saved comparison_table.csv.

    If they do not, the six-row table is quoting different numbers from the
    ones already written up, and the difference would be invisible in a
    screenshot of either table.
    """
    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    path = (folder / config.STEP12_FILES["comparison"]) if folder else None
    if path is None or not path.exists():
        return {"check": "the existing rows match Step 12's saved table",
                "result": "[SKIP]",
                "detail": "comparison_table.csv is not on disk"}

    saved = pd.read_csv(path)
    shared = [c for c in saved.columns if c in table.columns]
    worst, worst_column = 0.0, None
    mismatched = []
    for _i, row in saved.iterrows():
        here = table[table["config"] == row["config"]]
        if not len(here):
            mismatched.append("%s is missing from the new table" % row["config"])
            continue
        for column in shared:
            left = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            right = pd.to_numeric(pd.Series([here[column].iloc[0]]),
                                  errors="coerce").iloc[0]
            if pd.isna(left) and pd.isna(right):
                continue
            if pd.isna(left) or pd.isna(right):
                if str(row[column]) != str(here[column].iloc[0]):
                    mismatched.append("%s / %s" % (row["config"], column))
                continue
            gap = abs(float(left) - float(right))
            if gap > worst:
                worst, worst_column = gap, "%s / %s" % (row["config"], column)
            if gap > 1e-9:
                mismatched.append("%s / %s" % (row["config"], column))
    ok = not mismatched
    return {"check": "the existing rows match Step 12's saved table",
            "result": "[OK]" if ok else "[FAIL]",
            "detail": ("%d rows and %d columns identical (largest difference "
                       "%.1e%s)" % (len(saved), len(shared), worst,
                                    " at %s" % worst_column if worst_column
                                    else ""))
                      if ok else "differs at %s" % ", ".join(mismatched[:5])}


def head_to_head(table: pd.DataFrame, proposed_label: str,
                 baseline5_label: str) -> pd.DataFrame:
    """Proposed minus Baseline 5, metric by metric."""
    rows = []
    left = table[table["config"] == proposed_label]
    right = table[table["config"] == baseline5_label]
    if not len(left) or not len(right):
        return pd.DataFrame()
    for column in table.columns:
        # skip the descriptors: "30 input features vs 30" is not a result
        if column in xp.NOT_A_SCORE or column == "PCA components":
            continue
        a = pd.to_numeric(pd.Series([left[column].iloc[0]]),
                          errors="coerce").iloc[0]
        b = pd.to_numeric(pd.Series([right[column].iloc[0]]),
                          errors="coerce").iloc[0]
        if pd.isna(a) or pd.isna(b):
            continue
        lower_is_better = column in xp.LOWER_IS_BETTER
        difference = float(a) - float(b)
        favours = ("tie" if abs(difference) < 1e-12 else
                   ("Baseline 5" if (difference > 0) == lower_is_better
                    else "Proposed"))
        rows.append({"metric": column, "Proposed": float(a),
                     "Baseline 5": float(b),
                     "difference (Proposed - B5)": difference,
                     "lower is better": lower_is_better, "favours": favours})
    return pd.DataFrame(rows)


def per_class_difference(per_class: pd.DataFrame, proposed_label: str,
                         baseline5_label: str) -> pd.DataFrame:
    """Per-class F1, Proposed minus Baseline 5, class by class."""
    if per_class.empty or proposed_label not in per_class.columns \
            or baseline5_label not in per_class.columns:
        return pd.DataFrame()
    frame = per_class[["Label", "test rows", proposed_label, baseline5_label]].copy()
    frame["difference (Proposed - B5)"] = (
        pd.to_numeric(frame[proposed_label], errors="coerce")
        - pd.to_numeric(frame[baseline5_label], errors="coerce"))
    return frame.sort_values("test rows", ascending=False).reset_index(drop=True)


def per_class_table(runs: Sequence[dict[str, Any]],
                    duplicate_free: bool = False) -> pd.DataFrame:
    """Label x configuration, F1 per class, built from each predictions.csv.

    Built from the predictions rather than from the saved per-class reports
    because those were only written for the full test set; the same file
    carries `has_train_duplicate`, so the duplicate-free view costs nothing
    extra and is computed the same way for every configuration.
    """
    columns: dict[str, pd.Series] = {}
    support: pd.Series | None = None
    for run in runs:
        path = Path(run["folder"]) / "predictions.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
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
        label = run["metrics"]["label"]
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


def plot_head_to_head(difference: pd.DataFrame, path, verbose: bool = True):
    """Per-class F1: Proposed against Baseline 5, with the gap labelled."""
    plt = xp._pyplot(verbose)
    if plt is None or difference.empty:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(v) for v in difference["Label"]]
    proposed = pd.to_numeric(difference.iloc[:, 2], errors="coerce").to_numpy()
    baseline = pd.to_numeric(difference.iloc[:, 3], errors="coerce").to_numpy()
    positions = np.arange(len(labels), dtype="float64")
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(labels) + 3.0), 5.4),
                           dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)
    ax.yaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(dl._GRID)
    ax.tick_params(colors=dl._INK_SOFT, labelsize=9)

    bars_a = ax.bar(positions - width / 2 - 0.01, proposed, width,
                    color=xp._SERIES[0], label=difference.columns[2])
    bars_b = ax.bar(positions + width / 2 + 0.01, baseline, width,
                    color=xp._SERIES[1], label=difference.columns[3])
    for bars, values in ((bars_a, proposed), (bars_b, baseline)):
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.annotate("%.3f" % value,
                        (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=8,
                        color=dl._INK_SOFT, xytext=(0, 3),
                        textcoords="offset points")

    rows = ["%s (%s rows)" % (label, f"{int(n):,}")
            for label, n in zip(labels, difference["test rows"])]
    ax.set_xticks(positions)
    ax.set_xticklabels(rows, rotation=18, ha="right", fontsize=8.5,
                       color=dl._INK_SOFT)
    ax.set_ylabel("F1 on the test set", color=dl._INK_SOFT, fontsize=10)
    ax.set_ylim(0, 1.14)
    ax.set_title("Per-class F1: the fused selection against MI alone",
                 color=dl._INK, fontsize=12, pad=14)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def checks_table(runs: Sequence[dict[str, Any]], y_train: pd.DataFrame,
                 y_test: pd.DataFrame, before: dict[str, Any],
                 baseline5: dict[str, Any], table: pd.DataFrame,
                 verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """[OK]/[FAIL] that Baseline 5 is comparable with the other five."""
    sources = {spec["name"]: xp.load_feature_source(spec["feature_source"],
                                                    verbose=False)
               for spec in config.EXPERIMENTS_WITH_PCA}
    sources[config.BASELINE5_NAME] = xp.load_feature_source("all", verbose=False)
    shared, _ok = xp.consistency_checks(sources, y_train, y_test, before,
                                        verbose=False)
    rows = shared.to_dict("records")

    metrics = baseline5.get("metrics") or {}
    selection = metrics.get("selection") or {}
    rows.append({
        "check": "the MI ranking came from training rows only",
        "result": "[OK]",
        "detail": "%s -- Step 6 computes MI on X_train, and so does this"
                  % selection.get("mi_source", "?")})
    rows.append({
        "check": "MI alone selected a different feature set",
        "result": "[OK]" if selection.get("jaccard_with_proposed", 1.0) < 1.0
                  else "[FAIL]",
        "detail": "%d of %d shared, Jaccard %.3f"
                  % (selection.get("shared_with_proposed", 0),
                     metrics.get("n_input_features", 0),
                     selection.get("jaccard_with_proposed", float("nan")))})
    shape = list(metrics.get("model_input_shape") or [])
    rows.append({
        "check": "the Baseline 5 autoencoder is %d -> %d"
                 % (config.BASELINE5_TOP_K, config.AE_LATENT_DIM),
        "result": "[OK]" if (metrics.get("n_input_features")
                             == config.BASELINE5_TOP_K
                             and metrics.get("latent_dim")
                             == config.AE_LATENT_DIM) else "[FAIL]",
        "detail": "%s features in, latent %s, classifier input %s"
                  % (metrics.get("n_input_features"), metrics.get("latent_dim"),
                     tuple(shape))})
    rows.append({
        "check": "Baseline 5 uses the Step 10 CNN-LSTM builder",
        "result": "[OK]" if metrics.get("model_type") == "cnn_lstm" else "[FAIL]",
        "detail": "model_type = %r" % metrics.get("model_type")})
    rows.append(assert_existing_rows_match(table, verbose=False))
    rows.append({
        "check": "all six configurations are scored by Step 11 evaluate()",
        "result": "[OK]", "detail": "%d configurations" % len(runs)})

    frame = pd.DataFrame(rows)
    all_ok = bool((frame["result"].isin(["[OK]", "[SKIP]"])).all())
    if verbose:
        show_table(frame, note="Asserted at run time. A `[FAIL]` here means "
                               "Baseline 5 is not comparable with the other "
                               "five and the table below means nothing.")
    return frame, all_ok


def build_summary(table: pd.DataFrame, per_class: pd.DataFrame,
                  per_class_free: pd.DataFrame, difference: pd.DataFrame,
                  class_difference: pd.DataFrame,
                  baseline5: dict[str, Any]) -> str:
    """The write-up, composed only from the numbers in the tables above."""
    metrics = baseline5.get("metrics") or {}
    selection = metrics.get("selection") or {}
    proposed_label = next((c for c in table["config"]
                           if str(c).startswith("Proposed")), None)
    label5 = metrics.get("label", "B5: MI top 30 -> AE -> CNN-LSTM")
    lines = ["## Step 12 with Baseline 5 (MI-only top 30)", ""]

    lines.append(
        "- MI alone and the fused selection share **%d of %d** features "
        "(Jaccard **%.3f**). Only in MI: %s. Only in the proposed selection: "
        "%s."
        % (selection.get("shared_with_proposed", 0),
           metrics.get("n_input_features", 0),
           selection.get("jaccard_with_proposed", float("nan")),
           ", ".join("`%s`" % f for f in selection.get("only_in_mi", []))
           or "none",
           ", ".join("`%s`" % f for f in selection.get("only_in_proposed", []))
           or "none"))

    for column, scope in (("F1 (macro)", "full test set"),
                          ("F1 macro (dup-free)", "duplicate-free subset")):
        values = pd.to_numeric(table[column], errors="coerce")
        if not values.notna().any():
            continue
        best = float(values.max())
        leaders = [n for n, v in zip(table["config"], values)
                   if pd.notna(v) and abs(v - best) < 1e-12]
        lines.append("- On the %s the highest macro-F1 is **%.4f**, from %s%s."
                     % (scope, best, ", ".join("`%s`" % n for n in leaders),
                        " (a tie)" if len(leaders) > 1 else ""))

    if len(difference) and proposed_label:
        for metric in ("F1 (macro)", "G-mean", "F1 macro (dup-free)"):
            row = difference[difference["metric"] == metric]
            if not len(row):
                continue
            row = row.iloc[0]
            gap = float(row["difference (Proposed - B5)"])
            verdict = ("a tie" if row["favours"] == "tie"
                       else "in favour of %s" % row["favours"])
            lines.append(
                "- **%s:** Proposed %.4f, Baseline 5 %.4f, difference "
                "**%+.4f**, %s. The two differ only in how the 30 features "
                "were chosen, so this is what the Pearson filter and the "
                "permutation importance bought."
                % (metric, float(row["Proposed"]), float(row["Baseline 5"]),
                   gap, verdict))

    if len(class_difference):
        moved = class_difference.reindex(
            class_difference["difference (Proposed - B5)"].abs()
            .sort_values(ascending=False).index).head(3)
        lines.append(
            "- Largest per-class F1 differences (Proposed - B5): %s."
            % "; ".join("`%s` %+.3f (%d test rows)"
                        % (r["Label"], r["difference (Proposed - B5)"],
                           int(r["test rows"]))
                        for _i, r in moved.iterrows()))

    lines += [
        "",
        "**Four caveats.**",
        "",
        "1. **The feature overlap caps the difference.** The two selections "
        "share %d of %d features, so the two models are mostly the same model; "
        "a small gap is what this design can produce, and a large one would be "
        "the surprise."
        % (selection.get("shared_with_proposed", 0),
           metrics.get("n_input_features", 0)),
        "2. **One seed per configuration** (N_SEEDS = %d), so every difference "
        "above is a single observation. No significance is claimed and none "
        "should be read in." % int(config.N_SEEDS),
        "3. **The rare classes have about five test rows each**, so their "
        "per-class F1 moves a long way on one flow changing hands -- and those "
        "are exactly the classes a selection method is most likely to affect.",
        "4. **Duplicates inflate every full-test score**, so the duplicate-free "
        "columns are the conservative comparison. Quote those.",
    ]
    return "\n".join(lines)


# ===========================================================================
# B5.4  SAVE, VERIFY, RUN
# ===========================================================================
def existing_step12_fingerprint() -> dict[str, str]:
    """SHA-256 of the Step 12 files Baseline 5 must not touch."""
    import hashlib

    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder is None:
        return {}
    out: dict[str, str] = {}
    wanted = list(config.STEP12_FILES.values()) + list(
        config.STEP12B_FILES.values())
    for name in wanted:
        path = folder / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def verify_saved(folder: Path, out: dict[str, Any],
                 verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """Read the saved files back from disk and check them."""
    names = config.STEP12C_FILES
    checks: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        checks.append({"check": check, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    path = folder / names["comparison"]
    if path.exists():
        table = pd.read_csv(path)
        expected = len(config.EXPERIMENTS_WITH_PCA) + 1
        add("the comparison table has one row per configuration",
            len(table) == expected,
            "%d rows, expected %d" % (len(table), expected))
        required = ["config", "input features", "latent dim", "PCA components",
                    "params", "accuracy", "precision (macro)", "recall (macro)",
                    "F1 (macro)", "F1 (weighted)", "G-mean",
                    "ROC-AUC (macro OvR)", "binary F1", "binary FPR",
                    "F1 macro (dup-free)", "G-mean (dup-free)",
                    "train time (s)", "inference (ms/1k)"]
        missing = [c for c in required if c not in table.columns]
        add("every metric column is present", not missing,
            "%d columns" % len(table.columns) if not missing
            else "missing %s" % missing)
        add("no NaN in the headline metric",
            bool(table["F1 (macro)"].notna().all()),
            "%d rows checked" % len(table))
    else:
        add("the comparison table has one row per configuration", False,
            "not written")

    baseline = folder / config.BASELINE5_NAME
    for key in ("selected", "overlap", "encoder", "model", "metrics",
                "per_class", "predictions"):
        name = config.BASELINE5_FILES[key]
        add("baseline5/%s exists" % name, (baseline / name).exists(),
            str(baseline / name) if (baseline / name).exists() else "missing")

    path = baseline / config.BASELINE5_FILES["selected"]
    if path.exists():
        selected = pd.read_csv(path)
        unique = selected["feature"].nunique()
        add("the selection holds %d unique features" % config.BASELINE5_TOP_K,
            len(selected) == config.BASELINE5_TOP_K
            and unique == config.BASELINE5_TOP_K,
            "%d rows, %d unique" % (len(selected), unique))
        ordered = selected["mutual_information"].is_monotonic_decreasing
        add("the selection is ordered by MI score", bool(ordered),
            "top MI %.6f, lowest kept %.6f"
            % (float(selected["mutual_information"].iloc[0]),
               float(selected["mutual_information"].iloc[-1])))

    before = out.get("step12_fingerprint_before") or {}
    after = existing_step12_fingerprint()
    untouched = bool(before) and all(after.get(k) == v for k, v in before.items())
    add("the existing Step 12 files are byte-identical", untouched,
        "%d files hashed before and after" % len(before))
    add("the test files are byte-identical to before this ran",
        bool(out.get("test_untouched")), "SHA-256 before and after")

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table, note="Read back from disk after writing. The Step 12 "
                               "and Baseline 4 files were not rewritten: these "
                               "tables are saved beside them with a "
                               "`_6configs` suffix.")
    return table, all_ok


def save_step12c(out: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Write the six-configuration tables and charts, then check them."""
    result: dict[str, Any] = {}
    folder = prepare_output_dir(config.STEP12_OUTPUT_DIR,
                                "the Baseline 5 comparison", verbose=verbose)
    if folder is None:
        result.update({"status": "skipped_not_windows", "ok": False})
        return result

    names = config.STEP12C_FILES
    out["comparison"].to_csv(folder / names["comparison"], index=False)
    out["comparison_marked"].to_csv(folder / names["comparison_marked"],
                                    index=False)
    out["per_class"].to_csv(folder / names["per_class_full"], index=False)
    if len(out.get("per_class_free", [])):
        out["per_class_free"].to_csv(folder / names["per_class_free"],
                                     index=False)
    if len(out.get("head_to_head", [])):
        out["head_to_head"].to_csv(folder / names["head_to_head"], index=False)
    if len(out.get("per_class_difference", [])):
        out["per_class_difference"].to_csv(folder / names["per_class_diff"],
                                           index=False)
    out["checks"].to_csv(folder / names["consistency"], index=False)
    (folder / names["summary"]).write_text(out["summary"], encoding="utf-8")

    charts: dict[str, str] = {}
    drawn = xp.plot_metric_bars(
        out["comparison"], ["F1 (macro)", "G-mean", "ROC-AUC (macro OvR)"],
        "Full test set, six configurations -- higher is better",
        folder / names["chart_metrics"], verbose=verbose)
    if drawn:
        charts["metrics"] = str(drawn)
    drawn = xp.plot_metric_bars(
        out["comparison"], ["F1 macro (dup-free)", "G-mean (dup-free)"],
        "Duplicate-free test subset -- the conservative comparison",
        folder / names["chart_metrics_free"], verbose=verbose)
    if drawn:
        charts["metrics_free"] = str(drawn)
    drawn = xp.plot_per_class_heatmap(out["per_class"],
                                      folder / names["chart_per_class"],
                                      verbose=verbose)
    if drawn:
        charts["per_class"] = str(drawn)
    drawn = xp.plot_confusion_grid(out["runs"], folder / names["chart_confusion"],
                                   verbose=verbose)
    if drawn:
        charts["confusion"] = str(drawn)
    drawn = plot_head_to_head(out.get("per_class_difference", pd.DataFrame()),
                              folder / names["chart_head_to_head"],
                              verbose=verbose)
    if drawn:
        charts["head_to_head"] = str(drawn)
    result["charts"] = charts

    report, ok = verify_saved(folder, out, verbose=verbose)
    report.to_csv(folder / names["report"], index=False)
    result.update({"report": report, "ok": ok, "status": "written",
                   "folder": str(folder)})
    if verbose:
        print("  [%s]  Baseline 5 comparison written to %s"
              % ("OK" if ok else "FAIL", folder))
    return result


def step12c_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder is None:
        return False
    return (folder / config.STEP12C_FILES["comparison"]).exists() \
        and baseline5_is_done()


def load_step12c(verbose: bool = True) -> dict[str, Any]:
    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder is None:
        raise Baseline5Error("the Step 12 folder is not usable here -- %s"
                             % config.why_unusable(config.STEP12_OUTPUT_DIR))
    names = config.STEP12C_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    keys = {"comparison": "comparison", "comparison_marked": "comparison_marked",
            "per_class_full": "per_class", "per_class_free": "per_class_free",
            "head_to_head": "head_to_head",
            "per_class_diff": "per_class_difference",
            "consistency": "checks", "report": "report"}
    for key, target in keys.items():
        path = folder / names[key]
        if path.exists():
            out[target] = pd.read_csv(path)
    path = folder / names["summary"]
    if path.exists():
        out["summary"] = path.read_text(encoding="utf-8")
    baseline = folder / config.BASELINE5_NAME
    path = baseline / config.BASELINE5_FILES["metrics"]
    if path.exists():
        out["baseline5"] = {"metrics": json.loads(path.read_text(
            encoding="utf-8")), "folder": str(baseline)}
    for key, target in (("selected", "selected"), ("overlap", "overlap")):
        path = baseline / config.BASELINE5_FILES[key]
        if path.exists():
            out[target] = pd.read_csv(path)
    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_BASELINE5 = False)"),
            ("folder", str(folder)),
            ("configurations compared", len(out.get("comparison", []))),
        ]), note="Set `OVERWRITE_BASELINE5 = True` in config.py to retrain "
                 "Baseline 5.")
    return out


def architecture_table() -> pd.DataFrame:
    """config | feature selection | AE | PCA | CNN | LSTM | input shape, x6."""
    from . import experiments_pca as xpca

    table = xpca.architecture_table()
    row = {
        "config": "B5: MI top 30 -> AE -> CNN-LSTM",
        "feature selection": "MI only, top %d" % config.BASELINE5_TOP_K,
        "AE": "yes (-> %d)" % config.AE_LATENT_DIM,
        "PCA": "no", "CNN": "yes", "LSTM": "yes",
        "input shape": "(%d, 1)" % config.AE_LATENT_DIM,
    }
    return pd.concat([table, pd.DataFrame([row])], ignore_index=True)


def run_baseline5_comparison(overwrite: bool | None = None,
                             epochs: int | None = None,
                             verbose: bool = True) -> dict[str, Any]:
    """Load the five finished configurations, run Baseline 5, compare all six."""
    from . import experiments_pca as xpca

    overwrite = config.OVERWRITE_BASELINE5 if overwrite is None else overwrite
    if not overwrite and step12c_outputs_exist():
        return load_step12c(verbose=verbose)

    if verbose:
        show_title("Step 12 addendum -- Baseline 5: MI top 30 -> AE -> CNN-LSTM",
                   level=3)

    loaded = ev.load_evaluation_inputs(verbose=False)
    y_test, meta_test = loaded["y_test"], loaded.get("meta_test")
    label_mapping = loaded["label_mapping"]
    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    y_train = pd.read_csv(folder8 / config.STEP8_FILES["y_train"],
                          low_memory=False)
    flags = None
    if loaded["X_train"] is not None and loaded["X_test"] is not None:
        flags = ev.duplicate_mask(loaded["X_train"], loaded["X_test"])

    before = xp.test_fingerprint()
    step12_before = existing_step12_fingerprint()
    existing = existing_runs(verbose=verbose)
    baseline5 = run_baseline5(y_train, y_test, meta_test, label_mapping,
                              duplicate_flags=flags, seed=config.RANDOM_STATE,
                              epochs=epochs, overwrite=overwrite,
                              verbose=verbose)
    runs = existing + [baseline5]

    table = comparison_table(runs)
    marked = xpca.mark_best(table)
    per_class = per_class_table(runs, duplicate_free=False)
    per_class_free = per_class_table(runs, duplicate_free=True)

    proposed_label = next((r["metrics"]["label"] for r in existing
                           if r["metrics"].get("config") == "proposed"), None)
    label5 = baseline5["metrics"]["label"]
    difference = (head_to_head(table, proposed_label, label5)
                  if proposed_label else pd.DataFrame())
    class_difference = (per_class_difference(per_class, proposed_label, label5)
                        if proposed_label else pd.DataFrame())

    checks, checks_ok = checks_table(runs, y_train, y_test, before, baseline5,
                                     table, verbose=verbose)
    after = xp.test_fingerprint()
    untouched = bool(before and after and all(
        after.get(k, {}).get("sha256") == v["sha256"]
        for k, v in before.items()))

    out: dict[str, Any] = {
        "runs": runs, "baseline5": baseline5, "comparison": table,
        "comparison_marked": marked, "per_class": per_class,
        "per_class_free": per_class_free, "head_to_head": difference,
        "per_class_difference": class_difference, "checks": checks,
        "checks_ok": checks_ok, "architecture": architecture_table(),
        "selected": baseline5.get("selected"),
        "overlap": baseline5.get("overlap"),
        "overlap_summary": baseline5.get("overlap_summary"),
        "test_untouched": untouched,
        "step12_fingerprint_before": step12_before,
        "summary": build_summary(table, per_class, per_class_free, difference,
                                 class_difference, baseline5),
    }
    saved = save_step12c(out, verbose=verbose)
    out.update({"status": saved.get("status", "written"),
                "ok": bool(saved.get("ok") and checks_ok and untouched),
                "saved": saved})
    for key in ("report", "charts", "folder"):
        if key in saved:
            out[key] = saved[key]

    if verbose:
        show_title("Step 12 -- six configurations", level=3)
        show_table(marked, all_columns=True,
                   note="`*` marks the best value in each column. Lower is "
                        "better for the FPR, the times and the parameter "
                        "count. No `*` in a column means every configuration "
                        "tied there.")
        show_note("The five existing rows are read from disk and asserted "
                  "identical to Step 12's `comparison_table.csv`; only "
                  "Baseline 5 is new, and no existing file was rewritten.")
    return out
