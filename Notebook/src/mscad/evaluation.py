"""
src/mscad/evaluation.py -- STEP 11: evaluate on the untouched test set.

Input   C:\\MSCAD2022\\outputs\\step8\\   Z_test.csv, y_test.csv
        C:\\MSCAD2022\\outputs\\step5\\   X_train.csv / X_test.csv (the duplicate
                                        check), meta_test.csv (Orig_Row)
        C:\\MSCAD2022\\outputs\\step4\\   label_mapping.json
Output  C:\\MSCAD2022\\outputs\\step11\\

Reusable on purpose: Step 12's baselines are scored by calling `evaluate()` with
a different model, so a comparison is a comparison of models and not of
measurement code.

Why there are two test scores
-----------------------------
About half this dataset's rows share identical feature values once the
identifiers are gone.  A random split therefore puts feature-identical twins on
both sides, and a model can "recognise" a test row it has effectively already
seen.  `evaluate()` reports every headline metric twice:

    full            every test row
    duplicate-free  only test rows with NO feature-identical row in training

Nothing is retrained and nothing is dropped from the model's input -- the same
predictions are simply scored over two row sets.  The duplicate-free number is
the conservative estimate of generalisation and is the one to lead with.

What macro means here, and why it is the headline
-------------------------------------------------
Accuracy on this dataset is close to a measure of the majority class alone:
Brute_Force is ~84% of the rows, so predicting it for everything scores ~84%.
Macro-F1 and G-mean weight every class equally, so a model that ignores the
23-row classes cannot hide.  G-mean is the geometric mean of per-class recall,
which goes to zero the moment ANY class is never recalled -- the strictest of the
three, and the one that catches a model quietly abandoning a rare class.
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
from . import models as md  # noqa: E402
from . import splitting as sp  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = "Label_encoded"


class EvaluationError(Exception):
    """Raised when Step 11 cannot continue without guessing."""


# ===========================================================================
# 11.1  THE METRICS
# ===========================================================================
def geometric_mean_recall(y_true, y_pred, labels=None) -> float:
    """G-mean: the geometric mean of per-class recall.

    Computed directly rather than taken from a library, so it has no dependency
    and its definition is visible.  It is zero if ANY class has zero recall --
    which is the point: a model that abandons one class cannot hide behind a high
    average on the others.  `cross_check_gmean` compares it with
    imblearn's implementation when that package happens to be installed.
    """
    from sklearn.metrics import recall_score

    y_true = np.asarray(y_true)
    labels = np.unique(y_true) if labels is None else np.asarray(labels)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None,
                           zero_division=0)
    recalls = np.asarray(recalls, dtype="float64")
    if recalls.size == 0:
        return 0.0
    if np.any(recalls <= 0):
        return 0.0
    # in log space, so many small recalls cannot underflow to 0
    return float(np.exp(np.mean(np.log(recalls))))


def cross_check_gmean(y_true, y_pred) -> dict[str, Any]:
    """Compare our G-mean with imblearn's, when imblearn is installed."""
    ours = geometric_mean_recall(y_true, y_pred)
    try:
        from imblearn.metrics import geometric_mean_score
    except ImportError:
        return {"ours": ours, "imblearn": None, "agree": None,
                "note": "imbalanced-learn is not installed, so only the direct "
                        "computation is available (it is not a dependency of this "
                        "project)."}
    theirs = float(geometric_mean_score(np.asarray(y_true), np.asarray(y_pred),
                                        average="multiclass"))
    return {"ours": ours, "imblearn": theirs,
            "agree": bool(abs(ours - theirs) < 1e-9),
            "note": "both computations must agree"}


def roc_auc_ovr(y_true, probabilities: np.ndarray, n_classes: int):
    """One-vs-rest ROC-AUC, macro and weighted.  Classes absent from y are skipped.

    A class with no test rows has no ROC curve at all -- there are no positives --
    so it is excluded and named rather than being scored as 0, which would drag
    the macro average down for a reason that has nothing to do with the model.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    present = sorted({int(c) for c in np.unique(y_true)})
    absent = [c for c in range(n_classes) if c not in present]
    usable = [c for c in present if 0 < int((y_true == c).sum()) < len(y_true)]
    degenerate = [c for c in present if c not in usable]

    if len(usable) < 2:
        return {"macro": None, "weighted": None, "skipped_absent": absent,
                "skipped_degenerate": degenerate,
                "note": "fewer than 2 usable classes, so a one-vs-rest AUC is "
                        "undefined"}

    columns = probabilities[:, usable]
    # renormalise over the usable classes, so each row is still a distribution
    totals = columns.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    columns = columns / totals
    mapping = {c: i for i, c in enumerate(usable)}
    mapped = np.array([mapping[int(v)] for v in y_true])

    if len(usable) == 2:
        # sklearn's multi_class="ovr" path wants an (n, k) score matrix for k > 2
        # and a 1-D score for the positive class when k == 2; handing it (n, 2)
        # raises "y should be a 1d array".  With two classes macro and weighted
        # one-vs-rest are the same number anyway.
        auc = float(roc_auc_score(mapped, columns[:, 1]))
        macro = weighted = auc
    else:
        macro = float(roc_auc_score(mapped, columns, multi_class="ovr",
                                    average="macro",
                                    labels=list(range(len(usable)))))
        weighted = float(roc_auc_score(mapped, columns, multi_class="ovr",
                                       average="weighted",
                                       labels=list(range(len(usable)))))
    return {"macro": macro, "weighted": weighted, "skipped_absent": absent,
            "skipped_degenerate": degenerate,
            "note": "one-vs-rest over the %d classes present in this row set"
                    % len(usable)}


def headline_metrics(y_true, y_pred, probabilities: np.ndarray | None = None,
                     n_classes: int | None = None) -> dict[str, Any]:
    """Accuracy, macro/weighted precision-recall-F1, G-mean, and ROC-AUC."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = (int(max(y_true.max(), y_pred.max())) + 1 if n_classes is None
                 else n_classes)

    macro = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                            zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                               zero_division=0)
    out: dict[str, Any] = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro[0]), "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "g_mean": geometric_mean_recall(y_true, y_pred),
    }
    if probabilities is not None and len(probabilities):
        out["roc_auc"] = roc_auc_ovr(y_true, probabilities, n_classes)
    return out


def per_class_table(y_true, y_pred, label_names: dict[int, str] | None = None,
                    n_classes: int | None = None) -> pd.DataFrame:
    """Precision, recall, F1 and support for every class."""
    from sklearn.metrics import precision_recall_fscore_support

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = (int(max(y_true.max(), y_pred.max())) + 1 if n_classes is None
                 else n_classes)
    labels = list(range(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    names = label_names or {}
    table = pd.DataFrame({
        "Label": [names.get(c, str(c)) for c in labels],
        "encoded": labels,
        "precision": np.round(precision, 6),
        "recall": np.round(recall, 6),
        "f1": np.round(f1, 6),
        "support": support.astype("int64"),
    })
    return table.sort_values("support", ascending=False).reset_index(drop=True)


def confusion_tables(y_true, y_pred, label_names: dict[int, str] | None = None,
                     n_classes: int | None = None):
    """(counts, row-normalised).  Rows are truth, columns are predictions."""
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = (int(max(y_true.max(), y_pred.max())) + 1 if n_classes is None
                 else n_classes)
    labels = list(range(n_classes))
    names = [(label_names or {}).get(c, str(c)) for c in labels]

    counts = confusion_matrix(y_true, y_pred, labels=labels)
    frame = pd.DataFrame(counts, index=names, columns=names)
    frame.index.name = "true \\ predicted"

    totals = counts.sum(axis=1, keepdims=True).astype("float64")
    totals[totals == 0] = 1.0            # a class with no test rows stays all-zero
    normalised = pd.DataFrame(np.round(counts / totals, 6), index=names,
                              columns=names)
    normalised.index.name = "true \\ predicted"
    return frame, normalised


# ===========================================================================
# 11.2  THE BINARY VIEW
# ===========================================================================
def binary_view(y_true, probabilities: np.ndarray, benign_codes: Sequence[int],
                n_classes: int | None = None) -> dict[str, Any]:
    """Benign vs attack, derived from the multiclass probabilities.

    P(attack) = 1 - sum of the benign classes' probabilities, which is the right
    way round: summing the attack classes instead would drift from 1 by floating
    point and, worse, would quietly ignore a class that is neither.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)

    y_true = np.asarray(y_true)
    benign = {int(c) for c in benign_codes}
    truth = np.array([0 if int(v) in benign else 1 for v in y_true])

    usable = [c for c in benign if c < probabilities.shape[1]]
    p_benign = probabilities[:, usable].sum(axis=1) if usable else np.zeros(len(y_true))
    p_attack = 1.0 - p_benign
    predicted = (p_attack >= 0.5).astype("int64")

    precision, recall, f1, _support = precision_recall_fscore_support(
        truth, predicted, average="binary", pos_label=1, zero_division=0)

    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]),
                      int(matrix[1, 0]), int(matrix[1, 1]))
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0

    auc = None
    if len(np.unique(truth)) == 2:
        auc = float(roc_auc_score(truth, p_attack))

    return {"rows": int(len(truth)), "benign_classes": sorted(benign),
            "accuracy": float(accuracy_score(truth, predicted)),
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "roc_auc": auc,
            "false_positive_rate": fpr,
            "true_negatives": tn, "false_positives": fp,
            "false_negatives": fn, "true_positives": tp,
            "note": "attack probability = 1 - sum(benign class probabilities); "
                    "a benign flow predicted as an attack is a false positive"}


# ===========================================================================
# 11.3  DUPLICATES BETWEEN TRAIN AND TEST
# ===========================================================================
def duplicate_mask(X_train: pd.DataFrame, X_test: pd.DataFrame) -> np.ndarray:
    """True where a test row's exact feature vector also appears in training.

    Compared on the raw float64 bytes, so a match is exact and not approximate.
    Uses the Step 5 feature matrices, which are the full feature vectors -- two
    flows that differ only in a feature the autoencoder discarded are NOT twins.
    """
    if list(X_train.columns) != list(X_test.columns):
        raise EvaluationError(
            "the duplicate check needs the same columns in the same order; got "
            "%d train columns and %d test columns"
            % (X_train.shape[1], X_test.shape[1]))
    keys_train = set(sp._row_keys(X_train).tolist())
    keys_test = sp._row_keys(X_test)
    return np.array([k in keys_train for k in keys_test], dtype=bool)


def duplicate_comparison(y_true, y_pred, probabilities: np.ndarray,
                         mask: np.ndarray, label_names: dict[int, str] | None = None,
                         n_classes: int | None = None):
    """Headline metrics on the full test set and on the duplicate-free part.

    Same model, same predictions -- only the row set differs.  Nothing is
    retrained and nothing is dropped from the model's input.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    clean = ~np.asarray(mask, dtype=bool)

    full = headline_metrics(y_true, y_pred, probabilities, n_classes)
    if clean.sum() == 0:
        free = {"rows": 0, "note": "every test row has a feature-identical twin in "
                                   "training, so there is no duplicate-free subset"}
    else:
        free = headline_metrics(y_true[clean], y_pred[clean],
                                probabilities[clean], n_classes)

    rows = []
    for metric in ("rows", "accuracy", "f1_macro", "g_mean", "recall_macro",
                   "f1_weighted"):
        rows.append({
            "metric": metric,
            "full test set": full.get(metric),
            "duplicate-free": free.get(metric),
        })
    summary = pd.DataFrame(rows)

    # per class: how many test rows, and how many of them are duplicates
    names = label_names or {}
    per_class = []
    for code in sorted({int(c) for c in np.unique(y_true)}):
        selected = y_true == code
        per_class.append({
            "Label": names.get(code, str(code)), "encoded": code,
            "test rows": int(selected.sum()),
            "with a train duplicate": int(np.asarray(mask)[selected].sum()),
            "duplicate-free rows": int(clean[selected].sum()),
            "% duplicated": round(100.0 * float(np.asarray(mask)[selected].sum())
                                  / max(int(selected.sum()), 1), 3),
        })
    counts = pd.DataFrame(per_class)
    return summary, counts, full, free


# ===========================================================================
# 11.4  INFERENCE TIME
# ===========================================================================
def inference_time(model, X: np.ndarray, batch_size: int | None = None,
                   runs: int = 3) -> dict[str, Any]:
    """Wall-clock prediction time on this machine, after a warm-up batch.

    The warm-up matters: the first call builds the graph and allocates buffers,
    so timing it would measure compilation rather than inference.
    """
    batch_size = config.CLF_BATCH_SIZE if batch_size is None else batch_size
    warmup = X[:min(len(X), batch_size)]
    model.predict(warmup, batch_size=batch_size, verbose=0)      # discarded

    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        model.predict(X, batch_size=batch_size, verbose=0)
        timings.append(time.perf_counter() - started)

    seconds = float(np.mean(timings))
    rows = max(len(X), 1)
    seconds_per_row = seconds / rows
    return {"rows": int(len(X)), "runs": runs, "batch_size": batch_size,
            "total_seconds": seconds,
            "std_seconds": float(np.std(timings)),
            # seconds per row -> milliseconds per 1,000 rows
            "ms_per_1000_flows": float(seconds_per_row * 1000.0 * 1000.0),
            "note": "CPU wall clock, averaged over %d runs after one warm-up batch"
                    % runs}


# ===========================================================================
# 11.5  THE CONFUSION-MATRIX CHART
# ===========================================================================
# Row-normalised recall is a MAGNITUDE in [0, 1], so one hue light -> dark.
_SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
             "#0d366b"]


def plot_confusion(normalised: pd.DataFrame, counts: pd.DataFrame, path,
                   verbose: bool = True):
    """Row-normalised confusion matrix, with the raw count in each cell."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("  [SKIP]  matplotlib is not installed, so no chart was drawn.")
        return None
    from matplotlib.colors import LinearSegmentedColormap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(normalised.columns)
    n = len(names)
    values = normalised.to_numpy(dtype="float64")
    raw = counts.to_numpy()

    cmap = LinearSegmentedColormap.from_list("mscad_blue", _SEQ_RAMP)
    size = max(6.5, 0.85 * n + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.86), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0,
                      interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9,
                       color=dl._INK_SOFT)
    ax.set_yticklabels(names, fontsize=9, color=dl._INK_SOFT)
    ax.set_xlabel("predicted", color=dl._INK_SOFT, fontsize=10)
    ax.set_ylabel("true", color=dl._INK_SOFT, fontsize=10)
    ax.set_title("Confusion matrix -- each row is one true class, normalised to 1",
                 color=dl._INK, fontsize=12, pad=14)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)

    # every cell carries its share and its raw count: a log-free way to read both
    for i in range(n):
        for j in range(n):
            share = values[i, j]
            if share == 0 and raw[i, j] == 0:
                continue
            ax.annotate("%.2f\n(%s)" % (share, f"{int(raw[i, j]):,}"), (j, i),
                        ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if share > 0.55 else dl._INK)

    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("share of that true class", color=dl._INK_SOFT, fontsize=9)
    bar.ax.tick_params(colors=dl._INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


# ===========================================================================
# 11.6  THE ONE ENTRY POINT -- Step 12's baselines call this too
# ===========================================================================
def evaluate(model, X_test: np.ndarray, y_test, label_mapping=None,
             meta_test: pd.DataFrame | None = None,
             duplicate_flags: np.ndarray | None = None,
             n_classes: int | None = None, model_name: str = "proposed CNN-LSTM",
             training_seconds: float | None = None,
             verbose: bool = True) -> dict[str, Any]:
    """Score a trained model on the test set.  Nothing here changes the model.

    `X_test` is already arranged for the model (see models.arrange_inputs), and
    `duplicate_flags` is the per-row mask from `duplicate_mask`.
    """
    y_true = np.asarray(y_test[TARGET_COLUMN] if isinstance(y_test, pd.DataFrame)
                        else y_test)
    names = label_names_from_mapping(label_mapping)
    n_classes = (int(max(y_true.max() + 1, len(names) or 0))
                 if n_classes is None else n_classes)

    probabilities = md.predict_proba(model, X_test)
    y_pred = np.argmax(probabilities, axis=1)

    full = headline_metrics(y_true, y_pred, probabilities, n_classes)
    per_class = per_class_table(y_true, y_pred, names, n_classes)
    counts, normalised = confusion_tables(y_true, y_pred, names, n_classes)
    benign_codes = benign_codes_from(y_test, names)
    binary = binary_view(y_true, probabilities, benign_codes, n_classes)
    gmean_check = cross_check_gmean(y_true, y_pred)

    if duplicate_flags is None:
        duplicate_flags = np.zeros(len(y_true), dtype=bool)
    duplicate_flags = np.asarray(duplicate_flags, dtype=bool)
    comparison, duplicate_counts, full_again, free = duplicate_comparison(
        y_true, y_pred, probabilities, duplicate_flags, names, n_classes)

    timing = inference_time(model, X_test)

    out: dict[str, Any] = {
        "model_name": model_name,
        "full": full, "duplicate_free": free,
        "per_class": per_class, "confusion_counts": counts,
        "confusion_normalised": normalised, "binary": binary,
        "duplicate_comparison": comparison, "duplicate_counts": duplicate_counts,
        "gmean_cross_check": gmean_check, "inference": timing,
        "training_seconds": training_seconds,
        "probabilities": probabilities, "y_pred": y_pred, "y_true": y_true,
        "duplicate_flags": duplicate_flags, "label_names": names,
        "n_classes": n_classes,
    }

    if verbose:
        show_title("Step 11 -- %s on the untouched test set" % model_name, level=3)
        show_table(kv_table([
            ("test rows", f"{full['rows']:,}"),
            ("accuracy", "%.4f" % full["accuracy"]),
            ("macro-F1", "%.4f" % full["f1_macro"]),
            ("weighted F1", "%.4f" % full["f1_weighted"]),
            ("macro precision / recall", "%.4f / %.4f"
             % (full["precision_macro"], full["recall_macro"])),
            ("G-mean", "%.4f" % full["g_mean"]),
            ("ROC-AUC macro / weighted",
             "%s / %s" % (_fmt(full.get("roc_auc", {}).get("macro")),
                          _fmt(full.get("roc_auc", {}).get("weighted")))),
            ("training time", "%.1f s" % training_seconds
             if training_seconds else "-"),
            ("inference", "%.3f s for %s rows (%.1f ms per 1,000)"
             % (timing["total_seconds"], f"{timing['rows']:,}",
                timing["ms_per_1000_flows"])),
        ]), note="Accuracy is close to a measure of the majority class on this "
                 "dataset -- Brute_Force is ~84% of the rows. Macro-F1 and G-mean "
                 "are the numbers to lead with.")
        absent = full.get("roc_auc", {}).get("skipped_absent") or []
        if absent:
            show_note("ROC-AUC skipped %d class(es) with no test rows: %s."
                      % (len(absent), ", ".join(names.get(c, str(c)) for c in absent)))
    return out


def _fmt(value) -> str:
    return "-" if value is None else "%.4f" % value


def label_names_from_mapping(label_mapping) -> dict[int, str]:
    """{encoded: name} from Step 4's label_mapping.json, whichever way round."""
    if not label_mapping:
        return {}
    names: dict[int, str] = {}
    for key, value in dict(label_mapping).items():
        try:                                     # {"Normal": 3}
            names[int(value)] = str(key)
        except (TypeError, ValueError):          # {"3": "Normal"}
            try:
                names[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    return names


def benign_codes_from(y_test, names: dict[int, str]) -> list[int]:
    """The encoded labels whose Label_binary is 0 -- read from the data itself."""
    if isinstance(y_test, pd.DataFrame) and "Label_binary" in y_test.columns:
        benign = y_test.loc[y_test["Label_binary"] == 0, TARGET_COLUMN]
        return sorted({int(c) for c in benign.unique()})
    # fall back to the configured benign label names
    wanted = {str(n).lower() for n in config.BENIGN_LABELS}
    return sorted(code for code, name in names.items() if str(name).lower() in wanted)


# ===========================================================================
# 11.7  LOAD, SAVE, RUN
# ===========================================================================
def load_evaluation_inputs(verbose: bool = True) -> dict[str, Any]:
    """Z_test / y_test, the Step 5 matrices for the duplicate check, meta, labels."""
    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder8 is None:
        raise EvaluationError("the Step 8 output folder is not usable here -- %s"
                              % config.why_unusable(config.STEP8_OUTPUT_DIR))
    names8 = config.STEP8_FILES
    missing = [names8[k] for k in ("Z_test", "y_test")
               if not (folder8 / names8[k]).exists()]
    if missing:
        raise EvaluationError("Step 11 needs %s in %s. Run Step 8 first."
                              % (missing, folder8))

    out: dict[str, Any] = {
        "Z_test": pd.read_csv(folder8 / names8["Z_test"], low_memory=False),
        "y_test": pd.read_csv(folder8 / names8["y_test"], low_memory=False),
    }

    folder5 = config.usable_output_dir(config.STEP5_OUTPUT_DIR)
    names5 = config.STEP5_FILES
    for key in ("X_train", "X_test", "meta_test", "meta_train"):
        path = (folder5 / names5[key]) if folder5 else None
        out[key] = (pd.read_csv(path, low_memory=False)
                    if path is not None and path.exists() else None)

    out["label_mapping"] = {}
    folder4 = config.usable_output_dir(config.STEP4_OUTPUT_DIR)
    if folder4 is not None:
        path = folder4 / config.STEP4_FILES["label_mapping"]
        if path.exists():
            out["label_mapping"] = json.loads(path.read_text(encoding="utf-8"))
    return out


def build_predictions_frame(result: dict[str, Any],
                            meta_test: pd.DataFrame | None,
                            row_index: np.ndarray | None = None) -> pd.DataFrame:
    """One row per scored sample: identity, truth, prediction, probabilities."""
    names = result["label_names"]
    probabilities = result["probabilities"]
    n = len(result["y_true"])

    frame = pd.DataFrame()
    if meta_test is not None and "Orig_Row" in meta_test.columns:
        source = meta_test["Orig_Row"].to_numpy()
        frame["Orig_Row"] = (source[row_index] if row_index is not None
                             else source[:n] if len(source) >= n else np.arange(n))
    else:
        frame["Orig_Row"] = np.arange(n)
    if meta_test is not None and "Source_Capture" in meta_test.columns:
        source = meta_test["Source_Capture"].to_numpy()
        frame["Source_Capture"] = (source[row_index] if row_index is not None
                                   else source[:n])

    frame["y_true"] = result["y_true"]
    frame["y_pred"] = result["y_pred"]
    frame["y_true_label"] = [names.get(int(c), str(c)) for c in result["y_true"]]
    frame["y_pred_label"] = [names.get(int(c), str(c)) for c in result["y_pred"]]
    frame["correct"] = frame["y_true"] == frame["y_pred"]
    frame["has_train_duplicate"] = result["duplicate_flags"]
    for code in range(probabilities.shape[1]):
        frame["p_%s" % names.get(code, code)] = np.round(probabilities[:, code], 6)
    return frame


def aggregate_seeds(runs: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """mean +- std of every headline metric across seeds."""
    rows = []
    keys = ("accuracy", "f1_macro", "f1_weighted", "recall_macro",
            "precision_macro", "g_mean")
    for scope in ("full", "duplicate_free"):
        for key in keys:
            values = [float(r[scope][key]) for r in runs
                      if scope in r and key in r[scope]]
            if not values:
                continue
            rows.append({"scope": scope, "metric": key, "seeds": len(values),
                         "mean": round(float(np.mean(values)), 6),
                         "std": round(float(np.std(values, ddof=1)), 6)
                         if len(values) > 1 else 0.0,
                         "min": round(float(np.min(values)), 6),
                         "max": round(float(np.max(values)), 6)})
    return pd.DataFrame(rows)


def save_step11(result: dict[str, Any], predictions: pd.DataFrame,
                seeds_table: pd.DataFrame | None = None,
                verbose: bool = True) -> dict[str, Any]:
    """Write every Step 11 output, then re-read the key ones."""
    out: dict[str, Any] = {"folder": str(config.STEP11_OUTPUT_DIR), "files": {}}
    folder = prepare_output_dir(config.STEP11_OUTPUT_DIR, "the Step 11 outputs",
                                verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP11_FILES
    result["per_class"].to_csv(folder / names["per_class"], index=False)
    predictions.to_csv(folder / names["predictions"], index=False)
    result["confusion_counts"].to_csv(folder / names["confusion_counts"])
    result["confusion_normalised"].to_csv(folder / names["confusion_normalised"])
    result["duplicate_comparison"].to_csv(folder / names["duplicate_split"],
                                          index=False)
    pd.DataFrame([result["binary"]]).to_csv(folder / names["binary"], index=False)

    report = pd.DataFrame([
        {"step": "11.1 predict", "action": "softmax of the logits, argmax",
         "detail": "softmax applied exactly once, here"},
        {"step": "11.2 metrics", "action": "accuracy, macro/weighted P-R-F1, G-mean",
         "detail": "macro-F1 %.4f, G-mean %.4f on %s rows"
                   % (result["full"]["f1_macro"], result["full"]["g_mean"],
                      f"{result['full']['rows']:,}")},
        {"step": "11.3 duplicates", "action": "scored again on duplicate-free rows",
         "detail": "%s of %s test rows have a feature-identical twin in train"
                   % (f"{int(result['duplicate_flags'].sum()):,}",
                      f"{len(result['duplicate_flags']):,}")},
        {"step": "11.4 timing", "action": "CPU inference after a warm-up batch",
         "detail": "%.1f ms per 1,000 flows"
                   % result["inference"]["ms_per_1000_flows"]},
        {"step": "11.x not done", "action": "the test set was not resampled",
         "detail": "no reweighting, no rebalancing -- scored exactly as Step 8 "
                   "produced it"},
    ])
    report.to_csv(folder / names["report"], index=False)

    if seeds_table is not None and len(seeds_table):
        seeds_table.to_csv(folder / "seed_variation.csv", index=False)
        out["files"]["seed_variation"] = str(folder / "seed_variation.csv")

    metrics = {
        "model": result["model_name"],
        "full": {k: v for k, v in result["full"].items()},
        "duplicate_free": {k: v for k, v in result["duplicate_free"].items()},
        "binary": result["binary"],
        "g_mean_cross_check": result["gmean_cross_check"],
        "inference": result["inference"],
        "training_seconds": result["training_seconds"],
        "n_classes": result["n_classes"],
        "label_names": {str(k): v for k, v in result["label_names"].items()},
        "duplicate_rows": int(result["duplicate_flags"].sum()),
        "test_rows": int(len(result["duplicate_flags"])),
        "notes": [
            "The duplicate-free score is the conservative estimate: it excludes "
            "test rows whose exact feature vector also occurs in training.",
            "The rarest classes have only a handful of test rows, so their "
            "per-class scores move a lot on a single row changing hands. Set "
            "N_SEEDS to 3 or 5 and quote mean +- std for them.",
            "The test set was never resampled, reweighted or used for training, "
            "early stopping or tuning.",
        ],
    }
    (folder / names["metrics"]).write_text(json.dumps(metrics, indent=2,
                                                      default=float),
                                           encoding="utf-8")

    chart = plot_confusion(result["confusion_normalised"],
                           result["confusion_counts"],
                           folder / names["confusion_chart"], verbose=verbose)
    if chart is not None:
        out["files"]["confusion_chart"] = str(chart)

    for key in ("metrics", "per_class", "predictions", "confusion_counts",
                "confusion_normalised", "duplicate_split", "binary", "report"):
        out["files"][key] = str(folder / names[key])
    out.update({"status": "written", "ok": True})
    return out


def step11_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP11_OUTPUT_DIR)
    if folder is None:
        return False
    return all((folder / config.STEP11_FILES[k]).exists()
               for k in ("metrics", "per_class", "predictions"))


def load_step11(verbose: bool = True) -> dict[str, Any]:
    folder = config.usable_output_dir(config.STEP11_OUTPUT_DIR)
    if folder is None:
        raise EvaluationError("the Step 11 folder is not usable here -- %s"
                              % config.why_unusable(config.STEP11_OUTPUT_DIR))
    names = config.STEP11_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    out["metrics"] = json.loads((folder / names["metrics"]).read_text(encoding="utf-8"))
    for key in ("per_class", "predictions", "duplicate_split", "binary", "report"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    for key in ("confusion_counts", "confusion_normalised"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, index_col=0, low_memory=False)
    if verbose:
        metrics = out["metrics"]
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP11 = False)"),
            ("model", metrics.get("model")),
            ("test rows", f"{metrics.get('test_rows', 0):,}"),
            ("macro-F1 (full)", "%.4f" % metrics["full"]["f1_macro"]),
            ("macro-F1 (duplicate-free)",
             _fmt(metrics.get("duplicate_free", {}).get("f1_macro"))),
        ]), note="Set `OVERWRITE_STEP11 = True` in config.py to re-evaluate.")
    return out


def run_step11(model=None, training_seconds: float | None = None,
               overwrite: bool | None = None, model_name: str = "proposed CNN-LSTM",
               verbose: bool = True) -> dict[str, Any]:
    """STEP 11 end to end: load the test set, score it, save everything."""
    overwrite = config.OVERWRITE_STEP11 if overwrite is None else overwrite
    if not overwrite and step11_outputs_exist():
        return load_step11(verbose=verbose)

    loaded = load_evaluation_inputs(verbose=verbose)
    Z_test, y_test = loaded["Z_test"], loaded["y_test"]

    if model is None:
        step10 = md.load_step10(verbose=False)
        model = step10["model"]
        training_seconds = training_seconds or step10["info"].get("seconds")

    X_test, labels_test, row_index = md.arrange_inputs(
        Z_test, y_test, loaded.get("meta_test"), verbose=False)
    y_scored = (y_test.iloc[row_index].reset_index(drop=True)
                if len(row_index) != len(y_test) or not np.array_equal(
                    row_index, np.arange(len(y_test)))
                else y_test)

    flags = None
    if loaded["X_train"] is not None and loaded["X_test"] is not None:
        mask = duplicate_mask(loaded["X_train"], loaded["X_test"])
        flags = mask[row_index] if len(row_index) != len(mask) else mask
    elif verbose:
        show_note("**[WARN]** the Step 5 feature matrices are not on disk, so the "
                  "duplicate-free score could not be computed. Re-run Step 5.")

    result = evaluate(model, X_test, y_scored, loaded["label_mapping"],
                      meta_test=loaded.get("meta_test"), duplicate_flags=flags,
                      model_name=model_name, training_seconds=training_seconds,
                      verbose=verbose)

    predictions = build_predictions_frame(result, loaded.get("meta_test"), row_index)
    saved = save_step11(result, predictions, verbose=verbose)

    result.update({"predictions": predictions, "saved": saved,
                   "status": saved.get("status", "written"),
                   "ok": bool(saved.get("ok"))})
    return result
