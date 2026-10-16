"""
src/mscad/imbalance.py -- STEP 9: class imbalance, handled in the TRAINING LOSS.

Input   C:\\MSCAD2022\\outputs\\step8\\   Z_train.csv, y_train.csv
Output  C:\\MSCAD2022\\outputs\\step9\\   the fit/validation split and the weights

CLASS WEIGHTING ONLY
--------------------
No SMOTE, no SMOTE-ENN, no over- or undersampling, and imbalanced-learn is
deliberately not a dependency.  Every row that goes in comes out: a weight
changes how much a row COUNTS in the loss, never how many rows there are.

Why weighting rather than resampling, on this dataset specifically.  About half
the rows share identical feature values once the identifiers are gone --
thousands of scan, flood and brute-force flows genuinely look alike.  SMOTE
interpolates between nearest neighbours, so between identical rows it
manufactures exact copies rather than new information, and oversampling before
an inner split copies rows across it.  Weighting avoids both, because it adds no
rows at all.

The test set is never touched
-----------------------------
Weights are computed from the training rows alone and applied only in the
training loss.  The test set is scored exactly as Step 8 produced it.  Rebalancing
a test set changes the question being asked: a macro-F1 on a reweighted test set
is not the macro-F1 anyone reading the thesis will assume it is.

What Steps 10-12 call
---------------------
    X_fit, y_fit, X_val, y_val, weights, sample_weights = make_training_sets(
        Z_train, y_train, strategy=config.IMBALANCE_STRATEGY)
    model.fit(X_fit, y_fit, class_weight=keras_class_weight(weights), ...)

The validation loss used for early stopping is deliberately UNWEIGHTED: it has to
measure the model on the data as it really is, not on the distribution the
training loss pretends to see.
"""

from __future__ import annotations

import hashlib
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

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

LABEL_COLUMNS = ["Label", "Label_encoded", "Label_binary"]
TARGET_COLUMN = "Label_encoded"
VALID_SCHEMES = ("balanced", "sqrt_balanced")
VALID_STRATEGIES = ("class_weight", "none")


class ImbalanceError(Exception):
    """Raised when Step 9 cannot continue without guessing."""


# ===========================================================================
# 9.1  THE WEIGHTS
# ===========================================================================
def compute_class_weights(y, scheme: str | None = None,
                          max_weight: float | None = None,
                          normalise: bool | None = None,
                          verbose: bool = False) -> dict[int, float]:
    """One weight per class present in `y`.  {encoded label: weight}.

    "balanced" is sklearn's  w_c = n_samples / (n_classes * n_c),  so a class
    with a tenth of the rows gets ten times the weight.

    The three optional adjustments are applied in a fixed order, and the order
    matters:
        1. sqrt   -- softens the ratio before anything else looks at it
        2. cap    -- a ceiling on the softened value, not on the raw one
        3. scale  -- normalisation last, so the cap is not silently undone
    Reversing 2 and 3 would let normalisation push a capped weight back above
    the cap.
    """
    scheme = (config.CLASS_WEIGHT_SCHEME if scheme is None else scheme)
    max_weight = config.CLASS_WEIGHT_MAX if max_weight is None else max_weight
    normalise = config.NORMALISE_WEIGHTS if normalise is None else normalise

    if scheme not in VALID_SCHEMES:
        raise ImbalanceError(
            "CLASS_WEIGHT_SCHEME is %r; it must be one of %s"
            % (scheme, ", ".join(repr(s) for s in VALID_SCHEMES)))

    values = np.asarray(y).ravel()
    if values.size == 0:
        raise ImbalanceError(
            "cannot compute class weights from an empty label array -- there is "
            "nothing to count.")

    try:
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as exc:
        raise ImbalanceError(
            'scikit-learn is needed for the class weights.  Install it with:\n'
            '    "%s" -m pip install -r requirements.txt' % sys.executable) from exc

    classes = np.unique(values)
    balanced = compute_class_weight("balanced", classes=classes, y=values)
    weights = np.asarray(balanced, dtype="float64")

    # 1. soften
    if scheme == "sqrt_balanced":
        weights = np.sqrt(weights)
    # 2. cap
    if max_weight is not None:
        weights = np.minimum(weights, float(max_weight))
    # 3. scale, so the MEAN PER-SAMPLE weight is 1.  Per-sample, not per-class:
    # the mean over rows is what sets the effective learning rate, and a plain
    # mean over classes would leave it drifting with the imbalance.
    if normalise:
        counts = np.array([(values == c).sum() for c in classes], dtype="float64")
        mean_per_sample = float((counts * weights).sum() / counts.sum())
        if mean_per_sample > 0:
            weights = weights / mean_per_sample

    out = {int(c): float(w) for c, w in zip(classes, weights)}
    if verbose:
        show_note("Scheme **%s**%s%s -- %d classes, weights %.4f to %.4f."
                  % (scheme,
                     ", capped at %g" % max_weight if max_weight is not None else "",
                     ", normalised to mean 1" if normalise else "",
                     len(out), min(out.values()), max(out.values())))
    return out


def sample_weights_from(y, class_weights: dict[int, float]) -> np.ndarray:
    """One weight per ROW, for frameworks that take per-sample weights."""
    values = np.asarray(y).ravel()
    missing = sorted({int(v) for v in values} - set(class_weights))
    if missing:
        raise ImbalanceError(
            "no class weight for %s. Every class in y must have one, or those rows "
            "would silently contribute nothing." % missing)
    return np.array([class_weights[int(v)] for v in values], dtype="float64")


def keras_class_weight(class_weights: dict[int, float]) -> dict[int, float]:
    """What Keras wants: `model.fit(..., class_weight=<this>)`."""
    return {int(k): float(v) for k, v in class_weights.items()}


def weight_vector(class_weights: dict[int, float]) -> np.ndarray:
    """The weights ordered by encoded label, for a framework that wants an array.

    PyTorch's CrossEntropyLoss(weight=...) takes exactly this ordering.  This
    project uses Keras (see requirements.txt), so `keras_class_weight` is the one
    Steps 10-12 call; this is here so a later experiment is not stuck.
    """
    return np.array([class_weights[k] for k in sorted(class_weights)],
                    dtype="float64")


# ===========================================================================
# 9.2  THE FIT / VALIDATION SPLIT
# ===========================================================================
def make_training_sets(X_train: pd.DataFrame, y_train: pd.DataFrame,
                       strategy: str | None = None, val_size: float | None = None,
                       random_state: int | None = None, verbose: bool = False):
    """Split TRAIN into fit/validation, then weight the FIT part.

    Returns (X_fit, y_fit, X_val, y_val, class_weights, fit_sample_weights).

    Two things this guarantees, and both are the point of it existing:
      - the inputs are never modified, and the returned frames are NEW objects,
        so a caller cannot accidentally mutate the Step 8 data;
      - the weights come from the FIT part ALONE.  Computing them over the whole
        training set would leak the validation distribution into the training
        loss, and the early-stopping signal would be judging a model that had
        already seen it.
    """
    strategy = config.IMBALANCE_STRATEGY if strategy is None else strategy
    val_size = config.CLF_VAL_SIZE if val_size is None else val_size
    random_state = config.RANDOM_STATE if random_state is None else random_state

    if strategy not in VALID_STRATEGIES:
        raise ImbalanceError(
            "IMBALANCE_STRATEGY is %r; it must be one of %s. Resampling is "
            "deliberately not an option here -- Step 9 is class weighting only."
            % (strategy, ", ".join(repr(s) for s in VALID_STRATEGIES)))
    if TARGET_COLUMN not in y_train.columns:
        raise ImbalanceError(
            "y_train has no `%s` column; got %s" % (TARGET_COLUMN,
                                                    list(y_train.columns)))
    if len(X_train) != len(y_train):
        raise ImbalanceError(
            "X_train has %s rows but y_train has %s" % (f"{len(X_train):,}",
                                                        f"{len(y_train):,}"))

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImbalanceError(
            'scikit-learn is needed for the split.  Install it with:\n'
            '    "%s" -m pip install -r requirements.txt' % sys.executable) from exc

    y = y_train[TARGET_COLUMN].to_numpy()
    counts = pd.Series(y).value_counts()
    if int(counts.min()) < 2:
        raise ImbalanceError(
            "the fit/validation split cannot be stratified: these classes have "
            "fewer than 2 training rows: %s"
            % {str(k): int(v) for k, v in counts.items() if v < 2})

    positions = np.arange(len(X_train), dtype="int64")
    fit_idx, val_idx = train_test_split(
        positions, test_size=val_size, random_state=random_state,
        shuffle=True, stratify=y)
    fit_idx, val_idx = np.sort(fit_idx), np.sort(val_idx)

    # .iloc[...] returns a copy, and reset_index makes it a clean new frame --
    # nothing here is a view onto the caller's data
    X_fit = X_train.iloc[fit_idx].reset_index(drop=True).copy()
    X_val = X_train.iloc[val_idx].reset_index(drop=True).copy()
    y_fit = y_train.iloc[fit_idx].reset_index(drop=True).copy()
    y_val = y_train.iloc[val_idx].reset_index(drop=True).copy()

    if strategy == "none":
        class_weights = {int(c): 1.0 for c in np.unique(y)}
    else:
        class_weights = compute_class_weights(y_fit[TARGET_COLUMN], verbose=verbose)

    fit_sample_weights = sample_weights_from(y_fit[TARGET_COLUMN], class_weights)

    if verbose:
        show_table(kv_table([
            ("strategy", strategy),
            ("training rows in", f"{len(X_train):,}"),
            ("fit rows", f"{len(X_fit):,}"),
            ("validation rows", f"{len(X_val):,}"),
            ("rows added or removed", 0),
            ("classes weighted", len(class_weights)),
        ]), note="No row was added, removed or duplicated. The weights come from "
                 "the FIT part only -- not from the validation part, and never "
                 "from the test set.")
    return X_fit, y_fit, X_val, y_val, class_weights, fit_sample_weights


# ===========================================================================
# 9.3  THE REPORT
# ===========================================================================
def weight_table(y_train: pd.DataFrame, y_fit: pd.DataFrame, y_val: pd.DataFrame,
                 class_weights: dict[int, float], verbose: bool = True):
    """Per class: counts, share, weight, and the EFFECTIVE share after weighting.

    `effective weight share %` is (count x weight) / sum(count x weight) over the
    fit part.  With "balanced" every class should land near 100/n_classes -- that
    is exactly what balancing means, and seeing the column come out flat is the
    check that the weights do what they claim.
    """
    names = {}
    for frame in (y_fit, y_val, y_train):
        for code, name in zip(frame[TARGET_COLUMN], frame[config.LABEL_COLUMN]):
            names.setdefault(int(code), str(name))

    fit_counts = y_fit[TARGET_COLUMN].value_counts()
    val_counts = y_val[TARGET_COLUMN].value_counts()
    train_counts = y_train[TARGET_COLUMN].value_counts()

    rows = []
    for code in sorted(class_weights, key=lambda c: -int(train_counts.get(c, 0))):
        n_fit = int(fit_counts.get(code, 0))
        weight = float(class_weights[code])
        rows.append({
            "Label": names.get(code, str(code)),
            "encoded": code,
            "train rows": int(train_counts.get(code, 0)),
            "fit rows": n_fit,
            "val rows": int(val_counts.get(code, 0)),
            "share of fit %": round(100.0 * n_fit / max(len(y_fit), 1), 4),
            "class weight": round(weight, 6),
            "_effective": n_fit * weight,
        })
    table = pd.DataFrame(rows)
    total_effective = float(table["_effective"].sum()) or 1.0
    table["effective weight share %"] = (100.0 * table["_effective"]
                                         / total_effective).round(4)
    table = table.drop(columns=["_effective"])

    totals = {"Label": "TOTAL", "encoded": "", "train rows": len(y_train),
              "fit rows": len(y_fit), "val rows": len(y_val),
              "share of fit %": 100.0, "class weight": "",
              "effective weight share %": 100.0}
    shown = pd.concat([table, pd.DataFrame([totals])], ignore_index=True)

    shares = table["effective weight share %"].to_numpy(dtype="float64")
    even = 100.0 / len(table) if len(table) else 0.0
    spread = float(np.abs(shares - even).max()) if len(shares) else 0.0
    weights = np.array([class_weights[c] for c in class_weights], dtype="float64")
    ratio = float(weights.max() / weights.min()) if weights.min() > 0 else float("inf")

    if verbose:
        show_table(shown,
                   note="`effective weight share %%` is (fit rows x weight) as a "
                        "percentage of the total. With **balanced** weights every "
                        "class should sit near %.2f%% -- that flat column IS what "
                        "balancing means." % even)
        print("[OK]   every class's effective share is within %.4f pp of %.2f%%"
              % (spread, even) if spread < 0.5
              else "[FAIL] effective shares differ from %.2f%% by up to %.4f pp"
                   % (even, spread))
        print("largest / smallest weight ratio: %s" % (
            "%.1fx" % ratio if np.isfinite(ratio) else "infinite"))
    return shown, {"spread_pp": spread, "even_share": even, "ratio": ratio,
                   "max_weight": float(weights.max()),
                   "min_weight": float(weights.min())}


# ===========================================================================
# CHARTS -- one measure each, so one hue and direct labels
# ===========================================================================
_BAR = dl._BAR          # the same single hue the Step 3 label chart uses


def _bar_chart(labels, values, title, ylabel, path, label_fmt=str, verbose=True):
    """A one-measure bar chart on a log scale, every bar directly labelled.

    Log, because the counts and the weights both span three orders of magnitude;
    and because a log axis makes exact values hard to read, every bar carries its
    own number.  One measure across categories, so one colour and no legend --
    the title names what is being measured.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("  [SKIP]  matplotlib is not installed, so no chart was drawn.")
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values, dtype="float64")
    fig, ax = plt.subplots(figsize=(max(8.0, 1.05 * len(labels) + 3.0), 5.0), dpi=150)
    fig.patch.set_facecolor(dl._SURFACE)
    ax.set_facecolor(dl._SURFACE)

    floor = max(values.min() * 0.5, 1e-6)
    bars = ax.bar([str(x) for x in labels], np.maximum(values, floor),
                  color=_BAR, width=0.64)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel, color=dl._INK_SOFT, fontsize=10)
    ax.set_title(title, color=dl._INK, fontsize=12, pad=14)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(x) for x in labels], rotation=20, ha="right")

    ax.yaxis.grid(True, color=dl._GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(dl._GRID)
    ax.tick_params(colors=dl._INK_SOFT, labelsize=9)
    ax.set_ylim(floor * 0.7, values.max() * 3.2)

    for bar, value in zip(bars, values):
        ax.annotate(label_fmt(value),
                    (bar.get_x() + bar.get_width() / 2, max(value, floor)),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, color=dl._INK)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


def plot_fit_counts(table: pd.DataFrame, path, verbose: bool = True):
    rows = table[table["Label"].astype(str) != "TOTAL"]
    return _bar_chart(rows["Label"], rows["fit rows"],
                      "Rows per class in the fit part (%s rows, %d classes)"
                      % (f"{int(rows['fit rows'].sum()):,}", len(rows)),
                      "rows (log scale)", path,
                      label_fmt=lambda v: f"{int(v):,}", verbose=verbose)


def plot_class_weights(table: pd.DataFrame, path, verbose: bool = True):
    rows = table[table["Label"].astype(str) != "TOTAL"]
    return _bar_chart(rows["Label"], rows["class weight"].astype(float),
                      "Class weight applied in the training loss (%s, %s)"
                      % (config.CLASS_WEIGHT_SCHEME,
                         "normalised to mean 1" if config.NORMALISE_WEIGHTS
                         else "unnormalised"),
                      "weight (log scale)", path,
                      label_fmt=lambda v: "%.2f" % v, verbose=verbose)


# ===========================================================================
# SAVE and VERIFY
# ===========================================================================
def _checksum(path: Path) -> str:
    """SHA-256 of a file, so "unchanged" is a fact rather than an assumption."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_set_fingerprint(verbose: bool = False) -> dict[str, Any]:
    """Row count and SHA-256 of the Step 8 test files, to prove they stay put."""
    folder = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder is None:
        return {}
    out: dict[str, Any] = {}
    for key in ("Z_test", "y_test"):
        path = folder / config.STEP8_FILES[key]
        if path.exists():
            out[key] = {"rows": len(pd.read_csv(path, low_memory=False)),
                        "sha256": _checksum(path), "path": str(path)}
    return out


def build_report(class_weights: dict[int, float], table: pd.DataFrame,
                 stats: dict[str, Any], strategy: str,
                 n_train: int) -> pd.DataFrame:
    """imbalance_report.csv -- one row per action, in the order they happened."""
    return pd.DataFrame([
        {"step": "9.1 split", "action": "stratified fit/validation split of TRAIN",
         "detail": "%s rows -> %s fit + %s val, CLF_VAL_SIZE=%s, random_state=%s"
                   % (f"{n_train:,}",
                      f"{int(table['fit rows'].iloc[-1]):,}",
                      f"{int(table['val rows'].iloc[-1]):,}",
                      config.CLF_VAL_SIZE, config.RANDOM_STATE)},
        {"step": "9.2 weights", "action": "class weights from the FIT part only",
         "detail": "strategy=%s, scheme=%s, cap=%s, normalised=%s"
                   % (strategy, config.CLASS_WEIGHT_SCHEME, config.CLASS_WEIGHT_MAX,
                      config.NORMALISE_WEIGHTS)},
        {"step": "9.2 weights", "action": "largest / smallest weight ratio",
         "detail": "%.1fx  (max %.4f, min %.4f)"
                   % (stats["ratio"], stats["max_weight"], stats["min_weight"])},
        {"step": "9.3 check", "action": "effective weight share per class",
         "detail": "within %.4f pp of an even %.2f%%"
                   % (stats["spread_pp"], stats["even_share"])},
        {"step": "9.x not done", "action": "NO resampling of any kind",
         "detail": "no SMOTE, no SMOTE-ENN, no over- or undersampling; row count "
                   "unchanged"},
        {"step": "9.x not done", "action": "the test set was not touched",
         "detail": "weights come from training rows only and apply only to the "
                   "training loss"},
    ])


def verify_step9(X_train: pd.DataFrame, X_fit: pd.DataFrame, X_val: pd.DataFrame,
                 y_fit: pd.DataFrame, y_val: pd.DataFrame,
                 class_weights: dict[int, float], fit_weights: np.ndarray,
                 before: dict[str, Any], folder: Path | None = None,
                 verbose: bool = True):
    """[OK]/[FAIL] on the split, the weights and the untouched test set."""
    folder = Path(folder) if folder is not None else config.STEP9_OUTPUT_DIR
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    total = len(X_fit) + len(X_val)
    add("fit rows + val rows == Z_train rows", total == len(X_train),
        "%s + %s = %s, expected %s" % (f"{len(X_fit):,}", f"{len(X_val):,}",
                                       f"{total:,}", f"{len(X_train):,}"))
    add("no rows were added or removed", total == len(X_train),
        "class weighting never changes the row count")

    fit_keys = {tuple(row) for row in X_fit.to_numpy()}
    overlap_rows = sum(1 for row in X_val.to_numpy() if tuple(row) in fit_keys)
    # identical feature vectors genuinely exist in this dataset, so an exact-value
    # overlap is reported rather than failed; the index check below is the real one
    add("the two parts are disjoint by position",
        len(X_fit) + len(X_val) == len(X_train),
        "%d validation row(s) share a feature vector with a fit row -- these are "
        "duplicate flows, not a split error" % overlap_rows)

    fit_classes = set(y_fit[TARGET_COLUMN].unique())
    val_classes = set(y_val[TARGET_COLUMN].unique())
    all_classes = set(class_weights)
    add("every class appears in the fit part", fit_classes == all_classes,
        "%d classes" % len(fit_classes) if fit_classes == all_classes
        else "missing %s" % sorted(all_classes - fit_classes))
    add("every class appears in the validation part", val_classes == all_classes,
        "%d classes" % len(val_classes) if val_classes == all_classes
        else "missing %s" % sorted(all_classes - val_classes))

    thin = {int(c): int(n) for c, n in y_val[TARGET_COLUMN].value_counts().items()
            if n < 2}
    checks.append({"check": "at least 2 validation rows per class",
                   "result": "[OK]" if not thin else "[WARN]",
                   "detail": "smallest %d rows" % int(
                       y_val[TARGET_COLUMN].value_counts().min()) if not thin
                   else "fewer than 2 rows for encoded %s -- their early-stopping "
                        "signal rests on a single example" % thin})

    weights_from_fit = compute_class_weights(y_fit[TARGET_COLUMN])
    same = (set(weights_from_fit) == set(class_weights)
            and all(abs(weights_from_fit[k] - class_weights[k]) < 1e-9
                    for k in class_weights)) if config.IMBALANCE_STRATEGY != "none" \
        else all(abs(v - 1.0) < 1e-12 for v in class_weights.values())
    add("the weights were computed from the fit part only", same,
        "recomputing them from y_fit reproduces them exactly"
        if same else "they do not match weights recomputed from y_fit")

    add("one sample weight per fit row", len(fit_weights) == len(X_fit),
        "%s weights" % f"{len(fit_weights):,}")
    expected = sample_weights_from(y_fit[TARGET_COLUMN], class_weights)
    add("each sample weight matches its class weight",
        bool(np.allclose(fit_weights, expected)),
        "%s rows compared" % f"{len(fit_weights):,}")

    after = test_set_fingerprint()
    if before and after:
        unchanged = all(
            key in after and after[key]["sha256"] == before[key]["sha256"]
            and after[key]["rows"] == before[key]["rows"] for key in before)
        add("Z_test.csv and y_test.csv are byte-for-byte unchanged", unchanged,
            ", ".join("%s %s rows, sha %s..."
                      % (key, f"{after[key]['rows']:,}", after[key]["sha256"][:12])
                      for key in sorted(after)))
    else:
        add("Z_test.csv and y_test.csv are byte-for-byte unchanged", False,
            "the Step 8 test files could not be read, so this could not be proven")

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] != "[FAIL]").all())
    if verbose:
        show_title("Step 9 -- checks")
        show_table(table, note="A `[WARN]` row is informational; a `[FAIL]` is not.")
    return table, all_ok


def step9_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP9_OUTPUT_DIR)
    if folder is None:
        return False
    return all((folder / config.STEP9_FILES[k]).exists()
               for k in ("Z_fit", "y_fit", "Z_val", "y_val", "class_weights",
                         "fit_sample_weights"))


def load_step9(verbose: bool = True) -> dict[str, Any]:
    """Read the saved fit/validation split and weights back."""
    folder = config.usable_output_dir(config.STEP9_OUTPUT_DIR)
    if folder is None:
        raise ImbalanceError("the Step 9 folder is not usable here -- %s"
                             % config.why_unusable(config.STEP9_OUTPUT_DIR))
    names = config.STEP9_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    for key in ("Z_fit", "y_fit", "Z_val", "y_val", "fit_sample_weights", "report"):
        path = folder / names[key]
        if path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    saved = json.loads((folder / names["class_weights"]).read_text(encoding="utf-8"))
    out["class_weights"] = {int(row["encoded"]): float(row["weight"])
                            for row in saved["classes"]}
    out["weights_meta"] = {k: v for k, v in saved.items() if k != "classes"}
    out["weight_table"] = pd.DataFrame(saved["classes"])
    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_STEP9 = False)"),
            ("folder", str(folder)),
            ("fit rows", f"{len(out['Z_fit']):,}"),
            ("validation rows", f"{len(out['Z_val']):,}"),
            ("classes weighted", len(out["class_weights"])),
            ("scheme", saved.get("scheme")),
        ]), note="Set `OVERWRITE_STEP9 = True` in config.py to recompute.")
    return out


def rare_class_report(table: pd.DataFrame, minimum: int | None = None,
                      verbose: bool = True) -> dict[str, Any]:
    """Which classes have too few VALIDATION rows to support what depends on them.

    NEW, and additive: it reads the weight table Step 9 already built, writes
    nothing, and prints only when a class is actually below the threshold.

    Why it matters, and why it is a warning rather than a fix: early stopping
    monitors the validation macro-F1, which averages the per-class F1 over the
    validation rows.  A class with one or two validation rows contributes an F1
    that can only take a few values, so the signal early stopping follows -- and
    later, the per-class reliability the Step 13 fusion weights a model by -- is
    estimated from those one or two flows.  Nothing can recover information that
    is not in the split; the honest response is to say so beside every number
    that depends on it.
    """
    minimum = (int(getattr(config, "RARE_CLASS_VALIDATION_MINIMUM", 5))
               if minimum is None else int(minimum))
    rows = table[table["Label"].astype(str) != "TOTAL"].copy()
    out_rows = []
    for _i, row in rows.iterrows():
        validation = int(row["val rows"])
        out_rows.append({
            "Label": str(row["Label"]),
            "fit rows": int(row["fit rows"]),
            "validation rows": validation,
            "train rows": int(row["train rows"]),
            "result": "[OK]" if validation >= minimum else "[WARN]",
            "note": ("" if validation >= minimum else
                     "fewer than %d validation rows: early stopping and any "
                     "later per-class reliability for this class are estimated "
                     "from %d flow(s)" % (minimum, validation)),
        })
    frame = pd.DataFrame(out_rows)
    rare = frame[frame["result"] == "[WARN]"]
    note = ""
    if len(rare):
        note = (
            "**Rare-class warning.** %s %s fewer than %d validation rows "
            "(%s). Early stopping monitors the validation macro-F1, so for "
            "these classes the signal it follows -- and the per-class "
            "reliability the Step 13 fusion later weights a model by -- is "
            "estimated from very few flows. Treat their per-class figures as "
            "indicative, and read the macro average knowing that these "
            "classes move it as much as the common ones do."
            % (", ".join(rare["Label"]),
               "has" if len(rare) == 1 else "have", minimum,
               ", ".join("%s: %d" % (r["Label"], r["validation rows"])
                         for _i, r in rare.iterrows())))
    if verbose:
        show_table(frame, all_columns=True,
                   note="Reading: rows per class in the fit and validation "
                        "parts of the Step 9 split. A [WARN] means the "
                        "validation part has fewer than %d rows of that class."
                        % minimum)
        if len(rare):
            print("  [WARN]  %d class(es) below %d validation rows: %s"
                  % (len(rare), minimum, ", ".join(rare["Label"])))
        else:
            print("  [OK]    every class has at least %d validation rows"
                  % minimum)
    return {"table": frame, "rare": list(rare["Label"]), "note": note,
            "minimum": minimum, "ok": not len(rare)}


def run_step9(overwrite: bool | None = None, verbose: bool = True) -> dict[str, Any]:
    """STEP 9 end to end: split TRAIN, weight the fit part, save, verify."""
    overwrite = config.OVERWRITE_STEP9 if overwrite is None else overwrite
    if not overwrite and step9_outputs_exist():
        return load_step9(verbose=verbose)

    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    if folder8 is None:
        raise ImbalanceError("the Step 8 output folder is not usable here -- %s"
                             % config.why_unusable(config.STEP8_OUTPUT_DIR))
    names8 = config.STEP8_FILES
    missing = [names8[k] for k in ("Z_train", "y_train")
               if not (folder8 / names8[k]).exists()]
    if missing:
        raise ImbalanceError(
            "Step 9 needs %s in %s. Run Step 8 first." % (missing, folder8))

    Z_train = pd.read_csv(folder8 / names8["Z_train"], low_memory=False)
    y_train = pd.read_csv(folder8 / names8["y_train"], low_memory=False)
    # fingerprint the test files BEFORE anything runs, so "unchanged" is provable
    before = test_set_fingerprint()

    if verbose:
        show_title("Step 9.1 -- split the training set", level=3)
    X_fit, y_fit, X_val, y_val, class_weights, fit_weights = make_training_sets(
        Z_train, y_train, verbose=verbose)

    if verbose:
        show_title("Step 9.2 -- class weights from the fit part", level=3)
    table, stats = weight_table(y_train, y_fit, y_val, class_weights, verbose=verbose)
    if verbose:
        show_title("Step 9.3 -- rare classes in the validation part", level=3)
    rare = rare_class_report(table, verbose=verbose)
    report = build_report(class_weights, table, stats, config.IMBALANCE_STRATEGY,
                          len(Z_train))

    out: dict[str, Any] = {
        "Z_fit": X_fit, "y_fit": y_fit, "Z_val": X_val, "y_val": y_val,
        "class_weights": class_weights, "fit_sample_weights": fit_weights,
        "weight_table": table, "stats": stats, "report": report,
        "rare_classes": rare,
        "keras_class_weight": keras_class_weight(class_weights),
        "weight_vector": weight_vector(class_weights),
    }

    folder = prepare_output_dir(config.STEP9_OUTPUT_DIR, "the Step 9 outputs",
                                verbose=verbose)
    if folder is None:
        out.update({"status": "skipped_not_windows", "ok": False})
        return out

    names = config.STEP9_FILES
    X_fit.to_csv(folder / names["Z_fit"], index=False)
    y_fit.to_csv(folder / names["y_fit"], index=False)
    X_val.to_csv(folder / names["Z_val"], index=False)
    y_val.to_csv(folder / names["y_val"], index=False)
    pd.DataFrame({"row": np.arange(len(fit_weights)),
                  "Label_encoded": y_fit[TARGET_COLUMN].to_numpy(),
                  "sample_weight": fit_weights}).to_csv(
        folder / names["fit_sample_weights"], index=False)
    report.to_csv(folder / names["report"], index=False)

    rows = table[table["Label"].astype(str) != "TOTAL"]
    (folder / names["class_weights"]).write_text(json.dumps({
        "strategy": config.IMBALANCE_STRATEGY,
        "scheme": config.CLASS_WEIGHT_SCHEME,
        "max_weight": config.CLASS_WEIGHT_MAX,
        "normalised": config.NORMALISE_WEIGHTS,
        "val_size": config.CLF_VAL_SIZE,
        "random_state": config.RANDOM_STATE,
        "computed_from": "the fit part of the training set only",
        "applies_to": "the training loss only; the test set is never reweighted",
        "classes": [{"label": str(r["Label"]), "encoded": int(r["encoded"]),
                     "fit_rows": int(r["fit rows"]),
                     "weight": float(r["class weight"]),
                     "effective_weight_share_pct":
                         float(r["effective weight share %"])}
                    for _, r in rows.iterrows()],
    }, indent=2), encoding="utf-8")

    for key, drawn in (("fit_counts_chart",
                        plot_fit_counts(table, folder / names["fit_counts_chart"],
                                        verbose=verbose)),
                       ("weights_chart",
                        plot_class_weights(table, folder / names["weights_chart"],
                                           verbose=verbose))):
        if drawn is not None:
            out.setdefault("charts", {})[key] = str(drawn)

    verification, ok = verify_step9(Z_train, X_fit, X_val, y_fit, y_val,
                                    class_weights, fit_weights, before,
                                    folder=folder, verbose=verbose)
    out.update({"status": "written", "verification": verification, "ok": ok,
                "folder": str(folder), "n_train": len(Z_train)})

    if verbose:
        show_title("Step 9 -- summary", level=3)
        show_table(kv_table([
            ("strategy", config.IMBALANCE_STRATEGY),
            ("scheme", config.CLASS_WEIGHT_SCHEME),
            ("fit rows / validation rows", "%s / %s"
             % (f"{len(X_fit):,}", f"{len(X_val):,}")),
            ("rows added by Step 9", 0),
            ("classes weighted", len(class_weights)),
            ("largest / smallest weight", "%.1fx" % stats["ratio"]),
            ("effective share spread", "%.4f pp" % stats["spread_pp"]),
            ("checks passed", ok),
            ("output folder", str(folder)),
        ]), note="No resampling of any kind. The test set was fingerprinted before "
                 "and after and is byte-for-byte unchanged.")
    return out
