r"""
src/mscad/experiments_pca.py -- STEP 12 (addendum): Baseline 4, AE -> PCA -> LSTM.

NEW FILE.  Steps 1-14 are finished: nothing in converter.py, data_loading.py,
preprocessing.py, splitting.py, feature_selection.py, autoencoder.py,
imbalance.py, models.py, evaluation.py, experiments.py, fusion.py or
explainability.py is modified.  This module imports those and adds one more
configuration beside them.

Why a separate module
---------------------
Step 12's four configurations are finished and verified, and Steps 13 and 14
import `experiments.py` and address those configurations by name.  Editing that
module to thread a PCA step through `run_experiment` would put the fusion and
the explainability layers at risk for the sake of a fifth row in a table, so
Baseline 4 is built here instead and the two are compared afterwards.

What Baseline 4 is
------------------
    all Step 5 features -> AE encoder (all -> 15) -> PCA -> LSTM

It shares everything with Baseline 2 (AE -> LSTM) except the PCA step between
them, so the difference between those two rows is what PCA did and nothing
else.  The encoder is REUSED from Baseline 1 rather than retrained, for the
same reason: two configurations that differ in two places explain nothing.

Where the leakage guards are
----------------------------
The PCA is fitted on the TRAINING latent features only -- the same rows the
scaler and the autoencoder were fitted on -- and the test set is only ever
passed through `.transform()`.  `assert_pca_fitted_on_train_only` re-fits on a
deliberately corrupted test set and asserts the components do not move, so the
guarantee is measured rather than asserted in a comment.
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
from . import models as md  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = "Label_encoded"


class PCAExperimentError(Exception):
    """Raised when Baseline 4 cannot continue without guessing."""


# ===========================================================================
# 12B.1  THE PCA SETTING
# ===========================================================================
def pca_components_setting(n_latent: int, setting: Any = None) -> Any:
    """Validate PCA_N_COMPONENTS against the latent width, or say why not.

    A float in (0, 1) is a variance target and sklearn handles it directly.  An
    int is a fixed component count, and asking for more components than the
    latent space has is not a rounding problem -- it means the setting and the
    autoencoder disagree, so it raises.
    """
    setting = config.PCA_N_COMPONENTS if setting is None else setting
    n_latent = int(n_latent)

    if isinstance(setting, bool):
        raise PCAExperimentError(
            "PCA_N_COMPONENTS is %r; it must be a float in (0, 1) or an int"
            % setting)
    if isinstance(setting, float):
        if not 0.0 < setting < 1.0:
            raise PCAExperimentError(
                "PCA_N_COMPONENTS is %r; a float must lie strictly between 0 "
                "and 1 (the fraction of variance to keep). Use an int for a "
                "fixed number of components." % setting)
        return float(setting)
    if isinstance(setting, (int, np.integer)):
        value = int(setting)
        if value < 1:
            raise PCAExperimentError(
                "PCA_N_COMPONENTS is %d; it must be at least 1" % value)
        if value > n_latent:
            raise PCAExperimentError(
                "PCA_N_COMPONENTS is %d but the autoencoder latent space has "
                "only %d dimensions, so PCA cannot produce that many "
                "components. Lower PCA_N_COMPONENTS to %d or less, or raise "
                "AE_LATENT_DIM." % (value, n_latent, n_latent))
        return value
    raise PCAExperimentError(
        "PCA_N_COMPONENTS is %r (%s); it must be a float in (0, 1) or an int"
        % (setting, type(setting).__name__))


def pca_columns(n_components: int) -> list[str]:
    """pc_00, pc_01, ...  Zero-padded so the columns sort in order."""
    return ["pc_%02d" % i for i in range(int(n_components))]


def fit_pca_on_train(Z_train: pd.DataFrame, setting: Any = None,
                     seed: int | None = None):
    """Fit PCA on the TRAINING latent features.  Returns (pca, info).

    The test latent features are not an argument here on purpose: a function
    that never sees them cannot fit on them.
    """
    from sklearn.decomposition import PCA

    seed = config.RANDOM_STATE if seed is None else seed
    values = Z_train.to_numpy(dtype="float64")
    requested = pca_components_setting(values.shape[1], setting)
    pca = PCA(n_components=requested, random_state=seed, whiten=False)
    pca.fit(values)

    ratio = np.asarray(pca.explained_variance_ratio_, dtype="float64")
    info = {
        "requested": requested,
        "requested_kind": ("variance fraction" if isinstance(requested, float)
                           else "fixed components"),
        "n_components": int(pca.n_components_),
        "latent_dim": int(values.shape[1]),
        "variance_kept": float(ratio.sum()),
        "fitted_on": "the Step 5 TRAINING rows only (%s rows)"
                     % f"{len(values):,}",
        "seed": int(seed), "whiten": False,
    }
    return pca, info


def pca_variance_table(pca) -> pd.DataFrame:
    """Explained variance per component, and the running total."""
    ratio = np.round(np.asarray(pca.explained_variance_ratio_,
                                dtype="float64"), 8)
    return pd.DataFrame({
        "component": pca_columns(len(ratio)),
        "explained_variance": np.round(
            np.asarray(pca.explained_variance_, dtype="float64"), 8),
        "explained_variance_ratio": ratio,
        # the running total is built from the ROUNDED column, so adding that
        # column up by hand gives exactly the total printed beside it
        "cumulative_variance_ratio": np.round(np.cumsum(ratio), 8),
    })


def transform_with_pca(pca, Z: pd.DataFrame, what: str = "the frame"
                       ) -> pd.DataFrame:
    """.transform() only.  Never .fit(), whatever `Z` is."""
    if list(Z.columns) != list(getattr(pca, "feature_names_in_", Z.columns)):
        pass                      # column names are not part of the transform
    if Z.shape[1] != int(pca.n_features_in_):
        raise PCAExperimentError(
            "%s has %d columns but the PCA was fitted on %d"
            % (what, Z.shape[1], int(pca.n_features_in_)))
    values = pca.transform(Z.to_numpy(dtype="float64"))
    return pd.DataFrame(values, columns=pca_columns(values.shape[1]),
                        index=Z.index)


def assert_pca_fitted_on_train_only(pca, Z_train: pd.DataFrame,
                                    Z_test: pd.DataFrame,
                                    setting: Any = None,
                                    seed: int | None = None) -> dict[str, Any]:
    """Re-fit on the training rows while the test rows are corrupted.

    If a single number in the fitted PCA moved, the test set reached `.fit()`
    somewhere, and the comparison would be measuring the wrong thing.
    """
    spoiled = pd.DataFrame(
        Z_test.to_numpy(dtype="float64") * 7.0 + 13.0,
        columns=list(Z_test.columns), index=Z_test.index)

    # (a) re-fitting on the training rows alone reproduces it exactly
    again, _info = fit_pca_on_train(Z_train, setting=setting, seed=seed)
    components = float(np.abs(np.asarray(pca.components_)
                              - np.asarray(again.components_)).max())
    mean = float(np.abs(np.asarray(pca.mean_)
                        - np.asarray(again.mean_)).max())
    same = (components < 1e-12 and mean < 1e-12
            and int(pca.n_components_) == int(again.n_components_))

    # (b) and the corrupted test rows WOULD have moved it, had they reached
    #     .fit() -- without this the check above could pass on a PCA that is
    #     simply insensitive to its input
    mixed = pd.concat([Z_train, spoiled], ignore_index=True)
    leaky, _leaky_info = fit_pca_on_train(mixed, setting=setting, seed=seed)
    moved = float(np.abs(np.asarray(pca.mean_)
                         - np.asarray(leaky.mean_)).max())
    sensitive = moved > 1e-9

    return {"check": "the PCA was fitted on the training rows only",
            "result": "[OK]" if (same and sensitive) else "[FAIL]",
            "detail": ("re-fit on train reproduces it exactly (components "
                       "%.1e, mean %.1e); the same fit including the test rows "
                       "would move the mean by %.3e, so the test set never "
                       "reached .fit()" % (components, mean, moved))}


def plot_scree(table: pd.DataFrame, info: dict[str, Any], path,
               verbose: bool = True):
    """Variance per component as bars, the running total as a labelled line."""
    plt = xp._pyplot(verbose)
    if plt is None or table.empty:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ratio = table["explained_variance_ratio"].to_numpy(dtype="float64")
    cumulative = table["cumulative_variance_ratio"].to_numpy(dtype="float64")
    positions = np.arange(len(ratio), dtype="float64")

    fig, ax = plt.subplots(figsize=(max(7.5, 0.95 * len(ratio) + 3.0), 5.2),
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

    # one axis, one scale: both series are fractions of the same total, so no
    # second y-axis is needed and none is used
    bars = ax.bar(positions, ratio, 0.62, color=xp._SERIES[0],
                  label="variance of this component")
    ax.plot(positions, cumulative, color=xp._SERIES[1], linewidth=2.0,
            marker="o", markersize=4, label="cumulative variance")
    for position, value in zip(positions, ratio):
        ax.annotate("%.3f" % value, (position, value), ha="center", va="bottom",
                    fontsize=8, color=dl._INK_SOFT, xytext=(0, 3),
                    textcoords="offset points")
    for position, value in zip(positions, cumulative):
        # offset to the right as well as down: at the first component the
        # cumulative value equals the bar's, and two labels on one point is
        # one label nobody can read
        ax.annotate("%.3f" % value, (position, value), ha="left", va="top",
                    fontsize=8, color=xp._SERIES[1], xytext=(7, -5),
                    textcoords="offset points")

    kept = info.get("variance_kept", float(cumulative[-1]))
    if isinstance(info.get("requested"), float):
        ax.axhline(info["requested"], color=dl._INK, linewidth=1.2,
                   linestyle="--")
        ax.annotate("target %.2f" % info["requested"],
                    (len(ratio) - 0.4, info["requested"]), ha="right",
                    va="bottom", fontsize=9, color=dl._INK,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(positions)
    ax.set_xticklabels(table["component"], rotation=0, fontsize=8.5,
                       color=dl._INK_SOFT)
    ax.set_ylabel("fraction of the latent variance", color=dl._INK_SOFT,
                  fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("PCA on the %d-dimensional AE latent space: %d components "
                 "keep %.1f%%"
                 % (info.get("latent_dim", len(ratio)),
                    info.get("n_components", len(ratio)), 100 * kept),
                 color=dl._INK, fontsize=12, pad=14)
    legend = ax.legend(frameon=False, fontsize=9, loc="center right")
    for text in legend.get_texts():
        text.set_color(dl._INK_SOFT)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print("  [SAVE]  %s" % path)
    return path


# ===========================================================================
# 12B.2  THE ENCODER -- reused, never retrained
# ===========================================================================
def baseline1_folder() -> Path | None:
    """Where Baseline 1 keeps the encoder Baseline 4 borrows."""
    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder is None:
        return None
    return folder / "baseline1_ae_cnn_lstm"


def reused_encoder(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.DataFrame, out_folder: Path,
                   epochs: int | None = None, verbose: bool = True):
    """The encoder Baseline 4 uses, and where it came from.

    "selected30" borrows the saved Step 7 encoder; "all" borrows Baseline 1's.
    Neither is retrained.  Only if Baseline 1 has never been run is an
    autoencoder trained here -- once, with Step 7's hyperparameters -- and the
    provenance says so, so a reader is never left guessing which AE produced a
    latent space.
    """
    keras = md._keras()
    mode = str(config.BASELINE4_AE_FEATURES)

    if mode == "selected30":
        folder7 = config.usable_output_dir(config.STEP7_OUTPUT_DIR)
        path = (folder7 / config.STEP7_FILES["encoder"]) if folder7 else None
        if path is None or not path.exists():
            raise PCAExperimentError(
                "BASELINE4_AE_FEATURES = \"selected30\" reuses the Step 7 "
                "encoder, and it is missing:\n    %s\nRun Step 7 first. "
                "Baseline 4 never trains its own." % path)
        encode, encoder, order = ae.load_encoder(verbose=False)
        provenance = {"source": "step7", "path": str(path),
                      "trained_here": False,
                      "note": "the Step 7 encoder, as the proposed "
                              "configuration uses"}
        if verbose:
            print("          encoder reused from Step 7 (%d features -> %d)"
                  % (len(order), int(encoder.output_shape[-1])))
        return encode, encoder, list(order), provenance

    if mode != "all":
        raise PCAExperimentError(
            "BASELINE4_AE_FEATURES is %r; it must be \"all\" or \"selected30\""
            % mode)

    folder1 = baseline1_folder()
    encoder_path = (folder1 / "encoder.keras") if folder1 else None
    settings_path = (folder1 / "ae_config.json") if folder1 else None
    if (encoder_path is not None and encoder_path.exists()
            and settings_path is not None and settings_path.exists()):
        encoder = keras.models.load_model(encoder_path)
        order = list(json.loads(settings_path.read_text(encoding="utf-8"))
                     ["features"])
        encode = ae.make_encoder_function(encoder, order)
        provenance = {"source": "baseline1_ae_cnn_lstm",
                      "path": str(encoder_path), "trained_here": False,
                      "note": "Baseline 1's encoder, so Baseline 2 and "
                              "Baseline 4 differ only by the PCA step"}
        if verbose:
            print("          encoder reused from Baseline 1 (%d features -> %d,"
                  " not retrained)" % (len(order), int(encoder.output_shape[-1])))
        return encode, encoder, order, provenance

    # Baseline 1 has not been run: train the shared autoencoder once, here.
    if verbose:
        print("          Baseline 1 has not been run, so the autoencoder is "
              "trained once here with Step 7's hyperparameters")
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    _Z_train, _Z_test, info = xp.encode_with_new_ae(
        X_train, X_test, y_train, out_folder, epochs=epochs, verbose=False)
    encoder = keras.models.load_model(out_folder / "encoder.keras")
    order = list(X_train.columns)
    encode = ae.make_encoder_function(encoder, order)
    provenance = {"source": "trained_in_baseline4", "path": str(out_folder /
                                                               "encoder.keras"),
                  "trained_here": True, "ae_info": info,
                  "note": "Baseline 1 was not on disk. Run Step 12 first if "
                          "you want Baselines 1, 2 and 4 to share one encoder."}
    return encode, encoder, order, provenance


# ===========================================================================
# 12B.3  RUNNING BASELINE 4
# ===========================================================================
def baseline4_is_done() -> bool:
    folder = xp.experiment_folder(config.BASELINE4["name"])
    return folder is not None and (folder / "metrics.json").exists()


def _ms_per_1000(seconds: float, rows: int) -> float:
    return float(seconds / max(rows, 1) * 1000 * 1000)


def run_baseline4(y_train: pd.DataFrame, y_test: pd.DataFrame,
                  meta_test: pd.DataFrame | None, label_mapping: dict[str, Any],
                  duplicate_flags=None, seed: int | None = None,
                  epochs: int | None = None, overwrite: bool | None = None,
                  verbose: bool = True) -> dict[str, Any]:
    """Train (or load) Baseline 4 and score it with Step 11's evaluate()."""
    from joblib import dump

    overwrite = config.OVERWRITE_BASELINE4 if overwrite is None else overwrite
    spec = dict(config.BASELINE4)
    names = config.BASELINE4_FILES
    folder = xp.experiment_folder(spec["name"])
    if folder is None:
        raise PCAExperimentError(
            "the Step 12 folder is not usable here -- %s"
            % config.why_unusable(config.STEP12_OUTPUT_DIR))

    if not overwrite and baseline4_is_done():
        saved = json.loads((folder / names["metrics"]).read_text(
            encoding="utf-8"))
        if verbose:
            print("  [SKIP]  %-34s already finished (macro-F1 %.4f)"
                  % (spec["label"], saved["full"]["f1_macro"]))
        return {"spec": spec, "status": "loaded_existing", "metrics": saved,
                "folder": str(folder)}

    started = time.perf_counter()
    if verbose:
        print("  [RUN ]  %-34s ..." % spec["label"], flush=True)

    source = xp.load_feature_source(spec["feature_source"], verbose=False)
    X_train, X_test = source["X_train"], source["X_test"]
    folder.mkdir(parents=True, exist_ok=True)

    encode, encoder, order, provenance = reused_encoder(
        X_train, X_test, y_train, folder, epochs=epochs, verbose=verbose)
    Z_train = encode(X_train[order], what="the training features")
    Z_test = encode(X_test[order], what="the test features")

    pca, pca_info = fit_pca_on_train(Z_train, seed=seed)
    leak_check = assert_pca_fitted_on_train_only(pca, Z_train, Z_test, seed=seed)
    if leak_check["result"] != "[OK]":
        raise PCAExperimentError(
            "the PCA changed when the test rows changed, so it was not fitted "
            "on the training rows alone: %s" % leak_check["detail"])
    variance = pca_variance_table(pca)
    P_train = transform_with_pca(pca, Z_train, "the training latent features")
    P_test = transform_with_pca(pca, Z_test, "the test latent features")
    if verbose:
        print("          PCA %s -> %d components, %.1f%% of the latent "
              "variance kept"
              % (pca_info["requested_kind"], pca_info["n_components"],
                 100 * pca_info["variance_kept"]))

    model, history, info = xp.train_classifier(
        P_train, y_train, model_type=spec["model_type"], seed=seed,
        epochs=epochs, verbose=False)

    X_test_arr, _labels, row_index = md.arrange_inputs(P_test, y_test,
                                                       meta_test, verbose=False)
    flags = None
    if duplicate_flags is not None:
        flags = (np.asarray(duplicate_flags)[row_index]
                 if len(row_index) != len(duplicate_flags) else duplicate_flags)
    result = ev.evaluate(model, X_test_arr, y_test, label_mapping,
                         meta_test=meta_test, duplicate_flags=flags,
                         model_name=spec["label"],
                         training_seconds=info.get("seconds"), verbose=False)

    # end-to-end inference: the encoder and the PCA transform are part of what
    # this configuration costs in service, and the shared comparison column
    # measures the classifier alone, so both are reported rather than mixed
    clock = time.perf_counter()
    _warm = encode(X_test[order].iloc[: min(64, len(X_test))],
                   what="the test features")
    clock = time.perf_counter()
    encoded = encode(X_test[order], what="the test features")
    encode_seconds = time.perf_counter() - clock
    clock = time.perf_counter()
    transform_with_pca(pca, encoded, "the test latent features")
    pca_seconds = time.perf_counter() - clock
    classifier_ms = float(result["inference"]["ms_per_1000_flows"])
    pipeline = {
        "encoder_ms_per_1000_flows": round(
            _ms_per_1000(encode_seconds, len(X_test)), 4),
        "pca_transform_ms_per_1000_flows": round(
            _ms_per_1000(pca_seconds, len(X_test)), 4),
        "classifier_ms_per_1000_flows": round(classifier_ms, 4),
        "end_to_end_ms_per_1000_flows": round(
            _ms_per_1000(encode_seconds + pca_seconds, len(X_test))
            + classifier_ms, 4),
        "note": ("the shared comparison column measures the classifier alone, "
                 "as Step 11 does for every configuration; these four lines "
                 "add the encoder and the PCA transform for this one"),
    }

    # ---- save -----------------------------------------------------------
    model.save(folder / names["model"])
    encoder.save(folder / names["encoder"])
    dump(pca, folder / names["pca"])
    variance.to_csv(folder / names["pca_variance"], index=False)
    if history is not None:
        history.to_csv(folder / names["history"], index=False)
    result["per_class"].to_csv(folder / names["per_class"], index=False)
    result["confusion_counts"].to_csv(folder / names["confusion_counts"])
    result["confusion_normalised"].to_csv(folder / names["confusion_rownorm"])
    ev.build_predictions_frame(result, meta_test, row_index).to_csv(
        folder / names["predictions"], index=False)
    (folder / names["ae_config"]).write_text(json.dumps({
        "features": list(order), "n_features": len(order),
        "latent_dim": int(Z_train.shape[1]),
        "encoder_provenance": provenance,
        "trained_here": bool(provenance.get("trained_here")),
    }, indent=2, default=float), encoding="utf-8")
    plot_scree(variance, pca_info, folder / names["pca_scree"], verbose=verbose)

    metrics = {
        "config": spec["name"], "label": spec["label"],
        "feature_source": spec["feature_source"], "use_ae": True,
        "use_pca": True, "model_type": spec["model_type"],
        "n_input_features": int(source["n_features"]),
        "latent_dim": int(Z_train.shape[1]),
        "pca_components": int(pca_info["n_components"]),
        "pca": pca_info | {"variance_per_component":
                           variance["explained_variance_ratio"].tolist()},
        "encoder_provenance": provenance,
        "model_input_shape": list(info.get("input_shape", [])),
        "parameters": int(info.get("parameters", model.count_params())),
        "seed": info.get("seed", seed), "training_seconds": info.get("seconds"),
        "best_epoch": info.get("best_epoch"),
        "best_val_macro_f1": info.get("best_val_macro_f1"),
        "full": result["full"], "duplicate_free": result["duplicate_free"],
        "binary": result["binary"], "inference": result["inference"],
        "pipeline_inference": pipeline,
        "reused_step7_10": bool(spec.get("reuse_step7_10")),
    }
    (folder / names["metrics"]).write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    elapsed = time.perf_counter() - started
    if verbose:
        print("  [DONE]  %-34s macro-F1 %.4f (dup-free %s) in %.1f s"
              % (spec["label"], result["full"]["f1_macro"],
                 ev._fmt(result["duplicate_free"].get("f1_macro")), elapsed))
    return {"spec": spec, "status": "written", "metrics": metrics,
            "result": result, "model": model, "history": history,
            "pca": pca, "pca_info": pca_info, "variance": variance,
            "leak_check": leak_check, "folder": str(folder), "elapsed": elapsed}


# ===========================================================================
# 12B.4  THE FIVE-CONFIGURATION COMPARISON
# ===========================================================================
def load_step12_runs(verbose: bool = True) -> list[dict[str, Any]]:
    """The four finished Step 12 configurations, read from disk.

    Read rather than re-run: Step 12 is finished, and re-running it would
    rewrite its tables for nothing.  A missing configuration is named, with the
    step that produces it.
    """
    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder is None:
        raise PCAExperimentError(
            "the Step 12 output folder is not usable here -- %s"
            % config.why_unusable(config.STEP12_OUTPUT_DIR))
    runs, missing = [], []
    for spec in config.EXPERIMENTS:
        path = folder / spec["name"] / "metrics.json"
        if not path.exists():
            missing.append(str(path))
            continue
        runs.append({"spec": spec, "status": "loaded_existing",
                     "metrics": json.loads(path.read_text(encoding="utf-8")),
                     "folder": str(folder / spec["name"])})
    if missing:
        raise PCAExperimentError(
            "these Step 12 configurations have not been run:\n    %s\n"
            "Run Step 12 first (set OVERWRITE_STEP12 = True to rebuild it). "
            "Baseline 4 is compared against them, not instead of them."
            % "\n    ".join(missing))
    if verbose:
        print("  loaded %d finished Step 12 configurations from disk"
              % len(runs))
    return runs


def architecture_table() -> pd.DataFrame:
    """config | feature selection | AE | PCA | CNN | LSTM | input shape."""
    rows = []
    for spec in config.EXPERIMENTS_WITH_PCA:
        selection = ("Step 6 top %d" % config.N_SELECTED_FEATURES
                     if spec["feature_source"] == "selected30" else "none")
        uses_ae = bool(spec.get("use_ae"))
        uses_pca = bool(spec.get("use_pca"))
        if uses_pca:
            shape = "(n_components, 1)"
        elif uses_ae:
            shape = "(%d, 1)" % config.AE_LATENT_DIM
        else:
            shape = "(n_features, 1)"
        rows.append({
            "config": spec["label"],
            "feature selection": selection,
            "AE": ("yes (-> %d)" % config.AE_LATENT_DIM) if uses_ae else "no",
            "PCA": ("yes (%s)" % config.PCA_N_COMPONENTS) if uses_pca else "no",
            "CNN": "yes" if spec["model_type"] == "cnn_lstm" else "no",
            "LSTM": "yes",
            "input shape": shape,
        })
    return pd.DataFrame(rows)


def combined_comparison(runs: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Step 12's own table with a PCA column and the Baseline 4 row.

    `experiments.comparison_table` is called unchanged and the PCA column is
    inserted afterwards, so the four existing rows are produced by exactly the
    code that produced them before.
    """
    table = xp.comparison_table(runs)
    components = []
    for run in runs:
        value = run["metrics"].get("pca_components")
        latent = run["metrics"].get("latent_dim")
        components.append("%d of %d" % (int(value), int(latent))
                          if value and latent else "-")
    position = (list(table.columns).index("latent dim") + 1
                if "latent dim" in table.columns else len(table.columns))
    table.insert(position, "PCA components", components)
    return table


def mark_best(table: pd.DataFrame) -> pd.DataFrame:
    """experiments.mark_best, with the PCA column left alone.

    It is already a string ("8 of 15"), so `pd.to_numeric` makes it NaN and the
    existing function skips it -- no "best" is claimed for a component count.
    """
    return xp.mark_best(table)


def checks_table(runs: Sequence[dict[str, Any]], y_train: pd.DataFrame,
                 y_test: pd.DataFrame, before: dict[str, Any],
                 baseline4: dict[str, Any],
                 verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """[OK]/[FAIL] that Baseline 4 really is comparable with the other four."""
    sources = {spec["name"]: xp.load_feature_source(spec["feature_source"],
                                                    verbose=False)
               for spec in config.EXPERIMENTS_WITH_PCA}
    shared, _ok = xp.consistency_checks(sources, y_train, y_test, before,
                                        verbose=False)
    rows = shared.to_dict("records")

    if "leak_check" in baseline4:
        rows.append(baseline4["leak_check"])
    provenance = (baseline4.get("metrics") or {}).get("encoder_provenance", {})
    reused = not provenance.get("trained_here", True)
    rows.append({
        "check": "Baseline 4 reused an existing encoder rather than training one",
        "result": "[OK]" if reused else "[WARN]",
        "detail": ("from %s" % provenance.get("source", "?")) if reused
                  else "Baseline 1 was absent, so one was trained here once"})

    metrics = baseline4.get("metrics") or {}
    shape = list(metrics.get("model_input_shape") or [])
    components = metrics.get("pca_components")
    right_shape = bool(shape and components and int(shape[0]) == int(components)
                       and len(shape) == 2 and int(shape[1]) == 1)
    rows.append({
        "check": "Baseline 4's classifier input is (n_components, 1)",
        "result": "[OK]" if right_shape else "[FAIL]",
        "detail": "input shape %s for %s components" % (tuple(shape), components)})

    lstm_only = str(metrics.get("model_type")) == "lstm"
    rows.append({
        "check": "Baselines 2 and 4 are LSTM-only (no convolutions)",
        "result": "[OK]" if lstm_only else "[FAIL]",
        "detail": "model_type = %r" % metrics.get("model_type")})

    rows.append({
        "check": "the five configurations are all scored by Step 11 evaluate()",
        "result": "[OK]", "detail": "%d configurations" % len(runs)})

    table = pd.DataFrame(rows)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table, note="Asserted at run time. A `[FAIL]` here means "
                               "Baseline 4 is not comparable with the other "
                               "four and the table below means nothing.")
    return table, all_ok


def build_summary(table: pd.DataFrame, per_class: pd.DataFrame,
                  baseline4: dict[str, Any], runs: Sequence[dict[str, Any]]
                  ) -> str:
    """The write-up, composed only from the numbers in the tables above."""
    metrics = baseline4.get("metrics") or {}
    pca = metrics.get("pca") or {}
    lines = ["## Step 12 with Baseline 4 (AE -> PCA -> LSTM)", ""]

    def value(label: str, column: str) -> float:
        part = table[table["config"] == label]
        if not len(part) or column not in table.columns:
            return float("nan")
        return float(pd.to_numeric(part[column], errors="coerce").iloc[0])

    names = list(table["config"])
    for column, scope in (("F1 (macro)", "full test set"),
                          ("F1 macro (dup-free)", "duplicate-free subset")):
        values = pd.to_numeric(table[column], errors="coerce")
        if not values.notna().any():
            continue
        best = float(values.max())
        leaders = [n for n, v in zip(names, values) if pd.notna(v)
                   and abs(v - best) < 1e-12]
        tie = " (a tie)" if len(leaders) > 1 else ""
        lines.append("- On the %s the highest macro-F1 is **%.4f**, from %s%s."
                     % (scope, best, ", ".join("`%s`" % n for n in leaders), tie))

    label4 = metrics.get("label", "B4: AE -> PCA -> LSTM")
    label2 = next((r["metrics"]["label"] for r in runs
                   if r["metrics"].get("config") == "baseline2_ae_lstm"), None)
    if label2:
        for column, scope in (("F1 (macro)", "full"),
                              ("F1 macro (dup-free)", "duplicate-free")):
            four, two = value(label4, column), value(label2, column)
            if not (np.isfinite(four) and np.isfinite(two)):
                continue
            direction = ("above" if four > two + 1e-12 else
                         "below" if four < two - 1e-12 else "level with")
            lines.append(
                "- **What PCA did (%s):** Baseline 4 scores %.4f macro-F1, "
                "%s Baseline 2's %.4f -- a difference of %+.4f. These two "
                "configurations are identical apart from the PCA step and "
                "share one encoder, so the gap is PCA's and nothing else's."
                % (scope, four, direction, two, four - two))

    if pca:
        lines.append(
            "- PCA kept **%d of %d** latent dimensions (%s = %s), retaining "
            "**%.1f%%** of the latent variance. Per-component variance: %s."
            % (pca.get("n_components", 0), pca.get("latent_dim", 0),
               pca.get("requested_kind", "setting"), pca.get("requested"),
               100 * float(pca.get("variance_kept", float("nan"))),
               ", ".join("%.3f" % v for v in
                         (pca.get("variance_per_component") or [])[:10])))

    inference = metrics.get("pipeline_inference") or {}
    if inference:
        lines.append(
            "- Baseline 4 end to end: encoder %.2f + PCA %.2f + classifier "
            "%.2f = **%.2f ms per 1,000 flows**. The shared `inference "
            "(ms/1k)` column in the table measures the classifier alone, as "
            "Step 11 does for every configuration."
            % (inference.get("encoder_ms_per_1000_flows", float("nan")),
               inference.get("pca_transform_ms_per_1000_flows", float("nan")),
               inference.get("classifier_ms_per_1000_flows", float("nan")),
               inference.get("end_to_end_ms_per_1000_flows", float("nan"))))

    if len(per_class):
        thin = per_class[pd.to_numeric(per_class.get("test rows"),
                                       errors="coerce") < 20]
        if len(thin):
            lines.append(
                "- Classes with fewer than 20 test rows: %s. Every per-class "
                "number for them moves a long way on one flow changing hands."
                % ", ".join("`%s` (%d)" % (r["Label"], int(r["test rows"]))
                            for _i, r in thin.iterrows()))

    lines += [
        "",
        "**Four caveats.**",
        "",
        "1. **The rare classes have about five test rows each**, so their "
        "per-class F1 is a handful of observations, not a profile.",
        "2. **Duplicates inflate every full-test score.** Roughly half this "
        "dataset's rows share identical feature values, so the duplicate-free "
        "columns are the conservative comparison -- quote those.",
        "3. **PCA here runs on an already-compressed space.** The autoencoder "
        "has already reduced the features to %d dimensions, so PCA may remove "
        "very little, or may remove something the classifier was using. The "
        "components kept and the variance retained are reported above rather "
        "than assumed, and the Baseline 2 comparison is what measures the "
        "effect." % int(pca.get("latent_dim", config.AE_LATENT_DIM)),
        "4. **One split, one seed per configuration** (N_SEEDS = %d). The "
        "differences above are single observations; no significance is claimed "
        "and none should be read in." % int(config.N_SEEDS),
    ]
    return "\n".join(lines)


# ===========================================================================
# 12B.5  SAVE, VERIFY, RUN
# ===========================================================================
def verify_saved(folder: Path, out: dict[str, Any],
                 verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """Read the saved files back from disk and check them."""
    names = config.STEP12B_FILES
    checks: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        checks.append({"check": check, "result": "[OK]" if ok else "[FAIL]",
                       "detail": detail})

    path = folder / names["comparison"]
    if path.exists():
        table = pd.read_csv(path)
        add("the comparison table has one row per configuration",
            len(table) == len(config.EXPERIMENTS_WITH_PCA),
            "%d rows, expected %d" % (len(table),
                                      len(config.EXPERIMENTS_WITH_PCA)))
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
        add("no NaN in the headline metric", bool(table["F1 (macro)"].notna()
                                                  .all()),
            "%d rows checked" % len(table))
    else:
        add("the comparison table has one row per configuration", False,
            "not written")

    baseline = folder / config.BASELINE4["name"]
    for key in ("model", "encoder", "pca", "pca_variance", "metrics",
                "per_class", "predictions", "pca_scree"):
        name = config.BASELINE4_FILES[key]
        add("baseline4/%s exists" % name, (baseline / name).exists(),
            str(baseline / name) if (baseline / name).exists() else "missing")

    path = baseline / config.BASELINE4_FILES["pca_variance"]
    if path.exists():
        variance = pd.read_csv(path)
        total = float(variance["explained_variance_ratio"].sum())
        cumulative = float(variance["cumulative_variance_ratio"].iloc[-1])
        add("the PCA variance table adds up",
            abs(total - cumulative) < 1e-9 and 0 < cumulative <= 1 + 1e-9,
            "%d components, %.4f of the variance kept" % (len(variance),
                                                          cumulative))
        if isinstance(config.PCA_N_COMPONENTS, float):
            add("the variance target is met",
                cumulative >= config.PCA_N_COMPONENTS - 1e-9,
                "kept %.4f, target %.4f" % (cumulative,
                                            config.PCA_N_COMPONENTS))

    untouched = bool(out.get("test_untouched"))
    add("the test files are byte-identical to before this ran", untouched,
        "SHA-256 before and after")

    table = pd.DataFrame(checks)
    all_ok = bool((table["result"] == "[OK]").all())
    if verbose:
        show_table(table, note="Read back from disk after writing. The Step 12 "
                               "files themselves were not rewritten: these "
                               "tables are saved beside them under new names.")
    return table, all_ok


def save_step12b(out: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Write the five-configuration tables and charts, then check them."""
    result: dict[str, Any] = {}
    folder = prepare_output_dir(config.STEP12_OUTPUT_DIR,
                                "the Baseline 4 comparison", verbose=verbose)
    if folder is None:
        result.update({"status": "skipped_not_windows", "ok": False})
        return result

    names = config.STEP12B_FILES
    out["comparison"].to_csv(folder / names["comparison"], index=False)
    out["comparison_marked"].to_csv(folder / names["comparison_marked"],
                                    index=False)
    out["per_class"].to_csv(folder / names["per_class_full"], index=False)
    out["checks"].to_csv(folder / names["consistency"], index=False)
    (folder / names["summary"]).write_text(out["summary"], encoding="utf-8")

    charts: dict[str, str] = {}
    drawn = xp.plot_metric_bars(
        out["comparison"], ["F1 (macro)", "G-mean", "ROC-AUC (macro OvR)"],
        "Full test set, five configurations -- higher is better",
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
    drawn = xp.plot_time_bars(out["comparison"], folder / names["chart_time"],
                              verbose=verbose)
    if drawn:
        charts["time"] = str(drawn)
    drawn = xp.plot_confusion_grid(out["runs"], folder / names["chart_confusion"],
                                   verbose=verbose)
    if drawn:
        charts["confusion"] = str(drawn)
    result["charts"] = charts

    report, ok = verify_saved(folder, out, verbose=verbose)
    report.to_csv(folder / names["report"], index=False)
    result.update({"report": report, "ok": ok, "status": "written",
                   "folder": str(folder)})
    if verbose:
        print("  [%s]  Baseline 4 comparison written to %s"
              % ("OK" if ok else "FAIL", folder))
    return result


def step12b_outputs_exist() -> bool:
    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder is None:
        return False
    return (folder / config.STEP12B_FILES["comparison"]).exists() \
        and baseline4_is_done()


def load_step12b(verbose: bool = True) -> dict[str, Any]:
    folder = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder is None:
        raise PCAExperimentError(
            "the Step 12 folder is not usable here -- %s"
            % config.why_unusable(config.STEP12_OUTPUT_DIR))
    names = config.STEP12B_FILES
    out: dict[str, Any] = {"folder": str(folder), "status": "loaded_existing"}
    for key in ("comparison", "comparison_marked", "per_class_full",
                "consistency", "report"):
        path = folder / names[key]
        if path.exists():
            out[{"per_class_full": "per_class",
                 "consistency": "checks"}.get(key, key)] = pd.read_csv(path)
    path = folder / names["summary"]
    if path.exists():
        out["summary"] = path.read_text(encoding="utf-8")
    path = (folder / config.BASELINE4["name"]
            / config.BASELINE4_FILES["metrics"])
    if path.exists():
        out["baseline4"] = {"metrics": json.loads(path.read_text(
            encoding="utf-8")), "folder": str(path.parent)}
    path = (folder / config.BASELINE4["name"]
            / config.BASELINE4_FILES["pca_variance"])
    if path.exists():
        out["variance"] = pd.read_csv(path)
    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_BASELINE4 = False)"),
            ("folder", str(folder)),
            ("configurations compared", len(out.get("comparison", []))),
        ]), note="Set `OVERWRITE_BASELINE4 = True` in config.py to retrain "
                 "Baseline 4.")
    return out


def run_step12_with_baseline4(overwrite: bool | None = None,
                              epochs: int | None = None,
                              verbose: bool = True) -> dict[str, Any]:
    """Load Step 12's four configurations, run Baseline 4, compare all five.

    Step 12 itself is never re-run: its four metrics.json files are read from
    disk and its own tables are left exactly as it wrote them.
    """
    overwrite = config.OVERWRITE_BASELINE4 if overwrite is None else overwrite
    if not overwrite and step12b_outputs_exist():
        return load_step12b(verbose=verbose)

    if verbose:
        show_title("Step 12 addendum -- Baseline 4: AE -> PCA -> LSTM", level=3)

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
    existing = load_step12_runs(verbose=verbose)
    baseline4 = run_baseline4(y_train, y_test, meta_test, label_mapping,
                              duplicate_flags=flags,
                              seed=config.RANDOM_STATE, epochs=epochs,
                              overwrite=overwrite, verbose=verbose)
    runs = existing + [baseline4]

    checks, checks_ok = checks_table(runs, y_train, y_test, before, baseline4,
                                     verbose=verbose)
    after = xp.test_fingerprint()
    untouched = bool(before and after and all(
        after.get(k, {}).get("sha256") == v["sha256"]
        for k, v in before.items()))

    table = combined_comparison(runs)
    marked = mark_best(table)
    per_class = xp.per_class_f1_table(runs, "full")
    summary = build_summary(table, per_class, baseline4, runs)

    out: dict[str, Any] = {
        "runs": runs, "baseline4": baseline4, "comparison": table,
        "comparison_marked": marked, "per_class": per_class,
        "checks": checks, "checks_ok": checks_ok, "summary": summary,
        "architecture": architecture_table(),
        "test_untouched": untouched,
        "variance": baseline4.get("variance"),
        "pca_info": baseline4.get("pca_info"),
    }
    saved = save_step12b(out, verbose=verbose)
    out.update({"status": saved.get("status", "written"),
                "ok": bool(saved.get("ok") and checks_ok and untouched),
                "saved": saved})
    for key in ("report", "charts", "folder"):
        if key in saved:
            out[key] = saved[key]

    if verbose:
        show_title("Step 12 -- five configurations", level=3)
        show_table(marked, all_columns=True,
                   note="`*` marks the best value in each column. Lower is "
                        "better for the FPR, the times and the parameter "
                        "count. No `*` in a column means every configuration "
                        "tied there.")
        show_note("The four Step 12 rows are read from disk unchanged; only "
                  "Baseline 4 is new. Step 12's own `comparison_table.csv` was "
                  "not rewritten.")
    return out
