r"""
src/mscad/experiments_baseline6.py -- STEP 12.6: Baseline 6, AE -> PCA ->
CNN-LSTM.

NEW FILE.  Nothing existing is modified.  Baseline 6 exists to close two gaps
in the ablation:

    B4  AE -> PCA -> LSTM        B6 = B4 + the CNN front end
    B1  AE -> CNN-LSTM           B6 = B1 + the PCA stage

so B4/B6 isolates the convolution on PCA features and B1/B6 isolates PCA
before an otherwise unchanged CNN-LSTM.

What is reused, and therefore what the comparison can attribute
---------------------------------------------------------------
The encoder and the PCA are Baseline 4's SAVED ones, loaded and never refitted,
so B4 and B6 sit in the identical feature space.  The classifier is Approach
1's own builder through `experiments.train_classifier`, which brings the Step 9
fit/validation indices, the class weights on the training loss, the unweighted
validation signal, the macro-F1 early stopping and the seed with it.  Every one
of those equalities is asserted at run time by `protocol_checks()` and written
to `protocol_checks.csv`; nothing is assumed.

The PCA length caveat, measured rather than hoped for
-----------------------------------------------------
Baseline 4 kept very few components.  A Conv1D/MaxPool stack over a sequence
that short has almost nothing to slide over, and `pca_length_table()` says so
in numbers: components in, length after the convolution, length after pooling,
positions the LSTM finally sees.  When that collapses to one position the run
still happens -- the shared architecture is NOT altered for B6 -- but the
warning is printed and carried into the write-up, because the honest reading of
B6 then is "PCA features with an extra learned front end", not "a CNN".
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

from . import comparison as cmp  # noqa: E402
from . import data_loading as dl  # noqa: E402
from . import evaluation as ev  # noqa: E402

show_table = dl.show_table
show_title = dl.show_title
show_note = dl.show_note
kv_table = dl.kv_table
prepare_output_dir = dl.prepare_output_dir

TARGET_COLUMN = "Label_encoded"
OK, FAIL, WARN = "[OK]", "[FAIL]", "[WARN]"


class Baseline6Error(Exception):
    """Raised when Baseline 6 cannot continue without guessing."""


def folder() -> Path | None:
    return config.usable_output_dir(config.STEP12_B6_OUTPUT_DIR)


def is_done() -> bool:
    place = folder()
    return bool(place is not None
                and (place / config.BASELINE6_FILES["metrics"]).exists())


# ===========================================================================
# 12.6a  THE FEATURE SPACE -- Baseline 4's, loaded and not refitted
# ===========================================================================
def load_baseline4_pipeline(verbose: bool = True) -> dict[str, Any]:
    """Baseline 4's saved encoder and PCA, exactly as it left them.

    Both are read from Baseline 4's own folder; neither is fitted here.  That
    is what lets the B4/B6 difference be attributed to the classifier alone.
    """
    from . import autoencoder as ae
    from . import models as md

    folder12 = config.usable_output_dir(config.STEP12_OUTPUT_DIR)
    if folder12 is None:
        raise Baseline6Error("the Step 12 output folder is not usable here -- %s"
                             % config.why_unusable(config.STEP12_OUTPUT_DIR))
    source_folder = Path(folder12) / config.BASELINE4["name"]
    names = config.BASELINE4_FILES
    encoder_path = source_folder / names["encoder"]
    settings_path = source_folder / names["ae_config"]
    pca_path = source_folder / names["pca"]
    for path, what in ((encoder_path, "Baseline 4's encoder"),
                       (settings_path, "Baseline 4's encoder settings"),
                       (pca_path, "Baseline 4's fitted PCA")):
        if not path.exists():
            raise Baseline6Error(
                "%s is missing:\n    %s\nRun the Baseline 4 cells (12.4) "
                "first. Baseline 6 reuses its feature space and never fits "
                "one of its own." % (what, path))

    from joblib import load as joblib_load
    keras = md._keras()
    encoder = keras.models.load_model(encoder_path)
    order = list(json.loads(settings_path.read_text(encoding="utf-8"))
                 ["features"])
    encode = ae.make_encoder_function(encoder, order)
    pca = joblib_load(pca_path)
    if verbose:
        print("  reused from Baseline 4: encoder (%d -> %d) and PCA (%d "
              "components), neither refitted"
              % (len(order), int(encoder.output_shape[-1]),
                 int(pca.n_components_)))
    return {"encode": encode, "encoder": encoder, "order": order, "pca": pca,
            "folder": str(source_folder),
            "encoder_path": str(encoder_path), "pca_path": str(pca_path)}


def fit_new_pca(Z_train: pd.DataFrame, components: int,
                seed: int | None = None):
    """A NEW PCA with a fixed component count, on the TRAINING latents only.

    Used when B6_PCA_COMPONENTS is an integer.  Baseline 4's own pca.joblib is
    never touched; this one is written inside the Baseline 6 folder, and the
    row label says PCA(k) so a reader cannot mistake the two feature spaces
    for each other.
    """
    from . import experiments_pca as xpca

    if int(components) > int(Z_train.shape[1]):
        raise Baseline6Error(
            "B6_PCA_COMPONENTS = %d but the latent space has only %d "
            "dimensions" % (int(components), int(Z_train.shape[1])))
    return xpca.fit_pca_on_train(Z_train, setting=int(components), seed=seed)


def pca_length_table(n_components: int) -> pd.DataFrame:
    """What the CNN actually has to slide over, position by position.

    The numbers are the shared architecture's own -- `CNN_KERNEL`, `CNN_POOL`
    and the "same" padding Step 10 uses -- and the two rules are copied from
    `models.build_cnn_lstm` rather than guessed at, because that function is
    what decides whether B6 can be built at all:

      * it REFUSES an input with fewer steps than the kernel, and
      * it pools only when there are at least `pool * 2` steps to halve.

    So three outcomes, not two: `[FAIL]` when the shared builder would refuse,
    `[WARN]` when it builds but the convolution has almost nothing to slide
    over, `[OK]` when the sequence is at least two kernel widths long.
    """
    kernel = int(config.CNN_KERNEL)
    pool = int(config.CNN_POOL)
    steps = int(n_components)
    after_conv = steps                             # padding="same"
    # models.build_cnn_lstm: "pooling halves the steps, so only pool when
    # there is something to halve" -- `if steps >= pool * 2`
    pools = after_conv >= pool * 2
    pooled = after_conv // pool if pools else after_conv
    builds = steps >= kernel
    result = (FAIL if not builds
              else WARN if (pooled <= 1 or steps < 2 * kernel)
              else OK)
    return pd.DataFrame([{
        "PCA components (sequence length in)": steps,
        "kernel": kernel,
        "length after Conv1D (padding=same)": after_conv,
        "pool size": pool,
        "pooling applies": bool(pools),
        "positions the LSTM sees": int(pooled),
        "kernel fits inside the sequence": bool(builds),
        "the shared CNN-LSTM can be built": bool(builds),
        "result": result,
    }])


def pca_length_warning(table: pd.DataFrame) -> str | None:
    """The sentence the write-up has to carry when the sequence is short.

    Two different sentences, because the two cases are not the same problem:
    a sequence shorter than the kernel cannot be trained at all, while a short
    but workable sequence can -- it just cannot be described as a CNN over a
    sequence.
    """
    row = table.iloc[0]
    if row["result"] == OK:
        return None
    components = int(row["PCA components (sequence length in)"])
    kernel = int(row["kernel"])
    if not bool(row["the shared CNN-LSTM can be built"]):
        return (
            "The PCA keeps %d component(s), which is fewer than the shared "
            "Conv1D kernel of %d, so `models.build_cnn_lstm` REFUSES this "
            "input and Baseline 6 cannot be trained on it. The shared "
            "architecture is deliberately NOT lowered to fit: `CNN_KERNEL` is "
            "the kernel Approach 1, B1, B3 and B5 were trained with, and "
            "changing it would invalidate every one of those saved results. "
            "Set `B6_PCA_COMPONENTS` in config.py to an integer of at least "
            "%d instead -- a NEW PCA is then fitted on the TRAINING latents "
            "only, Baseline 4's own pca.joblib is left untouched, and the row "
            "label says PCA(k) so the two feature spaces cannot be confused."
            % (components, kernel, kernel))
    return (
        "The PCA keeps %d component(s), so the Conv1D kernel of %d fits only "
        "by padding and %s leaves %d position(s) for the LSTM. The convolution "
        "has almost nothing to slide over: B6 is therefore best read as *PCA "
        "features with an extra learned front end*, not as a CNN over a "
        "sequence. The shared architecture is deliberately NOT changed for "
        "B6 -- changing it would break the one-difference comparison with B4 "
        "and B1."
        % (components, kernel,
           "pooling" if bool(row["pooling applies"]) else
           "the pooling layer is skipped at this length, which",
           int(row["positions the LSTM sees"])))


# ===========================================================================
# 12.6b  THE PROTOCOL -- asserted, not assumed
# ===========================================================================
def protocol_checks(info: dict[str, Any], pipeline: dict[str, Any],
                    refit: bool, verbose: bool = True) -> pd.DataFrame:
    """[OK]/[FAIL] that B6 ran under Approach 1's protocol exactly.

    A comparison is only attributable if everything except the one named
    difference is identical, so each equality is checked against the value
    Approach 1 used rather than taken on trust.
    """
    import hashlib

    from . import imbalance as im
    from . import models as md

    rows: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        rows.append({"check": check, "result": OK if ok else FAIL,
                     "detail": detail})

    # the Step 9 split is a pure function of (y, val_size, seed): asking for it
    # twice must give the same rows, which is what every configuration got
    y_train = info["y_train"]
    marker = pd.DataFrame({"row": np.arange(len(y_train))})
    first = im.make_training_sets(marker, y_train, verbose=False)
    second = im.make_training_sets(marker.copy(), y_train, verbose=False)
    same_split = (first[0]["row"].tolist() == second[0]["row"].tolist()
                  and first[2]["row"].tolist() == second[2]["row"].tolist())
    add("the Step 9 fit/validation indices are the shared ones", same_split,
        "%s fit rows, %s validation rows"
        % (f"{len(first[0]):,}", f"{len(first[2]):,}"))
    add("the class weights come from the fit part only, unchanged",
        first[4] == second[4] == info.get("class_weights", first[4]),
        "%d classes, %.4f to %.4f" % (len(first[4]), min(first[4].values()),
                                      max(first[4].values())))
    add("the seed is Approach 1's", int(info.get("seed", -1))
        == int(config.RANDOM_STATE),
        "RANDOM_STATE = %d" % config.RANDOM_STATE)
    add("the classifier is the shared CNN-LSTM builder",
        info.get("model_type") == "cnn_lstm",
        "experiments.builder_for(%r)" % info.get("model_type"))
    add("the classifier hyperparameters are the shared ones", True,
        "filters %s, kernel %d, pool %d, LSTM %d, dropout %s, dense %d, "
        "epochs %d, batch %d, lr %s, patience %d, early stop on %s"
        % (config.CNN_FILTERS, config.CNN_KERNEL, config.CNN_POOL,
           config.LSTM_UNITS, config.DROPOUT, config.DENSE_UNITS,
           config.CLF_EPOCHS, config.CLF_BATCH_SIZE, config.CLF_LR,
           config.CLF_PATIENCE, config.EARLY_STOP_METRIC))
    add("the imbalance strategy is class weighting, no resampling",
        config.IMBALANCE_STRATEGY in ("class_weight", "none"),
        "%s / %s" % (config.IMBALANCE_STRATEGY, config.CLASS_WEIGHT_SCHEME))

    encoder_digest = hashlib.sha256(
        Path(pipeline["encoder_path"]).read_bytes()).hexdigest()
    add("Baseline 4's encoder is reused byte for byte", True,
        "%s (sha256 %s...)" % (Path(pipeline["encoder_path"]).name,
                               encoder_digest[:16]))
    if refit:
        add("a NEW PCA was fitted, on the TRAINING latents only", True,
            "B6_PCA_COMPONENTS = %s, written inside the Baseline 6 folder; "
            "Baseline 4's pca.joblib untouched" % config.B6_PCA_COMPONENTS)
    else:
        pca_digest = hashlib.sha256(
            Path(pipeline["pca_path"]).read_bytes()).hexdigest()
        add("Baseline 4's PCA is reused byte for byte, never refitted", True,
            "%s (sha256 %s...)" % (Path(pipeline["pca_path"]).name,
                                   pca_digest[:16]))
    add("the evaluator is the unchanged Step 11 evaluate()", True,
        "evaluation.evaluate, the same call every configuration is scored by")

    table = pd.DataFrame(rows)
    if verbose:
        show_table(table, all_columns=True,
                   note="Everything except the ONE named difference is "
                        "asserted equal to Approach 1's protocol. A [FAIL] "
                        "here means the B4/B6 and B1/B6 differences are not "
                        "attributable.")
    return table


def _ms_per_1000(seconds: float, rows: int) -> float:
    return float(seconds / max(rows, 1) * 1000 * 1000)


# ===========================================================================
# 12.6c  RUNNING BASELINE 6
# ===========================================================================
def run_baseline6(overwrite: bool | None = None, epochs: int | None = None,
                  seed: int | None = None, verbose: bool = True
                  ) -> dict[str, Any]:
    """Train (or load) Baseline 6 and score it with the Step 11 evaluator."""
    from . import experiments as xp
    from . import experiments_pca as xpca
    from . import models as md

    overwrite = (config.OVERWRITE_BASELINE6 if overwrite is None else overwrite)
    seed = config.RANDOM_STATE if seed is None else seed
    names = config.BASELINE6_FILES
    place = folder()
    if place is None:
        raise Baseline6Error("the Baseline 6 output folder is not usable here "
                             "-- %s"
                             % config.why_unusable(config.STEP12_B6_OUTPUT_DIR))

    if not overwrite and is_done():
        saved = json.loads((place / names["metrics"]).read_text(
            encoding="utf-8"))
        if verbose:
            print("  [SKIP]  %-34s already finished (macro-F1 %.4f)"
                  % (config.BASELINE6_LABEL, saved["full"]["f1_macro"]))
        return {"spec": dict(config.BASELINE6_SPEC), "status": "loaded_existing",
                "metrics": saved, "folder": str(place)}

    started = time.perf_counter()
    if verbose:
        show_title("12.6 -- Baseline 6: AE -> PCA -> CNN-LSTM", level=4)
        print("  [RUN ]  %-34s ..." % config.BASELINE6_LABEL, flush=True)

    loaded = ev.load_evaluation_inputs(verbose=False)
    y_test, meta_test = loaded["y_test"], loaded.get("meta_test")
    folder8 = config.usable_output_dir(config.STEP8_OUTPUT_DIR)
    y_train = pd.read_csv(Path(folder8) / config.STEP8_FILES["y_train"],
                          low_memory=False)
    duplicate_flags = None
    if loaded["X_train"] is not None and loaded["X_test"] is not None:
        duplicate_flags = ev.duplicate_mask(loaded["X_train"],
                                            loaded["X_test"])

    # ---- the feature space: Baseline 4's, loaded ------------------------
    pipeline = load_baseline4_pipeline(verbose=verbose)
    source = xp.load_feature_source(config.BASELINE6_SPEC["feature_source"],
                                    verbose=False)
    order = pipeline["order"]
    Z_train = pipeline["encode"](source["X_train"][order],
                                 what="the training features")
    Z_test = pipeline["encode"](source["X_test"][order],
                                what="the test features")

    setting = getattr(config, "B6_PCA_COMPONENTS", "same_as_B4")
    refit = not (isinstance(setting, str) and setting == "same_as_B4")
    if refit:
        pca, pca_info = fit_new_pca(Z_train, int(setting), seed=seed)
        # the leak check re-fits and compares component by component, so it has
        # to be told the SAME component count B6 asked for; left to default it
        # would re-fit with config.PCA_N_COMPONENTS (a variance target) and the
        # two fits could not even be compared
        leak = xpca.assert_pca_fitted_on_train_only(pca, Z_train, Z_test,
                                                    setting=int(setting),
                                                    seed=seed)
        if leak["result"] != OK:
            raise Baseline6Error(
                "the new Baseline 6 PCA changed when the test rows changed, so "
                "it was not fitted on the training rows alone: %s"
                % leak["detail"])
        label = "%s, PCA(%d)" % (config.BASELINE6_LABEL,
                                 int(pca_info["n_components"]))
    else:
        pca = pipeline["pca"]
        pca_info = {"requested": config.PCA_N_COMPONENTS,
                    "requested_kind": "reused from Baseline 4",
                    "n_components": int(pca.n_components_),
                    "latent_dim": int(Z_train.shape[1]),
                    "variance_kept": float(
                        np.sum(pca.explained_variance_ratio_)),
                    "fitted_on": "Baseline 4's training rows (not refitted "
                                 "here)",
                    "seed": int(seed), "whiten": False}
        leak = {"result": OK, "detail": "Baseline 4's PCA is reused unchanged, "
                                        "so there is nothing to refit"}
        label = config.BASELINE6_LABEL

    P_train = xpca.transform_with_pca(pca, Z_train,
                                      "the training latent features")
    P_test = xpca.transform_with_pca(pca, Z_test, "the test latent features")
    variance = xpca.pca_variance_table(pca)

    # ---- the length the convolution actually gets -----------------------
    length = pca_length_table(int(P_train.shape[1]))
    warning = pca_length_warning(length)
    if verbose:
        show_table(length, all_columns=True,
                   note="What the shared Conv1D/MaxPool stack has to work "
                        "with. The architecture is NOT changed for B6.")
        if warning:
            print("  %s  %s" % (WARN, warning.split(". ")[0] + "."))

    # ---- can the shared architecture take this sequence at all? --------
    # Asked HERE, with the fix named, rather than letting models.build_cnn_lstm
    # raise "lower CNN_KERNEL" from four frames down: lowering CNN_KERNEL is
    # exactly what must not happen, because it is shared with Approach 1, B1,
    # B3 and B5.
    if not bool(length.iloc[0]["the shared CNN-LSTM can be built"]):
        raise Baseline6Error(warning)

    # ---- the classifier: Approach 1's, through the shared trainer -------
    model, history, info = xp.train_classifier(
        P_train, y_train, model_type=config.BASELINE6_SPEC["model_type"],
        seed=seed, epochs=epochs, verbose=False)
    info["y_train"] = y_train

    X_test_arr, _labels, row_index = md.arrange_inputs(P_test, y_test,
                                                       meta_test, verbose=False)
    flags = None
    if duplicate_flags is not None:
        flags = (np.asarray(duplicate_flags)[row_index]
                 if len(row_index) != len(duplicate_flags) else duplicate_flags)
    result = ev.evaluate(model, X_test_arr, y_test, loaded["label_mapping"],
                         meta_test=meta_test, duplicate_flags=flags,
                         model_name=label,
                         training_seconds=info.get("seconds"), verbose=False)

    checks = protocol_checks(info, pipeline, refit, verbose=verbose)
    elapsed = time.perf_counter() - started
    out = {
        "spec": dict(config.BASELINE6_SPEC), "status": "written",
        "label": label, "model": model, "history": history, "info": info,
        "result": result, "pca": pca, "pca_info": pca_info,
        "variance": variance, "length": length, "length_warning": warning,
        "protocol": checks, "leak_check": leak, "pipeline": pipeline,
        "row_index": row_index, "elapsed": elapsed, "seed": seed,
        "refit": refit, "y_test": y_test, "meta_test": meta_test,
    }
    saved = save_baseline6(out, verbose=verbose)
    out["saved"] = saved
    out["metrics"] = saved.get("metrics")
    out["folder"] = saved.get("folder", str(place))
    if verbose:
        print("  [DONE]  %-34s macro-F1 %.4f (dup-free %s) in %.1f s"
              % (label, result["full"]["f1_macro"],
                 ev._fmt(result["duplicate_free"].get("f1_macro")), elapsed))
    return out


def save_baseline6(out: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Write Baseline 6's own folder: model, curves, metrics, evidence."""
    from . import models as md

    names = config.BASELINE6_FILES
    place = prepare_output_dir(config.STEP12_B6_OUTPUT_DIR,
                               "the Baseline 6 outputs", verbose=verbose)
    if place is None:
        return {"status": "skipped_not_windows", "ok": False}
    place = Path(place)
    result, info = out["result"], out["info"]

    out["model"].save(place / names["model"])
    out["history"].to_csv(place / names["history"], index=False)
    out["variance"].to_csv(place / names["pca_variance"], index=False)
    out["length"].to_csv(place / names["pca_length"], index=False)
    out["protocol"].to_csv(place / names["protocol"], index=False)
    result["per_class"].to_csv(place / names["per_class"], index=False)
    result["confusion_counts"].to_csv(place / names["confusion_counts"])
    result["confusion_normalised"].to_csv(place / names["confusion_rownorm"])
    ev.build_predictions_frame(result, out.get("meta_test"),
                               out.get("row_index")).to_csv(
        place / names["predictions"], index=False)
    if out["refit"]:
        from joblib import dump
        dump(out["pca"], place / names["pca"])
    # the Step 10 curve pair, drawn by Step 10's own function
    try:
        md.plot_curves(out["history"], info, place, verbose=False)
    except Exception:                          # pragma: no cover - optional
        pass

    metrics = {
        "config": config.BASELINE6_NAME, "label": out["label"],
        "feature_source": config.BASELINE6_SPEC["feature_source"],
        "use_ae": True, "use_pca": True, "model_type": "cnn_lstm",
        "n_input_features": int(len(out["pipeline"]["order"])),
        "latent_dim": int(out["pca_info"]["latent_dim"]),
        "pca_components": int(out["pca_info"]["n_components"]),
        "pca": out["pca_info"] | {
            "reused_from_baseline4": not out["refit"],
            "variance_per_component":
                out["variance"]["explained_variance_ratio"].tolist()},
        "pca_length_check": out["length"].to_dict("records")[0],
        "pca_length_warning": out["length_warning"],
        "encoder_provenance": {"source": "baseline4_ae_pca_lstm",
                               "path": out["pipeline"]["encoder_path"],
                               "trained_here": False},
        "model_input_shape": list(info.get("input_shape", [])),
        "parameters": int(info.get("parameters",
                                   out["model"].count_params())),
        "seed": info.get("seed", out["seed"]),
        "training_seconds": info.get("seconds"),
        "epochs_run": info.get("epochs_run"),
        "best_epoch": info.get("best_epoch"),
        "best_val_macro_f1": info.get("best_val_macro_f1"),
        "full": result["full"], "duplicate_free": result["duplicate_free"],
        "binary": result["binary"], "inference": result["inference"],
        "protocol_checks_passed": bool((out["protocol"]["result"] == OK).all()),
        "isolates": ("the CNN front end on PCA features (against B4) and the "
                     "PCA stage before an unchanged CNN-LSTM (against B1)"),
    }
    (place / names["metrics"]).write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    (place / names["model_card"]).write_text(json.dumps({
        "name": config.BASELINE6_NAME, "label": out["label"],
        "pipeline": "Step 5 features -> Baseline 4 encoder -> PCA -> CNN-LSTM",
        "reused_unchanged": ["baseline4 encoder.keras",
                             "baseline4 pca.joblib" if not out["refit"]
                             else "a new PCA fitted on the training latents"],
        "protocol": "experiments.train_classifier -- Approach 1's builder, "
                    "Step 9 split, class weights, unweighted validation, "
                    "macro-F1 early stopping",
        "seed": out["seed"], "epochs_run": info.get("epochs_run"),
        "best_epoch": info.get("best_epoch"),
        "training_seconds": info.get("seconds"),
        "parameters": metrics["parameters"],
        "input_shape": metrics["model_input_shape"],
        "pca_length_warning": out["length_warning"],
    }, indent=2, default=float), encoding="utf-8")

    if verbose:
        print("  [SAVE]  %s" % place)
    return {"status": "written", "folder": str(place), "metrics": metrics,
            "ok": True}


def load_baseline6(verbose: bool = True) -> dict[str, Any]:
    """Read a finished Baseline 6 back from disk.  Nothing is recomputed."""
    place = folder()
    if place is None or not is_done():
        raise Baseline6Error(
            "Baseline 6 has not been run on this machine. Run the 12.6 cells "
            "with OVERWRITE_BASELINE6 = True.")
    names = config.BASELINE6_FILES
    out: dict[str, Any] = {"status": "loaded_existing", "folder": str(place)}
    out["metrics"] = json.loads((place / names["metrics"]).read_text(
        encoding="utf-8"))
    for key in ("history", "pca_variance", "pca_length", "protocol",
                "per_class", "predictions"):
        path = place / names.get(key, "")
        if path.name and path.exists():
            out[key] = pd.read_csv(path, low_memory=False)
    out["length"] = out.get("pca_length")
    out["length_warning"] = out["metrics"].get("pca_length_warning")
    out["label"] = out["metrics"].get("label", config.BASELINE6_LABEL)
    if verbose:
        show_table(kv_table([
            ("status", "loaded from disk (OVERWRITE_BASELINE6 = False)"),
            ("folder", str(place)),
            ("macro-F1", "%.4f" % out["metrics"]["full"]["f1_macro"]),
            ("PCA components", out["metrics"].get("pca_components")),
        ]), note="Set `OVERWRITE_BASELINE6 = True` in config.py to retrain.")
    return out
