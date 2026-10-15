r"""
tests/test_fusion_all.py -- pytest tests for the Step 13 addendum
(adaptive evidence fusion across every approach and baseline).

NEW FILE.  No existing test file, src function, config value or saved output is
modified by these tests.  Nothing is trained and no Keras model is built: every
test works on tiny hand-made probability matrices, and the three model loaders
are exercised through their failure path only, which is the path that must
never train anything.  Everything is written inside tmp_path.
Never C:\\MSCAD2022.

No pathlib patching: the other operating system is simulated through
`config.running_on_windows`, and Windows paths are built with PureWindowsPath.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import json
import os
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import fusion as fu
from src.mscad import fusion_all as fa

LABELS = {0: "Normal", 1: "Port_Scan", 2: "Brute_Force"}
N_CLASSES = 3
BENIGN = [0]

# the real internal keys, so a rename in config.py breaks these tests loudly
KEYS = [k for k, _label in config.STEP12_DISPLAY_ORDER]
PROPOSED, APPROACH2, B1, B2, B3, B4, B5 = KEYS


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    """Every folder this module could write to points inside tmp_path."""
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR", tmp_path / "step12")
    monkeypatch.setattr(config, "STEP13_OUTPUT_DIR", tmp_path / "step13")
    monkeypatch.setattr(config, "STEP13ALL_OUTPUT_DIR", tmp_path / "step13_all")
    monkeypatch.setattr(config, "STEP10B_OUTPUT_DIR",
                        tmp_path / "step10_attention", raising=False)
    monkeypatch.setattr(config, "STEP11B_OUTPUT_DIR",
                        tmp_path / "step11_attention", raising=False)
    monkeypatch.setattr(config, "SUBSET_MAX_MODELS", 4)
    monkeypatch.setattr(config, "SUBSET_METRIC", "macro_f1")
    monkeypatch.setattr(config, "SUBSET_SELECTION", "greedy_forward")


# ===========================================================================
# hand-made probabilities
# ===========================================================================
def rows(*triples) -> np.ndarray:
    """A probability matrix from rows that each sum to 1."""
    out = np.asarray(triples, dtype="float64")
    assert np.allclose(out.sum(axis=1), 1.0), "the test rows must sum to 1"
    return out


def truth_frame(codes) -> pd.DataFrame:
    codes = np.asarray(codes, dtype="int64")
    return pd.DataFrame({"Label_encoded": codes,
                         "Label_binary": (codes != 0).astype("int64")})


def synthetic(n_rows=48, keys=(PROPOSED, B2, B3), seed=0,
              strengths=None):
    """(val_probs, test_probs, inputs) for `keys`, with no model anywhere.

    Each model is given a different per-class accuracy so the reliabilities,
    the weights and the greedy search all have something to separate.
    """
    rng = np.random.default_rng(seed)
    keys = list(keys)
    codes = np.resize(np.arange(N_CLASSES), n_rows)
    strengths = ({k: 0.5 + 0.12 * i for i, k in enumerate(keys)}
                 if strengths is None else strengths)

    def probs_for(key, y):
        strength = strengths[key]
        out = np.full((len(y), N_CLASSES), (1.0 - strength) / (N_CLASSES - 1))
        out[np.arange(len(y)), y] = strength
        noise = rng.uniform(0, 0.02, out.shape)
        out = out + noise
        return out / out.sum(axis=1, keepdims=True)

    y_val_codes = codes.copy()
    y_test_codes = np.resize(np.arange(N_CLASSES), n_rows)
    val_probs = {k: probs_for(k, y_val_codes) for k in keys}
    test_probs = {k: probs_for(k, y_test_codes) for k in keys}

    y_val = truth_frame(y_val_codes)
    y_test = truth_frame(y_test_codes)
    inputs = {
        "y_test": y_test, "y_val": y_val,
        "y_train": truth_frame(np.resize(np.arange(N_CLASSES), 4 * n_rows)),
        "meta_test": pd.DataFrame({"Orig_Row": np.arange(n_rows),
                                   "Source_Capture": ["cap"] * n_rows}),
        "label_names": LABELS, "n_classes": N_CLASSES, "benign_codes": BENIGN,
        "duplicate_flags": np.array([i % 4 == 0 for i in range(n_rows)]),
        "test_probs": test_probs,
    }
    return val_probs, test_probs, inputs


# ===========================================================================
# the keys and the labels
# ===========================================================================
def test_the_configured_keys_are_exactly_the_display_order_keys():
    """A typo here would silently fuse a different set of models."""
    assert list(config.FUSION_ALL_MODELS) == KEYS
    assert fa.model_keys(verbose=False) == KEYS


def test_an_unknown_key_is_refused_by_name():
    with pytest.raises(fa.FusionAllError, match="not keys of"):
        fa.model_keys(["proposed", "approach2_cnn_attention_lstm"], verbose=False)


def test_a_repeated_key_is_refused():
    with pytest.raises(fa.FusionAllError, match="more than once"):
        fa.model_keys(["proposed", "proposed"], verbose=False)


def test_the_labels_come_from_the_display_order():
    assert fa.display_label(PROPOSED) == "Approach 1: FS -> AE -> CNN-LSTM"
    assert fa.display_label(B4) == "B4: AE -> PCA -> LSTM"
    assert fa.short_label(APPROACH2) == "Approach 2"
    assert fa.short_label(B5) == "B5"
    # no spec label ("Proposed: ...") reaches a table
    assert not fa.display_label(PROPOSED).startswith("Proposed")


def test_the_keys_are_reordered_into_the_display_order():
    assert fa.ordered([B5, PROPOSED, B2]) == [PROPOSED, B2, B5]


def test_each_model_keeps_its_own_colour_whatever_the_subset():
    """Colour follows the model, not its position in the current subset."""
    colours = {k: fa.model_colour(k) for k in KEYS}

    assert len(set(colours.values())) == len(KEYS)      # seven distinct hues
    assert fa.model_colour(B5) == colours[B5]


# ===========================================================================
# a missing model is skipped, and the rest still fuse
# ===========================================================================
def test_a_missing_model_is_skipped_with_the_cells_that_produce_it(monkeypatch,
                                                                   capsys):
    """Approach 2 is not trained; the addendum must say so and carry on."""
    def loader(key, verbose=False):
        if key == APPROACH2:
            raise fa.FusionAllError(
                "no saved Approach 2 model at step10_attention")
        return {"name": key, "model": None,
                "X_train": pd.DataFrame({"z": [0.0]}),
                "X_test": pd.DataFrame({"z": [0.0]}), "source": "stub"}
    monkeypatch.setattr(fa, "load_bundle", loader)

    bundles, availability = fa.available_bundles(verbose=True)
    printed = capsys.readouterr().out

    assert APPROACH2 not in bundles
    assert len(bundles) == len(KEYS) - 1
    skipped = availability[~availability["available"]]
    assert list(skipped["key"]) == [APPROACH2]
    note = skipped.iloc[0]["note"]
    assert "Approach 2" in note and "10.2" in note     # which cells to run
    assert "[SKIP]" in printed and "Approach 2" in printed


def test_the_fusion_runs_on_the_models_that_are_there():
    """Four of seven available is a four-model fusion, not an error."""
    keys = [PROPOSED, B2, B3, B5]
    val_probs, test_probs, inputs = synthetic(keys=keys)

    variant = fa.adaptive_variant(keys, val_probs, test_probs, inputs, "FALL")

    assert list(variant["keys"]) == keys
    assert np.asarray(variant["weights"]).shape == (len(inputs["y_test"]), 4)
    assert variant["result"]["full"]["f1_macro"] >= 0


def test_everything_missing_is_an_error_rather_than_a_guess(monkeypatch):
    def loader(key, verbose=False):
        raise fa.FusionAllError("nothing on disk")
    monkeypatch.setattr(fa, "load_bundle", loader)

    with pytest.raises(fa.FusionAllError, match="nothing to fuse"):
        fa.available_bundles(verbose=False)


def test_one_model_is_not_a_fusion(monkeypatch):
    def loader(key, verbose=False):
        if key != PROPOSED:
            raise fa.FusionAllError("nothing on disk")
        return {"name": key, "model": None,
                "X_train": pd.DataFrame({"z": [0.0]}),
                "X_test": pd.DataFrame({"z": [0.0]}), "source": "stub"}
    monkeypatch.setattr(fa, "load_bundle", loader)

    with pytest.raises(fa.FusionAllError, match="at least two models"):
        fa.available_bundles(verbose=False)


@pytest.mark.parametrize("key", [APPROACH2, B4, B5])
def test_no_loader_trains_anything_when_the_files_are_absent(key, tmp_path,
                                                             monkeypatch):
    """The loaders must fail by naming a file, and write nothing at all.

    The message is checked for the name of what is missing, and the folder is
    checked for emptiness afterwards: a loader that trained a replacement would
    have left a model, an encoder or a history behind.
    """
    empty = tmp_path / "empty12"
    empty.mkdir()
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR", empty)
    monkeypatch.setattr(config, "STEP10B_OUTPUT_DIR", tmp_path / "no_step10b",
                        raising=False)

    with pytest.raises(Exception) as problem:
        fa.load_bundle(key, verbose=False)
    message = str(problem.value).lower()

    assert ("missing" in message or "no saved" in message
            or "not usable" in message), message
    assert sorted(p.name for p in empty.rglob("*")) == []
    assert not (tmp_path / "no_step10b").exists()


# ===========================================================================
# diversity, on a hand-made example
# ===========================================================================
def test_the_disagreement_is_the_share_of_differing_predictions():
    # A and B agree on 3 of 4 rows; A and C never agree
    val_probs = {
        PROPOSED: rows((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8),
                       (0.8, 0.1, 0.1)),
        B2: rows((0.7, 0.2, 0.1), (0.2, 0.7, 0.1), (0.1, 0.1, 0.8),
                 (0.1, 0.8, 0.1)),
        B3: rows((0.1, 0.8, 0.1), (0.8, 0.1, 0.1), (0.8, 0.1, 0.1),
                 (0.1, 0.1, 0.8)),
    }

    grid = fa.disagreement_table(val_probs)

    assert grid.loc["Approach 1: FS -> AE -> CNN-LSTM",
                    "Approach 1: FS -> AE -> CNN-LSTM"] == 0.0
    assert grid.loc["Approach 1: FS -> AE -> CNN-LSTM",
                    "B2: AE -> LSTM (no CNN)"] == pytest.approx(0.25)
    assert grid.loc["Approach 1: FS -> AE -> CNN-LSTM",
                    "B3: CNN-LSTM (no AE)"] == pytest.approx(1.0)
    # symmetric, and labelled with the display labels
    assert grid.equals(grid.T)
    assert list(grid.index) == list(grid.columns)


def test_the_risk_correlation_is_the_correlation_of_r_m():
    """Built on fusion.attack_risk, so it is the r_m the fusion combines."""
    val_probs = {
        PROPOSED: rows((0.9, 0.05, 0.05), (0.5, 0.25, 0.25), (0.1, 0.45, 0.45)),
        # r_B2 = 0.75 r_proposed + 0.125 exactly, so the correlation is 1
        B2: rows((0.8, 0.1, 0.1), (0.5, 0.25, 0.25), (0.2, 0.4, 0.4)),
        # r_B3 = 1 - r_proposed exactly, so the correlation is -1
        B3: rows((0.1, 0.45, 0.45), (0.5, 0.25, 0.25), (0.9, 0.05, 0.05)),
    }

    grid = fa.risk_correlation_table(val_probs, BENIGN)

    labels = [fa.display_label(k) for k in (PROPOSED, B2, B3)]
    assert grid.loc[labels[0], labels[0]] == pytest.approx(1.0)
    assert grid.loc[labels[0], labels[1]] == pytest.approx(1.0, abs=1e-9)
    assert grid.loc[labels[0], labels[2]] == pytest.approx(-1.0, abs=1e-9)


def test_a_constant_risk_has_no_correlation_rather_than_zero():
    val_probs = {
        PROPOSED: rows((0.9, 0.05, 0.05), (0.2, 0.4, 0.4)),
        B2: rows((0.5, 0.25, 0.25), (0.5, 0.25, 0.25)),     # r_m constant
    }

    grid = fa.risk_correlation_table(val_probs, BENIGN)

    assert np.isnan(grid.loc[fa.display_label(PROPOSED),
                             fa.display_label(B2)])


# ===========================================================================
# the subset search: validation only, deterministic, and it stops
# ===========================================================================
def test_the_greedy_search_starts_from_the_best_single_model():
    keys = [PROPOSED, B2, B3]
    val_probs, _test, inputs = synthetic(keys=keys)
    y_val_true = np.asarray(inputs["y_val"]["Label_encoded"])
    singles = {k: fa.validation_scores(y_val_true, val_probs[k],
                                       N_CLASSES)["macro_f1"] for k in keys}

    chosen, path = fa.greedy_forward_selection(keys, val_probs, inputs,
                                               verbose=False)

    best = max(singles, key=singles.get)
    assert path.iloc[0]["key added"] == best
    assert best in chosen
    assert path.iloc[0]["decision"] == "best single model on validation"


def test_the_greedy_search_never_sees_the_test_set():
    """Changing every test probability must not move the chosen subset.

    The function takes no test argument at all, which is the real guarantee;
    this asserts the consequence, so a future signature change that let one in
    would fail here.
    """
    keys = [PROPOSED, B2, B3, B5]
    val_probs, test_probs, inputs = synthetic(keys=keys)

    first, first_path = fa.greedy_forward_selection(keys, val_probs, inputs,
                                                    verbose=False)

    # a completely different test set, including its labels
    rng = np.random.default_rng(99)
    flipped = {k: rng.dirichlet(np.ones(N_CLASSES), len(test_probs[k]))
               for k in keys}
    inputs["test_probs"] = flipped
    inputs["y_test"] = truth_frame(
        rng.integers(0, N_CLASSES, len(inputs["y_test"])))

    second, second_path = fa.greedy_forward_selection(keys, val_probs, inputs,
                                                      verbose=False)

    assert first == second
    pd.testing.assert_frame_equal(first_path, second_path)


def test_the_greedy_search_is_deterministic():
    keys = [PROPOSED, B2, B3, B5]
    val_probs, _test, inputs = synthetic(keys=keys)

    runs = [fa.greedy_forward_selection(keys, val_probs, inputs,
                                        verbose=False)[0] for _i in range(3)]

    assert runs[0] == runs[1] == runs[2]


def test_the_greedy_search_stops_at_subset_max_models(monkeypatch):
    monkeypatch.setattr(config, "SUBSET_MAX_MODELS", 2)
    keys = [PROPOSED, B2, B3, B5]
    val_probs, _test, inputs = synthetic(keys=keys)

    chosen, path = fa.greedy_forward_selection(keys, val_probs, inputs,
                                               verbose=False)

    assert len(chosen) <= 2
    if len(chosen) == 2:
        assert "SUBSET_MAX_MODELS = 2 reached" in path.iloc[-1]["decision"]


def test_the_greedy_search_stops_when_nothing_improves():
    """Three identical models: adding the second cannot improve anything."""
    keys = [PROPOSED, B2, B3]
    val_probs, _test, inputs = synthetic(keys=keys)
    same = val_probs[PROPOSED]
    val_probs = {k: same.copy() for k in keys}

    chosen, path = fa.greedy_forward_selection(keys, val_probs, inputs,
                                               verbose=False)

    assert len(chosen) == 1
    assert "no candidate improved" in path.iloc[-1]["decision"]


def test_the_selection_path_records_every_candidate_it_tried():
    keys = [PROPOSED, B2, B3]
    val_probs, _test, inputs = synthetic(keys=keys)

    chosen, path = fa.greedy_forward_selection(keys, val_probs, inputs,
                                               verbose=False)

    tried = path[path["decision"].isin(["tried", "ADDED"])]
    assert len(tried) >= len(keys) - 1
    assert set(path.columns) >= {"step", "model added", "key added",
                                 "models in subset", "validation macro_f1",
                                 "improvement", "decision"}
    # the path names models by their display label
    assert all(any(label.startswith(name.split(":")[0])
                   for label in [fa.display_label(k) for k in KEYS])
               for name in path["model added"] if name != "(none)")


def test_an_unknown_subset_metric_is_refused():
    with pytest.raises(fa.FusionAllError, match="SUBSET_METRIC"):
        fa.subset_metric({"macro_f1": 0.5, "g_mean": 0.4}, metric="accuracy")


def test_the_subset_metric_can_be_the_g_mean(monkeypatch):
    monkeypatch.setattr(config, "SUBSET_METRIC", "g_mean")
    keys = [PROPOSED, B2, B3]
    val_probs, _test, inputs = synthetic(keys=keys)

    _chosen, path = fa.greedy_forward_selection(keys, val_probs, inputs,
                                                verbose=False)

    assert "validation g_mean" in path.columns


# ===========================================================================
# FALL over the three Step 13 models IS the Step 13 fusion
# ===========================================================================
def test_fall_over_the_three_step13_models_reproduces_the_step13_arithmetic():
    """The addendum must call Step 13's maths, not re-implement it.

    The same three models are fused here through `fusion_all.adaptive_variant`
    and, beside it, through the `fusion` calls Step 13's own `run_step13` makes.
    Both the fused distribution and tau have to match exactly.
    """
    keys = list(config.FUSION_MODELS)
    val_probs, test_probs, inputs = synthetic(keys=keys, seed=3)

    mine = fa.adaptive_variant(keys, val_probs, test_probs, inputs, "FALL")

    # --- Step 13, by hand ---------------------------------------------
    y_val_true = np.asarray(inputs["y_val"]["Label_encoded"])
    val_list = [val_probs[k] for k in keys]
    test_list = [test_probs[k] for k in keys]
    val_ctx, test_ctx, names = fu.context_sets(
        config.CONTEXT_MODE, val_list, test_list, inputs["y_train"],
        inputs["meta_test"], LABELS, N_CLASSES)
    _reliability, info = fu.build_reliability(
        {k: val_probs[k] for k in keys}, y_val_true, val_ctx, names,
        N_CLASSES, verbose=False)
    val_fused, _w, _r, _c = fu.fuse_once(val_list, keys, info["lookup"],
                                         info["macro"], val_ctx)
    tau = fu.choose_threshold(
        fu.attack_risk(val_fused, BENIGN),
        np.asarray(inputs["y_val"]["Label_binary"]))["tau"]
    theirs, weights, _r2, _c2 = fu.fuse_once(test_list, keys, info["lookup"],
                                             info["macro"], test_ctx)

    assert mine["fused"] == pytest.approx(theirs, abs=1e-12)
    assert mine["weights"] == pytest.approx(weights, abs=1e-12)
    assert mine["threshold"]["tau"] == pytest.approx(tau, abs=1e-12)
    assert mine["risk"] == pytest.approx(fu.attack_risk(theirs, BENIGN),
                                         abs=1e-12)


def test_the_f3_row_is_read_from_the_saved_metrics_not_recomputed(tmp_path):
    """F3 must be the Step 13 numbers in the write-up, byte for byte."""
    saved = {"fusion": {
        "full": {"accuracy": 0.9876, "precision_macro": 0.80,
                 "recall_macro": 0.70, "f1_macro": 0.7531,
                 "f1_weighted": 0.95, "g_mean": 0.6123,
                 "roc_auc": {"macro": 0.9911}},
        "duplicate_free": {"f1_macro": 0.6789, "g_mean": 0.5432},
        "binary": {"f1": 0.9321, "fpr": 0.0021, "roc_auc": 0.9876,
                   "pr_auc": 0.8765}},
        "fusion_overhead_ms_per_1000_flows": 12.5}

    row = fa.row_from_saved_metrics(fa.F3_NAME, saved)

    assert row["configuration"] == fa.F3_NAME
    assert row["F1 (macro)"] == 0.7531
    assert row["G-mean"] == 0.6123
    assert row["F1 macro (dup-free)"] == 0.6789
    assert row["binary FPR"] == 0.0021
    assert row["binary PR-AUC"] == 0.8765
    assert row["fusion overhead (ms/1k)"] == 12.5


def test_a_step13_metrics_without_a_fusion_block_is_refused():
    with pytest.raises(fa.FusionAllError, match="no \"fusion\" block"):
        fa.row_from_saved_metrics(fa.F3_NAME, {"step": 13})


def test_the_f3_row_is_omitted_rather_than_invented_when_step13_is_absent():
    """`load_step13_metrics` returns None, and the caller leaves the row out."""
    assert fa.load_step13_metrics() is None


# ===========================================================================
# the existing Step 13 outputs are never touched
# ===========================================================================
def fake_step13(tmp_path, f1_macro=0.7531):
    """A minimal `step13\\` folder, as Step 13 would have left it."""
    folder = Path(config.STEP13_OUTPUT_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / config.STEP13_FILES["metrics"]).write_text(json.dumps({
        "step": 13, "models_fused": list(config.FUSION_MODELS),
        "fusion": {"full": {"accuracy": 0.98, "precision_macro": 0.8,
                            "recall_macro": 0.7, "f1_macro": f1_macro,
                            "f1_weighted": 0.95, "g_mean": 0.61,
                            "roc_auc": {"macro": 0.99}},
                   "duplicate_free": {"f1_macro": 0.67, "g_mean": 0.54},
                   "binary": {"f1": 0.93, "fpr": 0.002, "roc_auc": 0.98,
                              "pr_auc": 0.87}},
        "fusion_overhead_ms_per_1000_flows": 12.5,
    }, indent=2), encoding="utf-8")
    (folder / config.STEP13_FILES["comparison"]).write_text(
        "configuration,F1 (macro)\nADAPTIVE EVIDENCE FUSION,%.4f\n" % f1_macro,
        encoding="utf-8")
    return folder


def test_the_step13_fingerprint_covers_every_file(tmp_path):
    folder = fake_step13(tmp_path)

    before = fa.step13_fingerprint()

    assert set(before) == {p.name for p in folder.glob("*")}
    assert all(len(v) == 64 for v in before.values())


def test_writing_the_addendum_leaves_step13_byte_identical(tmp_path, capsys):
    fake_step13(tmp_path)
    before = fa.step13_fingerprint()

    # the addendum writes only into its own folder
    Path(config.STEP13ALL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (Path(config.STEP13ALL_OUTPUT_DIR) / "comparison_table_all.csv").write_text(
        "configuration\nFALL\n", encoding="utf-8")

    table, ok = fa.assert_step13_unchanged(before, verbose=True)
    printed = capsys.readouterr().out

    assert ok
    assert set(table["result"]) == {"[OK]"}
    assert "[OK]  existing Step 13 outputs unchanged" in printed


def test_a_changed_step13_file_is_caught(tmp_path):
    folder = fake_step13(tmp_path)
    before = fa.step13_fingerprint()

    (folder / config.STEP13_FILES["metrics"]).write_text("{}", encoding="utf-8")

    table, ok = fa.assert_step13_unchanged(before, verbose=False)

    assert not ok
    assert "[FAIL]" in set(table["result"])


def test_a_deleted_step13_file_is_caught(tmp_path):
    folder = fake_step13(tmp_path)
    before = fa.step13_fingerprint()

    (folder / config.STEP13_FILES["comparison"]).unlink()

    _table, ok = fa.assert_step13_unchanged(before, verbose=False)

    assert not ok


def test_fusion_models_is_not_read_as_fusion_all_models():
    """Step 13 still fuses three models; the addendum offers seven."""
    assert list(config.FUSION_MODELS) == ["proposed", "baseline2_ae_lstm",
                                          "baseline3_cnn_lstm"]
    assert len(config.FUSION_ALL_MODELS) == 7


# ===========================================================================
# the whole addendum, assembled and written -- no model, no training
# ===========================================================================
def complete_run(keys=(PROPOSED, B2, B3, B5), seed=5, tmp_path=None):
    """Everything `run_step13all` assembles, from synthetic probabilities.

    The model loading and the probability cross-check are the only parts left
    out: those need saved Keras models, and the point here is the arithmetic,
    the tables and the files.
    """
    keys = list(keys)
    val_probs, test_probs, inputs = synthetic(keys=keys, seed=seed)
    y_val_true = np.asarray(inputs["y_val"]["Label_encoded"])

    variants = {"FALL": fa.adaptive_variant(keys, val_probs, test_probs, inputs,
                                            fa.FALL_NAME)}
    selected, path = fa.greedy_forward_selection(keys, val_probs, inputs,
                                                 verbose=False)
    if len(selected) >= 2:
        variants["FSEL"] = fa.adaptive_variant(selected, val_probs, test_probs,
                                               inputs, fa.FSEL_NAME)
    singles = fa.single_variants(keys, val_probs, test_probs, inputs)
    fixed = fa.fixed_variants(keys, val_probs, test_probs, inputs)
    ordered_variants = singles + fixed + list(variants.values())

    comparison = fa.comparison_table(ordered_variants)
    per_class, per_class_free = fa.per_class_tables(ordered_variants, inputs)
    availability = pd.DataFrame([
        {"model": fa.display_label(k), "key": k, "available": k in keys,
         "features into the classifier": 15, "loaded from": "stub",
         "note": "" if k in keys else "%s not available" % fa.display_label(k)}
        for k in KEYS])
    out = {
        "offered_keys": KEYS, "fused_keys": keys, "selected_keys": selected,
        "selection_path": path, "availability": availability,
        "disagreement": fa.disagreement_table(val_probs, keys),
        "risk_corr": fa.risk_correlation_table(val_probs, BENIGN, keys),
        "variants": variants, "results": ordered_variants,
        "comparison": comparison,
        "comparison_marked": fa.mark_best(comparison),
        "per_class": per_class, "per_class_free": per_class_free,
        "weights_by_class": {tag: fa.weights_by_class(v, inputs)
                             for tag, v in variants.items()},
        "fused_predictions": fa.fused_predictions_frame(variants, inputs, keys),
        "reliability": variants["FALL"]["reliability"],
        "alignment_ok": True, "step13_unchanged": True, "step13_before": {},
        "validation_macro_f1": {k: fu.macro_f1(
            y_val_true, np.argmax(val_probs[k], axis=1)) for k in keys},
        "rare_classes": [("Brute_Force", 5)],
        "y_test": inputs["y_test"], "y_val": inputs["y_val"],
        "meta_test": inputs["meta_test"], "label_names": LABELS,
        "n_classes": N_CLASSES, "benign_codes": BENIGN,
        "duplicate_flags": inputs["duplicate_flags"],
        "test_probs": test_probs, "val_probs": val_probs,
    }
    return out, inputs


def test_the_addendum_writes_every_output_and_verifies_it_from_disk(tmp_path,
                                                                   capsys):
    fake_step13(tmp_path)
    out, inputs = complete_run(tmp_path=tmp_path)

    saved = fa.save_step13all(out, verbose=False)
    capsys.readouterr()
    folder = Path(saved["folder"])

    assert saved["status"] == "written"
    report = saved["report"]
    assert set(report["result"]) == {"[OK]"}, report.to_string()
    assert saved["ok"]
    for key in ("availability", "disagreement", "risk_corr", "selection_path",
                "reliability", "thresholds", "fused_predictions",
                "comparison", "comparison_marked", "per_class",
                "per_class_free", "metrics", "summary", "report"):
        assert (folder / config.STEP13ALL_FILES[key]).exists(), key
    # and nothing was written into step13\
    assert sorted(p.name for p in Path(config.STEP13_OUTPUT_DIR).glob("*")) == [
        config.STEP13_FILES["comparison"], config.STEP13_FILES["metrics"]]


def test_the_saved_predictions_carry_the_evidence_for_both_variants(tmp_path):
    out, inputs = complete_run(tmp_path=tmp_path)

    saved = fa.save_step13all(out, verbose=False)
    frame = pd.read_csv(Path(saved["folder"])
                        / config.STEP13ALL_FILES["fused_predictions"])

    assert len(frame) == len(inputs["y_test"])
    assert list(frame["Orig_Row"]) == list(inputs["meta_test"]["Orig_Row"])
    for key in out["fused_keys"]:
        assert "r_%s" % fa.short_label(key) in frame.columns
    for tag in out["variants"]:
        weights = [c for c in frame.columns if c.startswith("w_%s_" % tag)]
        assert len(weights) == len(out["variants"][tag]["keys"])
        assert frame[weights].sum(axis=1).to_numpy() == pytest.approx(
            np.ones(len(frame)), abs=1e-6)
        fused = [c for c in frame.columns if c.startswith("P_%s_" % tag)]
        assert len(fused) == N_CLASSES
        assert frame[fused].sum(axis=1).to_numpy() == pytest.approx(
            np.ones(len(frame)), abs=1e-6)
        risk = frame["R_%s" % tag].to_numpy()
        assert risk.min() >= 0 and risk.max() <= 1
        assert set(frame["decision_%s" % tag].unique()) <= {0, 1}


def test_the_verification_catches_weights_that_do_not_sum_to_one(tmp_path):
    """The check reads the CSV, so a corrupted file must fail it."""
    out, _inputs = complete_run(tmp_path=tmp_path)
    saved = fa.save_step13all(out, verbose=False)
    folder = Path(saved["folder"])

    path = folder / config.STEP13ALL_FILES["fused_predictions"]
    frame = pd.read_csv(path)
    column = [c for c in frame.columns if c.startswith("w_FALL_")][0]
    frame[column] = frame[column] + 0.5
    frame.to_csv(path, index=False)

    report, ok = fa.verify_saved(folder, out, verbose=False)

    assert not ok
    failed = report[report["result"] == "[FAIL]"]
    assert any("weights sum to 1" in c for c in failed["check"])


def test_the_tables_use_the_display_labels_and_mark_the_best(tmp_path):
    out, _inputs = complete_run(tmp_path=tmp_path)
    comparison, marked = out["comparison"], out["comparison_marked"]

    names = list(comparison["configuration"])
    assert fa.display_label(PROPOSED) in names
    assert fa.FALL_NAME in names
    assert not any(str(n).startswith("single: ") for n in names)
    assert not any(str(n).startswith("Proposed:") for n in names)
    # exactly one best marked per column, on every tied row
    for column in ("F1 (macro)", "binary FPR"):
        values = pd.to_numeric(comparison[column], errors="coerce")
        target = values.min() if column in fa.LOWER_IS_BETTER else values.max()
        starred = [i for i, v in enumerate(marked[column])
                   if str(v).endswith("*")]
        expected = [i for i, v in enumerate(values)
                    if not pd.isna(v) and abs(v - target) <= 1e-9]
        assert starred == expected


def test_the_per_class_tables_cover_the_full_and_duplicate_free_views(tmp_path):
    out, inputs = complete_run(tmp_path=tmp_path)

    full, free = out["per_class"], out["per_class_free"]

    assert list(full["Label"]) == [LABELS[c] for c in range(N_CLASSES)]
    assert list(free["Label"]) == list(full["Label"])
    assert full["test rows"].sum() == len(inputs["y_test"])
    assert free["test rows"].sum() == int(
        (~np.asarray(inputs["duplicate_flags"])).sum())
    assert fa.FALL_NAME in full.columns and fa.FALL_NAME in free.columns


def test_the_weight_table_is_the_mean_weight_per_model_per_true_class(tmp_path):
    out, inputs = complete_run(tmp_path=tmp_path)

    frame = out["weights_by_class"]["FALL"]

    assert set(frame["true class"]) == {LABELS[c] for c in range(N_CLASSES)}
    assert set(frame["model"]) == {fa.display_label(k)
                                   for k in out["variants"]["FALL"]["keys"]}
    for _name, part in frame.groupby("true class"):
        # the weights are a partition of 1 per flow, so their means are too;
        # 5e-6 because `fusion.weights_summary_table` rounds each mean to 6
        # places and up to seven of them are added up here
        assert part["mean weight"].sum() == pytest.approx(1.0, abs=5e-6)


def test_the_summary_is_written_from_the_computed_numbers(tmp_path):
    fake_step13(tmp_path)
    out, _inputs = complete_run(tmp_path=tmp_path)
    out["comparison"] = fa.comparison_table(
        out["results"], [fa.row_from_saved_metrics(
            fa.F3_NAME, fa.load_step13_metrics())])

    text = fa.build_summary(out)

    assert "13.8.5" in text
    for name in ("F3", "FALL"):
        assert name in text
    assert "%.4f" % float(pd.to_numeric(
        out["comparison"].set_index("configuration").loc[
            fa.F3_NAME, "F1 (macro)"])) in text
    assert "validation" in text.lower()
    assert "Step 14 explains the original three-model fusion" in text
    assert "RANDOM_STATE = %d" % config.RANDOM_STATE in text
    assert "Brute_Force (5 test rows)" in text


def test_the_summary_names_the_models_that_were_skipped(tmp_path):
    out, _inputs = complete_run(keys=(PROPOSED, B2, B3), tmp_path=tmp_path)

    text = fa.build_summary(out)

    assert "Skipped, not trained on this machine" in text
    assert fa.display_label(B4) in text
    assert "Nothing was trained to fill the gap" in text


# ===========================================================================
# the guards
# ===========================================================================
def test_the_addendum_refuses_a_windows_path(tmp_path, monkeypatch, capsys,
                                             assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP13ALL_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step13_all"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)
    out, _inputs = complete_run(tmp_path=tmp_path)

    saved = fa.save_step13all(out, verbose=False)

    assert saved["status"] == "skipped_not_windows"
    assert saved["ok"] is False
    assert fa.step13all_outputs_exist() is False
    capsys.readouterr()
    assert_no_leaked_output("save_step13all")


def test_the_output_dir_is_under_the_data_root_not_the_repo():
    assert config.PROJECT_ROOT not in config.STEP13ALL_OUTPUT_DIR.parents
    assert config.STEP13ALL_OUTPUT_DIR.name == "step13_all"


def test_a_finished_addendum_is_read_back_rather_than_recomputed(tmp_path,
                                                                 capsys):
    out, _inputs = complete_run(tmp_path=tmp_path)
    fa.save_step13all(out, verbose=False)
    capsys.readouterr()

    assert fa.step13all_outputs_exist() is True
    loaded = fa.load_step13all(verbose=False)

    assert loaded["status"] == "loaded_existing"
    assert len(loaded["comparison"]) == len(out["comparison"])
    assert loaded["metrics"]["fused"] == [fa.display_label(k)
                                          for k in out["fused_keys"]]
    assert loaded["summary"].startswith("### 13.8.5")
    assert set(loaded["disagreement"].index) == set(
        loaded["disagreement"].columns)


def test_the_resumable_path_rechecks_the_step13_fingerprint(tmp_path, capsys):
    """Loading a finished addendum must still answer the add-only question."""
    fake_step13(tmp_path)
    out, _inputs = complete_run(tmp_path=tmp_path)
    out["step13_before"] = fa.step13_fingerprint()
    fa.save_step13all(out, verbose=False)
    capsys.readouterr()

    loaded = fa.load_step13all(verbose=False)

    assert loaded["step13_unchanged"] is True
    assert set(loaded["step13_unchanged_table"]["result"]) == {"[OK]"}

    # and it notices a Step 13 file that moved since the addendum was written
    (Path(config.STEP13_OUTPUT_DIR)
     / config.STEP13_FILES["metrics"]).write_text("{}", encoding="utf-8")

    again = fa.load_step13all(verbose=False)

    assert again["step13_unchanged"] is False
