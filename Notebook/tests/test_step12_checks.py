r"""
tests/test_step12_checks.py -- pytest tests for src/mscad/step12_checks.py.

NEW FILE.  Covers 12.0 (provenance and the three consistency questions), 12.0b
(the retraining lock) and 12.2 (the sanity baselines, including the
label-shuffle model).

Everything runs on small hand-made files inside tmp_path.  The only model that
is built is the label-shuffle sanity model, on a handful of synthetic rows for
two epochs on the CPU.  Never C:\\MSCAD2022.

No pathlib patching: the output folders are redirected through config, and the
other operating system is simulated through `config.running_on_windows`.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import step12_checks as s12

MODULE_PATH = (Path(__file__).resolve().parent.parent / "src" / "mscad"
               / "step12_checks.py")
B6_MODULE_PATH = (Path(__file__).resolve().parent.parent / "src" / "mscad"
                  / "experiments_baseline6.py")

PROPOSED = "proposed"
B2 = "baseline2_ae_lstm"
B3 = "baseline3_cnn_lstm"


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    """Every folder this module reads or writes points inside tmp_path."""
    for name, leaf in (("DATA_ROOT", "data"),
                       ("STEP8_OUTPUT_DIR", "step8"),
                       ("STEP10_OUTPUT_DIR", "step10"),
                       ("STEP10B_OUTPUT_DIR", "step10_attention"),
                       ("STEP11_OUTPUT_DIR", "step11"),
                       ("STEP11B_OUTPUT_DIR", "step11_attention"),
                       ("STEP12_OUTPUT_DIR", "step12"),
                       ("STEP12_B6_OUTPUT_DIR", "step12/baseline6"),
                       ("STEP12_SANITY_OUTPUT_DIR", "step12_sanity"),
                       ("STEP12_SEEDS_OUTPUT_DIR", "step12_seeds"),
                       ("STEP12_THESIS_DIR", "step12/thesis"),
                       ("STEP13_OUTPUT_DIR", "step13"),
                       ("STEP13ALL_OUTPUT_DIR", "step13_all")):
        monkeypatch.setattr(config, name, tmp_path / leaf, raising=False)


# ===========================================================================
# helpers -- the saved files a real run would leave behind
# ===========================================================================
def metrics_payload(label, accuracy=0.9, f1=0.5, g_mean=0.4):
    return {
        "config": label, "label": label, "n_input_features": 30,
        "latent_dim": 15, "parameters": 1000, "training_seconds": 10.0,
        "seed": config.RANDOM_STATE, "best_epoch": 3,
        "full": {"accuracy": accuracy, "precision_macro": 0.6,
                 "recall_macro": 0.55, "f1_macro": f1, "f1_weighted": 0.8,
                 "g_mean": g_mean, "roc_auc": {"macro": 0.9}},
        "duplicate_free": {"f1_macro": f1, "g_mean": g_mean},
        "binary": {"f1": 0.95, "false_positive_rate": 0.02},
        "inference": {"ms_per_1000_flows": 50.0},
    }


def predictions_payload(rows=6, benign_probability=0.7):
    """A predictions.csv with the `p_<label>` columns Step 13 fuses on."""
    orig = np.arange(rows)
    attack = 1.0 - benign_probability
    return pd.DataFrame({
        "Orig_Row": orig,
        "y_true": [0] * rows,
        "y_pred": [0] * rows,
        "p_Normal": [benign_probability] * rows,
        "p_Port_Scan": [attack / 2] * rows,
        "p_Brute_Force": [attack / 2] * rows,
    })


def write_config(tmp_path, key, metrics=None, predictions=None, model=True):
    """One configuration's folder, as Step 11/12 would have written it.

    The folder is the one the comparison module reads this key from, so a
    configuration that keeps its own output folder (Approach 2, Baseline 6) is
    written where it is actually looked for.
    """
    from src.mscad import comparison as cmp

    folder = Path(cmp.result_path(key)).parent
    folder.mkdir(parents=True, exist_ok=True)
    label = dict(config.STEP12_DISPLAY_ORDER)[key]
    payload = metrics_payload(label) if metrics is None else metrics
    (folder / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    if predictions is not None:
        predictions.to_csv(folder / "predictions.csv", index=False)
    if model:
        (folder / "model.keras").write_bytes(b"not a real model, just bytes")
    return folder


def write_fused(tmp_path, which, risks, rows=6):
    """A fused_predictions file recording r_m per flow, as Step 13 writes it."""
    folder = tmp_path / ("step13" if which == "step13" else "step13_all")
    folder.mkdir(parents=True, exist_ok=True)
    name = (config.STEP13_FILES["fused_predictions"] if which == "step13"
            else config.STEP13ALL_FILES["fused_predictions"])
    frame = pd.DataFrame({"Orig_Row": np.arange(rows)})
    for column, value in risks.items():
        frame[column] = value
    frame.to_csv(folder / name, index=False)
    return folder / name


# ===========================================================================
# 12.0  provenance: the fingerprints themselves
# ===========================================================================
def test_provenance_records_the_sha256_size_and_timestamp_of_every_file(tmp_path):
    write_config(tmp_path, PROPOSED, predictions=predictions_payload())

    table = s12.provenance_table([PROPOSED])

    assert set(table["file"]) == {"model", "metrics", "predictions"}
    found = table[table["exists"]]
    assert len(found) == 3
    for _i, row in found.iterrows():
        assert len(row["sha256"]) == 64, row["file"]
        assert row["size (bytes)"] > 0
        assert row["modified"].endswith("UTC")
        assert Path(row["path"]).exists()


def test_the_digest_is_the_sha256_of_those_exact_bytes(tmp_path):
    import hashlib

    folder = write_config(tmp_path, PROPOSED)
    path = folder / "metrics.json"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    record = s12.file_record(PROPOSED, "metrics", path)

    assert record["sha256"] == expected
    assert record["size (bytes)"] == path.stat().st_size


def test_a_missing_file_is_recorded_as_absent_rather_than_guessed(tmp_path):
    write_config(tmp_path, PROPOSED, model=False)

    table = s12.provenance_table([PROPOSED])
    model = table[table["file"] == "model"].iloc[0]

    assert bool(model["exists"]) is False
    assert model["sha256"] == ""
    assert model["size (bytes)"] == 0


def test_approach_1s_model_is_found_in_step10_not_only_beside_its_metrics(
        tmp_path):
    """The published Approach 1 model lives in step10\\, its metrics in step12\\."""
    write_config(tmp_path, PROPOSED, model=False)
    step10 = tmp_path / "step10"
    step10.mkdir(parents=True, exist_ok=True)
    (step10 / config.STEP10_FILES["model"]).write_bytes(b"approach 1 weights")

    table = s12.provenance_table([PROPOSED])
    model = table[table["file"] == "model"].iloc[0]

    assert bool(model["exists"]) is True
    assert Path(model["path"]).parent == step10


# ===========================================================================
# 12.0 (a)  did Step 13 / 13.8 fuse the predictions these models hold TODAY?
# ===========================================================================
def test_a_fusion_that_matches_the_current_predictions_is_ok(tmp_path):
    predictions = predictions_payload(benign_probability=0.7)
    for key in config.FUSION_MODELS:
        write_config(tmp_path, key, predictions=predictions)
    write_fused(tmp_path, "step13",
                {"r_%s" % key: 0.3 for key in config.FUSION_MODELS})

    checks = s12.Checks()
    table = s12.check_fusion_inputs(checks, verbose=False)
    step13 = table[table["fusion"] == "step13"]

    assert len(step13) == len(config.FUSION_MODELS)
    assert set(step13["result"]) == {"[OK]"}
    assert float(step13["max |r difference|"].max()) <= 1e-6
    assert not len(checks.failures)


def test_a_changed_predictions_file_is_detected_as_a_failure(tmp_path):
    """The bug this check exists for: the model was re-evaluated after Step 13."""
    for key in config.FUSION_MODELS:
        write_config(tmp_path, key,
                     predictions=predictions_payload(benign_probability=0.7))
    write_fused(tmp_path, "step13",
                {"r_%s" % key: 0.3 for key in config.FUSION_MODELS})

    # one model is re-evaluated: its benign probability moves, so r_m moves
    folder = tmp_path / "step12" / B2
    predictions_payload(benign_probability=0.4).to_csv(
        folder / "predictions.csv", index=False)

    checks = s12.Checks()
    table = s12.check_fusion_inputs(checks, verbose=False)
    step13 = table[table["fusion"] == "step13"]
    failed = step13[step13["result"] == "[FAIL]"]

    assert len(failed) == 1
    assert dict(config.STEP12_DISPLAY_ORDER)[B2] in list(failed["model"])
    assert float(failed.iloc[0]["max |r difference|"]) == pytest.approx(0.3,
                                                                       abs=1e-9)
    assert "retrained or re-evaluated" in failed.iloc[0]["detail"]
    assert len(checks.failures) == 1
    assert checks.ok is False


def test_the_benign_column_is_recognised_by_name(tmp_path):
    """`p_Normal` is benign because config.BENIGN_LABELS says so."""
    assert s12._is_benign_column("p_Normal") is True
    assert s12._is_benign_column("p_Benign") is True
    assert s12._is_benign_column("p_Port_Scan") is False


def test_a_fusion_that_never_ran_is_skipped_not_failed(tmp_path):
    for key in config.FUSION_MODELS:
        write_config(tmp_path, key, predictions=predictions_payload())

    checks = s12.Checks()
    table = s12.check_fusion_inputs(checks, verbose=False)

    assert not len(table)
    assert set(checks.frame()["result"]) == {"[SKIP]"}


def test_the_all_model_fusion_is_checked_under_its_short_column_names(tmp_path):
    predictions = predictions_payload(benign_probability=0.7)
    for key, _label in config.STEP12_DISPLAY_ORDER:
        write_config(tmp_path, key, predictions=predictions)
    write_fused(tmp_path, "step13_all",
                {"r_%s" % s12.short_label(key): 0.3
                 for key, _l in config.STEP12_DISPLAY_ORDER})

    checks = s12.Checks()
    table = s12.check_fusion_inputs(checks, verbose=False)
    all_rows = table[table["fusion"] == "step13_all"]

    assert len(all_rows) == len(config.STEP12_DISPLAY_ORDER) == 8
    assert set(all_rows["result"]) == {"[OK]"}


# ===========================================================================
# 12.0 (b)  Approach 1's metrics, side by side -- and nothing chosen
# ===========================================================================
def test_approach1_sources_are_reported_side_by_side_and_agreement_is_ok(
        tmp_path):
    label = dict(config.STEP12_DISPLAY_ORDER)[PROPOSED]
    payload = metrics_payload(label, f1=0.7)
    write_config(tmp_path, PROPOSED, metrics=payload)
    (tmp_path / "step11").mkdir(parents=True, exist_ok=True)
    (tmp_path / "step11" / "metrics.json").write_text(json.dumps(payload),
                                                      encoding="utf-8")

    checks = s12.Checks()
    table = s12.check_approach1_consistency(checks, verbose=False)

    assert list(table["source"]) == ["Step 11 (current)",
                                     "Step 12 row (current)"]
    assert not len(checks.failures)


def test_a_backup_that_disagrees_is_a_failure_that_names_the_metrics(tmp_path):
    label = dict(config.STEP12_DISPLAY_ORDER)[PROPOSED]
    write_config(tmp_path, PROPOSED, metrics=metrics_payload(label, f1=0.70))
    (tmp_path / "step11").mkdir(parents=True, exist_ok=True)
    (tmp_path / "step11" / "metrics.json").write_text(
        json.dumps(metrics_payload(label, f1=0.70)), encoding="utf-8")
    backup = tmp_path / "data" / "outputs_backup_2025" / "step11"
    backup.mkdir(parents=True, exist_ok=True)
    (backup / "metrics.json").write_text(
        json.dumps(metrics_payload(label, f1=0.61)), encoding="utf-8")

    checks = s12.Checks()
    table = s12.check_approach1_consistency(checks, verbose=False)
    failures = checks.failures

    assert len(table) == 3
    assert any(name.startswith("backup: ") for name in table["source"])
    assert len(failures) == 1
    assert "f1_macro" in failures.iloc[0]["detail"]
    assert "decide which run the write-up quotes" in failures.iloc[0][
        "what to do"]
    # nothing is repaired: both values are still on disk, exactly as found
    assert json.loads((tmp_path / "step11" / "metrics.json").read_text(
        encoding="utf-8"))["full"]["f1_macro"] == 0.70
    assert json.loads((backup / "metrics.json").read_text(
        encoding="utf-8"))["full"]["f1_macro"] == 0.61


def test_a_single_source_is_skipped_because_there_is_nothing_to_compare(
        tmp_path):
    write_config(tmp_path, PROPOSED)

    checks = s12.Checks()
    s12.check_approach1_consistency(checks, verbose=False)

    assert set(checks.frame()["result"]) == {"[SKIP]"}


def test_backup_folders_are_listed_newest_first(tmp_path):
    root = tmp_path / "data"
    for name in ("outputs_backup_a", "outputs_backup_b"):
        (root / name).mkdir(parents=True, exist_ok=True)
    older = root / "outputs_backup_a"
    os.utime(older, (time.time() - 600, time.time() - 600))
    (root / "not_a_backup").mkdir(parents=True, exist_ok=True)

    folders = s12.backup_folders()

    assert [f.name for f in folders] == ["outputs_backup_b",
                                         "outputs_backup_a"]


# ===========================================================================
# 12.0 (c)  a model written AFTER the metrics that describe it
# ===========================================================================
def test_a_model_newer_than_its_metrics_is_a_failure(tmp_path):
    folder = write_config(tmp_path, PROPOSED)
    # the metrics were saved an hour ago; the weights were written just now
    old = time.time() - 3600
    os.utime(folder / "metrics.json", (old, old))

    provenance = s12.provenance_table([PROPOSED])
    checks = s12.Checks()
    table = s12.check_model_timestamps(checks, provenance, verbose=False)

    assert bool(table.iloc[0]["model newer than metrics"]) is True
    assert table.iloc[0]["result"] == "[FAIL]"
    assert len(checks.failures) == 1
    assert "rewritten after its metrics" in checks.failures.iloc[0][
        "what to do"]


def test_a_model_older_than_its_metrics_is_ok(tmp_path):
    folder = write_config(tmp_path, PROPOSED)
    old = time.time() - 3600
    os.utime(folder / "model.keras", (old, old))

    provenance = s12.provenance_table([PROPOSED])
    checks = s12.Checks()
    table = s12.check_model_timestamps(checks, provenance, verbose=False)

    assert bool(table.iloc[0]["model newer than metrics"]) is False
    assert table.iloc[0]["result"] == "[OK]"
    assert not len(checks.failures)


def test_run_provenance_writes_its_evidence_and_its_report(tmp_path):
    for key, _label in config.STEP12_DISPLAY_ORDER:
        write_config(tmp_path, key, predictions=predictions_payload())

    out = s12.run_provenance(verbose=False)
    folder = tmp_path / "step12"
    names = config.STEP12_PROVENANCE_FILES

    assert (folder / names["provenance"]).exists()
    assert (folder / names["consistency"]).exists()
    assert (folder / names["report"]).exists()
    saved = pd.read_csv(folder / names["provenance"])
    assert len(saved) == 3 * len(config.STEP12_DISPLAY_ORDER)
    assert "# Step 12.0 -- provenance report" in out["report"]
    assert out["saved"]["status"] == "written"


def test_the_report_names_every_failure_and_claims_no_repair(tmp_path):
    for key in config.FUSION_MODELS:
        write_config(tmp_path, key,
                     predictions=predictions_payload(benign_probability=0.7))
    write_fused(tmp_path, "step13",
                {"r_%s" % key: 0.9 for key in config.FUSION_MODELS})

    out = s12.run_provenance(keys=list(config.FUSION_MODELS), verbose=False)

    assert out["ok"] is False
    assert "Nothing was repaired automatically" in out["report"]
    assert "### " in out["report"]


# ===========================================================================
# 12.0b  the retraining lock
# ===========================================================================
def test_the_lock_refuses_to_train_a_configuration_whose_metrics_exist(
        tmp_path, monkeypatch):
    write_config(tmp_path, PROPOSED)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True)
    calls = []

    with pytest.raises(s12.Step12CheckError) as error:
        s12.train_guard(PROPOSED, lambda: calls.append("trained"),
                        verbose=False)

    assert calls == []                      # the trainer was never reached
    assert "LOCK_TRAINED_MODELS is True" in str(error.value)
    assert "metrics.json" in str(error.value) or "step12" in str(error.value)
    assert s12.locked(PROPOSED) is True


def test_the_lock_allows_training_a_configuration_with_no_saved_result(
        tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True)

    assert s12.is_trained(config.BASELINE6_NAME) is False
    assert s12.locked(config.BASELINE6_NAME) is False
    assert s12.train_guard(config.BASELINE6_NAME, lambda: "trained",
                           verbose=False) == "trained"


def test_the_lock_ignores_the_overwrite_flags(tmp_path, monkeypatch):
    """OVERWRITE_* left on is exactly the accident the lock exists for."""
    write_config(tmp_path, PROPOSED)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True)
    monkeypatch.setattr(config, "OVERWRITE_STEP10", True, raising=False)
    monkeypatch.setattr(config, "OVERWRITE_STEP12", True, raising=False)

    with pytest.raises(s12.Step12CheckError):
        s12.train_guard(PROPOSED, lambda: "trained", verbose=False)


def test_turning_the_lock_off_deliberately_allows_the_retrain(tmp_path,
                                                              monkeypatch):
    write_config(tmp_path, PROPOSED)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", False)

    assert s12.locked(PROPOSED) is False
    assert s12.train_guard(PROPOSED, lambda: "trained", verbose=False) \
        == "trained"


def test_the_lock_table_says_what_would_happen_to_every_configuration(
        tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True)
    write_config(tmp_path, PROPOSED)

    table = s12.lock_table()
    by_model = dict(zip(table["model"], table["training now would"]))
    labels = dict(config.STEP12_DISPLAY_ORDER)

    assert len(table) == 8
    assert by_model[labels[PROPOSED]] == "be REFUSED"
    assert by_model[labels[config.BASELINE6_NAME]] == "be allowed"


def test_the_lock_default_in_config_is_on():
    assert config.LOCK_TRAINED_MODELS is True


# ===========================================================================
# 12.2  the sanity baselines
# ===========================================================================
def labels_frame(codes):
    return pd.DataFrame({s12.TARGET_COLUMN: np.asarray(codes, dtype="int64"),
                         "Label_binary": (np.asarray(codes) != 0).astype(
                             "int64")})


def test_the_majority_baseline_predicts_the_most_frequent_training_class():
    y_train = labels_frame([0, 0, 0, 0, 0, 0, 1, 1, 2])

    predictions = s12.majority_class_predictions(y_train, 5)

    assert predictions.tolist() == [0, 0, 0, 0, 0]
    assert len(predictions) == 5


def test_the_majority_baseline_follows_the_training_shares_not_class_zero():
    y_train = labels_frame([0, 1, 1, 1, 2])

    assert s12.majority_class_predictions(y_train, 3).tolist() == [1, 1, 1]


def test_the_stratified_baseline_draws_from_the_training_shares_and_is_seeded():
    y_train = labels_frame([0] * 80 + [1] * 15 + [2] * 5)

    first = s12.stratified_predictions(y_train, 400)
    again = s12.stratified_predictions(y_train, 400)
    other = s12.stratified_predictions(y_train, 400, seed=7)

    assert first.tolist() == again.tolist()          # RANDOM_STATE by default
    assert first.tolist() != other.tolist()
    assert set(np.unique(first)) <= {0, 1, 2}
    assert 0.6 < float((first == 0).mean()) < 0.95   # roughly the 80% share


def test_the_class_shares_table_adds_up_to_one_hundred_percent():
    y_test = labels_frame([0] * 7 + [1] * 2 + [2])

    shares = s12.class_shares(y_test, {0: "Normal", 1: "Port_Scan",
                                       2: "Brute_Force"})

    assert list(shares["class"]) == ["Normal", "Port_Scan", "Brute_Force"]
    assert list(shares["test rows"]) == [7, 2, 1]
    assert float(shares["share (%)"].sum()) == pytest.approx(100.0, abs=0.05)


def test_the_label_shuffle_permutes_the_fit_labels_and_nothing_else():
    """Only the Step 9 FIT rows change, and only by a permutation."""
    y_train = labels_frame(([0] * 40 + [1] * 20 + [2] * 20))

    permuted, record = s12.shuffle_fit_labels(y_train, seed=3)
    before = np.asarray(y_train[s12.TARGET_COLUMN])
    after = np.asarray(permuted[s12.TARGET_COLUMN])
    fit_rows = np.asarray(record["fit row"])
    untouched = np.setdiff1d(np.arange(len(y_train)), fit_rows)

    # the multiset of labels is unchanged on the fit rows: a permutation
    assert sorted(after[fit_rows].tolist()) == sorted(before[fit_rows].tolist())
    # the validation rows keep their real labels
    assert after[untouched].tolist() == before[untouched].tolist()
    assert sorted(record["true label"].tolist()) == sorted(
        record["shuffled label"].tolist())
    assert (after != before).any()
    # the originals are not modified in place
    assert np.asarray(y_train[s12.TARGET_COLUMN]).tolist() == before.tolist()


def test_the_label_shuffle_keeps_the_binary_column_consistent():
    y_train = labels_frame([0] * 40 + [1] * 20 + [2] * 20)

    permuted, _record = s12.shuffle_fit_labels(y_train, seed=3)

    expected = (np.asarray(permuted[s12.TARGET_COLUMN]) != 0).astype("int64")
    assert np.asarray(permuted["Label_binary"]).tolist() == expected.tolist()


def test_the_label_shuffle_is_reproducible_from_its_seed():
    y_train = labels_frame([0] * 40 + [1] * 20 + [2] * 20)

    first, _r1 = s12.shuffle_fit_labels(y_train, seed=11)
    again, _r2 = s12.shuffle_fit_labels(y_train, seed=11)
    other, _r3 = s12.shuffle_fit_labels(y_train, seed=12)

    assert first[s12.TARGET_COLUMN].tolist() == again[s12.TARGET_COLUMN].tolist()
    assert first[s12.TARGET_COLUMN].tolist() != other[s12.TARGET_COLUMN].tolist()


def test_the_shuffle_model_trains_on_permuted_fit_labels_only(monkeypatch):
    """The real model, two epochs, on synthetic rows: the labels it was fitted
    on are a permutation, and no test label is touched anywhere."""
    pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")
    monkeypatch.setattr(config, "CLF_EPOCHS", 2)
    monkeypatch.setattr(config, "CLF_BATCH_SIZE", 8)

    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.random((60, 4)),
                           columns=["f1", "f2", "f3", "f4"])
    y_train = labels_frame([0] * 30 + [1] * 15 + [2] * 15)

    out = s12.train_label_shuffle(X_train, y_train, seed=5, epochs=2,
                                  verbose=False)
    record = out["permutation"]

    assert len(out["history"]) <= 2
    assert out["fit_rows"] > 0 and out["validation_rows"] > 0
    assert sorted(record["true label"].tolist()) == sorted(
        record["shuffled label"].tolist())
    assert (np.asarray(record["true label"])
            != np.asarray(record["shuffled label"])).any()
    assert out["seed"] == 5
    # the training labels on disk are untouched
    assert y_train[s12.TARGET_COLUMN].tolist() == (
        [0] * 30 + [1] * 15 + [2] * 15)


def test_the_leakage_verdict_fires_when_the_shuffle_model_beats_the_floor():
    checks = s12.Checks()
    margin = float(config.SANITY_LEAKAGE_MARGIN)
    reference = 0.10

    # the same comparison run_sanity makes, on numbers chosen by hand
    leak = (reference + margin + 0.2) > reference + margin
    checks.add("12.2 sanity", "the label-shuffle model stays near the random "
                              "floor", not leak, "by hand", "POSSIBLE LEAKAGE")

    assert leak is True
    assert checks.failures.iloc[0]["result"] == "[FAIL]"
    assert margin == pytest.approx(0.10)


def test_the_sanity_report_is_written_from_the_computed_numbers():
    out = {
        "class_shares": pd.DataFrame([
            {"class": "Normal", "test rows": 700, "share (%)": 70.0},
            {"class": "Port_Scan", "test rows": 200, "share (%)": 20.0},
            {"class": "Brute_Force", "test rows": 100, "share (%)": 10.0}]),
        "table": pd.DataFrame([{"baseline": "majority class",
                                "F1 (macro)": 0.2745, "G-mean": 0.0}]),
        "majority_accuracy": 0.7013, "shuffle_macro_f1": 0.1002,
        "random_macro_f1": 0.1885, "leak": False,
    }

    report = s12.sanity_report(out)

    assert "0.7013" in report                   # the majority accuracy
    assert "90.0%" in report                    # the two largest classes
    assert "0.1002" in report and "0.1885" in report
    assert "macro-F1 and G-mean are the comparison" in report
    assert "[FAIL]" not in report


def test_the_sanity_report_states_the_failure_when_there_is_one():
    out = {
        "class_shares": pd.DataFrame([
            {"class": "Normal", "test rows": 700, "share (%)": 70.0},
            {"class": "Port_Scan", "test rows": 300, "share (%)": 30.0}]),
        "table": pd.DataFrame([{"baseline": "majority class",
                                "F1 (macro)": 0.41, "G-mean": 0.0}]),
        "majority_accuracy": 0.70, "shuffle_macro_f1": 0.88,
        "random_macro_f1": 0.18, "leak": True,
    }

    report = s12.sanity_report(out)

    assert "[FAIL]" in report
    assert "permuted labels cannot learn the task" in report


def test_the_sanity_baselines_are_never_part_of_the_ranking():
    """The sanity table has its own column names, so it cannot be concatenated
    into the comparison by accident."""
    from src.mscad import comparison as cmp

    assert "baseline" not in cmp.COLUMNS
    assert "what it is" not in cmp.COLUMNS


# ===========================================================================
# 12.11  the discussion says where B6 is NOT a one-difference comparison
# ===========================================================================
def comparison_table():
    """The smallest comparison table build_discussion can rank."""
    return pd.DataFrame([
        {"config": label, "F1 (macro)": 0.50 + 0.01 * i,
         "G-mean": 0.40 + 0.01 * i, "binary FPR": 0.02,
         "F1 macro (dup-free)": 0.50 + 0.01 * i,
         "G-mean (dup-free)": 0.40 + 0.01 * i}
        for i, (_key, label) in enumerate(config.STEP12_DISPLAY_ORDER)])


def test_the_discussion_states_that_b4_vs_b6_differs_in_two_things(tmp_path):
    write_config(tmp_path, "baseline4_ae_pca_lstm",
                 metrics=dict(metrics_payload(
                     dict(config.STEP12_DISPLAY_ORDER)[
                         "baseline4_ae_pca_lstm"]), pca_components=2))
    step12 = {"comparison": comparison_table(), "verified": pd.DataFrame()}
    baseline6 = {"metrics": {"pca_components": 3}, "length_warning": None}

    text = s12.build_discussion(step12, None, None, None, baseline6)

    assert "B4 vs B6 differs in two things" in text
    assert "keeps 2 component(s) and Baseline 6's keeps 3" in text
    assert "B1 vs B6 is the clean single-difference pair" in text


def test_the_discussion_makes_no_such_claim_when_the_widths_match(tmp_path):
    write_config(tmp_path, "baseline4_ae_pca_lstm",
                 metrics=dict(metrics_payload(
                     dict(config.STEP12_DISPLAY_ORDER)[
                         "baseline4_ae_pca_lstm"]), pca_components=3))
    step12 = {"comparison": comparison_table(), "verified": pd.DataFrame()}
    baseline6 = {"metrics": {"pca_components": 3}, "length_warning": None}

    text = s12.build_discussion(step12, None, None, None, baseline6)

    assert "differs in two things" not in text


# ===========================================================================
# 12.10  seeds -- only reported when more than one was trained
# ===========================================================================
def test_indistinguishable_pairs_needs_a_gap_smaller_than_one_std():
    summary = pd.DataFrame([
        {"model": "A1: proposed", "F1 (macro) mean": 0.70,
         "F1 (macro) std": 0.05},
        {"model": "B1: AE -> CNN-LSTM", "F1 (macro) mean": 0.69,
         "F1 (macro) std": 0.04},
        {"model": "B2: AE -> LSTM", "F1 (macro) mean": 0.40,
         "F1 (macro) std": 0.01},
    ])

    pairs = s12.indistinguishable_pairs(summary, "F1 (macro)")
    verdicts = dict(zip(pairs["pair"], pairs["verdict"]))

    assert len(pairs) == 3
    # 0.01 apart with a std of 0.05: this evidence cannot tell them apart
    assert verdicts["A1 vs B1"] == "not distinguishable"
    # 0.30 apart with a std of 0.05: the gap is larger than the spread
    assert verdicts["A1 vs B2"] == "gap exceeds one std"
    assert "significant" not in " ".join(pairs["verdict"]).lower()


def test_a_seed_folder_is_a_new_folder_per_seed(tmp_path):
    place = s12.seed_folder(PROPOSED, 7)

    assert place is not None
    assert place == tmp_path / "step12_seeds" / PROPOSED / "seed7"
    assert place.parent.parent.name == "step12_seeds"


# ===========================================================================
# the environment the brief fixes: Keras 2, no Keras-3-only API
# ===========================================================================
def test_the_new_modules_never_import_keras_ops():
    for path in (MODULE_PATH, B6_MODULE_PATH):
        text = path.read_text(encoding="utf-8")
        assert "keras.ops" not in text, path.name
        assert "from keras import ops" not in text, path.name
        assert "keras.saving.register" not in text, path.name


def test_the_new_modules_only_reach_keras_through_the_shared_helper():
    for path in (MODULE_PATH, B6_MODULE_PATH):
        text = path.read_text(encoding="utf-8")
        assert "import keras" not in text, path.name
        assert "import tensorflow" not in text, path.name


# ===========================================================================
# H)  Baseline 6 joins the addenda; Step 13's own fusion does not change
# ===========================================================================
def test_baseline6_is_in_the_all_model_fusion_and_explainability_lists():
    assert config.BASELINE6_NAME in config.FUSION_ALL_MODELS
    assert config.BASELINE6_NAME in config.XAI_ALL_MODELS
    assert len(config.FUSION_ALL_MODELS) == 8
    assert len(config.XAI_ALL_MODELS) == 8


def test_step_13s_own_three_model_fusion_is_unchanged():
    assert list(config.FUSION_MODELS) == ["proposed", "baseline2_ae_lstm",
                                          "baseline3_cnn_lstm"]
    assert config.BASELINE6_NAME not in config.FUSION_MODELS


def test_the_four_step_12_experiments_are_unchanged():
    """Baseline 6 is NOT an EXPERIMENTS entry: it has its own 12.6 runner, so
    re-running the 12.3 loop cannot touch it."""
    assert [spec["name"] for spec in config.EXPERIMENTS] == [
        "proposed", "baseline1_ae_cnn_lstm", "baseline2_ae_lstm",
        "baseline3_cnn_lstm"]
    assert config.BASELINE6_NAME not in [spec["name"]
                                         for spec in config.EXPERIMENTS]
