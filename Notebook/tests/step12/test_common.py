r"""
tests/step12/test_common.py -- the shared helpers behind the eight files:
the retraining lock, provenance, and the saved-metrics comparison.

NEW FILE.  Nothing is trained here and nothing is written outside tmp_path.
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
from src.mscad import comparison as cmp
from src.mscad.experiments_step12 import common

from .conftest import (KEYS, metrics_payload, predictions_frame,  # noqa: E402
                      write_result)

FOLDER = (Path(__file__).resolve().parent.parent.parent / "src" / "mscad"
          / "experiments_step12")
PROPOSED = "proposed"
B1 = "baseline1_ae_cnn_lstm"
B6 = config.BASELINE6_NAME


# ===========================================================================
# the lock
# ===========================================================================
def test_the_lock_refuses_to_retrain_a_configuration_whose_metrics_exist(
        finished):
    finished(B1)
    calls = []

    with pytest.raises(common.Step12ConfigError) as error:
        common.guard(B1, lambda: calls.append("trained"))

    assert calls == []                       # the trainer was never reached
    assert "LOCK_TRAINED_MODELS is True" in str(error.value)
    assert "metrics.json" in str(error.value)
    assert common.locked(B1) is True


def test_the_lock_ignores_the_overwrite_flags(finished, monkeypatch):
    """OVERWRITE_* left on is exactly the accident the lock exists for."""
    finished(B1)
    monkeypatch.setattr(config, "OVERWRITE_STEP12", True, raising=False)
    monkeypatch.setattr(config, "OVERWRITE_B6", True, raising=False)

    with pytest.raises(common.Step12ConfigError):
        common.guard(B1, lambda: "trained")


def test_a_missing_result_with_the_lock_on_stops_with_the_file_name(
        step12_dirs):
    """No silent training: the message names the file and the one line to set."""
    with pytest.raises(common.Step12ConfigError) as error:
        common.refuse_missing(B1)

    message = str(error.value)
    assert "has no saved result" in message
    assert str(step12_dirs / "step12" / B1 / "metrics.json") in message
    assert "set LOCK_TRAINED_MODELS = False in config.py" in message
    # 13-14, not 11-14: Step 11 evaluates Approach 1 and is not affected by
    # retraining a Step 12 baseline; 13 and 14 are what consume it
    assert "re-run Steps 13-14 for this dataset" in message
    assert "nothing was trained" in message


def test_turning_the_lock_off_deliberately_allows_the_call(finished,
                                                           monkeypatch):
    finished(B1)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", False)

    assert common.locked(B1) is False
    assert common.guard(B1, lambda: "trained") == "trained"


def test_an_unreachable_output_folder_is_not_reported_as_a_missing_result(
        monkeypatch):
    """The Windows-path guard and a missing file are different answers."""
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR", r"C:\MSCAD2022\outputs\step12")

    with pytest.raises(common.Step12ConfigError) as error:
        common.refuse_missing(B1)

    message = str(error.value)
    assert "cannot be read on this machine" in message
    assert "Windows-path guard, not a missing result" in message
    assert "has no saved result" not in message


def test_the_lock_table_says_what_would_happen_to_every_configuration(
        finished):
    finished(PROPOSED, B1)

    table = common.lock_table()
    by_config = dict(zip(table["configuration"], table["running it now would"]))
    labels = dict(config.STEP12_DISPLAY_ORDER)

    assert len(table) == 8
    assert by_config[labels[PROPOSED]] == "LOAD (locked)"
    assert by_config[labels[B6]] == "STOP: result missing"
    assert list(table["file"]) == ["%s.py" % common.short_name(key)
                                   for key in KEYS]


def test_the_lock_default_in_config_is_on_and_is_defined_once():
    """A second assignment would mean editing the first one does nothing."""
    text = (Path(__file__).resolve().parent.parent.parent
            / "config.py").read_text(encoding="utf-8")

    assert config.LOCK_TRAINED_MODELS is True
    assert text.count("\nLOCK_TRAINED_MODELS = ") == 2
    assert 'LOCK_TRAINED_MODELS = globals().get("LOCK_TRAINED_MODELS", True)' \
        in text


# ===========================================================================
# provenance
# ===========================================================================
def test_provenance_records_the_sha256_size_and_timestamp(finished):
    finished(B1)

    table = common.provenance(B1)

    assert set(table["file"]) >= {"model", "metrics", "predictions"}
    found = table[table["exists"]]
    assert len(found) >= 3
    for _i, row in found.iterrows():
        assert len(row["sha256"]) == 64, row["file"]
        assert row["size (bytes)"] > 0
        assert row["modified"].endswith("UTC")


def test_provenance_detects_a_changed_predictions_file(finished):
    """The question the fingerprint exists to answer: are these the same bytes?"""
    folders = finished(B1)

    before = common.provenance(B1)
    predictions_frame(seed=99, benign_probability=0.4).to_csv(
        folders[B1] / "predictions.csv", index=False)
    after = common.provenance(B1)

    digest = lambda table: dict(zip(table["file"], table["sha256"]))
    assert digest(before)["metrics"] == digest(after)["metrics"]
    assert digest(before)["predictions"] != digest(after)["predictions"]


def test_provenance_takes_the_extra_files_a_configuration_has(finished):
    folders = finished(B1)
    (folders[B1] / "encoder.keras").write_bytes(b"an encoder")

    table = common.provenance(B1, extra=[("encoder",
                                          folders[B1] / "encoder.keras")])
    row = table[table["file"] == "encoder"].iloc[0]

    assert bool(row["exists"]) is True
    assert len(row["sha256"]) == 64


def test_a_missing_file_is_recorded_as_absent_rather_than_guessed(finished):
    finished(B1, **{B1: metrics_payload("B1")})
    folder = Path(cmp.result_path(B1)).parent
    (folder / "model.keras").unlink()

    table = common.provenance(B1)
    model = table[table["file"] == "model"].iloc[0]

    assert bool(model["exists"]) is False
    assert model["sha256"] == ""


# ===========================================================================
# the saved-metrics snapshot  (rule 4)
# ===========================================================================
def test_the_snapshot_records_every_finished_configuration(all_finished):
    before = common.snapshot()

    assert set(before) == set(KEYS)
    assert "full.f1_macro" in before["proposed"]
    assert "duplicate_free.f1_macro" in before["proposed"]
    assert "binary.false_positive_rate" in before["proposed"]


def test_an_unchanged_file_compares_as_ok(all_finished):
    before = common.snapshot()

    table = common.compare_with_snapshot(before)

    assert len(table) == 8
    assert set(table["result"]) == {"[OK]"}
    assert float(table["largest difference"].max()) == 0.0
    assert (table["metrics compared"] > 0).all()


def test_a_changed_value_is_a_failure_that_names_the_metric(all_finished):
    before = common.snapshot()
    folder = Path(cmp.result_path(B1)).parent
    payload = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    payload["full"]["f1_macro"] = float(payload["full"]["f1_macro"]) + 0.01
    (folder / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    table = common.compare_with_snapshot(before)
    failed = table[table["result"] == "[FAIL]"]

    assert len(failed) == 1
    assert failed.iloc[0]["configuration"] == dict(
        config.STEP12_DISPLAY_ORDER)[B1]
    assert "full.f1_macro" in failed.iloc[0]["detail"]
    assert float(failed.iloc[0]["largest difference"]) == pytest.approx(
        0.01, abs=1e-9)


def test_a_difference_under_the_tolerance_still_fails_at_1e_9(all_finished):
    before = common.snapshot()
    folder = Path(cmp.result_path(B1)).parent
    payload = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    payload["full"]["g_mean"] = float(payload["full"]["g_mean"]) + 1e-8
    (folder / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    assert "[FAIL]" in set(common.compare_with_snapshot(before)["result"])
    assert set(common.compare_with_snapshot(before, tolerance=1e-6)
               ["result"]) == {"[OK]"}


def test_a_result_that_disappeared_is_a_failure(all_finished):
    before = common.snapshot()
    Path(cmp.result_path(B1)).unlink()

    table = common.compare_with_snapshot(before)
    failed = table[table["result"] == "[FAIL]"]

    assert len(failed) == 1
    assert "GONE" in failed.iloc[0]["detail"]


def test_a_configuration_that_is_new_in_this_run_is_not_a_failure(finished):
    """Baseline 6 on its first run was in no snapshot; that is not a change."""
    finished(B1)
    before = common.snapshot()
    finished(B6)

    table = common.compare_with_snapshot(before, [B1, B6])
    b6_row = table[table["configuration"] == dict(
        config.STEP12_DISPLAY_ORDER)[B6]].iloc[0]

    assert b6_row["result"] == "[OK]"
    assert "new in this run" in b6_row["detail"]


# ===========================================================================
# the protocol checks
# ===========================================================================
def test_an_unrecorded_setting_is_skipped_rather_than_claimed(finished):
    """A check that passes because nothing was looked at is worse than none."""
    label = dict(config.STEP12_DISPLAY_ORDER)[B1]
    payload = metrics_payload(label)
    payload.pop("seed")
    finished(B1, **{B1: payload})

    table = common.protocol_checks(B1)
    seed_row = table[table["check"].str.contains("seed")].iloc[0]

    assert seed_row["result"] == "[SKIP]"
    assert "did not record its seed" in seed_row["detail"]


def test_a_recorded_seed_that_differs_is_a_failure(finished):
    label = dict(config.STEP12_DISPLAY_ORDER)[B1]
    finished(B1, **{B1: metrics_payload(label,
                                        seed=config.RANDOM_STATE + 1)})

    table = common.protocol_checks(B1)
    seed_row = table[table["check"].str.contains("seed")].iloc[0]

    assert seed_row["result"] == "[FAIL]"


def test_the_protocol_table_names_the_shared_hyperparameters(finished):
    finished(B1)

    detail = common.shared_hyperparameters()

    for number in (config.CNN_KERNEL, config.CNN_POOL, config.LSTM_UNITS,
                   config.DENSE_UNITS, config.CLF_EPOCHS,
                   config.CLF_BATCH_SIZE, config.CLF_PATIENCE):
        assert str(number) in detail
    assert config.EARLY_STOP_METRIC in detail


# ===========================================================================
# the manifest -- the only file a loaded configuration writes
# ===========================================================================
def test_the_manifest_goes_beside_a_finished_folder_never_inside_it(finished,
                                                                    tmp_path):
    folders = finished(B1)
    before = sorted(p.name for p in folders[B1].iterdir())

    manifest = common.write_manifest(B1, cmp.load_result(B1),
                                     common.provenance(B1), common.LOADED,
                                     verbose=False)
    folder = Path(manifest["folder"])

    assert folder == tmp_path / "step12" / "baseline1"
    assert folder != folders[B1]
    assert sorted(p.name for p in folders[B1].iterdir()) == before
    payload = json.loads((folder / "manifest.json").read_text(
        encoding="utf-8"))
    assert payload["key"] == B1
    assert payload["status"] == "loaded"
    assert payload["lock"]["LOCK_TRAINED_MODELS"] is True
    assert payload["files"]
    assert (folder / "provenance.csv").exists()


def test_no_manifest_is_written_when_the_folder_is_unusable(monkeypatch,
                                                            tmp_path):
    r"""The Windows-path guard: nothing is created, not even a junk folder.

    Checked in an empty directory of its own rather than through the shared
    `assert_no_leaked_output` fixture, because this module's autouse fixtures
    legitimately put step8\ inside tmp_path.
    """
    empty = tmp_path / "nothing_should_appear_here"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.setattr(config, "STEP12_CONFIG_DIR",
                        r"C:\MSCAD2022\outputs\step12")

    out = common.write_manifest(B1, {}, pd.DataFrame(), common.LOADED,
                                verbose=False)

    assert out["status"] == "skipped_not_windows"
    assert out["folder"] is None
    assert os.listdir(empty) == [], os.listdir(empty)


# ===========================================================================
# the folder itself
# ===========================================================================
def test_no_keras_3_only_api_in_the_new_folder():
    for path in sorted(FOLDER.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "keras.ops" not in text, path.name
        assert "from keras import ops" not in text, path.name


def test_no_module_in_the_new_folder_imports_keras_or_tensorflow_directly():
    """Keras is reached only through the existing modules."""
    for path in sorted(FOLDER.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "import keras" not in text, path.name
        assert "import tensorflow" not in text, path.name


def test_no_module_in_the_new_folder_builds_or_fits_a_model():
    """No duplicated logic: the training lives in the existing modules."""
    for path in sorted(FOLDER.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert ".fit(" not in text, path.name
        assert "Sequential(" not in text, path.name
        assert "keras.layers" not in text, path.name


def test_every_configuration_has_one_short_name_everywhere():
    assert common.SHORT_NAMES == dict(config.STEP12_CONFIG_FILES)
    assert set(common.SHORT_NAMES) == set(KEYS)
    assert len(set(common.SHORT_NAMES.values())) == 8


# ===========================================================================
# the lock message, and which of the two situations it is
# ===========================================================================
# Each dataset has its own output root, so a configuration can be finished in
# one and never trained in the other. The message used to say "its model and
# metrics were produced by an earlier run" either way, which is false in a
# fresh root -- exactly what a newly switched DATASET_SOURCE looks like.
@pytest.fixture
def two_roots(tmp_path, monkeypatch):
    """Official mode, with both output roots inside tmp_path."""
    monkeypatch.setattr(config, "DATASET_SOURCE", "official", raising=False)
    monkeypatch.setattr(config, "RUNNING_OFFICIAL", True, raising=False)
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "outputs_official",
                        raising=False)
    monkeypatch.setattr(config, "OUTPUTS_ROOT_OFFICIAL",
                        tmp_path / "outputs_official", raising=False)
    monkeypatch.setattr(config, "OUTPUTS_ROOT_OWN", tmp_path / "outputs",
                        raising=False)
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR",
                        tmp_path / "outputs_official" / "step12")
    monkeypatch.setattr(config, "STEP12_CONFIG_DIR",
                        tmp_path / "outputs_official" / "step12")
    return tmp_path


def write_elsewhere(tmp_path, key):
    """The same configuration, finished in the OTHER dataset's root."""
    folder = tmp_path / "outputs" / "step12" / key
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(
        json.dumps({"full": {"f1_macro": 0.68}}), encoding="utf-8")
    return folder / "metrics.json"


def test_the_other_datasets_root_is_named_and_its_path_given(two_roots):
    elsewhere = write_elsewhere(two_roots, B1)

    with pytest.raises(common.Step12ConfigError) as error:
        common.refuse_missing(B1)

    message = str(error.value)
    assert "has no saved result in the official output root" in message
    assert "this root holds 0 of 8 Step 12 results" in message
    assert "the own root HAS it: %s" % elsewhere in message


def test_a_fresh_root_is_not_described_as_a_lost_result(two_roots):
    """Nothing was produced earlier here, so nothing can be lost."""
    write_elsewhere(two_roots, B1)

    with pytest.raises(common.Step12ConfigError) as error:
        common.refuse_missing(B1)

    message = str(error.value)
    assert "Step 12 has never been run against the official dataset" in message
    assert "nothing can be lost" in message
    assert "must NOT be read as this one's" in message
    assert "produced by an earlier run" not in message


def test_a_root_that_does_hold_results_still_warns_about_the_missing_one(
        two_roots, finished):
    finished("baseline2_ae_lstm")

    with pytest.raises(common.Step12ConfigError) as error:
        common.refuse_missing(B1)

    message = str(error.value)
    assert "this root holds 1 of 8 Step 12 results" in message
    assert "produced by an earlier run in this output root" in message
    assert "check what happened to this one before retraining it" in message
    assert "never been run against" not in message


def test_the_message_says_so_when_neither_root_has_it(two_roots):
    with pytest.raises(common.Step12ConfigError) as error:
        common.refuse_missing(B1)

    assert "the own root does not have it either" in str(error.value)


def test_both_messages_name_the_one_line_that_changes_it(two_roots, finished):
    for setup in (lambda: None, lambda: finished("baseline2_ae_lstm")):
        setup()
        with pytest.raises(common.Step12ConfigError) as error:
            common.refuse_missing(B1)
        message = str(error.value)
        assert "set LOCK_TRAINED_MODELS = False in config.py" in message
        assert "re-run Steps 13-14 for this dataset" in message


def test_approach_1_and_2_are_not_counted_as_step_12_results(two_roots,
                                                             monkeypatch):
    """Their metrics live in step11\\ and step11_attention\\, so they say
    nothing about whether Step 12 ever ran against this dataset."""
    monkeypatch.setattr(config, "STEP11B_OUTPUT_DIR",
                        two_roots / "outputs_official" / "step11_attention",
                        raising=False)
    folder = two_roots / "outputs_official" / "step11_attention"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text("{}", encoding="utf-8")

    assert common.step12_results_here() == []
    with pytest.raises(common.Step12ConfigError) as error:
        common.refuse_missing(B1)
    assert "never been run against the official dataset" in str(error.value)


def test_the_other_root_helper_follows_the_active_mode(two_roots, monkeypatch):
    assert common.other_dataset()[0] == "own"

    monkeypatch.setattr(config, "RUNNING_OFFICIAL", False, raising=False)
    assert common.other_dataset()[0] == "official"


def test_the_sibling_path_swaps_only_the_root(two_roots):
    path = common.metrics_path(B1)
    sibling = common.metrics_path_in_other_root(B1)

    assert str(path).startswith(str(two_roots / "outputs_official"))
    assert str(sibling).startswith(str(two_roots / "outputs"))
    assert sibling.parts[-3:] == path.parts[-3:]   # step12/<key>/metrics.json
