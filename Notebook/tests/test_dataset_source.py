r"""
tests/test_dataset_source.py -- the DATASET_SOURCE switch, the output-root
routing, and the retraining lock for Steps 10 and 10.2.

NEW FILE.  The two modes are checked in SUBPROCESSES: `config` reads the
environment once at import, and reloading it inside the test process would
leave the other tests looking at a half-switched module.  A subprocess answers
the question that actually matters -- what does a fresh kernel see -- and
cannot leak.

Nothing here writes outside tmp_path, and nothing is trained.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import training_lock as tl

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent

# Every STEP output folder config defines, so the routing test cannot miss one.
# CSV_OUTPUT_DIR and RAW_DIR are deliberately absent: they are INPUT folders for
# the own extraction (C:\MSCAD2022\CSVs) and do not follow the switch.
DIR_NAMES = sorted(name for name in dir(config)
                   if (name.startswith("STEP") and name.endswith("_OUTPUT_DIR"))
                   or name in ("OUTPUTS_DIR", "OUTPUTS_ROOT", "MERGED_DIR",
                               "STEP12_THESIS_DIR", "STEP12_CONFIG_DIR",
                               "STEP12_B6_OUTPUT_DIR"))


def config_in_mode(source: str, data_root: Path) -> dict:
    """What a fresh kernel sees with DATASET_SOURCE set to `source`."""
    script = (
        "import json, os, sys\n"
        "sys.path.insert(0, %r)\n"
        "import config\n"
        "out = {'DATASET_SOURCE': config.DATASET_SOURCE,\n"
        "       'RUNNING_OFFICIAL': config.RUNNING_OFFICIAL,\n"
        "       'BENIGN_LABELS': list(config.BENIGN_LABELS),\n"
        "       'METADATA_COLUMNS': list(config.METADATA_COLUMNS),\n"
        "       'METADATA_SHARED_COLUMNS': list(config.METADATA_SHARED_COLUMNS),\n"
        "       'SKIP_STEPS': list(config.SKIP_STEPS),\n"
        "       'OFFICIAL_CSV': str(config.OFFICIAL_CSV)}\n"
        "for name in %r:\n"
        "    out[name] = str(getattr(config, name))\n"
        "print(json.dumps(out))\n" % (str(NOTEBOOK_DIR), DIR_NAMES))
    environment = dict(os.environ)
    if source:
        environment["MSCAD_DATASET_SOURCE"] = source
    else:                      # unset, so config.py's own default is used
        environment.pop("MSCAD_DATASET_SOURCE", None)
    environment["MSCAD_DATA_ROOT"] = str(data_root)
    environment["TF_CPP_MIN_LOG_LEVEL"] = "3"
    finished = subprocess.run([sys.executable, "-c", script],
                              cwd=str(NOTEBOOK_DIR), env=environment,
                              capture_output=True, text=True, timeout=600)
    assert finished.returncode == 0, finished.stderr[-2000:]
    return json.loads(finished.stdout.strip().splitlines()[-1])


# ===========================================================================
# the switch itself
# ===========================================================================
def test_official_is_the_default(tmp_path):
    seen = config_in_mode("", tmp_path)          # empty -> the file's default

    assert seen["DATASET_SOURCE"] == "official"
    assert seen["RUNNING_OFFICIAL"] is True


def test_an_unknown_source_is_refused_rather_than_guessed(tmp_path):
    environment = dict(os.environ)
    environment["MSCAD_DATASET_SOURCE"] = "kaggle"
    environment["MSCAD_DATA_ROOT"] = str(tmp_path)
    finished = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, %r); import config"
         % str(NOTEBOOK_DIR)],
        cwd=str(NOTEBOOK_DIR), env=environment, capture_output=True,
        text=True, timeout=600)

    assert finished.returncode != 0
    assert "DATASET_SOURCE is 'kaggle'" in finished.stderr
    assert "official, own" in finished.stderr


# ===========================================================================
# the routing: every step folder follows the switch
# ===========================================================================
def test_official_mode_routes_every_output_folder_to_outputs_official(tmp_path):
    seen = config_in_mode("official", tmp_path)

    assert seen["OUTPUTS_ROOT"] == str(tmp_path / "outputs_official")
    for name in DIR_NAMES:
        value = seen[name]
        assert "outputs_official" in value, (name, value)
        parts = Path(value).parts
        assert "outputs" not in parts, (name, value)


def test_own_mode_routes_every_output_folder_exactly_as_before(tmp_path):
    seen = config_in_mode("own", tmp_path)

    assert seen["OUTPUTS_ROOT"] == str(tmp_path / "outputs")
    for name in DIR_NAMES:
        value = seen[name]
        assert "outputs_official" not in value, (name, value)
        assert str(tmp_path / "outputs") in value, (name, value)


def test_official_mode_never_names_the_own_output_root(tmp_path):
    """The guarantee the brief asks for: outputs\\ is not written in this mode."""
    seen = config_in_mode("official", tmp_path)
    own_root = Path(str(tmp_path / "outputs")).parts

    for name in DIR_NAMES:
        parts = Path(seen[name]).parts
        assert parts[:len(own_root)] != own_root, (name, seen[name])


# ===========================================================================
# own mode is unchanged
# ===========================================================================
def test_own_mode_keeps_the_original_benign_set_and_metadata(tmp_path):
    seen = config_in_mode("own", tmp_path)

    assert seen["BENIGN_LABELS"] == ["Benign", "Normal"]
    assert seen["METADATA_SHARED_COLUMNS"] == ["Protocol"]
    assert "Protocol" in seen["METADATA_COLUMNS"]
    assert "Source_Capture" in seen["METADATA_COLUMNS"]
    assert seen["SKIP_STEPS"] == []


def test_official_mode_uses_normal_as_the_only_benign_class(tmp_path):
    seen = config_in_mode("official", tmp_path)

    assert seen["BENIGN_LABELS"] == ["Normal"]
    assert "Benign" not in seen["BENIGN_LABELS"]
    assert seen["SKIP_STEPS"] == ["1", "2"]


def test_official_mode_keeps_every_identifier_and_orig_row_out_of_features(
        tmp_path):
    seen = config_in_mode("official", tmp_path)

    for name in ("Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port",
                 "Timestamp", "Orig_Row"):
        assert name in seen["METADATA_COLUMNS"], name
    # Protocol is a behavioural attribute, not an identifier: Step 4 one-hot
    # encodes it, so it must NOT be moved out of the features
    assert "Protocol" not in seen["METADATA_COLUMNS"]
    assert seen["METADATA_SHARED_COLUMNS"] == []


def test_the_own_extraction_quick_check_is_recorded_for_the_comparison():
    assert config.OWN_QUICK_CHECK["random forest accuracy"] == 0.5836
    assert config.OFFICIAL_QUICK_CHECK["random forest macro-F1"] == 0.8542
    assert config.OFFICIAL_EXPECTED_ROWS == 128_799
    assert config.OFFICIAL_EXPECTED_COLUMNS == 67
    assert sum(config.OFFICIAL_EXPECTED_LABELS.values()) == 128_799
    assert set(config.OFFICIAL_EXPECTED_LABELS) == set(config.OFFICIAL_LABELS)


# ===========================================================================
# the retraining lock
# ===========================================================================
@pytest.fixture
def step11_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STEP11_OUTPUT_DIR", tmp_path / "step11")
    monkeypatch.setattr(config, "STEP11B_OUTPUT_DIR",
                        tmp_path / "step11_attention", raising=False)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True)
    return tmp_path


def write_metrics(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "metrics.json"
    path.write_text(json.dumps({"full": {"f1_macro": 0.5}}), encoding="utf-8")
    return path


def test_the_lock_refuses_when_metrics_already_exist(step11_dirs):
    write_metrics(step11_dirs / "step11")
    calls = []

    with pytest.raises(tl.TrainingLocked) as error:
        tl.guard("approach1", lambda: calls.append("trained"))

    assert calls == []
    message = str(error.value)
    assert "LOCK_TRAINED_MODELS is True" in message
    assert "metrics.json" in message
    assert "set LOCK_TRAINED_MODELS = False" in message
    assert "OVERWRITE_* flag does not override this" in message


def test_the_lock_allows_training_when_there_are_no_metrics(step11_dirs):
    assert tl.is_trained("approach1") is False
    assert tl.locked("approach1") is False
    assert tl.guard("approach1", lambda: "trained") == "trained"


def test_the_lock_ignores_the_overwrite_flags(step11_dirs, monkeypatch):
    write_metrics(step11_dirs / "step11")
    monkeypatch.setattr(config, "OVERWRITE_STEP10", True, raising=False)

    with pytest.raises(tl.TrainingLocked):
        tl.guard("approach1", lambda: "trained")


def test_turning_the_lock_off_deliberately_allows_the_retrain(step11_dirs,
                                                              monkeypatch):
    write_metrics(step11_dirs / "step11")
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", False)

    assert tl.locked("approach1") is False
    assert tl.guard("approach1", lambda: "trained") == "trained"


def test_the_lock_covers_approach_2_separately(step11_dirs):
    write_metrics(step11_dirs / "step11_attention")

    assert tl.locked("approach2") is True
    assert tl.locked("approach1") is False
    with pytest.raises(tl.TrainingLocked):
        tl.guard("approach2", lambda: "trained")


def test_the_lock_table_says_what_would_happen_to_each_model(step11_dirs):
    write_metrics(step11_dirs / "step11")

    table = tl.lock_table()
    by_model = dict(zip(table["model"], table["running it now would"]))

    assert len(table) == 2
    # no Step 10 model is saved in this tmp_path, so the call would TRAIN --
    # and for Approach 1, whose metrics exist, that is refused
    assert by_model["Approach 1 (CNN-LSTM)"] == "be REFUSED (it would retrain)"
    assert by_model["Approach 2 (CNN-Attention-LSTM)"].startswith("TRAIN")
    assert list(table["trained by"]) == ["models.run_step10()",
                                         "models_attention.run_step10b()"]


def test_install_wraps_the_trainers_and_uninstall_restores_them(step11_dirs):
    from src.mscad import models as md
    from src.mscad import models_attention as ma

    write_metrics(step11_dirs / "step11")
    original = md.run_step10
    try:
        tl.install(verbose=False)
        assert md.run_step10 is not original
        assert md.run_step10.__wrapped__ is original
        assert md.run_step10.__name__ == "run_step10"
        with pytest.raises(tl.TrainingLocked):
            md.run_step10(verbose=False)       # a DIRECT call is guarded too
    finally:
        tl.uninstall(verbose=False)
    assert md.run_step10 is original
    assert not hasattr(md.run_step10, "_mscad_locked")


def test_step_11_is_not_locked_because_it_evaluates_rather_than_trains():
    """Re-scoring a saved model is how you check it, so it must stay possible."""
    assert set(tl.TARGETS) == {"approach1", "approach2"}
    for target in tl.TARGETS.values():
        assert target["function"] in ("run_step10", "run_step10b")
        assert "run_step11" not in target["function"]


# ===========================================================================
# the lock refuses a RETRAIN, not a LOAD
# ===========================================================================
# run_step10 and run_step10b are load-or-train functions: with their OVERWRITE_*
# flag off and their outputs on disk they call load_step10() and touch nothing.
# The wrapper used to refuse before that decision, so a cell that would have
# harmlessly loaded raised instead -- and told the reader to change a flag that
# was not the problem.
@pytest.fixture
def saved_approach1(step11_dirs, monkeypatch):
    """Approach 1 finished: its Step 10 model and its Step 11 metrics exist."""
    from src.mscad import models as md

    write_metrics(step11_dirs / "step11")
    monkeypatch.setattr(md, "step10_outputs_exist", lambda: True)
    monkeypatch.setattr(config, "OVERWRITE_STEP10", False, raising=False)
    return step11_dirs


def test_a_call_that_would_only_load_is_not_refused(saved_approach1):
    """The distinction the lock turns on: locked, but this call does not train."""
    assert tl.locked("approach1") is True          # the metrics do exist
    assert tl.saved_model_exists("approach1") is True
    assert tl.would_train("approach1") is False    # so this call only loads


def test_the_installed_wrapper_lets_a_load_through(saved_approach1):
    from src.mscad import models as md

    calls = []
    monkeypatch_target = md.run_step10
    try:
        tl.install(verbose=False)
        md.run_step10.__wrapped__          # the original is still reachable
        # replace the ORIGINAL inside the wrapper's closure by re-installing
        # over a stub, so nothing real is trained in a test
        tl.uninstall(verbose=False)
        md.run_step10 = lambda **kwargs: calls.append(kwargs) or {
            "status": "loaded_existing"}
        tl.install(verbose=False)

        out = md.run_step10(verbose=False)

        assert out["status"] == "loaded_existing"
        assert len(calls) == 1
    finally:
        tl.uninstall(verbose=False)
        md.run_step10 = monkeypatch_target


def test_the_wrapper_still_refuses_an_explicit_overwrite(saved_approach1):
    from src.mscad import models as md

    original = md.run_step10
    try:
        md.run_step10 = lambda **kwargs: {"status": "trained"}
        tl.install(verbose=False)

        with pytest.raises(tl.TrainingLocked):
            md.run_step10(overwrite=True, verbose=False)
    finally:
        tl.uninstall(verbose=False)
        md.run_step10 = original


def test_the_wrapper_reads_overwrite_as_the_first_positional_argument(
        saved_approach1):
    """run_step10(True) means overwrite=True, and must be refused too."""
    from src.mscad import models as md

    original = md.run_step10
    try:
        md.run_step10 = lambda *args, **kwargs: {"status": "trained"}
        tl.install(verbose=False)

        with pytest.raises(tl.TrainingLocked):
            md.run_step10(True)
    finally:
        tl.uninstall(verbose=False)
        md.run_step10 = original


def test_the_overwrite_flag_in_config_makes_the_call_a_retrain(saved_approach1,
                                                               monkeypatch):
    monkeypatch.setattr(config, "OVERWRITE_STEP10", True, raising=False)

    assert tl.would_train("approach1") is True
    assert tl.locked("approach1") is True


def test_a_missing_model_makes_the_call_a_train_whatever_the_flag(step11_dirs,
                                                                  monkeypatch):
    from src.mscad import models as md

    monkeypatch.setattr(md, "step10_outputs_exist", lambda: False)
    monkeypatch.setattr(config, "OVERWRITE_STEP10", False, raising=False)

    assert tl.would_train("approach1") is True


def test_the_lock_table_distinguishes_load_from_refuse(saved_approach1):
    table = tl.lock_table()
    row = table[table["model"] == "Approach 1 (CNN-LSTM)"].iloc[0]

    assert bool(row["saved metrics exist"]) is True
    assert bool(row["saved model exists"]) is True
    assert row["running it now would"] == \
        "LOAD the saved model (it trains nothing)"
    assert row["overwrite flag"] == "OVERWRITE_STEP10 = False"
    # one column for both rows, so neither is NaN
    assert "OVERWRITE_STEP10B" not in table.columns


def test_the_lock_table_says_refused_when_the_call_would_retrain(
        saved_approach1, monkeypatch):
    monkeypatch.setattr(config, "OVERWRITE_STEP10", True, raising=False)

    table = tl.lock_table()
    row = table[table["model"] == "Approach 1 (CNN-LSTM)"].iloc[0]

    assert row["running it now would"] == "be REFUSED (it would retrain)"


def test_guard_is_still_an_explicit_training_request(saved_approach1):
    """guard() means "train this now", so it refuses whenever the lock is on."""
    with pytest.raises(tl.TrainingLocked):
        tl.guard("approach1", lambda: "trained")
