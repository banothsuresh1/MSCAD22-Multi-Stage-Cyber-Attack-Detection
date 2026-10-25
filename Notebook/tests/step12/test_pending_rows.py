r"""
tests/step12/test_pending_rows.py -- a missing result renders, it does not crash.

NEW FILE.  Nothing is trained and nothing is written outside tmp_path; the two
output roots are built there and config is redirected with monkeypatch.  No
pathlib patching; Windows paths are PureWindowsPath.

B1-B6 have never been trained on the official dataset -- only approach1\\ and
approach2\\ exist under outputs_official\\step12\\.  `run_one` used to raise
Step12ConfigError for each of them, which took down the whole Step 10 cell, so
the two results that DO exist could not be seen either.  A missing result is a
PENDING row now.

What is pinned down:

  * with LOCK_TRAINED_MODELS = True and no metrics.json, `run_one` returns a
    row with status "pending" whose every headline value reads "not trained on
    official" -- and calls neither the loader nor the trainer;
  * the row is shaped like a real one (`metrics` a dict, a one-row protocol and
    snapshot table), so `report.status_table` renders it;
  * `ok` is False and `verdict` is "PENDING": a pending row is neither a pass
    nor a failure;
  * EXISTING BEHAVIOUR, unchanged: the own root's result for the same
    configuration is still never read, and a misconfigured DATASET_SOURCE or a
    missing output root still raises rather than rendering eight pending rows.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests/step12/test_pending_rows.py -v
"""

import json
import os
from pathlib import Path, PureWindowsPath

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp
from src.mscad.experiments_step12 import common
from src.mscad.experiments_step12 import report as rep

B1 = "baseline1_ae_cnn_lstm"
PROPOSED = "proposed"


def payload(value: float) -> dict:
    return {"full": {"accuracy": value, "precision_macro": value,
                     "recall_macro": value, "f1_macro": value,
                     "f1_weighted": value, "g_mean": value},
            "binary": {"false_positive_rate": 0.01},
            "duplicate_free": {"f1_macro": value},
            "training_seconds": 1.0, "best_epoch": 1, "parameters": 1000}


def put(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload(value)), encoding="utf-8")
    return path


@pytest.fixture
def official_root(tmp_path, monkeypatch):
    """The state on disk: Approach 1 and Approach 2 only, own root complete."""
    official = tmp_path / "outputs_official"
    own = tmp_path / "outputs"
    for name, value in (("DATASET_SOURCE", "official"),
                        ("RUNNING_OFFICIAL", True),
                        ("LOCK_TRAINED_MODELS", True),
                        ("OUTPUTS_ROOT", official),
                        ("OUTPUTS_DIR", official),
                        ("OUTPUTS_ROOT_OFFICIAL", official),
                        ("OUTPUTS_ROOT_OWN", own)):
        monkeypatch.setattr(config, name, value, raising=False)
    for name, folder in (("STEP11_OUTPUT_DIR", official / "step11"),
                         ("STEP11B_OUTPUT_DIR", official / "step11_attention"),
                         ("STEP12_OUTPUT_DIR", official / "step12"),
                         ("STEP12_CONFIG_DIR", official / "step12"),
                         ("STEP12_B6_OUTPUT_DIR",
                          official / "step12_baseline6")):
        monkeypatch.setattr(config, name, folder, raising=False)

    put(own / "step12" / B1 / "metrics.json", 0.68)     # the tempting file
    put(official / "step11" / "metrics.json", 0.91)
    official.mkdir(parents=True, exist_ok=True)
    return tmp_path


def refuse(**_kwargs):
    raise AssertionError("neither the loader nor the trainer may be called")


# ===========================================================================
# 1. the new behaviour: a pending row, not an exception
# ===========================================================================
def test_a_missing_result_renders_as_a_pending_row(official_root):
    result = common.run_one(B1, loader=refuse, trainer=refuse, verbose=False)

    assert result["status"] == common.PENDING
    assert result["pending"] is True
    assert result["ok"] is False
    assert common.verdict(result) == "PENDING"
    assert "not trained on official" in result["status_detail"]
    values = set(result["headline"]["value"])
    assert values == {"not trained on official"}


def test_the_pending_row_is_shaped_like_a_real_one(official_root):
    result = common.run_one(B1, loader=refuse, verbose=False)

    assert isinstance(result["metrics"], dict)
    assert (result["metrics"].get("full") or {}).get("f1_macro") is None
    assert len(result["protocol"]) == 1
    assert result["protocol"].iloc[0]["result"] == common.SKIP
    assert len(result["unchanged"]) == 1
    assert result["unchanged"].iloc[0]["result"] == common.SKIP
    assert result["manifest"] is None          # a pending row writes nothing


def test_the_status_table_renders_a_pending_row_beside_a_loaded_one(
        official_root):
    pending = common.run_one(B1, loader=refuse, verbose=False)
    loaded = common.run_one(PROPOSED, loader=lambda verbose=True: {},
                            verbose=False)

    table = rep.status_table({"baseline1": pending, "approach1": loaded})

    assert len(table) == 2
    row = table[table["file"] == "baseline1.py"].iloc[0]
    assert row["status"] == common.PENDING
    assert "not trained on official" in row["detail"]
    import pandas as pd
    assert pd.isna(row["macro-F1"])            # and no own-root 0.68


def test_the_cli_does_not_report_a_pending_configuration_as_a_failure(
        official_root, capsys):
    class Module:
        __name__ = "src.mscad.experiments_step12.baseline1"
        __doc__ = "Baseline 1."
        KEY = B1
        DISPLAY_LABEL = common.display_label(B1)
        COMPONENTS: dict = {}

        @staticmethod
        def run(verbose: bool = True):
            return common.run_one(B1, loader=refuse, verbose=False)

    code = common.cli(Module, ["--quiet"])

    assert code == 0                           # pending is not exit 1
    assert "[PENDING]" in capsys.readouterr().out


# ===========================================================================
# 2. existing behaviour, unchanged
# ===========================================================================
def test_the_pending_row_still_never_reads_the_other_roots_result(
        official_root):
    elsewhere = official_root / "outputs" / "step12" / B1 / "metrics.json"
    assert elsewhere.exists()

    result = common.run_one(B1, loader=refuse, verbose=False)

    assert cmp.load_result(B1) is None
    assert Path(result["metrics_path"]) != elsewhere
    assert cmp.in_active_root(result["metrics_path"])
    assert "0.68" not in json.dumps(result["metrics"])


def test_a_misconfigured_dataset_source_still_raises(official_root,
                                                     monkeypatch):
    monkeypatch.setattr(config, "DATASET_SOURCE", "offical", raising=False)

    with pytest.raises(common.Step12ConfigError) as error:
        common.run_one(B1, loader=refuse, verbose=False)

    assert "DATASET_SOURCE is 'offical'" in str(error.value)


def test_a_root_that_is_not_on_this_machine_still_raises(official_root,
                                                         monkeypatch):
    """The Windows-path guard: unreadable here is not the same as missing.

    `running_on_windows` is patched off so the guard fires on Windows too --
    there a C:\ path is perfectly readable, and the test asserted a refusal
    that only ever happened on Linux.
    """
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs_official"
                                        r"\step12"), raising=False)

    with pytest.raises(common.Step12ConfigError) as error:
        common.run_one(B1, loader=refuse, verbose=False)

    assert "cannot be read on this machine" in str(error.value)


def test_the_lock_message_is_still_there_for_the_callers_that_want_it(
        official_root):
    """`refuse_missing` is unchanged -- run_one just no longer calls it."""
    with pytest.raises(common.Step12ConfigError) as error:
        common.refuse_missing(B1)

    message = str(error.value)
    assert "no saved result in the official output root" in message
    assert "LOCK_TRAINED_MODELS is True" in message


# ===========================================================================
# 4. the Windows regression: an absent root is PENDING, not a crash
# ===========================================================================
def test_an_absent_but_readable_output_root_is_pending_not_an_error(
        tmp_path, monkeypatch):
    """What Windows sees when only outputs_official\\ exists.

    On Linux `C:\\MSCAD2022\\outputs` is refused by the Windows-path guard, so
    `refuse_if_the_root_is_not_a_root` never looked at it.  On Windows the path
    IS readable and simply does not exist -- which is "nothing has been run
    against this dataset", the pending case, not a misconfiguration.  Raising
    there took down all eight Step 10 sections.
    """
    absent = tmp_path / "outputs"                  # readable, does not exist
    monkeypatch.setattr(config, "DATASET_SOURCE", "own", raising=False)
    monkeypatch.setattr(config, "RUNNING_OFFICIAL", False, raising=False)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True, raising=False)
    monkeypatch.setattr(config, "OUTPUTS_ROOT", absent, raising=False)
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR", tmp_path / "redirected",
                        raising=False)
    assert not absent.exists()

    result = common.run_one(B1, loader=refuse, trainer=refuse, verbose=False)

    assert result["status"] == common.PENDING
    assert result["ok"] is False
