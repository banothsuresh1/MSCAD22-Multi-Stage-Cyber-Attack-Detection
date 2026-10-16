r"""
tests/step12/test_baseline1.py -- Baseline 1 (AE -> CNN-LSTM, no feature selection).

NEW FILE.  What it checks is the same for every load-only configuration: the
file LOADS and never trains while a saved result exists, the numbers it returns
are the saved ones, the training path it does have is the EXISTING function,
and the module runs on its own.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp
from src.mscad import experiments as xp
from src.mscad.experiments_step12 import baseline1 as module
from src.mscad.experiments_step12 import common

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent.parent
KEY = 'baseline1_ae_cnn_lstm'


def explode(*_args, **_kwargs):                 # pragma: no cover - must not run
    raise AssertionError("the existing trainer was called; this file must LOAD")


# ===========================================================================
# the interface every configuration file exposes
# ===========================================================================
def test_the_file_exposes_the_shared_interface():
    assert module.KEY == KEY
    assert module.DISPLAY_LABEL == dict(config.STEP12_DISPLAY_ORDER)[KEY]
    assert callable(module.run)
    assert isinstance(module.COMPONENTS, dict)
    for component in ("feature selection", "autoencoder", "PCA", "CNN",
                      "attention", "LSTM", "input features",
                      "what it isolates"):
        assert component in module.COMPONENTS, component
    assert module.COMPONENTS["what it isolates"] == config.STEP12_ISOLATES[KEY]


def test_the_file_is_named_after_its_configuration():
    assert common.short_name(KEY) == "baseline1"
    assert config.STEP12_CONFIG_FILES[KEY] == "baseline1"


# ===========================================================================
# it LOADS, and never trains
# ===========================================================================
def test_run_loads_the_saved_result_and_never_trains(finished, monkeypatch):
    finished(KEY)
    monkeypatch.setattr(xp, "run_experiment", explode)

    out = module.run(verbose=False)

    assert out["status"] == common.LOADED
    assert out["status_detail"] == "loaded (locked)"
    assert out["locked"] is True


def test_the_saved_metrics_come_back_unchanged(finished, monkeypatch):
    folders = finished(KEY)
    monkeypatch.setattr(xp, "run_experiment", explode)
    on_disk = json.loads(
        (Path(cmp.result_path(KEY))).read_text(encoding="utf-8"))

    out = module.run(verbose=False)

    assert out["metrics"]["full"] == on_disk["full"]
    assert out["metrics"]["duplicate_free"] == on_disk["duplicate_free"]
    assert set(out["unchanged"]["result"]) == {"[OK]"}
    assert float(out["unchanged"]["largest difference"].max()) == 0.0


def test_the_saved_folder_is_not_written_to(finished, monkeypatch):
    folders = finished(KEY)
    monkeypatch.setattr(xp, "run_experiment", explode)
    before = {path.name: path.stat().st_mtime
              for path in folders[KEY].iterdir()}

    module.run(verbose=False)

    after = {path.name: path.stat().st_mtime
             for path in folders[KEY].iterdir()}
    assert after == before


def test_the_manifest_is_written_to_its_own_folder(finished, tmp_path,
                                                   monkeypatch):
    finished(KEY)
    monkeypatch.setattr(xp, "run_experiment", explode)

    out = module.run(verbose=False)
    folder = Path(out["manifest"]["folder"])

    assert folder == tmp_path / "step12" / "baseline1"
    payload = json.loads((folder / "manifest.json").read_text(
        encoding="utf-8"))
    assert payload["key"] == KEY
    assert payload["file"] == "src/mscad/experiments_step12/baseline1.py"
    assert payload["status"] == "loaded"
    assert payload["metrics"]["full"]["f1_macro"] is not None


def test_the_provenance_fingerprints_the_files_it_read(finished, monkeypatch):
    finished(KEY)
    monkeypatch.setattr(xp, "run_experiment", explode)

    table = module.run(verbose=False)["provenance"]
    found = table[table["exists"]]

    assert len(found) >= 2
    for _i, row in found.iterrows():
        assert len(row["sha256"]) == 64


# ===========================================================================
# the training path is the EXISTING function, and only the lock opens it
# ===========================================================================
def test_a_missing_result_with_the_lock_on_stops_without_training(step12_dirs,
                                                                   monkeypatch):
    monkeypatch.setattr(xp, "run_experiment", explode)

    with pytest.raises(common.Step12ConfigError) as error:
        module.run(verbose=False)

    message = str(error.value)
    assert "LOCK_TRAINED_MODELS is True" in message
    assert "set LOCK_TRAINED_MODELS = False in config.py" in message


def test_the_trainer_is_the_existing_function_not_a_copy(step12_dirs,
                                                          monkeypatch):
    """With the lock off and nothing saved, this file calls xp.run_experiment."""
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", False)
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return {"metrics": {"full": {"f1_macro": 0.5, "accuracy": 0.9,
                                     "g_mean": 0.4},
                            "duplicate_free": {}, "binary": {},
                            "inference": {}}}

    monkeypatch.setattr(xp, "run_experiment", fake)
    monkeypatch.setattr(common, "evaluation_inputs",
                        lambda **_kwargs: {"y_train": None, "y_test": None,
                                          "meta_test": None,
                                          "label_mapping": {},
                                          "duplicate_flags": None})

    out = module.run(verbose=False)

    assert len(calls) == 1
    assert out["status"] == common.TRAINED
    assert "no saved result existed" in out["status_detail"]


def test_the_module_source_contains_no_training_code():
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert ".fit(" not in source
    assert "keras" not in source.replace("model.keras", "")
    assert "run_experiment" in source          # it calls the existing one


# ===========================================================================
# runnable on its own
# ===========================================================================
def test_the_module_runs_as_a_script():
    finished = subprocess.run(
        [sys.executable, "-m", "src.mscad.experiments_step12.baseline1",
         "--components"],
        cwd=str(NOTEBOOK_DIR), capture_output=True, text=True, timeout=600)

    assert finished.returncode == 0, finished.stderr[-2000:]
    assert module.DISPLAY_LABEL in finished.stdout
    assert "what it isolates" in finished.stdout
