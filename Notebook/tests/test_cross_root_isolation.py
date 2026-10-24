r"""
tests/test_cross_root_isolation.py -- one dataset per table.

NEW FILE.  Nothing is trained and nothing is written outside tmp_path; the two
output roots are built there and config is redirected with monkeypatch.  No
pathlib patching: Windows paths, where they appear at all, are PureWindowsPath.

Each dataset has its own output root -- `outputs\\` for the own extraction,
`outputs_official\\` for the published MSCAD.csv -- and a configuration can be
finished in one and never trained in the other.  A row read from the other root
would not be stale, it would be FALSE: the two datasets have different class
counts (6 vs 7) and different label distributions, so the numbers are not on
one scale.

What is pinned down here:

  * with the own root fully populated and the official root holding only
    Approach 1 and Approach 2, every baseline reads as "not trained on
    official" and no baseline's path leaves the official root;
  * `comparison.in_active_root` compares by path PARTS, so the own root is
    never mistaken for the official one ("outputs" is a prefix of
    "outputs_official");
  * the Step 12 loaders and the lock's refusal name the other root but never
    read a metric from it.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests/test_cross_root_isolation.py -v
"""

import json
import os
from pathlib import Path, PureWindowsPath

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp
from src.mscad.experiments_step12 import common

B1 = "baseline1_ae_cnn_lstm"
PROPOSED = "proposed"
APPROACH2 = config.APPROACH2_KEY


def payload(value: float) -> dict:
    """Enough of a metrics.json for a row to be built from it."""
    return {"full": {"accuracy": value, "precision_macro": value,
                     "recall_macro": value, "f1_macro": value,
                     "f1_weighted": value, "g_mean": value,
                     "roc_auc_macro_ovr": value},
            "binary": {"f1": value, "fpr": 0.01},
            "dup_free": {"f1_macro": value, "g_mean": value},
            "training_seconds": 1.0, "best_epoch": 1,
            "inference": {"ms_per_1000_flows": 1.0}}


def put(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload(value)), encoding="utf-8")
    return path


@pytest.fixture
def two_roots(tmp_path, monkeypatch):
    """Official mode: a full own root, an official root with the two approaches.

    This is the state on disk today -- Steps 3-11 and only approach1\\ and
    approach2\\ under the official step12\\ -- with the own root deliberately
    complete, so a fallback would have something to find.
    """
    official = tmp_path / "outputs_official"
    own = tmp_path / "outputs"
    for name, value in (("DATASET_SOURCE", "official"),
                        ("RUNNING_OFFICIAL", True),
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

    # the own root, complete: every configuration finished on the OTHER dataset
    for key, _label in config.STEP12_DISPLAY_ORDER:
        put(own / "step12" / key / "metrics.json", 0.68)
    put(own / "step11" / "metrics.json", 0.68)
    put(own / "step11_attention" / "metrics.json", 0.68)
    put(own / "step12_baseline6" / "metrics.json", 0.68)
    # the official root: the two approaches only
    put(official / "step11" / "metrics.json", 0.91)
    put(official / "step11_attention" / "metrics.json", 0.93)
    return tmp_path


# ===========================================================================
# 1. the loader never reaches into the other root
# ===========================================================================
def test_baseline1_is_not_trained_here_although_the_own_root_has_it(two_roots):
    """The case the brief names: outputs\\step12\\baseline1\\ exists on disk."""
    elsewhere = two_roots / "outputs" / "step12" / B1 / "metrics.json"
    assert elsewhere.exists()                      # the tempting file

    assert cmp.load_result(B1) is None
    path = Path(cmp.result_path(B1))
    assert cmp.in_active_root(path)
    assert path != elsewhere
    assert path.is_relative_to(two_roots / "outputs_official")
    assert not path.is_relative_to(two_roots / "outputs")


def test_every_baselines_path_stays_under_the_active_root(two_roots):
    here, pending = cmp.results_in_active_root()

    assert here == [PROPOSED, APPROACH2]
    assert len(pending) == 6
    for key in pending:
        assert cmp.load_result(key) is None, key
        assert cmp.in_active_root(cmp.result_path(key)), key


def test_the_honest_table_marks_the_baselines_not_trained_on_official(
        two_roots):
    honest = cmp.honest_comparison(verbose=False)

    assert honest["source"] == "official"
    assert honest["trained_keys"] == [PROPOSED, APPROACH2]
    text = honest["marked"]["input features"].astype(str).tolist()
    assert text[2:] == ["not trained on official"] * 6
    # and not one own-root number reached the table
    numbers = honest["table"]["accuracy"].dropna().tolist()
    assert 0.68 not in numbers
    assert len(numbers) == 2


def test_the_trained_sub_table_holds_the_two_approaches_only(two_roots):
    honest = cmp.honest_comparison(verbose=False)
    pair = cmp.trained_only_table(honest)

    assert len(pair) == 2
    assert list(pair.columns) == cmp.COLUMNS
    assert pair["config"].str.startswith("Approach").all()


# ===========================================================================
# 2. the root test is by path parts, not by text
# ===========================================================================
def test_the_own_root_is_not_read_as_the_official_root(monkeypatch):
    """"outputs" is a prefix of "outputs_official" -- as text, not as parts."""
    official = PureWindowsPath(r"C:\MSCAD2022\outputs_official")
    monkeypatch.setattr(config, "OUTPUTS_ROOT", official, raising=False)

    assert cmp.in_active_root(official / "step11" / "metrics.json")
    assert not cmp.in_active_root(
        PureWindowsPath(r"C:\MSCAD2022\outputs\step11\metrics.json"))
    assert not cmp.in_active_root(
        PureWindowsPath(r"C:\MSCAD2022\outputs\step12\baseline1_ae_cnn_lstm"
                        r"\metrics.json"))
    assert not cmp.in_active_root(None)


# ===========================================================================
# 3. the Step 12 refusal names the other root without reading it
# ===========================================================================
def test_the_lock_refuses_and_never_substitutes_the_other_roots_result(
        two_roots, monkeypatch):
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True, raising=False)

    with pytest.raises(common.Step12ConfigError) as error:
        common.refuse_missing(B1)

    message = str(error.value)
    assert "no saved result in the official output root" in message
    assert "the own root HAS it" in message
    assert cmp.load_result(B1) is None          # still not loaded from there
