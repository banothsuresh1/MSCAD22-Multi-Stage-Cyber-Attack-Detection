r"""
tests/test_proposed_lookup.py -- where each configuration's metrics.json lives.

NEW FILE.  Nothing is trained and nothing is written outside tmp_path; the
output root is redirected with monkeypatch.  No pathlib patching.

Three configurations are NOT written by the Step 12 runner, and each needs its
own branch in `comparison.result_path`:

    Approach 1  "proposed"                   step11\\metrics.json
                                             (evaluation.run_step11)
    Approach 2  config.APPROACH2_KEY         step11_attention\\metrics.json
                                             (models_attention.run_step11b)
    Baseline 6  config.BASELINE6_NAME        step12_baseline6\\metrics.json

Everything else is step12\\<key>\\metrics.json.  Approach 1's branch is the one
that was missing: before PR #46 the "proposed" key fell through to the generic
line and was looked for at step12\\proposed\\metrics.json, which the pipeline
never writes -- so the row that the whole comparison is ABOUT rendered as
"not yet trained - run 10.2" while all seven others rendered correctly.  These
tests exist so that cannot come back.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests/test_proposed_lookup.py -v
"""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp

PROPOSED = "proposed"
APPROACH2 = config.APPROACH2_KEY
BASELINE6 = config.BASELINE6_NAME
BASELINES = [key for key, _label in config.STEP12_DISPLAY_ORDER
             if key not in {PROPOSED, APPROACH2, BASELINE6}]


def payload(value: float, features: int = 30) -> dict:
    return {"full": {"accuracy": value, "precision_macro": value,
                     "recall_macro": value, "f1_macro": value,
                     "f1_weighted": value, "g_mean": value,
                     "roc_auc_macro_ovr": value},
            "binary": {"f1": value, "fpr": 0.01},
            "dup_free": {"f1_macro": value, "g_mean": value},
            "n_input_features": features, "latent_dim": 15,
            "parameters": 50438, "training_seconds": 1.0, "best_epoch": 1,
            "inference": {"ms_per_1000_flows": 1.0}}


def put(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload(value)), encoding="utf-8")
    return path


@pytest.fixture
def official(tmp_path, monkeypatch):
    """DATASET_SOURCE = "official", with the output root inside tmp_path."""
    root = tmp_path / "outputs_official"
    for name, value in (("DATASET_SOURCE", "official"),
                        ("RUNNING_OFFICIAL", True),
                        ("OUTPUTS_ROOT", root), ("OUTPUTS_DIR", root)):
        monkeypatch.setattr(config, name, value, raising=False)
    for name, folder in (("STEP11_OUTPUT_DIR", root / "step11"),
                         ("STEP11B_OUTPUT_DIR", root / "step11_attention"),
                         ("STEP12_OUTPUT_DIR", root / "step12"),
                         ("STEP12_CONFIG_DIR", root / "step12"),
                         ("STEP12_B6_OUTPUT_DIR", root / "step12_baseline6")):
        monkeypatch.setattr(config, name, folder, raising=False)
    root.mkdir(parents=True, exist_ok=True)
    return root


# ===========================================================================
# 1. the three configurations with a folder of their own
# ===========================================================================
def test_result_path_proposed_points_at_step11_on_official(official):
    expected = (Path(config.STEP11_OUTPUT_DIR)
                / config.STEP11_FILES["metrics"])

    assert Path(cmp.result_path(PROPOSED)) == expected
    # and NOT the path the generic line would build
    assert Path(cmp.result_path(PROPOSED)) != (
        Path(config.STEP12_OUTPUT_DIR) / PROPOSED / "metrics.json")


def test_the_proposed_row_loads_from_step11(official):
    put(Path(config.STEP11_OUTPUT_DIR) / config.STEP11_FILES["metrics"], 0.9896)

    metrics = cmp.load_result(PROPOSED)

    assert metrics is not None
    assert metrics["full"]["f1_macro"] == 0.9896


def test_a_step12_copy_of_the_proposed_result_is_preferred_when_it_exists(
        official):
    """It carries the descriptor fields as well as the metrics."""
    put(Path(config.STEP11_OUTPUT_DIR) / config.STEP11_FILES["metrics"], 0.11)
    step12_copy = put(
        Path(config.STEP12_OUTPUT_DIR) / PROPOSED / "metrics.json", 0.22)

    assert Path(cmp.result_path(PROPOSED)) == step12_copy
    assert cmp.load_result(PROPOSED)["full"]["f1_macro"] == 0.22


def test_result_path_approach2_points_at_step11_attention_on_official(official):
    expected = Path(config.STEP11B_OUTPUT_DIR) / "metrics.json"

    assert Path(cmp.result_path(APPROACH2)) == expected
    assert "step11_attention" in Path(cmp.result_path(APPROACH2)).parts


def test_result_path_baseline6_points_at_its_own_folder(official):
    expected = Path(config.STEP12_B6_OUTPUT_DIR) / "metrics.json"

    assert Path(cmp.result_path(BASELINE6)) == expected


# ===========================================================================
# 2. every other key is unchanged
# ===========================================================================
@pytest.mark.parametrize("key", BASELINES)
def test_result_path_baselines_unchanged(key, official):
    expected = Path(config.STEP12_OUTPUT_DIR) / key / "metrics.json"

    assert Path(cmp.result_path(key)) == expected


def test_every_key_resolves_under_the_active_output_root(official):
    for key, _label in config.STEP12_DISPLAY_ORDER:
        path = cmp.result_path(key)
        assert path is not None, key
        assert cmp.in_active_root(path), key


# ===========================================================================
# 3. the whole table, with every file in place
# ===========================================================================
def write_all_eight(value: float = 0.9) -> None:
    for key, _label in config.STEP12_DISPLAY_ORDER:
        put(Path(cmp.result_path(key)), value)


def test_step10_table_shows_all_eight_rows_when_files_exist(official):
    write_all_eight()

    built = cmp.build_full_comparison(verbose=False)
    table = built["table"]

    assert len(table) == 8
    assert built["missing"] == []
    live = cmp.trained_rows(table)
    assert bool(live.all()), table.loc[~live, "config"].tolist()
    assert table["accuracy"].notna().all()
    text = table["input features"].astype(str)
    assert not text.str.contains("not yet trained").any()
    assert not text.str.contains("not trained on").any()


def test_the_honest_table_lists_all_eight_as_trained_when_files_exist(official):
    write_all_eight()

    honest = cmp.honest_comparison(verbose=False)

    assert len(honest["trained_keys"]) == 8
    assert honest["pending_keys"] == []


def test_only_the_proposed_row_would_be_missing_without_its_branch(official):
    """The bug's signature: seven rows render, Approach 1 does not.

    The pre-PR-#46 lookup is spelled out here rather than imported, so this
    test keeps describing the failure even after the old code is gone.
    """
    write_all_eight()
    # the path the generic line builds, which the pipeline never writes: with
    # the branch in place, writing all eight results did not create it
    old_path = Path(config.STEP12_OUTPUT_DIR) / PROPOSED / "metrics.json"
    assert not old_path.exists()

    built = cmp.build_full_comparison(verbose=False)

    # and Approach 1 renders anyway -- from step11\metrics.json
    assert built["missing"] == []
    assert Path(cmp.result_path(PROPOSED)) == (
        Path(config.STEP11_OUTPUT_DIR) / config.STEP11_FILES["metrics"])
