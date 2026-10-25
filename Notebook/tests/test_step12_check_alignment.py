r"""
tests/test_step12_check_alignment.py -- the two Step 12 checks compare like
with like.

NEW FILE.  Nothing is trained and nothing is written outside tmp_path; the
output root is redirected with monkeypatch.  No pathlib patching.

Two checks were reported [FAIL] once Approach 1's row started rendering, and
neither failure meant what it said:

  * "table against disk (1e-9)" rebuilt each row from the RAW metrics.json.
    Step 11 never recorded Approach 1's parameter count, latent width or seed
    -- `comparison._with_descriptors` reads those from step10\\model_card.json
    and step6's selected_features.json -- so the check compared 50,438
    parameters against `null`, called the difference infinite, and said the
    table disagreed with disk when every number in it came from disk.
  * "headline numbers against the published figures" compared the official
    dataset's results with figures measured on the OWN EXTRACTION. Those rows
    cannot reproduce them: six classes instead of seven, a different label
    distribution. A red cross there meant only "this is the other dataset".

What is pinned down here: the enrichment is applied on both sides, the check
still fails when a real metric moves, the own dataset's figures still apply in
own mode, and the official dataset reports [SKIP] with the reason until its
own figures are recorded.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests/test_step12_check_alignment.py -v
"""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp

PROPOSED = "proposed"

# Approach 1 as Step 11 writes it: scores, and null where Step 11 records
# nothing.  The values are the shape of the file, not anybody's results.
STEP11_METRICS = {
    "full": {"rows": 100, "accuracy": 0.9, "precision_macro": 0.6,
             "recall_macro": 0.8, "f1_macro": 0.61, "f1_weighted": 0.94,
             "g_mean": 0.83, "roc_auc": {"macro": 0.98}},
    "duplicate_free": {"f1_macro": 0.58, "g_mean": 0.83},
    "binary": {"f1": 0.94, "false_positive_rate": 0.42},
    "inference": {"ms_per_1000_flows": 54.5},
    "parameters": None, "training_seconds": 354.1,
    "seed": None, "best_epoch": None,
}
MODEL_CARD = {"input_shape_per_sample": [15, 1],
              "architecture": {"total_parameters": 50438},
              "training": {"seed": 42, "best_epoch": 9,
                           "training_seconds": 354.1}}


def put(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def official(tmp_path, monkeypatch):
    """Official mode, with Approach 1's Step 11 result and its Step 10 card."""
    root = tmp_path / "outputs_official"
    # STEP12_EXPECTED_ROWS is chosen ONCE, when config is imported, from
    # RUNNING_OFFICIAL -- and this suite imports config with
    # MSCAD_DATASET_SOURCE=own (tests/conftest.py).  So a test that moves to
    # official mode has to set the dict as well; the notebook, which imports
    # config in official mode, gets STEP12_EXPECTED_ROWS_OFFICIAL directly.
    for name, value in (("DATASET_SOURCE", "official"),
                        ("RUNNING_OFFICIAL", True),
                        ("STEP12_EXPECTED_ROWS",
                         dict(config.STEP12_EXPECTED_ROWS_OFFICIAL)),
                        ("OUTPUTS_ROOT", root), ("OUTPUTS_DIR", root)):
        monkeypatch.setattr(config, name, value, raising=False)
    for name, folder in (("STEP6_OUTPUT_DIR", root / "step6"),
                         ("STEP10_OUTPUT_DIR", root / "step10"),
                         ("STEP11_OUTPUT_DIR", root / "step11"),
                         ("STEP11B_OUTPUT_DIR", root / "step11_attention"),
                         ("STEP12_OUTPUT_DIR", root / "step12"),
                         ("STEP12_B6_OUTPUT_DIR", root / "step12_baseline6")):
        monkeypatch.setattr(config, name, folder, raising=False)
    put(Path(cmp.result_path(PROPOSED)), STEP11_METRICS)
    put(Path(config.STEP10_OUTPUT_DIR) / "model_card.json", MODEL_CARD)
    put(Path(config.STEP6_OUTPUT_DIR) / config.STEP6_FILES["selected"],
        {"n_selected": 30})
    return root


# ===========================================================================
# 1. the disk check
# ===========================================================================
def test_the_disk_check_passes_for_a_row_whose_descriptors_came_from_step10(
        official):
    built = cmp.build_full_comparison(verbose=False)
    row = built["table"].iloc[0]
    # the descriptor columns are filled, and not from Step 11's file
    assert row["params"] == 50438
    assert STEP11_METRICS["parameters"] is None

    table = cmp.verify_rows_against_disk(built["table"])
    line = table[table["config"].str.startswith("Approach 1")].iloc[0]

    assert line["result"] == "[OK]", line["largest difference"]
    assert line["largest difference"] == 0.0
    assert line["columns compared"] >= 12


def test_the_disk_check_still_fails_when_a_real_metric_moves(official):
    """The guard is not weakened: a changed file is still caught."""
    built = cmp.build_full_comparison(verbose=False)
    moved = dict(STEP11_METRICS)
    moved["full"] = {**STEP11_METRICS["full"], "f1_macro": 0.42}
    put(Path(cmp.result_path(PROPOSED)), moved)    # the file changes, not the row

    table = cmp.verify_rows_against_disk(built["table"])
    line = table[table["config"].str.startswith("Approach 1")].iloc[0]

    assert line["result"] == "[FAIL]"
    assert line["largest difference"] > 1e-9


# ===========================================================================
# 2. the published figures
# ===========================================================================
def test_the_official_dataset_skips_the_published_figures_with_a_reason(
        official):
    built = cmp.build_full_comparison(verbose=False)

    table, ok = cmp.check_rows_unchanged(built, verbose=False)
    line = table[table["check"] == "published figures"].iloc[0]

    assert line["result"] == "[SKIP]"
    assert "no figures are recorded for the official dataset" in line["detail"]
    assert "STEP12_EXPECTED_ROWS_OFFICIAL" in line["detail"]
    assert ok is True                              # a [SKIP] is not a [FAIL]
    # and no own-extraction row is compared against an official result
    assert not table["check"].str.startswith("published figures:").any()


def test_the_own_dataset_still_checks_its_published_figures(official,
                                                            monkeypatch):
    """Own mode keeps the guard, and still fails when a number moves."""
    monkeypatch.setattr(config, "DATASET_SOURCE", "own", raising=False)
    monkeypatch.setattr(config, "STEP12_EXPECTED_ROWS",
                        {PROPOSED: {"accuracy": 0.9896, "f1_macro": 0.7170}})
    built = cmp.build_full_comparison(verbose=False)

    table, ok = cmp.check_rows_unchanged(built, verbose=False)
    line = table[table["check"] == "published figures: proposed"].iloc[0]

    assert line["result"] == "[FAIL]"              # 0.9 is not 0.9896
    assert "want 0.9896" in line["detail"]
    assert ok is False


def test_published_figures_reads_the_one_live_dict(official, monkeypatch):
    """One dict, so a caller or a test can still override it in one place."""
    monkeypatch.setattr(config, "STEP12_EXPECTED_ROWS",
                        {PROPOSED: {"accuracy": 0.2}})
    assert cmp.published_figures() == {PROPOSED: {"accuracy": 0.2}}

    monkeypatch.setattr(config, "STEP12_EXPECTED_ROWS", {})
    assert cmp.published_figures() == {}


def test_the_own_figures_in_config_are_kept_as_they_were():
    """The own extraction's published numbers are not edited by this change."""
    assert config.STEP12_EXPECTED_ROWS_OWN[PROPOSED] == {
        "accuracy": 0.9896, "f1_macro": 0.7170, "g_mean": 0.7505}
    assert len(config.STEP12_EXPECTED_ROWS_OWN) == 5
    assert config.STEP12_EXPECTED_ROWS_OFFICIAL == {}
