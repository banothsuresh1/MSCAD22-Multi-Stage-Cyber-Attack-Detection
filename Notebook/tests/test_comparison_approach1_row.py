r"""
tests/test_comparison_approach1_row.py -- Approach 1 appears in its own table.

NEW FILE.  Phase 1 of the audit found that the Step 10 comparison showed
Approach 1 as "not yet trained" even though Step 11 had evaluated it: a lookup
bug, not a training failure.  `comparison.result_path()` had a branch for
Approach 2 (step11_attention\) and one for Baseline 6 (step12_baseline6\) but
none for "proposed", whose evaluation Step 11 writes to step11\metrics.json.

What is pinned here:

  * result_path("proposed") resolves to Step 11's metrics.json, and prefers a
    Step 12 copy when a run of experiments.run_step12 has written one;
  * all eight configurations resolve to a path, and none of them collide;
  * the eight-row table has no "not yet trained" row when all eight files are
    on disk, and Approach 1's scores are the ones Step 11 saved;
  * the descriptor columns Step 11 never recorded are filled from the saved
    model_card.json and selected_features.json -- nothing retrained, nothing
    rewritten;
  * the best-value markings cover all eight rows;
  * the "existing rows identical" check passes at 1e-9;
  * the blank row names the step that produces THAT configuration.

Everything is synthetic and inside tmp_path.  No model is built or loaded, and
no file under C:\\MSCAD2022 is read or written.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp

PROPOSED = "proposed"


@pytest.fixture
def official_tree(tmp_path, monkeypatch):
    """Official mode with every output folder inside tmp_path."""
    root = tmp_path / "outputs_official"
    monkeypatch.setattr(config, "DATASET_SOURCE", "official")
    monkeypatch.setattr(config, "RUNNING_OFFICIAL", True)
    monkeypatch.setattr(config, "OUTPUTS_ROOT", root)
    for number in (6, 10, 11, 12):
        monkeypatch.setattr(config, "STEP%d_OUTPUT_DIR" % number,
                            root / ("step%d" % number))
    monkeypatch.setattr(config, "STEP11B_OUTPUT_DIR", root / "step11_attention")
    monkeypatch.setattr(config, "STEP12_B6_OUTPUT_DIR", root / "step12_baseline6")
    monkeypatch.setattr(config, "usable_output_dir", lambda folder, **k: Path(folder))
    monkeypatch.setattr(config, "why_unusable", lambda folder: "")
    return root


def metrics_block(f1, accuracy=0.9, g_mean=0.8, fpr=0.2, **extra):
    """One configuration's metrics.json, in the shape the readers expect."""
    full = {"accuracy": accuracy, "precision_macro": 0.7, "recall_macro": 0.7,
            "f1_macro": f1, "f1_weighted": 0.9, "g_mean": g_mean,
            "roc_auc": {"macro": 0.95}}
    block = {
        "full": full,
        "duplicate_free": {"f1_macro": f1 - 0.05, "g_mean": g_mean - 0.05},
        "binary": {"f1": 0.9, "false_positive_rate": fpr},
        "inference": {"ms_per_1000_flows": 12.5, "std_seconds": 0.01},
        "training_seconds": 30.0,
    }
    block.update(extra)
    return block


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def write_all_eight(root, proposed_in_step12=False):
    """A metrics.json for every configuration, each with a different macro-F1."""
    order = [key for key, _label in config.STEP12_DISPLAY_ORDER]
    scores = {}
    for index, key in enumerate(order):
        scores[key] = round(0.60 + index * 0.01, 4)
    for key in order:
        extra = {"n_input_features": 66, "latent_dim": 15, "parameters": 1000,
                 "seed": 42, "best_epoch": 7}
        if key == PROPOSED and not proposed_in_step12:
            # Step 11 writes the scores WITHOUT the descriptor fields
            write(root / "step11" / "metrics.json",
                  metrics_block(scores[key]))
            continue
        write(cmp.result_path(key), metrics_block(scores[key], **extra))
    return scores


# ===========================================================================
# 1. the lookup
# ===========================================================================
def test_result_path_for_approach1_is_step11s_metrics(official_tree):
    path = cmp.result_path(PROPOSED)

    assert path == official_tree / "step11" / "metrics.json", path


def test_a_step12_copy_of_approach1_wins_when_one_exists(official_tree):
    step12 = write(official_tree / "step12" / PROPOSED / "metrics.json",
                   metrics_block(0.61))

    assert cmp.result_path(PROPOSED) == step12


def test_every_configuration_resolves_to_its_own_path(official_tree):
    seen = {}
    for key, _label in config.STEP12_DISPLAY_ORDER:
        path = cmp.result_path(key)
        assert path is not None, key
        assert path.name == "metrics.json", (key, path)
        seen[key] = path

    assert len(seen) == 8, sorted(seen)
    assert len(set(seen.values())) == 8, "two configurations share a path"
    assert seen[PROPOSED].parent.name == "step11"
    assert seen[config.APPROACH2_KEY].parent.name == "step11_attention"
    assert seen[config.BASELINE6_NAME].parent.name == "step12_baseline6"


def test_approach1_is_not_found_when_step11_has_not_run(official_tree):
    assert cmp.load_result(PROPOSED) is None


# ===========================================================================
# 2. the eight-row table
# ===========================================================================
def test_all_eight_rows_are_filled_and_none_says_not_trained(official_tree):
    scores = write_all_eight(official_tree)

    built = cmp.build_full_comparison(verbose=False)
    table = built["table"]

    assert built["missing"] == [], built["missing"]
    assert len(table) == 8, len(table)
    assert not table["input features"].astype(str).str.contains(
        config.NOT_TRAINED_TEXT).any(), table["input features"].tolist()
    assert table["F1 (macro)"].notna().all()
    for key, label in config.STEP12_DISPLAY_ORDER:
        row = table[table["config"] == label]
        assert len(row) == 1, label
        assert float(row["F1 (macro)"].iloc[0]) == pytest.approx(scores[key])


def test_approach1s_row_carries_step11s_own_scores(official_tree):
    scores = write_all_eight(official_tree)
    label = dict(config.STEP12_DISPLAY_ORDER)[PROPOSED]

    table = cmp.build_full_comparison(verbose=False)["table"]
    row = table[table["config"] == label].iloc[0]

    saved = json.loads((official_tree / "step11" / "metrics.json")
                       .read_text(encoding="utf-8"))
    assert float(row["F1 (macro)"]) == pytest.approx(saved["full"]["f1_macro"])
    assert float(row["G-mean"]) == pytest.approx(saved["full"]["g_mean"])
    assert float(row["binary FPR"]) == pytest.approx(
        saved["binary"]["false_positive_rate"])
    assert float(row["F1 macro (dup-free)"]) == pytest.approx(
        saved["duplicate_free"]["f1_macro"])


def test_the_best_value_marking_covers_all_eight_rows(official_tree):
    write_all_eight(official_tree)
    table = cmp.build_full_comparison(verbose=False)["table"]

    marked = cmp.mark_best(table)

    assert len(marked) == 8
    starred = [c for c in marked.columns
               if marked[c].astype(str).str.contains(r"\*", regex=True).any()]
    assert starred, "no column marked a best value"
    # the highest macro-F1 is the last configuration in the display order
    best = marked["F1 (macro)"].astype(str)
    assert best.str.contains(r"\*").sum() == 1, best.tolist()
    assert "*" in best.iloc[-1], best.tolist()


def test_the_existing_rows_check_passes_at_1e_9(official_tree):
    """The "exact" half: every rebuilt row equals the metrics.json it came from.

    `check_rows_unchanged` also compares against the published figures in
    `config.STEP12_EXPECTED_ROWS`, which are write-up constants and have
    nothing to do with synthetic scores -- those rows are not this test's
    subject and are left to the run that quotes them.
    """
    write_all_eight(official_tree)
    built = cmp.build_full_comparison(verbose=False)

    table, _ok = cmp.check_rows_unchanged(built, verbose=False)
    exact = table[table["check"].str.startswith("exact: ")]

    assert len(exact) == 8, exact.to_string()
    assert (exact["result"] == "[OK]").all(), exact.to_string()
    assert not (exact["result"] == "[SKIP]").any(), exact.to_string()
    assert any("exact: proposed" == name for name in exact["check"])


# ===========================================================================
# 3. the descriptor columns Step 11 never recorded
# ===========================================================================
def test_the_descriptors_come_from_the_saved_model_card(official_tree):
    write_all_eight(official_tree)
    write(official_tree / "step6" / config.STEP6_FILES["selected"],
          {"features": ["f%d" % i for i in range(28)], "n_selected": 28})
    write(official_tree / "step10" / config.STEP10_FILES["model_card"],
          {"input_shape_per_sample": [15, 1],
           "architecture": {"total_parameters": 54321},
           "training": {"seed": 42, "best_epoch": 9,
                        "best_val_macro_f1": 0.66, "training_seconds": 77.5}})
    label = dict(config.STEP12_DISPLAY_ORDER)[PROPOSED]

    row = cmp.build_full_comparison(verbose=False)["table"]
    row = row[row["config"] == label].iloc[0]

    assert int(row["input features"]) == 28          # the SAVED count, not 30
    assert int(row["latent dim"]) == 15
    assert int(row["params"]) == 54321


def test_a_step12_copys_own_descriptors_are_never_overwritten(official_tree):
    write_all_eight(official_tree, proposed_in_step12=True)
    write(official_tree / "step6" / config.STEP6_FILES["selected"],
          {"features": ["f%d" % i for i in range(28)], "n_selected": 28})
    label = dict(config.STEP12_DISPLAY_ORDER)[PROPOSED]

    row = cmp.build_full_comparison(verbose=False)["table"]
    row = row[row["config"] == label].iloc[0]

    assert int(row["input features"]) == 66          # what that file recorded


def test_missing_cards_leave_the_descriptors_blank_rather_than_guessed(
        official_tree):
    write_all_eight(official_tree)
    label = dict(config.STEP12_DISPLAY_ORDER)[PROPOSED]

    row = cmp.build_full_comparison(verbose=False)["table"]
    row = row[row["config"] == label].iloc[0]

    assert pd.isna(row["input features"]) or row["input features"] in ("", None)
    assert float(row["F1 (macro)"]) > 0              # the scores are still there


# ===========================================================================
# 4. the blank row names the right step -- the regression guard
# ===========================================================================
def test_a_missing_row_names_its_own_step(official_tree):
    built = cmp.build_full_comparison(verbose=False)
    table = built["table"]

    assert len(built["missing"]) == 8
    for key, label in config.STEP12_DISPLAY_ORDER:
        cell = str(table[table["config"] == label]["input features"].iloc[0])
        assert config.NOT_TRAINED_TEXT in cell, (key, cell)
        assert cmp.which_step_produces(key) in cell, (key, cell)


def test_the_blank_row_no_longer_sends_everyone_to_section_10_2():
    assert "10.2" not in config.NOT_TRAINED_TEXT
    for key, _label in config.STEP12_DISPLAY_ORDER:
        if key == config.APPROACH2_KEY:
            continue
        assert "10.2" not in cmp.which_step_produces(key), key
