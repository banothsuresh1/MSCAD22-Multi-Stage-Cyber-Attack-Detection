r"""
tests/step12/test_step12_outputs.py -- what the eight files produce together:
the comparison, the figures, the exports, and the config they all read.

NEW FILE.  No model is built and nothing is trained; matplotlib draws through
the Agg backend into tmp_path.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

import config
from src.mscad import comparison as cmp
from src.mscad import fusion_all as fa
from src.mscad.experiments_step12 import common, report

from .conftest import KEYS                                      # noqa: E402

FOLDER = (Path(__file__).resolve().parent.parent.parent / "src" / "mscad"
          / "experiments_step12")
B6 = config.BASELINE6_NAME


# ===========================================================================
# the comparison: eight rows, in display order, equal to the saved files
# ===========================================================================
def test_the_comparison_has_eight_rows_in_display_order(all_finished):
    built = cmp.build_full_comparison(verbose=False)

    assert len(built["table"]) == 8
    assert list(built["table"]["config"]) == [label for _k, label
                                              in config.STEP12_DISPLAY_ORDER]
    assert built["missing"] == []
    assert built["table"].iloc[-1]["config"] == config.BASELINE6_LABEL


def test_every_row_equals_the_saved_file_it_came_from(all_finished):
    built = cmp.build_full_comparison(verbose=False)

    verified = cmp.verify_rows_against_disk(built["table"])

    assert len(verified) == 8
    assert set(verified["result"]) == {"[OK]"}
    assert float(verified["largest difference"].max()) <= 1e-9


def test_a_row_that_no_longer_matches_disk_is_a_failure(all_finished):
    built = cmp.build_full_comparison(verbose=False)
    tampered = built["table"].copy()
    tampered.loc[0, "F1 (macro)"] = float(tampered.loc[0, "F1 (macro)"]) + 0.01

    verified = cmp.verify_rows_against_disk(tampered)
    failed = verified[verified["result"] == "[FAIL]"]

    assert len(failed) == 1
    assert float(failed.iloc[0]["largest difference"]) == pytest.approx(
        0.01, abs=1e-9)


def test_an_untrained_configuration_leaves_its_row_blank_and_named(finished):
    finished(*[key for key in KEYS if key != B6])

    built = cmp.build_full_comparison(verbose=False)

    assert len(built["table"]) == 8
    assert built["missing"] == [B6]
    assert pd.isna(pd.to_numeric(built["table"].iloc[-1]["F1 (macro)"],
                                 errors="coerce"))


# ===========================================================================
# each file's own description against the published table
# ===========================================================================
def test_each_file_agrees_with_the_published_configuration_table(all_finished):
    out = report.configuration_tables(verbose=False)

    assert len(out["components"]) == 8
    assert len(out["agreement"]) == 8
    assert set(out["agreement"]["result"]) == {"[OK]"}, out["agreement"][
        out["agreement"]["result"] == "[FAIL]"]["detail"].tolist()
    assert out["ok"] is True


def test_a_file_that_disagreed_with_the_table_would_be_caught(all_finished,
                                                              monkeypatch):
    """The check has to be able to fail, or it is not a check."""
    from src.mscad.experiments_step12 import baseline2

    monkeypatch.setitem(baseline2.COMPONENTS, "CNN", "yes")   # B2 has no CNN

    table = report.files_agree_with_table()
    failed = table[table["result"] == "[FAIL]"]

    assert len(failed) == 1
    assert failed.iloc[0]["file"] == "baseline2.py"
    assert "CNN" in failed.iloc[0]["detail"]


def test_the_components_table_is_read_from_the_eight_files():
    table = report.components_table()

    assert list(table["file"]) == ["%s.py" % name for name in
                                   ("approach1", "approach2", "baseline1",
                                    "baseline2", "baseline3", "baseline4",
                                    "baseline5", "baseline6")]
    assert list(table["config"]) == [label for _k, label
                                     in config.STEP12_DISPLAY_ORDER]
    assert table[table["file"] == "approach2.py"].iloc[0][
        "attention"].startswith("yes")
    assert table[table["file"] == "baseline3.py"].iloc[0][
        "autoencoder"] == "no"


# ===========================================================================
# the figures: eight configurations, one fixed colour each
# ===========================================================================
def test_every_configuration_has_its_own_fixed_colour():
    colours = {key: cmp.model_colour(key) for key in KEYS}

    assert len(set(colours.values())) == 8, colours
    assert colours == dict(config.CONFIG_COLOURS)


def test_the_colour_table_ties_each_file_to_its_hue():
    table = report.colour_table()

    assert len(table) == 8
    assert set(table["result"]) == {"[OK]"}
    assert list(table["file"]) == ["%s.py" % common.short_name(key)
                                   for key in KEYS]


def test_the_fusion_addendum_uses_the_same_hues():
    """13.8's charts and 12.12's figures must not disagree about a colour."""
    for key in KEYS:
        assert fa.model_colour(key) == cmp.model_colour(key), key


def test_the_figures_are_written_as_png_and_pdf_for_eight_configurations(
        all_finished, tmp_path):
    pytest.importorskip("matplotlib")
    built = cmp.build_full_comparison(verbose=False)
    built["per_class"] = cmp.per_class_table()
    folder = tmp_path / "figures"

    charts = cmp.draw_figures_8(built, folder, majority_accuracy=0.70,
                               verbose=False)
    table = report.figure_table(charts)

    assert len(table)
    assert set(table["exists"]) == {True}
    assert set(table["format"]) == {"png", "pdf"}
    for stem in cmp.FIGURE_FILES.values():
        assert (folder / ("%s.png" % stem)).exists(), stem
        assert (folder / ("%s.pdf" % stem)).exists(), stem


def test_the_new_figure_names_never_overwrite_an_earlier_step_12_file():
    new = set(cmp.FIGURE_FILES.values())
    existing = set()
    for block in (config.STEP12_FILES, config.STEP12B_FILES,
                  config.STEP12C_FILES, config.STEP12_FULL_FILES):
        existing |= {Path(name).stem for name in block.values()}

    assert not (new & existing), sorted(new & existing)


# ===========================================================================
# the exports
# ===========================================================================
def test_every_table_is_exported_as_csv_and_latex(all_finished, tmp_path):
    built = cmp.build_full_comparison(verbose=False)
    step12 = {"comparison": built["table"],
              "per_class": cmp.per_class_table(),
              "head_to_head": cmp.all_head_to_heads_8(built["table"])}

    written = report.export_tables(step12, provenance=None, verbose=False)
    folder = tmp_path / "step12" / "thesis"

    assert written
    for key in ("comparison", "per_class", "head_to_head", "configuration",
                "training_setup"):
        stem = cmp.THESIS_TABLES[key][0]
        assert (folder / ("%s.csv" % stem)).exists(), stem
        assert (folder / ("%s.tex" % stem)).exists(), stem
    text = (folder / "step12_comparison.tex").read_text(encoding="utf-8")
    assert r"\toprule" in text and r"\bottomrule" in text
    assert r"\caption{" in text and r"\label{tab:" in text
    assert len(pd.read_csv(folder / "step12_comparison.csv")) == 8


# ===========================================================================
# the head-to-head pairs the brief names
# ===========================================================================
def test_the_seven_pairs_are_the_ones_asked_for(all_finished):
    built = cmp.build_full_comparison(verbose=False)

    head = cmp.all_head_to_heads_8(built["table"])
    pairs = list(dict.fromkeys(head["pair"]))
    highlights = report.head_to_head_highlights(head)

    assert len(cmp.HEAD_TO_HEAD_PAIRS_8) == 7
    assert len(pairs) == 7
    assert len(highlights) == 7
    for wanted in ("Approach 1 vs B1", "Approach 1 vs B5",
                   "Approach 1 vs Approach 2", "B1 vs B2", "B1 vs B3",
                   "B4 vs B6", "B1 vs B6"):
        assert any(wanted in pair for pair in pairs), wanted


def test_the_highlights_carry_the_direction(all_finished):
    built = cmp.build_full_comparison(verbose=False)

    highlights = report.head_to_head_highlights(
        cmp.all_head_to_heads_8(built["table"]))

    assert set(highlights.columns) >= {"pair", "left", "right",
                                       "difference (left - right)", "favours"}
    assert set(highlights["metric"]) == {"F1 (macro)"}


# ===========================================================================
# the status table and rule 4, end to end
# ===========================================================================
def test_the_status_table_says_loaded_for_every_finished_configuration(
        all_finished):
    results = report.run_all(verbose=False)

    table = report.status_table(results)

    assert len(table) == 8
    assert set(table["status"]) == {"loaded"}
    assert set(table["detail"]) == {"loaded (locked)"}
    assert set(table["numbers unchanged"]) == {"[OK]"}
    assert set(table["result"]) == {"[OK]"}
    assert list(table["file"]) == ["%s.py" % common.short_name(key)
                                   for key in KEYS]


def test_rule_four_holds_across_the_whole_run(all_finished):
    before = common.snapshot()

    report.run_all(verbose=False)
    table = report.verify_unchanged(before, verbose=False)

    assert len(table) == 8
    assert set(table["result"]) == {"[OK]"}
    assert float(table["largest difference"].max()) == 0.0


def test_running_the_eight_files_writes_eight_manifests(all_finished,
                                                        tmp_path):
    report.run_all(verbose=False)

    folders = sorted(
        path.parent.name for path in (tmp_path / "step12").glob(
            "*/manifest.json"))

    assert len(folders) == 8
    assert "baseline5_config" in folders        # its key is also its file name
    for name in ("approach1", "approach2", "baseline1", "baseline2",
                 "baseline3", "baseline4", "baseline6"):
        assert name in folders, name


# ===========================================================================
# the config the eight files read
# ===========================================================================
def test_baseline6_is_in_the_all_model_fusion_and_explainability_lists():
    assert B6 in config.FUSION_ALL_MODELS
    assert B6 in config.XAI_ALL_MODELS
    assert len(config.FUSION_ALL_MODELS) == 8
    assert len(config.XAI_ALL_MODELS) == 8


def test_step_13s_own_three_model_fusion_is_unchanged():
    assert list(config.FUSION_MODELS) == ["proposed", "baseline2_ae_lstm",
                                          "baseline3_cnn_lstm"]
    assert B6 not in config.FUSION_MODELS


def test_the_existing_internal_keys_are_unchanged():
    assert [key for key, _label in config.STEP12_DISPLAY_ORDER] == [
        "proposed", "approach2_attention", "baseline1_ae_cnn_lstm",
        "baseline2_ae_lstm", "baseline3_cnn_lstm", "baseline4_ae_pca_lstm",
        "baseline5", "baseline6_ae_pca_cnn_lstm"]
    assert [spec["name"] for spec in config.EXPERIMENTS] == [
        "proposed", "baseline1_ae_cnn_lstm", "baseline2_ae_lstm",
        "baseline3_cnn_lstm"]
    assert B6 not in [spec["name"] for spec in config.EXPERIMENTS]


def test_the_new_block_redefines_nothing_it_did_not_introduce():
    """Append-only: a setting that already existed is read, not reassigned."""
    text = (Path(__file__).resolve().parent.parent.parent
            / "config.py").read_text(encoding="utf-8")

    for name in ("LOCK_TRAINED_MODELS", "B6_PCA_COMPONENTS"):
        assert '%s = globals().get("%s"' % (name, name) in text, name
    assert "B6_OUTPUT_DIR = STEP12_B6_OUTPUT_DIR" in text
    assert "CONFIG_COLOURS = dict(MODEL_COLOURS)" in text


def test_the_eight_files_exist_and_nothing_else_does():
    expected = {"__init__.py", "common.py", "report.py", "approach1.py",
                "approach2.py", "baseline1.py", "baseline2.py",
                "baseline3.py", "baseline4.py", "baseline5.py",
                "baseline6.py"}

    assert {path.name for path in FOLDER.glob("*.py")} == expected


def test_no_keras_3_only_api_in_the_new_folder():
    for path in sorted(FOLDER.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "keras.ops" not in text, path.name
        assert "from keras import ops" not in text, path.name
