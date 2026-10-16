r"""
tests/test_step12_comparison8.py -- pytest tests for the EIGHT-configuration
Step 12 comparison: the extension appended to src/mscad/comparison.py.

NEW FILE.  No existing test file and no existing comparison function is
touched; these tests cover only what the Step 12 completion added -- the
eight-row table and its row-against-disk check, the fixed colour per
configuration, the publication figures, the head-to-head pairs and the thesis
exports (CSV + booktabs LaTeX).

Everything runs on small hand-made metrics.json files inside tmp_path; no model
is built and nothing is trained.  Matplotlib draws through the Agg backend.
Never C:\\MSCAD2022.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
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

LABELS = {0: "Normal", 1: "Port_Scan", 2: "Brute_Force"}
KEYS = [key for key, _label in config.STEP12_DISPLAY_ORDER]
B6 = config.BASELINE6_NAME


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    for step in (10, 11, 12):
        monkeypatch.setattr(config, "STEP%d_OUTPUT_DIR" % step,
                            tmp_path / ("step%d" % step))
    monkeypatch.setattr(config, "STEP11B_OUTPUT_DIR",
                        tmp_path / "step11_attention", raising=False)
    monkeypatch.setattr(config, "STEP12_B6_OUTPUT_DIR",
                        tmp_path / "step12_baseline6", raising=False)
    monkeypatch.setattr(config, "STEP12_THESIS_DIR", tmp_path / "thesis",
                        raising=False)
    monkeypatch.setattr(config, "STEP12_EXPECTED_ROWS", {})


# ===========================================================================
# helpers
# ===========================================================================
def metrics(label, accuracy=0.9, f1=0.5, g_mean=0.4, auc=0.9, fpr=0.02,
            params=1000, train=10.0, pca=None):
    out = {
        "config": label, "label": label, "n_input_features": 30,
        "latent_dim": 15, "parameters": params, "training_seconds": train,
        "seed": config.RANDOM_STATE, "best_epoch": 4,
        "full": {"accuracy": accuracy, "precision_macro": 0.6,
                 "recall_macro": 0.55, "f1_macro": f1, "f1_weighted": 0.8,
                 "g_mean": g_mean, "roc_auc": {"macro": auc}},
        "duplicate_free": {"f1_macro": f1, "g_mean": g_mean},
        "binary": {"f1": 0.95, "false_positive_rate": fpr},
        "inference": {"ms_per_1000_flows": 50.0},
    }
    if pca is not None:
        out["pca_components"] = pca
        out["pca"] = {"variance_kept": 0.912}
    return out


def predictions_frame(rows=12, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.array([0, 1, 2] * (rows // 3))
    y_pred = y_true.copy()
    y_pred[0] = 1
    probabilities = rng.random((len(y_true), 3))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    frame = pd.DataFrame({
        "Orig_Row": np.arange(len(y_true)),
        "y_true": y_true, "y_pred": y_pred,
        "y_true_label": [LABELS[int(c)] for c in y_true],
        "y_pred_label": [LABELS[int(c)] for c in y_pred],
        "has_train_duplicate": np.zeros(len(y_true), dtype=bool)})
    for code, name in LABELS.items():
        frame["p_%s" % name] = probabilities[:, code]
    return frame


def confusion_frame():
    """A row-normalised confusion matrix, as Step 11's evaluator writes it."""
    names = [LABELS[code] for code in sorted(LABELS)]
    values = np.eye(len(names)) * 0.9 + 0.05
    return pd.DataFrame(values, index=names, columns=names)


def write_result(key, payload, predictions=None, confusion=True):
    folder = Path(cmp.result_path(key)).parent
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    if predictions is not None:
        predictions.to_csv(folder / "predictions.csv", index=False)
    if confusion:
        confusion_frame().to_csv(folder / "confusion_matrix_rownorm.csv")
    return folder


def write_all_eight(with_predictions=True):
    for i, (key, label) in enumerate(config.STEP12_DISPLAY_ORDER):
        payload = metrics(label, f1=0.50 + 0.02 * i, g_mean=0.40 + 0.02 * i,
                          accuracy=0.90 + 0.001 * i, params=1000 + 10 * i,
                          train=10.0 + i,
                          pca=2 if key in (B6, "baseline4_ae_pca_lstm")
                          else None)
        write_result(key, payload,
                     predictions_frame(seed=i) if with_predictions else None)


# ===========================================================================
# the comparison has eight rows, in display order
# ===========================================================================
def test_the_comparison_has_eight_rows_in_display_order(tmp_path):
    write_all_eight()

    built = cmp.build_full_comparison(verbose=False)

    assert len(built["table"]) == 8
    assert list(built["table"]["config"]) == [label for _k, label
                                              in config.STEP12_DISPLAY_ORDER]
    assert built["missing"] == []
    assert built["table"].iloc[-1]["config"] == config.BASELINE6_LABEL


def test_baseline_6s_row_is_read_from_its_own_folder(tmp_path):
    write_all_eight()

    path = cmp.result_path(B6)

    assert Path(path).parent == tmp_path / "step12_baseline6"
    assert "12.6" in cmp.which_step_produces(B6)
    assert cmp.load_result(B6)["label"] == config.BASELINE6_LABEL


def test_an_untrained_baseline_6_leaves_its_row_blank_and_named(tmp_path):
    for key, label in config.STEP12_DISPLAY_ORDER:
        if key != B6:
            write_result(key, metrics(label))

    built = cmp.build_full_comparison(verbose=False)

    assert len(built["table"]) == 8
    assert built["missing"] == [B6]
    row = built["table"].iloc[-1]
    assert row["config"] == config.BASELINE6_LABEL
    assert pd.isna(pd.to_numeric(row["F1 (macro)"], errors="coerce"))


# ===========================================================================
# every rendered number against the metrics.json it came from
# ===========================================================================
def test_every_row_equals_the_saved_file_it_came_from(tmp_path):
    write_all_eight()
    built = cmp.build_full_comparison(verbose=False)

    verified = cmp.verify_rows_against_disk(built["table"])

    assert len(verified) == 8
    assert set(verified["result"]) == {"[OK]"}
    assert float(verified["largest difference"].max()) <= 1e-9
    assert (verified["columns compared"] > 0).all()
    for _i, row in verified.iterrows():
        assert Path(row["metrics.json"]).exists()


def test_a_table_that_no_longer_matches_disk_is_a_failure(tmp_path):
    """The check exists to catch exactly this: a number that is not the file's."""
    write_all_eight()
    built = cmp.build_full_comparison(verbose=False)
    tampered = built["table"].copy()
    tampered.loc[0, "F1 (macro)"] = float(tampered.loc[0, "F1 (macro)"]) + 0.01

    verified = cmp.verify_rows_against_disk(tampered)
    failed = verified[verified["result"] == "[FAIL]"]

    assert len(failed) == 1
    assert failed.iloc[0]["config"] == config.STEP12_DISPLAY_ORDER[0][1]
    assert float(failed.iloc[0]["largest difference"]) == pytest.approx(0.01,
                                                                       abs=1e-9)


def test_a_missing_row_is_skipped_rather_than_passed(tmp_path):
    for key, label in config.STEP12_DISPLAY_ORDER:
        if key != B6:
            write_result(key, metrics(label))
    built = cmp.build_full_comparison(verbose=False)

    verified = cmp.verify_rows_against_disk(built["table"])
    b6_row = verified[verified["config"] == config.BASELINE6_LABEL].iloc[0]

    assert b6_row["result"] == "[SKIP]"
    assert b6_row["metrics.json"] == "(absent)"


def test_the_tolerance_is_the_one_the_brief_asks_for(tmp_path):
    write_all_eight()
    built = cmp.build_full_comparison(verbose=False)
    tampered = built["table"].copy()
    tampered.loc[1, "G-mean"] = float(tampered.loc[1, "G-mean"]) + 1e-7

    assert set(cmp.verify_rows_against_disk(tampered, tolerance=1e-6)
               ["result"]) == {"[OK]"}
    assert "[FAIL]" in set(cmp.verify_rows_against_disk(tampered,
                                                        tolerance=1e-9)
                           ["result"])


# ===========================================================================
# one fixed colour per configuration
# ===========================================================================
def test_every_configuration_has_its_own_fixed_colour():
    colours = {key: cmp.model_colour(key) for key in KEYS}

    assert len(colours) == 8
    assert len(set(colours.values())) == 8, colours
    assert colours[B6] == config.MODEL_COLOURS[B6]


def test_the_colour_follows_the_model_not_its_position_in_a_subset():
    before = cmp.model_colour(B6)
    subset = [(B6, config.BASELINE6_LABEL)]

    assert cmp.model_colour(B6) == before
    assert cmp.model_colour(subset[0][0]) == before
    assert cmp.model_colour("proposed") != before


def test_the_fusion_addendum_gives_baseline_6_the_same_hue():
    """13.8's charts and 12.8's figures must not disagree about a colour."""
    from src.mscad import fusion_all as fa

    for key in KEYS:
        assert fa.model_colour(key) == cmp.model_colour(key), key


def test_the_explainability_addendum_gives_baseline_6_the_same_hue():
    from src.mscad import explainability_all as xa

    assert xa.model_colour(B6) == cmp.model_colour(B6)


def test_the_short_label_is_the_prefix_the_figures_use():
    assert cmp.short_label(B6) == "B6"
    assert cmp.short_label("proposed") == cmp.display_label(
        "proposed").split(":")[0].strip()
    assert len({cmp.short_label(key) for key in KEYS}) == 8


# ===========================================================================
# the configuration and training-setup tables
# ===========================================================================
def test_the_configuration_table_has_one_row_per_configuration(tmp_path):
    write_all_eight()

    table = cmp.configuration_table_8()

    assert len(table) == 8
    assert list(table["config"]) == [label for _k, label
                                     in config.STEP12_DISPLAY_ORDER]
    row = table.iloc[-1]
    assert row["config"] == config.BASELINE6_LABEL
    assert row["CNN"] == "yes"
    assert row["attention"] == "no"
    assert row["LSTM"] == "yes"
    assert row["AE"].startswith("yes")
    assert "2 of 15" in row["PCA"]
    assert config.STEP12_ISOLATES[B6] == row["what it isolates"]


def test_the_configuration_table_separates_b4_and_b6_by_the_classifier(
        tmp_path):
    write_all_eight()

    table = cmp.configuration_table_8().set_index("config")
    b4 = table.loc[dict(config.STEP12_DISPLAY_ORDER)["baseline4_ae_pca_lstm"]]
    b6 = table.loc[config.BASELINE6_LABEL]

    assert b4["PCA"] == b6["PCA"]            # the same feature space
    assert b4["AE"] == b6["AE"]
    assert b4["CNN"] == "no" and b6["CNN"] == "yes"


def test_the_training_setup_table_records_the_shared_protocol(tmp_path):
    write_all_eight()

    table = cmp.training_setup_table()

    assert len(table) == 8
    assert set(table["learning rate"]) == {config.CLF_LR}
    assert set(table["batch size"]) == {config.CLF_BATCH_SIZE}
    assert set(table["patience"]) == {config.CLF_PATIENCE}
    assert set(table["early stopping on"]) == {config.EARLY_STOP_METRIC}
    assert set(table["seed"]) == {config.RANDOM_STATE}


def test_an_untrained_configuration_shows_dashes_not_defaults(tmp_path):
    for key, label in config.STEP12_DISPLAY_ORDER:
        if key != B6:
            write_result(key, metrics(label))

    row = cmp.training_setup_table().iloc[-1]

    assert row["config"] == config.BASELINE6_LABEL
    assert row["seed"] == "-"
    assert row["best epoch"] == "-"


# ===========================================================================
# the head-to-head pairs
# ===========================================================================
def test_the_seven_pairs_each_differ_in_exactly_one_component(tmp_path):
    write_all_eight()
    built = cmp.build_full_comparison(verbose=False)

    head = cmp.all_head_to_heads_8(built["table"])
    pairs = list(dict.fromkeys(head["pair"]))

    assert len(cmp.HEAD_TO_HEAD_PAIRS_8) == 7
    assert len(pairs) == 7
    assert any("B4 vs B6" in pair for pair in pairs)
    assert any("B1 vs B6" in pair for pair in pairs)
    assert any("Approach 1 vs Approach 2" in pair for pair in pairs)
    for left, right, _why in cmp.HEAD_TO_HEAD_PAIRS_8:
        assert left in KEYS and right in KEYS


def test_a_head_to_head_difference_is_left_minus_right(tmp_path):
    write_all_eight()
    built = cmp.build_full_comparison(verbose=False)
    table = built["table"].set_index("config")
    labels = dict(config.STEP12_DISPLAY_ORDER)

    head = cmp.all_head_to_heads_8(built["table"])
    row = head[(head["pair"].str.contains("B4 vs B6"))
               & (head["metric"] == "F1 (macro)")].iloc[0]
    left = float(table.loc[labels["baseline4_ae_pca_lstm"]]["F1 (macro)"])
    right = float(table.loc[labels[B6]]["F1 (macro)"])

    assert float(row["difference (left - right)"]) == pytest.approx(
        left - right, abs=1e-9)


def test_a_pair_with_an_untrained_side_is_left_out(tmp_path):
    for key, label in config.STEP12_DISPLAY_ORDER:
        if key != B6:
            write_result(key, metrics(label))
    built = cmp.build_full_comparison(verbose=False)

    head = cmp.all_head_to_heads_8(built["table"])

    assert len(head)
    assert not any("B6" in pair for pair in head["pair"])


def test_the_per_class_differences_cover_the_same_pairs(tmp_path):
    write_all_eight()
    built = cmp.build_full_comparison(verbose=False)
    per_class = cmp.per_class_table()

    differences = cmp.per_class_difference_8(per_class)

    assert len(differences)
    assert set(differences.columns) >= {"pair", "Label", "difference",
                                        "favours"}
    assert set(differences["favours"]) <= set(
        [cmp.short_label(key) for key in KEYS] + ["tie"])
    assert set(differences["Label"]) <= set(LABELS.values())
    b4_vs_b6 = differences[differences["pair"].str.contains("B4 vs B6")]
    assert len(b4_vs_b6)
    assert {"B4", "B6"} <= set(b4_vs_b6.columns)


# ===========================================================================
# the publication figures
# ===========================================================================
def test_every_figure_is_written_as_png_and_pdf(tmp_path):
    pytest.importorskip("matplotlib")
    write_all_eight()
    built = cmp.build_full_comparison(verbose=False)
    built["per_class"] = cmp.per_class_table()
    folder = tmp_path / "figures"

    charts = cmp.draw_figures_8(built, folder, majority_accuracy=0.70,
                                random_macro_f1=0.19, verbose=False)

    assert charts
    for stem in (config.STEP12_FULL_FILES and cmp.FIGURE_FILES).values():
        png, pdf = folder / ("%s.png" % stem), folder / ("%s.pdf" % stem)
        assert png.exists() and png.stat().st_size > 0, stem
        assert pdf.exists() and pdf.stat().st_size > 0, stem


def test_the_figures_are_saved_at_the_thesis_resolution(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    write_all_eight()
    built = cmp.build_full_comparison(verbose=False)
    folder = tmp_path / "figures"

    cmp.figure_accuracy(built["table"], folder, cmp.FIGURE_FILES["accuracy"],
                        majority=0.70, verbose=False)
    png = folder / ("%s.png" % cmp.FIGURE_FILES["accuracy"])

    with Image.open(png) as image:
        dpi = image.info.get("dpi")
    assert dpi is not None
    assert int(round(dpi[0])) == int(config.THESIS_FIGURE_DPI) == 300


def test_a_figure_is_also_written_into_the_thesis_folder(tmp_path):
    pytest.importorskip("matplotlib")
    write_all_eight()
    built = cmp.build_full_comparison(verbose=False)
    folder, thesis = tmp_path / "figures", tmp_path / "thesis"

    cmp.figure_single_metric(built["table"], "train time (s)",
                             "Training time", "seconds", folder,
                             cmp.FIGURE_FILES["time"], lower_is_better=True,
                             verbose=False, also=thesis)

    for place in (folder, thesis):
        assert (place / ("%s.png" % cmp.FIGURE_FILES["time"])).exists()
        assert (place / ("%s.pdf" % cmp.FIGURE_FILES["time"])).exists()


def test_the_new_figure_names_never_overwrite_an_earlier_step_12_file():
    new = set(cmp.FIGURE_FILES.values())
    existing = set()
    for block in (config.STEP12_FILES, config.STEP12B_FILES,
                  config.STEP12C_FILES, config.STEP12_FULL_FILES):
        existing |= {Path(name).stem for name in block.values()}

    assert not (new & existing), sorted(new & existing)
    assert len(new) == len(cmp.FIGURE_FILES)


def test_a_figure_with_no_trained_row_is_skipped_rather_than_empty(tmp_path):
    pytest.importorskip("matplotlib")
    built = {"table": pd.DataFrame([{"config": "A1", "F1 (macro)": np.nan}])}
    folder = tmp_path / "figures"

    charts = cmp.figure_metric_bars(built["table"], ["F1 (macro)"], "t",
                                    folder, "fig_nothing", verbose=False)

    assert charts == {}
    assert not (folder / "fig_nothing.png").exists()


# ===========================================================================
# the thesis exports: every table as CSV and as booktabs LaTeX
# ===========================================================================
def test_every_exported_table_is_written_as_csv_and_latex(tmp_path):
    write_all_eight()
    built = cmp.build_full_comparison(verbose=False)
    tables = {"comparison": built["table"],
              "configuration": cmp.configuration_table_8(),
              "training_setup": cmp.training_setup_table(),
              "head_to_head": cmp.all_head_to_heads_8(built["table"])}

    written = cmp.export_thesis(tables, verbose=False)
    folder = tmp_path / "thesis"

    assert len(written) == 2 * len(tables)
    for key in tables:
        stem = cmp.THESIS_TABLES[key][0]
        assert (folder / ("%s.csv" % stem)).exists(), stem
        assert (folder / ("%s.tex" % stem)).exists(), stem
    saved = pd.read_csv(folder / "step12_comparison.csv")
    assert len(saved) == 8


def test_the_latex_is_booktabs_with_a_caption_and_a_label():
    frame = pd.DataFrame([{"config": "B6: AE -> PCA -> CNN-LSTM",
                           "F1 (macro)": 0.5123}])

    text = cmp.to_latex(frame, "The eight configurations.", "step12_demo",
                        note="Read with macro-F1.")

    assert r"\begin{table}" in text and r"\end{table}" in text
    assert r"\toprule" in text and r"\midrule" in text and r"\bottomrule" in text
    assert r"\caption{The eight configurations.}" in text
    assert r"\label{tab:step12_demo}" in text
    assert "0.5123" in text
    assert "Read with macro-F1." in text


def test_the_latex_escapes_what_latex_would_otherwise_eat():
    frame = pd.DataFrame([{"config": "B6: AE -> PCA", "note": "100% & _x_ #1"}])

    text = cmp.to_latex(frame, "A & B 50%", "demo")

    assert r"\%" in text and r"\&" in text and r"\_" in text and r"\#" in text
    assert r"$\rightarrow$" in text
    assert "100% &" not in text


def test_an_empty_table_is_a_comment_rather_than_a_broken_tex_file():
    text = cmp.to_latex(pd.DataFrame(), "Nothing here.", "demo")

    assert text.startswith("%")
    assert r"\begin{table}" not in text


def test_an_empty_table_is_not_exported(tmp_path):
    written = cmp.export_thesis({"comparison": pd.DataFrame()}, verbose=False)

    assert written == {}


def test_every_exportable_table_has_a_caption_waiting_for_it():
    for key in ("configuration", "training_setup", "sanity", "comparison",
                "per_class", "per_class_free", "head_to_head",
                "per_class_difference", "zero_recall", "provenance", "seeds"):
        stem, caption = cmp.THESIS_TABLES[key]
        assert stem.startswith("step12")
        assert caption.endswith(".")
        assert len(caption) > 30, key


# ===========================================================================
# the whole 12.7-12.9 run
# ===========================================================================
def test_the_complete_run_writes_eight_verified_rows_and_its_figures(tmp_path):
    pytest.importorskip("matplotlib")
    write_all_eight()

    out = cmp.run_step12_complete(majority_accuracy=0.70,
                                  random_macro_f1=0.19, verbose=False)
    folder = tmp_path / "step12"

    assert out["status"] == "written"
    assert out["ok"] is True
    assert len(out["comparison"]) == 8
    assert set(out["verified"]["result"]) == {"[OK]"}
    assert (folder / config.STEP12_FULL_FILES["comparison"]).exists()
    assert (folder / "configuration_table_8configs.csv").exists()
    assert (folder / "training_setup_8configs.csv").exists()
    assert (folder / "row_vs_disk_check.csv").exists()
    assert len(pd.read_csv(folder / "row_vs_disk_check.csv")) == 8
    assert out["charts"]


def test_the_complete_run_leaves_the_earlier_step_12_tables_alone(tmp_path):
    """Step 12's existing CSVs are not rewritten by the completion."""
    write_all_eight()
    folder = tmp_path / "step12"
    folder.mkdir(parents=True, exist_ok=True)
    guarded = folder / config.STEP12_FILES["comparison"]
    guarded.write_text("untouched", encoding="utf-8")

    cmp.run_step12_complete(verbose=False)

    assert guarded.read_text(encoding="utf-8") == "untouched"


def test_the_complete_run_reports_not_ok_when_a_row_does_not_match_disk(
        tmp_path, monkeypatch):
    write_all_eight()
    key = config.STEP12_DISPLAY_ORDER[0][0]
    monkeypatch.setattr(config, "STEP12_EXPECTED_ROWS",
                        {key: {"f1_macro": 0.0}})

    out = cmp.run_step12_complete(verbose=False)

    assert out["ok"] is False
    assert "[FAIL]" in set(out["unchanged"]["result"])
