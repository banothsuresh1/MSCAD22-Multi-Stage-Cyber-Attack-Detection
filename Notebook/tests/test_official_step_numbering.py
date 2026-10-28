r"""
tests/test_official_step_numbering.py -- the simplified official-dataset flow.

NEW FILE.  The notebook was restructured so that Step 1 is data acquisition from
one CSV and Step 2 is preprocessing; merging is gone.  Everything after
preprocessing was renumbered in MARKDOWN ONLY -- no code cell, no module and no
output folder changed -- and these tests are what holds that line:

  * the frozen modules under src/mscad/ are byte-identical to main (rule 2);
  * every code cell from the preprocessing step onwards is byte-identical to
    the cell it was before the renumbering (rule 2), checked against the
    recorded digests in tests/data/notebook_cell_identity.json;
  * the output folders keep their old names, so nothing has to be retrained
    (rules 3 and 4);
  * the own-extraction pipeline and its results are untouched (rule 5);
  * Step 1's new self-check and own-extraction comparison do what they say.

Nothing is trained, nothing is downloaded, and every path is inside tmp_path.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import official_dataset as od

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "MSCAD_Pipeline.ipynb"
FIXTURE = Path(__file__).resolve().parent / "data" / "notebook_cell_identity.json"

# The modules rule 2 freezes: the code after preprocessing, unchanged.
# The training modules: Steps 3 to 9, 11 and 12.  comparison.py and
# step12_checks.py are NOT here -- they build Step 10's report, they train
# nothing, and the Approach 1 row fix had to change them.
FROZEN_MODULES = (
    "splitting.py", "feature_selection.py", "imbalance.py",
    "models.py", "models_attention.py", "evaluation.py",
    "experiments_pca.py", "experiments_baseline5.py",
    "experiments_baseline6.py", "fusion.py", "fusion_all.py",
    "explainability.py", "explainability_all.py",
)
# PR #54 (AE latent-30 sensitivity): autoencoder.py and experiments.py are no
# longer in that tuple.  Each gained ONE parameter, `latent_dim=None`, which
# means config.AE_LATENT_DIM -- what every existing caller already got.  No
# training logic moved, and tests/test_ae30_sensitivity.py proves both halves:
# latent_dim=30 gives a 30-wide encoding, and the default still gives 15.
# They remain in STEP_1_AND_2_MODULES, so they may differ from main but are
# still named there rather than unchecked.

# The twelve headings of the simplified flow, in order.
NEW_HEADINGS = [
    "## Step 1 - Data acquisition",
    "## Step 2 - Preprocessing",
    "## Step 3 - Train-Test Validation Split",
    "## Step 4 - Feature Selection (MI + Pearson + PI to top 30)",
    "## Step 5 - Autoencoder feature extraction",
    "## Step 6 - Transform the training and test sets",
    "## Step 7 - Class imbalance handled in the training loss",
    "## Step 8 - Hybrid classifiers",
    "## Step 9 - Evaluation on the untouched test set",
    "## Step 10 - Model comparison",
    "## Step 11 - Adaptive Evidence Fusion",
    "## Step 12 - Explainability Layer",
]

# Notebook step -> the output folder it writes, which did NOT change.
STEP_FOLDERS = {1: "STEP3_OUTPUT_DIR", 2: "STEP4_OUTPUT_DIR",
                3: "STEP5_OUTPUT_DIR", 4: "STEP6_OUTPUT_DIR",
                5: "STEP7_OUTPUT_DIR", 6: "STEP8_OUTPUT_DIR",
                7: "STEP9_OUTPUT_DIR", 8: "STEP10_OUTPUT_DIR",
                9: "STEP11_OUTPUT_DIR", 10: "STEP12_OUTPUT_DIR",
                11: "STEP13_OUTPUT_DIR", 12: "STEP14_OUTPUT_DIR"}

OWN_BANNER = "## Appendix A - Own extraction (reference only)"


# ===========================================================================
# helpers
# ===========================================================================
def notebook():
    nbformat = pytest.importorskip("nbformat")
    if not NOTEBOOK.exists():                      # pragma: no cover
        pytest.skip("the notebook is not in this checkout")
    return nbformat.read(str(NOTEBOOK), as_version=4)


def official_flow_cells(nb):
    """Every cell: the notebook is the official flow and nothing else now."""
    return list(nb.cells)


def github_anchor(heading):
    """GitHub's auto-anchor for a heading's text, as the contents list uses."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(*args):
    try:
        done = subprocess.run(("git",) + args, cwd=str(REPO.parent),
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pytest.skip("git is not available here")
    if done.returncode:                            # pragma: no cover
        pytest.skip("git %s failed: %s" % (" ".join(args), done.stderr.strip()))
    return done.stdout


# ===========================================================================
# 1. rule 2 -- the code after preprocessing is not changed
# ===========================================================================
# The Step 1 and Step 2 modules may differ from main: they are the ones the
# official-dataset work touches.  Everything in FROZEN_MODULES may not.
# Each entry names the PR that justifies it, so a future reviewer can tell
# this set was widened on purpose and not by a test-silencing edit.
STEP_1_AND_2_MODULES = {
    "official_dataset.py",   # Step 1 itself -- the official-dataset loader
    "preprocessing.py",      # PR #45: COMBINED_ID_COLUMNS = [] verification
    "comparison.py",         # PR #46: Approach 1's row in the Step 10 table
    "step12_checks.py",      # PR #46: the same-file check behind that row
}
# Step 10's runner, which trains nothing either: PR #49 turns a missing result
# into a pending ROW instead of an exception that took the whole cell down.
STEP_1_AND_2_MODULES = STEP_1_AND_2_MODULES | {"common.py"}
# PR #54 (AE latent-30 sensitivity): both gain a `latent_dim` parameter whose
# default is None, i.e. config.AE_LATENT_DIM -- the behaviour every existing
# caller already had.  No training logic is changed.
STEP_1_AND_2_MODULES = STEP_1_AND_2_MODULES | {"autoencoder.py",
                                               "experiments.py"}

# PR #54: a NEW folder, which cannot change the frozen modules by existing.
# Its own rule -- that it imports the trainers instead of copying them -- is
# checked in tests/test_ae30_sensitivity.py.
NEW_FOLDERS = ("src/mscad/experiments_step12_ae30/",
               # PR #55 (paper replication): a NEW folder, additive, cannot
               # change a frozen module by existing; it imports the evaluator
               # and builds its own models, checked in test_paper_replication.
               "src/mscad/experiments_step12_paper_replication/")


def test_the_frozen_modules_are_byte_identical_to_main():
    """Only the Step 1 and Step 2 modules may differ from main."""
    base = None
    for candidate in ("origin/main", "main"):
        out = subprocess.run(("git", "rev-parse", "--verify", candidate),
                             cwd=str(REPO.parent), capture_output=True,
                             text=True)
        if out.returncode == 0:
            base = candidate
            break
    if base is None:                               # pragma: no cover
        pytest.skip("neither origin/main nor main is in this checkout")

    changed = [line.strip() for line in
               git("diff", "--name-only", base, "--", "Notebook/src/mscad/"
                   ).splitlines() if line.strip()]
    changed = [name for name in changed
               if not any(folder in name for folder in NEW_FOLDERS)]
    touched = {Path(name).name for name in changed}

    assert touched <= STEP_1_AND_2_MODULES, sorted(touched)
    for name in FROZEN_MODULES:
        assert name not in touched, name


# Code cells changed by a LATER, separately authorised PR.  The two tests below
# exist to prove the renumbering pass was presentation only, so a code cell that
# moves has to be named here with the PR that moved it -- a reviewer can then
# see the list did not grow quietly.
CODE_CELLS_CHANGED_SINCE = {
    "s12f-26": "PR #48: the Step 10 cell calls comparison.show_honest_step10",
}


def test_every_code_cell_from_preprocessing_onwards_is_unchanged():
    """The recorded digests are from BEFORE the renumbering."""
    nb = notebook()
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))["cells"]
    current = {c.get("id"): c for c in nb.cells}

    assert recorded, "the identity fixture is empty"
    code_rows = [r for r in recorded if r["cell_type"] == "code"]
    assert len(code_rows) > 100, len(code_rows)

    for row in code_rows:
        cell = current.get(row["id"])
        assert cell is not None, "code cell %s disappeared" % row["id"]
        assert cell.cell_type == "code", row["id"]
        if row["id"] in CODE_CELLS_CHANGED_SINCE:
            continue
        assert digest(cell.source) == row["sha256"], (
            "code cell %s changed; only markdown headings may differ, unless "
            "the cell is named in CODE_CELLS_CHANGED_SINCE with its PR"
            % row["id"])


def test_the_renumbering_touched_markdown_and_nothing_else():
    nb = notebook()
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))["cells"]
    current = {c.get("id"): c for c in nb.cells}

    changed = [r for r in recorded
               if digest(current[r["id"]].source) != r["sha256"]]

    assert changed, "nothing was renumbered at all"
    moved = [r["id"] for r in changed if r["cell_type"] != "markdown"]
    assert set(moved) <= set(CODE_CELLS_CHANGED_SINCE), moved


# ===========================================================================
# 2. the notebook's new numbering
# ===========================================================================
def test_the_twelve_step_headings_are_there_in_order():
    nb = notebook()
    lines = [line for cell in official_flow_cells(nb)
             if cell.cell_type == "markdown"
             for line in cell.source.splitlines()
             if re.match(r"^## Step \d", line)]
    found = [line for line in lines if " addendum" not in line]

    assert found == NEW_HEADINGS, found
    # No level-2 addendum heading is left: each addendum is a subsection of the
    # step it belongs to, so it reads in sequence rather than as a 13th step.
    assert [line for line in lines if " addendum" in line] == []
    body = "\n".join(c.source for c in official_flow_cells(nb)
                     if c.cell_type == "markdown")
    for heading in ("### 11.8 Fusion across all approaches and baselines "
                    "(Step 11 addendum)",
                    "### 12.9 Explainability across all approaches and "
                    "baselines (Step 12 addendum)"):
        assert heading in body, heading


def test_the_contents_list_has_twelve_links_that_all_resolve():
    nb = notebook()
    contents = next(c for c in nb.cells
                    if c.cell_type == "markdown"
                    and c.source.lstrip().startswith("## Contents"))
    links = re.findall(r"\]\(#([^()\s]+)\)", contents.source)
    headings = set()
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        for match in re.finditer(r"^#{2,4} (.+)$", cell.source, re.M):
            headings.add(github_anchor(match.group(1)))

    step_links = [link for link in links if link.startswith("step-")]
    assert len(step_links) == 12, step_links
    for link in links:
        assert link in headings, link


def test_the_numbering_note_is_at_the_top_of_the_notebook():
    nb = notebook()
    head = "\n".join(c.source for c in nb.cells[:6]
                     if c.cell_type == "markdown")
    sentence = ("Step numbers in this notebook refer to the simplified "
                "official-dataset flow; module names and function names inside "
                "src/mscad/ keep the earlier numbering for backward "
                "compatibility.")

    assert re.sub(r"[\s>]+", " ", sentence) in re.sub(r"[\s>]+", " ", head)


def test_the_note_maps_every_new_step_number_to_its_unchanged_folder():
    nb = notebook()
    head = "\n".join(c.source for c in nb.cells[:6])
    flat = re.sub(r"[\s>]+", " ", head)

    for number, attribute in STEP_FOLDERS.items():
        folder = Path(getattr(config, attribute)).name
        assert re.search(r"\*\*Step %d\*\*.{0,200}?`%s\\`"
                         % (number, folder), flat), (number, folder)


def test_the_official_flow_no_longer_mentions_the_pcap_conversion():
    nb = notebook()
    # A pointer to the companion notebook by name is fine; describing the
    # conversion, or calling the own-extraction functions, is not.
    forbidden = ("editcap", "tshark", "PCAPNG",
                 "acquire_captures(", "merge_flow_csvs(",
                 "check_column_consistency(", "save_step3_reports(")

    for i, cell in enumerate(official_flow_cells(nb)):
        for word in forbidden:
            assert word not in cell.source, (i, cell.get("id"), word)


def test_the_own_extraction_cells_are_gone():
    """Removed on purpose: the own extraction has its own notebook."""
    nb = notebook()
    body = "\n".join(c.source for c in nb.cells)

    assert OWN_BANNER not in body
    assert not [line for c in nb.cells if c.cell_type == "markdown"
                for line in c.source.splitlines() if line.startswith("### A.")]
    assert not [c for c in nb.cells if c.cell_type == "code"
                and "OWN_STEP3_SKIP_NOTE" in c.source]


# ===========================================================================
# 3. rules 3 and 4 -- the folders did not move, so nothing is retrained
# ===========================================================================
def test_no_output_folder_was_renamed_so_no_fallback_is_needed():
    """'Notebook markdown only, folders unchanged' is the chosen approach."""
    for number, attribute in STEP_FOLDERS.items():
        folder = Path(getattr(config, attribute))
        assert folder.name == "step%d" % (number + 2), (number, folder.name)
    assert Path(config.MERGED_CSV).parent.name == "step3"
    assert Path(config.STEP3_OUTPUT_DIR).name == "step3"


def test_the_retraining_lock_is_still_on():
    assert config.LOCK_TRAINED_MODELS is True


def test_the_lock_still_refuses_a_retrain_after_the_renumbering(tmp_path,
                                                               monkeypatch):
    from src.mscad import training_lock

    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True)
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "outputs_official")
    monkeypatch.setattr(config, "STEP11_OUTPUT_DIR",
                        tmp_path / "outputs_official" / "step11")
    config.STEP11_OUTPUT_DIR.mkdir(parents=True)
    (config.STEP11_OUTPUT_DIR / "metrics.json").write_text(
        json.dumps({"macro_f1": 0.5}), encoding="utf-8")

    name = next(iter(training_lock.TARGETS))
    assert training_lock.locked(name)
    with pytest.raises(training_lock.TrainingLocked) as error:
        training_lock.guard(name, lambda: "trained")

    assert "metrics.json" in str(error.value)
    assert "LOCK_TRAINED_MODELS" in str(error.value)


# ===========================================================================
# 4. Step 1 -- the loader, the self-check and the comparison
# ===========================================================================
@pytest.fixture
def official_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATASET_SOURCE", "official")
    monkeypatch.setattr(config, "RUNNING_OFFICIAL", True)
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "outputs_official")
    monkeypatch.setattr(config, "OUTPUTS_ROOT_OWN", tmp_path / "outputs")
    monkeypatch.setattr(config, "OUTPUTS_ROOT_OFFICIAL",
                        tmp_path / "outputs_official")
    monkeypatch.setattr(config, "STEP3_OUTPUT_DIR",
                        tmp_path / "outputs_official" / "step3")
    monkeypatch.setattr(config, "STEP4_OUTPUT_DIR",
                        tmp_path / "outputs_official" / "step4")
    monkeypatch.setattr(config, "MERGED_CSV",
                        tmp_path / "outputs_official" / "step3"
                        / "MSCAD_merged.csv")
    monkeypatch.setattr(config, "BENIGN_LABELS", ["Normal"])
    monkeypatch.setattr(config, "METADATA_COLUMNS",
                        list(config.OFFICIAL_IDENTIFIER_COLUMNS)
                        + list(config.OFFICIAL_ROW_COLUMNS))
    monkeypatch.setattr(config, "METADATA_SHARED_COLUMNS", [])
    monkeypatch.setattr(config, "OFFICIAL_CSV",
                        tmp_path / "CSV_Original_kaggle" / "MSCAD.csv")
    return tmp_path


def write_official_csv(rows_per_class=5):
    """A tiny file in the official layout: six classes, one quoted header."""
    features = ["Flow Duration", "Tot Fwd Pkts", "Flow Byts/s", "Idle Mean"]
    labels = []
    for name in config.OFFICIAL_LABELS:
        labels.extend([name] * rows_per_class)
    n = len(labels)
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({name: np.round(rng.random(n) * (i + 2), 4)
                          for i, name in enumerate(features)})
    frame["Protocol"] = 6
    frame["Label"] = labels
    frame.columns = ["'Idle Mean'" if c == "Idle Mean" else c
                     for c in frame.columns]
    path = Path(config.OFFICIAL_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame, path


def test_step_2_refuses_when_step_1_has_not_written_its_file(official_mode):
    from src.mscad import preprocessing

    with pytest.raises(Exception) as error:
        preprocessing.run_step4(verbose=False)

    message = str(error.value)
    assert str(Path(config.STEP3_OUTPUT_DIR) / "MSCAD_merged.csv") in message
    assert "Step 1" in message and "Step 2" in message
    assert "nothing was written" in message.lower()
    assert not Path(config.STEP4_OUTPUT_DIR).exists()


def test_the_self_check_passes_on_the_file_step_1_wrote(official_mode):
    source, _path = write_official_csv()

    od.run_step3(verbose=False)
    result = od.self_check(verbose=False)

    assert result["ok"] is True
    assert result["worst_difference"] == 0.0
    assert len(result["numeric_columns"]) >= 4
    table = result["table"]
    assert len(table) == 3
    assert (table["result"] == "[OK]").all()
    assert int(table.loc[0, "source (MSCAD.csv)"]) == len(source)


def test_the_self_check_fails_when_the_saved_file_is_tampered_with(
        official_mode):
    write_official_csv()
    od.run_step3(verbose=False)
    saved = Path(config.STEP3_OUTPUT_DIR) / "MSCAD_merged.csv"
    frame = pd.read_csv(saved)
    frame["Flow Duration"] = frame["Flow Duration"] + 1000.0
    frame.to_csv(saved, index=False)

    result = od.self_check(verbose=False)

    assert result["ok"] is False
    assert result["worst_column"] == "Flow Duration"
    assert result["worst_difference"] > 0


def test_the_self_check_says_so_before_step_1_has_run(official_mode):
    write_official_csv()

    result = od.self_check(verbose=False)

    assert result["ok"] is False
    assert "does not exist yet" in result["detail"]


def test_the_comparison_says_so_when_the_own_file_is_not_there(official_mode):
    write_official_csv()
    od.run_step3(verbose=False)

    result = od.own_extraction_label_comparison(verbose=False)

    assert result["available"] is False
    assert "no own-extraction" in result["detail"]
    assert list(result["table"].columns) == ["label", "official rows"]


def test_the_comparison_puts_the_two_label_tables_side_by_side(official_mode):
    write_official_csv()
    od.run_step3(verbose=False)
    own = Path(config.OUTPUTS_ROOT_OWN) / "step3" / "MSCAD_merged.csv"
    own.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Label": ["Benign"] * 7 + ["Brute_Force"] * 3}).to_csv(
        own, index=False)

    result = od.own_extraction_label_comparison(verbose=False)

    assert result["available"] is True
    assert result["own_total"] == 10
    table = result["table"]
    assert set(table.columns) == {"label", "official rows", "official %",
                                  "own extraction rows", "own extraction %",
                                  "in both"}
    benign = table[table["label"] == "Benign"].iloc[0]
    assert benign["own extraction rows"] == 7
    assert benign["official rows"] == 0
    assert bool(benign["in both"]) is False
    both = table[table["label"] == "Brute_Force"].iloc[0]
    assert bool(both["in both"]) is True


def test_the_comparison_is_the_note_the_write_up_needs(official_mode, capsys):
    write_official_csv()
    od.run_step3(verbose=False)
    own = Path(config.OUTPUTS_ROOT_OWN) / "step3" / "MSCAD_merged.csv"
    own.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Label": ["Normal"] * 4}).to_csv(own, index=False)

    od.own_extraction_label_comparison(verbose=True)

    printed = capsys.readouterr().out
    assert "not used for training" in printed


def test_the_loader_message_names_the_file_and_both_step_numbers(
        official_mode):
    Path(config.STEP3_OUTPUT_DIR).mkdir(parents=True)

    with pytest.raises(od.OfficialDatasetError) as error:
        od.load_merged_official(verbose=False)

    message = str(error.value)
    assert str(Path(config.STEP3_OUTPUT_DIR) / "MSCAD_merged.csv") in message
    assert "Step 2 (preprocessing)" in message
    assert "Step 1 (data acquisition)" in message
    assert "run_step3" in message            # the old name, on purpose


# ===========================================================================
# 5. rule 5 -- the own extraction's results are byte-identical afterwards
# ===========================================================================
def test_step_1_changes_no_byte_under_the_own_output_root(official_mode):
    """Three saved model files, checksummed before and after Step 1 runs."""
    own_root = Path(config.OUTPUTS_ROOT_OWN)
    saved = {
        own_root / "step7" / "encoder.keras": b"\x89KERAS-encoder\x00binary",
        own_root / "step10" / "cnn_lstm.keras": b"\x89KERAS-cnnlstm\x00binary",
        own_root / "step11" / "metrics.json":
            b'{"macro_f1": 0.5836, "accuracy": 0.5836}',
    }
    for path, blob in saved.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in saved}
    listing_before = sorted(p.relative_to(own_root).as_posix()
                            for p in own_root.rglob("*"))

    write_official_csv()
    od.run_step3(verbose=False)
    od.self_check(verbose=False)
    od.own_extraction_label_comparison(verbose=False)
    od.load_merged_official(verbose=False)

    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in saved}
    assert after == before
    assert sorted(p.relative_to(own_root).as_posix()
                  for p in own_root.rglob("*")) == listing_before


def test_own_mode_still_routes_everything_to_the_own_root(monkeypatch):
    """Rule 5: `DATASET_SOURCE = "own"` is exactly what it was."""
    import importlib

    monkeypatch.setenv("MSCAD_DATASET_SOURCE", "own")
    own = importlib.reload(config)
    try:
        assert own.DATASET_SOURCE == "own"
        assert own.RUNNING_OFFICIAL is False
        assert Path(own.OUTPUTS_ROOT).name == "outputs"
        assert own.BENIGN_LABELS == ["Benign", "Normal"]
        for attribute in STEP_FOLDERS.values():
            folder = Path(getattr(own, attribute))
            assert folder.parent.name == "outputs", folder
        assert Path(own.MERGED_CSV).parent.name == "step3"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_own_extraction_step_3_report_writer_is_untouched():
    """The own merge path still has all five of its entry points."""
    from src.mscad import data_loading

    for name in ("check_column_consistency", "merge_flow_csvs", "save_merged",
                 "save_step3_reports", "summarize_dataset"):
        assert callable(getattr(data_loading, name)), name
