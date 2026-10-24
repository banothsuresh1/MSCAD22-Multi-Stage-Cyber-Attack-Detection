r"""
tests/test_notebook_structure.py -- the notebooks read in step order.

NEW FILE.  Presentation only: it reads the two notebooks with nbformat and
asserts nothing about the pipeline's behaviour.  What it holds in place:

  * every code cell is byte-identical to the snapshot in
    tests/data/notebook_code_cells.json -- source, outputs and
    execution_count -- so a presentation pass can never touch code or results;
  * the step headings read 1 to 12 and then Appendix A, in that order;
  * no step number appears twice;
  * no drafting leftovers ("no code yet", "TODO", "coming next") survive;
  * every code cell sits under a heading a reader can name.

Nothing is executed and nothing is written: both notebooks are opened
read-only.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS = PROJECT_DIR / "notebooks"
MAIN = NOTEBOOKS / "MSCAD_Pipeline.ipynb"
CONVERTER = NOTEBOOKS / "convert_pcapng_to_csvflows.ipynb"
SNAPSHOT = Path(__file__).resolve().parent / "data" / "notebook_code_cells.json"

STEP_HEADINGS = [
    "Step 1 - Data acquisition",
    "Step 2 - Preprocessing",
    "Step 3 - Train-Test Validation Split",
    "Step 4 - Feature Selection (MI + Pearson + PI to top 30)",
    "Step 5 - Autoencoder feature extraction",
    "Step 6 - Transform the training and test sets",
    "Step 7 - Class imbalance handled in the training loss",
    "Step 8 - Hybrid classifiers",
    "Step 9 - Evaluation on the untouched test set",
    "Step 10 - Model comparison",
    "Step 11 - Adaptive Evidence Fusion",
    "Step 12 - Explainability Layer",
]
# The own extraction has its own notebook; this one is the official flow only.
APPENDIX = "Appendix A - Own extraction (reference only)"
# The sections that are not steps but legitimately hold code or prose.
NON_STEP_SECTIONS = {"Objective", "Pipeline overview", "Contents", "Setup",
                     "Summary"}


def read(path):
    nbformat = pytest.importorskip("nbformat")
    if not path.exists():                          # pragma: no cover
        pytest.skip("%s is not in this checkout" % path.name)
    return nbformat.read(str(path), as_version=4)


def level_two(nb):
    """Every level-2 heading, in document order."""
    out = []
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        for match in re.finditer(r"^## (.+)$", cell.source, re.M):
            out.append(match.group(1).strip())
    return out


def sections(nb):
    """(section title, cell) for every cell, by the heading above it."""
    current = None
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            found = re.findall(r"^## (.+)$", cell.source, re.M)
            if found:
                current = found[-1].strip()
        yield current, cell


def github_anchor(heading):
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


# ===========================================================================
# 1. no code cell changed -- source, outputs and execution_count
# ===========================================================================
@pytest.mark.parametrize("name", ["MSCAD_Pipeline.ipynb",
                                  "convert_pcapng_to_csvflows.ipynb"])
def test_every_code_cell_matches_the_snapshot(name, capsys):
    recorded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["notebooks"]
    assert name in recorded, name
    nb = read(NOTEBOOKS / name)

    code = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code) == len(recorded[name]), (
        "%d code cells now, %d in the snapshot" % (len(code),
                                                   len(recorded[name])))
    for cell, row in zip(code, recorded[name]):
        assert cell.get("id") == row["id"], (cell.get("id"), row["id"])
        assert cell.get("execution_count") == row["execution_count"], row["id"]
        source = hashlib.sha256(cell.source.encode("utf-8")).hexdigest()
        assert source == row["source_sha256"], "source changed: %s" % row["id"]
        outputs = hashlib.sha256(
            json.dumps([dict(o) for o in cell.get("outputs", [])],
                       sort_keys=True,
                       ensure_ascii=False).encode("utf-8")).hexdigest()
        assert outputs == row["outputs_sha256"], (
            "outputs changed: %s" % row["id"])

    with capsys.disabled():
        print("\n[OK] code cells unchanged  (%s: %d cells)" % (name, len(code)))


# ===========================================================================
# 2. the order a reader sees
# ===========================================================================
def test_the_step_headings_read_one_to_twelve():
    nb = read(MAIN)
    wanted = [h for h in level_two(nb) if h.startswith("Step ")]

    assert wanted == STEP_HEADINGS, wanted


def test_no_step_number_appears_twice():
    nb = read(MAIN)
    numbers = [re.match(r"Step (\d+) -", h).group(1)
               for h in level_two(nb) if re.match(r"Step \d+ -", h)]

    assert numbers == sorted(numbers, key=int), numbers
    assert len(numbers) == len(set(numbers)), numbers
    assert numbers == [str(n) for n in range(1, 13)], numbers


def test_the_own_extraction_appendix_is_gone():
    """It moved to convert_pcapng_to_csvflows.ipynb; nothing is left behind."""
    nb = read(MAIN)
    headings = level_two(nb)

    assert APPENDIX not in headings, headings[-3:]
    assert headings[-1] == "Summary", headings[-3:]
    assert headings[-2] == STEP_HEADINGS[-1], headings[-3:]
    body = "\n".join(c.source for c in nb.cells)
    for marker in ("## Appendix A", "### A.1 ", "### A.11 ",
                   "OWN_STEP3_SKIP_NOTE", "acquire_captures(",
                   "merge_flow_csvs(", "save_step3_reports("):
        assert marker not in body, marker


def test_the_contents_list_covers_every_section_in_order():
    nb = read(MAIN)
    contents = next(c for c in nb.cells if c.cell_type == "markdown"
                    and c.source.lstrip().startswith("## Contents"))
    links = re.findall(r"^\d+\. \[([^]]+)\]\(#([^()\s]+)\)$", contents.source,
                       re.M)

    assert [label for label, _anchor in links] == STEP_HEADINGS
    for label, anchor in links:
        assert anchor == github_anchor(label), (label, anchor)
    anchors = {github_anchor(h) for h in level_two(nb)}
    for _label, anchor in links:
        assert anchor in anchors, anchor


# ===========================================================================
# 3. what must not be left behind
# ===========================================================================
@pytest.mark.parametrize("path", [MAIN, CONVERTER])
def test_no_drafting_leftovers(path):
    nb = read(path)
    forbidden = re.compile(
        r"no code yet|coming next|\bTODO\b|\bFIXME\b|placeholder cell|"
        r"scratch cell|debug cell", re.I)

    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "markdown":
            continue
        found = forbidden.search(cell.source)
        assert not found, (path.name, index, found.group(0))


@pytest.mark.parametrize("path", [MAIN, CONVERTER])
def test_no_markdown_cell_is_only_a_rule_or_a_single_word(path):
    nb = read(path)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "markdown":
            continue
        text = cell.source.strip()
        assert text, (path.name, index, "empty markdown cell")
        assert text not in ("---", "***", "___"), (path.name, index)
        assert len(text.split()) > 1, (path.name, index, text)


def test_every_code_cell_sits_under_a_named_section():
    nb = read(MAIN)
    allowed = set(STEP_HEADINGS) | NON_STEP_SECTIONS

    for section, cell in sections(nb):
        if cell.cell_type != "code":
            continue
        assert section is not None, "a code cell precedes every heading"
        assert section in allowed, (section, cell.get("id"))


def test_the_converter_notebook_is_marked_as_the_appendix():
    nb = read(CONVERTER)
    first = nb.cells[0].source

    assert first.startswith("# Appendix A - Own extraction")
    assert "results in Steps 1-12" in first
    assert "Step 1 - PCAPNG" not in first          # the drifted old number
