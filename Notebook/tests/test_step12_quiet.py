r"""
tests/test_step12_quiet.py -- the Step 12 run cells without their superseded
tables.

NEW FILE.  No existing test file, src function or notebook cell is modified by
these tests; they only read.  Nothing is trained: the run functions are
replaced by stubs that stand in for them, and the real ones are checked
statically.  Never C:\\MSCAD2022.

What is pinned down:

  * `comparison.load_step12_quiet` returns the run function's own result,
    unchanged and by identity;
  * it calls it with `verbose=False` and passes every other argument through;
  * it prints no table -- one short status line per configuration instead;
  * the lines carry the `STEP12_DISPLAY_ORDER` labels, so no "Proposed:"
    reaches the output;
  * a configuration with no saved result is never reported as loaded;
  * every display call in the three real run modules is inside an
    `if verbose` branch, which is what makes `verbose=False` silent;
  * Step 12's notebook cells show the superseded tables no more, and the
    notebook carries no stored outputs.

No pathlib patching: the output folder is redirected with monkeypatch and
Windows paths are built with PureWindowsPath.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

# ---------------------------------------------------------------------------
# NOTEBOOK OUTPUT POLICY (shared with tests/test_notebook_structure.py)
# ---------------------------------------------------------------------------
# One policy, stated once, enforced in two places: THE NOTEBOOKS IN THIS REPO
# CARRY NO STORED OUTPUTS AND NO EXECUTION COUNTS.  Results live under the
# output root on disk (C:\MSCAD2022\outputs_official\), never inside the
# .ipynb, so a diff of a notebook shows changed code and nothing else.
#
#   * the snapshot in tests/data/notebook_code_cells.json therefore records a
#     CLEARED notebook: every execution_count is null and every outputs list
#     is empty.  It is not a record of "what was last run";
#   * test_notebook_structure pins cell ids, source, execution_count and
#     outputs against that snapshot;
#   * test_the_notebook_carries_no_stored_outputs, below, is the guardrail
#     that keeps the cleared state true for the Step 12 cells.
#
# So a failure here reading `assert 1 == None` is NOT a stale snapshot: it
# means a notebook was saved after being run.  Fix it at the cause -- in
# Jupyter, Kernel > Restart Kernel and Clear Outputs of All Cells, save, and
# re-run the suite.  Do not re-record the snapshot from a run notebook: that
# would bake results into the repo and silently retire the policy.


import ast
import io
import json
import os
import re
from pathlib import Path, PureWindowsPath

import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp

PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_DIR / "notebooks" / "MSCAD_Pipeline.ipynb"

# the four configurations the Step 12 run cell covers
STEP12_KEYS = [spec["name"] for spec in config.EXPERIMENTS]


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    """No test may look at the real output folder."""
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR", tmp_path / "step12")
    monkeypatch.setattr(config, "STEP11B_OUTPUT_DIR",
                        tmp_path / "step11_attention", raising=False)


def save_result(key: str, f1: float = 0.5) -> None:
    """A minimal metrics.json for one configuration, where 12.0 looks for it."""
    folder = config.STEP12_OUTPUT_DIR / key
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(
        json.dumps({"config": key, "full": {"f1_macro": f1}}), encoding="utf-8")


def fake_runner(result=None, keys=STEP12_KEYS, status="loaded_existing"):
    """A stand-in for a run function: prints a table when it is verbose.

    The table is printed exactly where the real functions print theirs -- at
    the end, under `if verbose` -- so a wrapper that forgot to switch verbose
    off would show up in the captured output.
    """
    calls = []
    runs = [{"spec": {"name": key, "label": "Proposed: FS -> AE -> CNN-LSTM"
                      if key == "proposed" else key.upper()},
             "status": status} for key in keys]
    out = {"runs": runs, "status": "written",
           "comparison": pd.DataFrame([{"config": run["spec"]["label"],
                                        "F1 (macro)": 0.5} for run in runs])}
    if result is not None:
        out = result

    def runner(verbose=True, **kwargs):
        calls.append({"verbose": verbose, **kwargs})
        if verbose:
            print("### Step 12 -- comparison")
            print(out["comparison"].to_string()
                  if isinstance(out.get("comparison"), pd.DataFrame) else "")
        return out

    runner.calls = calls
    runner.out = out
    return runner


# ===========================================================================
# the wrapper returns exactly what the run function returned
# ===========================================================================
def test_the_quiet_loader_returns_the_run_functions_own_result(capsys):
    """Presentation only: the result object itself must come straight back."""
    runner = fake_runner()

    out = cmp.load_step12_quiet(runner)
    capsys.readouterr()

    assert out is runner.out
    assert list(out) == list(runner.out)
    assert out["runs"] == runner.out["runs"]
    pd.testing.assert_frame_equal(out["comparison"], runner.out["comparison"])


def test_the_quiet_loader_asks_for_verbose_false_and_passes_the_rest_through(
        capsys):
    runner = fake_runner()

    cmp.load_step12_quiet(runner, epochs=3, overwrite=False)
    capsys.readouterr()

    assert runner.calls == [{"verbose": False, "epochs": 3, "overwrite": False}]


def test_the_quiet_loader_defaults_to_the_step_12_run_function(monkeypatch,
                                                              capsys):
    runner = fake_runner()
    monkeypatch.setattr(cmp.xp, "run_step12", runner)

    out = cmp.load_step12_quiet()
    capsys.readouterr()

    assert out is runner.out
    assert runner.calls == [{"verbose": False}]


# ===========================================================================
# ... and prints no table
# ===========================================================================
def test_the_quiet_loader_prints_no_table(capsys):
    runner = fake_runner()

    cmp.load_step12_quiet(runner)
    printed = capsys.readouterr().out

    assert "Step 12 -- comparison" not in printed
    assert "F1 (macro)" not in printed
    # the stub's table is a DataFrame repr; none of it may survive
    assert "0.5" not in printed
    assert len(printed.strip().split("\n")) == len(STEP12_KEYS)


def test_the_status_lines_use_the_display_order_labels(capsys):
    runner = fake_runner()

    cmp.load_step12_quiet(runner)
    printed = capsys.readouterr().out

    assert "Proposed:" not in printed          # the old spec label
    for key in STEP12_KEYS:
        label = dict(config.STEP12_DISPLAY_ORDER)[key]
        assert label in printed
    lines = printed.strip().split("\n")
    assert lines[0].startswith("Approach 1: FS -> AE -> CNN-LSTM")
    assert all(line.endswith("loaded (saved result)") for line in lines)


def test_one_line_per_configuration_whatever_the_section(capsys):
    """The Baseline 4 and Baseline 5 cells pass their own longer key lists."""
    keys = [spec["name"] for spec in config.EXPERIMENTS_WITH_PCA] \
        + [config.BASELINE5_NAME]
    runner = fake_runner(keys=keys)

    cmp.load_step12_quiet(runner, keys=keys)
    lines = capsys.readouterr().out.strip().split("\n")

    assert len(lines) == len(keys) == 6
    assert lines[-1].startswith(dict(config.STEP12_DISPLAY_ORDER)["baseline5"])


def test_a_trained_configuration_and_a_loaded_one_read_differently(capsys):
    runner = fake_runner(status="written")

    cmp.load_step12_quiet(runner)
    printed = capsys.readouterr().out

    assert printed.count("trained and saved") == len(STEP12_KEYS)
    assert "loaded (saved result)" not in printed


def test_show_false_prints_nothing_at_all(capsys):
    runner = fake_runner()

    out = cmp.load_step12_quiet(runner, show=False)

    assert capsys.readouterr().out == ""
    assert out is runner.out


# ===========================================================================
# a line never claims a result that is not there
# ===========================================================================
def test_a_configuration_with_no_saved_result_is_not_reported_as_loaded():
    """The load-from-disk path returns no `runs`, so the files are checked."""
    save_result("proposed")

    lines = cmp.run_status_lines({"status": "loaded_existing"},
                                 keys=STEP12_KEYS)

    assert lines[0].endswith("loaded (saved result)")
    for line in lines[1:]:
        assert "not available" in line
        assert "OVERWRITE_STEP12" in line      # says how to produce it


def test_the_run_functions_own_status_wins_over_the_files():
    """A configuration trained in this call reads `trained and saved` even
    before anything is on disk."""
    out = {"runs": [{"spec": {"name": "proposed"}, "status": "written"}]}

    lines = cmp.run_status_lines(out, keys=["proposed"])

    assert lines == ["Approach 1: FS -> AE -> CNN-LSTM ... trained and saved"]


def test_the_status_lines_survive_an_unusable_output_folder(monkeypatch,
                                                           capsys,
                                                           assert_no_leaked_output):
    """On a machine where C:\\MSCAD2022 is not reachable, the lines still
    print and nothing is written."""
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step12"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(Path(str(config.STEP11B_OUTPUT_DIR.parent)))

    lines = cmp.run_status_lines({}, keys=["proposed"])

    assert len(lines) == 1
    assert "not available" in lines[0]
    capsys.readouterr()
    assert_no_leaked_output("run_status_lines")


def test_an_unknown_key_falls_back_to_the_key_itself():
    assert cmp.display_label("not_a_configuration") == "not_a_configuration"
    assert cmp.display_label("baseline4_ae_pca_lstm") == "B4: AE -> PCA -> LSTM"


# ===========================================================================
# why verbose=False is enough: the real functions guard every display call
# ===========================================================================
DISPLAY_CALLS = {"show_table", "show_title", "show_note", "print", "display"}


def unguarded_display_calls(path: Path) -> list[tuple[int, str]]:
    """Every display call in a module that is NOT under an `if verbose` test.

    Lexical, which is the point: it does not matter what the flag is called at
    runtime, only that no table is printed on a code path a caller cannot turn
    off.  (An `if not verbose:` branch would count as guarded; there are none,
    and the test below would fail the moment one printed a table.)
    """
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    found: list[tuple[int, str]] = []

    def walk(node, guarded: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inner = guarded
            if isinstance(node, ast.If) and child in node.body:
                names = {n.id for n in ast.walk(node.test)
                         if isinstance(n, ast.Name)}
                inner = guarded or "verbose" in names
            if isinstance(child, ast.Call):
                name = (getattr(child.func, "id", None)
                        or getattr(child.func, "attr", None))
                if name in DISPLAY_CALLS and not inner:
                    found.append((child.lineno, name))
            walk(child, inner)

    walk(tree, False)
    return found


@pytest.mark.parametrize("module", ["experiments", "experiments_pca",
                                    "experiments_baseline5"])
def test_every_display_call_in_the_run_modules_is_behind_verbose(module):
    path = PROJECT_DIR / "src" / "mscad" / ("%s.py" % module)

    assert unguarded_display_calls(path) == []


@pytest.mark.parametrize("module,function", [
    ("experiments", "run_step12"),
    ("experiments_pca", "run_step12_with_baseline4"),
    ("experiments_baseline5", "run_baseline5_comparison"),
])
def test_the_run_functions_still_take_a_verbose_flag(module, function):
    import importlib
    import inspect

    fn = getattr(importlib.import_module("src.mscad.%s" % module), function)
    parameter = inspect.signature(fn).parameters.get("verbose")

    assert parameter is not None, "%s.%s lost its verbose flag" % (module,
                                                                  function)
    assert parameter.default is True          # the notebook asks for False


# ===========================================================================
# and the notebook itself
# ===========================================================================
def step12_cells():
    """The model-comparison cells of the notebook, or a skip without nbformat.

    Located by HEADING TEXT rather than by step number: the official-dataset
    flow renumbered the notebook's markdown headings (model comparison is
    "Step 10" there, the fusion that follows it "Step 11"), while every module,
    function and output folder keeps the earlier numbering -- so this file,
    which is about `comparison.py` and the Step 12 cells' behaviour, still
    tests exactly what it always did.
    """
    nbformat = pytest.importorskip("nbformat")
    if not NOTEBOOK.exists():                  # pragma: no cover
        pytest.skip("the notebook is not in this checkout")
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    cells = nb.cells
    starts = [i for i, c in enumerate(cells)
              if c.cell_type == "markdown"
              and re.search(r"^## Step \d+ - Model Performance Evaluation$", c.source,
                            re.M)]
    ends = [i for i, c in enumerate(cells)
            if c.cell_type == "markdown"
            and re.search(r"^## Step \d+ - Adaptive Evidence Fusion$",
                          c.source, re.M)]
    assert starts and ends, "Step 12 could not be located in the notebook"
    return cells, range(starts[0], ends[0])


def test_the_step_12_run_cells_never_call_a_runner_that_prints_the_old_tables():
    """The Step 12 completion goes one step further than the quiet loader.

    12.3 to 12.5 now LOAD the finished configurations and print one status line
    each -- no runner is called from Step 12 at all, so the superseded tables
    cannot be printed and no existing model can be retrained by re-running a
    cell.  The quiet loader itself is unchanged and still available for a
    configuration that has never been trained; what this test guards is that
    no Step 12 cell calls a runner bare, which is what printed the old tables.
    """
    cells, span = step12_cells()
    code = "\n".join(cells[i].source for i in span
                     if cells[i].cell_type == "code")

    for runner in ("experiments.run_step12", "run_step12_with_baseline4",
                   "run_baseline5_comparison"):
        assert "= %s()" % runner not in code
        assert "%s(" % runner not in code      # not called from Step 12 at all


def test_the_quiet_loader_is_still_there_for_an_untrained_configuration():
    """Nothing about the loader itself changed; only who calls it."""
    import inspect

    parameter = inspect.signature(cmp.load_step12_quiet).parameters
    assert "runner" in parameter and "show" in parameter
    assert parameter["show"].default is True


def test_step_12_shows_the_superseded_tables_no_more():
    cells, span = step12_cells()
    code = "\n".join(cells[i].source for i in span
                     if cells[i].cell_type == "code")

    assert "architecture_table()" not in code      # the 5- and 6-row versions
    assert 'e["label"]' not in code                # the 4-row version
    # 12.1 and 12.7 each keep exactly one table of their own.  The Step 12
    # completion replaced the 7-row pair with the 8-row pair, so the names
    # moved; the property this test exists for -- one, and only one, of each --
    # is the same.
    assert code.count("configuration_table_8()") == 1
    assert code.count("configuration_table()") == 0
    # PR #48: the one table path is the honest renderer, which calls
    # run_step12_complete itself (quietly) for the checks, CSVs and figures.
    # Still exactly one entry point, and still no second table.
    assert code.count("show_honest_step10(") == 1
    assert code.count("run_step12_complete(") == 0
    assert code.count("run_step12_full()") == 0


def test_the_notebook_carries_no_stored_outputs():
    cells, _span = step12_cells()

    for i, cell in enumerate(cells):
        if cell.cell_type != "code":
            continue
        assert not (cell.get("outputs") or []), (
            "cell %d has stored output -- clear outputs and execution counts "
            "and save again (see NOTEBOOK OUTPUT POLICY above)" % i)
        assert cell.get("execution_count") is None, (
            "cell %d has an execution count -- clear outputs and execution "
            "counts and save again (see NOTEBOOK OUTPUT POLICY above)" % i)
