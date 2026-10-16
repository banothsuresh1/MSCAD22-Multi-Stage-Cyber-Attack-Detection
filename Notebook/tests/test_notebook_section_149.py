r"""
tests/test_notebook_section_149.py -- section 14.9 of the notebook must run
from a fresh kernel.

NEW FILE.  No existing test file, src module, config value or saved output is
touched; this only reads the notebook.

Why it exists: 14.9 is the last section of the notebook and a reader opens it
on its own -- restart the kernel, run Step 14, run 14.9.  Nothing from Step 1's
cells is in the namespace then, so a cell that says `np.fill_diagonal(...)`
without an import raises `NameError: name 'np' is not defined`.  That is exactly
what happened, and this test is the guard: every name 14.9 uses must be
imported or defined inside 14.9 itself.

The check is a sequential-execution simulation over the section's code cells,
not a run: it collects what each cell binds and flags every name a later cell
loads without it ever having been bound.  One scope is used for the whole
section, which is how a notebook namespace actually behaves.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import ast
import builtins
from pathlib import Path

import pytest

NOTEBOOK = (Path(__file__).resolve().parent.parent / "notebooks"
            / "MSCAD_Pipeline.ipynb")
SECTION = "step149"                     # the cell-id prefix 14.9's cells carry
SETUP_ID = "step149-setup"


def section_cells():
    """The 14.9 code cells, in notebook order."""
    nbformat = pytest.importorskip("nbformat")
    if not NOTEBOOK.exists():                      # pragma: no cover
        pytest.skip("the notebook is not in this checkout")
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    cells = [c for c in nb.cells
             if str(c.get("id", "")).startswith(SECTION)
             and c.cell_type == "code"]
    assert cells, "no 14.9 code cells found in the notebook"
    return cells


def bound_names(tree: ast.AST) -> set[str]:
    """Every name one cell binds: imports, assignments, loops, defs, walrus."""
    names: set[str] = set()

    def add_target(node) -> None:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                add_target(element)
        elif isinstance(node, ast.Starred):
            add_target(node.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.For, ast.AsyncFor)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            for target in targets:
                add_target(target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            add_target(node.target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Lambda, ast.FunctionDef,
                               ast.AsyncFunctionDef)):
            pass
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.comprehension):
            add_target(node.target)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            add_target(node.optional_vars)

    for node in ast.walk(tree):                    # parameters of any callable
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda)):
            arguments = node.args
            for group in (arguments.posonlyargs, arguments.args,
                          arguments.kwonlyargs):
                names.update(argument.arg for argument in group)
            for single in (arguments.vararg, arguments.kwarg):
                if single is not None:
                    names.add(single.arg)
    return names


def loaded_names(tree: ast.AST) -> list[tuple[str, int]]:
    """(name, line) for every name a cell reads."""
    return [(node.id, node.lineno) for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)]


def undefined_names() -> list[tuple[int, str, int]]:
    """(cell number, name, line) for every name used before it is bound.

    Statement by statement, in order, with each statement's own bindings added
    before its loads are checked -- otherwise `for row in rows:` would report
    `row` as undefined inside its own loop.  One namespace spans the section,
    which is how a notebook kernel behaves.
    """
    return scan([cell.source for cell in section_cells()])


def scan(sources) -> list[tuple[int, str, int]]:
    """The same check over any list of cell sources, so it can be tested."""
    known = set(dir(builtins))
    missing: list[tuple[int, str, int]] = []
    for number, source in enumerate(sources, start=1):
        for statement in ast.parse(source).body:
            known |= bound_names(statement)
            for name, line in loaded_names(statement):
                if name not in known:
                    missing.append((number, name, line))
    return missing


# ===========================================================================
# the section stands on its own
# ===========================================================================
def test_every_name_14_9_uses_is_imported_or_defined_inside_14_9():
    """The regression that produced `NameError: name 'np' is not defined`."""
    missing = undefined_names()

    assert missing == [], (
        "14.9 uses %d name(s) it never imports or defines: %s"
        % (len(missing), ", ".join("%s (cell %d, line %d)" % (name, cell, line)
                                   for cell, name, line in missing)))


def test_the_first_cell_of_14_9_is_the_setup_cell():
    cells = section_cells()

    assert cells[0].get("id") == SETUP_ID
    assert "import" in cells[0].source


@pytest.mark.parametrize("name", ["np", "pd", "config", "explainability_all",
                                  "xa", "Path", "os", "sys", "tf"])
def test_the_setup_cell_provides_what_the_section_needs(name):
    """Named one by one, so a failure says which import went missing."""
    setup = section_cells()[0]

    assert name in bound_names(ast.parse(setup.source)), name


def test_the_setup_cell_puts_the_project_root_on_the_import_path():
    """A kernel started in notebooks\\ must still find config.py and src\\."""
    source = section_cells()[0].source

    assert "sys.path" in source
    assert "config.py" in source                  # how the root is recognised
    assert "Path.cwd()" in source


def test_the_setup_cell_survives_a_terminal_without_ipython():
    """`from IPython.display import ...` must not stop a plain python run."""
    source = section_cells()[0].source

    assert "IPython.display" in source
    assert "except ImportError" in source


def test_no_cell_imports_numpy_or_pandas_under_another_name():
    """np and pd mean numpy and pandas everywhere in this notebook."""
    for number, cell in enumerate(section_cells(), start=1):
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname == "np":
                        assert alias.name == "numpy", "cell %d" % number
                    if alias.asname == "pd":
                        assert alias.name == "pandas", "cell %d" % number


def test_the_section_still_carries_no_stored_output():
    for number, cell in enumerate(section_cells(), start=1):
        assert not (cell.get("outputs") or []), "cell %d" % number
        assert cell.get("execution_count") is None, "cell %d" % number


def test_the_checker_itself_catches_the_bug_it_is_here_for():
    """A guard that cannot fail is not a guard.

    The two cells below are the shape 14.9 had when it raised
    `NameError: name 'np' is not defined`: a cell that uses numpy, in a
    section whose first cell never imported it.
    """
    missing = scan(["xa.show_chart(path)",
                    "values = agreement.to_numpy()\n"
                    "np.fill_diagonal(values, np.nan)\n"])

    assert ("np" in {name for _cell, name, _line in missing})
    assert ("agreement" in {name for _cell, name, _line in missing})

    # ... and it stays quiet once the imports are there
    fixed = scan(["import numpy as np\nfrom src.mscad import "
                  "explainability_all as xa",
                  "agreement = xa.load()\n"
                  "values = agreement.to_numpy()\n"
                  "np.fill_diagonal(values, np.nan)\n"])

    assert fixed == []


def test_a_name_bound_inside_its_own_statement_is_not_reported():
    """`for row in rows:` must not report `row`, and a comprehension's
    variable is not a missing import either."""
    assert scan(["rows = [1, 2]\nfor row in rows:\n    print(row)\n",
                 "doubled = [value * 2 for value in rows]\n"
                 "print(doubled)\n"]) == []
