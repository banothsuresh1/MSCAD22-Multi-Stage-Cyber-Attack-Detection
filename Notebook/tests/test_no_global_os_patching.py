r"""
tests/test_no_global_os_patching.py -- a guard against ONE specific regression.

NEW FILE.  It adds no coverage of the pipeline; it watches the other test
files.

What happened
-------------
Several path-safety tests used to simulate Linux with

    monkeypatch.setattr(config.os, "name", "posix")

`config.os` is the `os` module itself, so that sets a PROCESS-GLOBAL attribute.
`pathlib.Path(...)` reads `os.name` at instantiation time to decide whether to
build a `WindowsPath` or a `PosixPath`, so for the duration of such a test every
`Path(...)` call anywhere in the process -- in the code under test, and in
pytest's own repr_failure, cache and tmpdir machinery -- built the foreign class.

On Python >= 3.12 that merely produces an odd object.  On Python < 3.12 the
foreign class cannot be instantiated at all, and Windows + Python 3.10 answered
with

    NotImplementedError: cannot instantiate 'PosixPath' on your system

from inside pytest, which ended the whole run with INTERNALERROR and lost every
later test.

The fix was to simulate the other operating system through the hook the guard
actually asks:

    monkeypatch.setattr(config, "running_on_windows", lambda: False)

That is a plain function on `config`, invisible to pathlib and to pytest, and
`config.is_windows_absolute` decides with `PureWindowsPath(...).drive`, which
works identically on every operating system.

This module fails if the old pattern comes back.
"""

import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SELF = Path(__file__).resolve().name

# (pattern, what to do instead)
FORBIDDEN = [
    (re.compile(r"setattr\(\s*(config\.)?os\s*,\s*[\"']name[\"']"),
     "patches the process-global os.name, which steers pathlib.Path -- use "
     "monkeypatch.setattr(config, \"running_on_windows\", lambda: False)"),
    (re.compile(r"setattr\(\s*(config\.)?sys\s*,\s*[\"']platform[\"']"),
     "patches sys.platform -- use config.running_on_windows, or "
     "@pytest.mark.skipif(sys.platform == \"win32\", ...) for a test that "
     "genuinely cannot run there"),
    (re.compile(r"setattr\(\s*pathlib\b"),
     "patches the pathlib module, which pytest itself uses"),
    (re.compile(r"setattr\([^,]+,\s*[\"']Path[\"']"),
     "replaces a module's Path -- build the path with PureWindowsPath or "
     "PurePosixPath instead of swapping the class"),
    (re.compile(r"pathlib\.Path\s*="),
     "rebinds pathlib.Path"),
    (re.compile(r"(?<!Pure)\bPosixPath\s*\("),
     "instantiates PosixPath, which raises NotImplementedError on Windows "
     "before Python 3.12 -- use PurePosixPath"),
    (re.compile(r"(?<!Pure)\bWindowsPath\s*\("),
     "instantiates WindowsPath, which raises NotImplementedError off Windows "
     "before Python 3.12 -- use PureWindowsPath"),
]


def _modules_to_scan():
    """Every Python file in tests/, conftest.py included, except this one."""
    return sorted(p for p in TESTS_DIR.glob("*.py") if p.name != SELF)


@pytest.mark.parametrize("module", _modules_to_scan(), ids=lambda p: p.name)
def test_no_test_module_patches_global_os_or_path_state(module):
    """Every test file is read as text and checked line by line."""
    offences = []
    for number, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue                      # a comment explaining the rule is fine
        for pattern, why in FORBIDDEN:
            if pattern.search(line):
                offences.append("%s:%d  %s\n    %s"
                                % (module.name, number, line.strip(), why))
    assert not offences, (
        "these lines would break the suite on Windows with Python < 3.12:\n\n"
        + "\n".join(offences))


def test_the_scanner_actually_catches_the_old_pattern(tmp_path):
    """A guard that cannot fail is not a guard."""
    offending = [
        'monkeypatch.setattr(config.os, "name", "posix")',
        'monkeypatch.setattr(os, "name", "nt")',
        'monkeypatch.setattr(config, "Path", PosixPath)',
        'folder = PosixPath("/tmp/x")',
        'folder = WindowsPath(r"C:\\x")',
        'monkeypatch.setattr(sys, "platform", "linux")',
    ]
    for line in offending:
        assert any(pattern.search(line) for pattern, _why in FORBIDDEN), line

    allowed = [
        'monkeypatch.setattr(config, "running_on_windows", lambda: False)',
        'monkeypatch.setattr(config, "STEP3_OUTPUT_DIR", PureWindowsPath(r"C:\\x"))',
        'assert PurePosixPath("/tmp/x").name == "x"',
        '@pytest.mark.skipif(sys.platform == "win32", reason="...")',
        'monkeypatch.setattr(config, "STEP4_OUTPUT_DIR", tmp_path / "step4")',
    ]
    for line in allowed:
        assert not any(pattern.search(line) for pattern, _why in FORBIDDEN), line


def test_the_running_on_windows_hook_is_what_the_guard_asks():
    """The replacement has to be sufficient, not merely harmless.

    `usable_output_dir` asks `running_on_windows()` and `is_windows_absolute()`,
    and the second uses PureWindowsPath, so patching the hook alone is enough to
    simulate either operating system -- on either operating system.
    """
    import config

    original = config.running_on_windows
    try:
        config.running_on_windows = lambda: False
        assert config.usable_output_dir(r"C:\MSCAD2022\outputs") is None
        config.running_on_windows = lambda: True
        assert config.usable_output_dir(r"C:\MSCAD2022\outputs") is not None
    finally:
        config.running_on_windows = original

    # and the detection underneath it never needs a concrete path class
    assert config.is_windows_absolute(r"C:\MSCAD2022") is True
    assert config.is_windows_absolute("/tmp/mscad") is False
