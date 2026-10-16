"""
tests/conftest.py -- fixtures shared by every test module.

pytest imports this automatically, so the fixtures below are available by name
in any test file without an import.
"""

import os
import re

import pytest

# ---------------------------------------------------------------------------
# The test suite runs in "own" mode unless a test says otherwise.
# ---------------------------------------------------------------------------
# config.DATASET_SOURCE now defaults to "official", which is right for the
# notebook: the official MSCAD.csv is the primary dataset.  Every test written
# before that switch existed, though, builds an OWN-extraction frame -- with
# Src IP, Timestamp and Source_Capture columns -- and asserts own-extraction
# behaviour (BENIGN_LABELS = ["Benign", "Normal"], Protocol shared into the
# metadata).  Running those tests under the official overrides would test
# nothing they were written to test.
#
# Setting it HERE works because pytest imports conftest.py before any test
# module, so `import config` inside a test module reads this value.  It is
# `setdefault`, so `MSCAD_DATASET_SOURCE=official python -m pytest tests` still
# runs the whole suite the other way round when that is what you want.
#
# The effect is a guard rather than a convenience: these 1,000+ tests now also
# prove that "own" mode is unchanged by the official-dataset work.  Official
# mode has its own tests -- tests/test_official_dataset.py and
# tests/test_dataset_source.py -- which set the mode explicitly.
os.environ.setdefault("MSCAD_DATASET_SOURCE", "own")

# A Windows drive prefix at the start of a NAME: "C:", "D:", ...
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def looks_like_windows_path(name: str) -> bool:
    """True when a directory entry's NAME is really a Windows path.

    This is what an unguarded write leaves behind on Linux.  There,
    Path(r"C:\\MSCAD2022") is a RELATIVE path whose single part is the literal
    text "C:\\MSCAD2022", so

        (Path(r"C:\\MSCAD2022") / "outputs" / "step3").mkdir(parents=True)

    creates a directory literally named  C:\\MSCAD2022  in the current folder,
    with outputs/step3 inside it.  That is exactly how such a folder once ended
    up committed and broke `git pull` on Windows with "invalid path".

    Both marks are checked: a leading drive letter, and a backslash anywhere in
    the name (which catches a drive-less Windows path such as
    "MSCAD2022\\outputs").  Neither can occur in a legitimate name on Linux, and
    Windows forbids both in a filename, so this never fires on a real file.

    It works on NAMES rather than on paths on purpose -- see
    assert_no_leaked_output for why a path-based check cannot work on Windows.
    """
    return bool(_DRIVE_PREFIX.match(name)) or "\\" in name


@pytest.fixture
def assert_no_leaked_output(tmp_path):
    """Assert that a guarded write created nothing in the current directory.

    Every test that exercises the Windows-path guard chdirs into tmp_path
    first, so a mistake lands there and nowhere else.

    Why this is a fixture and not `assert not (tmp_path / "C:").exists()`:
    that expression is meaningless on Windows.  Joining a bare drive letter
    resets the anchor, so on a C: temp drive `tmp_path / "C:"` IS tmp_path --
    which of course exists, and the assertion failed on every Windows run.  On
    any other drive it becomes the drive-relative path "C:", which is the
    current directory on C: and has nothing to do with the test.  Listing the
    directory and looking at the NAMES works the same way on both operating
    systems, so this fixture runs everywhere -- nothing in it is skipped on
    Windows.

    One limitation worth naming: on Windows the guarded path really is
    absolute, so a regression there would create C:\\MSCAD2022\\... rather than
    anything under tmp_path.  This fixture cannot catch that, and asserting
    that C:\\MSCAD2022 does not exist would fail on the machine that holds the
    real dataset.  What the tests check there is the return value -- None, or
    status "skipped_not_windows" -- which is the guard refusing.
    """
    def check(what: str = "a guarded write") -> None:
        entries = sorted(os.listdir(tmp_path))
        leaked = [name for name in entries if looks_like_windows_path(name)]
        assert not leaked, (
            "%s created a folder named after a Windows path: %s in %s"
            % (what, leaked, tmp_path))
        assert entries == [], (
            "%s created %s in %s; it should have created nothing"
            % (what, entries, tmp_path))
    return check
