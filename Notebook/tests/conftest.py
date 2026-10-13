"""
tests/conftest.py -- fixtures shared by every test module.

pytest imports this automatically, so the fixtures below are available by name
in any test file without an import.
"""

import os
import re

import pytest

# "C:", "D:" and friends -- a Windows drive prefix at the start of a NAME.
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


@pytest.fixture
def assert_no_leaked_output(tmp_path):
    """Assert that a guarded write created nothing in the current directory.

    The failure this guards against: on Linux, Path(r"C:\\MSCAD2022") is a
    RELATIVE path, so an unguarded mkdir() creates a folder literally named
    "C:\\MSCAD2022" wherever the process happens to be.  That is how such a
    folder once ended up committed and broke `git pull` on Windows with
    "invalid path".  Every test that exercises the guard chdirs into tmp_path
    first, so a mistake lands there and nowhere else.

    Why this is a fixture and not `assert not (tmp_path / "C:").exists()`:
    that expression is meaningless on Windows.  Joining a bare drive letter
    resets the anchor, so on a C: temp drive `tmp_path / "C:"` IS tmp_path --
    which of course exists, and the assertion failed on every Windows run.  On
    any other drive it becomes the drive-relative path "C:", which is the
    current directory on C: and has nothing to do with the test.  Listing the
    directory and looking at the NAMES works the same way on both operating
    systems.

    One limitation worth naming: on Windows the guarded path really is
    absolute, so a regression there would create C:\\MSCAD2022\\... rather than
    anything under tmp_path.  This fixture cannot catch that, and asserting
    that C:\\MSCAD2022 does not exist would fail on the machine that holds the
    real dataset.  What the tests check there is the return value -- None, or
    status "skipped_not_windows" -- which is the guard refusing.
    """
    def check(what: str = "a guarded write") -> None:
        entries = sorted(os.listdir(tmp_path))
        drive_like = [name for name in entries if _DRIVE_PREFIX.match(name)]
        assert not drive_like, (
            "%s created a folder named after a Windows path: %s in %s"
            % (what, drive_like, tmp_path))
        assert entries == [], (
            "%s created %s in %s; it should have created nothing"
            % (what, entries, tmp_path))
    return check
