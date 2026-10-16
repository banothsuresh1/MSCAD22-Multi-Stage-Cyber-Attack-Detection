r"""
tests/step12/ -- the tests for the eight Step 12 configuration files.

This is a PACKAGE on purpose.  Without an `__init__.py`, pytest adds this
directory to sys.path and imports its `conftest.py` under the bare module name
`conftest`, which collides with `tests/conftest.py` -- whichever is imported
first wins, and `tests/test_data_loading.py`'s
`from conftest import looks_like_windows_path` then fails.  As a package, this
directory's conftest is `step12.conftest` and nothing is shadowed.
"""
