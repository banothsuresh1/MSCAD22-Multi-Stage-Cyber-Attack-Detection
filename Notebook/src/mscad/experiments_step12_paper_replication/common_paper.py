r"""
src/mscad/experiments_step12_paper_replication/common_paper.py -- plumbing for
the AE+PCA+LSTM paper replication.

NEW FILE.  It mirrors experiments_step12_ae30/common_ae30.py: a dedicated
output root, this variant's own lock, a context-manager redirect, and a
`run_one_*` entry point.  What it does NOT carry is any hyperparameter -- those
are the paper's and live in baseline_ae64_pca30_lstm.py -- so this variant
cannot read or move the AE=15 / AE=30 pipelines' config state.

Only the OUTPUT LOCATION is redirected (Steps 5 and 6 are read from the active
official root, because the split and the selected features are properties of
the dataset, and the replication reuses the same test set so the comparison
against B4 is on the same rows).  Nothing under outputs_official\ or
outputs_official_ae30\ is opened for writing while the redirect is active.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m src.mscad.experiments_step12_paper_replication.baseline_ae64_pca30_lstm
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from .. import data_loading as dl  # noqa: E402
from ..experiments_step12 import common as base  # noqa: E402

Step12ConfigError = base.Step12ConfigError
LOADED, TRAINED, PENDING = base.LOADED, base.TRAINED, base.PENDING


class PaperReplicationError(Exception):
    """A problem with the replication run itself, not with a trainer."""


def root() -> Path | None:
    """The replication output root, or None when it cannot be written here."""
    return config.usable_output_dir(config.OUTPUTS_ROOT_PAPER)


def lock_is_on() -> bool:
    """This root's own lock, never the official or AE=30 one."""
    return bool(getattr(config, "LOCK_TRAINED_MODELS_PAPER", True))


def folder(name: str) -> Path | None:
    """outputs_official_paper_replication\\<name>\\ -- one folder per config."""
    place = root()
    return None if place is None else place / name


def metrics_path(name: str) -> Path | None:
    place = folder(name)
    return None if place is None else place / "metrics.json"


def is_finished(name: str) -> bool:
    path = metrics_path(name)
    return bool(path is not None and path.exists())


# ===========================================================================
# the redirect -- only the output location, never a hyperparameter
# ===========================================================================
def _redirected(place: Path) -> dict[str, Any]:
    """{config attribute: value} for one replication call.

    Only output folders and this variant's lock.  The AE width, the PCA count,
    the LSTM shape and the epoch budget are the paper's, passed straight to the
    builders in baseline_ae64_pca30_lstm.py, so none of the shared AE_* or
    classifier constants appear here.
    """
    # ONLY the root and this variant's lock.  The replication READS its inputs
    # (Steps 4, 5, 8: the labels, the scaled features, the test set) from the
    # active official root -- the dataset is the same, and the comparison
    # against B4 must be on the same rows -- so the STEP*_OUTPUT_DIR constants
    # are deliberately NOT redirected.  The replication WRITES only into its
    # own per-config folder (common_paper.folder), which is under
    # OUTPUTS_ROOT_PAPER already, so nothing lands on outputs_official\ or
    # outputs_official_ae30\ either way.
    return {
        "OUTPUTS_ROOT": place,
        "OUTPUTS_DIR": place,
        "LOCK_TRAINED_MODELS": lock_is_on(),
    }


@contextlib.contextmanager
def paper_root() -> Iterator[Path]:
    """Point config's output location at the replication root for one call.

    Restores every attribute afterwards, exception or not.
    """
    place = root()
    if place is None:
        raise PaperReplicationError(
            "%s cannot be written on this machine, so nothing was trained.\n"
            "    %s" % (config.OUTPUTS_ROOT_PAPER,
                        config.why_unusable(config.OUTPUTS_ROOT_PAPER)))
    wanted = _redirected(Path(place))
    missing = object()
    before = {name: getattr(config, name, missing) for name in wanted}
    for name, value in wanted.items():
        setattr(config, name, value)
    try:
        Path(place).mkdir(parents=True, exist_ok=True)
        yield Path(place)
    finally:
        for name, value in before.items():
            if value is missing:                   # pragma: no cover
                delattr(config, name)
            else:
                setattr(config, name, value)


def inside_paper_root(path: Path | str | None) -> bool:
    """True only when `path` sits under the replication root (by parts)."""
    if path is None:
        return False
    place = root()
    if place is None:
        return False
    parts, root_parts = Path(path).parts, Path(place).parts
    return bool(root_parts) and parts[:len(root_parts)] == root_parts


def inputs(verbose: bool = False) -> dict[str, Any]:
    """y_train, y_test, meta_test, label_mapping, duplicate_flags.

    From the ACTIVE official root, through the same
    experiments_step12.common.evaluation_inputs the other runs use -- the test
    set and the duplicate mask are the dataset's, so the replication is scored
    on the same rows as B4.
    """
    return base.evaluation_inputs(verbose=verbose)


# ===========================================================================
# the entry point, mirroring run_one / run_one_ae30
# ===========================================================================
def run_one_paper(name: str, trainer: Callable[..., Any],
                  verbose: bool = True) -> dict[str, Any]:
    """Load the replication result if it exists, else train it in the redirect.

    `trainer` is called with no arguments, from inside `paper_root()`.
    """
    path = metrics_path(name)
    if verbose:
        dl.show_title("Paper replication: %s" % name, level=4)
    if path is None:
        raise PaperReplicationError(
            "%s cannot be read on this machine, so nothing was loaded and "
            "nothing was trained.\n    %s"
            % (name, config.why_unusable(config.OUTPUTS_ROOT_PAPER)))

    if path.exists():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        status, detail = LOADED, "loaded from %s" % path
    elif lock_is_on():
        return {
            "name": name, "status": PENDING, "pending": True,
            "status_detail": "not trained (LOCK_TRAINED_MODELS_PAPER is True)",
            "metrics": {}, "metrics_path": str(path), "ok": False,
        }
    else:
        with paper_root():
            out = trainer() or {}
        metrics = out.get("metrics")
        if metrics is None:
            raise PaperReplicationError(
                "%s finished but returned no metrics" % name)
        status, detail = TRAINED, "trained into %s" % path.parent

    result = {"name": name, "status": status, "pending": False,
              "status_detail": detail, "metrics": metrics,
              "metrics_path": str(path), "ok": True}
    if verbose:
        print("  [%s] %s: %s"
              % ("LOAD" if status == LOADED else "TRAIN", name, detail))
        dl.show_table(base.headline(metrics),
                      note="%s -- %s" % (name, detail))
    return result
