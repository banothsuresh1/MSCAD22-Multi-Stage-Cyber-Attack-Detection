r"""
src/mscad/experiments_step12_ae30/common_ae30.py -- the AE=30 run's plumbing.

NEW FILE.  No training code is written in this folder: every wrapper imports
the existing trainer and calls it.  What this module provides is the one thing
those trainers cannot be asked for by argument -- WHERE they write -- and the
lock that decides whether they may run at all.

Why a context manager
---------------------
Every trainer resolves its output folder from `config` when it is CALLED
(`experiments.experiment_folder` reads `config.STEP12_OUTPUT_DIR`,
`autoencoder.save_step7` reads `config.STEP7_OUTPUT_DIR`, and so on).  So the
AE=30 run does not need copies of those functions: it needs config to say
`outputs_official_ae30\` for the duration of one call.  `ae30()` sets exactly
the attributes that decide that, and restores every one of them afterwards --
including on an exception, so a crashed configuration cannot leave the next
cell writing into the wrong root.

What is redirected, and what is NOT
-----------------------------------
Redirected (Steps 7 to 12): the autoencoder, the encoded features, the class
weights, the classifier, its evaluation, and the Step 12 folders.  Those are
all functions of the bottleneck width.

NOT redirected (Steps 5 and 6): the train/test split and the selected
features.  Those are properties of the DATASET, identical for both runs, and
reading them from the official root is what makes the two tables comparable --
a second split would change more than the bottleneck.

Nothing under `outputs_official\` is opened for writing while this is active:
the only paths the trainers can reach for output are under
`config.OUTPUTS_ROOT_AE30`.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m src.mscad.experiments_step12_ae30.baseline1_ae30
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from .. import comparison as cmp  # noqa: E402
from .. import data_loading as dl  # noqa: E402
from ..experiments_step12 import common as base  # noqa: E402

Step12ConfigError = base.Step12ConfigError
OK, FAIL, SKIP = base.OK, base.FAIL, base.SKIP
LOADED, TRAINED, PENDING = base.LOADED, base.TRAINED, base.PENDING

# the eight configurations, in the display order the AE=15 table uses
KEYS = [key for key, _label in config.STEP12_DISPLAY_ORDER]


class Ae30Error(Exception):
    """A problem with the AE=30 run itself, not with a trainer."""


# ===========================================================================
# 1. WHERE THIS RUN WRITES
# ===========================================================================
def root() -> Path | None:
    """The AE=30 output root, or None when it cannot be written here."""
    return config.usable_output_dir(config.OUTPUTS_ROOT_AE30)


def latent_dim() -> int:
    """The bottleneck width this run asks every autoencoder for."""
    return int(config.AE_LATENT_DIM_AE30)


def metrics_path(key: str) -> Path | None:
    """Where THIS configuration's result lands inside the AE=30 root.

    Resolved by `comparison.result_path` from inside the redirect, which is
    the point: the existing trainers each write to their own folder
    (`step12\\<key>\\`, `step12_baseline6\\`, `step11\\`,
    `step11_attention\\`), and the existing table builder reads exactly those
    paths.  Inventing a flat `<name>\\metrics.json` layout here would mean
    either rewriting the trainers or teaching the builder a second layout, and
    the whole point of this folder is that neither happens.  So the AE=30 root
    mirrors the AE=15 root's layout, one folder per configuration, under a
    different root.
    """
    if root() is None:
        return None
    with ae30():
        path = cmp.result_path(key)
    return None if path is None else Path(path)


def folder(key: str) -> Path | None:
    """The folder that holds this configuration's AE=30 result."""
    path = metrics_path(key)
    return None if path is None else path.parent


def is_finished(key: str) -> bool:
    path = metrics_path(key)
    return bool(path is not None and path.exists())


def lock_is_on() -> bool:
    """This root's own lock, never the AE=15 one."""
    return bool(getattr(config, "LOCK_TRAINED_MODELS_AE30", True))


def display_label(key: str) -> str:
    return base.display_label(key)


# ===========================================================================
# 2. THE REDIRECT
# ===========================================================================
# Every attribute here is one a trainer reads at call time to decide where to
# write.  Steps 5 and 6 are deliberately absent: see the module docstring.
def _redirected(place: Path) -> dict[str, Any]:
    """{config attribute: value} for one AE=30 call."""
    out: dict[str, Any] = {
        "OUTPUTS_ROOT": place,
        "OUTPUTS_DIR": place,
        "AE_LATENT_DIM": latent_dim(),
        # the trainers consult the MAIN lock; inside the redirect it has to be
        # this root's lock, or an empty AE=30 root would be refused because the
        # AE=15 root is finished
        "LOCK_TRAINED_MODELS": lock_is_on(),
    }
    for number in config.AE30_REDIRECTED_STEPS:
        out["STEP%d_OUTPUT_DIR" % number] = place / ("step%d" % number)
    out["STEP10B_OUTPUT_DIR"] = place / "step10_attention"
    out["STEP11B_OUTPUT_DIR"] = place / "step11_attention"
    out["STEP12_CONFIG_DIR"] = place / "step12"
    out["STEP12_B6_OUTPUT_DIR"] = place / "step12_baseline6"
    out["BASELINE5_OUTPUT_DIR"] = place / "step12" / "baseline5"
    out["STEP12_THESIS_DIR"] = place / "step12" / "thesis"
    return out


@contextlib.contextmanager
def ae30() -> Iterator[Path]:
    """Point config at the AE=30 root for the duration of one call.

    Restores every attribute afterwards, exception or not -- a crashed
    configuration must not leave the next one writing into the wrong root.
    """
    place = root()
    if place is None:
        raise Ae30Error(
            "%s cannot be written on this machine, so nothing was trained.\n"
            "    %s" % (config.OUTPUTS_ROOT_AE30,
                        config.why_unusable(config.OUTPUTS_ROOT_AE30)))
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


def inside_ae30(path: Path | str | None) -> bool:
    """True only when `path` sits under the AE=30 root (compared by parts)."""
    if path is None:
        return False
    place = root()
    if place is None:
        return False
    parts, root_parts = Path(path).parts, Path(place).parts
    return bool(root_parts) and parts[:len(root_parts)] == root_parts


# ===========================================================================
# 3. WHAT THE TRAINERS NEED, read from the DATASET's root
# ===========================================================================
def inputs(verbose: bool = False) -> dict[str, Any]:
    """y_train, y_test, meta_test, label_mapping, duplicate_flags.

    Read OUTSIDE the redirect, from the active (AE=15) root, through
    `experiments_step12.common.evaluation_inputs` -- the same function the
    AE=15 run uses, so there is no second copy of this plumbing.

    These are the labels, the test set and the duplicate mask: properties of
    the dataset, identical whatever the bottleneck is.  Reading them from the
    AE=30 root instead would mean the two tables were scored on different
    rows, and the comparison would answer nothing.
    """
    return base.evaluation_inputs(verbose=verbose)


def prepare_latent_features(verbose: bool = True) -> dict[str, Any]:
    """Steps 7, 8 and 9 at this run's latent width, into the AE=30 root.

    The autoencoder, the encoded train/test features and the class weights.
    Approach 1, Approach 2 and Baseline 6 all read Step 8's features rather
    than taking them as arguments, so they share this one chain -- the encoder
    is trained once, not three times.  Each runner is the existing one, and
    each is skipped when its outputs are already there.

    Step 6's selected features are read from the ACTIVE root: the selection is
    a property of the dataset, not of the bottleneck.
    """
    from .. import autoencoder as ae
    from .. import imbalance as im

    out: dict[str, Any] = {}
    with ae30():
        if verbose:
            dl.show_note(
                "Steps 7-9 at latent **%d**, into `%s`. Step 6's selected "
                "features are read from `%s`, because the selection is a "
                "property of the dataset and not of the bottleneck."
                % (latent_dim(), config.STEP7_OUTPUT_DIR,
                   config.STEP6_OUTPUT_DIR))
        out["step7"] = ae.run_step7(verbose=verbose)
        out["step8"] = ae.run_step8(verbose=verbose)
        out["step9"] = im.run_step9(verbose=verbose)
    return out


def train_classifier(verbose: bool = True) -> dict[str, Any]:
    """Step 10's CNN-LSTM on the AE=30 features, through `models.run_step10`."""
    from .. import models as md

    with ae30():
        return md.run_step10(verbose=verbose)


# ===========================================================================
# 3. THE STANDARD ENTRY POINT, mirroring experiments_step12.common.run_one
# ===========================================================================
def run_one_ae30(key: str, trainer: Callable[..., Any],
                 components: dict[str, Any] | None = None,
                 note: str | None = None,
                 verbose: bool = True) -> dict[str, Any]:
    """Load the AE=30 result if it exists, else train it inside the redirect.

    `trainer` is called with no arguments, from inside `ae30()`, and is always
    an existing training function imported by the wrapper -- never a copy.
    """
    label = display_label(key)
    if verbose:
        dl.show_title("%s  (AE=%d)" % (label, latent_dim()), level=4)

    path = metrics_path(key)
    if path is None:
        raise Ae30Error(
            "%s cannot be read on this machine, so nothing was loaded and "
            "nothing was trained.\n    %s"
            % (label, config.why_unusable(config.OUTPUTS_ROOT_AE30)))

    if path.exists():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        status, detail = LOADED, "loaded from %s" % path
    elif lock_is_on():
        return {
            "key": key, "label": label, "status": PENDING, "pending": True,
            "status_detail": "not trained at AE=%d (LOCK_TRAINED_MODELS_AE30 "
                             "is True)" % latent_dim(),
            "metrics": {}, "metrics_path": str(path), "ok": False,
            "latent_dim": latent_dim(), "note": note,
        }
    else:
        with ae30():
            out = trainer() or {}
        metrics = out.get("metrics")
        if metrics is None and path.exists():      # the trainer wrote the file
            metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics is None:
            raise Ae30Error(
                "%s finished but no metrics.json is readable at %s"
                % (label, path))
        if not path.exists():                      # write what the row reads
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metrics, indent=2, default=str),
                            encoding="utf-8")
        status, detail = TRAINED, "trained at AE=%d" % latent_dim()

    result = {
        "key": key, "label": label, "status": status, "pending": False,
        "status_detail": detail, "metrics": metrics, "metrics_path": str(path),
        "ok": True, "latent_dim": latent_dim(), "note": note,
    }
    if verbose:
        print("  [%s] %s: %s" % ("LOAD" if status == LOADED else "TRAIN",
                                 label, detail))
        if note:
            dl.show_note(note)
        dl.show_table(base.headline(metrics), note="%s -- %s" % (label, detail))
    return result


# ===========================================================================
# 4. THE TABLE, in the same format as 10.x
# ===========================================================================
def comparison_table(order: Sequence[tuple[str, str]] | None = None
                     ) -> pd.DataFrame:
    """The AE=30 table, built by the SAME builder the AE=15 table uses.

    `comparison.build_full_comparison` reads one metrics.json per key from the
    active root, so it is run inside the redirect -- the rows come from
    outputs_official_ae30\\, never from outputs_official\\.
    """
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    with ae30():
        built = cmp.build_full_comparison(order=order, verbose=False)
    return built["table"]


def marked_comparison_table(order: Sequence[tuple[str, str]] | None = None
                            ) -> pd.DataFrame:
    with ae30():
        built = cmp.build_full_comparison(order=order, verbose=False)
        return cmp.mark_best(built["table"])


def provenance(order: Sequence[tuple[str, str]] | None = None
               ) -> pd.DataFrame:
    """Every row, its state, and the exact AE=30 file it was read from."""
    order = config.STEP12_DISPLAY_ORDER if order is None else order
    rows = []
    for key, label in order:
        path = metrics_path(key)
        rows.append({
            "config": label,
            "state": ("read from disk" if is_finished(key)
                      else "not trained at AE=%d" % latent_dim()),
            "metrics.json": "(unreachable here)" if path is None else str(path),
            "file exists": "yes" if is_finished(key) else "no",
            "under the AE=30 root": "yes" if inside_ae30(path) else "no",
        })
    return pd.DataFrame(rows)
