r"""
src/mscad/training_lock.py -- the retraining lock for Steps 10 and 10.2.

NEW FILE.  `config.LOCK_TRAINED_MODELS = True` means: a model whose
`metrics.json` already exists in the ACTIVE output root is never retrained,
whatever its `OVERWRITE_*` flag says.  The flag and the lock answer different
questions -- `OVERWRITE_STEP10` says "recompute if asked", the lock says "this
model has published numbers beside it, so do not touch it by accident".

It is a WRAPPER, not an edit: no training logic changes.  `install()` replaces
the training entry points with guarded versions that refuse first and delegate
second, and `uninstall()` puts the originals back.  The functions themselves
are untouched and still honour their own flags once the lock lets them through.

Which models, and where their metrics live:

    approach1   Step 10  trains it, Step 11  scores it   -> step11/metrics.json
    approach2   10.2     trains it, 11.x    scores it    -> step11_attention/

Step 11's own runner is deliberately NOT locked: it evaluates a model rather
than training one, and re-scoring a saved model is how you check it.  Step 12's
configurations have their own lock in `step12_checks` and in
`experiments_step12/common.py`; this module does not duplicate it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

OK, WARN = "[OK]", "[WARN]"


class TrainingLocked(Exception):
    """Raised instead of retraining a model that already has saved metrics."""


# name -> (what it is, the folder its metrics live in, the file, the module and
#          function that trains it, what to re-run afterwards)
TARGETS: dict[str, dict[str, Any]] = {
    "approach1": {
        "label": "Approach 1 (CNN-LSTM)",
        "metrics_dir": "STEP11_OUTPUT_DIR",
        "metrics_file": "metrics",
        "metrics_files_block": "STEP11_FILES",
        "module": "models",
        "function": "run_step10",
        "rerun": "Steps 11 to 14, so every table that quotes it agrees again",
    },
    "approach2": {
        "label": "Approach 2 (CNN-Attention-LSTM)",
        "metrics_dir": "STEP11B_OUTPUT_DIR",
        "metrics_file": "metrics",
        "metrics_files_block": "STEP11B_FILES",
        "module": "models_attention",
        "function": "run_step10b",
        "rerun": "sections 11.x to 14, so every table that quotes it agrees "
                 "again",
    },
}


def metrics_path(name: str) -> Path | None:
    """Where this model's metrics.json lives in the ACTIVE output root."""
    if name not in TARGETS:
        raise KeyError("%r is not a locked model; known: %s"
                       % (name, ", ".join(TARGETS)))
    target = TARGETS[name]
    folder = config.usable_output_dir(getattr(config, target["metrics_dir"]))
    if folder is None:
        return None
    names = getattr(config, target["metrics_files_block"], {})
    return Path(folder) / names.get(target["metrics_file"], "metrics.json")


def is_trained(name: str) -> bool:
    """True when this model already has saved metrics in the active root."""
    path = metrics_path(name)
    return bool(path is not None and path.exists())


def lock_is_on() -> bool:
    return bool(getattr(config, "LOCK_TRAINED_MODELS", False))


def locked(name: str) -> bool:
    return bool(lock_is_on() and is_trained(name))


def lock_message(name: str) -> str:
    target = TARGETS[name]
    return (
        "LOCK_TRAINED_MODELS is True and %s already has saved metrics in the "
        "active output root:\n"
        "    %s\n"
        "Dataset: %s (%s)\n"
        "Training it again would change the model that those numbers, the "
        "Step 12 comparison, the Step 13 fusion and the Step 14 explanations "
        "were all computed from, and the numbers beside it would no longer "
        "describe it. Its OVERWRITE_* flag does not override this.\n"
        "To retrain on purpose: set LOCK_TRAINED_MODELS = False in config.py, "
        "run this step again, and then re-run %s."
        % (target["label"], metrics_path(name), config.DATASET_SOURCE,
           config.OUTPUTS_ROOT, target["rerun"]))


def guard(name: str, trainer: Callable[..., Any], *args: Any,
          **kwargs: Any) -> Any:
    """Call `trainer` only if the lock allows it."""
    if locked(name):
        raise TrainingLocked(lock_message(name))
    return trainer(*args, **kwargs)


def lock_table() -> pd.DataFrame:
    """What the lock would do to each model, right now, in this output root."""
    rows = []
    for name, target in TARGETS.items():
        path = metrics_path(name)
        rows.append({
            "model": target["label"],
            "trained by": "%s.%s()" % (target["module"], target["function"]),
            "metrics.json": "" if path is None else str(path),
            "saved metrics exist": is_trained(name),
            "training now would": ("be REFUSED" if locked(name)
                                   else "run (no saved metrics yet)"
                                   if not is_trained(name)
                                   else "run (LOCK_TRAINED_MODELS is False)"),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# install / uninstall -- the wrapper, so a direct call is guarded too
# ===========================================================================
_INSTALLED: dict[str, Callable[..., Any]] = {}


def install(verbose: bool = True) -> dict[str, str]:
    """Wrap the training entry points so even a direct call refuses.

    Idempotent, and reversible with `uninstall()`.  The wrapper keeps the
    original on `__wrapped__`, so the function that actually trains is always
    one attribute away and nothing about it is rewritten.
    """
    import importlib

    installed = {}
    for name, target in TARGETS.items():
        module = importlib.import_module("%s.%s" % (__package__,
                                                    target["module"]))
        function_name = target["function"]
        original = getattr(module, function_name)
        if getattr(original, "_mscad_locked", False):
            installed[name] = "already wrapped"
            continue

        def make(name=name, original=original):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if locked(name):
                    raise TrainingLocked(lock_message(name))
                return original(*args, **kwargs)
            wrapper.__name__ = getattr(original, "__name__", "wrapper")
            wrapper.__doc__ = getattr(original, "__doc__", None)
            wrapper.__wrapped__ = original
            wrapper._mscad_locked = True
            return wrapper

        _INSTALLED.setdefault("%s.%s" % (target["module"], function_name),
                              original)
        setattr(module, function_name, make())
        installed[name] = "%s.%s wrapped" % (target["module"], function_name)
    if verbose:
        print("  [LOCK]  LOCK_TRAINED_MODELS = %s; %d training function(s) "
              "wrapped (%s)"
              % (lock_is_on(), len(installed), ", ".join(TARGETS)))
    return installed


def uninstall(verbose: bool = True) -> int:
    """Put the original training functions back."""
    import importlib

    count = 0
    for key, original in list(_INSTALLED.items()):
        module_name, function_name = key.rsplit(".", 1)
        module = importlib.import_module("%s.%s" % (__package__, module_name))
        setattr(module, function_name, original)
        _INSTALLED.pop(key, None)
        count += 1
    if verbose:
        print("  [LOCK]  %d training function(s) restored" % count)
    return count
