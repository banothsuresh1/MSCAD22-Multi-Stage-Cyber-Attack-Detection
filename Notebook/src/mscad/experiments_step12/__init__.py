r"""
src/mscad/experiments_step12/ -- Step 12 as eight separate, individually
runnable configuration files.

NEW PACKAGE.  Nothing outside this folder is modified.  Every file here is a
thin wrapper: it imports and calls the EXISTING function for its
configuration, so there is exactly one implementation of each model in the
project and this folder adds none.

    configuration   what it is                        the code it calls
    --------------- --------------------------------- -----------------------
    approach1       FS -> AE -> CNN-LSTM              models.py / evaluation.py
    approach2       FS -> AE -> CNN-Attention-LSTM    models_attention.py
    baseline1       AE -> CNN-LSTM (no selection)     experiments.py
    baseline2       AE -> LSTM (no CNN)               experiments.py
    baseline3       CNN-LSTM (no AE)                  experiments.py
    baseline4       AE -> PCA -> LSTM                 experiments_pca.py
    baseline5       MI-only top 30 -> AE -> CNN-LSTM  experiments_baseline5.py
    baseline6       AE -> PCA -> CNN-LSTM             experiments_baseline6.py

Every file exposes the same three names -- KEY, DISPLAY_LABEL, COMPONENTS --
and the same entry point, `run(verbose=True)`, and each one is runnable on its
own from the Notebook folder:

    python -m src.mscad.experiments_step12.approach1
    python -m src.mscad.experiments_step12.baseline6

While `config.LOCK_TRAINED_MODELS` is True a configuration whose metrics.json
already exists is never retrained -- it is loaded, and the status says
"loaded (locked)".  A configuration whose result is MISSING stops with the
file name and what to set; nothing is ever trained silently.
"""

from . import common  # noqa: F401

CONFIG_MODULES = ("approach1", "approach2", "baseline1", "baseline2",
                  "baseline3", "baseline4", "baseline5", "baseline6")


def module_for(name: str):
    """Import one configuration module by its short name."""
    import importlib

    if name not in CONFIG_MODULES:
        raise KeyError("%r is not one of %s" % (name, ", ".join(CONFIG_MODULES)))
    return importlib.import_module("%s.%s" % (__name__, name))


def all_modules() -> list:
    """The eight configuration modules, in Step 12 display order."""
    return [module_for(name) for name in CONFIG_MODULES]
