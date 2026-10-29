r"""
src/mscad/experiments_step12_ae64/ -- the AE=64 bottleneck-sensitivity run.

ADDITIVE, mirroring experiments_step12_ae30/: a dedicated output root
(config.OUTPUTS_ROOT_AE64), its own lock (config.LOCK_TRAINED_MODELS_AE64),
the common_ae64.ae64() redirect, and one wrapper per configuration.

Covered: B1, B2, B4, B6 (trained at AE latent 64) and B3 (copied from the
AE=15 result -- it has no autoencoder).  B5 is deliberately ABSENT: it trains
its AE on 30 MI-selected features, so a 64 bottleneck would EXPAND 30 -> 64,
which is not a bottleneck; per the task it is skipped rather than forced onto
all features (which would erase its identity).

Approach 1 and Approach 2 are not part of this run.

    python -m src.mscad.experiments_step12_ae64.baseline1_ae64
"""

from . import common_ae64  # noqa: F401

# the six per-config folder names are reused verbatim from the AE=30 run;
# baseline5 is intentionally not here (see the module docstring)
CONFIG_MODULES = ("baseline1_ae64", "baseline2_ae64", "baseline3_ae64",
                  "baseline4_ae64", "baseline6_ae64")


def module_for(name: str):
    from importlib import import_module

    if name not in CONFIG_MODULES:
        raise KeyError("%r is not one of %s" % (name, list(CONFIG_MODULES)))
    return import_module("%s.%s" % (__name__, name))
