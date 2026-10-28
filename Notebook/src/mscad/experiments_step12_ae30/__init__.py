r"""
src/mscad/experiments_step12_ae30/ -- the eight configurations at AE latent 30.

NEW FOLDER.  It contains NO training code.  Every file here imports the
training function the AE=15 run uses and calls it with the bottleneck at
`config.AE_LATENT_DIM_AE30` and the output root at
`config.OUTPUTS_ROOT_AE30`; `common_ae30.ae30()` is what makes config say so
for the duration of one call, and restores it afterwards.

Nothing under `outputs_official\\` is opened for writing by anything in here.
Steps 5 and 6 -- the split and the selected features -- are READ from the
active root, because they are properties of the dataset and not of the
bottleneck: a second split would change more than the one thing under test.

    python -m src.mscad.experiments_step12_ae30.baseline1_ae30
"""

from . import common_ae30  # noqa: F401

CONFIG_MODULES = ("approach1_ae30", "approach2_ae30", "baseline1_ae30",
                  "baseline2_ae30", "baseline3_ae30", "baseline4_ae30",
                  "baseline5_ae30", "baseline6_ae30")


def module_for(name: str):
    """The wrapper module for one short name, imported on demand."""
    from importlib import import_module

    if name not in CONFIG_MODULES:
        raise KeyError("%r is not one of %s" % (name, list(CONFIG_MODULES)))
    return import_module("%s.%s" % (__name__, name))
