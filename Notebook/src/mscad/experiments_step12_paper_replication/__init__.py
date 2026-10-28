r"""
src/mscad/experiments_step12_paper_replication/ -- a faithful-as-possible
replication of the published AE+PCA+LSTM intrusion-detection pipeline
("Fusion of linear and non-linear dimensionality reduction techniques for
feature reduction in LSTM-based Intrusion Detection System"), for an
apples-to-apples comparison against this project's B4 in the thesis.

ADDITIVE: a dedicated output root (config.OUTPUTS_ROOT_PAPER), its own lock
(config.LOCK_TRAINED_MODELS_PAPER), the context-manager redirect in
common_paper.py, and ONE configuration file -- exactly the
outputs_official_ae30\ pattern.  Nothing here reads the shared AE_* /
N_SELECTED_FEATURES constants or writes under outputs_official\ or
outputs_official_ae30\.

Every hyperparameter is the PAPER's, hardcoded in
baseline_ae64_pca30_lstm.py.  Where the paper's spec and this codebase's
established preprocessing conflict, the file's module docstring and
docs/paper_replication_notes.md record the call made and why.

    python -m src.mscad.experiments_step12_paper_replication.baseline_ae64_pca30_lstm
"""

from . import common_paper  # noqa: F401

CONFIG_MODULES = ("baseline_ae64_pca30_lstm",)


def module_for(name: str):
    from importlib import import_module

    if name not in CONFIG_MODULES:
        raise KeyError("%r is not one of %s" % (name, list(CONFIG_MODULES)))
    return import_module("%s.%s" % (__name__, name))
