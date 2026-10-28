r"""
tests/test_paper_replication.py -- the AE+PCA+LSTM paper replication.

NEW FILE.  Nothing is written outside tmp_path; the keras end-to-end test
trains 1 AE epoch + 1 LSTM epoch on ~300 synthetic rows, which proves the
wiring and the shapes, not any result.  No pathlib patching.

What is pinned down:

  * the variant has its OWN root and lock, independent of the official and
    AE=30 ones, and the redirect restores config afterwards (exception or not);
  * every hyperparameter is hardcoded in the module, not read from the shared
    config.AE_* / N_SELECTED_FEATURES constants;
  * end to end: 66 features (zero_duration dropped), latent 64, PCA 30, a
    (30, 1) sequence, and a metrics.json under the replication root carrying
    the fields comparison.py reads -- with the official root byte-identical
    before and after.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests/test_paper_replication.py -v
"""

import ast
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad.experiments_step12_paper_replication import common_paper as cp
from src.mscad.experiments_step12_paper_replication import (
    baseline_ae64_pca30_lstm as pr)

MODULE = (Path(__file__).resolve().parent.parent / "src" / "mscad"
          / "experiments_step12_paper_replication"
          / "baseline_ae64_pca30_lstm.py")


def hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ===========================================================================
# 1. the paper's hyperparameters are hardcoded, not read from shared config
# ===========================================================================
def test_the_module_hardcodes_the_paper_values():
    assert pr.AE_LATENT == 64
    assert pr.AE_HIDDEN == []
    assert pr.PCA_COMPONENTS == 30
    assert (pr.LSTM_UNITS_1, pr.LSTM_UNITS_2) == (1024, 512)
    assert pr.LSTM_DROPOUT == 0.6
    assert pr.LSTM_ACT == "tanh"
    assert pr.AE_EPOCHS == 20 and pr.CLF_EPOCHS == 20
    assert pr.AE_BATCH == 850 and pr.CLF_BATCH == 850
    assert pr.DROP_ZERO_DURATION is True


def test_no_shared_ae_or_selection_constant_drives_the_pipeline():
    """The module may read RANDOM_STATE and ZERO_DURATION_COLUMN, nothing else
    from config that would tie it to the AE=15 / AE=30 pipelines."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    attrs = {node.attr for node in ast.walk(tree)
             if isinstance(node, ast.Attribute)
             and isinstance(node.value, ast.Name) and node.value.id == "config"}
    forbidden = {"AE_LATENT_DIM", "AE_LATENT_DIM_AE30", "AE_HIDDEN",
                 "AE_ACTIVATION", "AE_OUTPUT_ACT", "AE_EPOCHS", "AE_BATCH_SIZE",
                 "AE_PATIENCE", "N_SELECTED_FEATURES", "PCA_N_COMPONENTS",
                 "CLF_EPOCHS", "CLF_BATCH_SIZE", "LSTM_UNITS", "CLF_LR"}
    assert attrs & forbidden == set(), sorted(attrs & forbidden)
    assert attrs <= {"RANDOM_STATE", "ZERO_DURATION_COLUMN",
                     "OUTPUTS_ROOT_PAPER"}, sorted(attrs)


# ===========================================================================
# 2. its own root and lock, independent of the others
# ===========================================================================
def test_the_variant_has_its_own_root_and_lock(tmp_path, monkeypatch):
    paper = tmp_path / "outputs_official_paper_replication"
    monkeypatch.setattr(config, "OUTPUTS_ROOT_PAPER", paper, raising=False)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS_PAPER", False,
                        raising=False)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True, raising=False)

    assert cp.root() == paper
    assert cp.lock_is_on() is False
    assert config.LOCK_TRAINED_MODELS is True          # not this variant's lock
    assert Path(cp.metrics_path(pr.NAME)).is_relative_to(paper)


def test_the_redirect_touches_only_root_and_lock_and_restores(tmp_path,
                                                              monkeypatch):
    paper = tmp_path / "outputs_official_paper_replication"
    official = tmp_path / "outputs_official"
    monkeypatch.setattr(config, "OUTPUTS_ROOT_PAPER", paper, raising=False)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS_PAPER", False,
                        raising=False)
    monkeypatch.setattr(config, "OUTPUTS_ROOT", official, raising=False)
    monkeypatch.setattr(config, "STEP5_OUTPUT_DIR", official / "step5",
                        raising=False)
    monkeypatch.setattr(config, "STEP8_OUTPUT_DIR", official / "step8",
                        raising=False)
    before = (Path(config.STEP5_OUTPUT_DIR), Path(config.STEP8_OUTPUT_DIR),
              config.LOCK_TRAINED_MODELS)

    with cp.paper_root() as place:
        assert Path(place) == paper
        assert Path(config.OUTPUTS_ROOT) == paper
        assert config.LOCK_TRAINED_MODELS is False     # this variant's lock
        # the dataset-input steps are NOT redirected: they stay on official
        assert Path(config.STEP5_OUTPUT_DIR) == official / "step5"
        assert Path(config.STEP8_OUTPUT_DIR) == official / "step8"

    assert (Path(config.STEP5_OUTPUT_DIR), Path(config.STEP8_OUTPUT_DIR),
            config.LOCK_TRAINED_MODELS) == before


def test_the_redirect_restores_even_on_an_exception(tmp_path, monkeypatch):
    paper = tmp_path / "outputs_official_paper_replication"
    monkeypatch.setattr(config, "OUTPUTS_ROOT_PAPER", paper, raising=False)
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "off", raising=False)
    before = Path(config.OUTPUTS_ROOT)

    with pytest.raises(RuntimeError):
        with cp.paper_root():
            raise RuntimeError("boom")

    assert Path(config.OUTPUTS_ROOT) == before


def test_the_lock_makes_a_missing_result_pending_not_a_crash(tmp_path,
                                                             monkeypatch):
    paper = tmp_path / "outputs_official_paper_replication"
    monkeypatch.setattr(config, "OUTPUTS_ROOT_PAPER", paper, raising=False)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS_PAPER", True,
                        raising=False)

    def explode():                                     # pragma: no cover
        raise AssertionError("the trainer must not run under the lock")

    result = cp.run_one_paper(pr.NAME, explode, verbose=False)
    assert result["status"] == cp.PENDING
    assert "LOCK_TRAINED_MODELS_PAPER" in result["status_detail"]


def test_inside_paper_root_compares_by_parts(tmp_path, monkeypatch):
    # Real dirs rather than a C:\ path, so the Windows-path guard in
    # usable_output_dir does not nix the root on a non-Windows runner; the
    # property under test is the PARTS comparison, not the guard.
    paper = tmp_path / "outputs_official_paper_replication"
    monkeypatch.setattr(config, "OUTPUTS_ROOT_PAPER", paper, raising=False)

    assert cp.inside_paper_root(paper / "paper_ae64_pca30_lstm" / "metrics.json")
    # outputs_official is a prefix of the name as text, but not by parts
    assert not cp.inside_paper_root(
        tmp_path / "outputs_official" / "step11" / "metrics.json")
    assert not cp.inside_paper_root(None)


# ===========================================================================
# 3. end to end, on tiny synthetic data (keras)
# ===========================================================================
@pytest.fixture
def synthetic_official(tmp_path, monkeypatch):
    official = tmp_path / "outputs_official"
    paper = tmp_path / "outputs_official_paper_replication"
    monkeypatch.setattr(config, "DATASET_SOURCE", "official", raising=False)
    monkeypatch.setattr(config, "RUNNING_OFFICIAL", True, raising=False)
    monkeypatch.setattr(config, "OUTPUTS_ROOT", official, raising=False)
    monkeypatch.setattr(config, "OUTPUTS_DIR", official, raising=False)
    monkeypatch.setattr(config, "OUTPUTS_ROOT_PAPER", paper, raising=False)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS_PAPER", False,
                        raising=False)
    for number in (4, 5, 8):
        monkeypatch.setattr(config, "STEP%d_OUTPUT_DIR" % number,
                            official / ("step%d" % number), raising=False)

    rng = np.random.default_rng(0)
    feats = list(config.KAGGLE_FEATURES) + [config.ZERO_DURATION_COLUMN]
    def Xframe(n):
        return pd.DataFrame(rng.random((n, len(feats))), columns=feats)
    def yframe(y):
        return pd.DataFrame({"Label_encoded": y,
                             "Label_binary": (y != 3).astype(int)})
    ytr, yte = rng.integers(0, 6, 300), rng.integers(0, 6, 120)
    s5 = official / "step5"; s5.mkdir(parents=True)
    Xframe(300).to_csv(s5 / "X_train.csv", index=False)
    Xframe(120).to_csv(s5 / "X_test.csv", index=False)
    yframe(ytr).to_csv(s5 / "y_train.csv", index=False)
    yframe(yte).to_csv(s5 / "y_test.csv", index=False)
    pd.DataFrame({"Orig_Row": np.arange(120), "Source_Capture": "cap"}).to_csv(
        s5 / "meta_test.csv", index=False)
    pd.DataFrame({"Orig_Row": np.arange(300)}).to_csv(
        s5 / "meta_train.csv", index=False)
    s8 = official / "step8"; s8.mkdir(parents=True)
    yframe(ytr).to_csv(s8 / "y_train.csv", index=False)
    yframe(yte).to_csv(s8 / "y_test.csv", index=False)
    pd.DataFrame(rng.random((120, 15))).to_csv(s8 / "Z_test.csv", index=False)
    s4 = official / "step4"; s4.mkdir(parents=True)
    (s4 / "label_mapping.json").write_text(json.dumps(
        {"HTTP_DDoS": 0, "Port_Scan": 1, "Brute_Force": 2, "Normal": 3,
         "ICMP_Flood": 4, "Web_Crwling": 5}))
    return official, paper


def test_end_to_end_shapes_and_the_official_root_is_untouched(
        synthetic_official, monkeypatch):
    pytest.importorskip("keras")
    official, paper = synthetic_official
    monkeypatch.setattr(pr, "AE_EPOCHS", 1, raising=False)
    monkeypatch.setattr(pr, "CLF_EPOCHS", 1, raising=False)
    before = hashes(official)
    assert before

    result = pr.run(verbose=False)
    m = result["metrics"]

    assert result["status"] == cp.TRAINED
    assert m["n_input_features"] == 66          # zero_duration dropped
    assert m["dropped_column"] == config.ZERO_DURATION_COLUMN
    assert m["latent_dim"] == 64
    assert m["pca_components"] == 30
    assert m["model_input_shape"] == [30, 1]
    for field in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                  "f1_weighted", "g_mean"):
        assert field in m["full"], field
    assert "false_positive_rate" in m["binary"]
    assert "ms_per_1000_flows" in m["inference"]
    assert Path(result["metrics_path"]).is_relative_to(paper)
    assert hashes(official) == before           # not one byte moved


def test_the_saved_metrics_are_readable_as_json(synthetic_official,
                                                monkeypatch):
    pytest.importorskip("keras")
    _official, paper = synthetic_official
    monkeypatch.setattr(pr, "AE_EPOCHS", 1, raising=False)
    monkeypatch.setattr(pr, "CLF_EPOCHS", 1, raising=False)

    pr.run(verbose=False)
    saved = json.loads(
        (paper / pr.NAME / "metrics.json").read_text(encoding="utf-8"))

    assert saved["paper_replication"] is True
    assert saved["hyperparameters"]["source"].startswith("paper")
    assert saved["hyperparameters"]["lstm"]["units"] == [1024, 512]
