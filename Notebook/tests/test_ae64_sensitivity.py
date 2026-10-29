r"""
tests/test_ae64_sensitivity.py -- the AE=64 bottleneck-sensitivity run.

NEW FILE.  Nothing is written outside tmp_path; the keras end-to-end tests
train 1 AE epoch + 1 classifier epoch on ~300 synthetic rows, proving the
wiring and shapes, not any result.  No pathlib patching.

Pinned down:
  * B1, B2, B4, B6 are present and B5 is NOT (its 30-feature AE makes a 64
    bottleneck an expansion -- skipped by design);
  * each wrapper reuses the project's own code (the paper builders for the AE
    and the B2/B4 LSTM, experiments.train_classifier for B1/B6's CNN-LSTM) --
    no model is built in these files;
  * the redirect writes only under the AE=64 root and restores config;
  * end to end: 66 features (zero_duration dropped), latent 64, PCA 30 for
    B4/B6, metrics.json in the schema comparison.py reads, official root
    byte-identical before and after;
  * B3 copies the AE=15 result with copied_from / latent_dim_insensitive.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests/test_ae64_sensitivity.py -v
"""

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad.experiments_step12_ae64 import CONFIG_MODULES, module_for
from src.mscad.experiments_step12_ae64 import common_ae64 as c64

FOLDER = (Path(__file__).resolve().parent.parent / "src" / "mscad"
          / "experiments_step12_ae64")


def hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ===========================================================================
# 1. membership: B1,B2,B4,B6 in; B5 out; B3 present (copy)
# ===========================================================================
def test_b5_is_skipped_and_the_other_five_are_present():
    assert set(CONFIG_MODULES) == {"baseline1_ae64", "baseline2_ae64",
                                   "baseline3_ae64", "baseline4_ae64",
                                   "baseline6_ae64"}
    assert "baseline5_ae64" not in CONFIG_MODULES
    # the folder names reused verbatim from AE=30
    keys = {module_for(n).NAME for n in CONFIG_MODULES}
    assert keys == {"baseline1_ae_cnn_lstm", "baseline2_ae_lstm",
                    "baseline3_cnn_lstm", "baseline4_ae_pca_lstm",
                    "baseline6_ae_pca_cnn_lstm"}


@pytest.mark.parametrize("name", CONFIG_MODULES)
def test_a_wrapper_builds_no_model_of_its_own(name):
    """The AE and LSTM come from the paper builders; the CNN-LSTM from
    experiments.train_classifier -- never a model built in these files."""
    src = _code_only((FOLDER / ("%s.py" % name)).read_text(encoding="utf-8"))
    for forbidden in ("keras", ".fit(", ".compile(", "Sequential",
                      "build_cnn_lstm", "build_autoencoder", "build_paper"):
        assert forbidden not in src, (name, forbidden)


def _code_only(source: str) -> str:
    tree = ast.parse(source)
    spans = [(n.lineno, n.col_offset, n.end_lineno, n.end_col_offset)
             for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and n.end_lineno is not None]
    lines = source.splitlines(keepends=True)
    for sl, sc, el, ec in spans:
        for ln in range(sl, el + 1):
            line = lines[ln - 1]
            b = sc if ln == sl else 0
            e = ec if ln == el else len(line.rstrip("\n"))
            nl = "\n" if line.endswith("\n") else ""
            lines[ln - 1] = line[:b] + " " * max(0, e - b) + line[e:].rstrip("\n") + nl
    return "".join(lines)


# ===========================================================================
# 2. a synthetic official root
# ===========================================================================
@pytest.fixture
def official(tmp_path, monkeypatch):
    off = tmp_path / "outputs_official"
    ae64 = tmp_path / "outputs_official_ae64"
    for name, val in (("DATASET_SOURCE", "official"), ("RUNNING_OFFICIAL", True),
                      ("OUTPUTS_ROOT", off), ("OUTPUTS_DIR", off),
                      ("OUTPUTS_ROOT_AE64", ae64),
                      ("LOCK_TRAINED_MODELS_AE64", False)):
        monkeypatch.setattr(config, name, val, raising=False)
    for n in (4, 5, 6, 7, 8, 10, 11, 12):
        monkeypatch.setattr(config, "STEP%d_OUTPUT_DIR" % n,
                            off / ("step%d" % n), raising=False)
    monkeypatch.setattr(config, "STEP12_OUTPUT_DIR", off / "step12",
                        raising=False)
    # keep the keras runs tiny
    monkeypatch.setattr(c64, "AE_EPOCHS", 1, raising=False)
    monkeypatch.setattr(c64, "CLF_EPOCHS", 1, raising=False)
    monkeypatch.setattr(config, "CLF_EPOCHS", 1, raising=False)

    rng = np.random.default_rng(0)
    feats = list(config.KAGGLE_FEATURES) + [config.ZERO_DURATION_COLUMN]
    Xf = lambda n: pd.DataFrame(rng.random((n, len(feats))), columns=feats)
    yf = lambda y: pd.DataFrame({"Label_encoded": y,
                                 "Label_binary": (y != 3).astype(int)})
    ytr, yte = rng.integers(0, 6, 300), rng.integers(0, 6, 120)
    (off / "step5").mkdir(parents=True)
    Xf(300).to_csv(off / "step5/X_train.csv", index=False)
    Xf(120).to_csv(off / "step5/X_test.csv", index=False)
    yf(ytr).to_csv(off / "step5/y_train.csv", index=False)
    yf(yte).to_csv(off / "step5/y_test.csv", index=False)
    pd.DataFrame({"Orig_Row": np.arange(120), "Source_Capture": "c"}).to_csv(
        off / "step5/meta_test.csv", index=False)
    pd.DataFrame({"Orig_Row": np.arange(300)}).to_csv(
        off / "step5/meta_train.csv", index=False)
    (off / "step8").mkdir(parents=True)
    yf(ytr).to_csv(off / "step8/y_train.csv", index=False)
    yf(yte).to_csv(off / "step8/y_test.csv", index=False)
    pd.DataFrame(rng.random((120, 15))).to_csv(off / "step8/Z_test.csv", index=False)
    (off / "step4").mkdir(parents=True)
    (off / "step4/label_mapping.json").write_text(json.dumps(
        {"HTTP_DDoS": 0, "Port_Scan": 1, "Brute_Force": 2, "Normal": 3,
         "ICMP_Flood": 4, "Web_Crwling": 5}))
    return off, ae64


# ===========================================================================
# 3. end to end
# ===========================================================================
def test_b1_trains_at_latent_64_on_66_features_cnn_lstm(official):
    pytest.importorskip("keras")
    off, ae64 = official
    before = hashes(off)
    r = module_for("baseline1_ae64").run(verbose=False)
    m = r["metrics"]
    assert r["status"] == c64.TRAINED
    assert m["n_input_features"] == 66 and m["dropped_column"] == config.ZERO_DURATION_COLUMN
    assert m["latent_dim"] == 64 and m["pca_components"] is None
    assert m["classifier"].startswith("cnn_lstm")
    for f in ("accuracy", "f1_macro", "g_mean"):
        assert f in m["full"], f
    assert "false_positive_rate" in m["binary"] and "ms_per_1000_flows" in m["inference"]
    assert c64.inside_ae64(r["metrics_path"])
    assert hashes(off) == before            # official root untouched


def test_b4_adds_pca_30_and_the_paper_lstm(official):
    pytest.importorskip("keras")
    r = module_for("baseline4_ae64").run(verbose=False)
    m = r["metrics"]
    assert m["pca_components"] == 30 and m["latent_dim"] == 64
    assert "lstm 1024/512" in m["classifier"]
    assert m["model_input_shape"] == [30, 1]


def test_b6_pca_30_clears_the_cnn_kernel_minimum(official):
    pytest.importorskip("keras")
    r = module_for("baseline6_ae64").run(verbose=False)   # would raise if steps < kernel
    assert r["metrics"]["pca_components"] == 30
    assert r["metrics"]["classifier"].startswith("cnn_lstm")


def test_b3_copies_the_ae15_result_and_says_so(official):
    off, ae64 = official
    from src.mscad import comparison as cmp
    b3 = Path(cmp.result_path("baseline3_cnn_lstm"))
    b3.parent.mkdir(parents=True, exist_ok=True)
    b3.write_text(json.dumps({"label": "B3", "full": {"accuracy": 0.98,
                  "f1_macro": 0.76}, "binary": {}, "inference": {}}),
                  encoding="utf-8")
    before = hashes(off)

    r = module_for("baseline3_ae64").run(verbose=False)
    saved = json.loads(Path(r["metrics_path"]).read_text(encoding="utf-8"))

    assert saved["latent_dim_insensitive"] is True
    assert saved["copied_from"] == str(b3)
    assert saved["full"]["accuracy"] == 0.98     # the AE=15 numbers, carried over
    assert c64.inside_ae64(r["metrics_path"])
    assert hashes(off) == before                  # copy READS the official root


def test_the_lock_makes_a_missing_result_pending(official, monkeypatch):
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS_AE64", True, raising=False)

    def explode():                                # pragma: no cover
        raise AssertionError("trainer must not run under the lock")

    out = c64.run_one_ae64("baseline1_ae_cnn_lstm", explode, verbose=False)
    assert out["status"] == c64.PENDING
    assert "LOCK_TRAINED_MODELS_AE64" in out["status_detail"]
