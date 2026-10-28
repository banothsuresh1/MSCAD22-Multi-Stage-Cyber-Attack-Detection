r"""
tests/test_ae30_sensitivity.py -- the AE latent-30 run.

NEW FILE.  Nothing is written outside tmp_path and no real model is trained:
the one autoencoder here is 3 epochs on 60 synthetic rows, which is enough to
prove the encoder's output width and nothing else.  No pathlib patching.

What is pinned down:

  * every wrapper imports the EXISTING trainer rather than carrying a copy of
    it -- checked by reading the source for the import and for the absence of
    a model-building call;
  * `train_autoencoder(..., latent_dim=30)` really produces a 30-wide
    encoding, not the configured 15;
  * the redirect sends writes to outputs_official_ae30\\<name>\\ and restores
    every config attribute afterwards, exception or not;
  * the AE=15 root is byte-identical before and after an AE=30 run (sha256
    per file);
  * the 10.y table is built from the AE=30 root, not from the AE=15 one;
  * this root's lock is LOCK_TRAINED_MODELS_AE30, independent of
    LOCK_TRAINED_MODELS.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests/test_ae30_sensitivity.py -v
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import comparison as cmp
from src.mscad.experiments_step12_ae30 import CONFIG_MODULES, module_for
from src.mscad.experiments_step12_ae30 import common_ae30 as c30

B1 = "baseline1_ae_cnn_lstm"
FOLDER = (Path(__file__).resolve().parent.parent / "src" / "mscad"
          / "experiments_step12_ae30")



def _code_only(source: str) -> str:
    """`source` with every string literal blanked out, so prose cannot match."""
    import ast as _ast

    tree = _ast.parse(source)
    spans = [(n.lineno, n.col_offset, n.end_lineno, n.end_col_offset)
             for n in _ast.walk(tree)
             if isinstance(n, _ast.Constant) and isinstance(n.value, str)
             and n.end_lineno is not None]
    lines = source.splitlines(keepends=True)
    for start_line, start_col, end_line, end_col in spans:
        for number in range(start_line, end_line + 1):
            line = lines[number - 1]
            begin = start_col if number == start_line else 0
            finish = end_col if number == end_line else len(line.rstrip("\n"))
            keep_newline = "\n" if line.endswith("\n") else ""
            lines[number - 1] = (line[:begin] + " " * max(0, finish - begin)
                                 + line[finish:].rstrip("\n") + keep_newline)
    return "".join(lines)


def metrics(value: float) -> dict:
    return {"full": {"accuracy": value, "precision_macro": value,
                     "recall_macro": value, "f1_macro": value,
                     "f1_weighted": value, "g_mean": value,
                     "roc_auc": {"macro": value}},
            "duplicate_free": {"f1_macro": value, "g_mean": value},
            "binary": {"f1": value, "false_positive_rate": 0.01},
            "inference": {"ms_per_1000_flows": 1.0},
            "n_input_features": 67, "latent_dim": 30, "parameters": 50438,
            "training_seconds": 1.0}


def put(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def hashes(root: Path) -> dict[str, str]:
    """sha256 of every file under `root`, by relative path."""
    return {str(p.relative_to(root)):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def two_roots(tmp_path, monkeypatch):
    """The AE=15 root with all eight results, and an empty AE=30 root."""
    ae15 = tmp_path / "outputs_official"
    ae30 = tmp_path / "outputs_official_ae30"
    for name, value in (("DATASET_SOURCE", "official"),
                        ("RUNNING_OFFICIAL", True),
                        ("OUTPUTS_ROOT", ae15), ("OUTPUTS_DIR", ae15),
                        ("OUTPUTS_ROOT_AE30", ae30),
                        ("AE_LATENT_DIM", 15),
                        ("AE_LATENT_DIM_AE30", 30),
                        ("LOCK_TRAINED_MODELS", True),
                        ("LOCK_TRAINED_MODELS_AE30", False)):
        monkeypatch.setattr(config, name, value, raising=False)
    for number in (5, 6, 7, 8, 9, 10, 11, 12):
        monkeypatch.setattr(config, "STEP%d_OUTPUT_DIR" % number,
                            ae15 / ("step%d" % number), raising=False)
    monkeypatch.setattr(config, "STEP11B_OUTPUT_DIR",
                        ae15 / "step11_attention", raising=False)
    monkeypatch.setattr(config, "STEP12_B6_OUTPUT_DIR",
                        ae15 / "step12_baseline6", raising=False)
    for key, _label in config.STEP12_DISPLAY_ORDER:
        put(Path(cmp.result_path(key)), metrics(0.50))
    ae30.mkdir(parents=True, exist_ok=True)
    return ae15, ae30


# ===========================================================================
# 1. the wrappers import the existing trainers, they do not duplicate them
# ===========================================================================
def test_every_configuration_has_a_wrapper():
    assert len(CONFIG_MODULES) == 8
    for name in CONFIG_MODULES:
        assert (FOLDER / ("%s.py" % name)).exists(), name
        assert module_for(name).KEY in dict(config.STEP12_DISPLAY_ORDER), name
    keys = {module_for(name).KEY for name in CONFIG_MODULES}
    assert keys == {key for key, _label in config.STEP12_DISPLAY_ORDER}


@pytest.mark.parametrize("name", CONFIG_MODULES)
def test_a_wrapper_calls_an_existing_trainer_and_builds_no_model(name):
    source = (FOLDER / ("%s.py" % name)).read_text(encoding="utf-8")
    code = _code_only(source)

    # it reaches for the project's own training code ...
    assert "from .. import" in code, name
    # ... and contains no model building, no fit, no compile of its own.
    # Docstrings are stripped first: baseline6's explains WHY it keeps three
    # PCA components by naming build_cnn_lstm, which is prose, not a call.
    for forbidden in ("keras", ".fit(", ".compile(", "Sequential",
                      "build_cnn_lstm", "build_autoencoder"):
        assert forbidden not in code, (name, forbidden)


def test_baseline3_carries_the_ae15_result_and_says_so(two_roots):
    """No autoencoder, so the latent width cannot change it.

    Exercised through the public run(): commit 846a775 moved the copy into
    _make_train(source), binding the AE=15 source path BEFORE the ae30()
    redirect (otherwise it would read the empty AE=30 root), so there is no
    longer a module-level train() to call directly.
    """
    import json

    module = module_for("baseline3_ae30")
    ae15_source = cmp.result_path(module.KEY)      # official root, before redirect

    result = module.run(verbose=False)

    assert result["status"] == c30.TRAINED         # a copy counts as produced here
    saved = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert saved["latent_dim_insensitive"] is True
    assert "no autoencoder" in saved["note"]
    assert saved["copied_from"] == str(ae15_source)
    # and the copy landed in the AE=30 root, not the official one
    assert c30.inside_ae30(result["metrics_path"])


# ===========================================================================
# 2. latent_dim = 30 really is 30
# ===========================================================================
def test_train_autoencoder_honours_latent_dim_30(monkeypatch):
    """The parameter, not config: AE_LATENT_DIM stays 15 for this call."""
    pytest.importorskip("keras")
    from src.mscad import autoencoder as ae

    monkeypatch.setattr(config, "AE_LATENT_DIM", 15, raising=False)
    monkeypatch.setattr(config, "AE_EPOCHS", 3, raising=False)
    rows, width = 60, 40
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((rows, width), dtype="float32"),
                     columns=["f%d" % i for i in range(width)])
    y = pd.DataFrame({"Label_encoded": ([0, 1, 2] * rows)[:rows]})

    _auto, encoder, _history, _info = ae.train_autoencoder(
        X, y, epochs=3, latent_dim=30, verbose=False)
    Z = encoder.predict(X.to_numpy(dtype="float32"), verbose=0)

    assert Z.shape == (rows, 30)
    assert config.AE_LATENT_DIM == 15          # the call did not move config

    _auto2, encoder2, _h2, _i2 = ae.train_autoencoder(
        X, y, epochs=3, verbose=False)         # the default is unchanged
    assert encoder2.predict(X.to_numpy(dtype="float32"),
                            verbose=0).shape == (rows, 15)


# ===========================================================================
# 3. the redirect
# ===========================================================================
def test_the_redirect_points_every_output_at_the_ae30_root(two_roots):
    _ae15, ae30 = two_roots

    with c30.ae30() as place:
        assert Path(place) == ae30
        assert config.AE_LATENT_DIM == 30
        assert Path(config.STEP7_OUTPUT_DIR) == ae30 / "step7"
        assert Path(config.STEP12_OUTPUT_DIR) == ae30 / "step12"
        assert Path(config.STEP11B_OUTPUT_DIR) == ae30 / "step11_attention"
        assert Path(cmp.result_path(B1)).is_relative_to(ae30)
        # Steps 5 and 6 are NOT redirected: they belong to the dataset
        assert Path(config.STEP6_OUTPUT_DIR).is_relative_to(_ae15)


def test_the_redirect_is_restored_even_when_the_trainer_raises(two_roots):
    before = (config.AE_LATENT_DIM, Path(config.STEP12_OUTPUT_DIR),
              config.LOCK_TRAINED_MODELS)

    with pytest.raises(RuntimeError):
        with c30.ae30():
            raise RuntimeError("the trainer fell over")

    assert (config.AE_LATENT_DIM, Path(config.STEP12_OUTPUT_DIR),
            config.LOCK_TRAINED_MODELS) == before


def test_the_ae30_root_has_its_own_lock(two_roots, monkeypatch):
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS", True, raising=False)
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS_AE30", True,
                        raising=False)

    def explode():                             # pragma: no cover
        raise AssertionError("the trainer must not be called")

    result = c30.run_one_ae30(B1, explode, verbose=False)

    assert result["status"] == c30.PENDING
    assert "LOCK_TRAINED_MODELS_AE30" in result["status_detail"]

    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS_AE30", False,
                        raising=False)
    assert c30.lock_is_on() is False           # and the AE=15 lock is untouched
    assert config.LOCK_TRAINED_MODELS is True


# ===========================================================================
# 4. the AE=15 root is not touched, and 10.y reads the AE=30 one
# ===========================================================================
def test_a_training_run_writes_to_ae30_and_leaves_ae15_byte_identical(
        two_roots):
    ae15, ae30 = two_roots
    before = hashes(ae15)
    assert before                              # there is something to protect

    def trainer():
        """Stands in for the real one: writes where the redirect points."""
        assert Path(config.STEP12_OUTPUT_DIR).is_relative_to(ae30)
        put(Path(cmp.result_path(B1)), metrics(0.90))
        return {"metrics": metrics(0.90)}

    result = c30.run_one_ae30(B1, trainer, verbose=False)

    assert result["status"] == c30.TRAINED
    # the layout the existing trainers and the existing builder already use,
    # under the AE=30 root: step12\<key>\metrics.json for a Step 12 baseline
    written = ae30 / "step12" / B1 / "metrics.json"
    assert written.exists()
    assert Path(result["metrics_path"]) == written
    assert json.loads(written.read_text())["full"]["f1_macro"] == 0.90
    assert hashes(ae15) == before              # not one byte moved


def test_the_ae30_table_reads_the_ae30_root(two_roots):
    _ae15, ae30 = two_roots
    with c30.ae30():
        for key, _label in config.STEP12_DISPLAY_ORDER:
            put(Path(cmp.result_path(key)), metrics(0.90))

    table = c30.comparison_table()

    assert len(table) == 8
    # 0.90 is the AE=30 value; 0.50 is what the AE=15 root holds
    assert set(table["accuracy"]) == {0.90}
    assert 0.50 not in set(table["accuracy"])
    provenance = c30.provenance()
    assert (provenance["under the AE=30 root"] == "yes").all()
    assert (provenance["file exists"] == "yes").all()


def test_a_missing_ae30_result_reads_as_not_trained_not_as_the_ae15_row(
        two_roots, monkeypatch):
    monkeypatch.setattr(config, "LOCK_TRAINED_MODELS_AE30", True,
                        raising=False)

    table = c30.comparison_table()
    provenance = c30.provenance()

    assert len(table) == 8
    assert table["accuracy"].isna().all()      # no AE=15 number leaked in
    assert (provenance["file exists"] == "no").all()
    assert c30.is_finished(B1) is False


# ===========================================================================
# 5. the duplicate sub-table is gone from the 10.x renderer
# ===========================================================================
def test_the_duplicate_approach_pair_subtable_is_not_rendered():
    source = (Path(__file__).resolve().parent.parent / "src" / "mscad"
              / "comparison.py").read_text(encoding="utf-8")
    body = source.split("def show_honest_step10")[1]

    assert "trained_only_table(honest)" not in body
    assert 'Approach 1 vs Approach 2 (the %s dataset)' not in body
    # the helper itself is kept for a caller that wants the trained subset
    assert "def trained_only_table" in source
