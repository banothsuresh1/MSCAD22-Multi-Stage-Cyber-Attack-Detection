"""
tests/test_imbalance.py -- pytest tests for STEP 9 (class weighting).

Everything is built from tiny synthetic frames in a temporary folder: no real
data, never C:\\MSCAD2022, and no model is trained.  An autouse fixture redirects
config.STEP8_OUTPUT_DIR and config.STEP9_OUTPUT_DIR to tmp_path.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pytest tests -v
"""

import json

import numpy as np
import pandas as pd
import pytest

import config
from src.mscad import imbalance as im

REAL_STEP9_DIR = config.STEP9_OUTPUT_DIR          # captured before any patching


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STEP8_OUTPUT_DIR", tmp_path / "step8")
    monkeypatch.setattr(config, "STEP9_OUTPUT_DIR", tmp_path / "step9")
    monkeypatch.setattr(config, "IMBALANCE_STRATEGY", "class_weight")
    monkeypatch.setattr(config, "CLASS_WEIGHT_SCHEME", "balanced")
    monkeypatch.setattr(config, "CLASS_WEIGHT_MAX", None)
    monkeypatch.setattr(config, "NORMALISE_WEIGHTS", True)
    monkeypatch.setattr(config, "CLF_VAL_SIZE", 0.2)


# ===========================================================================
# helpers
# ===========================================================================
def make_zy(counts=(90, 9, 1), n_latent: int = 4, seed: int = 0):
    """A Z matrix and labels with the given per-class row counts."""
    rng = np.random.default_rng(seed)
    codes = np.concatenate([np.full(n, c) for c, n in enumerate(counts)])
    names = np.array(["class_%d" % c for c in codes])
    Z = pd.DataFrame(rng.random((len(codes), n_latent)),
                     columns=["z%02d" % (i + 1) for i in range(n_latent)])
    y = pd.DataFrame({"Label": names, "Label_encoded": codes,
                      "Label_binary": (codes > 0).astype("int64")})
    return Z, y


def write_step8(Z=None, y=None, Z_test=None, y_test=None):
    """Lay down the four Step 8 files Step 9 reads."""
    if Z is None:
        Z, y = make_zy(counts=(200, 60, 20, 10))
    if Z_test is None:
        Z_test, y_test = make_zy(counts=(50, 15, 5, 3), seed=7)
    folder = config.STEP8_OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    names = config.STEP8_FILES
    Z.to_csv(folder / names["Z_train"], index=False)
    y.to_csv(folder / names["y_train"], index=False)
    Z_test.to_csv(folder / names["Z_test"], index=False)
    y_test.to_csv(folder / names["y_test"], index=False)
    return Z, y, Z_test, y_test


# ===========================================================================
# 1. the balanced formula
# ===========================================================================
def test_balanced_weights_on_a_hand_made_example():
    """w_c = n / (k * n_c): counts 90/9/1 over 3 classes of 100 rows."""
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)

    weights = im.compute_class_weights(y, scheme="balanced", max_weight=None,
                                       normalise=False)

    # 100 / (3 * 90), 100 / (3 * 9), 100 / (3 * 1)
    assert weights[0] == pytest.approx(100 / (3 * 90))
    assert weights[1] == pytest.approx(100 / (3 * 9))
    assert weights[2] == pytest.approx(100 / (3 * 1))
    assert weights[2] > weights[1] > weights[0], "rarer must mean heavier"
    assert weights[2] / weights[0] == pytest.approx(90.0)


def test_balanced_weights_equalise_the_effective_share():
    """The point of 'balanced': count x weight is the same for every class."""
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    weights = im.compute_class_weights(y, scheme="balanced", normalise=False)

    effective = [np.sum(y == c) * weights[c] for c in (0, 1, 2)]
    assert effective[0] == pytest.approx(effective[1])
    assert effective[1] == pytest.approx(effective[2])


def test_sqrt_scheme_softens_the_ratio():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    balanced = im.compute_class_weights(y, scheme="balanced", normalise=False)
    softened = im.compute_class_weights(y, scheme="sqrt_balanced", normalise=False)

    for c in (0, 1, 2):
        assert softened[c] == pytest.approx(np.sqrt(balanced[c]))
    # 90x becomes sqrt(90) ~ 9.5x
    assert softened[2] / softened[0] == pytest.approx(np.sqrt(90.0))


def test_the_cap_is_applied_after_the_sqrt_and_before_normalising():
    """Order matters: normalising last means the cap is not silently undone."""
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    cap = 2.0

    capped = im.compute_class_weights(y, scheme="balanced", max_weight=cap,
                                      normalise=False)
    assert max(capped.values()) == pytest.approx(cap)
    assert capped[2] == pytest.approx(cap), "the rarest class hits the ceiling"
    assert capped[0] < cap, "a common class is untouched by the cap"

    # the cap applies to the SOFTENED value, not the raw one
    both = im.compute_class_weights(y, scheme="sqrt_balanced", max_weight=cap,
                                    normalise=False)
    raw_sqrt = im.compute_class_weights(y, scheme="sqrt_balanced", normalise=False)
    for c in (0, 1, 2):
        assert both[c] == pytest.approx(min(raw_sqrt[c], cap))


def test_normalisation_makes_the_mean_per_sample_weight_one():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    weights = im.compute_class_weights(y, scheme="balanced", normalise=True)

    per_sample = im.sample_weights_from(y, weights)
    assert float(per_sample.mean()) == pytest.approx(1.0)
    # normalising rescales but must not reorder
    raw = im.compute_class_weights(y, scheme="balanced", normalise=False)
    assert weights[2] / weights[0] == pytest.approx(raw[2] / raw[0])


def test_an_empty_label_array_is_refused():
    with pytest.raises(im.ImbalanceError) as err:
        im.compute_class_weights(np.array([]))
    assert "empty" in str(err.value)


def test_an_unknown_scheme_is_refused():
    with pytest.raises(im.ImbalanceError) as err:
        im.compute_class_weights(np.array([0, 1]), scheme="magic")
    assert "magic" in str(err.value) and "balanced" in str(err.value)


def test_every_class_present_gets_a_weight():
    y = np.array([3, 3, 7, 9, 9, 9])
    weights = im.compute_class_weights(y)
    assert set(weights) == {3, 7, 9}


# ===========================================================================
# 2. per-sample weights
# ===========================================================================
def test_sample_weights_match_the_class_weights_row_by_row():
    y = np.array([0, 1, 2, 1, 0])
    weights = {0: 0.5, 1: 2.0, 2: 10.0}

    per_sample = im.sample_weights_from(y, weights)

    assert per_sample.tolist() == [0.5, 2.0, 10.0, 2.0, 0.5]
    assert len(per_sample) == len(y)


def test_a_missing_class_weight_is_refused_rather_than_silently_zero():
    with pytest.raises(im.ImbalanceError) as err:
        im.sample_weights_from(np.array([0, 1, 5]), {0: 1.0, 1: 1.0})
    assert "5" in str(err.value)


def test_the_weight_vector_is_ordered_by_encoded_label():
    weights = {2: 30.0, 0: 10.0, 1: 20.0}
    assert im.weight_vector(weights).tolist() == [10.0, 20.0, 30.0]
    assert im.keras_class_weight(weights) == {0: 10.0, 1: 20.0, 2: 30.0}


# ===========================================================================
# 3. make_training_sets
# ===========================================================================
def test_the_split_is_stratified_deterministic_and_adds_no_rows():
    Z, y = make_zy(counts=(200, 60, 20, 10))

    X_fit, y_fit, X_val, y_val, weights, sample_w = im.make_training_sets(Z, y)

    assert len(X_fit) + len(X_val) == len(Z), "no row added or removed"
    assert len(y_fit) == len(X_fit) and len(y_val) == len(X_val)
    # every class on both sides, in roughly the original proportions
    for code in y["Label_encoded"].unique():
        assert (y_fit["Label_encoded"] == code).sum() >= 1
        assert (y_val["Label_encoded"] == code).sum() >= 1
    share_full = (y["Label_encoded"] == 0).mean()
    assert (y_fit["Label_encoded"] == 0).mean() == pytest.approx(share_full, abs=0.02)

    again = im.make_training_sets(Z, y)
    pd.testing.assert_frame_equal(X_fit, again[0])
    pd.testing.assert_frame_equal(y_val, again[3])
    assert weights == again[4]


def test_the_two_parts_do_not_overlap_by_position():
    Z, y = make_zy(counts=(100, 40, 20, 10))
    Z = Z.copy()
    Z["row_id"] = np.arange(len(Z))          # a unique marker per row

    X_fit, _yf, X_val, _yv, _w, _s = im.make_training_sets(Z, y)

    fit_ids = set(X_fit["row_id"])
    val_ids = set(X_val["row_id"])
    assert fit_ids & val_ids == set(), "a row may not be in both parts"
    assert fit_ids | val_ids == set(range(len(Z))), "every row must appear once"


def test_weights_depend_only_on_the_fit_part():
    """Changing validation rows must not move a single weight."""
    Z, y = make_zy(counts=(200, 60, 20, 10))
    _xf, _yf, _xv, _yv, first, _s = im.make_training_sets(Z, y)

    # rewrite the Z values of the rows that land in validation
    X_fit, y_fit, X_val, y_val, _w, _s2 = im.make_training_sets(Z, y)
    wrecked = Z.copy()
    rng = np.random.default_rng(5)
    wrecked.iloc[-len(X_val):] = rng.normal(100, 50, (len(X_val), Z.shape[1]))
    _xf2, _yf2, _xv2, _yv2, second, _s3 = im.make_training_sets(wrecked, y)

    assert first == second


def test_the_inputs_are_never_modified():
    Z, y = make_zy(counts=(100, 40, 20, 10))
    Z_before = Z.copy(deep=True)
    y_before = y.copy(deep=True)

    X_fit, y_fit, X_val, y_val, _w, _s = im.make_training_sets(Z, y)
    # mutate the outputs; the inputs must not follow
    X_fit.iloc[0, 0] = -999.0
    y_fit.iloc[0, 0] = "tampered"

    pd.testing.assert_frame_equal(Z, Z_before)
    pd.testing.assert_frame_equal(y, y_before)


def test_strategy_none_gives_every_class_a_weight_of_one():
    Z, y = make_zy(counts=(200, 60, 20, 10))

    _xf, y_fit, _xv, _yv, weights, sample_w = im.make_training_sets(
        Z, y, strategy="none")

    assert set(weights.values()) == {1.0}
    assert len(weights) == y["Label_encoded"].nunique()
    assert np.all(sample_w == 1.0)
    assert len(sample_w) == len(y_fit)


def test_an_unknown_strategy_names_resampling_as_deliberately_absent():
    Z, y = make_zy(counts=(100, 40, 20, 10))
    with pytest.raises(im.ImbalanceError) as err:
        im.make_training_sets(Z, y, strategy="smote")
    message = str(err.value)
    assert "smote" in message
    assert "class weighting only" in message


def test_a_class_with_one_row_is_refused_with_a_clear_message():
    Z, y = make_zy(counts=(100, 40, 1))
    with pytest.raises(im.ImbalanceError) as err:
        im.make_training_sets(Z, y)
    assert "fewer than 2" in str(err.value)


def test_mismatched_row_counts_are_refused():
    Z, y = make_zy(counts=(100, 40, 20))
    with pytest.raises(im.ImbalanceError) as err:
        im.make_training_sets(Z, y.iloc[:-5])
    assert "rows" in str(err.value)


# ===========================================================================
# 4. the report table
# ===========================================================================
def test_the_effective_weight_share_is_flat_under_balanced_weights():
    Z, y = make_zy(counts=(200, 60, 20, 10))
    _xf, y_fit, _xv, y_val, weights, _s = im.make_training_sets(Z, y)

    table, stats = im.weight_table(y, y_fit, y_val, weights, verbose=False)

    rows = table[table["Label"].astype(str) != "TOTAL"]
    even = 100.0 / len(rows)
    assert np.allclose(rows["effective weight share %"], even, atol=0.01)
    assert stats["spread_pp"] < 0.01
    assert stats["ratio"] > 1.0


# ===========================================================================
# 5. end to end
# ===========================================================================
def test_run_step9_writes_every_file_and_passes_its_checks():
    write_step8()
    result = im.run_step9(overwrite=True, verbose=False)

    assert result["ok"] is True
    folder = config.STEP9_OUTPUT_DIR
    for key, name in config.STEP9_FILES.items():
        if key in ("fit_counts_chart", "weights_chart"):
            continue                 # matplotlib may not be installed
        assert (folder / name).exists(), name
    assert "[FAIL]" not in set(result["verification"]["result"])

    saved = json.loads((folder / config.STEP9_FILES["class_weights"])
                       .read_text(encoding="utf-8"))
    assert saved["scheme"] == config.CLASS_WEIGHT_SCHEME
    assert saved["strategy"] == "class_weight"
    assert len(saved["classes"]) == len(result["class_weights"])
    for row in saved["classes"]:
        assert row["weight"] == pytest.approx(result["class_weights"][row["encoded"]])


def test_step9_never_touches_the_test_files():
    _z, _y, Z_test, y_test = write_step8()
    folder8 = config.STEP8_OUTPUT_DIR
    before = {p.name: p.read_bytes() for p in folder8.iterdir()}

    result = im.run_step9(overwrite=True, verbose=False)

    for name, content in before.items():
        assert (folder8 / name).read_bytes() == content, name
    row = result["verification"]
    checked = row[row["check"].str.contains("byte-for-byte")]
    assert checked["result"].iloc[0] == "[OK]"


def test_the_saved_split_adds_no_rows():
    Z, _y, _zt, _yt = write_step8()
    im.run_step9(overwrite=True, verbose=False)
    folder = config.STEP9_OUTPUT_DIR

    Z_fit = pd.read_csv(folder / config.STEP9_FILES["Z_fit"])
    Z_val = pd.read_csv(folder / config.STEP9_FILES["Z_val"])
    weights = pd.read_csv(folder / config.STEP9_FILES["fit_sample_weights"])

    assert len(Z_fit) + len(Z_val) == len(Z)
    assert len(weights) == len(Z_fit)
    assert list(Z_fit.columns) == list(Z_val.columns) == list(Z.columns)


def test_step9_loads_instead_of_recomputing_when_overwrite_is_off():
    write_step8()
    first = im.run_step9(overwrite=True, verbose=False)
    again = im.run_step9(overwrite=False, verbose=False)

    assert again["status"] == "loaded_existing"
    assert again["class_weights"] == pytest.approx(first["class_weights"])
    assert len(again["Z_fit"]) == len(first["Z_fit"])


# ===========================================================================
# 6. the Windows-path guard
# ===========================================================================
def test_step9_refuses_a_windows_path_off_windows(tmp_path, monkeypatch, capsys,
                                                  assert_no_leaked_output):
    """On Linux, Path(r"C:\\...") is RELATIVE -- nothing may be created."""
    monkeypatch.setattr(config, "STEP9_OUTPUT_DIR",
                        config.Path(r"C:\MSCAD2022\outputs\step9"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    assert im.prepare_output_dir(config.STEP9_OUTPUT_DIR, "the Step 9 outputs") is None
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("prepare_output_dir for the Step 9 folder")


def test_step9_output_dir_is_under_data_root_not_the_repo():
    assert REAL_STEP9_DIR.parent.parent == config.DATA_ROOT
    assert config.PROJECT_ROOT not in REAL_STEP9_DIR.parents
    assert REAL_STEP9_DIR.name == "step9"
