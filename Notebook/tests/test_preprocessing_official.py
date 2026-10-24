r"""
tests/test_preprocessing_official.py -- preprocessing.csv without identifiers.

NEW FILE.  The official MSCAD dataset carries none of Flow ID, Src IP, Dst IP,
Src Port, Dst Port, Protocol or Timestamp, so `preprocessing.csv` is the feature
list followed by `Label` and nothing else.  Step 2's verification used to demand
a seven-column identifier block in front of it and reported three `[FAIL]` rows
against a file that was correct.  These tests hold the fix in place:

  * with no identifier columns, the file is len(feature_list) + 1 columns wide,
    in feature-list order, and no check FAILS -- the identifier check is
    reported `[SKIP]` instead;
  * with identifiers (own extraction), the layout is unchanged: identifier
    block, then features, then `Label`, and every check passes;
  * the expected column count is read from `feature_list.json`, never
    hard-coded.

Every path is inside tmp_path; nothing is trained and C:\\MSCAD2022 is never
touched.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import json

import numpy as np
import pandas as pd
import pytest

import config
from src.mscad import preprocessing as pp

N_FEATURES = 10


@pytest.fixture(autouse=True)
def step4_in_tmp(tmp_path, monkeypatch):
    """Step 2's output folder, inside tmp_path for every test."""
    monkeypatch.setattr(config, "STEP4_OUTPUT_DIR", tmp_path / "step4")
    monkeypatch.setattr(config, "COMBINED_CSV",
                        tmp_path / "step4" / config.STEP4_FILES["combined"])
    return tmp_path


def official_frame(n_rows=20):
    """What Step 1 hands Step 2 for the official file: features, Orig_Row, Label."""
    rng = np.random.default_rng(3)
    data = {name: rng.uniform(1, 100, n_rows).round(4)
            for name in config.KAGGLE_FEATURES[:N_FEATURES]}
    data["Orig_Row"] = np.arange(n_rows, dtype="int64")
    data[config.LABEL_COLUMN] = (["Normal"] * (n_rows // 2)
                                 + ["Brute_Force"] * (n_rows - n_rows // 2))
    return pd.DataFrame(data)


def own_frame(n_rows=20):
    """The same, plus the six identifiers the own extraction recovers."""
    frame = official_frame(n_rows)
    frame.insert(0, "Src IP", ["10.0.0.%d" % (i % 4 + 1) for i in range(n_rows)])
    frame.insert(1, "Dst IP", ["10.0.0.9"] * n_rows)
    frame.insert(2, "Src Port", np.arange(40000, 40000 + n_rows, dtype="int64"))
    frame.insert(3, "Dst Port", np.full(n_rows, 22, dtype="int64"))
    frame.insert(4, "Timestamp", ["2022-01-01 00:00:%02d" % i
                                  for i in range(n_rows)])
    return frame


def official_mode(monkeypatch):
    """config as the official dataset leaves it: no non-feature block."""
    monkeypatch.setattr(config, "DATASET_SOURCE", "official")
    monkeypatch.setattr(config, "RUNNING_OFFICIAL", True)
    monkeypatch.setattr(config, "BENIGN_LABELS", ["Normal"])
    monkeypatch.setattr(config, "METADATA_COLUMNS", ["Orig_Row"])
    monkeypatch.setattr(config, "METADATA_SHARED_COLUMNS", [])
    monkeypatch.setattr(config, "COMBINED_ID_COLUMNS", [])
    monkeypatch.setattr(config, "COMBINED_METADATA_ID_COLUMNS", [])
    monkeypatch.setattr(config, "COMBINED_INT_COLUMNS", [])


def own_mode(monkeypatch):
    """config as the own extraction leaves it, minus Protocol (not in the frame)."""
    identifiers = ["Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp"]
    monkeypatch.setattr(config, "DATASET_SOURCE", "own")
    monkeypatch.setattr(config, "RUNNING_OFFICIAL", False)
    monkeypatch.setattr(config, "METADATA_COLUMNS", identifiers + ["Orig_Row"])
    monkeypatch.setattr(config, "METADATA_SHARED_COLUMNS", [])
    monkeypatch.setattr(config, "COMBINED_ID_COLUMNS", identifiers)
    monkeypatch.setattr(config, "COMBINED_METADATA_ID_COLUMNS", identifiers)
    monkeypatch.setattr(config, "COMBINED_INT_COLUMNS", ["Src Port", "Dst Port"])


def feature_list_on_disk():
    path = config.STEP4_OUTPUT_DIR / config.STEP4_FILES["feature_list"]
    assert path.exists(), "Step 2 did not write feature_list.json"
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def result_of(table, needle):
    rows = table[table["check"].str.contains(needle, regex=False)]
    assert len(rows) == 1, (needle, table["check"].tolist())
    return rows.iloc[0]["result"], rows.iloc[0]["detail"]


# ===========================================================================
# 1. the official dataset: features + Label, and no FAIL
# ===========================================================================
def test_no_identifiers_gives_feature_list_plus_label(monkeypatch):
    official_mode(monkeypatch)

    result = pp.run_step4(official_frame(), overwrite=True, verbose=False)

    features = feature_list_on_disk()
    combined = pd.read_csv(config.COMBINED_CSV, low_memory=False)
    assert list(combined.columns) == features + [config.LABEL_COLUMN]
    assert combined.shape[1] == len(features) + 1
    assert result["combined"]["ok"] is True


def test_ten_features_and_a_label_make_eleven_columns(monkeypatch):
    official_mode(monkeypatch)

    pp.run_step4(official_frame(), overwrite=True, verbose=False)

    # the frame has 10 features; the inf-capping marker is not added here,
    # so feature_list.json holds exactly those ten
    assert len(feature_list_on_disk()) == N_FEATURES
    combined = pd.read_csv(config.COMBINED_CSV, low_memory=False)
    assert combined.shape[1] == N_FEATURES + 1 == 11


def test_not_one_check_fails_and_the_identifier_check_is_skipped(monkeypatch):
    official_mode(monkeypatch)

    result = pp.run_step4(official_frame(), overwrite=True, verbose=False)
    table = result["combined"]["verification"]

    assert "[FAIL]" not in table["result"].tolist(), table.to_string()
    status, detail = result_of(table, "identifier values match metadata")
    assert status == "[SKIP]"
    assert detail == "the official dataset has no identifier columns"
    # and the three rows that used to fail now read for a dataset with none
    assert result_of(table, "columns == %d features + 1 label"
                     % N_FEATURES)[0] == "[OK]"
    assert result_of(table,
                     "columns are in the expected feature-list order")[0] == "[OK]"
    assert not [c for c in table["check"] if "are the identifiers" in c]


def test_the_expected_count_comes_from_feature_list_not_a_constant(monkeypatch):
    """Eight features must expect nine columns, not eleven."""
    official_mode(monkeypatch)
    frame = official_frame()
    frame = frame.drop(columns=list(config.KAGGLE_FEATURES[:2]))

    result = pp.run_step4(frame, overwrite=True, verbose=False)
    table = result["combined"]["verification"]

    features = feature_list_on_disk()
    assert len(features) == N_FEATURES - 2
    assert "[FAIL]" not in table["result"].tolist(), table.to_string()
    _status, detail = result_of(table, "columns ==")
    assert detail == "%d on disk vs %d expected" % (len(features) + 1,
                                                    len(features) + 1)


def test_a_wrong_column_count_still_fails(monkeypatch):
    """The check is loosened for a dataset with no identifiers, not disabled."""
    official_mode(monkeypatch)
    pp.run_step4(official_frame(), overwrite=True, verbose=False)

    combined = pd.read_csv(config.COMBINED_CSV, low_memory=False)
    combined["Stowaway"] = 1.0                    # a column nothing wrote
    combined.to_csv(config.COMBINED_CSV, index=False)
    features = pd.read_csv(config.STEP4_OUTPUT_DIR
                           / config.STEP4_FILES["features_scaled"],
                           low_memory=False)
    labels = pd.read_csv(config.STEP4_OUTPUT_DIR / config.STEP4_FILES["labels"],
                         low_memory=False)

    table, ok, _on_disk = pp.verify_combined(
        combined, features, labels, None, expected_rows=len(features),
        feature_order=feature_list_on_disk(), path=config.COMBINED_CSV,
        verbose=False)

    assert ok is False
    assert result_of(table, "columns ==")[0] == "[FAIL]"


# ===========================================================================
# 2. the own extraction: the identifier block is untouched
# ===========================================================================
def test_identifiers_still_come_first_in_own_mode(monkeypatch):
    own_mode(monkeypatch)

    result = pp.run_step4(own_frame(), overwrite=True, verbose=False)

    identifiers = list(config.COMBINED_ID_COLUMNS)
    features = feature_list_on_disk()
    combined = pd.read_csv(config.COMBINED_CSV, low_memory=False)
    assert list(combined.columns) == identifiers + features + [
        config.LABEL_COLUMN]
    assert combined.shape[1] == len(identifiers) + len(features) + 1
    assert result["combined"]["ok"] is True


def test_own_mode_keeps_its_identifier_checks(monkeypatch):
    own_mode(monkeypatch)

    result = pp.run_step4(own_frame(), overwrite=True, verbose=False)
    table = result["combined"]["verification"]

    assert "[FAIL]" not in table["result"].tolist(), table.to_string()
    assert "[SKIP]" not in table["result"].tolist(), table.to_string()
    n_ids = len(config.COMBINED_ID_COLUMNS)
    assert result_of(table, "the first %d columns are the identifiers" % n_ids
                     )[0] == "[OK]"
    assert result_of(table, "identifier values match metadata")[0] == "[OK]"
    assert result_of(table, "ports are integers")[0] == "[OK]"


def test_own_mode_still_fails_a_shuffled_identifier_block(monkeypatch):
    own_mode(monkeypatch)
    pp.run_step4(own_frame(), overwrite=True, verbose=False)

    combined = pd.read_csv(config.COMBINED_CSV, low_memory=False)
    identifiers = list(config.COMBINED_ID_COLUMNS)
    reordered = [identifiers[1], identifiers[0]] + identifiers[2:]
    combined = combined[reordered + [c for c in combined.columns
                                     if c not in identifiers]]
    combined.to_csv(config.COMBINED_CSV, index=False)
    features = pd.read_csv(config.STEP4_OUTPUT_DIR
                           / config.STEP4_FILES["features_scaled"],
                           low_memory=False)
    labels = pd.read_csv(config.STEP4_OUTPUT_DIR / config.STEP4_FILES["labels"],
                         low_memory=False)
    metadata = pd.read_csv(config.STEP4_OUTPUT_DIR
                           / config.STEP4_FILES["metadata"], low_memory=False)

    table, ok, _on_disk = pp.verify_combined(
        combined, features, labels, metadata, expected_rows=len(features),
        feature_order=feature_list_on_disk(), path=config.COMBINED_CSV,
        verbose=False)

    assert ok is False


# ===========================================================================
# 3. config routes both modes
# ===========================================================================
def test_official_config_asks_for_no_identifier_block():
    import importlib
    import os

    previous = os.environ.get("MSCAD_DATASET_SOURCE")
    os.environ["MSCAD_DATASET_SOURCE"] = "official"
    try:
        official = importlib.reload(config)
        assert official.COMBINED_ID_COLUMNS == []
        assert official.COMBINED_METADATA_ID_COLUMNS == []
        assert official.COMBINED_INT_COLUMNS == []
        # the never-train guard is a different list and stays populated
        assert "Src IP" in official.ID_COLUMNS_NOT_FOR_TRAINING
    finally:
        if previous is None:
            os.environ.pop("MSCAD_DATASET_SOURCE", None)
        else:
            os.environ["MSCAD_DATASET_SOURCE"] = previous
        importlib.reload(config)


def test_own_config_keeps_its_six_identifiers():
    assert config.COMBINED_ID_COLUMNS == ["Src IP", "Dst IP", "Src Port",
                                          "Dst Port", "Protocol", "Timestamp"]
    assert config.COMBINED_INT_COLUMNS == ["Src Port", "Dst Port"]
