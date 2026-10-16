r"""
tests/test_official_dataset.py -- Step 3 for the official MSCAD dataset.

NEW FILE.  Everything runs on tiny hand-made frames inside tmp_path: quote
stripping, column mapping, the guarantee that no identifier reaches the
features, the six labels, Normal as the only benign class, missing identifiers,
and label-conflict detection.  No model is built and nothing is trained.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import official_dataset as od

IDENTIFIERS = ("Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port",
               "Timestamp")


@pytest.fixture(autouse=True)
def official_mode(tmp_path, monkeypatch):
    """Official mode, with every folder inside tmp_path."""
    monkeypatch.setattr(config, "DATASET_SOURCE", "official")
    monkeypatch.setattr(config, "RUNNING_OFFICIAL", True)
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "outputs_official")
    monkeypatch.setattr(config, "OUTPUTS_ROOT_OWN", tmp_path / "outputs")
    monkeypatch.setattr(config, "OUTPUTS_ROOT_OFFICIAL",
                        tmp_path / "outputs_official")
    monkeypatch.setattr(config, "STEP3_OUTPUT_DIR",
                        tmp_path / "outputs_official" / "step3")
    monkeypatch.setattr(config, "STEP4_OUTPUT_DIR",
                        tmp_path / "outputs_official" / "step4")
    monkeypatch.setattr(config, "MERGED_CSV",
                        tmp_path / "outputs_official" / "step3"
                        / "MSCAD_merged.csv")
    monkeypatch.setattr(config, "BENIGN_LABELS", ["Normal"])
    monkeypatch.setattr(config, "METADATA_COLUMNS",
                        list(config.OFFICIAL_IDENTIFIER_COLUMNS)
                        + list(config.OFFICIAL_ROW_COLUMNS))
    monkeypatch.setattr(config, "METADATA_SHARED_COLUMNS", [])
    monkeypatch.setattr(config, "OFFICIAL_CSV",
                        tmp_path / "CSV_Original_kaggle" / "MSCAD.csv")
    return tmp_path


def official_frame(rows_per_class=4, with_identifiers=False,
                   quoted=("Idle Mean",), conflict=False):
    """A tiny frame in the official layout, six classes, some quoted headers."""
    features = ["Flow Duration", "Tot Fwd Pkts", "Flow Byts/s", "Idle Mean",
                "Bwd PSH Flags"]
    labels = []
    for name in config.OFFICIAL_LABELS:
        labels.extend([name] * rows_per_class)
    n = len(labels)
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({name: np.round(rng.random(n) * (i + 1) + i, 4)
                          for i, name in enumerate(features)})
    frame["Bwd PSH Flags"] = 0                       # a constant column
    frame["Protocol"] = 6
    if with_identifiers:
        frame["Flow ID"] = ["f%d" % i for i in range(n)]
        frame["Src IP"] = "10.0.0.1"
        frame["Dst IP"] = "10.0.0.2"
        frame["Src Port"] = 40000 + np.arange(n)
        frame["Dst Port"] = 22
        frame["Timestamp"] = "2022-01-01 00:00:00"
    frame["Label"] = labels
    if conflict:
        # two rows with IDENTICAL features and different labels
        frame.iloc[1, :len(features)] = frame.iloc[0, :len(features)].to_numpy()
        frame.loc[frame.index[1], "Protocol"] = frame.loc[frame.index[0],
                                                          "Protocol"]
        frame.loc[frame.index[1], "Label"] = "Normal"
        frame.loc[frame.index[0], "Label"] = "Brute_Force"
    frame.columns = ["'%s'" % c if c in quoted else c for c in frame.columns]
    return frame


def write_official(frame, path=None):
    path = Path(config.OFFICIAL_CSV if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


# ===========================================================================
# 1. reading and cleaning the header
# ===========================================================================
def test_surrounding_quotes_and_whitespace_are_stripped():
    assert od.clean_column_name("'Idle Mean'") == "Idle Mean"
    assert od.clean_column_name('  "Dst Port"  ') == "Dst Port"
    assert od.clean_column_name("Flow Byts/s") == "Flow Byts/s"
    assert od.clean_column_name('"\'Idle Mean\'"') == "Idle Mean"
    # nothing INSIDE the name is touched
    assert od.clean_column_name("Down/Up Ratio") == "Down/Up Ratio"


def test_the_file_is_read_with_the_cleaned_header_and_the_change_reported():
    write_official(official_frame(quoted=("Idle Mean", "Flow Byts/s")))

    frame, table = od.read_official_csv(verbose=False)

    assert "Idle Mean" in frame.columns
    assert "'Idle Mean'" not in frame.columns
    changed = table[table["changed"]]
    assert set(changed["cleaned column"]) == {"Idle Mean", "Flow Byts/s"}
    assert set(changed["raw column"]) == {"'Idle Mean'", "'Flow Byts/s'"}


def test_a_missing_file_names_the_path_and_the_setting():
    with pytest.raises(od.OfficialDatasetError) as error:
        od.read_official_csv(verbose=False)

    assert str(config.OFFICIAL_CSV) in str(error.value)
    assert "MSCAD_OFFICIAL_CSV_DIR" in str(error.value)


def test_cleaning_that_would_collide_two_columns_stops(official_mode):
    frame = official_frame()
    frame.columns = list(frame.columns[:-1]) + ["Label"]
    frame["'Flow Duration'"] = 1.0              # collides with Flow Duration
    write_official(frame)

    with pytest.raises(od.OfficialDatasetError) as error:
        od.read_official_csv(verbose=False)

    assert "share a name" in str(error.value)
    assert "Nothing was changed on disk" in str(error.value)


# ===========================================================================
# 2. the column mapping
# ===========================================================================
def test_names_are_matched_by_normalised_form():
    assert od.normalise_name("'Idle Mean'") == od.normalise_name("Idle Mean")
    assert od.normalise_name("Flow Byts/s") == od.normalise_name("flow_byts_per_s")
    assert od.normalise_name("Tot Fwd Pkts") == od.normalise_name(
        "Total Forward Packets")
    assert od.normalise_name("Dst Port") != od.normalise_name("Src Port")


def test_the_mapping_reports_unmatched_columns_on_both_sides():
    write_official(official_frame())
    frame, _table = od.read_official_csv(verbose=False)
    frame["Something The Project Never Had"] = 1.0

    out = od.map_columns(frame, verbose=False)

    assert "Something The Project Never Had" in out["unmatched_official"]
    # our extraction has 60+ features this tiny frame does not
    assert out["unmatched_ours"]
    assert out["matched"] >= 4
    assert len(out["table"]) == frame.shape[1]


# ===========================================================================
# 3. identifiers are never features
# ===========================================================================
def test_every_identifier_present_is_found_and_reported():
    write_official(official_frame(with_identifiers=True))
    frame, _t = od.read_official_csv(verbose=False)

    out = od.find_identifiers(frame, verbose=False)

    assert set(out["found"]) == set(IDENTIFIERS)
    assert out["missing"] == []
    assert set(out["table"]["identifier"]) == set(
        config.OFFICIAL_IDENTIFIER_COLUMNS)


def test_a_file_without_identifiers_records_them_as_unavailable():
    write_official(official_frame(with_identifiers=False))
    frame, _t = od.read_official_csv(verbose=False)

    out = od.find_identifiers(frame, verbose=False)

    assert out["found"] == {}
    assert set(out["missing"]) == set(IDENTIFIERS)
    assert set(out["table"]["status"]) == {"unavailable"}


def test_no_identifier_and_no_port_ever_reaches_the_features():
    """The guarantee the whole switch rests on: a port is not a feature."""
    write_official(official_frame(with_identifiers=True))
    frame, _t = od.read_official_csv(verbose=False)
    identifiers = od.find_identifiers(frame, verbose=False)

    merged, feature_columns = od.build_merged(frame, identifiers, verbose=False)

    for name in IDENTIFIERS:
        assert name not in feature_columns, name
        assert name in merged.columns            # kept as metadata
    assert "Src Port" not in feature_columns
    assert "Dst Port" not in feature_columns
    assert od.ROW_COLUMN not in feature_columns
    # ... and every identifier is listed in METADATA_COLUMNS, so Step 4 moves
    # it aside without being told again
    for name in IDENTIFIERS:
        assert name in config.METADATA_COLUMNS, name


def test_orig_row_is_added_and_is_not_a_feature():
    write_official(official_frame())
    frame, _t = od.read_official_csv(verbose=False)
    identifiers = od.find_identifiers(frame, verbose=False)

    merged, feature_columns = od.build_merged(frame, identifiers, verbose=False)

    assert od.ROW_COLUMN in merged.columns
    assert merged[od.ROW_COLUMN].tolist() == list(range(len(merged)))
    assert od.ROW_COLUMN not in feature_columns
    assert od.ROW_COLUMN in config.METADATA_COLUMNS


def test_an_existing_orig_row_column_is_refused_rather_than_overwritten():
    frame = official_frame()
    frame["Orig_Row"] = 99
    write_official(frame)
    read, _t = od.read_official_csv(verbose=False)
    identifiers = od.find_identifiers(read, verbose=False)

    with pytest.raises(od.OfficialDatasetError) as error:
        od.build_merged(read, identifiers, verbose=False)

    assert "already has an 'Orig_Row' column" in str(error.value)


# ===========================================================================
# 4. labels
# ===========================================================================
def test_the_six_classes_are_kept_exactly_as_the_authors_wrote_them():
    write_official(official_frame())
    frame, _t = od.read_official_csv(verbose=False)

    out = od.label_table(frame, verbose=False)

    assert out["classes"] == sorted(config.OFFICIAL_LABELS)
    assert "Web_Crwling" in out["classes"]       # the authors' spelling
    assert out["unexpected"] == []
    assert len(out["table"]) == 6


def test_normal_is_the_only_benign_class_and_there_is_no_benign_label():
    write_official(official_frame())
    frame, _t = od.read_official_csv(verbose=False)

    out = od.label_table(frame, verbose=False)

    assert list(config.BENIGN_LABELS) == ["Normal"]
    assert "Benign" not in out["classes"]


def test_a_label_the_published_list_does_not_have_is_reported():
    frame = official_frame()
    frame.loc[frame.index[0], "Label"] = "DoS"
    write_official(frame)
    read, _t = od.read_official_csv(verbose=False)

    out = od.label_table(read, verbose=False)

    assert "DoS" in out["unexpected"]
    assert out["table"][out["table"]["label"] == "DoS"].iloc[0][
        "result"] == "[SKIP]"


def test_a_file_without_a_label_column_stops_with_its_header():
    frame = official_frame().drop(columns=["Label"])
    write_official(frame)
    read, _t = od.read_official_csv(verbose=False)

    with pytest.raises(od.OfficialDatasetError) as error:
        od.label_table(read, verbose=False)

    assert "has no 'Label' column" in str(error.value)


# ===========================================================================
# 5. duplicates and label conflicts
# ===========================================================================
def test_label_conflicts_are_detected_on_a_hand_made_frame():
    """Identical features, different labels -- the ceiling on accuracy."""
    features = ["a", "b"]
    frame = pd.DataFrame({
        "a": [1.0, 1.0, 2.0, 3.0],
        "b": [5.0, 5.0, 6.0, 7.0],
        "Label": ["Brute_Force", "Normal", "Port_Scan", "Normal"]})

    out = od.label_conflicts(frame, features, verbose=False)

    assert out["rows"] == 2
    assert out["groups"] == 1
    assert len(out["summary"]) == 1
    assert out["summary"].iloc[0]["labels in conflict"] == "Brute_Force + Normal"
    assert int(out["summary"].iloc[0]["rows"]) == 2


def test_identical_rows_with_the_SAME_label_are_not_a_conflict():
    features = ["a", "b"]
    frame = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0],
                          "Label": ["Normal", "Normal"]})

    out = od.label_conflicts(frame, features, verbose=False)

    assert out["rows"] == 0
    assert out["groups"] == 0
    assert not len(out["summary"])


def test_duplicate_feature_vectors_are_counted_overall_and_per_class():
    features = ["a"]
    frame = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0],
                          "Label": ["Normal", "Normal", "Normal",
                                    "Brute_Force"]})

    out = od.duplicate_report(frame, features, verbose=False)

    assert out["count"] == 2
    assert out["share"] == pytest.approx(0.5)
    normal = out["table"][out["table"]["label"] == "Normal"].iloc[0]
    assert int(normal["rows sharing a feature vector"]) == 2


def test_the_published_duplicate_share_is_what_the_check_compares_against():
    assert config.OFFICIAL_EXPECTED_DUPLICATE_SHARE == 0.374


# ===========================================================================
# 6. Step 3 end to end
# ===========================================================================
def test_step3_writes_the_merged_file_and_its_reports(official_mode):
    write_official(official_frame(rows_per_class=5, with_identifiers=True))

    out = od.run_step3(verbose=False)
    folder = official_mode / "outputs_official" / "step3"
    names = config.OFFICIAL_STEP3_FILES

    assert out["status"] == "written"
    for key in ("merged", "report", "columns", "labels", "quality",
                "duplicates", "manifest", "summary"):
        assert (folder / names[key]).exists(), key
    merged = pd.read_csv(folder / names["merged"])
    assert len(merged) == 30
    assert merged.columns[-1] == "Label"
    assert od.ROW_COLUMN in merged.columns
    manifest = json.loads((folder / names["manifest"]).read_text(
        encoding="utf-8"))
    assert manifest["benign_labels"] == ["Normal"]
    assert manifest["identifiers_found"] == list(IDENTIFIERS)
    assert manifest["row_column"] == od.ROW_COLUMN
    for name in IDENTIFIERS:
        assert name not in manifest["feature_columns"], name


def test_step3_writes_nothing_under_the_own_output_root(official_mode):
    write_official(official_frame())

    od.run_step3(verbose=False)

    assert not (official_mode / "outputs").exists(), sorted(
        p.name for p in official_mode.iterdir())


def test_step3_refuses_to_run_in_own_mode(monkeypatch):
    monkeypatch.setattr(config, "RUNNING_OFFICIAL", False)
    monkeypatch.setattr(config, "DATASET_SOURCE", "own")

    with pytest.raises(od.OfficialDatasetError) as error:
        od.run_step3(verbose=False)

    assert "DATASET_SOURCE is 'own'" in str(error.value)
    assert "data_loading.save_merged" in str(error.value)


def test_the_verification_table_fails_when_the_file_is_not_the_published_one(
        official_mode):
    write_official(official_frame())                 # 24 rows, not 128,799

    out = od.run_step3(verbose=False)
    report = out["report"]
    failed = report[report["result"] == "[FAIL]"]

    assert out["ok"] is False
    assert "row count matches the published figure" in set(failed["check"])
    # and the checks that are about THIS file still pass
    passed = set(report[report["result"] == "[OK]"]["check"])
    assert "no identifier column reaches the features" in passed
    assert "Orig_Row is metadata, not a feature" in passed
    assert "the benign set for Label_binary is Normal only" in passed


def test_the_merged_file_reads_back_through_load_merged(official_mode):
    write_official(official_frame(with_identifiers=True))
    od.run_step3(verbose=False)

    from src.mscad import data_loading as dl

    frame = dl.load_merged(verbose=False)

    assert len(frame) == 24
    assert "Label" in frame.columns
    assert od.ROW_COLUMN in frame.columns


def test_the_summary_markdown_is_built_from_the_computed_numbers(official_mode):
    write_official(official_frame(rows_per_class=5, conflict=True))

    out = od.run_step3(verbose=False)
    text = out["summary"]

    assert "30 rows" in text
    assert "Web_Crwling" in text
    assert "Normal" in text
    assert "carry more than one label" in text
    assert str(out["conflicts"]["rows"]) in text


def test_is_inside_compares_paths_by_parts_not_by_text():
    assert od._is_inside("/d/outputs_official/step3", "/d/outputs_official")
    assert not od._is_inside("/d/outputs_official/step3", "/d/outputs")
    assert od._is_inside("/d/outputs/step3", "/d/outputs")


def test_no_keras_3_only_api_in_the_new_modules():
    root = Path(__file__).resolve().parent.parent / "src" / "mscad"
    for name in ("official_dataset.py", "training_lock.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "keras.ops" not in text, name
        assert "from keras import ops" not in text, name
