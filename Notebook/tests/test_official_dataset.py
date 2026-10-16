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


# ===========================================================================
# 7. the own-extraction Step 3 functions, in official mode
# ===========================================================================
# The bug these cover: `save_step3_reports` crosstabs Source_Capture x Label,
# and the official dataset has no Source_Capture column, so in official mode it
# failed with `KeyError: 'Source_Capture'` seven frames inside pandas -- which
# says nothing about what to do. Each one now says it itself and returns a
# skipped status, writing nothing, so a Run All in official mode walks past
# 3.1-3.7 instead of stopping.
def test_own_step3_does_not_apply_in_official_mode():
    from src.mscad import data_loading as dl

    assert dl.own_step3_applies() is False


def test_own_step3_applies_in_own_mode(monkeypatch):
    from src.mscad import data_loading as dl

    monkeypatch.setattr(config, "DATASET_SOURCE", "own")

    assert dl.own_step3_applies() is True


def test_the_report_writer_skips_instead_of_raising_a_key_error(official_mode):
    """The exact failure: a frame with no Source_Capture column."""
    from src.mscad import data_loading as dl

    frame = pd.DataFrame({"Label": ["Normal", "Brute_Force"],
                          "Flow Duration": [1.0, 2.0]})

    out = dl.save_step3_reports(frame, verbose=False)

    assert out["status"] == "skipped_official_mode"
    assert out["written"] == {}
    assert "Source_Capture" in out["note"]
    assert "3.O" in out["note"]
    assert 'DATASET_SOURCE = "own"' in out["note"]
    # and nothing was written anywhere
    assert not (official_mode / "outputs").exists()


def test_every_own_step3_entry_point_skips_in_official_mode(official_mode):
    from src.mscad import data_loading as dl

    frame = pd.DataFrame({"Label": ["Normal"], "Flow Duration": [1.0]})

    table, ok = dl.check_column_consistency({}, verbose=False)
    assert not len(table) and ok is True

    merged, infos = dl.merge_flow_csvs(frames={}, verbose=False)
    assert not len(merged) and infos == []

    assert dl.save_merged(frame, verbose=False)["status"] == \
        "skipped_official_mode"
    assert dl.summarize_dataset(frame, make_plot=False)["status"] == \
        "skipped_official_mode"
    assert dl.save_step3_reports(frame, verbose=False)["status"] == \
        "skipped_official_mode"


def test_save_merged_in_official_mode_writes_nothing(official_mode):
    """It must not overwrite the file 3.O wrote, or create one of its own."""
    from src.mscad import data_loading as dl

    write_official(official_frame())
    od.run_step3(verbose=False)
    merged_path = Path(config.MERGED_CSV)
    before = merged_path.read_bytes()

    out = dl.save_merged(pd.DataFrame({"Label": ["Normal"]}), verbose=False)

    assert out["status"] == "skipped_official_mode"
    assert merged_path.read_bytes() == before


def test_load_merged_still_works_in_official_mode(official_mode):
    """3.5 is NOT skipped: it round-trips the file 3.O wrote."""
    from src.mscad import data_loading as dl

    write_official(official_frame(with_identifiers=True))
    od.run_step3(verbose=False)

    frame = dl.load_merged(verbose=False)

    assert len(frame) == 24
    assert "Src Port" in frame.columns          # kept, as metadata
    assert od.ROW_COLUMN in frame.columns


def test_the_notebook_cells_for_3_1_to_3_7_are_mode_aware():
    """A Run All in official mode must not stop in the own-extraction cells."""
    nbformat = pytest.importorskip("nbformat")
    notebook = (Path(__file__).resolve().parent.parent / "notebooks"
                / "MSCAD_Pipeline.ipynb")
    if not notebook.exists():                  # pragma: no cover
        pytest.skip("the notebook is not in this checkout")
    nb = nbformat.read(str(notebook), as_version=4)
    start = next(i for i, c in enumerate(nb.cells)
                 if c.cell_type == "markdown" and "## Step 3" in c.source)
    end = next(i for i, c in enumerate(nb.cells)
               if c.cell_type == "markdown" and "## Step 4" in c.source)

    guarded = 0
    for cell in nb.cells[start:end]:
        if cell.cell_type != "code":
            continue
        own_calls = ("check_column_consistency(", "merge_flow_csvs(",
                     "save_merged(", "save_step3_reports(",
                     "summarize_dataset(")
        if any(call in cell.source for call in own_calls):
            assert "config.RUNNING_OFFICIAL" in cell.source, cell["id"]
            assert "OWN_STEP3_SKIP_NOTE" in cell.source, cell["id"]
            guarded += 1
    assert guarded == 5, guarded


# ===========================================================================
# 8. the readable Protocol column in preprocessing.csv
# ===========================================================================
# The second bug of the same shape: official mode left Protocol out of
# COMBINED_ID_COLUMNS, so the readable text column was never derived and a
# Step 4 cell failed with `KeyError: 'Protocol'`. It is derived here too now --
# and when the dataset has no Protocol column at all there are no one-hot
# features to derive it from, which is reported rather than raised.
def combined_parts(with_one_hot=True, rows=6):
    """The three frames build_combined() joins."""
    features = pd.DataFrame({
        "Flow Duration": np.arange(rows, dtype="float64"),
        "Tot Fwd Pkts": np.ones(rows),
    })
    if with_one_hot:
        features["Protocol_ICMP"] = [0] * rows
        features["Protocol_TCP"] = [1] * rows
        features["Protocol_UDP"] = [0] * rows
        features["Protocol_other"] = [0] * rows
    labels = pd.DataFrame({
        "Label": ["Normal"] * rows,
        "Label_encoded": [3] * rows,
        "Label_binary": [0] * rows})
    metadata = pd.DataFrame({
        "Flow ID": ["f%d" % i for i in range(rows)],
        "Src IP": ["10.0.0.1"] * rows,
        "Dst IP": ["10.0.0.2"] * rows,
        # ports must stay integers: build_combined casts COMBINED_INT_COLUMNS
        "Src Port": np.arange(40000, 40000 + rows, dtype="int64"),
        "Dst Port": np.full(rows, 22, dtype="int64"),
        "Timestamp": ["2022-01-01 00:00:00"] * rows,
        "Orig_Row": np.arange(rows, dtype="int64")})
    return features, labels, metadata


def test_official_mode_derives_the_readable_protocol_column():
    from src.mscad import preprocessing as pp

    features, labels, metadata = combined_parts(with_one_hot=True)

    combined = pp.build_combined(features, labels, metadata, verbose=False)

    assert "Protocol" in combined.columns
    assert set(combined["Protocol"]) == {"TCP"}
    # the one-hot features are replaced by the readable column in this view
    for name in config.PROTOCOL_TEXT_FROM_ONE_HOT:
        assert name not in combined.columns, name
    assert "Protocol" in config.COMBINED_ID_COLUMNS
    assert "Protocol" not in config.COMBINED_METADATA_ID_COLUMNS


def test_a_dataset_with_no_protocol_column_is_reported_not_raised(capsys):
    """The official file may have no Protocol column; that is a fact, not a bug."""
    from src.mscad import preprocessing as pp

    features, labels, metadata = combined_parts(with_one_hot=False)

    combined = pp.build_combined(features, labels, metadata, verbose=True)
    printed = capsys.readouterr().out

    assert "Protocol" not in combined.columns
    assert "No readable `Protocol` column" in printed
    assert "has no Protocol column" in printed
    assert "Flow Duration" in combined.columns       # the flows are unchanged
    assert len(combined) == 6


def test_own_mode_still_fails_loudly_when_its_one_hots_are_missing(monkeypatch):
    """There, a missing one-hot means Step 4's encoding is broken."""
    from src.mscad import preprocessing as pp

    monkeypatch.setattr(config, "DATASET_SOURCE", "own")
    monkeypatch.setattr(config, "COMBINED_ID_COLUMNS", ["Protocol"])
    monkeypatch.setattr(config, "COMBINED_METADATA_ID_COLUMNS", [])
    features, labels, _metadata = combined_parts(with_one_hot=False)

    with pytest.raises(pp.PreprocessingError) as error:
        pp.build_combined(features, labels, None, verbose=False)

    assert "cannot build the readable `Protocol` column" in str(error.value)
    assert "OVERWRITE_STEP4 = True" in str(error.value)


def test_the_protocol_check_cell_guards_on_the_column_being_there():
    nbformat = pytest.importorskip("nbformat")
    notebook = (Path(__file__).resolve().parent.parent / "notebooks"
                / "MSCAD_Pipeline.ipynb")
    if not notebook.exists():                  # pragma: no cover
        pytest.skip("the notebook is not in this checkout")
    nb = nbformat.read(str(notebook), as_version=4)

    cells = [c for c in nb.cells
             if c.cell_type == "code" and 'final["Protocol"]' in c.source]

    assert len(cells) == 1
    source = cells[0].source
    assert '"Protocol" not in final.columns' in source
    assert "[SKIP]" in source
    assert source.index('"Protocol" not in final.columns') < source.index(
        'final["Protocol"].value_counts()')
