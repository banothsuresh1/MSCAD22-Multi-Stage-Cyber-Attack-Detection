r"""
src/mscad/official_dataset.py -- STEP 3 for the OFFICIAL MSCAD dataset.

NEW FILE.  It replaces Steps 1-3 for `DATASET_SOURCE = "official"`: the
official file is already one labelled flow table, so there is nothing to
convert from PCAP and nothing to merge per capture.  What is left is the part
that still has to be done honestly -- read the authors' file, say exactly what
is in it, and hand Step 4 the same layout the own-extraction Step 3 handed it.

WHY the official file is the primary dataset
    Official MSCAD.csv: 128,799 rows x 67 columns (66 features + Label).
    Labels: Brute_Force 88,502 (68.71%), Normal 28,502 (22.13%), Port_Scan
    11,081 (8.60%), HTTP_DDoS 641 (0.50%), ICMP_Flood 45 (0.03%), Web_Crwling
    28 (0.02%).  37.4% of rows share their feature vector with another row.
    A quick stratified 80/20 check (seed 42): majority class 0.6872 accuracy;
    a depth-3 tree 0.7816 accuracy and 0.5457 macro-F1; a random forest 0.9981
    accuracy and 0.8542 macro-F1.
    Our own PCAP extraction had roughly 2,779 normal flows carrying
    Benign/Normal and Brute_Force/DoS label conflicts, and a random forest
    reached only 0.5836 accuracy on it.
    Those numbers were measured BEFORE this module existed; Step 3 prints what
    it finds beside them and says whether the two agree.

What this module does, in order:

1. READ.  `MSCAD.csv`, with the column names cleaned: surrounding whitespace
   and quotes stripped, so `"'Idle Mean'"` becomes `Idle Mean`.  Both the raw
   and the cleaned name are reported, so a reader can see what was changed.

2. MAP.  Each cleaned column is matched to this project's own feature names by
   NORMALISED comparison (lowercased, punctuation and spacing removed), so
   `Flow Byts/s` and `flow_bytes_per_s` land on each other.  Columns that
   match nothing on either side are reported -- on BOTH sides, because a
   feature this project expects and the official file does not have matters
   just as much as the other way round.

3. SEPARATE THE IDENTIFIERS.  Any of Flow ID, Src IP, Dst IP, Src Port, Dst
   Port, Timestamp that the file has is moved to metadata and excluded from
   the features.  A port number would let a model learn the target port
   instead of the behaviour, and an IP would let it learn the attacker's
   address.  Which were found -- and which were not -- is reported, and the
   missing ones are recorded as unavailable so later steps can skip the parts
   that need them instead of inventing a value.

4. REPORT.  Shape; label counts and shares; inf and NaN per column; constant
   columns; duplicate feature rows overall and per class; and LABEL CONFLICTS
   -- identical feature vectors carrying more than one label, counted and
   listed.  A conflict is a ceiling on accuracy that no model can pass, so it
   is named here rather than discovered in Step 11.

5. SAVE.  `outputs_official\\step3\\MSCAD_merged.csv` in the layout Step 4
   expects, plus the report files beside it.  Nothing is written anywhere near
   `outputs\\`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from .data_loading import kv_table, show_note, show_table, show_title  # noqa: E402

OK, FAIL, WARN, SKIP = "[OK]", "[FAIL]", "[WARN]", "[SKIP]"
ROW_COLUMN = "Orig_Row"


class OfficialDatasetError(Exception):
    """Raised when the official file cannot be read or is not what it claims."""


# ===========================================================================
# 1. READ -- and clean the header
# ===========================================================================
def clean_column_name(name: Any) -> str:
    """`"'Idle Mean'"` -> `Idle Mean`.

    Surrounding whitespace and quote characters only: nothing inside the name
    is touched, so `Flow Byts/s` keeps its slash and `Fwd IAT Min` keeps its
    spacing.  The official header carries single quotes around some names, and
    a column called `'Idle Mean'` would silently never match anything.
    """
    text = str(name).strip()
    while text and text[0] in "'\"`" and text[-1] in "'\"`" and len(text) > 1:
        text = text[1:-1].strip()
    return text.strip()


# The few abbreviations the CICFlowMeter family spells both ways.  Expanded
# before comparison so `Tot Fwd Pkts` and `Total Forward Packets` land on the
# same key; anything beyond this list is a real difference in vocabulary and is
# reported as unmatched rather than guessed at.
_ABBREVIATIONS = (
    ("totlen", "totallength"), ("tot", "total"), ("len", "length"),
    ("pkts", "packets"), ("pkt", "packet"), ("byts", "bytes"),
    ("byt", "byte"), ("fwd", "forward"), ("bwd", "backward"),
    ("cnt", "count"), ("seg", "segment"), ("init", "initial"),
    ("win", "window"), ("avg", "average"), ("std", "stddev"),
    ("iat", "interarrivaltime"), ("blk", "bulk"), ("act", "active"),
)


def normalise_name(name: str) -> str:
    """The form two names are compared in: letters and digits only, lowercased.

    `FLOW BYTS / S`, `Flow Byts/s` and `flow_byts_per_s` all normalise to the
    same key, and the abbreviation table above also makes `Tot Fwd Pkts` match
    `Total Forward Packets`. That is what lets the official header be matched
    against this project's own feature names without a hand-written table of 66
    pairs -- and what is left unmatched is reported on both sides rather than
    forced together.
    """
    text = clean_column_name(name).lower()
    text = text.replace("/s", " per s").replace("/", " per ")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    words = []
    for word in text.split():
        for short, long in _ABBREVIATIONS:
            if word == short:
                word = long
                break
        words.append(word)
    return "".join(words)


def read_official_csv(path: Path | str | None = None, verbose: bool = True
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read MSCAD.csv and return (frame with cleaned names, the rename table)."""
    path = Path(config.OFFICIAL_CSV if path is None else path)
    if not path.exists():
        raise OfficialDatasetError(
            "the official dataset is not where config says it is:\n    %s\n"
            "Put MSCAD.csv there, or set MSCAD_OFFICIAL_CSV_DIR to the folder "
            "that holds it." % path)
    frame = pd.read_csv(path, low_memory=False)
    rows = []
    cleaned = []
    for raw in frame.columns:
        name = clean_column_name(raw)
        rows.append({"raw column": str(raw), "cleaned column": name,
                     "changed": str(raw) != name})
        cleaned.append(name)
    if len(set(cleaned)) != len(cleaned):
        counts = pd.Series(cleaned).value_counts()
        raise OfficialDatasetError(
            "cleaning the header made two columns share a name: %s. Nothing "
            "was changed on disk; the file needs looking at."
            % ", ".join(counts[counts > 1].index))
    frame.columns = cleaned
    table = pd.DataFrame(rows)
    if verbose:
        show_title("1.1  The official file, as read", level=4)
        show_table(kv_table([
            ("path", str(path)),
            ("size (MB)", "%.1f" % (path.stat().st_size / 1024 / 1024)),
            ("rows", f"{len(frame):,}"),
            ("columns", frame.shape[1]),
            ("column names cleaned", int(table["changed"].sum())),
        ]), note="Reading: the shape is the authors' file as it stands; "
                 "`column names cleaned` counts headers that carried "
                 "surrounding quotes or whitespace, for example \"'Idle "
                 "Mean'\" -> \"Idle Mean\".")
        changed = table[table["changed"]]
        if len(changed):
            show_table(changed.head(20), all_columns=True,
                       note="Reading: the raw name on the left is what the "
                            "file has; the cleaned name on the right is what "
                            "every later step uses. Nothing inside a name was "
                            "altered.")
    return frame, table


# ===========================================================================
# 2. MAP -- the official header against this project's own feature names
# ===========================================================================
def project_feature_names() -> list[str]:
    """The feature names this project's own extraction produces."""
    names = [c for c in getattr(config, "FLOW_COLUMNS", [])
             if c not in set(getattr(config, "COMBINED_ID_COLUMNS", []))
             and c != config.LABEL_COLUMN]
    return names


def map_columns(frame: pd.DataFrame, verbose: bool = True) -> dict[str, Any]:
    """Match the official columns to this project's names, and report both sides.

    Nothing is renamed: the official names are kept, because they are what the
    authors published and what a reader of the thesis will look for.  The map
    exists so the two vocabularies can be compared -- and so a feature this
    project expects but the official file lacks is reported rather than
    silently missing.
    """
    official = [c for c in frame.columns]
    ours = project_feature_names()
    ours_by_key = {normalise_name(name): name for name in ours}
    official_by_key = {normalise_name(name): name for name in official}

    rows = []
    for name in official:
        key = normalise_name(name)
        match = ours_by_key.get(key)
        rows.append({"official column": name, "normalised": key,
                     "our feature name": match or "",
                     "matched": match is not None})
    table = pd.DataFrame(rows)
    unmatched_official = [r["official column"] for r in rows if not r["matched"]]
    unmatched_ours = [name for key, name in ours_by_key.items()
                      if key not in official_by_key]

    if verbose:
        show_title("1.2  Column mapping", level=4)
        show_table(kv_table([
            ("official columns", len(official)),
            ("our own feature names", len(ours)),
            ("matched by normalised name", int(table["matched"].sum())),
            ("in the official file only", len(unmatched_official)),
            ("in our extraction only", len(unmatched_ours)),
        ]), note="Reading: matching is by normalised name (lowercased, "
                 "punctuation removed), so `Flow Byts/s` and "
                 "`flow_bytes_per_s` match. The official names are KEPT -- "
                 "the map exists to compare the two vocabularies, not to "
                 "rename anything.")
        if unmatched_official:
            show_note("**In the official file only (%d):** %s"
                      % (len(unmatched_official),
                         ", ".join(unmatched_official)))
        if unmatched_ours:
            show_note("**In our extraction only (%d):** %s -- the official "
                      "file does not carry these, so nothing downstream may "
                      "assume them."
                      % (len(unmatched_ours), ", ".join(unmatched_ours)))
    return {"table": table, "unmatched_official": unmatched_official,
            "unmatched_ours": unmatched_ours,
            "matched": int(table["matched"].sum())}


# ===========================================================================
# 3. SEPARATE THE IDENTIFIERS
# ===========================================================================
def find_identifiers(frame: pd.DataFrame, verbose: bool = True
                     ) -> dict[str, Any]:
    """Which identifier columns the file has, and which it does not.

    IDENTIFIERS ARE NEVER FEATURES.  A model given Dst Port learns the target
    port rather than the behaviour that reaches it, and one given Src IP learns
    which machine the attacker used -- neither survives contact with a
    different network.
    """
    wanted = list(config.OFFICIAL_IDENTIFIER_COLUMNS)
    by_key = {normalise_name(c): c for c in frame.columns}
    found: dict[str, str] = {}
    for name in wanted:
        actual = by_key.get(normalise_name(name))
        if actual is not None:
            found[name] = actual
    missing = [name for name in wanted if name not in found]

    table = pd.DataFrame([
        {"identifier": name,
         "present in the official file": name in found,
         "column in the file": found.get(name, ""),
         "status": ("moved to metadata, excluded from the features"
                    if name in found else "unavailable")}
        for name in wanted])
    if verbose:
        show_title("1.3  Identifiers", level=4)
        show_table(table, all_columns=True,
                   note="Reading: every identifier the file has is moved to "
                        "metadata and kept out of the features -- a port "
                        "number would let a model learn the target port "
                        "instead of the behaviour. An `unavailable` row means "
                        "the official file does not carry it, so later steps "
                        "skip the parts that need it instead of inventing a "
                        "value.")
        if missing:
            show_note("**Unavailable identifiers:** %s. Step 5's per-capture "
                      "reporting, Step 11's alert context and Step 14's "
                      "same-source grouping need these; they are skipped with "
                      "a printed note rather than estimated."
                      % ", ".join(missing))
    return {"found": found, "missing": missing, "table": table}


# ===========================================================================
# 4. REPORT -- what is actually in the file
# ===========================================================================
def label_table(frame: pd.DataFrame, verbose: bool = True) -> dict[str, Any]:
    """Label counts and shares, beside the counts the authors published."""
    column = config.LABEL_COLUMN
    if column not in frame.columns:
        raise OfficialDatasetError(
            "the official file has no %r column; its header is: %s"
            % (column, ", ".join(frame.columns[:12])))
    counts = frame[column].astype(str).value_counts()
    total = int(counts.sum())
    expected = dict(config.OFFICIAL_EXPECTED_LABELS)
    rows = []
    for name, count in counts.items():
        want = expected.get(name)
        rows.append({"label": name, "rows": int(count),
                     "share (%)": round(100.0 * count / total, 2),
                     "published count": want if want is not None else "",
                     "result": (SKIP if want is None
                                else OK if int(count) == int(want) else FAIL)})
    table = pd.DataFrame(rows)
    unexpected = [name for name in counts.index if name not in expected]
    absent = [name for name in expected if name not in set(counts.index)]
    if verbose:
        show_title("1.4  Labels", level=4)
        show_table(table, all_columns=True,
                   note="Reading: `rows` and `share (%%)` are counted from the "
                        "file; `published count` is what the authors reported, "
                        "and `result` says whether the two agree. The labels "
                        "are kept exactly as written, Web_Crwling included -- "
                        "renaming an author's class would make this dataset "
                        "uncomparable with their paper.")
        if unexpected:
            show_note("**Labels in the file that the published list does not "
                      "have:** %s" % ", ".join(unexpected))
        if absent:
            show_note("**Published labels missing from the file:** %s"
                      % ", ".join(absent))
    return {"table": table, "counts": counts.to_dict(),
            "unexpected": unexpected, "absent": absent,
            "classes": sorted(counts.index)}


def column_quality(frame: pd.DataFrame, feature_columns: Sequence[str],
                   verbose: bool = True) -> dict[str, Any]:
    """inf, NaN and constant columns, per column, before anything is changed."""
    rows = []
    for name in feature_columns:
        series = frame[name]
        numeric = pd.to_numeric(series, errors="coerce")
        infinite = int(np.isinf(numeric.to_numpy(dtype="float64",
                                                 na_value=np.nan)).sum())
        missing = int(series.isna().sum())
        distinct = int(series.nunique(dropna=True))
        rows.append({"column": name, "dtype": str(series.dtype),
                     "inf": infinite, "NaN": missing,
                     "distinct values": distinct,
                     "constant": distinct <= 1,
                     "non-numeric": bool(numeric.isna().sum() > missing)})
    table = pd.DataFrame(rows)
    constant = list(table.loc[table["constant"], "column"])
    with_inf = list(table.loc[table["inf"] > 0, "column"])
    with_nan = list(table.loc[table["NaN"] > 0, "column"])
    non_numeric = list(table.loc[table["non-numeric"], "column"])
    if verbose:
        show_title("1.5  Column quality", level=4)
        show_table(kv_table([
            ("feature columns", len(feature_columns)),
            ("columns with inf", "%d: %s" % (len(with_inf),
                                             ", ".join(with_inf) or "none")),
            ("columns with NaN", "%d: %s" % (len(with_nan),
                                             ", ".join(with_nan) or "none")),
            ("constant columns", "%d: %s" % (len(constant),
                                             ", ".join(constant) or "none")),
            ("non-numeric columns", "%d: %s" % (len(non_numeric),
                                                ", ".join(non_numeric)
                                                or "none")),
        ]), note="Reading: counted on the file as read, before any cleaning. "
                 "Step 4 caps the infinities and drops the constant columns; "
                 "naming them here means that is a decision rather than a "
                 "side effect.")
    return {"table": table, "constant": constant, "with_inf": with_inf,
            "with_nan": with_nan, "non_numeric": non_numeric}


def duplicate_report(frame: pd.DataFrame, feature_columns: Sequence[str],
                     verbose: bool = True) -> dict[str, Any]:
    """Rows whose feature vector occurs more than once, overall and per class."""
    features = frame[list(feature_columns)]
    duplicated = features.duplicated(keep=False)
    total = int(len(frame))
    share = float(duplicated.mean()) if total else 0.0
    per_class = []
    for name, part in frame.groupby(frame[config.LABEL_COLUMN].astype(str)):
        mask = duplicated.loc[part.index]
        per_class.append({"label": name, "rows": int(len(part)),
                          "rows sharing a feature vector": int(mask.sum()),
                          "share (%)": round(100.0 * float(mask.mean()), 2)})
    table = pd.DataFrame(per_class).sort_values(
        "rows", ascending=False).reset_index(drop=True)
    published = float(getattr(config, "OFFICIAL_EXPECTED_DUPLICATE_SHARE", 0.0))
    if verbose:
        show_title("1.6  Duplicate feature vectors", level=4)
        show_table(kv_table([
            ("rows", f"{total:,}"),
            ("rows sharing a feature vector with another row",
             "%s (%.1f%%)" % (f"{int(duplicated.sum()):,}", 100.0 * share)),
            ("published share", "%.1f%%" % (100.0 * published)),
            ("result", OK if abs(share - published) <= 0.01 else WARN),
        ]), note="Reading: a duplicate here means two rows have identical "
                 "FEATURE values; they may carry the same label or different "
                 "ones. Because the split is random over rows, a test row can "
                 "be an exact copy of a training row -- which is why every "
                 "later step also reports a duplicate-free figure.")
        show_table(table, all_columns=True,
                   note="Reading: per class, how much of it is made of "
                        "repeated feature vectors. A class that is mostly "
                        "duplicates is easier to score than its row count "
                        "suggests.")
    return {"table": table, "mask": duplicated, "share": share,
            "count": int(duplicated.sum()),
            "matches_published": bool(abs(share - published) <= 0.01)}


def label_conflicts(frame: pd.DataFrame, feature_columns: Sequence[str],
                    verbose: bool = True, list_limit: int = 50
                    ) -> dict[str, Any]:
    """Identical feature vectors carrying MORE THAN ONE label.

    This is the hard ceiling on the dataset: two rows with the same features
    and different labels cannot both be classified correctly by any model that
    sees only those features.  Counted and listed, because a reader comparing
    our macro-F1 with the authors' needs to know the ceiling exists.
    """
    columns = list(feature_columns)
    labels = frame[config.LABEL_COLUMN].astype(str)
    grouped = frame.groupby(columns, dropna=False, sort=False)[
        config.LABEL_COLUMN]
    distinct = grouped.transform(lambda s: s.astype(str).nunique())
    conflicting = distinct > 1

    rows = int(conflicting.sum())
    groups = 0
    summary = pd.DataFrame()
    if rows:
        part = frame.loc[conflicting, columns + [config.LABEL_COLUMN]].copy()
        part["_label"] = labels.loc[conflicting]
        keys = part.groupby(columns, dropna=False, sort=False)
        groups = int(keys.ngroups)
        combos = keys["_label"].apply(
            lambda s: " + ".join(sorted(set(s.astype(str)))))
        counts = keys.size()
        summary = pd.DataFrame({
            "labels in conflict": combos.to_numpy(),
            "rows": counts.to_numpy()})
        summary = (summary.groupby("labels in conflict", as_index=False)
                   .agg(**{"feature vectors": ("rows", "size"),
                           "rows": ("rows", "sum")})
                   .sort_values("rows", ascending=False)
                   .reset_index(drop=True))
    if verbose:
        show_title("1.7  Label conflicts", level=4)
        show_table(kv_table([
            ("feature vectors carrying more than one label", f"{groups:,}"),
            ("rows involved", "%s (%.2f%% of the dataset)"
             % (f"{rows:,}", 100.0 * rows / max(len(frame), 1))),
            ("result", OK if not rows else WARN),
        ]), note="Reading: two rows with identical features and different "
                 "labels cannot both be classified correctly by any model "
                 "that sees only those features, so this is a ceiling on "
                 "accuracy rather than a modelling problem. 0 rows means the "
                 "features separate the classes at least in principle.")
        if len(summary):
            show_table(summary.head(list_limit), all_columns=True,
                       note="Reading: which label pairs collide, how many "
                            "distinct feature vectors each pair covers, and "
                            "how many rows that is.")
    return {"summary": summary, "rows": rows, "groups": groups,
            "mask": conflicting}


# ===========================================================================
# 5. SAVE -- the layout Step 4 expects
# ===========================================================================
def build_merged(frame: pd.DataFrame, identifiers: dict[str, Any],
                 verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """The frame Step 4 reads: features + identifiers as metadata + Label.

    `Orig_Row` is added here (0..n-1, the row's position in MSCAD.csv) and is
    metadata, never a feature: the official file is grouped by class, so a
    model given the row number would read the label straight off it.
    """
    out = frame.copy()
    if ROW_COLUMN in out.columns:
        raise OfficialDatasetError(
            "the official file already has an %r column; Step 1 would "
            "overwrite it" % ROW_COLUMN)
    out[ROW_COLUMN] = np.arange(len(out), dtype="int64")

    identifier_columns = list(identifiers["found"].values())
    feature_columns = [c for c in out.columns
                       if c != config.LABEL_COLUMN
                       and c != ROW_COLUMN
                       and c not in identifier_columns]
    ordered = feature_columns + identifier_columns + [ROW_COLUMN,
                                                      config.LABEL_COLUMN]
    out = out[ordered]
    if verbose:
        show_table(kv_table([
            ("feature columns", len(feature_columns)),
            ("identifier columns (metadata)", len(identifier_columns)),
            ("row column", ROW_COLUMN),
            ("label column", config.LABEL_COLUMN),
            ("total columns", out.shape[1]),
        ]), note="Reading: the layout Step 4 reads -- features first, then the "
                 "identifiers it must keep out of the features, then Orig_Row "
                 "and Label. config.METADATA_COLUMNS lists all of them, so "
                 "Step 4 moves them aside without being told again.")
    return out, feature_columns


def folder() -> Path | None:
    return config.usable_output_dir(config.STEP3_OUTPUT_DIR)


def is_done() -> bool:
    place = folder()
    if place is None:
        return False
    return (place / config.OFFICIAL_STEP3_FILES["merged"]).exists()


def save_step3(out: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Write MSCAD_merged.csv and the report files into outputs_official\\step3."""
    from .data_loading import prepare_output_dir

    names = config.OFFICIAL_STEP3_FILES
    place = prepare_output_dir(config.STEP3_OUTPUT_DIR,
                               "the official Step 1 outputs", verbose=verbose)
    if place is None:
        return {"status": "skipped_not_windows", "folder": None}
    place = Path(place)
    out["merged"].to_csv(place / names["merged"], index=False)
    for key, frame in (("report", out["report"]),
                       ("columns", out["columns"]["table"]),
                       ("labels", out["labels"]["table"]),
                       ("quality", out["quality"]["table"]),
                       ("duplicates", out["duplicates"]["table"]),
                       ("conflicts", out["conflicts"]["summary"])):
        if frame is not None and len(frame):
            frame.to_csv(place / names[key], index=False)
    (place / names["manifest"]).write_text(
        json.dumps(out["manifest"], indent=2, default=str), encoding="utf-8")
    (place / names["summary"]).write_text(out["summary"], encoding="utf-8")
    if verbose:
        print("  [SAVE]  %s" % (place / names["merged"]))
    return {"status": "written", "folder": str(place),
            "merged": str(place / names["merged"])}


def load_merged_official(verbose: bool = True) -> pd.DataFrame:
    """Read the official MSCAD_merged.csv back, using the manifest's dtypes.

    `data_loading.load_merged` delegates here in official mode: the own
    extraction's dtype map and its `Timestamp_dt` parsing do not apply to this
    file, and guessing would be worse than reading the manifest Step 3 wrote.
    """
    place = folder()
    names = config.OFFICIAL_STEP3_FILES
    if place is None:
        raise OfficialDatasetError(
            "the official Step 1 folder is not usable here -- %s"
            % config.why_unusable(config.STEP3_OUTPUT_DIR))
    path = place / names["merged"]
    if not path.exists():
        raise OfficialDatasetError(
            "Step 2 (preprocessing) needs the file Step 1 (data acquisition) "
            "writes, and it is not there:\n"
            "    %s\n"
            "Run the Step 1 cells first (official_dataset.run_step3 -- the "
            "function keeps its old name, see the numbering note at the top of "
            "the notebook). Nothing was read and nothing was written."
            % path)
    dtypes: dict[str, Any] = {}
    manifest_path = place / names["manifest"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in manifest.get("identifier_columns", []):
            dtypes[name] = "string"
        dtypes[config.LABEL_COLUMN] = "string"
        dtypes[ROW_COLUMN] = "int64"
    frame = pd.read_csv(path, dtype=dtypes or None, low_memory=False)
    if verbose:
        print("  loaded %s from %s" % ((frame.shape,), path))
    return frame


# ===========================================================================
# the summary markdown -- written from the computed numbers only
# ===========================================================================
def build_summary(out: dict[str, Any]) -> str:
    labels = out["labels"]["table"]
    conflicts = out["conflicts"]
    duplicates = out["duplicates"]
    identifiers = out["identifiers"]
    lines = ["## Step 1 (data acquisition) -- what is in the file", ""]
    lines.append("Read from `%s`: **%s rows x %d columns**, **%d feature "
                 "columns** after the identifiers were separated."
                 % (out["manifest"]["path"],
                    f"{out['manifest']['rows']:,}",
                    out["manifest"]["columns_in"],
                    out["manifest"]["feature_count"]))
    lines.append("")
    lines.append("### Labels")
    lines.append("")
    for _i, row in labels.iterrows():
        agree = ("" if row["result"] == SKIP else
                 " (published %s, %s)" % (f"{int(row['published count']):,}",
                                          "agrees" if row["result"] == OK
                                          else "**differs**"))
        lines.append("- **%s** - %s rows, %.2f%%%s"
                     % (row["label"], f"{int(row['rows']):,}",
                        float(row["share (%)"]), agree))
    lines.append("")
    lines.append("### Identifiers")
    lines.append("")
    found = list(identifiers["found"])
    missing = identifiers["missing"]
    lines.append("Found and moved to metadata: **%s**."
                 % (", ".join(found) if found else "none"))
    if missing:
        lines.append("")
        lines.append("Unavailable: **%s**. Later steps skip the parts that "
                     "need them -- per-capture reporting, alert context and "
                     "same-source grouping -- rather than inventing a value."
                     % ", ".join(missing))
    lines.append("")
    lines.append("### Duplicates and label conflicts")
    lines.append("")
    lines.append("**%s rows (%.1f%%)** share their feature vector with another "
                 "row%s. Because the split is random over rows, a test flow "
                 "can be an exact copy of a training flow, so every later step "
                 "also reports a duplicate-free figure."
                 % (f"{duplicates['count']:,}", 100.0 * duplicates["share"],
                    "" if not duplicates["matches_published"] else
                    ", which matches the published 37.4%"))
    lines.append("")
    if conflicts["rows"]:
        lines.append("**%s feature vectors carry more than one label**, "
                     "covering **%s rows (%.2f%%)**. Two rows with identical "
                     "features and different labels cannot both be classified "
                     "correctly by any model that sees only those features, so "
                     "this is a ceiling on accuracy rather than a modelling "
                     "problem, and it bounds every number in Steps 10-14."
                     % (f"{conflicts['groups']:,}", f"{conflicts['rows']:,}",
                        100.0 * conflicts["rows"]
                        / max(out["manifest"]["rows"], 1)))
    else:
        lines.append("**No feature vector carries more than one label**, so "
                     "the features separate the six classes at least in "
                     "principle. That is the difference from our own "
                     "extraction, where roughly 2,779 normal flows carried "
                     "Benign/Normal and Brute_Force/DoS conflicts.")
    return "\n".join(lines)


def _is_inside(path: Path | str, parent: Path | str) -> bool:
    """Is `path` the same folder as `parent`, or inside it?

    By path PARTS rather than by text: `outputs` is a prefix of
    `outputs_official`, so `str(parent) in str(path)` answers this question
    wrongly in exactly the case that matters here.
    """
    path_parts = Path(str(path).replace("\\", "/")).parts
    parent_parts = Path(str(parent).replace("\\", "/")).parts
    return path_parts[:len(parent_parts)] == parent_parts


def build_report(out: dict[str, Any]) -> pd.DataFrame:
    """One verification table for the whole step."""
    manifest = out["manifest"]
    expected_rows = int(getattr(config, "OFFICIAL_EXPECTED_ROWS", 0))
    expected_columns = int(getattr(config, "OFFICIAL_EXPECTED_COLUMNS", 0))
    labels = out["labels"]
    rows = [
        {"check": "the file was read from the configured path",
         "expected": str(config.OFFICIAL_CSV), "actual": manifest["path"],
         "result": OK if manifest["path"] == str(config.OFFICIAL_CSV) else WARN},
        {"check": "row count matches the published figure",
         "expected": f"{expected_rows:,}", "actual": f"{manifest['rows']:,}",
         "result": OK if manifest["rows"] == expected_rows else FAIL},
        {"check": "column count matches the published figure",
         "expected": f"{expected_columns:,}",
         "actual": f"{manifest['columns_in']:,}",
         "result": OK if manifest["columns_in"] == expected_columns else FAIL},
        {"check": "the six published classes are present",
         "expected": ", ".join(config.OFFICIAL_LABELS),
         "actual": ", ".join(labels["classes"]),
         "result": OK if set(labels["classes"]) == set(config.OFFICIAL_LABELS)
         else FAIL},
        {"check": "every label count matches the published figure",
         "expected": "6 of 6",
         "actual": "%d of %d" % (int((labels["table"]["result"] == OK).sum()),
                                 len(labels["table"])),
         "result": OK if not (labels["table"]["result"] == FAIL).any() else FAIL},
        {"check": "no identifier column reaches the features",
         "expected": "0 identifier columns in the features",
         "actual": "%d of %d identifiers found, all in metadata"
                   % (len(out["identifiers"]["found"]),
                      len(config.OFFICIAL_IDENTIFIER_COLUMNS)),
         "result": OK if not (set(out["identifiers"]["found"].values())
                              & set(manifest["feature_columns"])) else FAIL},
        {"check": "Orig_Row is metadata, not a feature",
         "expected": "%s not in the features" % ROW_COLUMN,
         "actual": "%s in the features: %s"
                   % (ROW_COLUMN, ROW_COLUMN in manifest["feature_columns"]),
         "result": OK if ROW_COLUMN not in manifest["feature_columns"] else FAIL},
        {"check": "the benign set for Label_binary is Normal only",
         "expected": "['Normal']", "actual": str(list(config.BENIGN_LABELS)),
         "result": OK if list(config.BENIGN_LABELS) == ["Normal"] else FAIL},
        {"check": "duplicate share matches the published figure",
         "expected": "%.1f%%" % (100.0 * float(
             getattr(config, "OFFICIAL_EXPECTED_DUPLICATE_SHARE", 0.0))),
         "actual": "%.1f%%" % (100.0 * out["duplicates"]["share"]),
         "result": OK if out["duplicates"]["matches_published"] else WARN},
        {"check": "label conflicts (identical features, different labels)",
         "expected": "0 rows", "actual": "%s rows in %s feature vectors"
                                         % (f"{out['conflicts']['rows']:,}",
                                            f"{out['conflicts']['groups']:,}"),
         "result": OK if not out["conflicts"]["rows"] else WARN},
        {"check": "the outputs went to the official root, not the own one",
         "expected": "under %s" % config.OUTPUTS_ROOT,
         "actual": str(config.STEP3_OUTPUT_DIR),
         # compared by PATH PARTS, not as text: "outputs" is a substring of
         # "outputs_official", so a text test would say the official root was
         # the own one
         "result": OK if _is_inside(config.STEP3_OUTPUT_DIR,
                                    config.OUTPUTS_ROOT)
         and not _is_inside(config.STEP3_OUTPUT_DIR, config.OUTPUTS_ROOT_OWN)
         else FAIL},
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# STEP 3 end to end
# ===========================================================================
def run_step3(path: Path | str | None = None, overwrite: bool | None = None,
              verbose: bool = True) -> dict[str, Any]:
    """STEP 3 (official): read, map, separate, report, save, verify."""
    overwrite = (getattr(config, "OVERWRITE_MERGED", False)
                 if overwrite is None else overwrite)
    if not config.RUNNING_OFFICIAL:
        raise OfficialDatasetError(
            "this is the OFFICIAL Step 3, but DATASET_SOURCE is %r. Set "
            "DATASET_SOURCE = \"official\" in config.py, or run the own "
            "extraction's Step 3 (data_loading.save_merged)."
            % config.DATASET_SOURCE)

    if verbose:
        show_title("Step 1 - Data acquisition: the MSCAD dataset as published",
                   level=3)
        print(config.SKIP_STEPS_NOTE)
        print()

    if not overwrite and is_done():
        if verbose:
            show_note("`%s` already exists and `OVERWRITE_MERGED` is False, so "
                      "Step 1 is reading it back rather than rebuilding it."
                      % (folder() / config.OFFICIAL_STEP3_FILES["merged"]))
        merged = load_merged_official(verbose=verbose)
        return {"status": "loaded_existing", "merged": merged,
                "folder": str(folder())}

    frame, renames = read_official_csv(path, verbose=verbose)
    columns = map_columns(frame, verbose=verbose)
    identifiers = find_identifiers(frame, verbose=verbose)
    labels = label_table(frame, verbose=verbose)
    merged, feature_columns = build_merged(frame, identifiers, verbose=verbose)
    quality = column_quality(merged, feature_columns, verbose=verbose)
    duplicates = duplicate_report(merged, feature_columns, verbose=verbose)
    conflicts = label_conflicts(merged, feature_columns, verbose=verbose)

    manifest = {
        "dataset_source": config.DATASET_SOURCE,
        "path": str(Path(config.OFFICIAL_CSV if path is None else path)),
        "rows": int(len(frame)),
        "columns_in": int(frame.shape[1]),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "identifier_columns": list(identifiers["found"].values()),
        "identifiers_found": list(identifiers["found"]),
        "identifiers_unavailable": identifiers["missing"],
        "row_column": ROW_COLUMN,
        "label_column": config.LABEL_COLUMN,
        "classes": labels["classes"],
        "benign_labels": list(config.BENIGN_LABELS),
        "renamed_columns": int(renames["changed"].sum()),
        "unmatched_official": columns["unmatched_official"],
        "unmatched_ours": columns["unmatched_ours"],
        "constant_columns": quality["constant"],
        "duplicate_rows": duplicates["count"],
        "duplicate_share": duplicates["share"],
        "label_conflict_rows": conflicts["rows"],
        "label_conflict_groups": conflicts["groups"],
    }

    out: dict[str, Any] = {
        "status": "written", "merged": merged, "renames": renames,
        "columns": columns, "identifiers": identifiers, "labels": labels,
        "quality": quality, "duplicates": duplicates, "conflicts": conflicts,
        "feature_columns": feature_columns, "manifest": manifest,
    }
    out["report"] = build_report(out)
    out["summary"] = build_summary(out)
    saved = save_step3(out, verbose=verbose)
    out["saved"] = saved
    out["folder"] = saved.get("folder")
    out["ok"] = bool(not (out["report"]["result"] == FAIL).any())

    if verbose:
        show_title("1.8  Step 1 verification", level=4)
        show_table(out["report"], all_columns=True,
                   note="Reading: every check this step can make, in one "
                        "table. [OK] passed; [WARN] is a property of the "
                        "dataset to carry forward (duplicates, label "
                        "conflicts); [FAIL] means the file is not what the "
                        "published figures say and nothing downstream should "
                        "be trusted until it is explained.")
        print("[%s]  Step 1 (data acquisition): %d checks, %d failed"
              % ("OK" if out["ok"] else "FAIL", len(out["report"]),
                 int((out["report"]["result"] == FAIL).sum())))
    return out


# ===========================================================================
# THE END-OF-PART SUMMARY  (printed at the end of Step 11)
# ===========================================================================
def _read_json(path: Path) -> dict[str, Any] | None:
    return (json.loads(path.read_text(encoding="utf-8")) if path.exists()
            else None)


def _label_names() -> dict[int, str]:
    """encoded -> Label, from the Step 4 mapping that Step 4 itself wrote."""
    place = config.usable_output_dir(config.STEP4_OUTPUT_DIR)
    if place is None:
        return {}
    payload = _read_json(Path(place) / config.STEP4_FILES["label_mapping"])
    if not payload:
        return {}
    mapping = payload.get("label_mapping", payload)
    out: dict[int, str] = {}
    for key, value in mapping.items():
        try:                                   # {"Brute_Force": 0, ...}
            out[int(value)] = str(key)
        except (TypeError, ValueError):        # {"0": "Brute_Force", ...}
            try:
                out[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    return out


def part1_summary(verbose: bool = True) -> dict[str, Any]:
    """Steps 3 to 11 on the official dataset, in one block, from the saved files.

    Every number here is READ from the active output root -- nothing is
    recomputed and nothing is estimated.  A file that is not there is reported
    as not available rather than filled in, so the summary can never be ahead
    of the run.
    """
    lines: list[str] = []
    out: dict[str, Any] = {}

    def add(text: str = "") -> None:
        lines.append(text)

    root = config.OUTPUTS_ROOT
    add("## Summary -- the official MSCAD dataset, Steps 1 to 9")
    add("")
    add("Dataset source: **%s**; output root `%s`."
        % (config.DATASET_SOURCE, root))
    add("")

    # ---- Step 3 ---------------------------------------------------------
    place3 = config.usable_output_dir(config.STEP3_OUTPUT_DIR)
    manifest = (_read_json(Path(place3) / config.OFFICIAL_STEP3_FILES["manifest"])
                if place3 is not None else None)
    out["step3"] = manifest
    add("### Step 1 -- the file as published")
    add("")
    if manifest is None:
        add("- Not available: Step 1 has not been run in this output root.")
    else:
        add("- **%s rows x %d columns**, %d feature columns after the "
            "identifiers were separated."
            % (f"{manifest['rows']:,}", manifest["columns_in"],
               manifest["feature_count"]))
        add("- Classes (%d): %s."
            % (len(manifest["classes"]), ", ".join(manifest["classes"])))
        add("- Benign set for `Label_binary`: **%s**."
            % ", ".join(manifest["benign_labels"]))
        found = manifest["identifiers_found"]
        missing = manifest["identifiers_unavailable"]
        add("- Identifiers found and moved to metadata: **%s**."
            % (", ".join(found) if found else "none"))
        if missing:
            add("- Identifiers unavailable: **%s** -- the parts of later steps "
                "that need them are skipped with a note."
                % ", ".join(missing))
        add("- Duplicate feature vectors: **%s rows (%.1f%%)**."
            % (f"{manifest['duplicate_rows']:,}",
               100.0 * float(manifest["duplicate_share"])))
        if manifest["label_conflict_rows"]:
            add("- **Label conflicts: %s rows in %s feature vectors** carry "
                "more than one label. No model that sees only these features "
                "can classify all of them correctly, so this bounds every "
                "number below."
                % (f"{manifest['label_conflict_rows']:,}",
                   f"{manifest['label_conflict_groups']:,}"))
        else:
            add("- **No label conflicts**: no feature vector carries more than "
                "one label, so the features separate the six classes at least "
                "in principle. Our own extraction did not have this property.")
        if manifest["constant_columns"]:
            add("- Constant columns (dropped by Step 4): %s."
                % ", ".join(manifest["constant_columns"]))
    add("")

    # ---- Step 6 ---------------------------------------------------------
    place6 = config.usable_output_dir(config.STEP6_OUTPUT_DIR)
    selected = None
    if place6 is not None:
        path = Path(place6) / config.STEP6_FILES.get("selected",
                                                     "selected_features.json")
        payload = _read_json(path)
        if payload is not None:
            selected = payload.get("features") or payload.get("selected")
    out["step6_selected"] = selected
    add("### Step 4 -- the selected features")
    add("")
    if not selected:
        add("- Not available: Step 4 has not been run in this output root.")
    else:
        add("**%d features** (MI + Pearson + permutation importance, training "
            "rows only):" % len(selected))
        add("")
        add(", ".join("`%s`" % name for name in selected))
    add("")

    # ---- Step 9 ---------------------------------------------------------
    place9 = config.usable_output_dir(config.STEP9_OUTPUT_DIR)
    minimum = int(getattr(config, "RARE_CLASS_VALIDATION_MINIMUM", 5))
    counts: dict[str, int] = {}
    if place9 is not None:
        # counted from the SAVED validation labels, not from the weight file:
        # class_weights.json records fit rows and weights, not validation rows,
        # and a key that is not there must not be read as a zero
        y_val_path = Path(place9) / config.STEP9_FILES["y_val"]
        if y_val_path.exists():
            y_val = pd.read_csv(y_val_path, low_memory=False)
            names = _label_names()
            column = ("Label" if "Label" in y_val.columns
                      else "Label_encoded")
            for code, number in y_val[column].value_counts().items():
                label = (names.get(int(code), str(code))
                         if column == "Label_encoded" else str(code))
                counts[label] = int(number)
    out["step9_validation_rows"] = counts
    add("### Step 7 -- rare classes in the validation part")
    add("")
    if not counts:
        add("- Not available: Step 7 has not been run in this output root.")
    else:
        thin = {label: number for label, number in counts.items()
                if number < minimum}
        add("Validation rows per class: %s."
            % ", ".join("%s %d" % (label, number)
                        for label, number in sorted(counts.items(),
                                                    key=lambda kv: -kv[1])))
        add("")
        if thin:
            add("**RARE-CLASS WARNING -- %d class(es) below %d validation "
                "rows:** %s. Early stopping monitors the validation macro-F1, "
                "so for these classes the signal it follows -- and the "
                "per-class reliability the Step 13 fusion will weight a model "
                "by -- is estimated from very few flows. Their per-class "
                "figures are indicative, and they move the macro average as "
                "much as the common classes do."
                % (len(thin), minimum,
                   ", ".join("%s (%d)" % (label, number)
                             for label, number in sorted(thin.items(),
                                                         key=lambda kv: kv[1]))))
        else:
            add("- Every class has at least %d validation rows." % minimum)
    add("")

    # ---- Step 11 --------------------------------------------------------
    add("### Step 9 -- Approach 1 and Approach 2 on the untouched test set")
    add("")
    rows = []
    for name, folder_name, label in (
            ("approach1", "STEP11_OUTPUT_DIR", "Approach 1 (CNN-LSTM)"),
            ("approach2", "STEP11B_OUTPUT_DIR",
             "Approach 2 (CNN-Attention-LSTM)")):
        place = config.usable_output_dir(getattr(config, folder_name, None))
        metrics = (_read_json(Path(place) / "metrics.json")
                   if place is not None else None)
        if metrics is None:
            rows.append({"model": label, "status": "not available"})
            continue
        full = metrics.get("full") or {}
        free = metrics.get("duplicate_free") or {}
        binary = metrics.get("binary") or {}
        # the per-class recalls live in per_class_report.csv beside
        # metrics.json, not inside it
        per_class: dict[str, dict[str, Any]] = {}
        per_class_path = (Path(place)
                          / config.STEP11_FILES.get("per_class",
                                                    "per_class_report.csv"))
        if per_class_path.exists():
            frame = pd.read_csv(per_class_path)
            for _i, entry in frame.iterrows():
                per_class[str(entry.get("Label"))] = entry.to_dict()
        rows.append({
            "model": label, "status": "read from metrics.json",
            "accuracy": full.get("accuracy"),
            "F1 (macro)": full.get("f1_macro"),
            "G-mean": full.get("g_mean"),
            "binary FPR": binary.get("false_positive_rate"),
            "F1 macro (dup-free)": free.get("f1_macro"),
            "recall HTTP_DDoS": (per_class.get("HTTP_DDoS") or {}).get("recall"),
            "recall ICMP_Flood": (per_class.get("ICMP_Flood") or {}).get("recall"),
            "recall Web_Crwling": (per_class.get("Web_Crwling") or {}).get("recall"),
        })
    table = pd.DataFrame(rows)
    out["step11"] = table
    quick = dict(getattr(config, "OFFICIAL_QUICK_CHECK", {}))
    if len(table) and "F1 (macro)" in table.columns:
        for _i, row in table.iterrows():
            if row.get("F1 (macro)") is None:
                add("- **%s:** %s." % (row["model"], row["status"]))
                continue
            add("- **%s:** accuracy %.4f, macro-F1 %.4f, G-mean %.4f, binary "
                "FPR %.4f, duplicate-free macro-F1 %s."
                % (row["model"], float(row["accuracy"]),
                   float(row["F1 (macro)"]), float(row["G-mean"] or 0.0),
                   float(row["binary FPR"] or 0.0),
                   "%.4f" % float(row["F1 macro (dup-free)"])
                   if row["F1 macro (dup-free)"] is not None else "(not saved)"))
            for rare in ("HTTP_DDoS", "ICMP_Flood", "Web_Crwling"):
                value = row.get("recall %s" % rare)
                if value is not None:
                    add("  - recall %s: **%.4f**" % (rare, float(value)))
    else:
        add("- Not available: Step 9 has not been run in this output root.")
    add("")
    if quick:
        add("Against the quick stratified 80/20 check (seed 42) the switch was "
            "decided on: a random forest reached **macro-F1%s**, a depth-3 "
            "tree %s, and the majority class %s accuracy. Those are not deep "
            "models and they see the raw features rather than the autoencoder "
            "latents, so they are a REFERENCE for what the dataset supports, "
            "not a competitor."
            % (" %.4f" % quick.get("random forest macro-F1", float("nan")),
               "%.4f macro-F1" % quick.get("decision tree (depth 3) macro-F1",
                                           float("nan")),
               "%.4f" % quick.get("majority class accuracy", float("nan"))))
    text = "\n".join(lines)
    out["text"] = text
    if verbose:
        print(text)
    return out


# ===========================================================================
# STEP 1 SELF-CHECK -- the saved file against the file it came from
# ===========================================================================
def self_check(path: Path | str | None = None, verbose: bool = True
               ) -> dict[str, Any]:
    """Does the file Step 1 wrote still say what MSCAD.csv says?

    A SELF-check, against one source and no other: the saved
    `MSCAD_merged.csv` is re-read from disk and its label counts and per-column
    medians are compared with the same quantities recomputed from
    `MSCAD.csv`.  There is no second dataset to check against, and inventing
    one would be worse than saying so -- what this proves is that Step 1
    reordered and separated columns without changing a single value.

    A median rather than a mean, because a median is unchanged by the
    infinities the rate columns carry, and Step 1 does not cap them -- Step 2
    does.
    """
    place = folder()
    if place is None:
        return {"table": pd.DataFrame(), "ok": False,
                "detail": config.why_unusable(config.STEP3_OUTPUT_DIR)}
    saved_path = place / config.OFFICIAL_STEP3_FILES["merged"]
    if not saved_path.exists():
        return {"table": pd.DataFrame(), "ok": False,
                "detail": "%s does not exist yet" % saved_path}

    source, _renames = read_official_csv(path, verbose=False)
    saved = load_merged_official(verbose=False)

    rows: list[dict[str, Any]] = []
    rows.append({"check": "row count", "source (MSCAD.csv)": len(source),
                 "saved (MSCAD_merged.csv)": len(saved),
                 "largest difference": 0 if len(source) == len(saved) else
                 abs(len(source) - len(saved)),
                 "result": OK if len(source) == len(saved) else FAIL})

    labels_source = source[config.LABEL_COLUMN].astype(str).value_counts()
    labels_saved = saved[config.LABEL_COLUMN].astype(str).value_counts()
    same_labels = (set(labels_source.index) == set(labels_saved.index)
                   and all(int(labels_source[name]) == int(labels_saved[name])
                           for name in labels_source.index))
    rows.append({"check": "label counts, all %d classes" % len(labels_source),
                 "source (MSCAD.csv)": "%d classes" % len(labels_source),
                 "saved (MSCAD_merged.csv)": "%d classes" % len(labels_saved),
                 "largest difference": 0 if same_labels else 1,
                 "result": OK if same_labels else FAIL})

    shared = [c for c in source.columns if c in saved.columns]
    numeric = []
    worst, worst_column = 0.0, None
    for name in shared:
        left = pd.to_numeric(source[name], errors="coerce")
        right = pd.to_numeric(saved[name], errors="coerce")
        if left.notna().sum() == 0 or right.notna().sum() == 0:
            continue
        numeric.append(name)
        a = float(left.replace([np.inf, -np.inf], np.nan).median())
        b = float(right.replace([np.inf, -np.inf], np.nan).median())
        if np.isnan(a) and np.isnan(b):
            continue
        gap = abs(a - b) if not (np.isnan(a) or np.isnan(b)) else float("inf")
        if gap > worst:
            worst, worst_column = gap, name
    rows.append({"check": "median of every numeric column (%d compared)"
                           % len(numeric),
                 "source (MSCAD.csv)": "%d columns" % len(numeric),
                 "saved (MSCAD_merged.csv)": "%d columns" % len(numeric),
                 "largest difference": worst,
                 "result": OK if worst == 0.0 else FAIL})

    table = pd.DataFrame(rows)
    ok = bool(not (table["result"] == FAIL).any())
    if verbose:
        show_title("1.9  Self-check: the saved file against MSCAD.csv", level=4)
        show_table(table, all_columns=True,
                   note="Reading: the file Step 1 wrote, re-read from disk, "
                        "against the same quantities recomputed from "
                        "MSCAD.csv. There is no second dataset to compare "
                        "with, so this proves only what it can prove -- that "
                        "separating the identifiers and reordering the columns "
                        "changed no value. %s"
                        % ("Every column agrees exactly."
                           if ok else "A [FAIL] here means the saved file is "
                                      "not the file it came from."))
        print("[%s]  Step 1 self-check: %d checks, largest median difference "
              "%.3e" % ("OK" if ok else "FAIL", len(table), worst))
    return {"table": table, "ok": ok, "worst_column": worst_column,
            "worst_difference": worst, "numeric_columns": numeric}


def own_extraction_label_comparison(verbose: bool = True) -> dict[str, Any]:
    """The official label counts beside our own extraction's, as CONTEXT only.

    Nothing here is used for training, and the two columns are not two
    measurements of the same thing: they are two different datasets that happen
    to carry the same class names.  It is shown because the switch between them
    is a decision the write-up has to justify, and the clearest justification
    is the two tables side by side -- our own extraction's labels disagreed
    with themselves, and the official file's do not.
    """
    own_root = getattr(config, "OUTPUTS_ROOT_OWN", None)
    own_path = (None if own_root is None else
                Path(own_root) / "step3" / "MSCAD_merged.csv")
    official = cmp_counts = None
    place = folder()
    if place is not None:
        saved = place / config.OFFICIAL_STEP3_FILES["merged"]
        if saved.exists():
            official = pd.read_csv(saved, usecols=[config.LABEL_COLUMN],
                                   low_memory=False)[
                config.LABEL_COLUMN].astype(str).value_counts()
    if official is None:
        return {"table": pd.DataFrame(), "available": False,
                "detail": "Step 1 has not written its file yet"}
    if own_path is None or not own_path.exists():
        if verbose:
            show_note("**No own-extraction file to compare with.** "
                      "`%s` is not on this machine, so the official counts "
                      "below stand on their own." % own_path)
        return {"table": official.rename("official rows").to_frame()
                .reset_index(names="label"), "available": False,
                "detail": "no own-extraction MSCAD_merged.csv"}

    own = pd.read_csv(own_path, usecols=[config.LABEL_COLUMN],
                      low_memory=False)[config.LABEL_COLUMN].astype(
                          str).value_counts()
    names = sorted(set(official.index) | set(own.index))
    rows = []
    for name in names:
        rows.append({
            "label": name,
            "official rows": int(official.get(name, 0)),
            "official %": round(100.0 * int(official.get(name, 0))
                                / max(int(official.sum()), 1), 2),
            "own extraction rows": int(own.get(name, 0)),
            "own extraction %": round(100.0 * int(own.get(name, 0))
                                      / max(int(own.sum()), 1), 2),
            "in both": bool(name in official.index and name in own.index)})
    table = pd.DataFrame(rows)
    if verbose:
        show_title("1.10  The two datasets' labels, side by side", level=4)
        show_table(table, all_columns=True,
                   note="Reading: **comparison, not used for training.** The "
                        "left columns are the official dataset this pipeline "
                        "runs on; the right are our own PCAP extraction, kept "
                        "for reference. These are two different datasets that "
                        "share class names, not two measurements of one thing. "
                        "A class present in one and absent from the other is "
                        "the clearest reason the official file became the "
                        "primary dataset.")
        only_own = [r["label"] for r in rows if not r["official rows"]]
        only_official = [r["label"] for r in rows if not r["own extraction rows"]]
        if only_own:
            show_note("**Only in our own extraction:** %s." % ", ".join(only_own))
        if only_official:
            show_note("**Only in the official dataset:** %s."
                      % ", ".join(only_official))
    return {"table": table, "available": True,
            "own_path": str(own_path), "official_total": int(official.sum()),
            "own_total": int(own.sum())}
