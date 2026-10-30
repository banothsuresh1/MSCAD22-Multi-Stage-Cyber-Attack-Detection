r"""audit_step1.py -- READ-ONLY audit of Step 1 (data acquisition).

Computes the ground truth straight from the raw MSCAD.csv, reads whatever the
existing Step 1 outputs claim, compares them field by field, and saves every
result into outputs_official\step1\.  It changes NOTHING: MSCAD.csv and the
existing step3\ files are opened read-only; only the new step1\ folder is
written.  It does not re-run the Step 1 pipeline.

    cd C:\Users\<you>\MSCAD22-Multi-Stage-Cyber-Attack-Detection\Notebook
    python audit_step1.py

Outputs written to <outputs_official>\step1\:
    step1_raw_ground_truth.json      -- #1, computed independently from the CSV
    step1_existing_outputs_dump.json -- #2, verbatim content of every step3 file
    step1_comparison.csv             -- #3, field-by-field MATCH / MISMATCH
    step1_audit_report.md            -- a readable summary of all of the above
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

RAW = Path(config.OFFICIAL_CSV)
STEP3 = Path(config.STEP3_OUTPUT_DIR)
OUT = Path(config.OUTPUTS_DIR) / "step1"


def _sha256(path: Path, limit: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit))     # first 8 MB only -- a fingerprint, not full hash
    return h.hexdigest()


def raw_ground_truth() -> dict:
    df = pd.read_csv(RAW, low_memory=False)
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    cols = [str(c) for c in df.columns]

    label_col = next((c for c in (getattr(config, "LABEL_COLUMN", "Label"),
                                  "Label", "label") if c in df.columns),
                     cols[-1])
    vc = df[label_col].value_counts(dropna=False)
    label_dist = {str(k): {"count": int(v), "pct": round(100.0 * v / n_rows, 4)}
                  for k, v in vc.items()}

    num = df.apply(pd.to_numeric, errors="coerce")
    inf_by_col = {}
    for c in cols:
        arr = num[c].to_numpy(dtype="float64", na_value=np.nan)
        k = int(np.isinf(arr).sum())
        if k:
            inf_by_col[c] = k
    missing_by_col = {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()}

    return {
        "raw_file": str(RAW),
        "exists": True,
        "sha256_first_8mb": _sha256(RAW),
        "rows": n_rows,
        "columns": n_cols,
        "column_names_in_order": cols,
        "dtypes": {c: str(df[c].dtype) for c in cols},
        "label_column_used": label_col,
        "label_distribution": label_dist,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_by_column": missing_by_col,
        "infinite_values_total": int(sum(inf_by_col.values())),
        "infinite_by_column": inf_by_col,
        "config_KAGGLE_FEATURES_count": len(getattr(config, "KAGGLE_FEATURES", [])),
        "config_LABEL_COLUMN": getattr(config, "LABEL_COLUMN", None),
    }


def read_existing_outputs() -> dict:
    """Verbatim content of every file in step3\\, whatever its name/schema."""
    claims = {}
    if not STEP3.exists():
        return {"_step3_folder": str(STEP3), "_exists": False}
    claims["_step3_folder"] = str(STEP3)
    claims["_exists"] = True
    claims["_files"] = sorted(p.name for p in STEP3.iterdir() if p.is_file())
    for p in sorted(STEP3.glob("*")):
        if not p.is_file():
            continue
        try:
            if p.suffix.lower() == ".json":
                claims[p.name] = json.loads(p.read_text(encoding="utf-8"))
            elif p.suffix.lower() == ".csv":
                c = pd.read_csv(p, low_memory=False)
                claims[p.name] = {"_shape": list(c.shape),
                                  "_columns": [str(x) for x in c.columns],
                                  "_head": c.head(30).to_dict("list")}
            elif p.suffix.lower() in (".md", ".txt"):
                claims[p.name] = p.read_text(encoding="utf-8")[:6000]
            else:
                claims[p.name] = "(not parsed: %s)" % p.suffix
        except Exception as exc:                      # noqa: BLE001 -- audit, report anything
            claims[p.name] = {"_error": repr(exc)}
    return claims


def _find_ints(obj, needles):
    """Every int anywhere in a nested dict/list whose key contains a needle."""
    found = []

    def walk(o, keyhint=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, str(k).lower())
        elif isinstance(o, list):
            for v in o:
                walk(v, keyhint)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            if any(n in keyhint for n in needles) and float(o).is_integer():
                found.append((keyhint, int(o)))
    walk(obj)
    return found


def compare(truth: dict, claims: dict) -> pd.DataFrame:
    rows = []

    def add(field, raw_val, claimed_val, verdict, note=""):
        rows.append({"field": field, "raw_file": raw_val,
                     "step3_claim": claimed_val, "verdict": verdict, "note": note})

    # row count / column count -- scan every claim file for matching integers
    row_hits = _find_ints(claims, ("row", "record", "sample", "instance"))
    col_hits = _find_ints(claims, ("column", "feature", "field"))
    row_claim = sorted({v for _, v in row_hits})
    col_claim = sorted({v for _, v in col_hits})
    add("row count", truth["rows"], row_claim or "(not found)",
        "MATCH" if truth["rows"] in row_claim else
        ("UNKNOWN" if not row_claim else "MISMATCH"),
        "all integers found under row/record-like keys")
    add("column count", truth["columns"], col_claim or "(not found)",
        "MATCH" if truth["columns"] in col_claim else
        ("UNKNOWN" if not col_claim else "REVIEW"),
        "raw count includes Label and any metadata; compare deliberately")
    add("'66 CICFlowMeter features' assumption",
        "%d raw columns (KAGGLE_FEATURES=%d)"
        % (truth["columns"], truth["config_KAGGLE_FEATURES_count"]),
        "n/a",
        "MATCH" if truth["columns"] == 66 else "REVIEW",
        "raw column count vs the 66 assumption -- 66 is the kaggle feature "
        "subset, the raw file also carries Label (+ maybe metadata)")
    # label classes
    add("label classes (raw)",
        sorted(truth["label_distribution"].keys()), "(see dump)", "INFO",
        "compare against label_distribution-type file in the dump")
    add("duplicate rows (raw)", truth["duplicate_rows"], "(see dump)", "INFO",
        "Step 1 must NOT dedup -- this is just the raw count")
    add("missing values (raw)", truth["missing_values_total"], "(see dump)",
        "INFO", "Step 1 must NOT fill/handle missing")
    add("infinite values (raw)", truth["infinite_values_total"], "(see dump)",
        "INFO", "Step 1 must NOT cap/replace infinities")
    return pd.DataFrame(rows)


def main() -> int:
    print("RAW   :", RAW)
    print("STEP3 :", STEP3)
    print("OUT   :", OUT)
    if not RAW.exists():
        print("[STOP] raw MSCAD.csv not found at the path above. Set "
              "MSCAD_OFFICIAL_CSV_DIR / MSCAD_DATA_ROOT or move the file.")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    truth = raw_ground_truth()
    (OUT / "step1_raw_ground_truth.json").write_text(
        json.dumps(truth, indent=2, default=str), encoding="utf-8")

    claims = read_existing_outputs()
    (OUT / "step1_existing_outputs_dump.json").write_text(
        json.dumps(claims, indent=2, default=str), encoding="utf-8")

    table = compare(truth, claims)
    table.to_csv(OUT / "step1_comparison.csv", index=False)

    lines = ["# Step 1 audit (read-only)", "",
             "Raw file: `%s`" % RAW, "",
             "## Ground truth (from the raw CSV)",
             "- rows: **%d**" % truth["rows"],
             "- columns: **%d**" % truth["columns"],
             "- label column: `%s`" % truth["label_column_used"],
             "- duplicate rows: %d" % truth["duplicate_rows"],
             "- missing values (total): %d" % truth["missing_values_total"],
             "- infinite values (total): %d" % truth["infinite_values_total"],
             "", "### Label distribution"]
    for k, v in truth["label_distribution"].items():
        lines.append("- %s: %d (%.4f%%)" % (k, v["count"], v["pct"]))
    try:                                   # to_markdown needs the optional
        table_text = table.to_markdown(index=False)   # 'tabulate' package;
    except Exception:                                  # fall back if absent
        table_text = table.to_string(index=False)
    lines += ["", "## Existing step3 files found",
              ", ".join(claims.get("_files", [])) or "(none)",
              "", "## Field-by-field comparison", "```", table_text, "```",
              "", "_Auto-extraction is best-effort; the full verbatim content "
              "of every step3 file is in step1_existing_outputs_dump.json._"]
    (OUT / "step1_audit_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n[OK] wrote 4 files to", OUT)
    for f in ("step1_raw_ground_truth.json", "step1_existing_outputs_dump.json",
              "step1_comparison.csv", "step1_audit_report.md"):
        print("   -", f)
    print("\nRaw: rows=%d columns=%d dup=%d missing=%d inf=%d"
          % (truth["rows"], truth["columns"], truth["duplicate_rows"],
             truth["missing_values_total"], truth["infinite_values_total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
