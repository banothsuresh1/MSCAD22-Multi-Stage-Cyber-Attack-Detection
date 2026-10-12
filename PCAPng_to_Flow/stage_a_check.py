"""
stage_a_check.py -- STAGE A: environment check + capture inventory + Kaggle header check.

This script only LOOKS at things.  It creates the four output folders and writes
logs\\pcap_inventory.csv, and nothing else.  It never modifies, moves or deletes
any capture or the Kaggle file.

Run it from cmd (not PowerShell):

    cd /d C:\\MSCAD2022\\PCAPng_to_Flow
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" stage_a_check.py

Then paste the whole output back into the chat.
"""

import csv
import importlib.util
import os
import subprocess
import sys

# Make sure "import config" works no matter which folder you run this from.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config  # noqa: E402  (must come after the sys.path line above)


# ===========================================================================
# tiny printing helpers so the output is easy to read / paste
# ===========================================================================
def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def sub(title):
    print()
    print("-- " + title + " " + "-" * max(0, 72 - len(title)))


def ok(msg):
    print("  [OK]    " + msg)


def warn(msg):
    print("  [WARN]  " + msg)


def bad(msg):
    print("  [MISS]  " + msg)


# ===========================================================================
# A.1  find the capture folder
# ===========================================================================
def find_pcapng_dir():
    """Return (folder, how_we_found_it).

    First choice is config.PCAPNG_DIR.  If that does not exist we walk
    config.BASE_DIR looking for a folder called 'pcapngtopcap' (any casing) or
    any folder that actually contains .pcapng files, and report which one we used.
    """
    if os.path.isdir(config.PCAPNG_DIR):
        return config.PCAPNG_DIR, "config.PCAPNG_DIR (expected location)"

    warn("config.PCAPNG_DIR does not exist: %s" % config.PCAPNG_DIR)
    if not os.path.isdir(config.BASE_DIR):
        bad("config.BASE_DIR does not exist either: %s" % config.BASE_DIR)
        return None, "not found"

    named = []    # folders literally called pcapngtopcap
    with_cap = []  # folders that contain .pcapng files
    for root, dirs, files in os.walk(config.BASE_DIR):
        if os.path.basename(root).lower().replace("_", "").replace("-", "") == "pcapngtopcap":
            named.append(root)
        if any(f.lower().endswith(".pcapng") for f in files):
            with_cap.append(root)

    for cand, why in ((named, "folder named 'pcapngtopcap' found by search"),
                      (with_cap, "folder containing .pcapng files found by search")):
        if cand:
            cand.sort(key=len)  # shallowest first
            if len(cand) > 1:
                warn("several candidates, using the shallowest: %s" % cand)
            return cand[0], why
    return None, "not found"


# ===========================================================================
# A.2  inventory every capture file
# ===========================================================================
def build_inventory(pcapng_dir):
    """Walk pcapng_dir recursively and return a list of dict rows."""
    rows = []
    if not pcapng_dir:
        return rows
    for root, _dirs, files in os.walk(pcapng_dir):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext not in (".pcapng", ".pcap"):
                continue
            full = os.path.join(root, name)
            size_bytes = os.path.getsize(full)
            rows.append({
                "stem": os.path.splitext(name)[0],
                "file_name": name,
                "extension": ext,
                "full_path": full,
                "relative_path": os.path.relpath(full, pcapng_dir),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "expected_label": config.CAPTURE_LABELS.get(
                    os.path.splitext(name)[0], "UNKNOWN - not in config.CAPTURE_LABELS"),
            })
    rows.sort(key=lambda r: r["relative_path"].lower())
    return rows


def write_inventory(rows):
    """Write logs\\pcap_inventory.csv.  Uses the stdlib, so no pandas needed."""
    fields = ["stem", "file_name", "extension", "size_mb", "size_bytes",
              "expected_label", "relative_path", "full_path"]
    with open(config.PCAP_INVENTORY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    return config.PCAP_INVENTORY_CSV


def print_inventory(rows):
    if not rows:
        bad("no .pcapng / .pcap files found")
        return
    print("  %-16s %-8s %12s  %-34s %s"
          % ("STEM", "EXT", "SIZE (MB)", "EXPECTED LABEL", "RELATIVE PATH"))
    print("  " + "-" * 104)
    total_mb = 0.0
    for r in rows:
        total_mb += r["size_mb"]
        print("  %-16s %-8s %12.2f  %-34s %s" % (
            r["stem"], r["extension"], r["size_mb"],
            r["expected_label"], r["relative_path"]))
    print("  " + "-" * 104)
    n_ng = sum(1 for r in rows if r["extension"] == ".pcapng")
    n_p = sum(1 for r in rows if r["extension"] == ".pcap")
    print("  TOTAL: %d files (%d .pcapng + %d .pcap), %.2f MB" % (len(rows), n_ng, n_p, total_mb))

    # Which of the six expected captures are present / missing?
    found_stems = {r["stem"] for r in rows}
    expected = list(config.CAPTURE_LABELS.keys())
    missing = [s for s in expected if s not in found_stems]
    extra = sorted(s for s in found_stems if s not in config.CAPTURE_LABELS)
    print()
    print("  expected captures present : %s" % ([s for s in expected if s in found_stems] or "NONE"))
    if missing:
        warn("expected captures MISSING  : %s" % missing)
    else:
        ok("all %d expected captures are present" % len(expected))
    if extra:
        warn("extra capture files not in config.CAPTURE_LABELS: %s" % extra)


# ===========================================================================
# A.3  environment: python, packages, wireshark tools
# ===========================================================================
PACKAGES = [
    ("cicflowmeter", "0.5.0"),  # (import name, required version or None)
    ("scapy", None),
    ("numpy", None),
    ("scipy", None),
    ("pandas", None),
    ("tqdm", None),
]

# This probe runs INSIDE config.PYTHON_EXE so we check the interpreter that will
# actually do the conversion, not the one that happens to run this script.
PROBE = r"""
import importlib.metadata as md, importlib.util, json, sys
out = {"version": sys.version, "version_info": list(sys.version_info[:3]),
       "executable": sys.executable, "packages": {}}
for name in ["cicflowmeter", "scapy", "numpy", "scipy", "pandas", "tqdm"]:
    info = {"importable": importlib.util.find_spec(name) is not None, "version": None}
    try:
        info["version"] = md.version(name)
    except Exception:
        pass
    out["packages"][name] = info
print("PROBE_JSON:" + json.dumps(out))
"""


def probe_interpreter():
    """Ask config.PYTHON_EXE about itself.  Falls back to the running python."""
    import json

    exe = config.PYTHON_EXE
    which = "config.PYTHON_EXE"
    if not os.path.isfile(exe):
        bad("config.PYTHON_EXE not found: %s" % exe)
        warn("falling back to the interpreter running this script: %s" % sys.executable)
        exe = sys.executable
        which = "sys.executable (fallback)"

    # LIST of args + no shell=True  ->  spaces in the path are safe.
    try:
        res = subprocess.run([exe, "-c", PROBE], capture_output=True, text=True, timeout=120)
    except Exception as exc:
        bad("could not run %s : %r" % (exe, exc))
        return None, exe, which
    if res.returncode != 0:
        bad("probe failed (exit %d)" % res.returncode)
        print("    stderr: " + (res.stderr or "").strip()[:800])
        return None, exe, which
    for line in res.stdout.splitlines():
        if line.startswith("PROBE_JSON:"):
            return json.loads(line[len("PROBE_JSON:"):]), exe, which
    bad("probe produced no JSON; stdout was:")
    print("    " + (res.stdout or "").strip()[:800])
    return None, exe, which


def check_python_and_packages():
    info, exe, which = probe_interpreter()
    print("  interpreter used : %s  (%s)" % (exe, which))
    if info is None:
        return [p for p, _ in PACKAGES]  # treat everything as missing

    vi = info["version_info"]
    print("  python version   : %d.%d.%d" % tuple(vi))
    if (vi[0], vi[1]) == (3, 13):
        ok("Python 3.13 confirmed")
    else:
        warn("expected Python 3.13, found %d.%d -- cicflowmeter 0.5.0 declares an "
             "older python_requires, which is why we install it with "
             "--ignore-requires-python" % (vi[0], vi[1]))

    missing = []
    print()
    print("  %-14s %-12s %-12s %s" % ("PACKAGE", "INSTALLED", "REQUIRED", "STATUS"))
    print("  " + "-" * 60)
    for name, required in PACKAGES:
        p = info["packages"][name]
        installed = p["version"] or ("present" if p["importable"] else "-")
        if not p["importable"]:
            status = "MISSING"
            missing.append(name)
        elif required and p["version"] != required:
            status = "WRONG VERSION"
            missing.append(name)
        else:
            status = "ok"
        print("  %-14s %-12s %-12s %s" % (name, installed, required or "any", status))
    return missing


def print_install_commands(missing):
    if not missing:
        ok("all required packages are installed")
        return
    print()
    warn("missing / wrong version: %s" % missing)
    print("  Run these two commands in cmd (quotes matter -- the path has spaces):")
    print()
    print('      "%s" -m pip install scapy numpy scipy pandas requests tqdm' % config.PYTHON_EXE)
    print('      "%s" -m pip install cicflowmeter==0.5.0 --no-deps --ignore-requires-python'
          % config.PYTHON_EXE)
    print()
    print("  (--no-deps + --ignore-requires-python are required: cicflowmeter 0.5.0")
    print("   pins old dependency versions and declares python_requires <3.13.)")


def check_wireshark_tools():
    """editcap converts pcapng->pcap; tshark counts packets for verification."""
    all_ok = True
    for label, exe, version_flag in (("editcap", config.EDITCAP_EXE, "--version"),
                                     ("tshark", config.TSHARK_EXE, "--version")):
        if not os.path.isfile(exe):
            bad("%s not found at %s" % (label, exe))
            all_ok = False
            continue
        try:
            res = subprocess.run([exe, version_flag], capture_output=True, text=True, timeout=60)
            first = (res.stdout or res.stderr or "").strip().splitlines()
            ok("%-8s %s" % (label, first[0] if first else "(no version output)"))
        except Exception as exc:
            bad("%s found but would not run: %r" % (label, exc))
            all_ok = False
    if not all_ok:
        print()
        print("  Install Wireshark (it ships both tools) from https://www.wireshark.org/download.html")
        print("  then re-check the two paths in config.py: EDITCAP_EXE / TSHARK_EXE")
    return all_ok


# ===========================================================================
# A.4  the Kaggle MSCAD reference file
# ===========================================================================
def inspect_kaggle_file():
    """Print shape, numbered columns and Label counts.  Returns the column list."""
    if importlib.util.find_spec("pandas") is None:
        bad("pandas is not installed in THIS interpreter (%s), so the Kaggle file "
            "cannot be read here." % sys.executable)
        print("      Re-run this script with config.PYTHON_EXE once pandas is installed.")
        return None

    import pandas as pd

    path = config.find_kaggle_file()
    if path is None:
        bad("no .csv / .xlsx found in %s" % config.CSV_ORIGINAL_DIR)
        return None
    ok("Kaggle file: %s  (%.2f MB)" % (path, os.path.getsize(path) / (1024 * 1024)))

    try:
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)          # needs openpyxl for .xlsx
        else:
            df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        bad("could not read the Kaggle file: %r" % exc)
        return None

    print("  shape (rows, columns): %s" % (df.shape,))

    cols = list(df.columns)
    sub("Kaggle columns, numbered, EXACTLY as stored (| | marks the string bounds)")
    for i, c in enumerate(cols, start=1):
        flag = ""
        if c != c.strip():
            flag = "   <-- has leading/trailing whitespace"
        print("  %3d  |%s|%s" % (i, c, flag))

    # Label value counts
    label_col = None
    for c in cols:
        if str(c).strip().lower() == "label":
            label_col = c
            break
    sub("Label value counts")
    if label_col is None:
        bad("no 'Label' column found")
    else:
        if label_col != "Label":
            warn("the label column is stored as |%s|, not |Label|" % label_col)
        vc = df[label_col].value_counts(dropna=False)
        for value, count in vc.items():
            print("  %-18s %8d   (%5.2f%%)" % (str(value), count, 100.0 * count / len(df)))
        print("  %-18s %8d" % ("TOTAL", len(df)))
        print()
        print("  labels in config.CAPTURE_LABELS : %s"
              % sorted(set(config.CAPTURE_LABELS.values()) | {config.BENIGN_LABEL}))
        kaggle_labels = {str(v) for v in vc.index}
        mine = set(config.CAPTURE_LABELS.values()) | {config.BENIGN_LABEL}
        only_kaggle = sorted(kaggle_labels - mine)
        only_mine = sorted(mine - kaggle_labels)
        if only_kaggle:
            warn("labels in Kaggle but NOT in config.CAPTURE_LABELS: %s" % only_kaggle)
        if only_mine:
            warn("labels in config.CAPTURE_LABELS but NOT in Kaggle: %s" % only_mine)
        if not only_kaggle and not only_mine:
            ok("label spellings match config.CAPTURE_LABELS exactly")
    return cols


# ===========================================================================
# A.5  reference column list vs the real Kaggle header
# ===========================================================================
def compare_headers(kaggle_cols):
    """The Kaggle header WINS.  This only reports the differences."""
    sub("my reference list vs the real Kaggle header")
    print("  reference FLOW_COLUMNS in config.py : %d columns "
          "(%d identifiers + %d features + Label)"
          % (len(config.FLOW_COLUMNS), len(config.ID_COLUMNS),
             len(config.KAGGLE_FEATURES_REFERENCE)))

    if kaggle_cols is None:
        warn("Kaggle header could not be read, so no comparison was possible.")
        return

    kaggle_clean = [str(c).strip() for c in kaggle_cols]
    kaggle_feats = [c for c in kaggle_clean if c.lower() != "label"]
    ref_feats = config.KAGGLE_FEATURES_REFERENCE

    print("  Kaggle feature columns (Label excluded)   : %d" % len(kaggle_feats))

    missing_in_ref = [c for c in kaggle_feats if c not in ref_feats]
    extra_in_ref = [c for c in ref_feats if c not in kaggle_feats]

    if missing_in_ref:
        bad("in Kaggle but NOT in my reference list (%d): %s" % (len(missing_in_ref), missing_in_ref))
    if extra_in_ref:
        bad("in my reference list but NOT in Kaggle (%d): %s" % (len(extra_in_ref), extra_in_ref))

    if not missing_in_ref and not extra_in_ref:
        ok("same set of feature names")
        order_diff = [(i + 1, a, b) for i, (a, b) in enumerate(zip(ref_feats, kaggle_feats)) if a != b]
        if order_diff:
            bad("same names but DIFFERENT ORDER at %d positions:" % len(order_diff))
            for pos, a, b in order_diff:
                print("      position %2d: reference |%s|  vs  Kaggle |%s|" % (pos, a, b))
        else:
            ok("identical order -- config.FLOW_COLUMNS needs no change")

    if missing_in_ref or extra_in_ref or len(kaggle_feats) != len(ref_feats):
        print()
        print("  ==> The Kaggle header wins.  Paste this block over")
        print("      KAGGLE_FEATURES_REFERENCE in config.py (Stage C will do it for you):")
        print("      KAGGLE_FEATURES_REFERENCE = [")
        for c in kaggle_feats:
            print('          "%s",' % c)
        print("      ]")


# ===========================================================================
# main
# ===========================================================================
def main():
    header("STAGE A -- ENVIRONMENT CHECK AND INVENTORY (read-only)")
    print("  BASE_DIR : %s" % config.BASE_DIR)
    print("  this script writes ONLY: the 4 output folders + %s"
          % os.path.basename(config.PCAP_INVENTORY_CSV))

    header("A.0  output folders")
    created = config.ensure_output_dirs()
    for d in config.WRITABLE_DIRS:
        print("  %-55s %s" % (d, "CREATED" if d in created else "already existed"))

    header("A.1  locate the capture folder")
    pcapng_dir, how = find_pcapng_dir()
    if pcapng_dir:
        ok("using: %s" % pcapng_dir)
        print("         reason: %s" % how)
    else:
        bad("no capture folder could be found under %s" % config.BASE_DIR)

    header("A.2  capture inventory (recursive)")
    rows = build_inventory(pcapng_dir)
    print_inventory(rows)
    if rows:
        print()
        ok("inventory written to %s" % write_inventory(rows))

    header("A.3  environment check")
    sub("python + packages")
    missing = check_python_and_packages()
    print_install_commands(missing)
    sub("wireshark command-line tools")
    check_wireshark_tools()

    header("A.4  Kaggle MSCAD reference file (read-only)")
    kaggle_cols = inspect_kaggle_file()

    header("A.5  column-name comparison")
    compare_headers(kaggle_cols)

    header("STAGE A FINISHED -- paste this whole output back into the chat")
    print("  Nothing was converted and nothing was modified.")


if __name__ == "__main__":
    main()
