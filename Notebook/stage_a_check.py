"""
stage_a_check.py -- STAGE A: check what exists (read-only).

What it does, in order:
  A.1  list C:\\MSCAD2022 two levels deep, and locate the capture folder
  A.2  find any OLDER converter script/notebook and dump its source
  A.3  recursively inventory every .pcapng/.pcap (case-insensitive)
  A.4  check python 3.13, the packages, editcap and tshark
  A.5  read the Kaggle MSCAD file: shape, numbered columns, Label counts
  A.6  compare my 72-column reference list with the real Kaggle header

It CONVERTS NOTHING and MODIFIES NOTHING.  The only file it writes is the
plain-text report  Notebook\\stage_a_report.txt  (a copy of everything it
printed, so you can attach the file instead of pasting a long console dump).

Run it from cmd -- NOT PowerShell:

    cd /d C:\\MSCAD2022\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" stage_a_check.py

Then paste the output (or attach stage_a_report.txt) into the chat.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Make "import config" work no matter which folder you launched this from.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402  (must follow the sys.path line above)

REPORT_PATH = HERE / "stage_a_report.txt"
_REPORT_LINES: list[str] = []

# How many lines of an old script we echo to the console.  The full text always
# goes into the report file.
CONSOLE_DUMP_LIMIT = 120


# ===========================================================================
# printing helpers -- everything goes to the screen AND into the report
# ===========================================================================
def say(line: str = "") -> None:
    print(line)
    _REPORT_LINES.append(line)


def say_only_to_report(line: str) -> None:
    _REPORT_LINES.append(line)


def header(title: str) -> None:
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def sub(title: str) -> None:
    say()
    say("-- " + title + " " + "-" * max(0, 72 - len(title)))


def ok(msg: str) -> None:
    say("  [OK]    " + msg)


def warn(msg: str) -> None:
    say("  [WARN]  " + msg)


def bad(msg: str) -> None:
    say("  [MISS]  " + msg)


def mb(n_bytes: int) -> float:
    return round(n_bytes / (1024 * 1024), 2)


# ===========================================================================
# A.1  what is on disk, two levels deep
# ===========================================================================
def list_two_levels(root: Path) -> None:
    """Print root's children and grandchildren -- enough to see the layout."""
    if not root.is_dir():
        bad("DATA_ROOT does not exist: %s" % root)
        return
    say("  %s\\" % root)
    for level1 in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if level1.is_dir():
            try:
                children = sorted(level1.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except PermissionError:
                say("  |-- %-26s <dir>  (permission denied)" % (level1.name + "\\"))
                continue
            n_files = sum(1 for c in children if c.is_file())
            n_dirs = sum(1 for c in children if c.is_dir())
            say("  |-- %-26s <dir>   %d files, %d subfolders" % (level1.name + "\\", n_files, n_dirs))
            for c in children[:40]:            # second level
                if c.is_dir():
                    say("  |     |-- %-22s <dir>" % (c.name + "\\"))
                else:
                    say("  |     |-- %-22s %10.2f MB" % (c.name, mb(c.stat().st_size)))
            if len(children) > 40:
                say("  |     |-- ... %d more entries" % (len(children) - 40))
        else:
            say("  |-- %-26s %10.2f MB" % (level1.name, mb(level1.stat().st_size)))


def find_pcapng_dir() -> tuple[Path | None, str]:
    """Return (folder, why).  Falls back to searching DATA_ROOT."""
    if config.PCAPNG_DIR.is_dir():
        return config.PCAPNG_DIR, "config.PCAPNG_DIR (the expected location)"

    warn("config.PCAPNG_DIR does not exist: %s" % config.PCAPNG_DIR)
    if not config.DATA_ROOT.is_dir():
        bad("config.DATA_ROOT does not exist either: %s" % config.DATA_ROOT)
        return None, "not found"

    named: list[Path] = []     # folders literally called pcapngtopcap
    with_caps: list[Path] = []  # folders that hold .pcapng files
    for dirpath, _dirnames, filenames in os.walk(config.DATA_ROOT):
        d = Path(dirpath)
        flat = d.name.lower().replace("_", "").replace("-", "").replace(" ", "")
        if flat == "pcapngtopcap":
            named.append(d)
        if any(f.lower().endswith(".pcapng") for f in filenames):
            with_caps.append(d)

    for candidates, why in (
        (named, "a folder named 'pcapngtopcap' found by searching DATA_ROOT"),
        (with_caps, "a folder containing .pcapng files, found by searching DATA_ROOT"),
    ):
        if candidates:
            candidates.sort(key=lambda p: len(p.parts))   # shallowest first
            if len(candidates) > 1:
                warn("several candidates found: %s" % [str(p) for p in candidates])
                warn("using the shallowest one")
            return candidates[0], why
    return None, "not found"


# ===========================================================================
# A.2  older converter code
# ===========================================================================
# A file is "interesting" if its name contains one of these words.
OLD_CODE_KEYWORDS = ("cicflow", "flow", "convert", "pcap")

# Folders we do not search (virtualenvs and site-packages would match "flow").
SKIP_DIR_NAMES = {".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv",
                  "env", "site-packages", "node_modules", ".pytest_cache"}


def find_old_code(root: Path) -> list[Path]:
    """Find *.py / *.ipynb whose NAME contains one of OLD_CODE_KEYWORDS."""
    hits: list[Path] = []
    if not root.is_dir():
        return hits
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() not in (".py", ".ipynb"):
                continue
            # Do not report the Stage A files we just shipped.
            if p.resolve() in (Path(__file__).resolve(), (HERE / "config.py").resolve()):
                continue
            low = name.lower()
            if any(k in low for k in OLD_CODE_KEYWORDS):
                hits.append(p)
    hits.sort(key=lambda p: str(p).lower())
    return hits


def notebook_to_text(path: Path) -> str:
    """Return a notebook's markdown + code cells as plain text."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return "<<could not parse notebook: %r>>" % (exc,)
    parts: list[str] = []
    for i, cell in enumerate(nb.get("cells", []), start=1):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        parts.append("### cell %d [%s]\n%s" % (i, cell.get("cell_type", "?"), src))
    return "\n\n".join(parts) if parts else "<<notebook has no cells>>"


def dump_old_code(hits: list[Path]) -> None:
    if not hits:
        ok("no older converter script or notebook found under %s" % config.DATA_ROOT)
        say("          Nothing to reuse -- Stage C writes src\\converter.py from scratch.")
        return

    warn("found %d existing script(s)/notebook(s) that may be an earlier converter:" % len(hits))
    for p in hits:
        say("          %-60s %8.2f MB" % (p, mb(p.stat().st_size)))
    say()
    say("  These files are NOT modified by this script.  Their full source is in")
    say("  %s so I can read it and tell you exactly what to reuse." % REPORT_PATH.name)

    for p in hits:
        text = notebook_to_text(p) if p.suffix.lower() == ".ipynb" else \
            p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        sub("SOURCE OF %s  (%d lines)" % (p, len(lines)))
        for i, line in enumerate(lines, start=1):
            tagged = "  %5d| %s" % (i, line)
            if i <= CONSOLE_DUMP_LIMIT:
                say(tagged)
            else:
                say_only_to_report(tagged)
        if len(lines) > CONSOLE_DUMP_LIMIT:
            say("  ... %d more lines -- full text is in %s"
                % (len(lines) - CONSOLE_DUMP_LIMIT, REPORT_PATH.name))


# ===========================================================================
# A.3  capture inventory
# ===========================================================================
def build_inventory(pcapng_dir: Path | None) -> list[dict]:
    """Recursive, case-insensitive scan for .pcapng / .pcap."""
    rows: list[dict] = []
    if pcapng_dir is None or not pcapng_dir.is_dir():
        return rows
    for dirpath, dirnames, filenames in os.walk(pcapng_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in sorted(filenames):
            p = Path(dirpath) / name
            # .lower() on the suffix = case-insensitive, so .PCAPNG matches too.
            if p.suffix.lower() not in config.CAPTURE_EXTENSIONS:
                continue
            size = p.stat().st_size
            rows.append({
                "stem": p.stem,
                "file_name": p.name,
                "extension": p.suffix.lower(),
                "size_mb": mb(size),
                "size_bytes": size,
                "relative_path": str(p.relative_to(pcapng_dir)),
                "full_path": str(p),
                "expected_label": config.CAPTURE_LABELS.get(
                    p.stem, "UNKNOWN (no label in config)"),
            })
    rows.sort(key=lambda r: r["relative_path"].lower())
    return rows


def print_inventory(rows: list[dict]) -> None:
    if not rows:
        bad("no .pcapng / .pcap files found")
        return
    say("  %-16s %-8s %11s  %-36s %s"
        % ("STEM", "EXT", "SIZE (MB)", "EXPECTED LABEL", "RELATIVE PATH"))
    say("  " + "-" * 104)
    total = 0.0
    for r in rows:
        total += r["size_mb"]
        say("  %-16s %-8s %11.2f  %-36s %s"
            % (r["stem"], r["extension"], r["size_mb"], r["expected_label"], r["relative_path"]))
    say("  " + "-" * 104)
    n_ng = sum(1 for r in rows if r["extension"] == ".pcapng")
    n_p = sum(1 for r in rows if r["extension"] == ".pcap")
    say("  TOTAL: %d capture files (%d .pcapng + %d .pcap), %.2f MB"
        % (len(rows), n_ng, n_p, total))

    # Are the six documented captures all here?
    found = {r["stem"] for r in rows}
    expected = list(config.CAPTURE_LABELS)
    present = [s for s in expected if s in found]
    missing = [s for s in expected if s not in found]
    extra = sorted(s for s in found if s not in config.CAPTURE_LABELS)
    say()
    say("  documented captures present : %s" % (present or "NONE"))
    if missing:
        warn("documented captures MISSING : %s" % missing)
    else:
        ok("all %d documented captures are present" % len(expected))
    if extra:
        warn("capture files with no label in config.CAPTURE_LABELS: %s" % extra)

    # Duplicate stems matter: Stage B would overwrite one .pcap with the other,
    # so it has to prefix the subfolder name instead.
    seen: dict[str, list[str]] = {}
    for r in rows:
        seen.setdefault(r["stem"].lower(), []).append(r["relative_path"])
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        warn("DUPLICATE stems -- Stage B will prefix the subfolder name:")
        for stem, paths in dupes.items():
            say("            %s -> %s" % (stem, paths))
    else:
        ok("every capture stem is unique, so no subfolder prefixing is needed")


# ===========================================================================
# A.4  environment
# ===========================================================================
# (import name, required version or None)
PACKAGES = [
    ("cicflowmeter", "0.5.0"),
    ("scapy", None),
    ("numpy", None),
    ("scipy", None),
    ("pandas", None),
    ("tqdm", None),
    ("pytest", None),
    ("jupyter_core", None),   # installed by the "jupyter" meta-package
    ("notebook", None),       # the classic notebook / nbclassic front-end
]

# Runs INSIDE config.PYTHON_EXE, so we check the interpreter that will actually
# do the conversion -- not whichever python happens to run this file.
PROBE = r"""
import importlib.metadata as md, importlib.util, json, sys
names = ["cicflowmeter","scapy","numpy","scipy","pandas","tqdm","pytest","jupyter_core","notebook"]
out = {"version_info": list(sys.version_info[:3]), "executable": sys.executable, "packages": {}}
for n in names:
    info = {"importable": importlib.util.find_spec(n) is not None, "version": None}
    for dist in (n, {"jupyter_core":"jupyter-core"}.get(n, n)):
        try:
            info["version"] = md.version(dist); break
        except Exception:
            pass
    out["packages"][n] = info
print("PROBE_JSON:" + json.dumps(out))
"""


def probe_interpreter():
    exe = config.PYTHON_EXE
    which = "config.PYTHON_EXE"
    if not Path(exe).is_file():
        bad("config.PYTHON_EXE does not exist: %s" % exe)
        warn("falling back to the interpreter running this script: %s" % sys.executable)
        exe, which = sys.executable, "sys.executable (fallback)"
    try:
        # LIST of arguments + no shell=True  ->  spaces in the path are safe.
        res = subprocess.run([exe, "-c", PROBE], capture_output=True, text=True, timeout=180)
    except Exception as exc:
        bad("could not run %s : %r" % (exe, exc))
        return None, exe, which
    if res.returncode != 0:
        bad("the probe failed (exit code %d)" % res.returncode)
        say("    stderr: " + (res.stderr or "").strip()[:800])
        return None, exe, which
    for line in res.stdout.splitlines():
        if line.startswith("PROBE_JSON:"):
            return json.loads(line[len("PROBE_JSON:"):]), exe, which
    bad("the probe printed no JSON.  stdout was:")
    say("    " + (res.stdout or "").strip()[:800])
    return None, exe, which


def check_python_and_packages() -> list[str]:
    info, exe, which = probe_interpreter()
    say("  interpreter : %s" % exe)
    say("  source      : %s" % which)
    if info is None:
        return [n for n, _ in PACKAGES]

    v = info["version_info"]
    say("  version     : %d.%d.%d" % tuple(v))
    if (v[0], v[1]) == (3, 13):
        ok("Python 3.13 confirmed")
    else:
        warn("expected Python 3.13 but found %d.%d -- cicflowmeter 0.5.0 declares an "
             "older python_requires, which is why it is installed with "
             "--ignore-requires-python" % (v[0], v[1]))

    missing: list[str] = []
    say()
    say("  %-14s %-12s %-10s %s" % ("PACKAGE", "INSTALLED", "REQUIRED", "STATUS"))
    say("  " + "-" * 58)
    for name, required in PACKAGES:
        p = info["packages"][name]
        shown = p["version"] or ("present" if p["importable"] else "-")
        if not p["importable"]:
            status = "MISSING"
            missing.append(name)
        elif required and p["version"] != required:
            status = "WRONG VERSION"
            missing.append(name)
        else:
            status = "ok"
        say("  %-14s %-12s %-10s %s" % (name, shown, required or "any", status))

    # jupyter and pytest are really used as commands, so check them that way.
    sub("command-line checks")
    for label, args in (("jupyter", ["-m", "jupyter", "--version"]),
                        ("pytest", ["-m", "pytest", "--version"])):
        try:
            res = subprocess.run([exe, *args], capture_output=True, text=True, timeout=180)
            first = (res.stdout or res.stderr or "").strip().splitlines()
            if res.returncode == 0:
                ok("%-8s %s" % (label, first[0] if first else "(no output)"))
            else:
                bad("%-8s exit code %d: %s" % (label, res.returncode,
                                               (first[0] if first else "")[:200]))
                if label not in missing:
                    missing.append(label)
        except Exception as exc:
            bad("%-8s would not run: %r" % (label, exc))
            missing.append(label)
    return missing


def print_install_commands(missing: list[str]) -> None:
    if not missing:
        ok("every required package is installed")
        return
    say()
    warn("missing or wrong version: %s" % missing)
    say("  Run these in cmd (the quotes matter -- the path contains spaces):")
    say()
    say('      "%s" -m pip install scapy numpy scipy pandas requests tqdm jupyter pytest'
        % config.PYTHON_EXE)
    say('      "%s" -m pip install cicflowmeter==0.5.0 --no-deps --ignore-requires-python'
        % config.PYTHON_EXE)
    say()
    say("  Why the two flags: cicflowmeter 0.5.0 pins old dependency versions and")
    say("  declares python_requires < 3.13, so a plain install fails on Python 3.13.")
    say("  --no-deps skips its pins (we install the libraries ourselves on the first")
    say("  line) and --ignore-requires-python skips the version gate.")


def check_wireshark() -> bool:
    """editcap converts pcapng->pcap; tshark counts packets for verification."""
    all_ok = True
    for label, exe in (("editcap", config.EDITCAP_EXE), ("tshark", config.TSHARK_EXE)):
        if not Path(exe).is_file():
            bad("%s not found at %s" % (label, exe))
            all_ok = False
            continue
        try:
            res = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=120)
            first = (res.stdout or res.stderr or "").strip().splitlines()
            ok("%-8s %s" % (label, first[0] if first else "(no version output)"))
        except Exception as exc:
            bad("%s exists but would not run: %r" % (label, exc))
            all_ok = False
    if not all_ok:
        say()
        say("  Install Wireshark -- it ships BOTH editcap and tshark:")
        say("      https://www.wireshark.org/download.html")
        say("  Accept the default folder so the paths become:")
        say(r'      C:\Program Files\Wireshark\editcap.exe')
        say(r'      C:\Program Files\Wireshark\tshark.exe')
        say("  If you install it elsewhere, either edit EDITCAP_EXE / TSHARK_EXE in")
        say("  config.py, or add the Wireshark folder to your PATH and RESTART Jupyter")
        say("  (config.py falls back to shutil.which, which reads the PATH).")
    return all_ok


# ===========================================================================
# A.5  the Kaggle reference file
# ===========================================================================
def inspect_kaggle() -> list[str] | None:
    try:
        import pandas as pd
    except ImportError:
        bad("pandas is not installed in the interpreter running THIS script (%s)."
            % sys.executable)
        say("          Install the packages, then run this script again with config.PYTHON_EXE.")
        return None

    path = config.find_kaggle_file()
    if path is None:
        bad("no .csv / .xlsx found in %s" % config.KAGGLE_DIR)
        return None
    ok("Kaggle file: %s  (%.2f MB)" % (path, mb(path.stat().st_size)))

    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)          # needs openpyxl for .xlsx
        else:
            df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        bad("could not read it: %r" % (exc,))
        if path.suffix.lower() in (".xlsx", ".xls"):
            say('          For .xlsx you also need:  "%s" -m pip install openpyxl'
                % config.PYTHON_EXE)
        return None

    say("  shape (rows, columns): %s" % (df.shape,))
    cols = [str(c) for c in df.columns]

    sub("columns, numbered, EXACTLY as stored (|...| marks the string bounds)")
    for i, c in enumerate(cols, start=1):
        flag = "   <-- leading/trailing whitespace!" if c != c.strip() else ""
        say("  %3d  |%s|%s" % (i, c, flag))

    sub("Label value counts")
    label_col = next((c for c in cols if c.strip().lower() == "label"), None)
    if label_col is None:
        bad("no 'Label' column found")
    else:
        if label_col != "Label":
            warn("the label column is stored as |%s|, not |Label|" % label_col)
        vc = df[label_col].value_counts(dropna=False)
        for value, count in vc.items():
            say("  %-18s %9d   (%5.2f%%)" % (str(value), count, 100.0 * count / len(df)))
        say("  %-18s %9d" % ("TOTAL", len(df)))
        say()
        mine = set(config.CAPTURE_LABELS.values()) | {config.BENIGN_LABEL}
        theirs = {str(v) for v in vc.index}
        say("  labels my config expects : %s" % sorted(mine))
        only_theirs = sorted(theirs - mine)
        only_mine = sorted(mine - theirs)
        if only_theirs:
            warn("in Kaggle but NOT in config.CAPTURE_LABELS: %s" % only_theirs)
        if only_mine:
            warn("in config.CAPTURE_LABELS but NOT in Kaggle: %s  "
                 "('Benign' is expected here -- we create it ourselves)" % only_mine)
        if not only_theirs and only_mine == [config.BENIGN_LABEL]:
            ok("every Kaggle label spelling matches config.CAPTURE_LABELS exactly")
    return cols


# ===========================================================================
# A.6  my reference list vs the real header
# ===========================================================================
def compare_headers(kaggle_cols: list[str] | None) -> None:
    say("  config.FLOW_COLUMNS : %d columns = %d identifiers + %d features + Label"
        % (len(config.FLOW_COLUMNS), len(config.ID_COLUMNS), len(config.KAGGLE_FEATURES)))
    say("  (your message called this list '72 columns'; counted item by item it is")
    say("   6 + 66 + 1 = 73.  The 66 feature names themselves are unchanged.)")

    if kaggle_cols is None:
        warn("the Kaggle header could not be read, so no comparison was possible.")
        return

    clean = [c.strip() for c in kaggle_cols]
    kaggle_feats = [c for c in clean if c.lower() != "label"]
    ref = config.KAGGLE_FEATURES
    say("  Kaggle feature columns (Label excluded) : %d" % len(kaggle_feats))

    only_kaggle = [c for c in kaggle_feats if c not in ref]
    only_ref = [c for c in ref if c not in kaggle_feats]

    if only_kaggle:
        bad("in Kaggle but NOT in my reference list (%d): %s" % (len(only_kaggle), only_kaggle))
    if only_ref:
        bad("in my reference list but NOT in Kaggle (%d): %s" % (len(only_ref), only_ref))

    if not only_kaggle and not only_ref:
        ok("identical set of 66 feature names")
        diffs = [(i + 1, a, b) for i, (a, b) in enumerate(zip(ref, kaggle_feats)) if a != b]
        if diffs:
            bad("same names, DIFFERENT ORDER at %d position(s):" % len(diffs))
            for pos, a, b in diffs:
                say("      position %2d: mine |%s|   Kaggle |%s|" % (pos, a, b))
        else:
            ok("identical order -- config.KAGGLE_FEATURES needs no change")

    # Whatever differs, print a ready-to-paste corrected list.  Kaggle wins.
    if only_kaggle or only_ref or len(kaggle_feats) != len(ref):
        say()
        say("  ==> The Kaggle header WINS.  Paste this over KAGGLE_FEATURES in config.py:")
        say("      KAGGLE_FEATURES = [")
        for c in kaggle_feats:
            say('          "%s",' % c)
        say("      ]")

    # The rename map must cover every output column.
    sub("RENAME_MAP coverage")
    targets = set(config.RENAME_MAP.values())
    needed = set(config.FLOW_COLUMNS) - {config.LABEL_COLUMN}
    uncovered = sorted(needed - targets)
    values = list(config.RENAME_MAP.values())
    dupes = sorted({v for v in values if values.count(v) > 1})
    say("  RENAME_MAP entries: %d   (needs to cover %d columns)" % (len(config.RENAME_MAP), len(needed)))
    if uncovered:
        bad("columns with NO entry in RENAME_MAP: %s" % uncovered)
    else:
        ok("every output column except 'Label' has exactly one RENAME_MAP entry")
    if dupes:
        bad("two snake_case names map to the same output column: %s" % dupes)
    say("  extractor columns we expect to DROP (%d): %s"
        % (len(config.EXPECTED_EXTRA_COLUMNS), config.EXPECTED_EXTRA_COLUMNS))


# ===========================================================================
# main
# ===========================================================================
def main() -> None:
    header("STAGE A -- CHECK WHAT EXISTS  (read-only)")
    say("  DATA_ROOT    : %s" % config.DATA_ROOT)
    say("  PROJECT_ROOT : %s" % config.PROJECT_ROOT)
    say("  This script converts nothing and modifies nothing.")
    say("  The only file it writes is %s" % REPORT_PATH)

    header("A.1  what is on disk (two levels deep)")
    list_two_levels(config.DATA_ROOT)
    sub("the capture folder")
    pcapng_dir, why = find_pcapng_dir()
    if pcapng_dir is not None:
        ok("using: %s" % pcapng_dir)
        say("          reason: %s" % why)
    else:
        bad("no capture folder found under %s" % config.DATA_ROOT)

    header("A.2  older converter code (read, never modified)")
    dump_old_code(find_old_code(config.DATA_ROOT))

    header("A.3  capture inventory (recursive, case-insensitive)")
    print_inventory(build_inventory(pcapng_dir))
    say()
    say("  (Nothing written yet.  outputs\\step1\\pcap_inventory.csv is a Stage B output.)")

    header("A.4  environment")
    sub("python and packages")
    missing = check_python_and_packages()
    print_install_commands(missing)
    sub("Wireshark command-line tools")
    check_wireshark()

    header("A.5  Kaggle MSCAD reference file (read-only)")
    kaggle_cols = inspect_kaggle()

    header("A.6  column names: my reference list vs the Kaggle header")
    compare_headers(kaggle_cols)

    header("STAGE A FINISHED")
    say("  Nothing was converted, moved, renamed or deleted.")
    try:
        REPORT_PATH.write_text("\n".join(_REPORT_LINES) + "\n", encoding="utf-8")
        print("  full report written to: %s" % REPORT_PATH)
    except Exception as exc:
        print("  [WARN] could not write the report file: %r" % (exc,))
    print("  Paste the output above (or attach the report file) into the chat.")


if __name__ == "__main__":
    main()
