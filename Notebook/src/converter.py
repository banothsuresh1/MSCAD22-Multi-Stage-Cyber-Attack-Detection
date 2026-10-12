"""
src/converter.py -- ALL the Step 1 logic.

The notebook and stage_a_check.py are thin: they only call functions from here.
Nothing in this file contains a hard-coded path; everything comes from config.py.

Contents
--------
    reporting            Report -- collects every printed line so it can be saved
    discovery            find_captures, output_pcap_name, write_inventory
    external tools       build_*_cmd, run_cmd, count_packets, count_ipv4_packets
    Stage A              stage_a_* functions + stage_a_report()
    Stage B              convert_one, convert_all   (pcapng -> pcap with editcap)
    Stage C (pure)       PacketRecord, flow_key, post-processing, labelling
    Stage C (extractor)  extract_flows -- NOT IMPLEMENTED YET, see the note there

Why subprocess.run([...]) everywhere
------------------------------------
Windows paths contain spaces ("C:\\Program Files\\Wireshark\\tshark.exe").  Passing
a LIST of arguments hands each one to Windows separately, so a space can never
split an argument in two.  Using shell=True with one big string would break on
every one of those paths, so this project never does that.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# config.py sits one folder above src/, so make sure that folder is importable.
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402


# ===========================================================================
# errors
# ===========================================================================
class ConverterError(Exception):
    """Base class for every error this module raises on purpose."""


class MissingColumnError(ConverterError):
    """A column required by config.FLOW_COLUMNS is absent after renaming.

    This is deliberately fatal for that capture.  The alternative would be to
    invent a value, and a fabricated feature would quietly poison every model
    trained on it later.
    """


class ToolError(ConverterError):
    """An external program (editcap / tshark / capinfos) failed or is missing."""


# ===========================================================================
# reporting -- print to the screen and remember every line
# ===========================================================================
class Report:
    """Collects output lines so a stage can be printed AND saved to a file.

    Used by both the notebook and stage_a_check.py, so the two always show
    exactly the same thing.
    """

    def __init__(self, echo: bool = True) -> None:
        self.echo = echo
        self.lines: list[str] = []

    def say(self, line: str = "", echo: bool | None = None) -> None:
        self.lines.append(line)
        if self.echo if echo is None else echo:
            print(line)

    def quiet(self, line: str) -> None:
        """Keep the line in the report, but do not print it (long file dumps)."""
        self.say(line, echo=False)

    def header(self, title: str) -> None:
        self.say()
        self.say("=" * 78)
        self.say(title)
        self.say("=" * 78)

    def sub(self, title: str) -> None:
        self.say()
        self.say("-- " + title + " " + "-" * max(0, 72 - len(title)))

    def ok(self, msg: str) -> None:
        self.say("  [OK]    " + msg)

    def warn(self, msg: str) -> None:
        self.say("  [WARN]  " + msg)

    def bad(self, msg: str) -> None:
        self.say("  [MISS]  " + msg)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return path


def mb(n_bytes: int) -> float:
    """Bytes -> megabytes, rounded to 2 decimals."""
    return round(n_bytes / (1024 * 1024), 2)


# ===========================================================================
# discovery
# ===========================================================================
# Folders never searched: virtualenvs and site-packages contain library files
# whose names also match words like "flow".
SKIP_DIR_NAMES = {".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv",
                  "env", "site-packages", "node_modules", ".pytest_cache"}


@dataclass
class CaptureFile:
    """One capture file found on disk."""

    path: Path
    root: Path                    # the folder the scan started from
    size_bytes: int

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def ext(self) -> str:
        """Lower-cased suffix, so .PCAPNG and .pcapng are the same thing."""
        return self.path.suffix.lower()

    @property
    def rel_path(self) -> str:
        return str(self.path.relative_to(self.root))

    @property
    def size_mb(self) -> float:
        return mb(self.size_bytes)

    @property
    def is_already_pcap(self) -> bool:
        return self.ext == ".pcap"

    @property
    def expected_label(self) -> str:
        return config.CAPTURE_LABELS.get(self.stem, "UNKNOWN (no label in config)")


def find_captures(folder: Path | str | None = None) -> list[CaptureFile]:
    """Recursively find every .pcapng / .pcap under `folder` (case-insensitive).

    Returns them sorted by relative path.  An empty list means nothing matched --
    it never raises, so the caller can report the problem nicely.
    """
    root = Path(folder) if folder is not None else config.PCAPNG_DIR
    out: list[CaptureFile] = []
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in sorted(filenames):
            p = Path(dirpath) / name
            # .lower() on the suffix is what makes this case-insensitive.
            if p.suffix.lower() not in config.CAPTURE_EXTENSIONS:
                continue
            out.append(CaptureFile(path=p, root=root, size_bytes=p.stat().st_size))
    out.sort(key=lambda c: c.rel_path.lower())
    return out


def duplicate_stems(captures: Sequence[CaptureFile]) -> dict[str, list[str]]:
    """Return {lower-cased stem: [relative paths]} for stems used more than once.

    Two captures with the same stem would both want to become <stem>.pcap and
    one would overwrite the other, so output_pcap_name() prefixes them instead.
    """
    seen: dict[str, list[str]] = {}
    for c in captures:
        seen.setdefault(c.stem.lower(), []).append(c.rel_path)
    return {k: v for k, v in seen.items() if len(v) > 1}


def _sanitise(text: str) -> str:
    """Make a folder name safe to use inside a file name."""
    out = []
    for ch in text:
        out.append(ch if (ch.isalnum() or ch in "-_") else "_")
    return "".join(out).strip("_") or "root"


def output_pcap_name(capture: CaptureFile, captures: Sequence[CaptureFile]) -> str:
    """The .pcap file name for `capture`, unique within `captures`.

    Normal case:        N-0.pcapng            -> N-0.pcap
    Duplicate stem:     extra\\N-0.PCAPNG      -> extra_N-0.pcap
    Still clashing:     a second 'extra_N-0'   -> extra_N-0__2.pcap
    """
    dupes = duplicate_stems(captures)
    if capture.stem.lower() not in dupes:
        return capture.stem + ".pcap"

    parent = capture.path.parent
    prefix = _sanitise(parent.name) if parent != capture.root else "root"
    base = "%s_%s" % (prefix, capture.stem)

    # Build the same name for every capture, then break any remaining tie by
    # position, so the result is deterministic and never collides.
    taken: list[str] = []
    for c in captures:
        if c.stem.lower() not in dupes:
            continue
        c_parent = c.path.parent
        c_prefix = _sanitise(c_parent.name) if c_parent != c.root else "root"
        c_base = "%s_%s" % (c_prefix, c.stem)
        n = taken.count(c_base)
        taken.append(c_base)
        name = c_base if n == 0 else "%s__%d" % (c_base, n + 1)
        if c.path == capture.path:
            return name + ".pcap"
    return base + ".pcap"   # unreachable in practice; kept as a safe default


def inventory_rows(captures: Sequence[CaptureFile]) -> list[dict[str, Any]]:
    """One plain dict per capture, ready for a CSV or a DataFrame."""
    return [
        {
            "stem": c.stem,
            "file_name": c.path.name,
            "extension": c.ext,
            "size_mb": c.size_mb,
            "size_bytes": c.size_bytes,
            "expected_label": c.expected_label,
            "relative_path": c.rel_path,
            "output_pcap": output_pcap_name(c, captures),
            "full_path": str(c.path),
        }
        for c in captures
    ]


INVENTORY_FIELDS = ["stem", "file_name", "extension", "size_mb", "size_bytes",
                    "expected_label", "relative_path", "output_pcap", "full_path"]


def write_inventory(captures: Sequence[CaptureFile], path: Path | None = None) -> Path:
    """Write the inventory CSV (stdlib csv, so pandas is not required)."""
    out = Path(path) if path is not None else config.PCAP_INVENTORY_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INVENTORY_FIELDS)
        w.writeheader()
        for row in inventory_rows(captures):
            w.writerow({k: row[k] for k in INVENTORY_FIELDS})
    return out


# ===========================================================================
# external tools
# ===========================================================================
def build_editcap_cmd(src: Path | str, dst: Path | str,
                      editcap: str | None = None) -> list[str]:
    """editcap -F pcap <src> <dst>, as a LIST (spaces in paths stay safe)."""
    return [editcap or config.EDITCAP_EXE, "-F", "pcap", str(src), str(dst)]


def build_capinfos_count_cmd(path: Path | str, capinfos: str | None = None) -> list[str]:
    """capinfos -c -M <file>: packet count only, machine-readable, fast."""
    return [capinfos or config.CAPINFOS_EXE, "-c", "-M", str(path)]


def build_tshark_count_cmd(path: Path | str, tshark: str | None = None) -> list[str]:
    """tshark printing one line per packet -- we count the lines."""
    return [tshark or config.TSHARK_EXE, "-r", str(path),
            "-T", "fields", "-e", "frame.number"]


def build_tshark_ip_count_cmd(path: Path | str, tshark: str | None = None) -> list[str]:
    """Same, but only IPv4 packets (-Y ip).  Used by the Stage D cross-check."""
    return [tshark or config.TSHARK_EXE, "-r", str(path), "-Y", "ip",
            "-T", "fields", "-e", "frame.number"]


def run_cmd(cmd: Sequence[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run an external program.  LIST of args, never shell=True."""
    if not Path(cmd[0]).is_file() and shutil.which(cmd[0]) is None:
        raise ToolError("program not found: %s" % cmd[0])
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)


def tool_version(exe: str) -> str:
    """First line of `<exe> -v`, or a message explaining why it failed."""
    if not Path(exe).is_file() and shutil.which(exe) is None:
        return "NOT FOUND at %s" % exe
    try:
        res = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=120)
    except Exception as exc:
        return "found but would not run: %r" % (exc,)
    text = (res.stdout or res.stderr or "").strip().splitlines()
    return text[0] if text else "(no version output)"


def count_packets(path: Path | str) -> int:
    """Total packets in a capture.

    capinfos reads only the file index, so it is much faster than walking every
    packet.  If capinfos is missing we fall back to counting tshark's output.
    """
    try:
        res = run_cmd(build_capinfos_count_cmd(path), timeout=1800)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                # "-M" prints e.g.  Number of packets = 651234
                digits = "".join(ch for ch in line if ch.isdigit())
                if digits and "packet" in line.lower():
                    return int(digits)
    except ToolError:
        pass   # capinfos not installed -> use tshark instead

    res = run_cmd(build_tshark_count_cmd(path), timeout=7200)
    if res.returncode != 0:
        raise ToolError("tshark failed on %s: %s" % (path, (res.stderr or "").strip()[:400]))
    return sum(1 for line in res.stdout.splitlines() if line.strip())


def count_ipv4_packets(path: Path | str) -> int:
    """Independent IPv4 packet count, for the Stage D verification table."""
    res = run_cmd(build_tshark_ip_count_cmd(path), timeout=7200)
    if res.returncode != 0:
        raise ToolError("tshark -Y ip failed on %s: %s"
                        % (path, (res.stderr or "").strip()[:400]))
    return sum(1 for line in res.stdout.splitlines() if line.strip())


# ===========================================================================
# STAGE A -- check what exists
# ===========================================================================
def stage_a_paths(rep: Report) -> None:
    """Print every configured path and whether it exists."""
    rep.say("  %-32s %-8s %s" % ("NAME", "EXISTS", "PATH"))
    rep.say("  " + "-" * 100)
    for name, p in config.all_paths().items():
        rep.say("  %-32s %-8s %s" % (name, "yes" if Path(p).exists() else "NO", p))


def stage_a_disk_layout(rep: Report, root: Path | None = None, max_entries: int = 40) -> None:
    """List DATA_ROOT two levels deep, so the real layout is visible."""
    root = Path(root) if root is not None else config.DATA_ROOT
    if not root.is_dir():
        rep.bad("DATA_ROOT does not exist: %s" % root)
        return
    rep.say("  %s\\" % root)
    for level1 in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if level1.is_file():
            rep.say("  |-- %-26s %10.2f MB" % (level1.name, mb(level1.stat().st_size)))
            continue
        try:
            children = sorted(level1.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            rep.say("  |-- %-26s <dir>  (permission denied)" % (level1.name + "\\"))
            continue
        rep.say("  |-- %-26s <dir>   %d files, %d subfolders"
                % (level1.name + "\\",
                   sum(1 for c in children if c.is_file()),
                   sum(1 for c in children if c.is_dir())))
        for c in children[:max_entries]:
            if c.is_dir():
                rep.say("  |     |-- %-22s <dir>" % (c.name + "\\"))
            else:
                rep.say("  |     |-- %-22s %10.2f MB" % (c.name, mb(c.stat().st_size)))
        if len(children) > max_entries:
            rep.say("  |     |-- ... %d more entries" % (len(children) - max_entries))


def locate_capture_dir(rep: Report | None = None) -> tuple[Path | None, str]:
    """config.PCAPNG_DIR, or a folder found by searching DATA_ROOT.

    Fallback order: a folder named 'pcapngtopcap', then any folder that actually
    contains .pcapng files.  Returns (folder or None, reason).
    """
    def _warn(msg: str) -> None:
        if rep is not None:
            rep.warn(msg)

    if config.PCAPNG_DIR.is_dir():
        return config.PCAPNG_DIR, "config.PCAPNG_DIR (the expected location)"
    _warn("config.PCAPNG_DIR does not exist: %s" % config.PCAPNG_DIR)
    if not config.DATA_ROOT.is_dir():
        _warn("config.DATA_ROOT does not exist either: %s" % config.DATA_ROOT)
        return None, "not found"

    named: list[Path] = []
    with_caps: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(config.DATA_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        d = Path(dirpath)
        flat = d.name.lower().replace("_", "").replace("-", "").replace(" ", "")
        if flat == "pcapngtopcap":
            named.append(d)
        if any(f.lower().endswith(".pcapng") for f in filenames):
            with_caps.append(d)
    for cands, why in ((named, "a folder named 'pcapngtopcap', found by searching DATA_ROOT"),
                       (with_caps, "a folder containing .pcapng files, found by searching DATA_ROOT")):
        if cands:
            cands.sort(key=lambda p: len(p.parts))
            if len(cands) > 1:
                _warn("several candidates: %s -- using the shallowest" % [str(p) for p in cands])
            return cands[0], why
    return None, "not found"


OLD_CODE_KEYWORDS = ("cicflow", "flow", "convert", "pcap")


def find_old_code(root: Path | None = None) -> list[Path]:
    """Find *.py / *.ipynb whose NAME hints at an earlier converter."""
    root = Path(root) if root is not None else config.DATA_ROOT
    ours = {(_PROJECT_DIR / n).resolve() for n in
            ("config.py", "stage_a_check.py", "src/converter.py",
             "src/__init__.py", "tests/test_converter.py")}
    hits: list[Path] = []
    if not root.is_dir():
        return hits
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() not in (".py", ".ipynb"):
                continue
            if p.resolve() in ours or p.name in ("converter.py", "config.py"):
                continue
            if any(k in name.lower() for k in OLD_CODE_KEYWORDS):
                hits.append(p)
    hits.sort(key=lambda p: str(p).lower())
    return hits


def notebook_to_text(path: Path) -> str:
    """A notebook's cells as plain text (markdown + code), for reading it."""
    try:
        nb = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return "<<could not parse notebook: %r>>" % (exc,)
    parts = []
    for i, cell in enumerate(nb.get("cells", []), start=1):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        parts.append("### cell %d [%s]\n%s" % (i, cell.get("cell_type", "?"), src))
    return "\n\n".join(parts) if parts else "<<notebook has no cells>>"


def stage_a_old_code(rep: Report, console_limit: int = 120,
                     report_name: str = "the saved report") -> list[Path]:
    """List any earlier converter code and dump its source so it can be reused."""
    hits = find_old_code()
    if not hits:
        rep.ok("no older converter script or notebook found under %s" % config.DATA_ROOT)
        rep.say("          Nothing to reuse -- src/converter.py is written from scratch.")
        return hits
    rep.warn("found %d file(s) that may be an earlier converter:" % len(hits))
    for p in hits:
        rep.say("          %-62s %8.2f MB" % (p, mb(p.stat().st_size)))
    rep.say()
    rep.say("  They are NOT modified.  Their full source is in %s." % report_name)
    for p in hits:
        text = notebook_to_text(p) if p.suffix.lower() == ".ipynb" \
            else p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        rep.sub("SOURCE OF %s  (%d lines)" % (p, len(lines)))
        for i, line in enumerate(lines, start=1):
            tagged = "  %5d| %s" % (i, line)
            rep.say(tagged) if i <= console_limit else rep.quiet(tagged)
        if len(lines) > console_limit:
            rep.say("  ... %d more lines -- full text is in %s"
                    % (len(lines) - console_limit, report_name))
    return hits


def stage_a_inventory(rep: Report, captures: Sequence[CaptureFile]) -> None:
    """Print the capture table, missing/extra captures and duplicate stems."""
    if not captures:
        rep.bad("no .pcapng / .pcap files found")
        return
    rep.say("  %-16s %-8s %11s  %-30s %s"
            % ("STEM", "EXT", "SIZE (MB)", "EXPECTED LABEL", "RELATIVE PATH"))
    rep.say("  " + "-" * 100)
    total = 0.0
    for c in captures:
        total += c.size_mb
        rep.say("  %-16s %-8s %11.2f  %-30s %s"
                % (c.stem, c.ext, c.size_mb, c.expected_label, c.rel_path))
    rep.say("  " + "-" * 100)
    n_ng = sum(1 for c in captures if c.ext == ".pcapng")
    n_p = sum(1 for c in captures if c.ext == ".pcap")
    rep.say("  TOTAL: %d capture files (%d .pcapng + %d .pcap), %.2f MB"
            % (len(captures), n_ng, n_p, total))

    found = {c.stem for c in captures}
    expected = list(config.CAPTURE_LABELS)
    missing = [s for s in expected if s not in found]
    extra = sorted(s for s in found if s not in config.CAPTURE_LABELS)
    rep.say()
    rep.say("  documented captures present : %s" % ([s for s in expected if s in found] or "NONE"))
    if missing:
        rep.warn("documented captures MISSING : %s" % missing)
    else:
        rep.ok("all %d documented captures are present" % len(expected))
    if extra:
        rep.warn("capture files with no label in config.CAPTURE_LABELS: %s" % extra)

    dupes = duplicate_stems(captures)
    if dupes:
        rep.warn("DUPLICATE stems -- the subfolder name is used as a prefix:")
        for stem, paths in dupes.items():
            rep.say("            %s -> %s" % (stem, paths))
        for c in captures:
            if c.stem.lower() in dupes:
                rep.say("            %-34s -> %s" % (c.rel_path, output_pcap_name(c, captures)))
    else:
        rep.ok("every capture stem is unique, so no prefixing is needed")


# (import name, required version or None)
REQUIRED_PACKAGES = [
    ("cicflowmeter", "0.5.0"),
    ("scapy", None),
    ("numpy", None),
    ("scipy", None),
    ("pandas", None),
    ("tqdm", None),
    ("pytest", None),
    ("jupyter_core", None),
    ("notebook", None),
]

_PROBE = r"""
import importlib.metadata as md, importlib.util, json, sys
names = %s
out = {"version_info": list(sys.version_info[:3]), "executable": sys.executable, "packages": {}}
for n in names:
    info = {"importable": importlib.util.find_spec(n) is not None, "version": None}
    for dist in (n, {"jupyter_core": "jupyter-core"}.get(n, n)):
        try:
            info["version"] = md.version(dist); break
        except Exception:
            pass
    out["packages"][n] = info
print("PROBE_JSON:" + json.dumps(out))
""" % ([n for n, _ in REQUIRED_PACKAGES],)


def probe_interpreter(rep: Report) -> tuple[dict | None, str]:
    """Ask config.PYTHON_EXE about its own version and packages."""
    exe = config.PYTHON_EXE
    if not Path(exe).is_file():
        rep.bad("config.PYTHON_EXE does not exist: %s" % exe)
        rep.warn("falling back to the interpreter running this code: %s" % sys.executable)
        exe = sys.executable
    try:
        res = subprocess.run([exe, "-c", _PROBE], capture_output=True, text=True, timeout=300)
    except Exception as exc:
        rep.bad("could not run %s : %r" % (exe, exc))
        return None, exe
    if res.returncode != 0:
        rep.bad("the probe failed (exit code %d)" % res.returncode)
        rep.say("    stderr: " + (res.stderr or "").strip()[:800])
        return None, exe
    for line in res.stdout.splitlines():
        if line.startswith("PROBE_JSON:"):
            return json.loads(line[len("PROBE_JSON:"):]), exe
    rep.bad("the probe printed no JSON; stdout was: " + (res.stdout or "").strip()[:400])
    return None, exe


def stage_a_environment(rep: Report) -> list[str]:
    """Check Python, the packages, jupyter, pytest, editcap, tshark, capinfos."""
    info, exe = probe_interpreter(rep)
    rep.say("  interpreter : %s" % exe)
    missing: list[str] = []
    if info is None:
        missing = [n for n, _ in REQUIRED_PACKAGES]
    else:
        v = info["version_info"]
        rep.say("  version     : %d.%d.%d" % tuple(v))
        if (v[0], v[1]) == (3, 13):
            rep.ok("Python 3.13 confirmed")
        else:
            rep.warn("expected Python 3.13, found %d.%d -- cicflowmeter 0.5.0 declares an "
                     "older python_requires, hence --ignore-requires-python" % (v[0], v[1]))
        rep.say()
        rep.say("  %-14s %-12s %-10s %s" % ("PACKAGE", "INSTALLED", "REQUIRED", "STATUS"))
        rep.say("  " + "-" * 58)
        for name, required in REQUIRED_PACKAGES:
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
            rep.say("  %-14s %-12s %-10s %s" % (name, shown, required or "any", status))

        rep.sub("jupyter and pytest, as commands")
        for label, args in (("jupyter", ["-m", "jupyter", "--version"]),
                            ("pytest", ["-m", "pytest", "--version"])):
            try:
                res = subprocess.run([exe, *args], capture_output=True, text=True, timeout=300)
                first = (res.stdout or res.stderr or "").strip().splitlines()
                if res.returncode == 0:
                    rep.ok("%-8s %s" % (label, first[0] if first else "(no output)"))
                else:
                    rep.bad("%-8s exit %d: %s" % (label, res.returncode,
                                                  (first[0] if first else "")[:200]))
                    missing.append(label)
            except Exception as exc:
                rep.bad("%-8s would not run: %r" % (label, exc))
                missing.append(label)

    rep.sub("Wireshark command-line tools")
    tools_ok = True
    for label, exe_path in (("editcap", config.EDITCAP_EXE),
                            ("tshark", config.TSHARK_EXE),
                            ("capinfos", config.CAPINFOS_EXE)):
        line = tool_version(exe_path)
        if line.startswith("NOT FOUND") or line.startswith("found but"):
            rep.bad("%-9s %s" % (label, line))
            if label != "capinfos":      # capinfos is optional: tshark can do it
                tools_ok = False
        else:
            rep.ok("%-9s %s" % (label, line))
    if not tools_ok:
        rep.say()
        rep.say("  Install Wireshark -- it ships editcap, tshark AND capinfos:")
        rep.say("      https://www.wireshark.org/download.html")
        rep.say("  Accept the default folder so the paths become")
        rep.say(r"      C:\Program Files\Wireshark\editcap.exe  (and tshark.exe, capinfos.exe)")
        rep.say("  Installed elsewhere? Either edit EDITCAP_EXE / TSHARK_EXE in config.py,")
        rep.say("  or add the Wireshark folder to PATH and RESTART Jupyter -- config.py")
        rep.say("  falls back to shutil.which(), which reads the PATH.")

    if missing:
        rep.say()
        rep.warn("missing or wrong version: %s" % missing)
        rep.say("  Run these in cmd (quotes matter -- the path has spaces):")
        rep.say()
        rep.say('      "%s" -m pip install scapy numpy scipy pandas requests tqdm jupyter pytest'
                % config.PYTHON_EXE)
        rep.say('      "%s" -m pip install cicflowmeter==0.5.0 --no-deps --ignore-requires-python'
                % config.PYTHON_EXE)
        rep.say()
        rep.say("  cicflowmeter 0.5.0 pins old dependencies and declares python_requires < 3.13,")
        rep.say("  so --no-deps (the first line installs the libraries) and")
        rep.say("  --ignore-requires-python are both required on Python 3.13.")
    else:
        rep.ok("every required package is installed")
    return missing


def read_kaggle(rep: Report):
    """Read the Kaggle reference file; print shape, numbered columns, labels.

    Returns (DataFrame or None, list of column names or None).
    """
    try:
        import pandas as pd
    except ImportError:
        rep.bad("pandas is not installed in this interpreter (%s)" % sys.executable)
        return None, None

    path = config.find_kaggle_file()
    if path is None:
        rep.bad("no .csv / .xlsx found in %s" % config.KAGGLE_DIR)
        return None, None
    rep.ok("Kaggle file: %s  (%.2f MB)" % (path, mb(path.stat().st_size)))
    try:
        df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") \
            else pd.read_csv(path, low_memory=False)
    except Exception as exc:
        rep.bad("could not read it: %r" % (exc,))
        if path.suffix.lower() in (".xlsx", ".xls"):
            rep.say('          .xlsx also needs:  "%s" -m pip install openpyxl' % config.PYTHON_EXE)
        return None, None

    rep.say("  shape (rows, columns): %s" % (df.shape,))
    cols = [str(c) for c in df.columns]
    rep.sub("columns, numbered, EXACTLY as stored (|...| marks the string bounds)")
    for i, c in enumerate(cols, start=1):
        flag = "   <-- leading/trailing whitespace!" if c != c.strip() else ""
        rep.say("  %3d  |%s|%s" % (i, c, flag))

    rep.sub("Label value counts")
    label_col = next((c for c in cols if c.strip().lower() == "label"), None)
    if label_col is None:
        rep.bad("no 'Label' column found")
    else:
        if label_col != config.LABEL_COLUMN:
            rep.warn("the label column is stored as |%s|" % label_col)
        vc = df[label_col].value_counts(dropna=False)
        for value, count in vc.items():
            rep.say("  %-18s %9d   (%5.2f%%)" % (str(value), count, 100.0 * count / len(df)))
        rep.say("  %-18s %9d" % ("TOTAL", len(df)))
        mine = set(config.CAPTURE_LABELS.values()) | {config.BENIGN_LABEL}
        theirs = {str(v) for v in vc.index}
        only_theirs, only_mine = sorted(theirs - mine), sorted(mine - theirs)
        rep.say()
        if only_theirs:
            rep.warn("in Kaggle but NOT in config.CAPTURE_LABELS: %s" % only_theirs)
        if only_mine:
            rep.warn("in config.CAPTURE_LABELS but NOT in Kaggle: %s   "
                     "('Benign' belongs here -- we create it ourselves)" % only_mine)
        if not only_theirs and only_mine == [config.BENIGN_LABEL]:
            rep.ok("every Kaggle label spelling matches config.CAPTURE_LABELS exactly")
    return df, cols


def compare_headers(rep: Report, kaggle_cols: Sequence[str] | None) -> list[str] | None:
    """Diff config.KAGGLE_FEATURES against the real header.  The header wins.

    Returns the Kaggle feature names when they could be read, else None.
    """
    rep.say("  config.FLOW_COLUMNS : %d columns = %d identifiers + %d features + Label"
            % (len(config.FLOW_COLUMNS), len(config.ID_COLUMNS), len(config.KAGGLE_FEATURES)))
    rep.say("  (this list was described as '72 columns'; counted item by item it is")
    rep.say("   6 + 66 + 1 = 73.  The 66 feature names themselves are unchanged.)")
    if kaggle_cols is None:
        rep.warn("the Kaggle header could not be read, so no comparison was possible")
        return None

    clean = [str(c).strip() for c in kaggle_cols]
    feats = [c for c in clean if c.lower() != "label"]
    ref = config.KAGGLE_FEATURES
    rep.say("  Kaggle feature columns (Label excluded) : %d" % len(feats))

    only_kaggle = [c for c in feats if c not in ref]
    only_ref = [c for c in ref if c not in feats]
    if only_kaggle:
        rep.bad("in Kaggle but NOT in config.KAGGLE_FEATURES (%d): %s" % (len(only_kaggle), only_kaggle))
    if only_ref:
        rep.bad("in config.KAGGLE_FEATURES but NOT in Kaggle (%d): %s" % (len(only_ref), only_ref))
    if not only_kaggle and not only_ref:
        rep.ok("identical set of %d feature names" % len(feats))
        diffs = [(i + 1, a, b) for i, (a, b) in enumerate(zip(ref, feats)) if a != b]
        if diffs:
            rep.bad("same names, DIFFERENT ORDER at %d position(s):" % len(diffs))
            for pos, a, b in diffs:
                rep.say("      position %2d: config |%s|   Kaggle |%s|" % (pos, a, b))
        else:
            rep.ok("identical order -- config.KAGGLE_FEATURES needs no change")
    if only_kaggle or only_ref or len(feats) != len(ref):
        rep.say()
        rep.say("  ==> The Kaggle header WINS.  Paste this over KAGGLE_FEATURES in config.py:")
        rep.say("      KAGGLE_FEATURES = [")
        for c in feats:
            rep.say('          "%s",' % c)
        rep.say("      ]")

    rep.sub("RENAME_MAP coverage")
    targets = set(config.RENAME_MAP.values())
    needed = set(config.FLOW_COLUMNS) - {config.LABEL_COLUMN}
    uncovered = sorted(needed - targets)
    values = list(config.RENAME_MAP.values())
    dupes = sorted({v for v in values if values.count(v) > 1})
    rep.say("  RENAME_MAP entries: %d  (must cover %d columns)" % (len(config.RENAME_MAP), len(needed)))
    if uncovered:
        rep.bad("columns with NO RENAME_MAP entry: %s" % uncovered)
    else:
        rep.ok("every output column except 'Label' has exactly one RENAME_MAP entry")
    if dupes:
        rep.bad("two extractor names map to the same output column: %s" % dupes)
    rep.say("  extractor columns we expect to DROP (%d): %s"
            % (len(config.EXPECTED_EXTRA_COLUMNS), config.EXPECTED_EXTRA_COLUMNS))
    return feats


def stage_a_report(echo: bool = True, save_to: Path | None = None) -> Report:
    """Run every Stage A check.  Read-only apart from the saved report file."""
    rep = Report(echo=echo)
    save_to = Path(save_to) if save_to is not None else (_PROJECT_DIR / "stage_a_report.txt")

    rep.header("STAGE A -- CHECK WHAT EXISTS  (read-only)")
    rep.say("  DATA_ROOT    : %s" % config.DATA_ROOT)
    rep.say("  PROJECT_ROOT : %s" % config.PROJECT_ROOT)
    rep.say("  Converts nothing, modifies nothing.  Only writes %s" % save_to)

    rep.header("A.0  configured paths")
    stage_a_paths(rep)

    rep.header("A.1  what is on disk (two levels deep)")
    stage_a_disk_layout(rep)
    rep.sub("the capture folder")
    capture_dir, why = locate_capture_dir(rep)
    if capture_dir is not None:
        rep.ok("using: %s" % capture_dir)
        rep.say("          reason: %s" % why)
    else:
        rep.bad("no capture folder found under %s" % config.DATA_ROOT)

    rep.header("A.2  older converter code (read, never modified)")
    stage_a_old_code(rep, report_name=save_to.name)

    rep.header("A.3  capture inventory (recursive, case-insensitive)")
    captures = find_captures(capture_dir) if capture_dir else []
    stage_a_inventory(rep, captures)
    rep.say()
    rep.say("  (Nothing written yet -- outputs\\step1\\pcap_inventory.csv is a Stage B output.)")

    rep.header("A.4  environment")
    stage_a_environment(rep)

    rep.header("A.5  Kaggle MSCAD reference file (read-only)")
    _df, cols = read_kaggle(rep)

    rep.header("A.6  column names: config vs the Kaggle header")
    compare_headers(rep, cols)

    rep.header("STAGE A FINISHED")
    rep.say("  Nothing was converted, moved, renamed or deleted.")
    try:
        rep.save(save_to)
        if echo:
            print("  full report written to: %s" % save_to)
    except Exception as exc:
        if echo:
            print("  [WARN] could not write the report file: %r" % (exc,))
    return rep


# ===========================================================================
# STAGE B -- pcapng -> pcap
# ===========================================================================
def convert_one(capture: CaptureFile, out_path: Path,
                overwrite: bool | None = None, verify: bool = True) -> dict[str, Any]:
    """Convert (or copy) ONE capture into out_path.  Never touches the input.

    Returns a dict with status / counts / error, ready for the inventory table.
    status is one of: converted, copied, skipped, failed.
    """
    overwrite = config.OVERWRITE if overwrite is None else overwrite
    row: dict[str, Any] = {
        "stem": capture.stem,
        "source": str(capture.path),
        "source_ext": capture.ext,
        "source_mb": capture.size_mb,
        "output": str(out_path),
        "status": "",
        "src_packets": "",
        "out_packets": "",
        "packets_match": "",
        "output_mb": "",
        "error": "",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        row["status"] = "skipped"
        row["output_mb"] = mb(out_path.stat().st_size)
        row["error"] = "output already exists and config.OVERWRITE is False"
        return row

    try:
        if capture.is_already_pcap:
            # Already a .pcap: COPY it.  copy2 keeps the timestamps, and the
            # original stays exactly where it is.
            shutil.copy2(capture.path, out_path)
            row["status"] = "copied"
        else:
            cmd = build_editcap_cmd(capture.path, out_path)
            res = run_cmd(cmd, timeout=7200)
            if res.returncode != 0:
                row["status"] = "failed"
                row["error"] = "editcap exit %d: %s" % (res.returncode,
                                                        (res.stderr or "").strip()[:500])
                return row
            if not out_path.exists():
                row["status"] = "failed"
                row["error"] = "editcap reported success but %s does not exist" % out_path
                return row
            row["status"] = "converted"
        row["output_mb"] = mb(out_path.stat().st_size)

        if verify:
            src_n = count_packets(capture.path)
            out_n = count_packets(out_path)
            row["src_packets"] = src_n
            row["out_packets"] = out_n
            row["packets_match"] = (src_n == out_n)
            if src_n != out_n:
                row["status"] = "failed"
                row["error"] = ("packet count mismatch: source %d vs output %d "
                                "(a conversion must not lose packets)" % (src_n, out_n))
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    return row


def convert_all(captures: Sequence[CaptureFile] | None = None,
                overwrite: bool | None = None, verify: bool = True,
                rep: Report | None = None, use_tqdm: bool = True) -> list[dict[str, Any]]:
    """Stage B for every capture.  Returns one row per capture."""
    rep = rep or Report()
    if captures is None:
        capture_dir, _why = locate_capture_dir(rep)
        captures = find_captures(capture_dir) if capture_dir else []
    if not captures:
        rep.bad("no captures to convert")
        return []

    config.PCAP_DIR.mkdir(parents=True, exist_ok=True)
    iterator: Iterable[CaptureFile] = captures
    if use_tqdm:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(captures, desc="pcapng -> pcap", unit="file")
        except ImportError:
            pass

    rows: list[dict[str, Any]] = []
    for cap in iterator:
        out = config.PCAP_DIR / output_pcap_name(cap, captures)
        rows.append(convert_one(cap, out, overwrite=overwrite, verify=verify))

    rep.sub("Stage B results")
    rep.say("  %-16s %-10s %12s %12s %7s  %s"
            % ("STEM", "STATUS", "SRC PKTS", "OUT PKTS", "MATCH", "OUTPUT"))
    rep.say("  " + "-" * 104)
    for r in rows:
        rep.say("  %-16s %-10s %12s %12s %7s  %s"
                % (r["stem"], r["status"], r["src_packets"], r["out_packets"],
                   r["packets_match"], Path(r["output"]).name))
    rep.say("  " + "-" * 104)
    for status in ("converted", "copied", "skipped", "failed"):
        n = sum(1 for r in rows if r["status"] == status)
        rep.say("  %-10s %d" % (status, n))
    bad_rows = [r for r in rows if r["status"] == "failed"]
    if bad_rows:
        rep.bad("%d capture(s) FAILED -- nothing is skipped silently:" % len(bad_rows))
        for r in bad_rows:
            rep.say("          %s: %s" % (r["stem"], r["error"]))
    else:
        rep.ok("every capture converted or copied, and all packet counts match")
    return rows


# ===========================================================================
# STAGE C -- the slim packet record and the flow key
# ===========================================================================
# Protocol numbers we care about.
PROTO_ICMP = 1
PROTO_TCP = 6
PROTO_UDP = 17


@dataclass(slots=True)
class PacketRecord:
    """A SLIM per-packet record -- fix #6 for cicflowmeter 0.5.0.

    Stock cicflowmeter keeps the whole scapy packet object for every packet of
    every open flow, which exhausts memory on the flood captures.  We copy out
    only the handful of fields the 66 features actually need, and let scapy throw
    the packet away immediately.

    time        capture timestamp, in SECONDS (float, as scapy gives it)
    cap_len     CAPTURED bytes of the packet (do not rebuild the packet)
    payload_len bytes of TCP/UDP payload (0 for ICMP and for bare ACKs)
    header_len  IP + transport header length, for Fwd/Bwd Header Len
    flags       raw TCP flag byte; 0 when there is no TCP layer
    window      TCP window size; 0 when there is no TCP layer
    """

    time: float
    cap_len: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    flags: int = 0
    payload_len: int = 0
    header_len: int = 0
    window: int = 0

    # --- TCP flag helpers.  Fix #4: these read the TCP flag byte, NOT the IP
    # --- flags field, which is what stock cicflowmeter wrongly looked at.
    @property
    def fin(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def syn(self) -> bool:
        return bool(self.flags & 0x02)

    @property
    def rst(self) -> bool:
        return bool(self.flags & 0x04)

    @property
    def psh(self) -> bool:
        return bool(self.flags & 0x08)

    @property
    def ack(self) -> bool:
        return bool(self.flags & 0x10)

    @property
    def urg(self) -> bool:
        return bool(self.flags & 0x20)

    @property
    def ece(self) -> bool:
        return bool(self.flags & 0x40)

    @property
    def cwr(self) -> bool:
        return bool(self.flags & 0x80)


def flow_key(rec: PacketRecord) -> tuple[str, str, int, int, int]:
    """The 5-tuple that identifies a flow: (src ip, dst ip, src port, dst port, proto).

    Fix #2: stock cicflowmeter throws away everything that is not TCP or UDP, so
    the App-02 ICMP flood produced no flows at all.  We keep every IPv4 packet;
    anything without ports (ICMP and the rest) gets port 0 on both sides, and
    keeps its real IP protocol number -- 1 for ICMP.
    """
    if rec.protocol in (PROTO_TCP, PROTO_UDP):
        return (rec.src_ip, rec.dst_ip, int(rec.src_port), int(rec.dst_port), int(rec.protocol))
    return (rec.src_ip, rec.dst_ip, 0, 0, int(rec.protocol))


def reverse_flow_key(rec: PacketRecord) -> tuple[str, str, int, int, int]:
    """The key of the same flow seen from the other side.

    A flow's first packet fixes the "forward" direction; later packets coming
    back match this reversed key and are counted as backward packets.
    """
    src, dst, sp, dp, proto = flow_key(rec)
    return (dst, src, dp, sp, proto)


def packet_record_from_scapy(pkt) -> PacketRecord | None:
    """Turn one scapy packet into a PacketRecord, or None if it is not IPv4.

    scapy is imported lazily so this module (and the test suite) still work on a
    machine where scapy is not installed.
    """
    from scapy.layers.inet import ICMP, IP, TCP, UDP   # noqa: WPS433 (lazy on purpose)

    if IP not in pkt:
        return None            # ARP / IPv6 / anything else: excluded by design
    ip = pkt[IP]
    # Fix: packet length is the CAPTURED byte count.  len(pkt) would re-serialise
    # the packet, which is slow and can disagree with what is on the wire.
    cap_len = len(bytes(pkt)) if not hasattr(pkt, "wirelen") or pkt.wirelen is None \
        else int(pkt.wirelen)
    ip_header_len = int(ip.ihl) * 4 if getattr(ip, "ihl", None) else 20

    if TCP in pkt:
        t = pkt[TCP]
        tcp_header_len = int(t.dataofs) * 4 if getattr(t, "dataofs", None) else 20
        return PacketRecord(
            time=float(pkt.time), cap_len=cap_len,
            src_ip=str(ip.src), dst_ip=str(ip.dst),
            src_port=int(t.sport), dst_port=int(t.dport), protocol=PROTO_TCP,
            flags=int(t.flags), payload_len=len(bytes(t.payload)),
            header_len=ip_header_len + tcp_header_len, window=int(t.window),
        )
    if UDP in pkt:
        u = pkt[UDP]
        return PacketRecord(
            time=float(pkt.time), cap_len=cap_len,
            src_ip=str(ip.src), dst_ip=str(ip.dst),
            src_port=int(u.sport), dst_port=int(u.dport), protocol=PROTO_UDP,
            flags=0, payload_len=max(0, int(u.len) - 8),
            header_len=ip_header_len + 8, window=0,
        )
    # ICMP and every other IPv4 protocol: ports 0, real protocol number.
    return PacketRecord(
        time=float(pkt.time), cap_len=cap_len,
        src_ip=str(ip.src), dst_ip=str(ip.dst),
        src_port=0, dst_port=0, protocol=int(ip.proto),
        flags=0, payload_len=len(bytes(ip.payload.payload)) if ICMP in pkt else 0,
        header_len=ip_header_len + (8 if ICMP in pkt else 0), window=0,
    )


def extract_flows(pcap_path: Path | str, **kwargs):
    """Read a .pcap and return the raw per-flow DataFrame.  STAGE C -- not yet written.

    Everything it will need is already here and unit-tested: PacketRecord
    (fix #6), flow_key (fix #2), the 120 s / 240 s timeouts in config, the TCP
    flag helpers (fix #4) and the post-processing below.  What is missing is the
    flow table itself: accumulating the features, the per-flow timeout check plus
    periodic sweep (fix #7), and the first-packet double-count fix (fix #5).
    """
    raise NotImplementedError(
        "extract_flows() is Stage C and is not implemented yet.\n"
        "Stage B (pcapng -> pcap) and all the post-processing below are ready.\n"
        "Say 'go' for Stage C and this is the next thing written."
    )


# ===========================================================================
# STAGE C -- post-processing (pure pandas, no captures needed)
# ===========================================================================
def strip_column_names(df):
    """Remove stray leading/trailing spaces from the column names."""
    return df.rename(columns={c: str(c).strip() for c in df.columns})


def apply_rename_map(df, rename_map: dict[str, str] | None = None):
    """Rename the extractor's snake_case columns to the MSCAD names."""
    return df.rename(columns=dict(rename_map or config.RENAME_MAP))


def check_required_columns(df, required: Sequence[str] | None = None) -> None:
    """Raise MissingColumnError if any required output column is absent.

    Never fabricates a column: a missing feature means the extractor is wrong,
    and a made-up value would silently corrupt every later stage.
    """
    needed = list(required) if required is not None else \
        [c for c in config.FLOW_COLUMNS if c != config.LABEL_COLUMN]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise MissingColumnError(
            "%d required column(s) missing after renaming: %s. "
            "Add the matching entry to config.RENAME_MAP -- do not fill in a value."
            % (len(missing), missing))


def split_extra_columns(df, keep: Sequence[str] | None = None):
    """Drop columns MSCAD does not keep.  Returns (df_without_extras, dropped)."""
    keep_list = list(keep) if keep is not None else list(config.FLOW_COLUMNS)
    dropped = [c for c in df.columns if c not in keep_list]
    return df.drop(columns=dropped), dropped


def reorder_columns(df, columns: Sequence[str] | None = None):
    """Put the columns in exactly config.FLOW_COLUMNS order."""
    order = list(columns) if columns is not None else list(config.FLOW_COLUMNS)
    present = [c for c in order if c in df.columns]
    return df[present]


def to_microseconds(df, columns: Sequence[str] | None = None):
    """Seconds -> MICROSECONDS for every time feature, to match MSCAD's scale.

    Call this EXACTLY ONCE per capture: it multiplies in place, so calling it
    twice would scale by 10^12.
    """
    cols = list(columns) if columns is not None else list(config.MICROSECOND_COLUMNS)
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c] * 1_000_000
    return out


def detect_attackers(df, threshold: float | None = None):
    """Find the attacker IPs by traffic volume.

    An attacker is any Src IP whose TOTAL forward packets reach
    threshold x the top sender's total.  Returns (attackers, top_senders)
    where top_senders is a descending Series of Src IP -> total forward packets,
    so the top 5 can be printed and checked by eye.
    """
    thr = config.ATTACKER_THRESHOLD if threshold is None else threshold
    totals = df.groupby("Src IP")["Tot Fwd Pkts"].sum().sort_values(ascending=False)
    if totals.empty:
        return [], totals
    cutoff = float(totals.iloc[0]) * thr
    attackers = [ip for ip, v in totals.items() if float(v) >= cutoff]
    return attackers, totals


def label_flows(df, stem: str, attackers: Sequence[str] | None = None):
    """Add the Label column.

    N-0 is documented as pure normal traffic, so every flow there is "Normal".
    In an attack capture, a flow is given the capture's attack label when an
    attacker IP is on either end, and "Benign" otherwise.

    The label comes from the documented capture -> attack mapping plus the
    measured attacker IPs.  It is never guessed from packet contents.
    """
    out = df.copy()
    if stem in config.NORMAL_CAPTURES:
        out[config.LABEL_COLUMN] = config.CAPTURE_LABELS[stem]
        return out
    if stem not in config.CAPTURE_LABELS:
        raise ConverterError(
            "capture '%s' has no entry in config.CAPTURE_LABELS, so it cannot be "
            "labelled. Add it there -- labels are never inferred." % stem)
    attack_label = config.CAPTURE_LABELS[stem]
    attacker_set = set(attackers or [])
    involved = out["Src IP"].isin(attacker_set) | out["Dst IP"].isin(attacker_set)
    out[config.LABEL_COLUMN] = config.BENIGN_LABEL
    out.loc[involved, config.LABEL_COLUMN] = attack_label
    return out


def postprocess(raw_df, stem: str, rep: Report | None = None) -> tuple[Any, dict[str, Any]]:
    """Raw extractor output -> the final per-capture DataFrame.

    Order matters:
      1 strip whitespace from the column names
      2 apply RENAME_MAP (snake_case -> MSCAD names)
      3 ERROR if any required column is still missing
      4 seconds -> microseconds (on the renamed names)
      5 find attackers and add Label
      6 drop the extra columns, list them
      7 reorder to exactly config.FLOW_COLUMNS
    Values are NOT cleaned: inf, NaN and duplicates are kept on purpose.
    """
    rep = rep or Report(echo=False)
    df = strip_column_names(raw_df)
    df = apply_rename_map(df)
    check_required_columns(df)
    df = to_microseconds(df)

    if stem in config.NORMAL_CAPTURES:
        attackers: list[str] = []
        totals = None
    else:
        attackers, totals = detect_attackers(df)
    df = label_flows(df, stem, attackers)

    df, dropped = split_extra_columns(df)
    df = reorder_columns(df)

    info: dict[str, Any] = {
        "stem": stem,
        "flows": len(df),
        "dropped_columns": dropped,
        "attackers": attackers,
        "label_method": ("all Normal (documented normal capture)"
                         if stem in config.NORMAL_CAPTURES
                         else "attacker IPs >= %.2f x top sender" % config.ATTACKER_THRESHOLD),
        "label_counts": df[config.LABEL_COLUMN].value_counts().to_dict(),
        "top_senders": (totals.head(5).to_dict() if totals is not None else {}),
    }
    rep.say("  %s: %d flows, dropped %d extra column(s)" % (stem, len(df), len(dropped)))
    return df, info
