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


# ===========================================================================
# STAGE C -- the flow engine
# ===========================================================================
def _sample_std(values: Sequence[float]) -> float:
    """Sample standard deviation (the n-1 kind), 0.0 for fewer than 2 values.

    The Java CICFlowMeter uses Apache Commons' StandardDeviation, which is the
    sample version and which reports 0 for a single observation, so we match it
    rather than using numpy's population default.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def _sample_var(values: Sequence[float]) -> float:
    """Sample variance, 0.0 for fewer than 2 values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _mean(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _max(values: Sequence[float]) -> float:
    return max(values) if values else 0.0


def _min(values: Sequence[float]) -> float:
    return min(values) if values else 0.0


def _diffs(times: Sequence[float]) -> list[float]:
    """Inter-arrival gaps between consecutive timestamps."""
    return [times[i] - times[i - 1] for i in range(1, len(times))]


# A subflow boundary in CICFlowMeter is a gap of more than one second.
SUBFLOW_GAP_S = 1.0


class Flow:
    """One network flow, accumulating everything the 66 features need.

    A flow is created by its FIRST packet, and that packet's direction becomes
    "forward".  Packets arriving with the reversed 5-tuple are "backward".

    Fix #5 (first packet counted twice): the constructor calls add() exactly
    once for the first packet and the main loop never adds it again.  Stock
    cicflowmeter created the flow with the packet AND then added it, so every
    flow's first packet was counted twice.  The Stage D check
    "sum(Tot Fwd Pkts + Tot Bwd Pkts) == independent IPv4 count" is what proves
    this is fixed.
    """

    def __init__(self, rec: PacketRecord, key: tuple) -> None:
        self.key = key                      # the dict key this flow lives under
        self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol = key
        self.start_time = rec.time
        self.last_time = rec.time

        # per-direction packet lengths and arrival times
        self.fwd_lens: list[int] = []
        self.bwd_lens: list[int] = []
        self.fwd_times: list[float] = []
        self.bwd_times: list[float] = []
        self.all_times: list[float] = []

        self.fwd_header = 0
        self.bwd_header = 0
        self.fwd_payload_bytes = 0
        self.bwd_payload_bytes = 0
        self.fwd_act_data_pkts = 0

        # flag counters over the whole flow (CICFlowMeter counts packets)
        self.fin = self.syn = self.rst = self.psh = 0
        self.ack = self.urg = self.ece = self.cwr = 0
        # direction-specific PSH/URG, which MSCAD keeps only for backward
        self.fwd_psh = self.bwd_psh = 0
        self.fwd_urg = self.bwd_urg = 0
        # FIN seen per direction, so a TCP flow closes only after both sides
        self.fwd_fin = self.bwd_fin = 0

        self.init_fwd_win: int | None = None
        self.init_bwd_win: int | None = None
        self.fwd_seg_size_min: int | None = None

        self.add(rec, forward=True)         # the first packet, added ONCE

    # -- accumulate -------------------------------------------------------
    def add(self, rec: PacketRecord, forward: bool) -> None:
        self.last_time = rec.time
        self.all_times.append(rec.time)

        if forward:
            self.fwd_lens.append(rec.cap_len)
            self.fwd_times.append(rec.time)
            self.fwd_header += rec.header_len
            self.fwd_payload_bytes += rec.payload_len
            if rec.payload_len > 0:
                self.fwd_act_data_pkts += 1
            if rec.psh:
                self.fwd_psh += 1
            if rec.urg:
                self.fwd_urg += 1
            if rec.fin:
                self.fwd_fin += 1
            if self.init_fwd_win is None and rec.protocol == PROTO_TCP:
                self.init_fwd_win = rec.window
            if self.fwd_seg_size_min is None or rec.header_len < self.fwd_seg_size_min:
                self.fwd_seg_size_min = rec.header_len
        else:
            self.bwd_lens.append(rec.cap_len)
            self.bwd_times.append(rec.time)
            self.bwd_header += rec.header_len
            self.bwd_payload_bytes += rec.payload_len
            if rec.psh:
                self.bwd_psh += 1
            if rec.urg:
                self.bwd_urg += 1
            if rec.fin:
                self.bwd_fin += 1
            if self.init_bwd_win is None and rec.protocol == PROTO_TCP:
                self.init_bwd_win = rec.window

        # Fix #4: these come from the TCP flag byte (see PacketRecord), not from
        # the IP flags field that stock cicflowmeter read by mistake.
        self.fin += rec.fin
        self.syn += rec.syn
        self.rst += rec.rst
        self.psh += rec.psh
        self.ack += rec.ack
        self.urg += rec.urg
        self.ece += rec.ece
        self.cwr += rec.cwr

    # -- closing ----------------------------------------------------------
    @property
    def packet_count(self) -> int:
        return len(self.fwd_lens) + len(self.bwd_lens)

    @property
    def duration_s(self) -> float:
        return self.last_time - self.start_time

    def tcp_finished(self) -> bool:
        """True when a TCP flow has closed itself: FIN both ways, or a RST."""
        if self.protocol != PROTO_TCP:
            return False
        return (self.fwd_fin >= 1 and self.bwd_fin >= 1) or self.rst >= 1

    def active_idle_periods(self, activity_timeout_s: float) -> tuple[list[float], list[float]]:
        """Split the flow's timeline into active periods and idle gaps."""
        times = self.all_times
        if len(times) < 2:
            return [0.0], []
        active: list[float] = []
        idle: list[float] = []
        start = prev = times[0]
        for t in times[1:]:
            gap = t - prev
            if gap > activity_timeout_s:
                active.append(prev - start)
                idle.append(gap)
                start = t
            prev = t
        active.append(prev - start)
        return active, idle

    def subflow_count(self) -> int:
        """Number of subflows: a gap over one second starts a new one."""
        gaps = _diffs(self.all_times)
        return 1 + sum(1 for g in gaps if g > SUBFLOW_GAP_S)

    # -- the 72 raw columns ----------------------------------------------
    def to_row(self, activity_timeout_s: float,
               init_win_absent: int) -> dict[str, Any]:
        """Build one RAW row: snake_case names, all times still in SECONDS.

        The microsecond conversion happens later, in postprocess(), so that the
        raw CSV stays exactly what the extractor produced.
        """
        from datetime import datetime, timezone

        fwd_n, bwd_n = len(self.fwd_lens), len(self.bwd_lens)
        total_n = fwd_n + bwd_n
        totlen_fwd, totlen_bwd = sum(self.fwd_lens), sum(self.bwd_lens)
        all_lens = self.fwd_lens + self.bwd_lens
        duration = self.duration_s

        flow_iat = _diffs(self.all_times)
        fwd_iat = _diffs(self.fwd_times)
        bwd_iat = _diffs(self.bwd_times)
        active, idle = self.active_idle_periods(activity_timeout_s)
        sf = max(1, self.subflow_count())

        # Rates: a zero-duration flow divides by zero.  float("inf") is the
        # CORRECT answer here and the spec says to keep it, so we do -- the
        # Stage D table counts the inf values instead of hiding them.
        def per_second(value: float) -> float:
            if duration > 0:
                return value / duration
            return float("inf") if value > 0 else 0.0

        return {
            # --- identifiers --------------------------------------------
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": int(self.src_port),
            "dst_port": int(self.dst_port),
            "protocol": int(self.protocol),
            # flow START time, in UTC.  Stage D's duplicate check is
            # "same 5-tuple + same start Timestamp", so this must be the start.
            "timestamp": datetime.fromtimestamp(self.start_time, tz=timezone.utc)
                                 .strftime("%Y-%m-%d %H:%M:%S.%f"),
            # --- duration and counts ------------------------------------
            "flow_duration": duration,
            "tot_fwd_pkts": fwd_n,
            "tot_bwd_pkts": bwd_n,
            "totlen_fwd_pkts": totlen_fwd,
            "totlen_bwd_pkts": totlen_bwd,
            # --- per-direction packet lengths ---------------------------
            "fwd_pkt_len_max": _max(self.fwd_lens),
            "fwd_pkt_len_min": _min(self.fwd_lens),
            "fwd_pkt_len_mean": _mean(self.fwd_lens),
            "fwd_pkt_len_std": _sample_std(self.fwd_lens),
            "bwd_pkt_len_max": _max(self.bwd_lens),
            "bwd_pkt_len_min": _min(self.bwd_lens),
            "bwd_pkt_len_mean": _mean(self.bwd_lens),
            "bwd_pkt_len_std": _sample_std(self.bwd_lens),
            # --- rates ---------------------------------------------------
            "flow_byts_s": per_second(totlen_fwd + totlen_bwd),
            "flow_pkts_s": per_second(total_n),
            "fwd_pkts_s": per_second(fwd_n),
            "bwd_pkts_s": per_second(bwd_n),
            # --- inter-arrival times (seconds here) ---------------------
            "flow_iat_mean": _mean(flow_iat),
            "flow_iat_std": _sample_std(flow_iat),
            "flow_iat_max": _max(flow_iat),
            "flow_iat_min": _min(flow_iat),
            "fwd_iat_tot": sum(fwd_iat),
            "fwd_iat_mean": _mean(fwd_iat),
            "fwd_iat_std": _sample_std(fwd_iat),
            "fwd_iat_max": _max(fwd_iat),
            "fwd_iat_min": _min(fwd_iat),
            "bwd_iat_tot": sum(bwd_iat),
            "bwd_iat_mean": _mean(bwd_iat),
            "bwd_iat_std": _sample_std(bwd_iat),
            "bwd_iat_max": _max(bwd_iat),
            "bwd_iat_min": _min(bwd_iat),
            # --- flags and headers --------------------------------------
            "fwd_psh_flags": self.fwd_psh,      # dropped later: not in MSCAD
            "bwd_psh_flags": self.bwd_psh,
            "fwd_urg_flags": self.fwd_urg,      # dropped later: not in MSCAD
            "bwd_urg_flags": self.bwd_urg,
            "fwd_header_len": self.fwd_header,
            "bwd_header_len": self.bwd_header,
            # --- packet-length statistics over BOTH directions ----------
            "pkt_len_min": _min(all_lens),
            "pkt_len_max": _max(all_lens),
            "pkt_len_mean": _mean(all_lens),
            "pkt_len_std": _sample_std(all_lens),
            "pkt_len_var": _sample_var(all_lens),
            # --- TCP flag counters --------------------------------------
            "fin_flag_cnt": self.fin,
            "syn_flag_cnt": self.syn,
            "rst_flag_cnt": self.rst,
            "psh_flag_cnt": self.psh,
            "ack_flag_cnt": self.ack,
            "urg_flag_cnt": self.urg,
            "cwr_flag_count": self.cwr,        # becomes "CWE Flag Count"
            "ece_flag_cnt": self.ece,
            # --- ratios and averages ------------------------------------
            "down_up_ratio": (bwd_n / fwd_n) if fwd_n else 0.0,
            "pkt_size_avg": ((totlen_fwd + totlen_bwd) / total_n) if total_n else 0.0,
            "fwd_seg_size_avg": (totlen_fwd / fwd_n) if fwd_n else 0.0,
            "bwd_seg_size_avg": (totlen_bwd / bwd_n) if bwd_n else 0.0,
            # --- subflows (averages per subflow) ------------------------
            "subflow_fwd_pkts": fwd_n / sf,
            "subflow_fwd_byts": totlen_fwd / sf,
            "subflow_bwd_pkts": bwd_n / sf,
            "subflow_bwd_byts": totlen_bwd / sf,
            # --- window sizes and data packets --------------------------
            "init_fwd_win_byts": init_win_absent if self.init_fwd_win is None
                                 else self.init_fwd_win,     # dropped later
            "init_bwd_win_byts": init_win_absent if self.init_bwd_win is None
                                 else self.init_bwd_win,
            "fwd_act_data_pkts": self.fwd_act_data_pkts,
            "fwd_seg_size_min": self.fwd_seg_size_min or 0,  # dropped later
            # --- active / idle (seconds here) ---------------------------
            "active_mean": _mean(active),
            "active_std": _sample_std(active),
            "active_max": _max(active),
            "active_min": _min(active),
            "idle_mean": _mean(idle),
            "idle_std": _sample_std(idle),
            "idle_max": _max(idle),
            "idle_min": _min(idle),
            # --- bulk-rate columns: CICFlowMeter emits 0 and MSCAD drops them
            "fwd_byts_b_avg": 0,
            "fwd_pkts_b_avg": 0,
            "fwd_blk_rate_avg": 0,
            "bwd_byts_b_avg": 0,
            "bwd_pkts_b_avg": 0,
            "bwd_blk_rate_avg": 0,
        }


def iter_packet_records(pcap_path: Path | str):
    """Stream a capture and yield one PacketRecord per IPv4 packet.

    Fix #6: scapy's PcapReader reads one packet at a time, and we immediately
    copy out the few fields we need, so no scapy packet is ever kept alive.
    rdpcap() would load the whole file into RAM, which is what crashed on the
    flood captures.

    Fix #3: no BPF capture filter is used anywhere.  Stock cicflowmeter's filter
    needs tcpdump, which does not exist on Windows.  Filtering to IPv4 happens
    in Python, in packet_record_from_scapy().

    Yields (record, total_packets_seen) so the caller can count non-IPv4
    packets (ARP / IPv6), which are excluded from flows by design.

    IMPORTANT -- why the layer modules are imported HERE, before PcapReader:
    PcapReader picks the link-layer class when it opens the file, from the
    link-type in the file header (1 = Ethernet).  scapy only knows that 1 means
    Ether once scapy.layers.l2 has been imported.  Importing the layers lazily
    further down meant the FIRST capture of a session was decoded as Raw -- so
    "IP in pkt" was False for every packet and the capture produced 0 flows,
    while every later capture worked.  Importing both modules up front fixes it.
    """
    import scapy.layers.inet  # noqa: F401  (registers IP/TCP/UDP/ICMP)
    import scapy.layers.l2  # noqa: F401  (registers Ether for link-type 1)
    from scapy.utils import PcapReader

    with PcapReader(str(pcap_path)) as reader:
        for seen, pkt in enumerate(reader, start=1):
            rec = packet_record_from_scapy(pkt)
            yield rec, seen


def extract_flows(pcap_path: Path | str,
                  flow_timeout_s: float | None = None,
                  activity_timeout_s: float | None = None,
                  progress_every: int | None = None,
                  sweep_every: int | None = None,
                  rep: Report | None = None,
                  show_progress: bool = True):
    """Read a .pcap and return (raw DataFrame, stats dict).

    The DataFrame has the extractor's snake_case column names and all times
    still in SECONDS; postprocess() renames, rescales and labels it.

    Fix #1: this is a library call.  The cicflowmeter 0.5.0 CLI ("-f file")
    crashes with "'bool' object has no attribute 'split'", so the CLI is never
    used.
    Fix #7: a flow's timeout is checked only when a packet for that flow
    arrives, plus a periodic sweep of all open flows -- not a full rescan after
    every FIN, which was quadratic.
    """
    import pandas as pd

    flow_timeout_s = config.FLOW_TIMEOUT_S if flow_timeout_s is None else flow_timeout_s
    activity_timeout_s = config.ACTIVITY_TIMEOUT_S if activity_timeout_s is None \
        else activity_timeout_s
    progress_every = config.PROGRESS_EVERY if progress_every is None else progress_every
    sweep_every = config.SWEEP_EVERY if sweep_every is None else sweep_every
    rep = rep or Report(echo=show_progress)

    open_flows: dict[tuple, Flow] = {}
    rows: list[dict[str, Any]] = []
    packets_total = 0      # every packet in the file
    packets_ipv4 = 0       # the IPv4 ones, which are the only ones in flows
    now = 0.0

    def close(flow: Flow) -> None:
        open_flows.pop(flow.key, None)
        rows.append(flow.to_row(activity_timeout_s, config.INIT_WIN_ABSENT_VALUE))

    for rec, seen in iter_packet_records(pcap_path):
        packets_total = seen
        if rec is None:
            continue                       # ARP / IPv6 / non-IPv4: excluded
        packets_ipv4 += 1
        now = rec.time

        key = flow_key(rec)
        rkey = reverse_flow_key(rec)
        flow = open_flows.get(key)
        forward = True
        if flow is None:
            flow = open_flows.get(rkey)
            forward = flow is None          # stays True only if still not found

        if flow is not None:
            # per-flow timeout check (fix #7)
            if (rec.time - flow.start_time) > flow_timeout_s:
                close(flow)
                flow = None

        if flow is None:
            flow = Flow(rec, key)           # first packet added exactly once
            open_flows[key] = flow
        else:
            flow.add(rec, forward)

        if flow.tcp_finished():
            close(flow)

        if sweep_every and packets_ipv4 % sweep_every == 0:
            for f in [f for f in open_flows.values()
                      if (now - f.last_time) > flow_timeout_s]:
                close(f)

        if progress_every and packets_ipv4 % progress_every == 0:
            rep.say("      %10s packets read | %7d flows done | %7d flows open"
                    % (f"{packets_ipv4:,}", len(rows), len(open_flows)))

    for f in list(open_flows.values()):     # flush whatever is still open
        close(f)

    df = pd.DataFrame(rows)
    stats = {
        "pcap": str(pcap_path),
        "packets_total": packets_total,
        "packets_ipv4": packets_ipv4,
        "packets_non_ipv4": packets_total - packets_ipv4,
        "flows": len(rows),
        "flow_timeout_s": flow_timeout_s,
        "activity_timeout_s": activity_timeout_s,
    }
    return df, stats


# ===========================================================================
# STAGE C/D -- run one capture, run all six, write the logs
# ===========================================================================
def process_capture(pcap_path: Path | str, stem: str,
                    rep: Report | None = None,
                    save_raw: bool = True, save_csv: bool = True,
                    show_progress: bool = True):
    """pcap -> raw CSV -> final <stem>_flows.csv.  Returns (DataFrame, info).

    Writes CSVs\\raw\\<stem>_raw.csv (exactly what the extractor produced) and
    CSVs\\<stem>_flows.csv (renamed, rescaled, reordered, labelled).
    Values are never cleaned: inf, NaN and duplicate rows are kept on purpose.
    """
    import time as _time

    rep = rep or Report(echo=show_progress)
    t0 = _time.perf_counter()

    raw, stats = extract_flows(pcap_path, rep=rep, show_progress=show_progress)
    if raw.empty:
        raise ConverterError(
            "%s produced 0 flows from %d packets -- refusing to write an empty CSV"
            % (stem, stats["packets_total"]))

    if save_raw:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = config.RAW_DIR / ("%s_raw.csv" % stem)
        raw.to_csv(raw_path, index=False)
        stats["raw_csv"] = str(raw_path)

    df, info = postprocess(raw, stem, rep=Report(echo=False))

    if save_csv:
        config.CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.CSV_OUTPUT_DIR / ("%s_flows.csv" % stem)
        df.to_csv(out_path, index=False)
        info["output_csv"] = str(out_path)

    info.update(stats)
    info["runtime_s"] = round(_time.perf_counter() - t0, 2)
    return df, info


def print_capture_summary(rep: Report, df, info: dict[str, Any]) -> None:
    """The per-capture block: packets, flows, attackers, top senders, labels."""
    rep.say("  packets in file      : %s" % f"{info['packets_total']:,}")
    rep.say("  IPv4 packets (used)  : %s" % f"{info['packets_ipv4']:,}")
    rep.say("  non-IPv4 (ARP/IPv6)  : %s   <- excluded by design"
            % f"{info['packets_non_ipv4']:,}")
    rep.say("  flows                : %s" % f"{info['flows']:,}")
    rep.say("  shape                : %s" % (df.shape,))
    rep.say("  runtime              : %.2f s" % info["runtime_s"])
    rep.say("  dropped columns (%d)  : %s" % (len(info["dropped_columns"]),
                                              info["dropped_columns"]))
    rep.say("  label method         : %s" % info["label_method"])
    rep.say("  attacker IPs         : %s" % (info["attackers"] or "(none - normal capture)"))
    if info["top_senders"]:
        rep.say("  top 5 senders by forward packets:")
        for ip, pkts in list(info["top_senders"].items())[:5]:
            rep.say("        %-18s %s" % (ip, f"{int(pkts):,}"))
    rep.say("  label counts         : %s" % info["label_counts"])


LOG_FIELDS = ["stem", "status", "output_csv", "raw_csv", "flows",
              "packets_total", "packets_ipv4", "packets_non_ipv4",
              "dropped_columns", "label_method", "attackers", "label_counts",
              "runtime_s", "error"]

LABEL_FIELDS = ["stem", "label", "flows", "share_pct", "attackers",
                "attacker_count", "label_method", "top_senders"]


def run_all(captures: Sequence[CaptureFile] | None = None,
            overwrite: bool | None = None,
            rep: Report | None = None,
            use_tqdm: bool = True,
            only: Sequence[str] | None = None):
    """STAGE D: process every capture, write conversion_log.csv + label_report.csv.

    Never skips a capture silently: a failure is recorded with its exact error
    and reported at the end.
    """
    rep = rep or Report()
    overwrite = config.OVERWRITE if overwrite is None else overwrite

    if captures is None:
        capture_dir, _why = locate_capture_dir(rep)
        captures = find_captures(capture_dir) if capture_dir else []
    todo = [c for c in captures if c.stem in config.CAPTURE_LABELS]
    if only:
        todo = [c for c in todo if c.stem in set(only)]
    skipped_unlabelled = [c.stem for c in captures if c.stem not in config.CAPTURE_LABELS]
    if skipped_unlabelled:
        rep.warn("not processed (no label in config.CAPTURE_LABELS): %s" % skipped_unlabelled)

    iterator: Iterable[CaptureFile] = todo
    if use_tqdm:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(todo, desc="extracting flows", unit="capture")
        except ImportError:
            pass

    log_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    frames: dict[str, Any] = {}

    for cap in iterator:
        pcap = config.PCAP_DIR / output_pcap_name(cap, captures)
        out_csv = config.CSV_OUTPUT_DIR / ("%s_flows.csv" % cap.stem)
        row: dict[str, Any] = {k: "" for k in LOG_FIELDS}
        row["stem"] = cap.stem
        row["output_csv"] = str(out_csv)

        rep.sub("%s  (%s)" % (cap.stem, config.CAPTURE_LABELS[cap.stem]))

        if not pcap.exists():
            row["status"] = "failed"
            row["error"] = "%s does not exist -- run Stage B first" % pcap
            rep.bad(row["error"])
            log_rows.append(row)
            continue
        if out_csv.exists() and not overwrite:
            row["status"] = "skipped"
            row["error"] = "output exists and config.OVERWRITE is False"
            rep.warn("skipped: %s already exists (OVERWRITE is False)" % out_csv.name)
            log_rows.append(row)
            continue

        try:
            df, info = process_capture(pcap, cap.stem, rep=rep)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = "%s: %s" % (type(exc).__name__, exc)
            rep.bad("FAILED: %s" % row["error"])
            log_rows.append(row)
            continue

        frames[cap.stem] = df
        print_capture_summary(rep, df, info)
        row.update({
            "status": "success",
            "raw_csv": info.get("raw_csv", ""),
            "flows": info["flows"],
            "packets_total": info["packets_total"],
            "packets_ipv4": info["packets_ipv4"],
            "packets_non_ipv4": info["packets_non_ipv4"],
            "dropped_columns": "|".join(info["dropped_columns"]),
            "label_method": info["label_method"],
            "attackers": "|".join(info["attackers"]),
            "label_counts": json.dumps(info["label_counts"]),
            "runtime_s": info["runtime_s"],
        })
        log_rows.append(row)

        total = max(1, info["flows"])
        for label, n in sorted(info["label_counts"].items(), key=lambda kv: -kv[1]):
            label_rows.append({
                "stem": cap.stem,
                "label": label,
                "flows": n,
                "share_pct": round(100.0 * n / total, 2),
                "attackers": "|".join(info["attackers"]),
                "attacker_count": len(info["attackers"]),
                "label_method": info["label_method"],
                "top_senders": json.dumps({k: int(v) for k, v in info["top_senders"].items()}),
            })

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_dicts(config.CONVERSION_LOG_CSV, LOG_FIELDS, log_rows)
    _write_dicts(config.LABEL_REPORT_CSV, LABEL_FIELDS, label_rows)

    rep.sub("Stage D summary")
    n_ok = sum(1 for r in log_rows if r["status"] == "success")
    n_skip = sum(1 for r in log_rows if r["status"] == "skipped")
    n_fail = sum(1 for r in log_rows if r["status"] == "failed")
    total_flows = sum(int(r["flows"] or 0) for r in log_rows if r["status"] == "success")
    rep.say("  captures attempted : %d" % len(log_rows))
    rep.say("  successful         : %d" % n_ok)
    rep.say("  skipped            : %d" % n_skip)
    rep.say("  failed             : %d" % n_fail)
    rep.say("  total flows written: %s" % f"{total_flows:,}")
    rep.say("  output folder      : %s" % config.CSV_OUTPUT_DIR)
    rep.say("  conversion log     : %s" % config.CONVERSION_LOG_CSV)
    rep.say("  label report       : %s" % config.LABEL_REPORT_CSV)
    if n_fail:
        rep.bad("%d capture(s) FAILED -- none were skipped silently:" % n_fail)
        for r in log_rows:
            if r["status"] == "failed":
                rep.say("          %s: %s" % (r["stem"], r["error"]))
    return log_rows, label_rows, frames


def _write_dicts(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path


# ===========================================================================
# STAGE D -- verification
# ===========================================================================
# Pairs that must satisfy min <= max.
MIN_MAX_PAIRS = [
    ("Fwd Pkt Len Min", "Fwd Pkt Len Max"),
    ("Bwd Pkt Len Min", "Bwd Pkt Len Max"),
    ("Pkt Len Min", "Pkt Len Max"),
    ("Flow IAT Min", "Flow IAT Max"),
    ("Fwd IAT Min", "Fwd IAT Max"),
    ("Bwd IAT Min", "Bwd IAT Max"),
    ("Active Min", "Active Max"),
    ("Idle Min", "Idle Max"),
]

VERIFICATION_FIELDS = [
    "stem", "label", "packets_in_pcap", "ipv4_packets_in_flows",
    "tshark_ipv4_packets", "packet_counts_match", "non_ipv4_gap",
    "flows", "duplicate_flow_keys", "nan_cells", "negative_cells",
    "min_gt_max_rows", "invalid_ports", "inf_flow_byts_s", "inf_flow_pkts_s",
    "identical_feature_rows", "header_matches_flow_columns", "label_counts",
]


def verify_capture(df, stem: str, pcap_path: Path | None = None,
                   packets_in_pcap: int | None = None,
                   tshark_ipv4: int | None = None) -> dict[str, Any]:
    """Every Stage D check for one capture.  Returns a plain dict of counts."""
    import numpy as np

    feature_cols = [c for c in config.KAGGLE_FEATURES if c in df.columns]
    numeric = df[feature_cols].apply(lambda s: s.astype("float64"), axis=0)

    # packets: Tot Fwd Pkts + Tot Bwd Pkts summed over flows must equal the
    # independent IPv4 count.  This is the check that would catch a
    # double-counted first packet (fix #5) or a dropped protocol (fix #2).
    ipv4_in_flows = int(df["Tot Fwd Pkts"].sum() + df["Tot Bwd Pkts"].sum())
    if tshark_ipv4 is None and pcap_path is not None:
        try:
            tshark_ipv4 = count_ipv4_packets(pcap_path)
        except Exception:
            tshark_ipv4 = None

    # a flow must not be written twice: same 5-tuple AND same start Timestamp
    key_cols = ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Timestamp"]
    duplicate_keys = int(df.duplicated(subset=key_cols).sum())

    # inf is EXPECTED for zero-duration flows: count it, never fix it
    inf_byts = int(np.isinf(numeric["Flow Byts/s"]).sum()) if "Flow Byts/s" in numeric else 0
    inf_pkts = int(np.isinf(numeric["Flow Pkts/s"]).sum()) if "Flow Pkts/s" in numeric else 0

    nan_cells = int(numeric.isna().to_numpy().sum())
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    negative_cells = int((finite < 0).to_numpy().sum())

    min_gt_max = 0
    for lo, hi in MIN_MAX_PAIRS:
        if lo in df.columns and hi in df.columns:
            min_gt_max += int((numeric[lo] > numeric[hi]).sum())

    bad_ports = 0
    for col in ("Src Port", "Dst Port"):
        p = df[col].astype("int64")
        bad_ports += int(((p < 0) | (p > 65535)).sum())

    identical = int(df.duplicated(subset=feature_cols).sum())

    return {
        "stem": stem,
        "label": config.CAPTURE_LABELS.get(stem, ""),
        "packets_in_pcap": packets_in_pcap if packets_in_pcap is not None else "",
        "ipv4_packets_in_flows": ipv4_in_flows,
        "tshark_ipv4_packets": tshark_ipv4 if tshark_ipv4 is not None else "",
        "packet_counts_match": (ipv4_in_flows == tshark_ipv4) if tshark_ipv4 is not None else "",
        "non_ipv4_gap": (packets_in_pcap - ipv4_in_flows)
                        if packets_in_pcap is not None else "",
        "flows": len(df),
        "duplicate_flow_keys": duplicate_keys,
        "nan_cells": nan_cells,
        "negative_cells": negative_cells,
        "min_gt_max_rows": min_gt_max,
        "invalid_ports": bad_ports,
        "inf_flow_byts_s": inf_byts,
        "inf_flow_pkts_s": inf_pkts,
        "identical_feature_rows": identical,
        "header_matches_flow_columns": list(df.columns) == config.FLOW_COLUMNS,
        "label_counts": json.dumps(df[config.LABEL_COLUMN].value_counts().to_dict()),
    }


def verify_all(log_rows: Sequence[dict] | None = None,
               captures: Sequence[CaptureFile] | None = None,
               rep: Report | None = None, use_tshark: bool = True):
    """Build the Stage D verification table from the CSVs on disk."""
    import pandas as pd

    rep = rep or Report()
    if captures is None:
        capture_dir, _why = locate_capture_dir(rep)
        captures = find_captures(capture_dir) if capture_dir else []
    by_stem = {r["stem"]: r for r in (log_rows or [])}

    rows: list[dict[str, Any]] = []
    for cap in captures:
        out_csv = config.CSV_OUTPUT_DIR / ("%s_flows.csv" % cap.stem)
        if not out_csv.exists():
            continue
        df = pd.read_csv(out_csv, low_memory=False)
        pcap = config.PCAP_DIR / output_pcap_name(cap, captures)
        packets = by_stem.get(cap.stem, {}).get("packets_total") or None
        if packets in ("", None) and pcap.exists():
            try:
                packets = count_packets(pcap)
            except Exception:
                packets = None
        rows.append(verify_capture(df, cap.stem,
                                   pcap_path=pcap if (use_tshark and pcap.exists()) else None,
                                   packets_in_pcap=int(packets) if packets else None))

    table = pd.DataFrame(rows, columns=VERIFICATION_FIELDS) if rows else pd.DataFrame()
    if rows:
        _write_dicts(config.VERIFICATION_CSV, VERIFICATION_FIELDS, rows)

    rep.sub("verification")
    if not rows:
        rep.bad("no *_flows.csv found in %s -- run Stage D first" % config.CSV_OUTPUT_DIR)
        return table

    must_be_zero = ["duplicate_flow_keys", "nan_cells", "negative_cells",
                    "min_gt_max_rows", "invalid_ports"]
    for r in rows:
        problems = [k for k in must_be_zero if int(r[k] or 0) != 0]
        if r["packet_counts_match"] is False:
            problems.append("packet_counts_match")
        if r["header_matches_flow_columns"] is not True:
            problems.append("header_matches_flow_columns")
        if problems:
            rep.bad("%-8s FAILED: %s" % (r["stem"], {k: r[k] for k in problems}))
        else:
            rep.ok("%-8s clean (%s flows, %s inf rates, %s identical-feature rows)"
                   % (r["stem"], f"{r['flows']:,}",
                      r["inf_flow_byts_s"], r["identical_feature_rows"]))
    rep.say()
    rep.say("  'inf' in Flow Byts/s and Flow Pkts/s for zero-duration flows is NORMAL:")
    rep.say("  it is counted above, never replaced.  Identical 66-feature rows are")
    rep.say("  expected in floods, scans and brute force: counted, never dropped.")
    rep.say("  verification table : %s" % config.VERIFICATION_CSV)
    return table


def compare_with_kaggle(rep: Report | None = None):
    """Reference-only: our per-capture label counts next to the Kaggle file's.

    Returns None when CSV_ORIGINAL holds no Kaggle file, which is not an error:
    Step 1 never needs it, and nothing in the output depends on it.
    """
    import pandas as pd

    rep = rep or Report()
    ours: dict[str, int] = {}
    for p in sorted(config.CSV_OUTPUT_DIR.glob("*_flows.csv")):
        s = pd.read_csv(p, usecols=[config.LABEL_COLUMN], low_memory=False)
        for label, n in s[config.LABEL_COLUMN].value_counts().items():
            ours[str(label)] = ours.get(str(label), 0) + int(n)
    if not ours:
        rep.bad("no *_flows.csv to compare")
        return None

    path = config.find_kaggle_file()
    if path is None:
        rep.warn("no Kaggle file in %s, so there is nothing to compare against."
                 % config.KAGGLE_DIR)
        rep.say("          Our own label totals:")
        for label, n in sorted(ours.items(), key=lambda kv: -kv[1]):
            rep.say("            %-18s %s" % (label, f"{n:,}"))
        return None

    kag = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") \
        else pd.read_csv(path, low_memory=False, usecols=lambda c: str(c).strip() == "Label")
    col = [c for c in kag.columns if str(c).strip() == config.LABEL_COLUMN][0]
    theirs = {str(k): int(v) for k, v in kag[col].value_counts().items()}

    rep.say("  %-18s %12s %12s   %s" % ("LABEL", "OURS", "KAGGLE", "NOTE"))
    rep.say("  " + "-" * 62)
    for label in sorted(set(ours) | set(theirs)):
        note = ""
        if label not in theirs:
            note = "only ours (we create Benign)"
        elif label not in ours:
            note = "only Kaggle"
        rep.say("  %-18s %12s %12s   %s"
                % (label, f"{ours.get(label, 0):,}", f"{theirs.get(label, 0):,}", note))
    rep.say()
    rep.say("  Reference comparison ONLY: the Kaggle file was produced by the Java")
    rep.say("  CICFlowMeter with its own settings, so the counts are not expected to")
    rep.say("  be equal and nothing here is adjusted to make them match.")
    return {"ours": ours, "kaggle": theirs}
