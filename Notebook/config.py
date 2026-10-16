"""
config.py -- the single source of truth for every path and setting in Step 1.

NOTHING else in this project may contain a hard-coded path.  Import from here:

    import config
    print(config.PCAPNG_DIR)

Beginner notes
--------------
1. Windows paths use backslashes (\\).  In Python a backslash starts an escape
   sequence (\\n = newline), so Windows paths are always written as RAW strings:
       Path(r"C:\\MSCAD2022")        <- the r means "take this literally"
2. Paths may contain SPACES, e.g. "C:\\Program Files\\Wireshark".  That is safe
   as long as we never build a command line as one big string.  Everywhere in
   this project external programs are called as a LIST of arguments:
       subprocess.run([EDITCAP_EXE, "-F", "pcap", str(src), str(dst)])
   and NEVER with shell=True.  With a list, Windows receives each argument
   separately, so a space can never split one argument into two.
3. pathlib.Path handles the slashes for you:  PCAP_DIR / "N-0.pcap".
   Call str(p) only when handing a path to subprocess.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path, PureWindowsPath

# ===========================================================================
# 1. FOLDERS
# ===========================================================================
# DATA_ROOT holds the data; PROJECT_ROOT holds all code and notebooks.
# The two environment variables are ONLY a unit-test hook, so the test suite
# can run on a machine without a C:\ drive.  On your PC they are not set and
# the literal paths below are used.
DATA_ROOT = Path(os.environ.get("MSCAD_DATA_ROOT", r"C:\MSCAD2022"))
# PROJECT_ROOT is the folder this file sits in, so the project works wherever you
# keep the code (C:\MSCAD2022\Notebook, or a git clone) with nothing to edit.
PROJECT_ROOT = Path(os.environ.get("MSCAD_PROJECT_ROOT",
                                   str(Path(__file__).resolve().parent)))
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# INPUT -- read-only.  Never modified, moved, renamed or deleted by this project.
PCAPNG_DIR = DATA_ROOT / "PCAPNG_ORIGINAL"

# OUTPUT of Stage B: the .pcap files produced by editcap.
PCAP_DIR = DATA_ROOT / "PCAPS"

# OUTPUT of Stage C/D: one flow CSV per capture, <stem>_flows.csv.
CSV_OUTPUT_DIR = DATA_ROOT / "CSVs"
RAW_DIR = CSV_OUTPUT_DIR / "raw"          # untouched extractor output

# REFERENCE ONLY -- the Kaggle MSCAD file.  Read, never written.
KAGGLE_DIR = DATA_ROOT / "CSV_ORIGINAL"

# Where the Kaggle MSCAD file actually lives.  Step 2's reference comparison
# uses THIS folder; KAGGLE_DIR above is kept because Step 1 still points at it.
# Read-only either way: nothing in this project ever writes to them.
KAGGLE_CSV_DIR = Path(os.environ.get("MSCAD_KAGGLE_CSV_DIR",
                                     r"C:\MSCAD2022\CSV_Original_kaggle"))

# Code-side folders.
MERGED_DIR = DATA_ROOT / "outputs" / "step3"       # see the Step 2/3 block below
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step1"    # logs and reports
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# Report files written by Step 1 (all inside OUTPUT_DIR).
PCAP_INVENTORY_CSV = OUTPUT_DIR / "pcap_inventory.csv"   # Stage B
CONVERSION_LOG_CSV = OUTPUT_DIR / "conversion_log.csv"   # Stage D
LABEL_REPORT_CSV = OUTPUT_DIR / "label_report.csv"       # Stage C/D
VERIFICATION_CSV = OUTPUT_DIR / "verification.csv"       # Stage D

# ---------------------------------------------------------------------------
# STEP 2 -- load and merge the per-capture flow CSVs
# ---------------------------------------------------------------------------
# Only the Step 1 outputs match this pattern, so CSVs\raw\*_raw.csv and any
# other file in the folder are ignored.
FLOW_CSV_PATTERN = "*_flows.csv"

# Where Step 2 looks for those CSVs.  CSV_OUTPUT_DIR (what Step 1 wrote) is
# tried first; the others are fallbacks in case the folder was renamed or the
# CSVs were collected somewhere else.  find_flow_csvs() reports which one it
# used, so the choice is never silent.
MSCAD_DATASET_DIR = DATA_ROOT / "MSCAD_Dataset"
CSV_CANDIDATE_DIRS = [
    CSV_OUTPUT_DIR,                     # C:\MSCAD2022\CSVs  (Step 1's output)
    MSCAD_DATASET_DIR,                  # C:\MSCAD2022\MSCAD_Dataset
    PROJECT_ROOT / "MSCAD_Dataset",
    DATA_ROOT / "MSCAD_CSVs",
]

# ---------------------------------------------------------------------------
# STEP 2 (acquisition) and STEP 3 (merging) -- ABSOLUTE paths
# ---------------------------------------------------------------------------
# These deliberately hang off DATA_ROOT, not PROJECT_ROOT.  PROJECT_ROOT is
# wherever config.py happens to sit, so a second copy of the code (a git clone
# beside C:\MSCAD2022\Notebook, say) silently moved the outputs with it and the
# merged file appeared to vanish.  Anchoring them to the data drive means the
# destination is the same no matter where the notebook is started from, and
# nothing here is ever resolved against the current working directory.
CSV_INPUT_DIR = CSV_OUTPUT_DIR          # C:\MSCAD2022\CSVs -- Step 2 reads here
OUTPUTS_DIR = DATA_ROOT / "outputs"     # C:\MSCAD2022\outputs
STEP2_OUTPUT_DIR = OUTPUTS_DIR / "step2"   # acquisition report
STEP3_OUTPUT_DIR = OUTPUTS_DIR / "step3"   # merged dataset + its reports
STEP4_OUTPUT_DIR = OUTPUTS_DIR / "step4"   # preprocessing outputs

MERGED_CSV = STEP3_OUTPUT_DIR / "MSCAD_merged.csv"
MERGED_PARQUET = STEP3_OUTPUT_DIR / "MSCAD_merged.parquet"

# The order the captures are concatenated in: the chronological order of the
# attack scenario, so the merged file reads like the session it came from.
CAPTURE_ORDER = ["N-0", "Scan-1", "App-01", "App-02", "W-B-01", "W-B-02"]

# False = keep an existing merged file and skip the save.
OVERWRITE_MERGED = False

# Every folder this project is allowed to create.  PCAPNG_DIR and KAGGLE_DIR
# are deliberately NOT in this list.
DIRS_TO_CREATE = [
    PCAP_DIR,
    CSV_OUTPUT_DIR,
    RAW_DIR,
    PROJECT_ROOT,
    SRC_DIR,
    SRC_DIR / "mscad",
    TESTS_DIR,
    MERGED_DIR,
    OUTPUT_DIR,
    OUTPUTS_DIR,
    STEP2_OUTPUT_DIR,
    STEP3_OUTPUT_DIR,
    STEP4_OUTPUT_DIR,
    NOTEBOOKS_DIR,
]
# STEP5_OUTPUT_DIR is appended to this list at the bottom of the file, where it
# is defined -- DIRS_TO_CREATE is declared here, before the Step 5 block.

# Folders that must never be written to.
READ_ONLY_DIRS = [PCAPNG_DIR, KAGGLE_DIR, KAGGLE_CSV_DIR]

# ===========================================================================
# 2. EXTERNAL PROGRAMS
# ===========================================================================
# The Python interpreter that has cicflowmeter / scapy installed.
PYTHON_EXE = os.environ.get(
    "PYTHON_EXE",
    r"C:\Users\banot\AppData\Local\Programs\Python\Python313\python.exe",
)


def _resolve_tool(default_path: str, exe_name: str) -> str:
    """Return the default Wireshark path if it exists, else look on the PATH.

    Wireshark ships editcap (format conversion) and tshark (packet counting).
    If Wireshark was installed somewhere else but its folder is on the PATH,
    shutil.which() finds it, so the code keeps working.  If neither is found we
    still return the default path, and the Stage A check reports it as missing
    with instructions, instead of crashing here at import time.
    """
    env_override = os.environ.get(exe_name.upper() + "_EXE")
    if env_override:
        return env_override
    if Path(default_path).is_file():
        return default_path
    found = shutil.which(exe_name)
    return found if found else default_path


EDITCAP_EXE = _resolve_tool(r"C:\Program Files\Wireshark\editcap.exe", "editcap")
TSHARK_EXE = _resolve_tool(r"C:\Program Files\Wireshark\tshark.exe", "tshark")
# capinfos also ships with Wireshark.  It reads only the capture header/index,
# so counting packets with it is much faster than walking every packet with
# tshark.  converter.py falls back to tshark if capinfos is absent.
CAPINFOS_EXE = _resolve_tool(r"C:\Program Files\Wireshark\capinfos.exe", "capinfos")

# ===========================================================================
# 3. SETTINGS
# ===========================================================================
RANDOM_SEED = 42          # fixed so every run is reproducible

FLOW_TIMEOUT_S = 120      # close a flow after 120 s, as the Java CICFlowMeter
IDLE_TIMEOUT_S = 240      # silence that ends an "active" period

OVERWRITE = False         # False = skip a capture whose output already exists
PROGRESS_EVERY = 100_000  # print a progress line every N packets

# How often the extractor sweeps ALL open flows for expiry.  Fix #7: stock
# cicflowmeter re-scanned every open flow after every FIN, which is quadratic
# and crawls on the port-scan and brute-force captures.  We check only the flow
# a packet belongs to, plus this periodic sweep as a safety net.
SWEEP_EVERY = 50_000

# The gap (in seconds) separating an ACTIVE period from an IDLE period.  Used
# only for the Active*/Idle* features, and taken from IDLE_TIMEOUT_S.
#
# HEADS UP: IDLE_TIMEOUT_S (240 s) is LARGER than FLOW_TIMEOUT_S (120 s), and no
# flow can live longer than 120 s, so a 240 s silent gap can never happen inside
# one flow.  The consequence is arithmetic, not a bug: every Idle* feature comes
# out 0 and Active Max equals Flow Duration.  The Java CICFlowMeter pairs
# flowTimeout = 120 s with a much smaller activityTimeout (5 s by default), and
# that is what makes its Idle* columns non-zero.  Set ACTIVITY_TIMEOUT_S = 5 if
# you want populated Idle* features; every run reports which value it used.
ACTIVITY_TIMEOUT_S = IDLE_TIMEOUT_S

# What "packet length" means for the length features.
#   "frame" -> captured frame length (includes the 14-byte Ethernet header)
#   "ip"    -> the IP total-length field (excludes the Ethernet header)
# "frame" is the captured-bytes definition the spec asks for.  If a later
# comparison against the Kaggle file shows a constant 14-bytes-per-packet
# offset, switch this to "ip".
PACKET_LENGTH_MODE = "frame"

# Value written for Init Fwd/Bwd Win Byts when a direction has no TCP packet.
# CICFlowMeter writes -1; we write 0 because the Stage D verification requires
# zero negative values.  Set it to -1 to match CICFlowMeter byte-for-byte.
INIT_WIN_ABSENT_VALUE = 0


# A Src IP is an attacker if its total forward packets are at least
# ATTACKER_THRESHOLD x the top sender's total forward packets.
ATTACKER_THRESHOLD = 0.5

# Extensions we treat as captures (compared case-insensitively).
CAPTURE_EXTENSIONS = {".pcapng", ".pcap"}

# ===========================================================================
# 4. COLUMN NAMES
# ===========================================================================
# The 6 flow identifiers the Kaggle MSCAD.csv does NOT have -- the whole reason
# we regenerate the flows from the raw captures.
ID_COLUMNS = [
    "Src IP",
    "Dst IP",
    "Src Port",
    "Dst Port",
    "Protocol",
    "Timestamp",
]

# The 66 CICFlowMeter feature names.  THIS IS A REFERENCE ONLY: Stage A reads
# the real header out of the Kaggle file and prints every difference, and the
# Kaggle header always wins.  "CWE Flag Count" is CICFlowMeter's own
# (mis)spelling of CWR -- keep it exactly as the dataset has it.
KAGGLE_FEATURES = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "CWE Flag Count",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

LABEL_COLUMN = "Label"

# The exact header, in the exact order, that EVERY output CSV must have.
# 6 identifiers + 66 features + Label = 73 columns.
FLOW_COLUMNS = ID_COLUMNS + KAGGLE_FEATURES + [LABEL_COLUMN]

# Bookkeeping columns Step 2 adds on top of FLOW_COLUMNS:
#   Source_Capture -- which capture the flow came from (from the file name)
#   Orig_Row       -- the row's position inside its own file, so a merged row
#                     can always be traced back to its source CSV
#   Timestamp_dt   -- the Timestamp text parsed into a real datetime.  The
#                     original text column is KEPT, untouched, as well.
MERGE_EXTRA_COLUMNS = ["Source_Capture", "Orig_Row", "Timestamp_dt"]
MERGED_COLUMNS = FLOW_COLUMNS + MERGE_EXTRA_COLUMNS

# ---------------------------------------------------------------------------
# RENAME_MAP: extractor name (snake_case)  ->  MSCAD name
# ---------------------------------------------------------------------------
# cicflowmeter 0.5.0 writes snake_case column names.  This map is EXPLICIT and
# COMPLETE on purpose: a test asserts that every column of FLOW_COLUMNS except
# "Label" appears exactly once as a value here.  "Label" is not renamed -- it is
# added afterwards by the labelling step.
#
# If a MSCAD column has no entry here, the converter raises an ERROR for that
# capture.  It must never invent or fabricate a value.
RENAME_MAP = {
    # --- the 6 flow identifiers -------------------------------------------
    "src_ip": "Src IP",
    "dst_ip": "Dst IP",
    "src_port": "Src Port",
    "dst_port": "Dst Port",
    "protocol": "Protocol",
    "timestamp": "Timestamp",
    # --- counts and sizes --------------------------------------------------
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Tot Fwd Pkts",
    "tot_bwd_pkts": "Tot Bwd Pkts",
    "totlen_fwd_pkts": "TotLen Fwd Pkts",
    "totlen_bwd_pkts": "TotLen Bwd Pkts",
    "fwd_pkt_len_max": "Fwd Pkt Len Max",
    "fwd_pkt_len_min": "Fwd Pkt Len Min",
    "fwd_pkt_len_mean": "Fwd Pkt Len Mean",
    "fwd_pkt_len_std": "Fwd Pkt Len Std",
    "bwd_pkt_len_max": "Bwd Pkt Len Max",
    "bwd_pkt_len_min": "Bwd Pkt Len Min",
    "bwd_pkt_len_mean": "Bwd Pkt Len Mean",
    "bwd_pkt_len_std": "Bwd Pkt Len Std",
    # --- rates -------------------------------------------------------------
    "flow_byts_s": "Flow Byts/s",
    "flow_pkts_s": "Flow Pkts/s",
    "fwd_pkts_s": "Fwd Pkts/s",
    "bwd_pkts_s": "Bwd Pkts/s",
    # --- inter-arrival times (microseconds) --------------------------------
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Tot",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Tot",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    # --- flags and headers -------------------------------------------------
    "bwd_psh_flags": "Bwd PSH Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Len",
    "bwd_header_len": "Bwd Header Len",
    # --- packet-length statistics -----------------------------------------
    "pkt_len_min": "Pkt Len Min",
    "pkt_len_max": "Pkt Len Max",
    "pkt_len_mean": "Pkt Len Mean",
    "pkt_len_std": "Pkt Len Std",
    "pkt_len_var": "Pkt Len Var",
    # --- TCP flag counters -------------------------------------------------
    "fin_flag_cnt": "FIN Flag Cnt",
    "syn_flag_cnt": "SYN Flag Cnt",
    "rst_flag_cnt": "RST Flag Cnt",
    "psh_flag_cnt": "PSH Flag Cnt",
    "ack_flag_cnt": "ACK Flag Cnt",
    "urg_flag_cnt": "URG Flag Cnt",
    "cwr_flag_count": "CWE Flag Count",   # CICFlowMeter's own spelling mismatch
    "ece_flag_cnt": "ECE Flag Cnt",
    # --- ratios and averages ----------------------------------------------
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Pkt Size Avg",
    "fwd_seg_size_avg": "Fwd Seg Size Avg",
    "bwd_seg_size_avg": "Bwd Seg Size Avg",
    # --- subflows ----------------------------------------------------------
    "subflow_fwd_pkts": "Subflow Fwd Pkts",
    "subflow_fwd_byts": "Subflow Fwd Byts",
    "subflow_bwd_pkts": "Subflow Bwd Pkts",
    "subflow_bwd_byts": "Subflow Bwd Byts",
    # --- window size and data packets -------------------------------------
    "init_bwd_win_byts": "Init Bwd Win Byts",
    "fwd_act_data_pkts": "Fwd Act Data Pkts",
    # --- active / idle periods (microseconds) ------------------------------
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
}

# Columns cicflowmeter produces that MSCAD does not keep.  They are dropped and
# listed in the conversion log -- never silently discarded.  This list is only
# a convenience for the report; anything not in RENAME_MAP is dropped anyway.
EXPECTED_EXTRA_COLUMNS = [
    "fwd_psh_flags",
    "fwd_urg_flags",
    "init_fwd_win_byts",
    "fwd_seg_size_min",
    "fwd_byts_b_avg",
    "fwd_pkts_b_avg",
    "fwd_blk_rate_avg",
    "bwd_byts_b_avg",
    "bwd_pkts_b_avg",
    "bwd_blk_rate_avg",
    "fragmented_pkts",      # how many packets of the flow were IP fragments
]

# Time features that must be converted from SECONDS to MICROSECONDS so they
# match the scale of the Kaggle MSCAD file.
MICROSECOND_COLUMNS = [
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# ===========================================================================
# 5. LABELS  (official MSCAD spellings -- do NOT "correct" them)
# ===========================================================================
# Key   = capture file stem, exactly as the .pcapng is named.
# Value = label for flows that involve an attacker IP.
# Flows in an attack capture that do NOT involve an attacker IP get
# BENIGN_LABEL.  N-0 is pure normal traffic: every flow there gets "Normal".
#
# These labels come from the DOCUMENTED capture -> attack mapping of the MSCAD
# dataset plus the attacker IPs measured from the traffic.  They are never
# guessed from packet contents.
CAPTURE_LABELS = {
    "N-0": "Normal",            # normal traffic only
    "Scan-1": "Port_Scan",      # full / SYN / FIN / UDP port scans
    "App-01": "DoS slowloris",  # HTTP Slowloris, several attacker IPs
    "App-02": "ICMP_Flood",     # ICMP flood + IP-fragmented large pings
    "W-B-01": "Web_Crwling",    # web crawling (the dataset spells it "Crwling")
    "W-B-02": "Brute_Force",    # password cracking
}

# Captures in which every flow is normal, so no attacker logic is needed.
NORMAL_CAPTURES = ["N-0"]

# Label for flows in an attack capture that do not involve an attacker IP.
BENIGN_LABEL = "Benign"

# Attacker IPs seen in earlier runs.  Written here ONLY so the Stage C/D report
# can be compared against them.  The code always measures attackers from the
# traffic; it never labels from this dict.
EXPECTED_ATTACKERS_REFERENCE = {
    "Scan-1": ["192.168.159.1"],
    "W-B-01": ["192.168.159.1"],
    "App-01": ["192.168.159.10", "192.168.159.11", "192.168.159.12"],
    "App-02": ["192.168.159.14", "192.168.159.10", "192.168.159.12"],
}

# ===========================================================================
# 6. SMALL HELPERS
# ===========================================================================


def all_paths() -> dict:
    """Every configured path, for the notebook's 'print paths + exists' cell."""
    return {
        "DATA_ROOT": DATA_ROOT,
        "PROJECT_ROOT": PROJECT_ROOT,
        "PCAPNG_DIR (input, read-only)": PCAPNG_DIR,
        "PCAP_DIR (output B)": PCAP_DIR,
        "CSV_OUTPUT_DIR (output C/D)": CSV_OUTPUT_DIR,
        "RAW_DIR": RAW_DIR,
        "KAGGLE_DIR (reference)": KAGGLE_DIR,
        "KAGGLE_CSV_DIR (reference)": KAGGLE_CSV_DIR,
        "CSV_INPUT_DIR (step 2 reads)": CSV_INPUT_DIR,
        "MERGED_DIR (= step 3 out)": MERGED_DIR,
        "MERGED_CSV": MERGED_CSV,
        "OUTPUT_DIR (step 1)": OUTPUT_DIR,
        "STEP2_OUTPUT_DIR": STEP2_OUTPUT_DIR,
        "STEP3_OUTPUT_DIR": STEP3_OUTPUT_DIR,
        "STEP4_OUTPUT_DIR": STEP4_OUTPUT_DIR,
        "SRC_DIR": SRC_DIR,
        "TESTS_DIR": TESTS_DIR,
        "PYTHON_EXE": Path(PYTHON_EXE),
        "EDITCAP_EXE": Path(EDITCAP_EXE),
        "TSHARK_EXE": Path(TSHARK_EXE),
        "CAPINFOS_EXE": Path(CAPINFOS_EXE),
    }


def find_kaggle_file() -> Path | None:
    """Return the Kaggle MSCAD file inside KAGGLE_DIR, or None if absent.

    Prefers an exact 'MSCAD.csv', then any .csv, then any .xlsx/.xls.
    """
    if not KAGGLE_DIR.is_dir():
        return None
    entries = sorted(p for p in KAGGLE_DIR.iterdir() if p.is_file())
    for p in entries:
        if p.name.lower() == "mscad.csv":
            return p
    for ext in (".csv", ".xlsx", ".xls"):
        for p in entries:
            if p.suffix.lower() == ext:
                return p
    return None


def find_kaggle_csv(folder: Path | None = None) -> tuple[Path | None, list[Path]]:
    """Find the Kaggle MSCAD file in KAGGLE_CSV_DIR.

    Returns (chosen file or None, every candidate found) so the caller can show
    what was there and say which one it picked -- the choice is never silent.

    Order of preference:
      1. a .csv whose name starts with "mscad"   (MSCAD.csv, MSCAD_v2.csv, ...)
      2. any other .csv
      3. an .xlsx / .xls whose name starts with "mscad"
      4. any other .xlsx / .xls           (these need openpyxl to read)
    """
    base = Path(folder) if folder is not None else KAGGLE_CSV_DIR
    if not base.is_dir():
        return None, []

    candidates = sorted(p for p in base.iterdir()
                        if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls"))
    if not candidates:
        return None, []

    def pick(suffixes, named_mscad):
        for p in candidates:
            if p.suffix.lower() not in suffixes:
                continue
            if named_mscad and not p.name.lower().startswith("mscad"):
                continue
            if not named_mscad and p.name.lower().startswith("mscad"):
                continue
            return p
        return None

    for suffixes, named in (({".csv"}, True), ({".csv"}, False),
                            ({".xlsx", ".xls"}, True), ({".xlsx", ".xls"}, False)):
        chosen = pick(suffixes, named)
        if chosen is not None:
            return chosen, candidates
    return candidates[0], candidates


# ===========================================================================
# 7. CROSS-PLATFORM SAFETY
# ===========================================================================
# A Windows absolute path such as r"C:\MSCAD2022" is NOT absolute on Linux or
# macOS -- Path() treats it as a RELATIVE folder literally named "C:\MSCAD2022".
# Running this code off-Windows therefore used to create that folder inside the
# repository, and git on Windows then refuses to check it out:
#     error: invalid path 'Notebook/C:\MSCAD2022/outputs/step3/...'
# These helpers make the code refuse to write there instead.


def is_windows_absolute(path) -> bool:
    r"""True for a path like C:\... or \\server\share, on ANY operating system."""
    return PureWindowsPath(str(path)).drive != ""


def running_on_windows() -> bool:
    return os.name == "nt"


def usable_output_dir(path) -> Path | None:
    """The folder to write into, or None when it would be unsafe.

    None means: this is a Windows path and we are not on Windows, so creating
    it would silently make a junk folder in the current working directory.
    Callers report that and skip writing rather than producing garbage.
    """
    if not running_on_windows() and is_windows_absolute(path):
        return None
    return Path(path)


def why_unusable(path) -> str:
    """A one-line explanation for a caller to print."""
    return ("%s is a Windows path (drive %r) but this is %s, where it would be "
            "created as a RELATIVE folder of that literal name. Nothing was "
            "written. Run on Windows, or point DATA_ROOT at a local folder "
            "(MSCAD_DATA_ROOT=/some/dir)."
            % (path, PureWindowsPath(str(path)).drive, os.name))


# ===========================================================================
# 8. STEP 4 -- PREPROCESSING (clean, encode, scale)
# ===========================================================================
# Step 4 cleans, encodes and scales the WHOLE merged dataset.  It deliberately
# does NOT split into train/test, does NOT touch the class imbalance, and does
# NOT select features or model anything -- those are later steps.

# What to do with inf in the rate columns (zero-duration flows divide by zero):
#   "cap"        -> +inf becomes that column's maximum FINITE value, and a
#                   binary column "zero_duration" records which rows they were
#   "nan_median" -> inf becomes NaN, then median imputation fills it
INF_STRATEGY = "cap"

# "minmax" scales every numeric feature into [0, 1]; "standard" centres it on
# mean 0 with unit variance.
SCALER = "minmax"

# False = if the Step 4 outputs already exist, load them instead of redoing it.
OVERWRITE_STEP4 = False

# Columns that are NOT features.  They are removed from the feature matrix and
# kept in metadata.csv, in the same row order, because the attack-sequence and
# attack-state-graph work in a later step needs them.  Protocol is deliberately
# NOT here: it IS a feature, and gets one-hot encoded.
METADATA_COLUMNS = [
    "Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol",
    "Timestamp", "Timestamp_dt", "Source_Capture", "Orig_Row",
]

# Columns that appear in BOTH metadata.csv and the feature matrix.
# Protocol is an identifier AND a feature: metadata.csv keeps the raw number so
# every identifier lives in one place, while the feature pipeline still one-hot
# encodes it.  Without this exception, moving Protocol into metadata would strip
# it from the features and the Protocol_* columns would never be created.
METADATA_SHARED_COLUMNS = ["Protocol"]

# Labels that count as "not an attack" for the binary target.
# "Benign" (a non-attacker flow inside an attack capture) and "Normal" (the N-0
# capture) stay SEPARATE classes for the multi-class target; they only share
# the binary 0.
BENIGN_LABELS = ["Benign", "Normal"]

# Optional label fixes, applied ONLY if non-empty and always reported.
# The dataset's own spelling is kept by default -- nothing is renamed silently.
# Example:  LABEL_MAP = {"Web_Crwling": "Web_Crawling"}
LABEL_MAP: dict[str, str] = {}

# Protocol numbers that get their own one-hot column; anything else (IGMP = 2,
# say) falls into Protocol_other so an unseen value can never be dropped.
PROTOCOL_ONE_HOT = {1: "Protocol_ICMP", 6: "Protocol_TCP", 17: "Protocol_UDP"}
PROTOCOL_OTHER_COLUMN = "Protocol_other"

# The binary column added when INF_STRATEGY == "cap".
ZERO_DURATION_COLUMN = "zero_duration"

# Step 4 output files (all inside STEP4_OUTPUT_DIR).
STEP4_FILES = {
    "features_clean": "features_clean.csv",      # cleaned + encoded, NOT scaled
    "features_scaled": "features_scaled.csv",    # cleaned + encoded + scaled
    "labels": "labels.csv",                      # Label, Label_encoded, Label_binary
    "metadata": "metadata.csv",                  # the non-feature columns
    "label_mapping": "label_mapping.json",
    "feature_list": "feature_list.json",
    "artifacts": "preprocess_artifacts.joblib",
    "report": "step4_report.csv",
    "combined": "preprocessing.csv",
}

# ---------------------------------------------------------------------------
# The single combined file: the final preprocessed dataset in one CSV.
# ---------------------------------------------------------------------------
# It is built by COLUMN-JOINING the frames Step 4 already produced -- nothing is
# recomputed and nothing is re-scaled, so it cannot drift from the separate
# files.  The other eight Step 4 outputs are unchanged.
COMBINED_CSV = STEP4_OUTPUT_DIR / "preprocessing.csv"

# True  -> the SCALED features (features_scaled.csv)
# False -> the cleaned but UNSCALED features (features_clean.csv), which is what
#          a later step wants if it will fit its own scaler on a training split
COMBINED_USE_SCALED = True

# The flow identifiers placed at the FRONT of preprocessing.csv, in this order.
# They carry their ORIGINAL values from the Step 3 merged data: IP strings as
# they were, ports as integers, Timestamp as the original text.  None of them is
# scaled.
# Set it to [] for the old layout, features and labels only.
COMBINED_ID_COLUMNS = ["Src IP", "Dst IP", "Src Port", "Dst Port",
                       "Protocol", "Timestamp"]

# Of those, the ones READ FROM metadata.csv.  Protocol is deliberately absent:
# in preprocessing.csv it is a readable TEXT column derived from the Protocol_*
# one-hot features (see PROTOCOL_TEXT_FROM_ONE_HOT), not a number copied out of
# metadata.  That is what lets the file be rebuilt from an OLD metadata.csv
# written before Protocol was added to METADATA_COLUMNS.
COMBINED_METADATA_ID_COLUMNS = [c for c in COMBINED_ID_COLUMNS if c != "Protocol"]

# The derived Protocol column in preprocessing.csv: one readable name instead of
# four 0/1 columns.  Exactly one of the four one-hot columns is 1 in every row,
# so the mapping is lossless and reversible.
COMBINED_PROTOCOL_COLUMN = "Protocol"
PROTOCOL_TEXT_FROM_ONE_HOT = {
    "Protocol_ICMP": "ICMP",
    "Protocol_TCP": "TCP",
    "Protocol_UDP": "UDP",
    PROTOCOL_OTHER_COLUMN: "OTHER",
}
PROTOCOL_TEXT_VALUES = ["ICMP", "TCP", "UDP", "OTHER"]

# The four one-hot columns are LEFT OUT of preprocessing.csv -- the single
# Protocol text column replaces them there.  They stay in features_clean.csv,
# features_scaled.csv and feature_list.json, which is what a model is trained
# on, because a model needs numeric input.
COMBINED_DROP_FEATURE_COLUMNS = list(PROTOCOL_TEXT_FROM_ONE_HOT)

# preprocessing.csv carries ONLY the original Label string, as its last column.
# Label_encoded and Label_binary are not in it; labels.csv still has all three.
COMBINED_LABEL_COLUMNS = [LABEL_COLUMN]

# The same list under the name that says what it is FOR: these columns must
# never be fed to a model.  The attacker IPs decided the labels back in Step 1,
# so training on them would be label leakage and would score near-perfectly
# while learning nothing.  They are in preprocessing.csv so a row can be traced
# back to its flow, and for the sequence / attack-state-graph work later.
# preprocessing.get_feature_matrix(df) returns only the feature_list.json
# columns, which is what a model should actually see.
ID_COLUMNS_NOT_FOR_TRAINING = COMBINED_ID_COLUMNS

# Port numbers must stay integers in preprocessing.csv.  Protocol is NOT here:
# it is the readable text column ("TCP", "UDP", "ICMP", "OTHER").
COMBINED_INT_COLUMNS = ["Src Port", "Dst Port"]


# ===========================================================================
# STEP 5 -- train / test validation split
# ===========================================================================
# Input:  the Step 4 outputs (features_clean.csv, labels.csv, metadata.csv).
# Output: STEP5_OUTPUT_DIR, created if missing.
#
# Step 5 does the split and NOTHING else: no SMOTE, no resampling, no class
# weights, no feature selection, no modelling.  Those are later steps.
STEP5_OUTPUT_DIR = OUTPUTS_DIR / "step5"      # C:\MSCAD2022\outputs\step5

# 20% of the rows go to the test set.
TEST_SIZE = 0.20

# Fixed so the split is reproducible; the same value as RANDOM_SEED above, kept
# as its own name because sklearn calls this argument random_state.
RANDOM_STATE = RANDOM_SEED

# Stratify on the MULTICLASS label, not the binary one.  Stratifying on
# Label_binary would balance attack-vs-benign while letting a 23-row class land
# entirely in one side of the split.
STRATIFY_COLUMN = LABEL_COLUMN

# False = if the Step 5 outputs already exist, load them instead of resplitting.
OVERWRITE_STEP5 = False

# Scaled test values may fall slightly outside [0, 1]: the scaler is fitted on
# the training rows only, so a test row can hold a larger value than any
# training row.  That is expected and is REPORTED, not hidden.  Clipping would
# quietly destroy real information, so it is off by default.
CLIP_TEST_TO_TRAIN_RANGE = False

# A class share may differ between train and test by less than this many
# PERCENTAGE POINTS for the split to be called stratified.
SHARE_TOLERANCE_PP = 0.1

# The row count Step 4 is expected to hand over.  It is only a cross-check that
# prints a note: the real check is that features_clean.csv, labels.csv and
# metadata.csv all have the SAME number of rows, which is what matters for the
# split.  Set it to None to drop the note.
STEP5_EXPECTED_ROWS = 105_387

# Step 5 output files (all inside STEP5_OUTPUT_DIR).
STEP5_FILES = {
    "X_train": "X_train.csv",                    # scaled with the TRAIN-fitted scaler
    "X_test": "X_test.csv",
    "X_train_unscaled": "X_train_unscaled.csv",  # straight from features_clean.csv
    "X_test_unscaled": "X_test_unscaled.csv",
    "y_train": "y_train.csv",                    # Label, Label_encoded, Label_binary
    "y_test": "y_test.csv",
    "meta_train": "meta_train.csv",              # identifiers, incl. Orig_Row
    "meta_test": "meta_test.csv",
    "train_indices": "train_indices.csv",        # row positions in the Step 4 files
    "test_indices": "test_indices.csv",
    "train_view": "train_view.csv",              # preprocessing.csv layout, for viewing
    "test_view": "test_view.csv",
    "class_balance": "class_balance.csv",        # per-class train/test counts
    "binary_balance": "binary_balance.csv",      # the same for Label_binary
    "leakage": "leakage.csv",                    # test rows also present in train
    "split_summary": "split_summary.csv",
    "scaler": "scaler_train.joblib",             # fitted on the TRAIN rows only
    "report": "step5_report.csv",
    "chart": "split_class_counts.png",
}


# ===========================================================================
# STEP 6 -- feature selection (MI + Pearson + permutation importance)
# ===========================================================================
# Input:  the Step 5 outputs (X_train.csv, X_test.csv, y_train.csv, y_test.csv).
# Output: STEP6_OUTPUT_DIR, created if missing.
#
# EVERY selection decision is made on the TRAINING set alone.  X_test is touched
# once, at the very end, only to take the same columns.  Choosing features by
# looking at the test set is the same leak as scaling before the split: the
# reported score stops meaning anything.
STEP6_OUTPUT_DIR = OUTPUTS_DIR / "step6"      # C:\MSCAD2022\outputs\step6

# 6.3 -- Pearson redundancy filter.
# |r| >= PEARSON_THRESHOLD means the two features carry the same information, so
# one of them is dropped.  0.95 is deliberately strict: it removes near-copies
# (Subflow Fwd Pkts vs Tot Fwd Pkts, say) without touching features that merely
# move together.
PEARSON_THRESHOLD = 0.95
# Pairs in [PEARSON_REPORT_LOWER, PEARSON_THRESHOLD) are REPORTED and kept.  They
# matter because permutation importance understates both halves of such a pair:
# shuffling one lets the model lean on the other.
PEARSON_REPORT_LOWER = 0.80

# 6.6 -- how many features survive.
N_SELECTED_FEATURES = 30

# 6.4 -- permutation importance.
# An inner split of the TRAINING set only: fit the forest on one part, measure
# the importance drop on the other.  The Step 5 test set is not involved.
PI_VALIDATION_SIZE = 0.20
PI_N_REPEATS = 10
# macro-F1 weights every class equally, so shuffling a feature that only the
# 23-row classes depend on still shows up.  "accuracy" would hide it entirely.
PI_SCORING = "f1_macro"
RF_PARAMS = dict(n_estimators=200, class_weight="balanced",
                 n_jobs=-1, random_state=RANDOM_STATE)

# 6.2 -- mutual information.
MI_RANDOM_STATE = RANDOM_STATE
MI_N_NEIGHBORS = 3

# A negative permutation importance means shuffling the feature made the score
# BETTER, which is noise around zero, not evidence against the feature.  Clipping
# to 0 keeps the min-max normalisation from being stretched by that noise.
CLIP_NEGATIVE_PI = True

# False = if the Step 6 outputs already exist, load them instead of reselecting.
OVERWRITE_STEP6 = False

# Step 6 output files (all inside STEP6_OUTPUT_DIR).
STEP6_FILES = {
    "mi_all": "mi_scores_all.csv",               # every feature, before Pearson
    "mi_kept": "mi_scores.csv",                  # the Pearson survivors only
    "corr_matrix": "pearson_corr_matrix.csv",
    "dropped_pairs": "pearson_dropped_pairs.csv",
    "remaining_pairs": "pearson_remaining_pairs.csv",
    "pi_scores": "pi_scores.csv",
    "per_class_f1": "rf_per_class_f1.csv",
    "pi_overlap": "pi_duplicate_overlap.csv",
    "ranking": "combined_ranking.csv",
    "selected": "selected_features.json",
    "X_train_selected": "X_train_selected.csv",
    "X_test_selected": "X_test_selected.csv",
    "y_train": "y_train.csv",                    # copied from Step 5, unchanged
    "y_test": "y_test.csv",
    "rf_baseline": "rf_baseline.joblib",
    "report": "step6_report.csv",
    "heatmap": "pearson_heatmap.png",
    "ranking_chart": "combined_ranking_top30.png",
    "funnel": "feature_funnel.png",
}


# ===========================================================================
# STEP 7 -- autoencoder feature extraction  /  STEP 8 -- transform both sets
# ===========================================================================
# Input:  the Step 6 outputs (X_train_selected.csv, X_test_selected.csv, the
#         labels, and selected_features.json).
# Output: STEP7_OUTPUT_DIR (the trained autoencoder) and STEP8_OUTPUT_DIR (the
#         encoded Z matrices), both created if missing.
#
# The autoencoder is UNSUPERVISED: it is trained to reconstruct its own input
# and never sees a label.  The labels appear in Step 7 for exactly one purpose,
# to STRATIFY the inner split used for early stopping, so the 23-row classes are
# represented in the validation part instead of landing entirely in the fit part
# by chance.
STEP7_OUTPUT_DIR = OUTPUTS_DIR / "step7"      # C:\MSCAD2022\outputs\step7
STEP8_OUTPUT_DIR = OUTPUTS_DIR / "step8"      # C:\MSCAD2022\outputs\step8

# The framework, picked once and used for Steps 7-8 and 10-12 alike.  See
# AE_FRAMEWORK_NOTE for why.
AE_FRAMEWORK = "tensorflow"

# 30 -> 24 -> 15, and the decoder mirrors it back 15 -> 24 -> 30.
AE_LATENT_DIM = 15
AE_HIDDEN = [24]
AE_ACTIVATION = "relu"
# The Step 5 scaler put every feature into [0, 1], so a sigmoid output can
# reach every legal value and nothing else.  A linear output would be free to
# predict -3 or 7, which no input can be.
AE_OUTPUT_ACT = "sigmoid"
AE_LOSS = "mse"

AE_EPOCHS = 100
AE_BATCH_SIZE = 256
AE_LR = 1e-3

# An inner split of the TRAINING set, for early stopping only.  The Step 5 test
# set is not involved in Step 7 at all.
AE_VAL_SIZE = 0.10
AE_PATIENCE = 10

# False = if the outputs already exist, load them instead of retraining.
OVERWRITE_STEP7 = False
OVERWRITE_STEP8 = False

# Z columns are z01 ... z15, zero-padded so they sort correctly as text.
Z_COLUMN_PREFIX = "z"

STEP7_FILES = {
    "autoencoder": "autoencoder.keras",          # the full encoder + decoder
    "encoder": "encoder.keras",                  # the half Step 8 needs
    "history": "training_history.csv",
    "ae_config": "ae_config.json",               # hyperparameters + feature order
    "architecture": "architecture.csv",
    "per_feature_mse": "per_feature_mse.csv",
    "report": "step7_report.csv",
    "loss_curve": "loss_curve.png",
}

STEP8_FILES = {
    "Z_train": "Z_train.csv",
    "Z_test": "Z_test.csv",
    "y_train": "y_train.csv",                    # copied from Step 6, unchanged
    "y_test": "y_test.csv",
    "recon_mse": "reconstruction_mse.csv",
    "report": "step8_report.csv",
}


# ===========================================================================
# STEP 9 -- class imbalance, handled in the TRAINING LOSS only
# ===========================================================================
# Input:  the Step 8 outputs (Z_train.csv, y_train.csv).
# Output: STEP9_OUTPUT_DIR, created if missing.
#
# CLASS WEIGHTING ONLY.  No SMOTE, no SMOTE-ENN, no over- or undersampling,
# and imbalanced-learn is deliberately not a dependency.  Every row that goes in
# comes out: the weights change how much each row COUNTS in the loss, never how
# many rows there are.
#
# Why weighting and not resampling, on this dataset.  About half the rows share
# identical feature values once the identifiers are gone -- thousands of scan,
# flood and brute-force flows genuinely look alike.  SMOTE interpolates between
# nearest neighbours, so on identical rows it manufactures copies rather than new
# information, and oversampling before an inner split copies rows across it.
# Weighting avoids both: it adds no rows at all.
#
# The TEST SET IS NEVER TOUCHED by any of this.  Weights are computed from the
# training rows alone, applied only in the training loss, and the test set is
# scored exactly as it comes out of Step 8.  Reweighting the test set would make
# the reported score a different question from the one being asked.
STEP9_OUTPUT_DIR = OUTPUTS_DIR / "step9"      # C:\MSCAD2022\outputs\step9

# "class_weight" -> weight the loss.  "none" -> every weight is 1.0, which is the
# honest baseline to compare against.
IMBALANCE_STRATEGY = "class_weight"

# "balanced"      : w_c = n_samples / (n_classes * n_c)   -- sklearn's formula
# "sqrt_balanced" : the square root of that.  Milder: it still favours the rare
#                   classes but by ~32x rather than ~1000x, which is worth
#                   comparing if training turns out unstable.
CLASS_WEIGHT_SCHEME = "balanced"

# An optional ceiling on any single weight, e.g. 100.0.  None = no cap.
CLASS_WEIGHT_MAX = None

# Rescale so the MEAN PER-SAMPLE weight is 1.  This leaves the relative weights
# untouched and only fixes the overall scale, so the effective learning rate does
# not change when the scheme does.
NORMALISE_WEIGHTS = True

# The inner validation split Step 10 uses for early stopping.  It is carved out
# of the training set here, and the class weights are computed from the FIT part
# only -- never from the validation part, or the early-stopping signal would be
# contaminated by the thing it is meant to judge.
CLF_VAL_SIZE = 0.10

# False = if the Step 9 outputs already exist, load them instead of recomputing.
OVERWRITE_STEP9 = False

STEP9_FILES = {
    "Z_fit": "Z_fit.csv",
    "y_fit": "y_fit.csv",
    "Z_val": "Z_val.csv",
    "y_val": "y_val.csv",
    "fit_sample_weights": "fit_sample_weights.csv",
    "class_weights": "class_weights.json",
    "report": "imbalance_report.csv",
    "fit_counts_chart": "fit_class_counts.png",
    "weights_chart": "class_weights.png",
}


# ===========================================================================
# STEP 10 -- the proposed hybrid CNN-LSTM  /  STEP 11 -- evaluation
# ===========================================================================
# Input:  Step 8's Z matrices, Step 9's fit/validation split and class weights.
# Output: STEP10_OUTPUT_DIR (the trained model) and STEP11_OUTPUT_DIR (the test
#         metrics), both created if missing.
STEP10_OUTPUT_DIR = OUTPUTS_DIR / "step10"
STEP11_OUTPUT_DIR = OUTPUTS_DIR / "step11"

# ---------------------------------------------------------------------------
# HOW A FLOW IS ARRANGED FOR THE NETWORK -- read this before quoting a result
# ---------------------------------------------------------------------------
# "feature_seq" (default)
#   Each flow becomes a tensor of shape (AE_LATENT_DIM, 1): the 15 autoencoder
#   features laid out as 15 "steps" of one channel.  The Conv1D layers learn
#   local patterns across NEIGHBOURING LATENT FEATURES, and the LSTM then reads
#   their output feature map as a sequence of learned feature patterns.
#
#   This is NOT a time sequence, and the model is not doing temporal modelling.
#   Worse, the autoencoder's latent units have NO NATURAL ORDER: which feature
#   lands at position 3 versus position 11 is an arbitrary consequence of how the
#   encoder initialised, so "neighbouring" means nothing in particular.  The
#   arrangement is fixed and reproducible (the encoder is saved), and the
#   architecture is the one the thesis proposes, so it is implemented as
#   specified -- but any claim that the CNN-LSTM captures temporal attack
#   structure needs INPUT_MODE = "time_window", not this.
#
# "time_window"
#   Windows of WINDOW_SIZE consecutive flows, ordered by Timestamp WITHIN one
#   Source_Capture, labelled by the last flow in the window.  Windows are built
#   separately inside train and inside test so none ever straddles the split.
#   The caveat here is different: the Step 5 split was random and stratified, so
#   consecutive rows on one side were not consecutive in the capture -- there are
#   gaps, and a "window" spans more wall-clock time than it appears to.
INPUT_MODE = "feature_seq"
WINDOW_SIZE = 10

# The architecture the thesis proposes.
CNN_FILTERS = [64, 64]
CNN_KERNEL = 3
CNN_POOL = 2
LSTM_UNITS = 64
DROPOUT = 0.3
DENSE_UNITS = 64

CLF_EPOCHS = 50
CLF_BATCH_SIZE = 256
CLF_LR = 1e-3
CLF_PATIENCE = 8

# Early stopping watches macro-F1 on the validation part, not loss: with a 3849:1
# imbalance the loss is dominated by the majority class, and a model that quietly
# stops predicting the rare classes can still improve it.
EARLY_STOP_METRIC = "val_macro_f1"

# 1 trains once.  Set 3 or 5 to retrain with different seeds and report
# mean +- std, which is what the rare classes need: with ~5 test rows each, a
# single run's per-class F1 moves by a lot on one row changing hands.
N_SEEDS = 1

OVERWRITE_STEP10 = False
OVERWRITE_STEP11 = False

STEP10_FILES = {
    "model": "cnn_lstm.keras",
    "history": "training_history.csv",
    "model_card": "model_card.json",
    "architecture": "architecture.csv",
    "report": "step10_report.csv",
    "loss_curve": "loss_curve.png",
    "f1_curve": "val_macro_f1_curve.png",
}

STEP11_FILES = {
    "metrics": "metrics.json",
    "per_class": "per_class_report.csv",
    "predictions": "predictions.csv",
    "confusion_counts": "confusion_matrix_counts.csv",
    "confusion_normalised": "confusion_matrix_rownorm.csv",
    "duplicate_split": "duplicate_free_comparison.csv",
    "binary": "binary_view.csv",
    "report": "step11_report.csv",
    "confusion_chart": "confusion_matrix.png",
}

# Declared above, before the Step 5-11 blocks existed.
DIRS_TO_CREATE.append(STEP5_OUTPUT_DIR)
DIRS_TO_CREATE.append(STEP6_OUTPUT_DIR)
DIRS_TO_CREATE.append(STEP7_OUTPUT_DIR)
DIRS_TO_CREATE.append(STEP8_OUTPUT_DIR)
DIRS_TO_CREATE.append(STEP9_OUTPUT_DIR)
DIRS_TO_CREATE.append(STEP10_OUTPUT_DIR)
DIRS_TO_CREATE.append(STEP11_OUTPUT_DIR)


# ---------------- Step 12 - Model comparison ----------------
# APPENDED for Step 12.  Nothing above this line was changed.
#
# Step 12 compares the proposed pipeline with three baselines by CALLING the
# Step 7-11 functions, never by modifying them.  Each configuration removes one
# piece of the proposal, so a difference between two rows is attributable:
#
#   Proposed    feature selection -> AE(30->15) -> CNN-LSTM
#   Baseline 1  (no selection)    -> AE(all->15) -> CNN-LSTM   -- what selection buys
#   Baseline 2  (no selection)    -> AE(all->15) -> LSTM only  -- what the CNN buys
#   Baseline 3  (no selection, no AE)           -> CNN-LSTM    -- what the AE buys
STEP12_OUTPUT_DIR = OUTPUTS_DIR / "step12"

# "all"        -> every Step 5 feature, which is the honest "no feature
#                 selection, no autoencoder" baseline.
# "selected30" -> the Step 6 features without the autoencoder, which isolates
#                 the AE alone rather than selection and the AE together.
BASELINE3_FEATURES = "all"

# feature_source: "selected30" (Step 6) or "all" (Step 5)
# use_ae:         train an autoencoder on the TRAINING rows and encode both sides
# model_type:     "cnn_lstm" or "lstm"
EXPERIMENTS = [
    {"name": "proposed", "label": "Proposed: FS -> AE -> CNN-LSTM",
     "feature_source": "selected30", "use_ae": True, "model_type": "cnn_lstm",
     "reuse_step7_10": True},
    {"name": "baseline1_ae_cnn_lstm", "label": "B1: AE -> CNN-LSTM (no selection)",
     "feature_source": "all", "use_ae": True, "model_type": "cnn_lstm",
     "reuse_step7_10": False},
    {"name": "baseline2_ae_lstm", "label": "B2: AE -> LSTM (no CNN)",
     "feature_source": "all", "use_ae": True, "model_type": "lstm",
     "reuse_step7_10": False},
    {"name": "baseline3_cnn_lstm", "label": "B3: CNN-LSTM (no AE)",
     "feature_source": BASELINE3_FEATURES, "use_ae": False,
     "model_type": "cnn_lstm", "reuse_step7_10": False},
]

# False = a configuration whose metrics.json is already on disk is skipped.
# Training four models is slow, so this makes the notebook resumable.
OVERWRITE_STEP12 = False

STEP12_FILES = {
    "comparison": "comparison_table.csv",
    "per_class_full": "per_class_f1_full.csv",
    "per_class_free": "per_class_f1_duplicate_free.csv",
    "consistency": "consistency_checks.csv",
    "seed_comparison": "seed_comparison.csv",
    "environment": "environment.json",
    "summary": "summary.md",
    "chart_metrics": "comparison_metrics.png",
    "chart_metrics_free": "comparison_metrics_duplicate_free.png",
    "chart_per_class": "per_class_f1_heatmap.png",
    "chart_time": "training_and_inference_time.png",
    "chart_confusion": "confusion_matrices.png",
}

DIRS_TO_CREATE.append(STEP12_OUTPUT_DIR)


# ---------------- Step 13 - Adaptive Evidence Fusion ----------------
# APPENDED for Step 13.  Nothing above this line was changed.
#
# Three heterogeneous models are combined into one fused risk score per flow.
# Unlike a fixed-weight ensemble, each model's weight is set PER FLOW from two
# measured quantities: how reliable that model has been in the current
# behavioural context, and how confident it is on this particular flow.
STEP13_OUTPUT_DIR = OUTPUTS_DIR / "step13"

# The Step 12 configurations to fuse.  These names must exist in EXPERIMENTS;
# fusion.check_fusion_models() verifies that at run time rather than assuming it.
FUSION_MODELS = ["proposed", "baseline2_ae_lstm", "baseline3_cnn_lstm"]

# How a model's reliability in a context is measured on the validation part.
RELIABILITY_METRIC = "f1"          # or "precision"

# Shrinkage towards the model's global macro-F1.  A context with few validation
# rows gives an unstable per-context estimate, so it is pulled towards the global
# figure; K is the number of rows at which the two contribute equally.
SHRINKAGE_K = 20

# Multiply reliability by a per-flow confidence term, 1 - normalised entropy.
CONFIDENCE_TERM = True

# "proportional" -> w_m = s_m / sum_j s_j   (default)
# "softmax"      -> w_m = exp(s_m / T) / sum_j exp(s_j / T), T = WEIGHT_TEMPERATURE
WEIGHT_MODE = "proportional"
WEIGHT_TEMPERATURE = 0.2

# When every s_m is effectively zero the weights are undefined, so the flow falls
# back to equal weights rather than dividing by zero.
EPSILON = 1e-6

# "consensus_class" is label-free -- it uses the models' own mean prediction, so
# it is available at test time.  "protocol" uses the flow's Protocol instead.
CONTEXT_MODE = "consensus_class"   # or "protocol"

# The decision threshold on the fused risk, chosen on VALIDATION and then applied
# unchanged to the test set.
THRESHOLD_RULE = "max_f1"          # or "fpr_at_most"
TARGET_FPR = 0.01

# Exponential moving average of the risk over consecutive flows within one
# capture.  Off by default: the Step 5 split is random, so consecutive test flows
# are not consecutive in the original capture.
TEMPORAL_SMOOTHING = False
EMA_ALPHA = 0.3

OVERWRITE_STEP13 = False

STEP13_FILES = {
    "reliability": "reliability_table.csv",
    "weights_summary": "weights_summary.csv",
    "fused_predictions": "fused_predictions.csv",
    "comparison": "comparison_table.csv",
    "ablation": "ablation_table.csv",
    "per_class": "per_class_f1.csv",
    "threshold": "threshold.json",
    "metrics": "metrics.json",
    "report": "step13_report.csv",
    "chart_reliability": "reliability_heatmap.png",
    "chart_weights": "mean_weight_per_class.png",
    "chart_risk": "risk_distribution.png",
    "chart_curves": "roc_pr_curves.png",
    "chart_comparison": "comparison_bars.png",
    "chart_comparison_free": "comparison_bars_duplicate_free.png",
}

DIRS_TO_CREATE.append(STEP13_OUTPUT_DIR)


# ---------------- Step 14 - Explainability ----------------
# APPENDED for Step 14.  Nothing above this line was changed.
#
# Step 13 produces a number; an analyst needs a reason.  This step turns each
# alert into an evidence report: which input features moved each model, how the
# adaptive fusion combined them into R(x), how confident that evidence is, and
# what to check next.
#
# Nothing here changes a Step 13 decision.  The models, the encoders, the
# fusion weights and tau are all fixed; Step 14 only reads them.
STEP14_OUTPUT_DIR = OUTPUTS_DIR / "step14"

# --- Integrated Gradients -------------------------------------------------
# IG_i(x) = (x_i - x'_i) * integral_0^1 dF(x' + a(x - x'))/dx_i da, evaluated
# with the trapezoidal rule over IG_STEPS intervals.  64 is the usual working
# value: completeness is checked per flow, so the choice is verified rather
# than assumed.
IG_STEPS = 64
IG_BASELINE = "benign_median"      # "benign_median" | "zeros" | "train_mean"
IG_BATCH_SIZE = 256
EXPLAIN_OTHER_MODELS = True        # attribute all three fused models, not just one

# How many test flows to explain per FUSED class.  A class with fewer rows than
# this contributes all of them, so the rare classes are never dropped.
EXPLAIN_SAMPLE_PER_CLASS = 200
EXPLAIN_RANDOM_STATE = RANDOM_STATE
TOP_K_FEATURES = 5

# --- checking the explanations -------------------------------------------
COMPLETENESS_TOL = 0.05            # relative error allowed in sum IG = F(x) - F(x')
STABILITY_NOISE_STD = 0.01         # Gaussian noise added to scaled features
STABILITY_N_REPEATS = 5
PERTURB_MODE = "baseline_replace"  # x_i := x'_i, one feature at a time

# --- the decision bands ---------------------------------------------------
# tau comes from Step 13 and is NOT recomputed.  tau_s is the lower edge of a
# "suspicious" band, chosen on the Step 9 VALIDATION split as the lowest risk
# threshold that still catches SUSPICIOUS_RECALL_TARGET of the attacks there.
SUSPICIOUS_BAND = True
SUSPICIOUS_RECALL_TARGET = 0.99

# --- the confidence column ------------------------------------------------
# High   = fused max probability >= CONF_HIGH_PROB AND all models agree
#          AND the IG completeness check passed
# Medium = fused max probability >= CONF_MEDIUM_PROB OR a majority agrees
# Low    = otherwise
CONF_HIGH_PROB = 0.90
CONF_MEDIUM_PROB = 0.70

# --- analyst context ------------------------------------------------------
# Identifiers are CONTEXT ONLY.  They never enter a model: the MSCAD labels
# were assigned by attacker IP, so a model given Src IP would learn the label.
SOURCE_CONTEXT = True
SOURCE_CONTEXT_WINDOW_MIN = 60

# A label-to-tactic lookup, not an inferred attack progression.  A label that
# is not in this map is reported as "Unmapped" -- never guessed.
ATTACK_STAGE_MAP = {
    "Port_Scan": "Reconnaissance (MITRE ATT&CK TA0043)",
    "Web_Crwling": "Reconnaissance / Discovery",
    "Brute_Force": "Credential Access (TA0006)",
    "DoS slowloris": "Impact - Denial of Service (TA0040)",
    "ICMP_Flood": "Impact - Denial of Service (TA0040)",
    "Benign": "None",
    "Normal": "None",
}

# Suggestions for an analyst to check -- not automated actions, and nothing
# here is executed by any code in this project.
ACTION_TEMPLATES = {
    "Port_Scan": ("Review scanning source {src_ip}; check firewall rules and "
                  "exposed services on {dst_ip}."),
    "Web_Crwling": ("Review web access logs for {src_ip} on {dst_ip}; apply rate "
                    "limiting if unexpected."),
    "Brute_Force": ("Block or rate-limit {src_ip}; audit authentication logs and "
                    "reset credentials on {dst_ip}."),
    "DoS slowloris": ("Apply connection limits/timeouts for {src_ip} on {dst_ip}; "
                      "monitor service availability."),
    "ICMP_Flood": ("Rate-limit or filter ICMP from {src_ip} to {dst_ip}; monitor "
                   "link load."),
}

# False = a finished Step 14 is read back from disk instead of recomputed.
OVERWRITE_STEP14 = False

STEP14_FILES = {
    "evidence": "evidence_records.csv",
    "explanations": "explanations.csv",
    "global_importance": "global_feature_importance.csv",
    "validation": "explanation_validation.csv",
    "agreement": "model_agreement.csv",
    "alerts": "alert_explanations.csv",
    "examples": "example_alert_reports.md",
    "thresholds": "thresholds_step14.json",
    "metrics": "explanation_metrics.json",
    "report": "step14_report.csv",
    "chart_global": "global_top_features.png",
    "chart_heatmap": "class_feature_ig_heatmap.png",
    "chart_signed": "signed_ig_boxplot.png",
    "chart_scatter": "ig_vs_perturbation.png",
    "chart_deletion": "deletion_check.png",
    "chart_contributions": "example_alert_contributions.png",
    "chart_bands": "risk_bands.png",
    "chart_temporal": "temporal_attribution.png",   # time_window mode only
}

DIRS_TO_CREATE.append(STEP14_OUTPUT_DIR)


# ------- Step 12 (addendum) - Baseline 4: AE -> PCA -> LSTM -------
# APPENDED.  Nothing above this line was changed.
#
# Step 12 is finished and its four configurations are verified; Steps 13 and 14
# import that module and address those configurations by name.  So Baseline 4
# is added BESIDE them rather than by editing the list above:
#
#   EXPERIMENTS            the four Step 12 configurations, untouched -- this is
#                          what experiments.run_step12() still iterates
#   EXPERIMENTS_WITH_PCA   those four plus Baseline 4, which is what the new
#                          experiments_pca module compares
#
# Appending Baseline 4 to EXPERIMENTS itself would be worse than it looks: the
# Step 12 runner has no PCA step, so it would quietly train an AE -> LSTM under
# the name "baseline4" -- a second copy of Baseline 2 wearing the wrong label.
BASELINE4_AE_FEATURES = "all"      # "all" -> the Baseline 1 encoder (all -> 15)
                                   # "selected30" -> the saved Step 7 encoder

# PCA on the 15-dimensional AE latent space, fitted on the TRAINING rows only.
#   float in (0, 1) -> keep at least that fraction of the variance
#   int             -> that many components, and it must be <= the latent width
PCA_N_COMPONENTS = 0.95

BASELINE4 = {
    "name": "baseline4_ae_pca_lstm",
    "label": "B4: AE -> PCA -> LSTM",
    "feature_source": BASELINE4_AE_FEATURES,
    "use_ae": True,
    "use_pca": True,
    "model_type": "lstm",
    # the encoder is REUSED, never retrained: from Baseline 1 for "all", from
    # Step 7 for "selected30"
    "reuse_step7_10": BASELINE4_AE_FEATURES == "selected30",
    "reuse_encoder_from": ("baseline1_ae_cnn_lstm"
                           if BASELINE4_AE_FEATURES == "all" else "step7"),
}

# The five-configuration comparison.  EXPERIMENTS keeps its four entries.
EXPERIMENTS_WITH_PCA = EXPERIMENTS + [BASELINE4]

# False = a finished Baseline 4 is read back from disk instead of retrained.
OVERWRITE_BASELINE4 = False

# Written beside the Step 12 files, never over them: the four-configuration
# tables stay exactly as Step 12 produced them.
STEP12B_FILES = {
    "comparison": "comparison_table_with_baseline4.csv",
    "comparison_marked": "marked_comparison_table_with_baseline4.csv",
    "per_class_full": "per_class_f1_full_with_baseline4.csv",
    "consistency": "consistency_checks_with_baseline4.csv",
    "summary": "summary_with_baseline4.md",
    "report": "step12_baseline4_report.csv",
    "chart_metrics": "comparison_metrics_with_baseline4.png",
    "chart_metrics_free": "comparison_metrics_duplicate_free_with_baseline4.png",
    "chart_per_class": "per_class_f1_heatmap_with_baseline4.png",
    "chart_time": "training_and_inference_time_with_baseline4.png",
    "chart_confusion": "confusion_matrices_with_baseline4.png",
}

# Inside step12\baseline4_ae_pca_lstm\
BASELINE4_FILES = {
    "model": "model.keras",
    "encoder": "encoder.keras",
    "ae_config": "ae_config.json",
    "pca": "pca.joblib",
    "pca_variance": "pca_variance.csv",
    "pca_scree": "pca_scree.png",
    "history": "history.csv",
    "metrics": "metrics.json",
    "per_class": "per_class_report.csv",
    "confusion_counts": "confusion_matrix_counts.csv",
    "confusion_rownorm": "confusion_matrix_rownorm.csv",
    "predictions": "predictions.csv",
}


# ---------------- Step 12 addendum - Baseline 5 ----------------
# APPENDED.  Nothing above this line was changed.
#
# Baseline 5: MI (top 30) -> AE (30 -> 15) -> hybrid CNN-LSTM.
#
# The proposed model selects its 30 features with three signals -- mutual
# information, a Pearson redundancy filter and permutation importance.
# Baseline 5 keeps everything else identical and selects with MI alone, so the
# difference between the two rows is what the Pearson filter and the
# permutation importance bought.  If the two feature lists come out identical
# the run STOPS and says so: there would be nothing to compare.
BASELINE5_NAME = "baseline5"
BASELINE5_SPEC = dict(name="baseline5", feature_source="mi_top30",
                      use_ae=True, use_pca=False, model_type="cnn_lstm")
BASELINE5_TOP_K = 30
BASELINE5_OUTPUT_DIR = STEP12_OUTPUT_DIR / "baseline5"

# False = a finished Baseline 5 is read back from disk instead of retrained.
OVERWRITE_BASELINE5 = False

# Baseline 5 is deliberately NOT added to EXPERIMENTS: that list is what
# experiments.run_step12() iterates and what Steps 13 and 14 address by name.
BASELINE5_FILES = {
    "mi_scores": "mi_scores.csv",
    "selected": "selected_features_mi30.csv",
    "overlap": "feature_overlap.csv",
    "encoder": "encoder.keras",
    "ae_config": "ae_config.json",
    "ae_history": "ae_history.csv",
    "model": "model.keras",
    "history": "history.csv",
    "metrics": "metrics.json",
    "per_class": "per_class_report.csv",
    "confusion_counts": "confusion_matrix_counts.csv",
    "confusion_rownorm": "confusion_matrix_rownorm.csv",
    "predictions": "predictions.csv",
}

# Written beside the Step 12 files, never over them.
STEP12C_FILES = {
    "comparison": "comparison_table_6configs.csv",
    "comparison_marked": "marked_comparison_table_6configs.csv",
    "per_class_full": "per_class_f1_full_6configs.csv",
    "per_class_free": "per_class_f1_duplicate_free_6configs.csv",
    "head_to_head": "proposed_vs_baseline5_6configs.csv",
    "per_class_diff": "proposed_vs_baseline5_per_class_6configs.csv",
    "consistency": "consistency_checks_6configs.csv",
    "summary": "summary_6configs.md",
    "report": "step12_baseline5_report_6configs.csv",
    "chart_metrics": "comparison_metrics_6configs.png",
    "chart_metrics_free": "comparison_metrics_duplicate_free_6configs.png",
    "chart_per_class": "per_class_f1_heatmap_6configs.png",
    "chart_confusion": "confusion_matrices_6configs.png",
    "chart_head_to_head": "proposed_vs_baseline5_per_class_6configs.png",
}


# ---------------- Step 12 integration - display order and full tables ------
# APPENDED for the Step 12 integration.  Nothing above this line was changed:
# EXPERIMENTS keeps its four entries and every INTERNAL key ("proposed",
# "baseline2_ae_lstm", "baseline3_cnn_lstm", ...) is untouched, because Step 13
# validates FUSION_MODELS against them.  Only DISPLAY labels are defined here.
#
# One order, used by every Step 12 table, chart, legend and confusion grid.
# Approach 2 is row 2 from the start: before it is trained its row reads
# "not yet trained - run 10.2" and is excluded from best-value marking, so the
# order never changes once it exists.
APPROACH2_KEY = "approach2_attention"

STEP12_DISPLAY_ORDER = [
    ("proposed", "Approach 1: FS -> AE -> CNN-LSTM"),
    (APPROACH2_KEY, "Approach 2: FS -> AE -> CNN-Attention-LSTM"),
    ("baseline1_ae_cnn_lstm", "B1: AE -> CNN-LSTM (no selection)"),
    ("baseline2_ae_lstm", "B2: AE -> LSTM (no CNN)"),
    ("baseline3_cnn_lstm", "B3: CNN-LSTM (no AE)"),
    ("baseline4_ae_pca_lstm", "B4: AE -> PCA -> LSTM"),
    ("baseline5", "B5: MI-only top 30 -> AE -> CNN-LSTM"),
]

# What each configuration isolates, for the 12.1 table.
STEP12_ISOLATES = {
    "proposed": "the proposed pipeline (reference)",
    APPROACH2_KEY: "what attention over the CNN feature map adds",
    "baseline1_ae_cnn_lstm": "what the feature selection buys",
    "baseline2_ae_lstm": "what the CNN front end buys",
    "baseline3_cnn_lstm": "what the autoencoder buys",
    "baseline4_ae_pca_lstm": "linear PCA in place of the CNN",
    "baseline5": "MI alone in place of the fused MI + Pearson + PI selection",
}

# The text shown in a row whose model has not been trained yet.
NOT_TRAINED_TEXT = "not yet trained - run 10.2"

# Written beside the existing Step 12 files, never over them.  (You asked for
# `per_class_f1_full.csv`; Step 12 already uses that name for its four-row
# table, so the all-configuration version is `per_class_f1_table_full.csv`.)
STEP12_FULL_FILES = {
    "comparison": "comparison_table_full.csv",
    "comparison_marked": "marked_comparison_table_full.csv",
    "per_class": "per_class_f1_table_full.csv",
    "per_class_free": "per_class_f1_table_duplicate_free_full.csv",
    "zero_recall": "zero_recall_diagnostic_full.csv",
    "head_to_head": "head_to_head_full.csv",
    "unchanged_check": "existing_rows_unchanged_full.csv",
    "summary": "summary_full.md",
    "report": "step12_full_report.csv",
    "chart_metrics": "comparison_metrics_full.png",
    "chart_metrics_free": "comparison_metrics_duplicate_free_full.png",
    "chart_per_class": "per_class_f1_heatmap_full.png",
    "chart_time": "training_and_inference_time_full.png",
    "chart_confusion": "confusion_matrices_full.png",
}

# The readable cross-check from the thesis write-up: every one of these rows
# must still produce these numbers after the integration (4 decimals, 5e-5).
# A key that is absent from the saved results is reported as skipped, never
# silently passed.
STEP12_EXPECTED_ROWS = {
    "proposed": {"accuracy": 0.9896, "f1_macro": 0.7170, "g_mean": 0.7505},
    "baseline1_ae_cnn_lstm": {"accuracy": 0.9859, "f1_macro": 0.6848,
                              "g_mean": 0.7388},
    "baseline2_ae_lstm": {"accuracy": 0.9580, "f1_macro": 0.5101,
                          "g_mean": 0.0000},
    "baseline3_cnn_lstm": {"accuracy": 0.9689, "f1_macro": 0.5249,
                           "g_mean": 0.0000},
    "baseline4_ae_pca_lstm": {"accuracy": 0.9851, "f1_macro": 0.5735,
                              "g_mean": 0.6384},
}
STEP12_EXPECTED_TOLERANCE = 5e-5      # on values rounded to 4 decimals
STEP12_EXACT_TOLERANCE = 1e-9         # saved metrics.json against rebuilt rows


# ---------------- Step 10 Approach 2 - CNN-Attention-LSTM ----------------
# APPENDED for Approach 2.  Nothing above this line was changed: Approach 1
# (Step 10 / Step 11) keeps its settings, its folders and its saved results.
#
# Approach 2 is Approach 1 with one block added -- additive attention over the
# LSTM output sequence -- and EVERY other hyperparameter reused from Approach 1
# (CNN_FILTERS, CNN_KERNEL, CNN_POOL, LSTM_UNITS, DROPOUT, DENSE_UNITS,
# CLF_EPOCHS, CLF_BATCH_SIZE, CLF_LR, CLF_PATIENCE, EARLY_STOP_METRIC,
# RANDOM_STATE, INPUT_MODE).  That is the whole point: a difference between the
# two rows has one cause.
STEP10B_OUTPUT_DIR = OUTPUTS_DIR / "step10_attention"
STEP11B_OUTPUT_DIR = OUTPUTS_DIR / "step11_attention"

# The width of the attention scoring layer: e_t = v^T tanh(W h_t + b), with
# W mapping LSTM_UNITS -> ATTN_UNITS.
ATTN_UNITS = 64

# False = a finished Approach 2 is loaded from disk instead of retrained.
OVERWRITE_STEP10B = False

STEP10B_FILES = {
    "model": "cnn_attention_lstm.keras",
    "history": "training_history.csv",
    "model_card": "model_card.json",
    "architecture": "architecture.csv",
    "loss_curve": "loss_curve.png",
    "macro_f1_curve": "val_macro_f1_curve.png",
    "report": "step10_attention_report.csv",
}

STEP11B_FILES = {
    "metrics": "metrics.json",
    "per_class": "per_class_report.csv",
    "confusion_counts": "confusion_matrix_counts.csv",
    "confusion_rownorm": "confusion_matrix_rownorm.csv",
    "predictions": "predictions.csv",
    "attention": "attention_weights_summary.csv",
    "attention_per_class": "attention_weights_per_class.csv",
    "selection": "approach_selection.json",
    "comparison": "approach1_vs_approach2.csv",
    "per_class_diff": "approach1_vs_approach2_per_class.csv",
    "report": "step11_attention_report.csv",
    "chart_attention": "attention_heatmap.png",
    "chart_comparison": "approach1_vs_approach2.png",
    "chart_confusion": "approach1_vs_approach2_confusion.png",
}

DIRS_TO_CREATE.extend([STEP10B_OUTPUT_DIR, STEP11B_OUTPUT_DIR])


# ---------------- Step 13 addendum - fusion across all models ----------------
# APPENDED.  Nothing above this line was changed.
#
# Step 13 fuses three models (FUSION_MODELS, unchanged above: Approach 1, B2,
# B3).  This addendum asks the next question without disturbing that result:
# what happens when EVERY approach and baseline that exists on disk is offered
# to the same adaptive fusion, and what happens when a subset is chosen on the
# validation split instead of all of them?
#
# Nothing here is retrained.  Every model is loaded from the folder its own step
# wrote, and a model that has not been trained is skipped by name rather than
# filled in.
STEP13ALL_OUTPUT_DIR = OUTPUTS_DIR / "step13_all"

# The configurations to offer the fusion, by their INTERNAL keys.  These are the
# keys of STEP12_DISPLAY_ORDER, named through the same constants the rest of the
# project uses so a rename cannot leave this list behind:
#   APPROACH2_KEY  == "approach2_attention"   (Step 10.2 / 11.3, not EXPERIMENTS)
#   BASELINE5_NAME == "baseline5"             (the Step 12 addendum)
# fusion_all.model_keys() verifies every entry against STEP12_DISPLAY_ORDER at
# run time rather than trusting this list.
FUSION_ALL_MODELS = [
    "proposed",
    APPROACH2_KEY,
    "baseline1_ae_cnn_lstm",
    "baseline2_ae_lstm",
    "baseline3_cnn_lstm",
    "baseline4_ae_pca_lstm",
    BASELINE5_NAME,
]

# How the FSEL subset is chosen.  "greedy_forward" starts from the best single
# model and adds the model that most improves the adaptive fusion's VALIDATION
# score; "none" skips the subset variant entirely.
SUBSET_SELECTION = "greedy_forward"     # or "none"
SUBSET_MAX_MODELS = 4
SUBSET_METRIC = "macro_f1"              # or "g_mean" -- a VALIDATION metric

# False = a finished addendum is read back from disk instead of recomputed.
OVERWRITE_STEP13ALL = False

# Every fusion setting is reused from the Step 13 block above -- WEIGHT_MODE,
# WEIGHT_TEMPERATURE, SHRINKAGE_K, CONFIDENCE_TERM, CONTEXT_MODE,
# RELIABILITY_METRIC, THRESHOLD_RULE, TARGET_FPR, EPSILON -- so FALL and FSEL
# differ from the Step 13 fusion only in which models they are given.

# Written in their own folder: no Step 13 file is touched.
STEP13ALL_FILES = {
    "availability": "model_availability.csv",
    "disagreement": "diversity_disagreement.csv",
    "risk_corr": "diversity_risk_corr.csv",
    "selection_path": "subset_selection_path.csv",
    "reliability": "reliability_table_all.csv",
    "thresholds": "thresholds_all.json",
    "fused_predictions": "fused_predictions_all.csv",
    "comparison": "comparison_table_all.csv",
    "comparison_marked": "marked_comparison_table_all.csv",
    "per_class": "per_class_f1_all.csv",
    "per_class_free": "per_class_f1_duplicate_free_all.csv",
    "weights_all": "mean_weight_per_class_all.csv",
    "weights_subset": "mean_weight_per_class_subset.csv",
    "metrics": "metrics_all.json",
    "summary": "summary_all.md",
    "report": "step13_all_report.csv",
    "chart_metrics": "comparison_bars_all.png",
    "chart_metrics_free": "comparison_bars_duplicate_free_all.png",
    "chart_weights_all": "mean_weight_per_class_all.png",
    "chart_weights_subset": "mean_weight_per_class_subset.png",
    "chart_diversity": "diversity_heatmap.png",
}

DIRS_TO_CREATE.append(STEP13ALL_OUTPUT_DIR)


# ---------------- Step 14 addendum - XAI across all models ----------------
# APPENDED.  Nothing above this line was changed.
#
# Step 14 explains the three-model fusion (FUSION_MODELS).  This addendum
# explains EVERY approach and baseline on its own named input features, puts
# their explanations side by side, and explains each of the three fusion
# variants F3 (Step 13), FALL and FSEL (13.8).
#
# Nothing is retrained and nothing in step14\ is touched: the models, encoders,
# the Baseline 4 PCA, the fusion weights and every threshold are loaded.
STEP14ALL_OUTPUT_DIR = OUTPUTS_DIR / "step14_all"

# The configurations to explain, by their internal keys -- the keys of
# STEP12_DISPLAY_ORDER, in that order.  Built from it rather than retyped, so
# the two lists cannot drift apart.
XAI_ALL_MODELS = [key for key, _label in STEP12_DISPLAY_ORDER]

# Which fusion variant the 14.9.5 alert reports explain.
#   "auto"  the variant with the best VALIDATION macro-F1 among those available
#   "F3"    the Step 13 three-model fusion
#   "FALL"  13.8's fusion over every available model
#   "FSEL"  13.8's fusion over the validation-chosen subset
ALERT_FUSION_VARIANT = "auto"

# One colour per model, in display order, used by every 14.9 chart so a model
# keeps its colour from one figure to the next.  The first five are the
# project's existing series hues; the last two extend them to seven.  Checked
# for colour-blind separation and the chroma floor against the chart surface
# (worst adjacent pair dE 9.1 protan); three sit below 3:1 contrast, so every
# chart that uses them also carries direct labels or a legend.
MODEL_COLOURS = dict(zip(XAI_ALL_MODELS, [
    "#2a78d6",   # Approach 1
    "#eb6834",   # Approach 2
    "#1baf7a",   # B1
    "#eda100",   # B2
    "#e87ba4",   # B3
    "#8a63d2",   # B4
    "#00a9b5",   # B5
]))

# How many features a cross-model top-K table and the overlap measure use.
XAI_TOP_FEATURES = 10

# False = a finished addendum is read back from disk instead of recomputed.
OVERWRITE_STEP14ALL = False

# Everything else is reused from the Step 14 block above, unchanged: IG_STEPS,
# IG_BASELINE, IG_BATCH_SIZE, EXPLAIN_SAMPLE_PER_CLASS, TOP_K_FEATURES,
# COMPLETENESS_TOL, STABILITY_NOISE_STD, STABILITY_N_REPEATS, PERTURB_MODE,
# SUSPICIOUS_BAND, SUSPICIOUS_RECALL_TARGET, CONF_HIGH_PROB, CONF_MEDIUM_PROB,
# SOURCE_CONTEXT, ATTACK_STAGE_MAP and ACTION_TEMPLATES.

STEP14ALL_FILES = {
    "composed_checks": "composed_model_checks.csv",
    "ig_baselines": "ig_baselines_all.csv",
    "explanations": "explanations_all.csv",
    "global_importance": "global_importance_all.csv",
    "quality": "explanation_quality_all.csv",
    "agreement": "model_explanation_agreement.csv",
    "consensus": "consensus_features_all.csv",
    "attention": "attention_summary_approach2.csv",
    "contributions": "fusion_contributions_all.csv",
    "variants": "fusion_variant_comparison.csv",
    "dominant": "dominant_model_per_variant.csv",
    "disagreement": "fusion_variant_disagreement.csv",
    "example_rows": "example_alert_report_rows.csv",
    "alerts": "alert_explanations_all.csv",
    "examples": "example_alert_reports_all.md",
    "metrics": "explanation_metrics_all.json",
    "verification": "verification_all.csv",
    "summary": "summary_all.md",
    "report": "step14_all_report.csv",
    "chart_global": "global_importance_per_model.png",
    "chart_feature_heatmap": "feature_by_model_heatmap.png",
    "chart_agreement": "model_agreement_heatmap.png",
    "chart_quality": "explanation_quality_bars.png",
    "chart_deletion": "deletion_check_per_model.png",
    "chart_dominant": "dominant_model_per_variant.png",
    "chart_attention": "attention_heatmap_approach2.png",
}

DIRS_TO_CREATE.append(STEP14ALL_OUTPUT_DIR)


# ---------------- Step 12 completion - B6, provenance, sanity ----------------
# APPENDED.  Nothing above this line was changed: EXPERIMENTS keeps its four
# entries, FUSION_MODELS still names the three models Step 13 fuses, and every
# existing key keeps its value.  The three lists below are EXTENDED in place
# rather than rewritten, so a reader can see exactly what was added.
#
# Baseline 6: AE -> PCA -> CNN-LSTM.  It differs from Baseline 4 only by the
# classifier (CNN-LSTM instead of LSTM) and from Baseline 1 only by the PCA
# stage, so the pair B4/B6 isolates the CNN front end on PCA features and the
# pair B1/B6 isolates PCA before an unchanged CNN-LSTM.
BASELINE6_NAME = "baseline6_ae_pca_cnn_lstm"
BASELINE6_LABEL = "B6: AE -> PCA -> CNN-LSTM"
STEP12_B6_OUTPUT_DIR = OUTPUTS_DIR / "step12_baseline6"

# "same_as_B4" reuses Baseline 4's SAVED PCA, unchanged and never refitted, so
# B4 and B6 sit in the identical feature space.  An integer instead fits a NEW
# PCA with that many components on the B4 encoder's TRAINING latents only,
# writes it inside the B6 folder, and the row label then reads "PCA(k)";
# Baseline 4's own pca.joblib is never touched either way.
# How many PCA components Baseline 6 uses.
#   "same_as_B4" -> Baseline 4's fitted pca.joblib, reused byte for byte
#   int          -> a NEW PCA with that many components, fitted on the
#                   TRAINING latents only, written inside the Baseline 6
#                   folder; Baseline 4's own pca.joblib is never touched, and
#                   the row label says PCA(k) so the two feature spaces cannot
#                   be confused.
# This is an int, and not "same_as_B4", for a reason worth keeping in the
# write-up: PCA_N_COMPONENTS = 0.95 is a VARIANCE target, and on these latents
# it selects 2 components.  `models.build_cnn_lstm` refuses an input with fewer
# steps than CNN_KERNEL (3), so B6 cannot be trained in Baseline 4's own
# 2-component space.  CNN_KERNEL is NOT lowered to fit: it is the kernel
# Approach 1, B1, B3 and B5 were trained with, and changing it would invalidate
# every one of those saved results.  3 is the smallest count the shared stack
# accepts; at that length the pooling layer is skipped, so the LSTM sees 3
# positions (12.6 prints this, and 12.11 carries the caveat).
B6_PCA_COMPONENTS = 3

BASELINE6_SPEC = {
    "name": BASELINE6_NAME,
    "label": BASELINE6_LABEL,
    "feature_source": BASELINE4_AE_FEATURES,   # the same inputs B4 starts from
    "use_ae": True,
    "use_pca": True,
    "model_type": "cnn_lstm",                  # the ONE difference from B4
    "reuse_encoder_from": "baseline4_ae_pca_lstm",
}

# False = a finished Baseline 6 is read back from disk instead of retrained.
OVERWRITE_BASELINE6 = False

BASELINE6_FILES = {
    "model": "model.keras",
    "pca": "pca.joblib",                       # only when B6_PCA_COMPONENTS is an int
    "pca_variance": "pca_variance.csv",
    "history": "history.csv",
    "loss_curve": "loss_curve.png",
    "macro_f1_curve": "val_macro_f1_curve.png",
    "model_card": "model_card.json",
    "metrics": "metrics.json",
    "per_class": "per_class_report.csv",
    "confusion_counts": "confusion_matrix_counts.csv",
    "confusion_rownorm": "confusion_matrix_rownorm.csv",
    "predictions": "predictions.csv",
    "protocol": "protocol_checks.csv",
    "pca_length": "pca_length_check.csv",
}

# --- the display order, extended ------------------------------------------
# Appended rather than rewritten, so the first seven rows keep the positions
# every saved Step 12, 13 and 14 file already uses.
STEP12_DISPLAY_ORDER = STEP12_DISPLAY_ORDER + [(BASELINE6_NAME, BASELINE6_LABEL)]
STEP12_ISOLATES[BASELINE6_NAME] = ("a CNN front end on the PCA features "
                                   "(B4 + CNN, B1 + PCA)")

# --- the Step 13.8 and 14.9 lists, extended -------------------------------
# FUSION_MODELS is NOT touched: Step 13 still fuses Approach 1, B2 and B3.
FUSION_ALL_MODELS = FUSION_ALL_MODELS + [BASELINE6_NAME]
XAI_ALL_MODELS = XAI_ALL_MODELS + [BASELINE6_NAME]
MODEL_COLOURS[BASELINE6_NAME] = "#a06a2c"      # the eighth hue: brown,
# checked with the other seven for the chroma floor and colour-blind
# separation against the chart surface (worst adjacent pair dE 9.1 protan)

# --- 12.0  the retraining lock --------------------------------------------
# True = a training call for a configuration whose metrics.json already exists
# is REFUSED, whatever its OVERWRITE_* flag says.  This exists because a model
# silently retrained after its metrics were published makes every table that
# quotes it wrong.  Set it to False deliberately when a retrain is intended.
LOCK_TRAINED_MODELS = True

STEP12_PROVENANCE_FILES = {
    "provenance": "provenance.csv",
    "consistency": "provenance_consistency.csv",
    "approach1": "approach1_consistency.csv",
    "report": "provenance_report.md",
}

# Where a backup of an earlier run would be, if one exists.  Any folder under
# DATA_ROOT whose name starts with this is read and REPORTED beside the current
# numbers; nothing is ever restored or chosen automatically.
BACKUP_PREFIX = "outputs_backup"

# --- 12.2  the sanity baselines -------------------------------------------
STEP12_SANITY_OUTPUT_DIR = OUTPUTS_DIR / "step12_sanity"
SANITY_FILES = {
    "table": "sanity_baselines.csv",
    "model": "label_shuffle_model.keras",
    "history": "label_shuffle_history.csv",
    "metrics": "label_shuffle_metrics.json",
    "predictions": "label_shuffle_predictions.csv",
    "permutation": "label_shuffle_permutation.csv",
    "report": "sanity_report.md",
}
# The label-shuffle model must not beat chance by more than this in macro-F1,
# or the pipeline is leaking the label somewhere.
SANITY_LEAKAGE_MARGIN = 0.10

# False = a finished label-shuffle model is read back from disk instead of
# retrained.  It matters more here than it looks: the sanity table's macro-F1
# is a number the write-up quotes, and a model retrained on every re-run gives
# a slightly different one each time.  Trained once, loaded afterwards -- the
# same convention as every other configuration.  Set it True deliberately to
# retrain the sanity model.
OVERWRITE_STEP12_SANITY = False

# --- 12.10  extra seeds ----------------------------------------------------
# Only used when N_SEEDS >= 3; seed 0's folders are never overwritten.
STEP12_SEEDS_OUTPUT_DIR = OUTPUTS_DIR / "step12_seeds"

# --- the thesis exports ----------------------------------------------------
STEP12_THESIS_DIR = STEP12_OUTPUT_DIR / "thesis"
THESIS_FIGURE_DPI = 300

DIRS_TO_CREATE.extend([STEP12_B6_OUTPUT_DIR, STEP12_SANITY_OUTPUT_DIR,
                       STEP12_THESIS_DIR])
