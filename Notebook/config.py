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

# Declared above, before the Step 5 block existed.
DIRS_TO_CREATE.append(STEP5_OUTPUT_DIR)
