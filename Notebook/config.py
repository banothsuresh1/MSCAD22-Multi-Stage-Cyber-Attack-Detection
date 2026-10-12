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
from pathlib import Path

# ===========================================================================
# 1. FOLDERS
# ===========================================================================
# DATA_ROOT holds the data; PROJECT_ROOT holds all code and notebooks.
# The two environment variables are ONLY a unit-test hook, so the test suite
# can run on a machine without a C:\ drive.  On your PC they are not set and
# the literal paths below are used.
DATA_ROOT = Path(os.environ.get("MSCAD_DATA_ROOT", r"C:\MSCAD2022"))
PROJECT_ROOT = Path(os.environ.get("MSCAD_PROJECT_ROOT", str(DATA_ROOT / "Notebook")))

# INPUT -- read-only.  Never modified, moved, renamed or deleted by this project.
PCAPNG_DIR = DATA_ROOT / "PCAPNG_ORIGINAL"

# OUTPUT of Stage B: the .pcap files produced by editcap.
PCAP_DIR = DATA_ROOT / "PCAPS"

# OUTPUT of Stage C/D: one flow CSV per capture, <stem>_flows.csv.
CSV_OUTPUT_DIR = DATA_ROOT / "CSVs"
RAW_DIR = CSV_OUTPUT_DIR / "raw"          # untouched extractor output

# REFERENCE ONLY -- the Kaggle MSCAD file.  Read, never written.
KAGGLE_DIR = DATA_ROOT / "CSV_ORIGINAL"

# Code-side folders.
MERGED_DIR = PROJECT_ROOT / "data" / "merged"      # stays empty in Step 1
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step1"    # logs and reports
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# Report files written by Step 1 (all inside OUTPUT_DIR).
PCAP_INVENTORY_CSV = OUTPUT_DIR / "pcap_inventory.csv"   # Stage B
CONVERSION_LOG_CSV = OUTPUT_DIR / "conversion_log.csv"   # Stage D
LABEL_REPORT_CSV = OUTPUT_DIR / "label_report.csv"       # Stage C/D
VERIFICATION_CSV = OUTPUT_DIR / "verification.csv"       # Stage D

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
]

# Folders that must never be written to.
READ_ONLY_DIRS = [PCAPNG_DIR, KAGGLE_DIR]

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
        "MERGED_DIR": MERGED_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
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
