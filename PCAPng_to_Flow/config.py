"""
config.py -- every path, constant and name used by the PCAPNG -> PCAP -> flow-CSV tool.

This is the ONLY file you should ever need to edit when you move the project to a
different machine.  Nothing in here does any work; it just stores settings.

Beginner note on Windows paths
------------------------------
Windows paths contain backslashes (\\).  In Python a backslash starts an "escape
sequence" (\\n = newline), so we always write Windows paths as *raw strings*:
    r"C:\\MSCAD2022"          <- the r in front means "take it literally"
Paths may also contain SPACES (e.g. "C:\\Program Files\\Wireshark").  That is fine
as long as we never build a shell command string by hand.  Everywhere in this
project we call external programs with
    subprocess.run([exe, "-F", "pcap", infile, outfile])     # a LIST of arguments
and never with shell=True.  With a list, Python hands each argument to Windows
separately, so spaces can never split an argument in two.
"""

import os

# ---------------------------------------------------------------------------
# 1. FOLDERS
# ---------------------------------------------------------------------------
# BASE_DIR is the project root on your machine.  The environment variable
# MSCAD_BASE_DIR is only a testing hook (it lets the code be unit-tested on a
# machine that has no C:\ drive).  On your Windows PC it is not set, so the
# literal path below is used.
BASE_DIR = os.environ.get("MSCAD_BASE_DIR", r"C:\MSCAD2022")

# INPUT  -- read-only.  The tool never writes, moves or deletes anything here.
PCAPNG_DIR = os.path.join(BASE_DIR, "PCAPNG_ORIGINAL")

# OUTPUT of Stage B: the .pcap files produced by editcap.
PCAP_DIR = os.path.join(BASE_DIR, "PCAPS")

# OUTPUT of Stage C/D: one flow CSV per capture (<stem>_flows.csv).
CSV_OUTPUT_DIR = os.path.join(BASE_DIR, "CSVs")

# Untouched cicflowmeter output, before renaming/reordering (<stem>_raw.csv).
RAW_DIR = os.path.join(CSV_OUTPUT_DIR, "raw")

# Logs and reports written by this tool.
LOG_DIR = os.path.join(BASE_DIR, "PCAPng_to_Flow", "logs")

# REFERENCE ONLY -- the Kaggle MSCAD file.  Read-only; never modified.
# Set to the folder; find_kaggle_file() below locates the actual .csv/.xlsx.
CSV_ORIGINAL_DIR = os.path.join(BASE_DIR, "CSV_ORIGINAL")
KAGGLE_CSV_PATH = os.path.join(CSV_ORIGINAL_DIR, "MSCAD.csv")

# Log / report file names (full paths).
PCAP_INVENTORY_CSV = os.path.join(LOG_DIR, "pcap_inventory.csv")
CONVERSION_LOG_CSV = os.path.join(LOG_DIR, "conversion_log.csv")
LABEL_REPORT_CSV = os.path.join(LOG_DIR, "label_report.csv")
VERIFICATION_CSV = os.path.join(LOG_DIR, "verification.csv")

# Folders this tool is allowed to create / write into.
WRITABLE_DIRS = [PCAP_DIR, CSV_OUTPUT_DIR, RAW_DIR, LOG_DIR]

# ---------------------------------------------------------------------------
# 2. EXTERNAL PROGRAMS
# ---------------------------------------------------------------------------
# Wireshark ships editcap (format conversion) and tshark (packet counting).
# Both paths contain a space -- see the note at the top of this file.
EDITCAP_EXE = os.environ.get("EDITCAP_EXE", r"C:\Program Files\Wireshark\editcap.exe")
TSHARK_EXE = os.environ.get("TSHARK_EXE", r"C:\Program Files\Wireshark\tshark.exe")

# The Python interpreter that has cicflowmeter / scapy installed.
PYTHON_EXE = os.environ.get(
    "PYTHON_EXE",
    r"C:\Users\banot\AppData\Local\Programs\Python\Python313\python.exe",
)

# ---------------------------------------------------------------------------
# 3. FLOW-EXTRACTION PARAMETERS
# ---------------------------------------------------------------------------
# A flow is closed after this many seconds, even if packets keep arriving.
# 120 s is what the Java CICFlowMeter used to build the Kaggle MSCAD file.
FLOW_TIMEOUT_S = 120

# A flow is closed after this much silence (used for the Idle/Active features).
IDLE_TIMEOUT_S = 240

# If an output file already exists: False = skip it, True = rebuild it.
OVERWRITE = False

# Print a progress line every N packets (big captures are ~650k packets).
PROGRESS_EVERY = 100_000

# ---------------------------------------------------------------------------
# 4. COLUMN NAMES
# ---------------------------------------------------------------------------
# The 6 flow identifiers that the Kaggle MSCAD.csv does NOT have, and which are
# the whole reason we regenerate the flows ourselves.
ID_COLUMNS = [
    "Src IP",
    "Dst IP",
    "Src Port",
    "Dst Port",
    "Protocol",
    "Timestamp",
]

# The 66 CICFlowMeter feature names, in the order written down by hand from the
# MSCAD paper / earlier inspection.  THIS IS ONLY A REFERENCE.
# Stage A reads the real header out of the Kaggle file and prints every
# difference; the Kaggle header always wins and replaces this list.
# Note: "CWE Flag Count" is CICFlowMeter's own (mis)spelling -- keep it exactly.
KAGGLE_FEATURES_REFERENCE = [
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

# The exact header, in the exact order, that every output CSV must have.
# 6 identifiers + 66 features + Label = 73 columns.
# After Stage A confirms the Kaggle header, replace KAGGLE_FEATURES_REFERENCE
# above (if it differs) and this list follows automatically.
FLOW_COLUMNS = ID_COLUMNS + KAGGLE_FEATURES_REFERENCE + [LABEL_COLUMN]

# ---------------------------------------------------------------------------
# 5. LABELS  (official MSCAD spellings -- do not "fix" them)
# ---------------------------------------------------------------------------
# Key   = the capture file stem, exactly as the .pcapng is named.
# Value = the attack label given to flows that involve an attacker IP.
#         Flows in an attack capture that do NOT involve an attacker IP are
#         labelled BENIGN_LABEL instead.  N-0 is pure normal traffic, so every
#         flow there gets "Normal".
CAPTURE_LABELS = {
    "N-0": "Normal",          # normal traffic only
    "Scan-1": "Port_Scan",    # full / SYN / FIN / UDP port scans
    "App-01": "DoS slowloris",  # HTTP Slowloris, several attacker IPs
    "App-02": "ICMP_Flood",   # ICMP flood + IP-fragmented large pings
    "W-B-01": "Web_Crwling",  # web crawling (the dataset really spells it "Crwling")
    "W-B-02": "Brute_Force",  # password cracking
}

# Captures where every single flow is normal (no attacker-IP logic needed).
NORMAL_CAPTURES = ["N-0"]

# Label for flows in an attack capture that do not involve an attacker IP.
BENIGN_LABEL = "Benign"

# Attacker detection: a Src IP is an attacker if its TOTAL forward packets are
# at least ATTACKER_THRESHOLD x the top sender's total forward packets.
ATTACKER_THRESHOLD = 0.5

# Attacker IPs found in earlier runs.  VERIFY these from the data every run --
# they are written here only so you can compare, never to label from.
EXPECTED_ATTACKERS_REFERENCE = {
    "Scan-1": ["192.168.159.1"],
    "W-B-01": ["192.168.159.1"],
    "App-01": ["192.168.159.10", "192.168.159.11", "192.168.159.12"],
    "App-02": ["192.168.159.14", "192.168.159.10", "192.168.159.12"],
}

# ---------------------------------------------------------------------------
# 6. SMALL HELPERS (no heavy imports, safe to use anywhere)
# ---------------------------------------------------------------------------


def ensure_output_dirs():
    """Create the output folders if they do not exist yet.  Never deletes."""
    created = []
    for d in WRITABLE_DIRS:
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            created.append(d)
    return created


def find_kaggle_file():
    """Return the path of the Kaggle MSCAD file inside CSV_ORIGINAL, or None.

    Prefers an exact 'MSCAD.csv'; otherwise takes the first .csv, then .xlsx.
    """
    if not os.path.isdir(CSV_ORIGINAL_DIR):
        return None
    names = sorted(os.listdir(CSV_ORIGINAL_DIR))
    for n in names:                                  # exact match first
        if n.lower() == "mscad.csv":
            return os.path.join(CSV_ORIGINAL_DIR, n)
    for ext in (".csv", ".xlsx", ".xls"):            # then any spreadsheet
        for n in names:
            if n.lower().endswith(ext):
                return os.path.join(CSV_ORIGINAL_DIR, n)
    return None
