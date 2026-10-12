# Step 1 - PCAPNG to flow CSV (one CSV per capture)

Tool folder. Code only - no data lives here.
Reads `C:\MSCAD2022\PCAPNG_ORIGINAL` (read-only) and writes to
`C:\MSCAD2022\PCAPS`, `C:\MSCAD2022\CSVs` and `logs\`.

## Status

| Stage | What it does | State |
|-------|--------------|-------|
| A | environment check + capture inventory + Kaggle header check | `stage_a_check.py` ready - **run it and paste the output back** |
| B | pcapng -> pcap with editcap + packet-count verification | not written yet |
| C | `src\converter.py` + N-0 test conversion | not written yet |
| D | full run (6 captures) + verification report | not written yet |

## Run Stage A (cmd, NOT PowerShell)

```
cd /d C:\MSCAD2022\PCAPng_to_Flow
"C:\Users\banot\AppData\Local\Programs\Python\Python313\python.exe" stage_a_check.py
```

It creates `PCAPS\`, `CSVs\`, `CSVs\raw\`, `logs\` and writes
`logs\pcap_inventory.csv`. It converts nothing and modifies nothing.

If pandas is not installed yet, section A.4 will say so - install the packages
with the two commands the script prints, then run it again.

## Files

| File | Purpose |
|------|---------|
| `config.py` | every path, constant, column name and label. The only file to edit per machine. |
| `stage_a_check.py` | Stage A checks (read-only). |
| `requirements.txt` | packages + the two-step cicflowmeter install. |
| `src\converter.py` | Stage C: all conversion logic (notebook only calls it). |
| `tests\test_converter.py` | Stage C: pytest, needs no pcap files. |
| `logs\` | pcap_inventory.csv, conversion_log.csv, label_report.csv, verification.csv |
