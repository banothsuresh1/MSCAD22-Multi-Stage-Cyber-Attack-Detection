# Step 1 - PCAPNG to flow CSV (one CSV per capture)

All code lives in `C:\MSCAD2022\Notebook`. No data files here.
Reads `C:\MSCAD2022\PCAPNG_ORIGINAL` (read-only) and writes only to
`C:\MSCAD2022\PCAPS`, `C:\MSCAD2022\CSVs` and `Notebook\outputs\step1`.

## Stage status

| Stage | What it does | State |
|-------|--------------|-------|
| A | check what exists: disk layout, old converter code, inventory, environment, Kaggle header | **`stage_a_check.py` ready - run it and paste the output back** |
| B | create the `Notebook\` structure + pcapng -> pcap + packet-count check | not written yet |
| C | `src\converter.py` (patched extractor) + `tests\test_converter.py` + N-0 test run | not written yet |
| D | full run of all six captures + verification report | not written yet |

## Run Stage A (cmd, NOT PowerShell)

```
cd /d C:\MSCAD2022\Notebook
"C:\Users\banot\AppData\Local\Programs\Python\Python313\python.exe" stage_a_check.py
```

Stage A is read-only: it converts nothing and modifies nothing. The only file it
writes is `Notebook\stage_a_report.txt`, a copy of everything it printed - attach
that file instead of pasting a long console dump if you prefer.

If pandas is not installed yet, section A.5 says so; install the packages with
the two commands the script prints, then run it again.

## Files so far

| File | Purpose |
|------|---------|
| `config.py` | every path and setting, as `pathlib.Path`. The only file with hard-coded paths. |
| `stage_a_check.py` | Stage A checks. |
| `requirements.txt` | Step 1 packages + the separate cicflowmeter line + later stages commented out. |

Created in Stage B/C: `convert_pcapng_to_csvflows.ipynb`, `MSCAD_Pipeline.ipynb`
(skeleton only), `src\converter.py`, `src\mscad\*.py` (docstring stubs),
`tests\test_converter.py`, `data\merged\`, `outputs\step1\`.
