# Step 1 - PCAPNG to flow CSV (one CSV per capture)

All code lives in `C:\MSCAD2022\Notebook`; no data files here. Reads
`C:\MSCAD2022\PCAPNG_ORIGINAL` and `C:\MSCAD2022\CSV_ORIGINAL` **read-only**, and
writes only to `C:\MSCAD2022\PCAPS`, `C:\MSCAD2022\CSVs` and
`Notebook\outputs\step1`.

## How to run it

Copy this whole `Notebook` folder to `C:\MSCAD2022\Notebook`, then in **cmd**
(not PowerShell):

```
"C:\Users\banot\AppData\Local\Programs\Python\Python313\python.exe" -m pip install scapy numpy scipy pandas requests tqdm jupyter pytest
"C:\Users\banot\AppData\Local\Programs\Python\Python313\python.exe" -m pip install cicflowmeter==0.5.0 --no-deps --ignore-requires-python

cd /d C:\MSCAD2022\Notebook
"C:\Users\banot\AppData\Local\Programs\Python\Python313\python.exe" -m pytest tests -v
"C:\Users\banot\AppData\Local\Programs\Python\Python313\python.exe" -m jupyter notebook
```

Then open `convert_pcapng_to_csvflows.ipynb` and run cells 1-6.
Cell 1 is setup, cells 2-5 are the Stage A checks, cell 6 is the pcapng -> pcap
conversion. Cells 7-12 need the Stage C extractor; they print a clear
"STAGE C PENDING" message instead of raising.

There is also a non-Jupyter shortcut for the Stage A checks alone:

```
cd /d C:\MSCAD2022\Notebook
"C:\Users\banot\AppData\Local\Programs\Python\Python313\python.exe" stage_a_check.py
```

It writes `Notebook\stage_a_report.txt`, a copy of everything it printed.

## Stage status

| Stage | What it does | State |
|-------|--------------|-------|
| A | disk layout, old converter code, inventory, environment, Kaggle header | **ready** - notebook cells 2-5, or `stage_a_check.py` |
| B | pcapng -> pcap with `editcap` + packet-count verification + inventory CSV | **ready** - notebook cell 6 |
| C | the patched flow extractor (`converter.extract_flows`) + N-0 test run | **pending** - post-processing, labelling and tests are done; the flow table itself is not |
| D | all six captures + verification table | **pending** - needs Stage C |

## Files

| File | Purpose |
|------|---------|
| `convert_pcapng_to_csvflows.ipynb` | Step 1 notebook, 25 cells. Thin: every cell calls `src\converter.py`. |
| `MSCAD_Pipeline.ipynb` | **skeleton only** - title, setup cell, one empty header per Stage 2-8. No code. |
| `config.py` | every path and setting. The only file with hard-coded paths. |
| `stage_a_check.py` | 26-line wrapper that runs the Stage A checks from cmd. |
| `src\converter.py` | all Step 1 logic: discovery, editcap, post-processing, labelling. |
| `src\mscad\*.py` | Stage 2-8 stubs: a docstring and nothing else. |
| `tests\test_converter.py` | 49 pytest tests. No captures, no Wireshark, no scapy needed. |
| `requirements.txt` | Step 1 packages + the separate cicflowmeter line + later stages commented out. |
| `outputs\step1\` | pcap_inventory.csv (B), conversion_log.csv / label_report.csv / verification.csv (C/D) |

## What is NOT done here, on purpose

No merging of the CSVs, no cleaning (`inf` / `NaN` / duplicates are kept), no
encoding, scaling, splitting, feature selection, windowing or training. Those are
Step 2 onward, and `MSCAD_Pipeline.ipynb` stays a skeleton until then.
