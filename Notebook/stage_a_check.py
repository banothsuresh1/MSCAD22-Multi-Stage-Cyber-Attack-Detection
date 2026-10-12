"""
stage_a_check.py -- run every Stage A check from cmd, without Jupyter.

The notebook does the same thing in cells 2-5; this script is just a shortcut
for when you do not want to start Jupyter.  All the logic lives in
src/converter.py, so the two can never disagree.

    cd /d C:\\MSCAD2022\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" stage_a_check.py

It converts nothing and modifies nothing.  The only file it writes is
Notebook\\stage_a_report.txt -- a copy of everything it printed, so you can
attach the file instead of pasting a long console dump.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src import converter  # noqa: E402

if __name__ == "__main__":
    converter.stage_a_report(echo=True, save_to=HERE / "stage_a_report.txt")
