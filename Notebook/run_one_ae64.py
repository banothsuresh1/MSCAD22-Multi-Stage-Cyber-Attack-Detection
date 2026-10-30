r"""run_one_ae64.py -- train (or load) one or more AE=64 baselines from a plain
terminal, outside Jupyter, so the heavy paper-LSTM runs get a fresh process and
all the free RAM on the machine instead of sharing the notebook kernel's memory.

Run it from the Notebook\ directory (where config.py and src\ live):

    cd C:\Users\<you>\MSCAD22-Multi-Stage-Cyber-Attack-Detection\Notebook
    python run_one_ae64.py baseline2_ae64
    python run_one_ae64.py baseline2_ae64 baseline4_ae64 baseline6_ae64
    python run_one_ae64.py all          # every AE=64 baseline, in order
    python run_one_ae64.py --list       # show each one's finished state, run nothing

What it does, and does NOT do:
  * It sets config.LOCK_TRAINED_MODELS_AE64 = False FOR THIS PROCESS ONLY, so a
    stray lock can't turn training into a silent "pending".  Your config.py file
    is not edited.
  * It calls each baseline's own run(), which LOADS a result that already exists
    and only TRAINS one that is missing -- it never retrains or overwrites a
    saved metrics.json, and it writes only under outputs_official_ae64\.
  * B5 is not an AE=64 baseline (skipped by design) and is not run.
  * The architecture is untouched: AE=64, PCA=30, LSTM 1024/512, batch 850,
    20 epochs -- exactly as defined in the modules.

Exit code is 0 only when every requested baseline ends finished on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import config` / `from src.mscad...` work no matter the current directory
# by putting this file's own folder (the Notebook\ root) first on the path.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config  # noqa: E402

from src.mscad.experiments_step12_ae64 import (  # noqa: E402
    CONFIG_MODULES, module_for, common_ae64 as c64)


def _headline(metrics: dict) -> str:
    """A short one-line summary of a finished result, or '' when there is none."""
    if not metrics:
        return ""
    full = metrics.get("full") or {}
    bits = []
    if metrics.get("training_seconds") is not None:
        bits.append("train %.1fs" % float(metrics["training_seconds"]))
    for key, label in (("accuracy", "acc"), ("f1_macro", "F1-macro"),
                       ("g_mean", "G-mean")):
        if full.get(key) is not None:
            bits.append("%s %.4f" % (label, float(full[key])))
    return "  (" + ", ".join(bits) + ")" if bits else ""


def _run_one(name: str) -> bool:
    """Run (or load) one baseline; return True iff it is finished on disk."""
    try:
        mod = module_for(name)
    except Exception as exc:                           # unknown name, etc.
        print("  [SKIP] %s: %s" % (name, exc))
        return False
    key = mod.NAME
    print("=" * 70)
    print("=== %s  (%s)" % (name, key))
    result = mod.run()                                 # verbose by default
    finished = c64.is_finished(key)
    print("  STATUS %s | FINISHED %s%s"
          % (result.get("status"), finished, _headline(result.get("metrics", {}))))
    if not finished:
        print("  detail: %s" % result.get("status_detail"))
    return bool(finished)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train or load AE=64 baselines from the terminal.")
    parser.add_argument("configs", nargs="*",
                        help="module names (e.g. baseline2_ae64), or 'all'")
    parser.add_argument("--list", action="store_true",
                        help="show each baseline's finished state and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("AE=64 baselines (LOCK_TRAINED_MODELS_AE64 = %s on disk):"
              % config.LOCK_TRAINED_MODELS_AE64)
        for name in CONFIG_MODULES:
            key = module_for(name).NAME
            print("  %-16s %s" % (name,
                  "saved" if c64.is_finished(key) else "not trained yet"))
        print("  baseline5_ae64   skipped by design (30-feature AE -> 64 expands)")
        return 0

    requested = list(CONFIG_MODULES) if args.configs == ["all"] else args.configs
    if not requested:
        parser.print_help()
        return 2

    unknown = [n for n in requested if n not in CONFIG_MODULES]
    if unknown:
        print("Unknown config(s): %s" % ", ".join(unknown))
        print("Valid: %s, or 'all'." % ", ".join(CONFIG_MODULES))
        print("(baseline5_ae64 is skipped by design and cannot be run.)")
        return 2

    # This process only -- the file on disk is NOT changed.
    config.LOCK_TRAINED_MODELS_AE64 = False
    print("LOCK_TRAINED_MODELS_AE64 = False (this process only; config.py unchanged)")

    done = [_run_one(name) for name in requested]
    print("=" * 70)
    ok = all(done)
    print("Finished %d of %d requested baseline(s)." % (sum(done), len(done)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
