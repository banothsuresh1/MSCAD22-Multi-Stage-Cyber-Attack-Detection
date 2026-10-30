# Migration plan — folder renumber (step3…step14 → step1…step12) + remove `Orig_Row`

**Status: PLAN ONLY. Nothing has been edited, renamed, regenerated, or trained.**
Awaiting explicit go-ahead, plus decisions on the flagged open questions below.

Decisions already locked by you:
- **Orig_Row:** remove completely, everywhere; no replacement artificial id; use
  *verified positional alignment* where rows cross files/models.
- **Rename:** full renumber step3…step14 → step1…step12 (folders only; logic,
  seeds, splits, AE/PCA dims unchanged).
- **Cost:** accepted — all outputs regenerate, all models retrain (GPU/Colab for LSTMs).

---

## 0. Confirmations from the code

- **No merging exists.** `official_dataset.build_merged` contains no `concat`/`merge`
  — it assembles the single source frame, (currently) adds `Orig_Row`, reorders, and
  writes `MSCAD_merged.csv`. So `MSCAD_merged.csv` is the **single intermediate file**,
  not a combination of two files. Removing `Orig_Row` from Step 1 removes it from this
  file. (You can optionally also rename it to `MSCAD_Dataset.csv` — see A-4; that was an
  earlier ask, not in your last message, so I've flagged it rather than assumed it.)
- **`run_stepN()` entry points exist** exactly as you listed (run_step3 → official_dataset,
  run_step4 → preprocessing, … run_step14 → explainability; plus run_step10b/run_step11b
  attention, run_step13all/run_step14all).

---

## ⚠️ OPEN DECISIONS — I need these before writing diffs (not resolving on my own)

**D1 — the existing `step2` folder (collision).** There is already a
`STEP2_OUTPUT_DIR = outputs_official/step2` ("acquisition report", written via
`data_loading.py:447`), and there is **no `step1`** today. Current folders are
`step2, step3, step4, …, step14`. Your mapping starts at `step3→step1` and does not
mention the existing `step2`. So `step4→step2` **collides** with the current `step2`.
Options: (a) the current `step2` is the own-extraction acquisition report and is unused
in official mode → leave it / retire it; (b) fold it into `step1`; (c) give it its own
new name. **Tell me which.**

**D2 — derived/suffixed folders.** Several folders encode the old step number and must
renumber consistently (your explicit list only covered step3–step14):
`step10_attention→step8_attention`, `step11_attention→step9_attention`,
`step12_baseline6→step10_baseline6`, `step12/thesis→step10/thesis`,
`step13_all→step11_all`, `step14_all→step12_all`, `step12_sanity→step10_sanity`,
`step12_seeds→step10_seeds`. **Confirm these follow the same −2 shift.**

**D3 — function naming (Option A vs B).** See §3. *Do not resolve — your call.*

**D4 — config constant naming.** Parallel to D3: keep the constant *names*
(`STEP3_OUTPUT_DIR`) and change only their *values* (→ `step1`), or rename the constants
(`STEP3_OUTPUT_DIR`→`STEP1_OUTPUT_DIR`) and update every reference. Keeping names is far
lower risk; renaming is cleaner but cascades like Option B.

**D5 — `MSCAD_merged.csv` rename** to `MSCAD_Dataset.csv`? (earlier ask; confirm yes/no.)

---

## 1. Recommended order

1. **Workstream A (folder renumber) first** — mechanical, mostly `config.py` *values*,
   lowest risk.
2. **Workstream B (Orig_Row removal) second** — this is where "bugs will hide"
   (fusion row-alignment, provenance checks), and it edits **frozen** modules.
3. **Validate the test suite** after A, then after B (code-level; data-less tests).
4. **Regenerate** outputs step1…step12, then **retrain** all models (GPU/Colab for LSTMs).

Rationale: do *all code* first and validate, so you only pay the multi-day
regenerate+retrain **once**, at the very end, against final code.

---

## 2. WORKSTREAM A — folder renumber (diffs-to-be)

### A-1 `config.py` — change the folder *values* (constant names kept; see D4)

| line | constant | old value | new value |
|---|---|---|---|
| 150 | `STEP3_OUTPUT_DIR` | `OUTPUTS_DIR/"step3"` | `/"step1"` |
| 151 | `STEP4_OUTPUT_DIR` | `"step4"` | `"step2"` |
| 805 | `STEP5_OUTPUT_DIR` | `"step5"` | `"step3"` |
| 872 | `STEP6_OUTPUT_DIR` | `"step6"` | `"step4"` |
| 948 | `STEP7_OUTPUT_DIR` | `"step7"` | `"step5"` |
| 949 | `STEP8_OUTPUT_DIR` | `"step8"` | `"step6"` |
| 1024 | `STEP9_OUTPUT_DIR` | `"step9"` | `"step7"` |
| 1072 | `STEP10_OUTPUT_DIR` | `"step10"` | `"step8"` |
| 1073 | `STEP11_OUTPUT_DIR` | `"step11"` | `"step9"` |
| 1172 | `STEP12_OUTPUT_DIR` | `"step12"` | `"step10"` |
| 1227 | `STEP13_OUTPUT_DIR` | `"step13"` | `"step11"` |
| 1301 | `STEP14_OUTPUT_DIR` | `"step14"` | `"step12"` |
| 1639 | `STEP10B_OUTPUT_DIR` | `"step10_attention"` | `"step8_attention"` |
| 1640 | `STEP11B_OUTPUT_DIR` | `"step11_attention"` | `"step9_attention"` |
| 1691 | `STEP13ALL_OUTPUT_DIR` | `"step13_all"` | `"step11_all"` |
| 1763 | `STEP14ALL_OUTPUT_DIR` | `"step14_all"` | `"step12_all"` |
| 1849 | `STEP12_B6_OUTPUT_DIR` | `"step12_baseline6"` | `"step10_baseline6"` |
| 1941 | `STEP12_SANITY_OUTPUT_DIR` | `"step12_sanity"` | `"step10_sanity"` |
| 1965 | `STEP12_SEEDS_OUTPUT_DIR` | `"step12_seeds"` | `"step10_seeds"` |

Auto-follow (no edit needed, they derive from the above): `STEP12_THESIS_DIR`,
`STEP12_CONFIG_DIR`, `B6_OUTPUT_DIR`. The trailing `# C:\MSCAD2022\outputs\stepN`
comments also get updated for truthfulness.
`STEP2_OUTPUT_DIR` (line 149): **pending D1.**

### A-2 `test_official_step_numbering.py` — `STEP_FOLDERS`
Maps notebook step N → the constant whose folder name it checks. The folder *names*
change, so this mapping's expected names change; also `NEW_FOLDERS` strings and the
frozen-module path checks reference `src/mscad/…` (unaffected) but the step-folder
assertions update to the new names.

### A-3 AE30/AE64/paper redirects
- `experiments_step12_ae30/common_ae30._redirected` builds `place / ("step%d" % n)`
  from `config.AE30_REDIRECTED_STEPS` — those are **hardcoded old numbers**; they shift −2.
- `experiments_step12_ae64/common_ae64._table_redirect` uses the **constants**
  (`STEP12_OUTPUT_DIR`, `STEP12_B6_OUTPUT_DIR`) → auto-follows if constant names kept (D4=keep).
- paper-replication redirect similarly via constants.

### A-4 notebook
- Top-of-notebook note that maps **Step N → `stepM\`** (checked by
  `test_the_note_maps_every_new_step_number_to_its_unchanged_folder`) → rewrite to the new
  names; update that test's expectation.
- Any code cell printing/const-referencing a literal `"stepN"` path → update.
- Refreshing the notebook ⇒ refresh `tests/data/notebook_code_cells.json` and
  `notebook_cell_identity.json` **only if code cells change** (markdown-only note change
  does not).

### A-5 grep sweep
A final sweep for hardcoded `"step3".."step14"` strings (docstrings, comments, `README`,
any `MIGRATION`/report files) to update for truthfulness.

**Blast radius of A (with D4=keep names): low** — mostly `config.py` values + one test +
the notebook note + the AE30 hardcoded step numbers. All `config.STEPx_OUTPUT_DIR`
*references* keep working because only the values move.

---

## 3. Function naming — Option A vs Option B (D3, your decision)

**Option A — keep `run_step3()`…`run_step14()`; change only folders/config.**
- *Changes:* none to function names, callers, imports, notebook run cells, or tests.
- *Pro:* smallest diff, lowest risk, no notebook-snapshot churn from code cells.
- *Con:* the names now mismatch the new numbers (e.g. `run_step3` = data acquisition =
  "Step 1"). It moves the step-number confusion from folders into function names.

**Option B — rename `run_step3→run_step1` … `run_step14→run_step12`** (+ `run_step10b→
run_step8b`, `run_step11b→run_step9b`, `run_step13all→run_step11all`,
`run_step14all→run_step12all`).
- *Changes:* every definition, every caller across `src/mscad/`, **every notebook code
  cell that calls them**, and tests referencing them → then refresh the notebook snapshot
  + identity fixtures.
- *Pro:* fully consistent naming end-to-end.
- *Con:* large, touches many call sites + notebook code cells (snapshot refresh), higher
  chance of a missed caller → runtime error. Best done as its own commit after A+B are green.

Same tradeoff applies to **D4** (config constant names). My suggestion: **A + keep
constant names** for the first pass (get folders right with minimal risk), then decide on
B as a separate, isolated rename if you want the cosmetics.

---

## 4. WORKSTREAM B — remove `Orig_Row` (diffs-to-be, file by file)

> ⚠️ **7 of these are FROZEN modules** in `test_official_step_numbering.py`
> (`splitting, feature_selection, evaluation, fusion, fusion_all, explainability,
> explainability_all`). Editing them **breaks the byte-identity frozen test**, so that
> test must be updated too (move these out of `FROZEN_MODULES` into the named
> "changed with reason" set) — a visible, deliberate change, not silent.

| file | step | what to change | risk |
|---|---|---|---|
| `config.py` | — | `METADATA_COLUMNS` (line 689 / rebuilt at 2139): drop `'Orig_Row'`. Remove `ROW_COLUMN`-related use. | low |
| `official_dataset.py` | 1 | `build_merged`: delete `out[ROW_COLUMN]=np.arange(...)`, the "already has Orig_Row" guard, and `Orig_Row` from the `ordered` column list; drop `ROW_COLUMN="Orig_Row"`. `build_report`: remove the "Orig_Row is metadata" check. | med (frozen? no — it's a Step 1/2 module, editable) |
| `preprocessing.py` | 2 | `split_metadata` keys off `METADATA_COLUMNS`; once `Orig_Row` is gone from both config and the input, no code path needs it. Verify metadata.csv no longer emits it. | low |
| `splitting.py` 🧊 | 3 | Remove the `Orig_Row` split-position verification (≈ lines 802-808); **replace with explicit positional-alignment checks** (train/test index arrays map to source rows by position, verified by length + order). | **high** |
| `feature_selection.py` 🧊 | 4 | Remove `"Orig_Row"` from the exclude list (line 144). | trivial |
| `evaluation.py` 🧊 | 9 | `build_predictions_frame` (614-619): drop the `Orig_Row` branch; key `predictions.csv` on **verified test-set position** (not a new id). Update the duplicate-free/`meta_test` plumbing that assumed `Orig_Row`. | **high** |
| `fusion.py` 🧊 | 11 | `align_frames` (474-500) requires `Orig_Row` → **rewrite to align models by position**, with an explicit check that every per-model frame has identical length and row order. Lines 1125-1129, 1628: build the fused frame on position. | **highest** |
| `fusion_all.py` 🧊 | 11 | same rewrite as `fusion.py`. | high |
| `step12_checks.py` | 10.0 | Provenance check merges saved predictions on `Orig_Row` (234-316) → rewrite to compare by position, or drop the check (your call). | med |
| `explainability.py` 🧊 | 12 | 21 references — per-row attribution mapping on `Orig_Row` → switch to positional. | high |
| `explainability_all.py` 🧊 | 12 | 10 references — same. | high |
| `data_loading.py` | — | Official load: remove `Orig_Row` dtype/handling (line 909 etc.). The own-extraction `Orig_Row` (`merge_flow_csvs`, 427/605) is a *separate* path; since "no merging exists," confirm whether to touch it (D1-adjacent). | med |

🧊 = currently in `FROZEN_MODULES`.

**What "verified positional alignment" means in the diffs:** wherever two files or two
model frames are combined, add an explicit assertion that they have the **same row count
and same order** (e.g. identical length, and — where a stable sort key exists that is NOT
an added id — compare on that), and fail loudly otherwise. No new index/id column is
introduced.

**Tests touched by B:** `test_official_step_numbering.py` (frozen set), plus any test
that constructs `meta_*`/`predictions` with `Orig_Row` or asserts fusion alignment
(`test_fusion*`, `test_evaluation*`, `test_splitting*`, `test_step12_checks*`,
`test_comparison_approach1_row` uses `meta`), and the AE sensitivity tests
(`test_ae64_sensitivity`, `test_ae30_sensitivity`) which build synthetic `meta_test` with
`Orig_Row`. These get updated alongside the code.

---

## 5. What becomes invalid (regenerate / retrain)

**Everything. Nothing survives unchanged**, for two independent reasons (folders move
*and* the dataset schema loses a column):

- **All step outputs** step1…step12 (new numbering) — regenerate in order.
- **All trained models** — retrain: Approach 1, Approach 2, B1, B3, B6 (done), B2, B4
  (in progress). Their `metrics.json`/`model.keras`/`predictions.csv` all assume
  `Orig_Row` and the old paths.
- **All sensitivity roots** — `outputs_official_ae30/`, `outputs_official_ae64/`,
  `outputs_official_paper_replication/` — stale (they read the official step dirs and
  evaluate via `meta_test`/`Orig_Row`).
- **Nothing can be reused** — not even the raw `MSCAD.csv` output, because Step 1 itself
  changes (drops `Orig_Row`) and its folder moves to `step1`.

---

## 6. Validation gates (before regenerate/retrain)

1. After A: `pytest tests -q` green (data-less tests; folder-name + note + STEP_FOLDERS).
2. After B: `pytest tests -q` green (fusion/eval/split/explain + frozen-set update + AE tests).
3. A synthetic end-to-end dry run (no real data) proving positional alignment holds
   through split → eval → fusion, as I've done for prior changes.
4. Only then: regenerate step1…step12 and retrain (GPU/Colab for LSTMs).

---

## 7. Summary of what I need from you

Answer **D1** (existing `step2`), **D2** (derived folders −2), **D3** (function names A/B),
**D4** (config constant names keep/rename), **D5** (`MSCAD_merged.csv`→`MSCAD_Dataset.csv`?).
Then I'll produce the literal diffs for review (still nothing applied) and, on your
"go", execute Workstream A, validate, Workstream B, validate — and hand you the
regenerate/retrain runbook.
