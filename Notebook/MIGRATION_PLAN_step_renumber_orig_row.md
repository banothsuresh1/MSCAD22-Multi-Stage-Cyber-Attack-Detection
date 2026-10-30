# Migration plan v2 — folder+constant renumber (−2) and full `Orig_Row` removal

**Status: PLANNING ONLY. Nothing edited, renamed, deleted, regenerated, or trained.**
Push this document only; wait for separate explicit approval before implementation.

## Locked decisions (your confirmations)
- **D1** — keep the existing `step2`; **move its acquisition-report file(s) into the new
  `step1`** after a filename-conflict check; update `data_loading.py` + dependents. New
  `step2` = preprocessing (old `step4`).
- **D2** — apply the **−2 renumber to every** step folder (derived, nested, suffixed,
  experiment-specific); repo-wide, every occurrence.
- **D3** — **Option A**: keep all `run_stepN()` function names unchanged (folders/paths only).
- **D4** — **rename the constants** to the new numbers (`STEP3_OUTPUT_DIR`→`STEP1_OUTPUT_DIR`
  … `STEP14_*`→`STEP12_*`), consistently.
- **D5** — keep the filename `MSCAD_merged.csv`; confirmed **single-source intermediate**
  (`build_merged` has no `concat`/`merge`); strip `Orig_Row` from it and everything that
  creates/reads/depends on it. No merging, no replacement id.
- Complete `Orig_Row` removal; **no replacement ids / hidden index columns**; verified
  positional alignment instead.
- Update `FROZEN_MODULES` + affected tests with explicit reasons.
- Preserve all architectures, AE/PCA dims, seeds, split settings, methodology.

## Magnitude (measured, so the risk is explicit)
- **~1,400 references** to **80 distinct** `STEP<n>…` constant tokens across
  `src/mscad/` + `tests/` + `config.py`.
- **~96** hardcoded `"stepN"` path strings in `.py`; **many** in the notebook
  (e.g. `step6`×38, `STEP14ALL_FILES`×16 — note `step138`/`step149` are **cell ids**, not folders).
- Report filenames encode step numbers (`step4_report.csv`, `step10_attention_report.csv`,
  `step3_official_report.csv`, …).
This is a very large, mechanical-but-collision-prone change. See hazards below.

---

## ⚠️ HAZARDS to engineer around (not blockers, but must be done right)

**H1 — rename collision (the big one).** The −2 map makes **new names equal old names**:
old `STEP5`→new `STEP3`, while old `STEP3`→new `STEP1`. A sequential find/replace (sed)
would corrupt this. **Implementation must be a single atomic tokenized remap**: phase 1
rewrite every `STEP<n>` token to a unique placeholder (`__STEPTMP_n__`), phase 2 rewrite
placeholders to final names — applied uniformly to `config.py`, all modules, tests, and
notebook, matching **whole tokens only** (so `STEP12_EXACT_TOLERANCE` and `STEP12_B6` are
handled as units, and substrings like `step138` cell-ids are never touched).

**H2 — on-disk folder collision.** Same issue for the data folders
(`outputs_official\step5` → `step3` while `step3` exists). Since **all outputs regenerate
anyway**, the safe path is: **archive** the whole current `outputs_official\` to a backup,
then **regenerate fresh** into the new-numbered tree — not an in-place rename. (The one
exception is D1's `acquisition_report.csv`, copied into the new `step1` after regen.)

**H3 — D4 scope question (needs one more confirm).** The 80 tokens include non-folder
constants that merely *carry* a step number:
`STEP12_DISPLAY_ORDER`, `STEP12_ISOLATES`, `STEP12_FULL_FILES`, `STEP12_EXPECTED_ROWS*`,
`STEP12_EXACT_TOLERANCE`, `STEP5_EXPECTED_ROWS`, every `OVERWRITE_STEP<n>`, every
`STEP<n>_FILES`, `OFFICIAL_STEP3_FILES`, `STEP12B_FILES`/`STEP12C_FILES`.
**Confirm:** rename **all** of them for consistency (my recommendation, matches "every
occurrence"), or only the `*_OUTPUT_DIR` constants? This plan assumes **all**.

---

## Part 1 — WORKSTREAM A: renumber (−2), Option A functions, D4 constants

### A-0 Rule
Any `STEP<n>` with n≥3 → `STEP<n−2>`; suffix preserved
(`STEP10B`→`STEP8B`, `STEP13ALL`→`STEP11ALL`, `STEP12_B6`→`STEP10_B6`,
`STEP12_SANITY`→`STEP10_SANITY`, `STEP12_SEEDS`→`STEP10_SEEDS`,
`STEP12_THESIS_DIR`/`STEP12_CONFIG_DIR` derive and auto-follow).
Folder strings `"step<n>"`→`"step<n−2>"` (incl. `step10_attention`→`step8_attention`,
`step12_baseline6`→`step10_baseline6`, `step13_all`→`step11_all`, `step14_all`→`step12_all`,
`step12_sanity/seeds`→`step10_…`). `run_stepN()` function names **unchanged** (D3).
`MSCAD_merged.csv` filename **unchanged** (D5).

### A-1 `config.py` — `*_OUTPUT_DIR` definitions (names AND values change; D4)
| old constant → new constant | old value → new value |
|---|---|
| `STEP3_OUTPUT_DIR`→`STEP1_OUTPUT_DIR` | `step3`→`step1` |
| `STEP4_OUTPUT_DIR`→`STEP2_OUTPUT_DIR` | `step4`→`step2` |
| `STEP5_OUTPUT_DIR`→`STEP3_OUTPUT_DIR` | `step5`→`step3` |
| `STEP6_OUTPUT_DIR`→`STEP4_OUTPUT_DIR` | `step6`→`step4` |
| `STEP7_OUTPUT_DIR`→`STEP5_OUTPUT_DIR` | `step7`→`step5` |
| `STEP8_OUTPUT_DIR`→`STEP6_OUTPUT_DIR` | `step8`→`step6` |
| `STEP9_OUTPUT_DIR`→`STEP7_OUTPUT_DIR` | `step9`→`step7` |
| `STEP10_OUTPUT_DIR`→`STEP8_OUTPUT_DIR` | `step10`→`step8` |
| `STEP11_OUTPUT_DIR`→`STEP9_OUTPUT_DIR` | `step11`→`step9` |
| `STEP12_OUTPUT_DIR`→`STEP10_OUTPUT_DIR` | `step12`→`step10` |
| `STEP13_OUTPUT_DIR`→`STEP11_OUTPUT_DIR` | `step13`→`step11` |
| `STEP14_OUTPUT_DIR`→`STEP12_OUTPUT_DIR` | `step14`→`step12` |
| `STEP10B_OUTPUT_DIR`→`STEP8B_OUTPUT_DIR` | `step10_attention`→`step8_attention` |
| `STEP11B_OUTPUT_DIR`→`STEP9B_OUTPUT_DIR` | `step11_attention`→`step9_attention` |
| `STEP13ALL_OUTPUT_DIR`→`STEP11ALL_OUTPUT_DIR` | `step13_all`→`step11_all` |
| `STEP14ALL_OUTPUT_DIR`→`STEP12ALL_OUTPUT_DIR` | `step14_all`→`step12_all` |
| `STEP12_B6_OUTPUT_DIR`→`STEP10_B6_OUTPUT_DIR` | `step12_baseline6`→`step10_baseline6` |
| `STEP12_SANITY_OUTPUT_DIR`→`STEP10_SANITY_OUTPUT_DIR` | `step12_sanity`→`step10_sanity` |
| `STEP12_SEEDS_OUTPUT_DIR`→`STEP10_SEEDS_OUTPUT_DIR` | `step12_seeds`→`step10_seeds` |

Derived (auto-follow): `STEP12_THESIS_DIR`→`STEP10_THESIS_DIR`,
`STEP12_CONFIG_DIR`→`STEP10_CONFIG_DIR`, `B6_OUTPUT_DIR`. Trailing `# …\stepN` comments updated.

### A-2 `config.py` — the other step-numbered constants (D4, pending H3 confirm)
Renamed by the same rule: all `OVERWRITE_STEP<n>`; all `STEP<n>_FILES` (incl.
`STEP12B_FILES`→`STEP10B_FILES`, `STEP12C_FILES`→`STEP10C_FILES`,
`OFFICIAL_STEP3_FILES`→`OFFICIAL_STEP1_FILES`); `STEP*_EXPECTED_ROWS*`,
`STEP12_DISPLAY_ORDER`→`STEP10_DISPLAY_ORDER`, `STEP12_ISOLATES`,
`STEP12_FULL_FILES`, `STEP12_PROVENANCE_FILES`, `STEP12_EXACT_TOLERANCE`,
`STEP12_EXPECTED_TOLERANCE`, the `config_echo` dict keys (line ~554), and `DIRS_TO_CREATE`.
**Report filenames inside the `_FILES` dicts** also shift:
`step4_report.csv`→`step2_report.csv`, `step5_report.csv`→`step3_report.csv`,
`step10_attention_report.csv`→`step8_attention_report.csv`,
`step3_official_report.csv`→`step1_official_report.csv`,
`step3_official_summary.md`→`step1_official_summary.md`, …, `step14*`→`step12*`.

### A-3 every module + test + notebook referencing these tokens
All ~1,400 references updated by the atomic remap (H1). Specific non-config spots to verify:
- `experiments_step12_ae30/common_ae30._redirected` — hardcoded `"step%d" % n` from
  `AE30_REDIRECTED_STEPS` → shift the numbers.
- `experiments_step12_ae64/common_ae64._table_redirect` — uses constants → follows the rename.
- `test_official_step_numbering.py` — `STEP_FOLDERS` mapping, `NEW_FOLDERS`, and the
  step-heading/contents anchors; the notebook **top-note** "Step N → `stepM\`" and its test
  `test_the_note_maps_every_new_step_number_to_its_unchanged_folder`.
- Notebook code cells referencing `STEP*`/`"stepN"` → update, then refresh
  `tests/data/notebook_code_cells.json` + `notebook_cell_identity.json`.

### A-4 D1 collision-safe `step2`→`step1` move
Old `step2` holds only `acquisition_report.csv` (written by `data_loading.py:447`). New
`step1` = old `step3` contents (`MSCAD_merged.csv`, `column_mapping.csv`,
`official_manifest.json`, `label_counts.csv`, `column_quality.csv`, `duplicate_rows.csv`,
`label_conflicts.csv`, `step1_official_report.csv`, `step1_official_summary.md`).
**No filename conflict** (`acquisition_report.csv` is unique). Action: point
`data_loading.py`'s acquisition-report writer at the new `step1` (`STEP1_OUTPUT_DIR`), and
on migration copy any existing `acquisition_report.csv` into `step1`. (In single-source
official mode this file may be absent — the own-extraction path; still handled.)

---

## Part 2 — WORKSTREAM B: remove `Orig_Row` (verified positional alignment)

Same file-by-file set as v1; unchanged by D1–D5 except folder/constant names now follow A.

> ⚠️ **7 FROZEN modules are edited** — `splitting, feature_selection, evaluation, fusion,
> fusion_all, explainability, explainability_all`. `test_official_step_numbering.py` must
> move these out of `FROZEN_MODULES` into the named "changed with reason" set, annotated:
> "Orig_Row removal (migration vN): provenance key dropped; row alignment now positional."

| file | step (new #) | change | risk |
|---|---|---|---|
| `config.py` | — | drop `'Orig_Row'` from `METADATA_COLUMNS` (and the rebuild at ~2139); remove `ROW_COLUMN` usage | low |
| `official_dataset.py` | 1 | `build_merged`: delete the `out[ROW_COLUMN]=np.arange` add, the guard, and `Orig_Row` from `ordered`; drop `ROW_COLUMN`; remove the "Orig_Row is metadata" check in `build_report` | med |
| `preprocessing.py` | 2 | confirm `split_metadata`/metadata.csv no longer emit `Orig_Row` (follows config) | low |
| `splitting.py` 🧊 | 3 | remove `Orig_Row` split-position verification (~802-808); add explicit **positional** split↔source checks (length+order) | **high** |
| `feature_selection.py` 🧊 | 4 | drop `"Orig_Row"` from exclude list (line 144) | trivial |
| `evaluation.py` 🧊 | 7 | `build_predictions_frame` (614-619): remove `Orig_Row` branch; key `predictions.csv` on **verified test-row position** (no new id); fix duplicate-free/`meta_test` plumbing | **high** |
| `fusion.py` 🧊 | 9 | `align_frames` (474-500) requires `Orig_Row` → **rewrite to positional alignment** with explicit equal-length/equal-order assertions; fused frame (1125-1129, 1628) built on position | **highest** |
| `fusion_all.py` 🧊 | 9 | same rewrite | high |
| `step12_checks.py` | 8 (10.0) | provenance check merges on `Orig_Row` (234-316) → compare by position, or drop (your call) | med |
| `explainability.py` 🧊 | 10 | 21 refs — per-row attribution mapping → positional | high |
| `explainability_all.py` 🧊 | 10 | 10 refs → positional | high |
| `data_loading.py` | — | official load: remove `Orig_Row` dtype/handling (line 909 etc.). Own-extraction `Orig_Row` (merge path, 427/605) is separate; since no merging occurs in official mode, confirm whether to also strip it there | med |

**Tests touched by B:** the frozen-set update above, plus `test_fusion*`, `test_evaluation*`,
`test_splitting*`, `test_step12_checks*`, `test_comparison_approach1_row` (uses `meta`), and
the synthetic `meta_test` builders in `test_ae64_sensitivity`/`test_ae30_sensitivity`.

---

## Part 3 — implementation order
1. **Workstream A** (atomic token remap: config constants+values, modules, tests, notebook;
   D1 move; refresh notebook fixtures). Validate: `pytest tests -q` green (data-less).
2. **Workstream B** (Orig_Row removal; fusion/eval/split/explain rewrites; FROZEN_MODULES
   update). Validate: `pytest tests -q` green.
3. **Synthetic end-to-end dry run** (no real data): split→eval→fusion positional alignment holds.
4. **Regenerate + retrain** (see Part 4) — the only paid-once, multi-day phase.

A before B: A is mechanical/global and makes the tree consistent; B is the surgical,
bug-prone part and is easier to review on an already-renumbered tree.

---

## Part 4 — what must be regenerated / retrained (nothing survives)
Because folders move **and** the schema loses `Orig_Row`:
- **All step outputs** `step1…step12` (new numbering) — regenerate in order.
- **All trained models** — retrain: **Approach 1, Approach 2, B1, B3, B6** (done), **B2, B4**
  (in progress). Their `metrics.json`/`model.keras`/`predictions.csv` assume old paths + `Orig_Row`.
- **All sensitivity roots** — `outputs_official_ae30/`, `outputs_official_ae64/`,
  `outputs_official_paper_replication/` — stale (read official step dirs; evaluate via `meta`/`Orig_Row`).
- **Nothing reusable** — Step 1 itself changes (drops `Orig_Row`) and its folder moves to `step1`.
- Use the GPU/Colab path for the LSTM stages (B2/B4) as before.

---

## Part 5 — collision-safe migration & rollback

**Code (git):** do A and B on branch `claude/admiring-einstein-0u69k4`, one commit per
workstream, with the atomic remap. **Rollback = `git revert` / `git checkout`** the commits.

**Data (your machine, not in git):**
1. **Back up first:** `robocopy C:\MSCAD2022\outputs_official C:\MSCAD2022\_backup_outputs_official_YYYYMMDD /E`
   (and the three sensitivity roots). This is the rollback for data.
2. Do **not** in-place rename folders (H2). After pulling the renumbered code, **regenerate
   fresh** into the new-numbered folders by re-running the pipeline (Steps 1→…→12) + retrain.
3. Copy the old `step2\acquisition_report.csv` into the new `step1\` (D1), if present.
4. Once the new tree is verified, the old numbered folders in the backup can be deleted.
**Rollback:** `git revert` the two commits, restore `outputs_official` (and sensitivity
roots) from `_backup_…`, and you are exactly back to today's state.

---

## Part 6 — remaining confirmations before I write literal diffs
1. **H3 scope:** rename **all** 80 step-numbered constants (incl. `_FILES`,
   `OVERWRITE_STEP*`, `*_EXPECTED_ROWS`, `STEP12_DISPLAY_ORDER`, tolerances), or only
   `*_OUTPUT_DIR`? (Plan assumes all.)
2. **`step12_checks` provenance check:** rewrite to positional, or drop it?
3. **Own-extraction `Orig_Row`** in `data_loading.merge_flow_csvs` (427/605): strip too, or
   leave (it's the companion PCAPNG flow, unused in official single-source mode)?

On your "go" + these three, I'll produce the literal diffs (still nothing applied), then
execute A → validate → B → validate, and hand you the regenerate/retrain runbook.
