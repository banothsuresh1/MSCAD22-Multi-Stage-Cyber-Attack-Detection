# FINAL migration plan (v3) — step renumber (−2), constant rename, full `Orig_Row` removal

**Status: PLANNING / DOCUMENTATION ONLY.** No pipeline code, config, tests, notebooks,
folders, outputs, or models have been modified. This document is the only changed file.
Implementation waits for separate explicit approval.

## 0. Final approved decisions
- **Folders:** renumber **−2** for the official flow, every folder incl. derived/nested/
  suffixed/experiment-specific. Mapping: step3→step1, step4→step2, step5→step3, step6→step4,
  step7→step5, step8→step6, step9→step7, step10→step8, step11→step9, step12→step10,
  step13→step11, step14→step12.
- **Constants (H3 = YES):** rename **all** step-numbered constants to the new numbers —
  `*_OUTPUT_DIR`, `*_FILES`, `OVERWRITE_STEP*`, `*_EXPECTED_ROWS`, `STEP12_DISPLAY_ORDER`,
  tolerances, and every other step-numbered identifier — **names only; values unchanged**.
- **OVERWRITE_STEP\*:** KEEP all of them; rename numbers; **preserve every Boolean**
  (all currently `False`); no behavior/default change; no auto-overwrite introduced.
- **Functions (Option A):** `run_stepN()` names unchanged.
- **`MSCAD_merged.csv`:** filename kept; confirmed single-source (`build_merged` has no
  `concat`/`merge`); `Orig_Row` stripped from it and all dependents.
- **`Orig_Row`:** removed completely everywhere (incl. the PCAPNG/own-extraction path and
  all unused code); **no replacement id / hidden index**; verified positional alignment instead.
- **step2 collision (D1):** keep `acquisition_report.csv`; fold into new `step1` (no name
  conflict); new `step2` = preprocessing (old step4).
- **FROZEN_MODULES:** updated with explicit per-module reasons; no silent test bypass.
- **Regeneration:** all affected outputs + approved experiments regenerated after validation;
  configs/seeds/dims preserved; GPU/Colab for LSTM stages.
- **Collision safety:** atomic/tokenized remap only; never sequential text replacement.

## 0.1 Measured magnitude (repo findings, not assumptions)
~**1,400** references to **80** distinct `STEP<n>…` tokens across `src/mscad/`, `tests/`,
`config.py`; ~**96** hardcoded `"stepN"` path strings in `.py`; many in the notebook
(`step6`×38, `STEP14ALL_FILES`×16, … — note `step138`/`step149` are **cell ids**, not folders).

---

## 1. Folder mapping (complete, incl. derived/nested/suffixed)

| old folder | new folder | entry point (unchanged name) |
|---|---|---|
| `step3` | `step1` | `official_dataset.run_step3()` |
| `step4` | `step2` | `preprocessing.run_step4()` |
| `step5` | `step3` | `splitting.run_step5()` |
| `step6` | `step4` | `feature_selection.run_step6()` |
| `step7` | `step5` | `autoencoder.run_step7()` |
| `step8` | `step6` | `autoencoder.run_step8()` |
| `step9` | `step7` | `imbalance.run_step9()` |
| `step10` | `step8` | `models.run_step10()` |
| `step11` | `step9` | `evaluation.run_step11()` |
| `step12` | `step10` | `comparison.run_step12_complete()` / `experiments_step12\` |
| `step13` | `step11` | `fusion.run_step13()` |
| `step14` | `step12` | `explainability.run_step14()` |
| `step10_attention` | `step8_attention` | `models_attention.run_step10b()` |
| `step11_attention` | `step9_attention` | `models_attention.run_step11b()` |
| `step13_all` | `step11_all` | `fusion_all.run_step13all()` |
| `step14_all` | `step12_all` | `explainability_all.run_step14all()` |
| `step12_baseline6` | `step10_baseline6` | `experiments_baseline6` |
| `step12_sanity` | `step10_sanity` | sanity checks |
| `step12_seeds` | `step10_seeds` | seed sweep |
| `step12/thesis` | `step10/thesis` | thesis exports (derives from parent) |
| `step2` (acquisition) | **fold into `step1`** (D1) | `data_loading` acquisition report |

New official tree: `outputs_official\step1 … step12` (+ `step8_attention`, `step9_attention`,
`step10_baseline6`, `step10_sanity`, `step10_seeds`, `step11_all`, `step12_all`, `step10/thesis`).

---

## 2. Constant rename table — `*_OUTPUT_DIR` (names + values; `config.py` lines)

| line | old constant → new constant | old value → new value |
|---|---|---|
| 150 | `STEP3_OUTPUT_DIR`→`STEP1_OUTPUT_DIR` | `step3`→`step1` |
| 151 | `STEP4_OUTPUT_DIR`→`STEP2_OUTPUT_DIR` | `step4`→`step2` |
| 805 | `STEP5_OUTPUT_DIR`→`STEP3_OUTPUT_DIR` | `step5`→`step3` |
| 872 | `STEP6_OUTPUT_DIR`→`STEP4_OUTPUT_DIR` | `step6`→`step4` |
| 948 | `STEP7_OUTPUT_DIR`→`STEP5_OUTPUT_DIR` | `step7`→`step5` |
| 949 | `STEP8_OUTPUT_DIR`→`STEP6_OUTPUT_DIR` | `step8`→`step6` |
| 1024 | `STEP9_OUTPUT_DIR`→`STEP7_OUTPUT_DIR` | `step9`→`step7` |
| 1072 | `STEP10_OUTPUT_DIR`→`STEP8_OUTPUT_DIR` | `step10`→`step8` |
| 1073 | `STEP11_OUTPUT_DIR`→`STEP9_OUTPUT_DIR` | `step11`→`step9` |
| 1172 | `STEP12_OUTPUT_DIR`→`STEP10_OUTPUT_DIR` | `step12`→`step10` |
| 1227 | `STEP13_OUTPUT_DIR`→`STEP11_OUTPUT_DIR` | `step13`→`step11` |
| 1301 | `STEP14_OUTPUT_DIR`→`STEP12_OUTPUT_DIR` | `step14`→`step12` |
| 1639 | `STEP10B_OUTPUT_DIR`→`STEP8B_OUTPUT_DIR` | `step10_attention`→`step8_attention` |
| 1640 | `STEP11B_OUTPUT_DIR`→`STEP9B_OUTPUT_DIR` | `step11_attention`→`step9_attention` |
| 1691 | `STEP13ALL_OUTPUT_DIR`→`STEP11ALL_OUTPUT_DIR` | `step13_all`→`step11_all` |
| 1763 | `STEP14ALL_OUTPUT_DIR`→`STEP12ALL_OUTPUT_DIR` | `step14_all`→`step12_all` |
| 1849 | `STEP12_B6_OUTPUT_DIR`→`STEP10_B6_OUTPUT_DIR` | `step12_baseline6`→`step10_baseline6` |
| 1941 | `STEP12_SANITY_OUTPUT_DIR`→`STEP10_SANITY_OUTPUT_DIR` | `step12_sanity`→`step10_sanity` |
| 1965 | `STEP12_SEEDS_OUTPUT_DIR`→`STEP10_SEEDS_OUTPUT_DIR` | `step12_seeds`→`step10_seeds` |
| 149 | `STEP2_OUTPUT_DIR` (acquisition) | **folded into `STEP1_OUTPUT_DIR`** (D1); see §6 |

Derive & auto-follow (rename the name, value tracks parent): `STEP12_THESIS_DIR`→
`STEP10_THESIS_DIR` (1968), `STEP12_CONFIG_DIR`→`STEP10_CONFIG_DIR` (2002),
`B6_OUTPUT_DIR` (2022). Update trailing `# …\stepN` comments too.

## 3. Constant rename table — `OVERWRITE_STEP*` (KEEP, rename number, value preserved)

All currently `False`; **keep value `False`** for every one.

| line | old constant → new constant | value (unchanged) |
|---|---|---|
| 683 | `OVERWRITE_STEP4`→`OVERWRITE_STEP2` | `False` |
| 820 | `OVERWRITE_STEP5`→`OVERWRITE_STEP3` | `False` |
| 909 | `OVERWRITE_STEP6`→`OVERWRITE_STEP4` | `False` |
| 975 | `OVERWRITE_STEP7`→`OVERWRITE_STEP5` | `False` |
| 976 | `OVERWRITE_STEP8`→`OVERWRITE_STEP6` | `False` |
| 1051 | `OVERWRITE_STEP9`→`OVERWRITE_STEP7` | `False` |
| 1126 | `OVERWRITE_STEP10`→`OVERWRITE_STEP8` | `False` |
| 1127 | `OVERWRITE_STEP11`→`OVERWRITE_STEP9` | `False` |
| 1200 | `OVERWRITE_STEP12`→`OVERWRITE_STEP10` | `False` |
| 1268 | `OVERWRITE_STEP13`→`OVERWRITE_STEP11` | `False` |
| 1374 | `OVERWRITE_STEP14`→`OVERWRITE_STEP12` | `False` |
| 1647 | `OVERWRITE_STEP10B`→`OVERWRITE_STEP8B` | `False` |
| 1718 | `OVERWRITE_STEP13ALL`→`OVERWRITE_STEP11ALL` | `False` |
| 1797 | `OVERWRITE_STEP14ALL`→`OVERWRITE_STEP12ALL` | `False` |
| 1961 | `OVERWRITE_STEP12_SANITY`→`OVERWRITE_STEP10_SANITY` | `False` |

(No `OVERWRITE_STEP3` exists today; after rename a NEW `OVERWRITE_STEP3` appears, inherited
from old `OVERWRITE_STEP5` — expected, not a duplicate. This overlap is exactly why the remap
must be atomic, see §9/H1.)

## 4. Constant rename — `*_FILES` and all other step-numbered identifiers (H3=all)

Same −2 rule, **names only, values/contents unchanged** except the embedded report-filename
strings (§5):
`STEP4_FILES`→`STEP2_FILES`, `STEP5_FILES`→`STEP3_FILES`, `STEP6_FILES`→`STEP4_FILES`,
`STEP7_FILES`→`STEP5_FILES`, `STEP8_FILES`→`STEP6_FILES`, `STEP9_FILES`→`STEP7_FILES`,
`STEP10_FILES`→`STEP8_FILES`, `STEP11_FILES`→`STEP9_FILES`, `STEP12_FILES`→`STEP10_FILES`,
`STEP13_FILES`→`STEP11_FILES`, `STEP14_FILES`→`STEP12_FILES`,
`STEP10B_FILES`→`STEP8B_FILES`, `STEP11B_FILES`→`STEP9B_FILES`,
`STEP12B_FILES`→`STEP10B2_FILES`*, `STEP12C_FILES`→`STEP10C_FILES`,
`STEP13ALL_FILES`→`STEP11ALL_FILES`, `STEP14ALL_FILES`→`STEP12ALL_FILES`,
`STEP12_FULL_FILES`→`STEP10_FULL_FILES`, `STEP12_CONFIG_FILES`→`STEP10_CONFIG_FILES`,
`STEP12_PROVENANCE_FILES`→`STEP10_PROVENANCE_FILES`, `OFFICIAL_STEP3_FILES`→`OFFICIAL_STEP1_FILES`;
`STEP11_METRICS`→`STEP9_METRICS`, `STEP12_KEYS`→`STEP10_KEYS`,
`STEP12_DISPLAY_ORDER`→`STEP10_DISPLAY_ORDER`, `STEP12_ISOLATES`→`STEP10_ISOLATES`,
`STEP12_SANITY`→`STEP10_SANITY`, `STEP13ALL`/`STEP14ALL`/`STEP10B`/… tokens accordingly;
`STEP5_EXPECTED_ROWS`→`STEP3_EXPECTED_ROWS` (value `105_387` unchanged),
`STEP12_EXPECTED_ROWS`→`STEP10_EXPECTED_ROWS`,
`STEP12_EXPECTED_ROWS_OFFICIAL/OWN`→`STEP10_EXPECTED_ROWS_OFFICIAL/OWN`,
`STEP12_EXACT_TOLERANCE`→`STEP10_EXACT_TOLERANCE` (value `1e-9` unchanged),
`STEP12_EXPECTED_TOLERANCE`→`STEP10_EXPECTED_TOLERANCE` (value `5e-5` unchanged),
plus any `STEP<n>_DIR` aliases, the `config_echo` dict keys (~line 554), and `DIRS_TO_CREATE`.
`OWN_STEP3_SKIP_NOTE` (own-extraction) → `OWN_STEP1_SKIP_NOTE` per §6.

\* `STEP12B_FILES`/`STEP12C_FILES` are the own-extraction baseline-4/5 report sets; their
number is 12→10 but they must not collide with `STEP10B_FILES` (attention). Final names to be
fixed in the literal-diff pass; placeholder `STEP10B2_FILES` flagged for your naming
preference (or keep a distinct suffix). **This is the one naming nicety to confirm at diff time.**

## 5. Report / intermediate filename changes (inside the `_FILES` dicts)

| old filename → new filename |
|---|
| `step4_report.csv`→`step2_report.csv` |
| `step5_report.csv`→`step3_report.csv` |
| `step6_report.csv`→`step4_report.csv` |
| `step7_report.csv`→`step5_report.csv` |
| `step8_report.csv`→`step6_report.csv` |
| `step10_report.csv`→`step8_report.csv` |
| `step11_report.csv`→`step9_report.csv` |
| `step13_report.csv`→`step11_report.csv` |
| `step14_report.csv`→`step12_report.csv`, `step14.json`→`step12.json` |
| `step10_attention_report.csv`→`step8_attention_report.csv` |
| `step11_attention_report.csv`→`step9_attention_report.csv` |
| `step13_all_report.csv`→`step11_all_report.csv` |
| `step14_all_report.csv`→`step12_all_report.csv` |
| `step12_full_report.csv`→`step10_full_report.csv` |
| `step12_baseline4_report.csv`→`step10_baseline4_report.csv` |
| `step12_baseline5_report_6configs.csv`→`step10_baseline5_report_6configs.csv` |
| `step3_official_report.csv`→`step1_official_report.csv` |
| `step3_official_summary.md`→`step1_official_summary.md` |

(These regenerate anyway; the config strings change so new runs emit the new names.)

## 6. Own-extraction / PCAPNG path (D1 + §4 PCAPNG)
- `data_loading.py` writes `acquisition_report.csv` to `STEP2_OUTPUT_DIR` (old `step2`) in the
  companion PCAPNG flow. Repoint to **new `step1`** (`STEP1_OUTPUT_DIR`); on migration copy any
  existing `acquisition_report.csv` into `step1` (no conflict with step1's official files).
- `OWN_STEP3_SKIP_NOTE` and the companion-notebook step references renumber per the map.
- **Remove `Orig_Row` from the PCAPNG path entirely:** `data_loading.merge_flow_csvs`
  (`Orig_Row` created at ~427 and ~605), the dtype entry (~909), and any export — **even though
  the path is unused in official single-source mode.** No fallback `Orig_Row` left in dead code;
  no replacement id.

---

## 7. `Orig_Row` removal — complete impact (file by file; verified positional alignment)

> ⚠️ **FROZEN modules edited:** `splitting, feature_selection, evaluation, fusion, fusion_all,
> explainability, explainability_all`. `test_official_step_numbering.FROZEN_MODULES` must move
> each into the named "changed-with-reason" set, annotated e.g.:
> "Orig_Row removed (migration v3): provenance key dropped; alignment now positional."

| file | new step # | change | risk |
|---|---|---|---|
| `config.py` | — | drop `'Orig_Row'` from `METADATA_COLUMNS` (689 and the rebuild ~2139) | low |
| `official_dataset.py` | 1 | `build_merged`: remove `out[ROW_COLUMN]=np.arange`, the guard, `Orig_Row` in `ordered`; drop `ROW_COLUMN`; remove "Orig_Row is metadata" check in `build_report` | med |
| `preprocessing.py` | 2 | confirm `split_metadata`/`metadata.csv` emit no `Orig_Row` (follows config) | low |
| `splitting.py` 🧊 | 3 | remove `Orig_Row` split verification (~802-808); add explicit positional split↔source checks (length **and** order) | **high** |
| `feature_selection.py` 🧊 | 4 | drop `"Orig_Row"` from exclude list (144) | trivial |
| `evaluation.py` 🧊 | 9 | `build_predictions_frame` (614-619): drop `Orig_Row` branch; key `predictions.csv` on verified test-row position (no id); fix duplicate-free/`meta_test` plumbing | **high** |
| `fusion.py` 🧊 | 11 | `align_frames` (474-500) + fused frame (1125-1129, 1628): rewrite to positional alignment with equal-length **and** equal-order assertions | **highest** |
| `fusion_all.py` 🧊 | 11 | same rewrite | high |
| `step12_checks.py` | 10 | provenance merge-on-`Orig_Row` (234-316) → **rewrite positional** (see §8) | med |
| `explainability.py` 🧊 | 12 | 21 refs → positional | high |
| `explainability_all.py` 🧊 | 12 | 10 refs → positional | high |
| `data_loading.py` | — | official load: remove `Orig_Row` dtype/handling (909 …); PCAPNG path per §6 | med |

Also affected tests: `test_fusion*`, `test_evaluation*`, `test_splitting*`,
`test_step12_checks*`, `test_comparison_approach1_row` (uses `meta`), and the synthetic
`meta_test` builders in `test_ae64_sensitivity`/`test_ae30_sensitivity`.

---

## 8. Step-12 consistency checks — positional redesign (rewrite, not drop)
Replace `Orig_Row`-merge provenance with explicit positional validation, **keeping** these
checks and making each reject on mismatch rather than proceed:
- **Row-count** equality across features/labels/metadata/predictions and vs the Step-1 input.
- **Feature/label ordering**: same length AND same order (no reindexing between files).
- **Train/test correspondence**: split index arrays reproduce the same rows positionally.
- **Prediction/ground-truth correspondence**: `y_pred[i]` ↔ `y_true[i]` by position.
- **Cross-model prediction alignment** and **fusion input ordering**: all per-model frames
  have identical length AND identical row order before fusion.

**Rule (explicit in code + docstring):** *equal row counts alone never imply equal order.*
Where a natural, non-synthetic stable key exists it may be asserted; otherwise order is fixed
by construction (no shuffle between stages) and asserted by length + element-wise checks.
**No replacement id, hidden index, or persisted artificial identifier is introduced.**

---

## 9. Implementation order, collision safety (H1/H2)
1. **Workstream A** — atomic tokenized remap of all 80 constants + values + report filenames +
   ~96 hardcoded strings + notebook + tests; D1 repoint. Validate `pytest tests -q`.
2. **Workstream B** — `Orig_Row` removal + positional rewrites + FROZEN update. Validate.
3. **Synthetic dry run** (no real data): split→eval→fusion positional alignment holds.
4. **Regenerate + retrain** (§10).

**H1 (name collision):** new names equal old names (new `STEP3`=old `STEP5`; new
`OVERWRITE_STEP3`=old `OVERWRITE_STEP5`). Use a **two-phase, whole-token** remap
(`STEP<n>`→`__STEPTMP_n__`→final) applied uniformly; **never** sequential sed; never re-map an
already-renamed token. Match whole tokens so `step138`/`step149` cell-ids and unrelated
substrings are untouched.
**H2 (on-disk collision):** do **not** in-place rename data folders; **back up then regenerate
fresh** into the new tree (§10/§11).

---

## 10. Invalidation & regeneration (nothing survives)
- **All outputs** step1…step12 (new numbering) + derived folders — regenerate in order.
- **All trained models** — retrain: Approach 1, Approach 2, B1, B3, B6 (done), B2, B4
  (in progress). Every `metrics.json`/`model.keras`/`predictions.csv` assumes old paths + `Orig_Row`.
- **All sensitivity roots** — `outputs_official_ae30/`, `outputs_official_ae64/`,
  `outputs_official_paper_replication/` — stale (read official step dirs; evaluate via `meta`).
- **Nothing reusable** — Step 1 itself changes (drops `Orig_Row`) and its folder moves.
- GPU/Colab for LSTM stages (B2/B4), per the established route.

## 11. Collision-safe migration + rollback
**Code:** Workstream A and B as one commit each on `claude/admiring-einstein-0u69k4`.
Rollback = `git revert` those commits.
**Data (not in git):**
1. Back up: `robocopy C:\MSCAD2022\outputs_official C:\MSCAD2022\_backup_outputs_official_YYYYMMDD /E`
   (and the three sensitivity roots).
2. Pull renumbered code; **regenerate fresh** into new-numbered folders (re-run Steps 1→12 +
   retrain) — no in-place folder rename.
3. Copy old `step2\acquisition_report.csv` → new `step1\` (if present).
4. Verify, then delete old numbered folders from the backup.
Rollback (data) = restore from `_backup_…`.

## 12. Validation & acceptance criteria
- `pytest tests -q` **green after A**, and **green after B** (incl. updated FROZEN set + AE tests).
- No remaining references to old constant names or `"step3".."step14"` official paths
  (grep clean), and **no occurrence of `Orig_Row`** anywhere (grep clean) outside this doc.
- Every `OVERWRITE_STEP*` present with its original `False` value.
- Synthetic dry run passes positional-alignment assertions end to end.
- Numeric values/seeds/dims/tolerances/expected-rows unchanged (diff shows name changes only).
- After regeneration: new `step1…step12` populated; the Step-10 comparison + sensitivity tables
  render; model metrics reproduce within the pipeline's existing tolerances.

## 13. Open micro-confirm (diff-time only, non-blocking)
- Final name for `STEP12B_FILES`/`STEP12C_FILES` after →10 to avoid colliding with
  `STEP10B_FILES` (attention). I'll propose concrete names in the literal-diff pass.

On your explicit "go" I will produce the literal file-by-file diffs for review (still applying
nothing), then execute A → validate → B → validate, and hand over the regenerate/retrain runbook.
