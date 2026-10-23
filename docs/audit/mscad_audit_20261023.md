# MSCAD project audit -- Phase 1, read-only

**Date:** 2026-10-23
**Repository:** `banothsuresh1/MSCAD22-Multi-Stage-Cyber-Attack-Detection`
**Audited commit:** `25a0687` (`main`, after the Step 1 / Step 2 restructure)
**Branch for this report:** `audit/20261023`
**Scope:** inspection only. No file was edited, no model retrained, no metric
regenerated, nothing renamed or deleted. This report and nothing else is added.

> **Step numbering.** The notebook now numbers the official flow Step 1 to
> Step 12; the modules, functions and output folders keep the earlier
> numbering. Where this report names a folder (`step11\`) or a function
> (`run_step11`) it uses the code's name, and gives the notebook's number
> beside it on first use. Notebook Step 8 = `step10\`, Step 9 = `step11\`,
> Step 10 = `step12\`, Step 11 = `step13\`, Step 12 = `step14\`.

---

## Executive summary

The pipeline's methodology holds up: the scaler, the feature selector, the
autoencoder and both PCAs are each fitted on training rows only, every
configuration goes through one `train_classifier` with the same seed, split,
class weights, epochs, batch size, patience and early-stopping metric, every
configuration is scored by the same `evaluate()` on the same test rows, and the
duplicate-free subset has a single definition shared by all eight. The missing
Approach 1 row is **not** a training failure and not the retraining lock: it is
a lookup bug. `comparison.result_path()` has a special branch for Approach 2
(reads `step11_attention\metrics.json`) and one for Baseline 6 (reads
`step12_baseline6\metrics.json`), but none for `"proposed"`, whose evaluation
Step 9 writes to `step11\metrics.json`; so the comparison looks for
`step12\proposed\metrics.json`, does not find it, and prints the blank row's
placeholder text. Three further reporting defects are confirmed: Approach 2's
`input features` is copied from `config.N_SELECTED_FEATURES` instead of measured,
the blank-row placeholder tells the reader to "run 10.2" whichever row is
missing, and Baseline 6's 3-component PCA rests on a rationale
(`PCA_N_COMPONENTS = 0.95` keeps only 2 components) that was measured on the
own-extraction data and is false on the official dataset, where Baseline 4 keeps
6 -- so B6 differs from B4 in two factors at once and cannot be read as
isolating the CNN. B3 beating the autoencoder configurations is a legitimate
result, not a bug: B3's protocol is identical to the others and its only
advantage is structural, the full feature set at full width with no 15-unit
bottleneck. The 22-47% binary FPR is a property of the dataset and the decision
rule rather than a code defect, and the first thing to read against it is Step
1's own `label_conflicts.csv`, which sets a floor under it. One latent crash is
present (`meta_test["Source_Capture"]` unguarded in the fusion smoothing path,
reachable only if `TEMPORAL_SMOOTHING` is turned on) and one audit capability is
absent entirely: nothing anywhere records a hash of the training code, so "was
this saved model produced by the code at HEAD?" cannot currently be answered.

### What this audit could not verify from here, and why

This audit ran in a Linux cloud container with **no access to
`C:\MSCAD2022`**. Every statement below about *code* is verified at
`25a0687` with a file:line citation. Every statement that would need the
dataset or the saved artifacts -- the CSV header, the actual feature counts,
`metrics.json` contents, model file dates and sizes, the PCA variance kept,
the confusion counts -- is marked **NOT VERIFIED HERE** and carries the exact
command to produce it on your machine. No number in this report is invented,
and no artifact figure is quoted as if it had been read.

---

## Findings

| ID | Verdict | Title | Evidence | Impact | Minimum fix |
|---|---|---|---|---|---|
| H1 | **CONFIRMED DEFECT** | Approach 1's row is looked up in the wrong place | `src/mscad/comparison.py:78-100` (`result_path` has branches for Approach 2 and B6, none for `"proposed"`), `:104-109`, `:287-295` | The proposed pipeline is absent from its own eight-row comparison even though Step 9 has evaluated it | `src/mscad/comparison.py` (add the `"proposed"` branch); `src/mscad/evaluation.py` (record the descriptor fields); one test |
| B1-a | **CONFIRMED DEFECT** | Approach 2 reports intended, not measured, input width | `src/mscad/models_attention.py:1079` -- `"n_input_features": int(config.N_SELECTED_FEATURES)` | If Step 4 selected fewer than 30 features the table still says 30; every other writer measures it | `src/mscad/models_attention.py` (one line: use the loaded frame's width) |
| H1-b | **CONFIRMED DEFECT** | The blank row tells the reader to run the wrong section | `config.py:1567` -- `NOT_TRAINED_TEXT = "not yet trained - run 10.2"`; used for every key at `src/mscad/comparison.py:270-276` | "10.2" trains Approach 2, and 10.2 is the old number for notebook section 8.2; the instruction is wrong for seven of the eight rows | `src/mscad/comparison.py` `_blank_row` to call the existing `which_step_produces(key)`; `config.py` to drop the section number |
| D2-a | **CONFIRMED DEFECT** (experimental design) | B6's component count rests on a figure that is false for this dataset | `config.py:1843-1851` ("on these latents it selects 2 components"), `config.py:1851` `B6_PCA_COMPONENTS = 3`, `src/mscad/experiments_baseline6.py:131-142` | B4 keeps 6 components on the official data, so B6 could have used B4's space; as it stands B6 differs from B4 in model type **and** dimensionality, and "B6 isolates the CNN" does not hold | `config.py` (the comment, and `B6_PCA_COMPONENTS = "same_as_B4"`) -- but this requires retraining B6, so Phase 2 may instead state the confound in section 10.14 |
| F-a | **CONFIRMED GAP** | No record of which code produced a saved model | every SHA-256 in the tree is over output bytes: `src/mscad/comparison.py:905-915`, `src/mscad/experiments.py:308-337`, `experiments_baseline6.py:279-292`; no `inspect.getsource` anywhere | "Is this model still the code at HEAD?" is unanswerable; a model from before the eight-file refactor is indistinguishable from one trained now | `src/mscad/step12_checks.py` (a source-hash helper), `src/mscad/experiments.py` and `src/mscad/models.py` (record it at save time); one test |
| A2-a | **RISK** (latent crash) | `Source_Capture` subscripted without a presence check | `src/mscad/fusion.py:1675-1677`, `src/mscad/fusion_all.py:588-590` | `KeyError: 'Source_Capture'` in notebook Step 11 and 11.8 the moment `TEMPORAL_SMOOTHING` is set to `True`; the official file has no such column. Not reachable today (`config.py:1265` is `False`) | `src/mscad/fusion.py`, `src/mscad/fusion_all.py` (one presence check each) |
| B3-a | **RISK** | "Step 6 top 30" in the configuration table is intent, not the saved count | `src/mscad/comparison.py:1044-1053`; the selector takes `head(min(n, available))` at `src/mscad/feature_selection.py:622` and `:1002` | If the Pearson filter leaves fewer than 30 features, the configuration table claims 30 while the comparison table's `input features` shows the truth -- two columns disagreeing in one report | `src/mscad/comparison.py` (read the count from `selected_features.json` or from the row's `n_input_features`) |
| C-a | **RISK** (small leak) | Constant columns are dropped over the whole dataset, before the split | `src/mscad/preprocessing.py:231-258`, called at `:318` | Which columns exist is decided with test rows in scope. Harmless in effect (a column constant over 128,799 rows is almost certainly constant over the 103,039 training rows) but it is a decision taken with test data | Document it in the notebook's Step 2 "Why" block; moving it into Step 3 changes the feature set and would require retraining everything, so it should not be moved now |
| A1-a | **INFORMATIONAL** | The identifier references that remain, and their guards | listed in full in section A below | None in the current configuration: every reference in an active official-mode path is guarded except A2-a | none |
| A1-b | **INFORMATIONAL** | The per-alert evidence report prints `unknown:? -> unknown:?` | `src/mscad/explainability.py:1159-1172` | In official mode `meta_test` exists (it carries `Orig_Row`) but has no identifiers, so the "Flow context" line reads like missing data rather than "this dataset has none"; the `"identifiers unavailable"` branch is only taken when `meta is None` | `src/mscad/explainability.py` (treat "no identifier columns" the same as `meta is None`) |
| A3-a | **INFORMATIONAL** | No `Timestamp` column is handled correctly, and silently | `src/mscad/preprocessing.py:100-101` keeps only metadata columns that exist; `src/mscad/official_dataset.py:251-297` reports the absent ones as *unavailable* | Nothing errors and nothing is invented. It is silent by design, and Step 1's identifier table is where it is said out loud | none |
| E-a | **INFORMATIONAL** | The training protocol is identical by construction | `src/mscad/experiments.py:121-176`, `src/mscad/imbalance.py:175-215`, `config.py:812, 1048, 1111-1119` | The protocol table in section E is not a coincidence of three runs agreeing: one function with one set of config constants trains all of them | none |
| G1-a | **INFORMATIONAL** | The majority-class line is computed on the test split, and Brute_Force is the majority | `src/mscad/experiments_step12/report.py:206-242`; 88,502 / 128,799 = 0.68713 | `0.6872` is right and it is Brute_Force, not Normal. The notebook does not quote the stored constant for this -- it recomputes it | none |
| G3-a | **INFORMATIONAL** | One duplicate definition, shared by every configuration | `src/mscad/evaluation.py:291-307`; the same call in all nine runners (`evaluation.py:801`, `experiments.py:997`, `experiments_pca.py:972`, `experiments_baseline5.py:1017`, `experiments_baseline6.py:352`, `models_attention.py:1054`, `fusion.py:1614`, `fusion_all.py:1701`, `step12_checks.py:821`) | Comparable across rows | none |
| G5-a | **INFORMATIONAL** | Inference timing is uniform in policy, machine-dependent in fact | `src/mscad/evaluation.py:358-385` | One warm-up batch discarded, `batch_size = 256`, mean of 3 runs -- identical for all eight. But configurations timed in different sessions were timed under different machine load; `std_seconds` is saved, so the spread is auditable | Quote `ms_per_1k` with its `std_seconds`, or re-time all eight in one session (no retraining needed) |
| I-a | **INFORMATIONAL** | B3's advantage is structural, not procedural | `config.py:1192-1194` (`use_ae: False`), `src/mscad/experiments.py:192-223` (`"all"` = every Step 3 feature), `:386-399` (the AE branch is skipped) | B3 is not unfairly advantaged by weighting, seed, split or test set. It keeps every feature at full width and has no 15-unit bottleneck. That is the ablation working, and it is a result | none; report it as a result |
| J-a | **INFORMATIONAL** | The binary FPR is a dataset-and-threshold property | `src/mscad/evaluation.py:245-285` | Negatives are the `Normal` rows only; the rule is `P(attack) >= 0.5`. Diagnostic, not a defect | none; section J gives the numbers to read |
| DOC-a | **INFORMATIONAL** | A stale note still says the official flow starts at Step 3 | `config.py:2136-2143` (`SKIP_STEPS_NOTE`) | Printed by the Step 1 cell; contradicts the notebook's own numbering since `25a0687` | `config.py` (reword the note) |

---

## A) Dataset assumptions

### A1 -- every reference to `Src IP`, `Dst IP`, `Src Port`, `Dst Port`, `Protocol`, `Flow ID`, `Timestamp` in the active official-mode paths

The project's own code already records the belief in your section 2:
`config.py:275-284` declares these six as "**the 6 flow identifiers the Kaggle
MSCAD.csv does NOT have** -- the whole reason we regenerate the flows from the
raw captures". Official mode therefore lists all six as *candidates* and keeps
only the ones the file turns out to have.

**NOT VERIFIED HERE:** the CSV header itself. I cannot read
`C:\MSCAD2022\CSV_Original_kaggle\MSCAD.csv`. To settle it in one line:

```
cd /d C:\Users\banot\MSCAD22-Multi-Stage-Cyber-Attack-Detection\Notebook
conda activate jupyfix
python -c "import pandas as pd, config; print(list(pd.read_csv(config.OFFICIAL_CSV, nrows=0).columns))"
```

Every reference, by module:

| Where | Line | What it does | Guarded? |
|---|---|---|---|
| `config.py` | 277-284 | declares the six as own-mode identifiers | n/a (data) |
| `config.py` | 2074-2076 | `OFFICIAL_IDENTIFIER_COLUMNS` -- the six, as *candidates* | n/a (data) |
| `config.py` | 2116-2131 | official overrides: metadata, combined, int-cast lists | n/a (data) |
| `preprocessing.py` | 100-101 | `meta_cols = [c for c in METADATA_COLUMNS if c in df.columns]` | **yes** -- absence is normal |
| `preprocessing.py` | 392-396 | `if "Protocol" not in features.columns: return` with a note | **yes** |
| `preprocessing.py` | 849-874 | `allow_missing_identifiers` defaults to `True` in official mode; the readable `Protocol` column is skipped with a note when the one-hots do not exist | **yes** |
| `preprocessing.py` | 944-946 | port int-cast: `if col in ids.columns` | **yes** |
| `official_dataset.py` | 251-297 | finds the identifiers present, records the rest as *unavailable* | **yes** -- this is the report |
| `evaluation.py` | 620-622 | `if meta_test is not None and "Source_Capture" in meta_test.columns` | **yes** |
| `models.py` | 104-109 | `INPUT_MODE = "time_window"` needs `Source_Capture` + `Timestamp`; raises a named `ModelError` listing what is missing | **yes**, and unreachable: `config.py:1100` is `"feature_seq"` |
| `fusion.py` | 406-427 | `protocol_contexts`: `if ... or "Protocol" not in meta_test.columns` | **yes**; and `CONTEXT_MODE` is `"consensus_class"` (`config.py:1255`) |
| `fusion.py` | 1133-1134 | `if meta is not None and "Source_Capture" in meta.columns` | **yes** |
| **`fusion.py`** | **1675-1677** | **`meta_test["Source_Capture"]` -- bare subscript** | **NO** -- see A2 |
| **`fusion_all.py`** | **588-590** | **the same bare subscript** | **NO** -- see A2 |
| `fusion_all.py` | 1134-1135 | guarded copy | **yes** |
| `fusion.py` | 1628 | `meta_test["Orig_Row"]`, guarded only on `meta_test is not None` | partial; `Orig_Row` *is* added by Step 1 (`config.py:2082`), so it exists in official mode |
| `explainability.py` | 1039-1053 | `if meta is None or "Src IP" not in meta.columns: return` | **yes** |
| `explainability.py` | 1163-1170 | `line.get("Src IP", "unknown")` etc. | **yes**, but misleading -- see A1-b |
| `explainability.py` | 1899-1903 | `if column in meta.columns` before copying | **yes** |
| `splitting.py` | 7, 548 | docstrings only | n/a |
| `feature_selection.py` | 144, 183 | a name list used to *detect leakage* and an error message | n/a -- this is a check, not a dependency |
| `data_loading.py` | own Step 2/3 functions | guarded by `own_step3_applies()` / `OWN_STEP3_SKIP_NOTE` | **yes** (five entry points; `tests/test_official_dataset.py` asserts it) |

**Verdict: INFORMATIONAL**, with the one exception promoted to A2.

### A2 -- code that reconstructs a session or groups by identifiers that do not exist

One place, two copies:

```
src/mscad/fusion.py:1675    if config.TEMPORAL_SMOOTHING and meta_test is not None:
src/mscad/fusion.py:1676        risk = ema_within_capture(risk, meta_test["Source_Capture"],
src/mscad/fusion.py:1677                                  meta_test.get("Timestamp"))
```

`ema_within_capture` (`fusion.py:282-307`) groups the risk series by capture and
smooths within each group, ordered by `Timestamp`. The `Timestamp` argument is
fetched with `.get` and degrades to row order when absent; `Source_Capture` is
not. In official mode the metadata carries no `Source_Capture` -- the column
does not exist in `MSCAD.csv`, `OFFICIAL_ROW_COLUMNS` lists it only so it can
never reach the features (`config.py:2078-2083`), and `split_metadata` keeps
only columns that exist. So this raises `KeyError: 'Source_Capture'`, which is
the same shape as the Step 1 failure you already hit.

It does **not** fire today: `config.py:1265` sets `TEMPORAL_SMOOTHING = False`,
and `fusion_all.py:588` sits behind the same flag. `CONTEXT_MODE` is
`"consensus_class"`, so the per-protocol context -- which *is* guarded, and
*would* raise a named error (`fusion_all.py:580-583`) -- is not used either.

**Verdict: RISK (latent defect).** Minimum fix: one presence check in each of
the two lines.

### A3 -- does a missing `Timestamp` error silently?

No. `preprocessing.split_metadata` (`preprocessing.py:100-101`) intersects
`METADATA_COLUMNS` with the frame's real columns and reports the remainder in
its own table ("not present: ..."). Step 1 reports the same thing as a first-class
result: `official_dataset.find_identifiers` lists what was found and what is
*unavailable* (`official_dataset.py:251-297`), and the Step 1 summary prints
"Unavailable: Flow ID, Src IP, Dst IP, Src Port, Dst Port, Timestamp. Later steps
skip the parts that need them -- per-capture reporting, alert context and
same-source grouping -- rather than inventing a value."

**Verdict: INFORMATIONAL.** The absence is reported, not swallowed, and nothing
downstream reads the column. The one place that would read it without asking is
A2.

---

## B) Feature counts and label

### B1 -- the exact feature count each configuration's model sees

How each writer obtains the number that lands in `metrics.json` and therefore in
the comparison table's `input features` column:

| # | Config | Source of `n_input_features` | Measured or intended? |
|---|---|---|---|
| 0 | Approach 1 | *not written at all* -- Step 9's `metrics.json` has no such key (`evaluation.py:700-721`) | neither; see H1 |
| 1 | Approach 2 | `int(config.N_SELECTED_FEATURES)` -- `models_attention.py:1079` | **intended** |
| 2 | B1 | `source["n_features"]` = `X_train.shape[1]` -- `experiments.py:383, 440` | measured |
| 3 | B2 | same | measured |
| 4 | B3 | same | measured |
| 5 | B4 | `int(source["n_features"])` -- `experiments_pca.py:507` | measured |
| 6 | B5 | `int(len(features))` -- `experiments_baseline5.py:361` | measured |
| 7 | B6 | `int(len(out["pipeline"]["order"]))` -- `experiments_baseline6.py:496` | measured |

So six of the eight report what actually entered the model, one reports its
configured intent, and one reports nothing.

**B1-a, CONFIRMED DEFECT:** `models_attention.py:1079`. Approach 2 shares
Approach 1's inputs, which are the Step 4 selected features -- and the selector
can return fewer than 30 (see B3). If it does, Approach 2's row says 30 and is
wrong, while B5's row beside it says the truth.

**NOT VERIFIED HERE** -- what the counts actually are. The two numbers that
settle `66` and `30`:

```
cd /d C:\Users\banot\MSCAD22-Multi-Stage-Cyber-Attack-Detection\Notebook
conda activate jupyfix
python -c "import pandas as pd, config, json; print('Step 3 X_train width:', pd.read_csv(config.STEP5_OUTPUT_DIR/'X_train.csv', nrows=0).shape[1]); print('Step 4 selected:', len(json.loads((config.STEP6_OUTPUT_DIR/'selected_features.json').read_text())['selected']))"
```

Note that `66` is not guaranteed: Step 2 drops zero-variance columns
(`preprocessing.py:231-258`, called at `:318`), so if any of the official
columns is constant across the dataset the "no selection" configurations see
fewer than 66. Whatever the number is, the comparison table reports it
correctly for B1-B6 -- it is the *intent* columns that may disagree.

### B2 -- is `Label` ever a model or AE input?

No, and it is checked rather than assumed:

* `preprocessing.split_features_labels` removes the label column from the
  feature frame (`preprocessing.py:107-114`);
* Step 4's input check fails loudly if a label column is present in `X_train`:
  `leaked_labels = [c for c in LABEL_COLUMNS if c in X_train.columns]`
  (`feature_selection.py:138-143`), and the same block fails on leaked
  identifiers (`:145-150`);
* the autoencoder reads `X_train_selected.csv` and asserts its columns equal
  `selected_features.json` in order (`autoencoder.py:152-166`);
* the classifiers receive `arrange_inputs(Z, y)`, which takes `y` separately
  (`models.py:157-172`).

**Verdict: INFORMATIONAL (clean).**

### B3 -- do the 30-feature configurations really select 30, with the intended selector?

The selectors are the intended ones:

* Approach 1 and Approach 2 use the fused ranking -- mutual information
  (`feature_selection.py:292`), the Pearson redundancy filter at
  `PEARSON_THRESHOLD = 0.95` (`config.py:875-879`), then permutation importance,
  combined at 6.5 and cut at 6.6;
* B5 uses MI **only**, and says why: "ranking it would smuggle the Pearson step
  into an 'MI only' baseline" (`experiments_baseline5.py:83`), calling
  `fs.mutual_information(X_train, y_train)` directly (`:120`).

The count is **not** guaranteed to be 30. The cut is
`taken = ranking.head(min(n, available))` (`feature_selection.py:622`) and the
step's own check expects `min(config.N_SELECTED_FEATURES, len(pearson_kept))`
(`:1002`). With a 0.95 redundancy threshold over CICFlowMeter columns -- which
carry several near-duplicate pairs by construction (`Pkt Size Avg` against
`Pkt Len Mean`, `Fwd Seg Size Avg` against `Fwd Pkt Len Mean`, the `Subflow *`
columns against their `Tot *` counterparts) -- fewer than 30 surviving is a real
possibility.

**B3-a, RISK:** if that happened, `configuration_table_8` still prints
"Step 6 top 30" because that string is built from `config.N_SELECTED_FEATURES`
(`comparison.py:1051-1053`), not from the saved selection. Run the command in B1
to get the real number.

---

## C) Pipeline integrity -- fitted on train, applied to val/test

Stated once, as requested. Every stage, with its evidence:

| Stage | Fitted on | Applied to | Evidence | Verdict |
|---|---|---|---|---|
| Scaler (the one that matters) | `X_train` rows only | train and test | `splitting.py:396-399` -- `scaler.fit(X_train[columns])`, then `.transform` on each side. Step 3 deliberately reads `features_clean.csv`, the **unscaled** frame, never `features_scaled.csv` (`splitting.py:5, 38-40, 82-83, 100`) | **fitted on train only -> applied to val/test** |
| Feature selection (MI + Pearson + PI) | `X_train` alone | the same 30 columns taken from `X_test` once | `feature_selection.py:40` -- "X_test is read once, at 6.6, to take the same 30 columns"; MI/Pearson/PI all consume `X_train` | **fitted on train only -> applied to val/test** |
| Autoencoder (Step 5, Approach 1/2) | `X_train_selected` alone, with its own inner split and early stopping | both sides encoded | `autoencoder.py:31` | **fitted on train only -> applied to val/test** |
| Autoencoder (per-baseline, B1/B2/B4/B6) | the training rows | both sides | `experiments.py:226-229` (`encode_with_new_ae` -- "trains a baseline autoencoder on the TRAINING rows, then encodes both sides") | **fitted on train only -> applied to val/test** |
| PCA (B4) | the **training latents** | the test latents | `experiments_pca.py:119-132` -- `fit_pca_on_train(Z_train)`, and the docstring at `:123-128` records that the test latents are deliberately not an argument | **fitted on train only -> applied to val/test** |
| PCA (B6) | the training latents of B4's encoder | the test latents | `experiments_baseline6.py:131-142` | **fitted on train only -> applied to val/test** |
| Class weights | the **fit** part only, not the whole training set | the training loss; validation is unweighted | `imbalance.py:182-188` -- "the weights come from the FIT part ALONE"; `experiments.py:166` -- `validation_data=(...)  # UNWEIGHTED` | **fitted on train only** |
| Early stopping | the inner validation split of the training rows | restores best weights | `experiments.py:160-168`, `config.py:1114-1119` | clean |

One qualification, and it is the only one:

**C-a, RISK (small leak).** Step 2 drops zero-variance columns over the whole
dataset, before any split exists (`preprocessing.py:231-258`, called at `:318`).
The decision "which columns exist" therefore has test rows in scope. The effect
is almost certainly nil -- a column constant across 128,799 rows is constant
across the 103,039 training rows too -- but it is a choice made with test data
and the write-up should say so rather than leave it to be found. Moving the drop
into Step 3 would change the feature set and invalidate every saved model, so it
should **not** be moved now.

The test set itself is used once, for scoring: `models_attention.py:1094-1098`
records that check explicitly, and no stage above reads `X_test` except to apply
a fitted transform or to compute the duplicate mask (which uses features only,
never labels).

---

## D) Dimensions through the stack

### D1 -- B4: is the PCA input the 15-dim AE latent, or the raw features?

**The latent.** `fit_pca_on_train(Z_train, ...)` takes the encoded frame; the
module docstring at `experiments_pca.py:29` states "The PCA is fitted on the
TRAINING latent features only", and the returned info records
`"latent_dim": int(values.shape[1])` beside `"n_components"` and
`"variance_kept"` (`:139-141`). The comparison table's `PCA components` column
is then `"%d of %d" % (pca_components, latent_dim)` (`comparison.py:250-251`),
which is why your run shows "6 of 15" -- 6 components out of a 15-unit latent
space, not out of 66 raw features. The requested setting is
`PCA_N_COMPONENTS = 0.95` (`config.py:1421`), a **variance target**: sklearn
keeps the fewest components reaching 95% explained variance.

**NOT VERIFIED HERE:** the variance actually kept. It is saved:

```
python -c "import pandas as pd, config; print(pd.read_csv(config.STEP12_OUTPUT_DIR/'baseline4_ae_pca_lstm'/'pca_variance.csv'))"
```

### D2 -- B6: the same, with 3 components

B6 does **not** use a variance threshold. `B6_PCA_COMPONENTS = 3`
(`config.py:1851`) is a hard integer, and `experiments_baseline6.py:131-142`
fits a *new* PCA with exactly that many components on the same training latents,
writing it inside the B6 folder and leaving B4's `pca.joblib` untouched.

So, answering your question directly: **the threshold that produced 6 is
`PCA_N_COMPONENTS = 0.95`; nothing produced the 3 -- it was chosen by hand.**

The recorded reason (`config.py:1843-1851`) is:

> `PCA_N_COMPONENTS = 0.95` is a VARIANCE target, and on these latents it
> selects 2 components. `models.build_cnn_lstm` refuses an input with fewer
> steps than `CNN_KERNEL` (3) [...] 3 is the smallest count the shared stack
> accepts.

**D2-a, CONFIRMED DEFECT (experimental design).** That rationale was measured on
the own-extraction latents. On the official dataset the same 0.95 target keeps
**6** components, which is comfortably above `CNN_KERNEL = 3`, so the obstacle
the integer was introduced to work around does not exist here. The consequence
is not cosmetic:

* `STEP12_ISOLATES[BASELINE6_NAME]` claims B6 isolates "a CNN front end on the
  PCA features (B4 + CNN, B1 + PCA)" (`config.py:1887-1888`);
* but B6 runs on 3 components and B4 on 6, so **two** things differ between
  them, and B6's lower macro-F1 (0.6786 against B4's 0.6836) cannot be
  attributed to the CNN;
* at 3 steps the architecture also degrades: `pca_length_table`
  (`experiments_baseline6.py:145-176`) returns `[WARN]` because `steps < 2 *
  kernel` and the pooling layer is skipped, so the LSTM reads 3 positions.

Setting `B6_PCA_COMPONENTS = "same_as_B4"` would make the comparison
one-factor -- but it requires retraining B6, which Phase 1 forbids and which
your retraining lock is there to prevent. The Phase 2 decision is yours: retrain
B6 in B4's 6-component space, or keep the result and state the confound in the
discussion. Either way `config.py:1843-1851` is now factually wrong about this
dataset and should be corrected.

### D3 -- B2 and B4: LSTM input shape, and any accidental 1D collapse

No collapse. `INPUT_MODE = "feature_seq"` (`config.py:1100`) routes through
`arrange_inputs -> as_feature_sequence(Z)` (`models.py:165-167`), which lays
each flow out as `(steps, 1)`: B2 sees `(15, 1)` -- 15 latent units as 15 steps
of one channel -- and B4 sees `(6, 1)`. The shape that actually ran is recorded
per configuration as `model_input_shape` (`experiments.py:442`) and printed in
the configuration table (`comparison.py:1069`), so it is checkable rather than
assumed. The model card carries the warning that this is not a time axis: "the
latent units have no natural order -- which feature sits at position 3 is an
arbitrary consequence of how the encoder initialised" (`models.py:497-505`).

### D4 -- B3: raw features reshaped for Conv1D, and no AE applied by mistake

Confirmed, three ways: the spec sets `"use_ae": False` (`config.py:1192-1194`);
the AE branch in `run_experiment` is inside `if spec["use_ae"]:`
(`experiments.py:386-399`), so with `use_ae` false `X_train`/`X_test` pass
through untouched and `latent_dim` stays `None`; and the saved row therefore
shows `latent dim = "-"` (`comparison.py:249`). Its input is
`load_feature_source("all")` -- every Step 3 feature, already scaled with the
train-fitted scaler (`experiments.py:195-197`) -- reshaped to `(n_features, 1)`.

---

## E) Training protocol across the eight configurations

Every configuration that trains, trains through one function:
`experiments.train_classifier` (`experiments.py:121-176`), which is
`models.train_model` with a pluggable builder and, in its own words, "Same Step 9
split, same class weights on the training loss, same unweighted validation, same
macro-F1 early stopping."

| Item | Value | Where it comes from | Same for all eight? |
|---|---|---|---|
| Split (train/test) | stratified 80/20 on row positions, `shuffle=True` | `splitting.py:245-247`; `TEST_SIZE = 0.20` (`config.py:808`), `random_state = RANDOM_STATE` | **yes** -- one saved split; all configurations read `X_train`/`X_test` or their encodings |
| Split indices | saved as `train_indices.csv` / `test_indices.csv` | `splitting.py` Step 3 outputs | **yes**, reproducible without re-running |
| Seed | `42` | `RANDOM_SEED = 42` -> `RANDOM_STATE` (`config.py:225, 812`); `md.set_all_seeds(seed)` (`experiments.py:134`) | **yes** |
| Inner fit/validation split | 10%, stratified, `random_state = 42` | `imbalance.make_training_sets` (`imbalance.py:190-215`), `CLF_VAL_SIZE = 0.10` (`config.py:1048`) | **yes** -- identical `y_train` and identical seed give identical indices |
| Class weights | computed on the **fit** part only | `imbalance.py:182-188`; applied via `class_weight=` (`experiments.py:167`) | **yes** |
| Validation weighting | none | `experiments.py:166` | **yes** |
| Max epochs | 50 | `CLF_EPOCHS` (`config.py:1111`) | **yes** |
| Batch size | 256 | `CLF_BATCH_SIZE` (`config.py:1112`) | **yes** |
| Learning rate | 1e-3 | `CLF_LR` (`config.py:1113`) | **yes** |
| Patience | 8 | `CLF_PATIENCE` (`config.py:1114`) | **yes** |
| Early-stopping metric | `val_macro_f1`, mode `max`, best weights restored | `EARLY_STOP_METRIC` (`config.py:1119`), `experiments.py:160-163` | **yes** |
| CNN / LSTM hyperparameters | filters `[64, 64]`, kernel 3, pool 2, LSTM 64 | `config.py:1104-1107` | **yes**; B6 is the one place pooling is skipped, because 3 steps cannot be halved (`experiments_baseline6.py:164-167`) |
| Autoencoder | latent 15, hidden `[24]`, 100 epochs, batch 256, lr 1e-3, patience 10 | `config.py:956-972` | **yes** for every configuration that trains one |
| Scorer | `evaluation.evaluate` | called identically by all runners | **yes** |
| Test set | the same Step 3 test rows, in the same order | `load_feature_source` / `load_evaluation_inputs` | **yes** |
| Duplicate flags | `ev.duplicate_mask(X_train, X_test)` on the Step 3 matrices | nine call sites, all identical (G3) | **yes** |

Two honest caveats:

1. Approach 1 and Approach 2 were trained by the Step 8 runners
   (`models.run_step10`, `models_attention.run_step10b`), not by
   `train_classifier`. They read the same config constants, so the *values*
   agree; the code path differs.
2. The per-configuration values that the table reports (`seed`, `best epoch`,
   `training_seconds`) come from each run's own `metrics.json`
   (`comparison.py:1077-1115`), so a configuration trained before a config
   change would show its old value -- and nothing currently detects that. See
   F-a.

**NOT VERIFIED HERE:** the saved values. `training_setup_table()` renders them;
read it from the saved CSV:

```
python -c "import pandas as pd, config; print(pd.read_csv(config.STEP12_OUTPUT_DIR/'training_setup.csv').to_string())"
```

---

## F) Saved artifacts versus code

**NOT VERIFIED HERE, in full.** Model file dates, sizes and `metrics.json`
contents live under `C:\MSCAD2022\outputs_official\` and this container cannot
reach them. What I can report is the machinery that exists and the one piece
that does not.

**What exists.** Step 10 (notebook) fingerprints every file behind every
published number -- path, size, modified time, SHA-256 -- through
`step12_checks.provenance_table`, and `comparison.py:905-915` hashes each
configuration's model and `metrics.json`. `check_rows_unchanged`
(`comparison.py:352-360`) re-reads every finished configuration's metrics after
the section runs and compares at 1e-9, so a published number that moved is a
`[FAIL]` rather than a rounding difference. `experiments.py:300-337` does the
same for the Step 10 folder, and `experiments_baseline6.py:279-292` records the
encoder's and the PCA's digests. To print the whole provenance table on your
machine:

```
python -c "from src.mscad import step12_checks as s12; print(s12.provenance_table().to_string())"
python -c "import json,config; print(json.dumps(json.loads((config.STEP11_OUTPUT_DIR/'metrics.json').read_text()), indent=1)[:2000])"
```

**F-a, CONFIRMED GAP -- what does not exist.** Nothing in the repository hashes
*code*. Every SHA-256 in the tree is over an output file; `inspect.getsource`
appears nowhere. So the specific question you asked -- "whether the training code
at HEAD still matches what produced that model" -- **cannot be answered from the
saved artifacts today**. The nearest available substitute is indirect and
partial: `model_card.json` records the architecture constants that were in force
at training time (`models.py:513-519` -- `cnn_filters`, `cnn_kernel`,
`cnn_pool`, `lstm_units`, `dropout`, `dense_units`, `total_parameters`), so a
change to one of *those* can be detected by comparing the card against current
`config.py`. A change to the *logic* -- a different loss, a different ordering, a
changed builder -- cannot.

This matters concretely for this project because the code behind these models has
moved three times since some of them may have been trained: the Step 12 eight-file
refactor, the Baseline 6 `k = 3` change, and the official-dataset switch.

Minimum fix (Phase 2): a helper that hashes `inspect.getsource` of the builder
and the trainer, written into `metrics.json` and `model_card.json` at save time
and compared on load. Files: `src/mscad/step12_checks.py` (the helper),
`src/mscad/experiments.py` and `src/mscad/models.py` (record it), plus one test.
Note that this cannot retroactively validate the models already on disk -- it
only makes the question answerable from the next training run onwards.

---

## G) Metrics and reporting

### G1 -- the majority-class line

**Confirmed, and your reading is right: Brute_Force is the majority, not
Normal.** 88,502 / 128,799 = **0.68713**, which is the 0.6872 you see.

It is computed the correct way -- on the **test split**, by a predictor that
answers with the most frequent **training** class:

```
src/mscad/experiments_step12/report.py:220-229
    loaded = ev.load_evaluation_inputs(verbose=False)      # y_test
    predictions = s12.majority_class_predictions(y_train, len(y_test))
    headline = ev.headline_metrics(truth, predictions, None, n_classes)
```

The notebook passes that computed figure into the comparison
(`majority = report.majority_accuracy()`, then
`run_step12_complete(majority_accuracy=majority.get("accuracy"))`, notebook cell
`s12f-26`). It does **not** quote the stored constant
`OFFICIAL_QUICK_CHECK["majority class accuracy"]` (`config.py:2061`) here; that
constant is a reference figure in the Step 1 and Step 9 narrative only.

**Verdict: INFORMATIONAL (clean).**

### G2 -- the benign-vs-attack mapping

**Class 0 is `Normal`, and nothing else.** The mapping is read from the data,
not from a constant: `benign_codes_from` takes the encoded labels whose
`Label_binary == 0` (`evaluation.py:558-562`), and `Label_binary` is built in
Step 2 from `config.BENIGN_LABELS`, which official mode sets to `["Normal"]`
(`config.py:2089, 2111`). The official file has no `Benign` class, and the
config says so explicitly.

The binary decision is then (`evaluation.py:245-285`):

* `P(attack) = 1 - P(Normal)` -- derived from the multiclass probabilities, not
  from a separate head;
* predicted attack when `P(attack) >= 0.5`;
* `FPR = FP / (FP + TN)`, where the negatives are the `Normal` test rows alone;
* so **a binary false positive is a `Normal` flow called an attack**, and the
  FPR denominator is the number of `Normal` rows in the test set -- roughly
  5,700, not 25,760.

### G3 -- the duplicate-free definition

One definition, one implementation, every configuration:
`duplicate_mask(X_train, X_test)` marks a test row True when its **exact
float64 feature vector** also appears in training (`evaluation.py:291-307`),
compared on the **Step 3 full feature matrices** -- deliberately not on the
latent features, so that "two flows that differ only in a feature the
autoencoder discarded are NOT twins" (`evaluation.py:296-299`). All nine runners
call that same function on `loaded["X_train"], loaded["X_test"]` from
`load_evaluation_inputs` (call sites listed in finding G3-a).

**Verdict: INFORMATIONAL (clean, and comparable across rows).** Note the scale
of it: the published duplicate share for this dataset is 37.4%
(`config.py:2056`), so the duplicate-free column is computed on a much smaller
and harder subset, and that is the conservative number to quote.

### G4 -- parameter counts

**Measured on the Keras model, never estimated.** `experiments.py:443` is
`int(info.get("parameters", model.count_params()))`; `models_attention.py:1082`
is `int(model.count_params())`; the Step 8 model card takes
`architecture["parameters"].iloc[-1]`, the total row of the real layer table
(`models.py:518`).

### G5 -- inference ms/1k

Identical policy for all eight (`evaluation.py:358-385`), called from the one
`evaluate()`:

* **warm-up:** one batch, predicted and discarded, with the reason recorded --
  "the first call builds the graph and allocates buffers, so timing it would
  measure compilation rather than inference";
* **batch size:** `config.CLF_BATCH_SIZE` = 256 for every configuration;
* **runs:** 3, averaged, with `std_seconds` saved alongside;
* **unit:** seconds/row x 1e6 -> ms per 1,000 flows.

**G5-a, INFORMATIONAL.** The policy is uniform; the measurement is not
necessarily comparable. Each configuration was timed whenever it last ran, on
whatever load the machine had then. Quote `ms_per_1k` with its `std_seconds`, or
re-time all eight in one sitting -- re-timing loads the saved models and predicts,
so it needs no retraining and does not touch the lock.

---

## H) Approach 1 missing from this output root

### H1 -- where the trainer is, which cell runs it, and why the row is empty

**The trainer and its cell.** Approach 1 is trained by `models.run_step10`
(`models.py:603`), run from notebook cell `3bcde01f` (`step10 = models.run_step10()`,
notebook Step 8), and evaluated by `evaluation.run_step11` (`evaluation.py:776`),
which writes `outputs_official\step11\metrics.json` (notebook Step 9).

**Is the lock the explanation? No.** Two pieces of evidence rule it out. First,
the `TrainingLocked` message you saw named
`C:\MSCAD2022\outputs_official\step11\metrics.json` as *already existing* --
that message is produced by `training_lock.is_trained`, which is literally
`metrics_path(name).exists()` (`training_lock.py:89-92`). So Approach 1 **has**
been trained and evaluated in this output root. Second, the lock only ever
refuses a *retrain*: `would_train(name, overwrite)` returns False when the model
is already saved and `OVERWRITE_*` is off (`training_lock.py:104-116`), so a
load passes through.

**The actual cause is a lookup bug.** `comparison.result_path(key)` decides where
each row's `metrics.json` lives, and it carries **per-key special cases**:

```
src/mscad/comparison.py:78-100
    if key == APPROACH2_KEY:                      -> STEP11B_OUTPUT_DIR / "metrics.json"
    if key == BASELINE6_NAME:                     -> STEP12_B6_OUTPUT_DIR / "metrics.json"
    folder = usable_output_dir(STEP12_OUTPUT_DIR) -> step12\<key>\metrics.json
```

Approach 2 gets a branch because "Approach 2 is written by Step 11b rather than
by Step 12"; Baseline 6 gets one "for the same reason". **`"proposed"` gets no
branch, although exactly the same thing is true of it** -- its evaluation is
written by Step 9 to `step11\metrics.json`, not by the Step 10 runner to
`step12\proposed\metrics.json`.

So `build_full_comparison` (`comparison.py:287-295`) calls `load_result("proposed")`,
finds nothing at `step12\proposed\metrics.json`, appends the key to `missing`
and emits `_blank_row(label)` -- whose `input features` cell is
`config.NOT_TRAINED_TEXT` (`comparison.py:270-276`). That is the string you are
reading in the table.

The per-configuration file for Approach 1 does **not** close the gap, because it
is load-only by design: `experiments_step12/approach1.py:37-47` counts Approach 1
as finished when `ev.step11_outputs_exist()` is true, and `:67-76` reads Step 11's
metrics -- but it writes no `step12\proposed\metrics.json`. Notebook cell
`s12f-05` (`a1 = approach1.run()`) therefore reports Approach 1 correctly in its
own section while the eight-row table two cells later still shows it blank.

**H1-b, the second half of the same defect.** `NOT_TRAINED_TEXT` is the fixed
string `"not yet trained - run 10.2"` (`config.py:1567`), used for whichever row
is missing. Section 10.2 trains **Approach 2**; and since `25a0687` the notebook
calls that section 8.2. The correct instruction already exists in the codebase --
`which_step_produces("proposed")` returns `"Step 10 and Step 11, then Step 12"`
(`comparison.py:124-137`) and is printed in the note under the table -- it is
simply not used in the cell.

**Two ways to fill the row, for Phase 2 to choose between:**

1. *Without running anything* -- add the `"proposed"` branch to `result_path` so
   the row is read from `step11\metrics.json`. `_row_from_metrics` is
   `.get`-based (`comparison.py:237-267`), so every metric column fills
   correctly. But Step 9's metrics block has no `n_input_features`,
   `latent_dim`, `parameters`, `seed` or `best_epoch` (`evaluation.py:700-721`),
   so those four columns would show blank until `run_step11` records them the
   way `run_step11b` already does for Approach 2
   (`models_attention.py:1072-1093`). That is the second file in the fix.
2. *By re-scoring, without retraining* -- `experiments.run_experiment` for
   `"proposed"` loads the saved Step 5 encoder and the saved Step 8 model and
   only re-runs `evaluate()` (`experiments.py:386-411`, "reusing the trained
   Step 10 model (not retrained)"), then writes a full
   `step12\proposed\metrics.json` with all the descriptor fields. It trains
   nothing and the lock allows it. It does, however, write a new file into
   `outputs_official\step12\`, which Phase 1 forbids and which needs your
   approval.

I recommend option 1 plus the `run_step11` enrichment: it is two small edits, it
writes nothing into the output root, and it makes Approach 1 read the same way
Approach 2 and B6 already do.

---

## I) B3 beating Approach 1 and the AE configurations

You asked only whether anything gives B3 an unfair advantage. Checked, and the
answer is **no** -- with one structural advantage that is the ablation working as
intended rather than a flaw.

| Possible advantage | Finding | Evidence |
|---|---|---|
| Wider feature set | **Yes, by design.** B3 uses every Step 3 feature; Approach 1 and 2 use 30, then 15 latents | `config.py:1178, 1192-1194`; `experiments.py:195-197` |
| Different class weighting | No. Same `make_training_sets` on the same `y_train`, weights from the fit part only | `experiments.py:139-140`; `imbalance.py:182-215` |
| Different seed | No. `42` for all | `experiments.py:132-134`; `config.py:225, 812` |
| Different split | No. One saved stratified 80/20; same inner 10% validation split | `splitting.py:245-247`; `imbalance.py:190-215` |
| Different epochs / patience / batch / LR | No. All from the same constants | `config.py:1111-1119` |
| Different scorer or test set | No. The same `evaluate()` on the same test rows, the same duplicate mask | `experiments.py:421-424`; `evaluation.py:291-307` |
| No AE bottleneck | **Yes, by design.** `use_ae: False`, so the AE branch is skipped and `latent_dim` stays `None` | `experiments.py:386-399` |
| An extra architectural advantage | No. Same builder, same kernel, same filters, same LSTM width | `models.py:178-196` |

So B3's two advantages are precisely the two components the baseline exists to
remove: it keeps all the features (no selection) and it keeps them at full width
(no autoencoder). On a dataset where 37.4% of rows share a feature vector and
where a plain random forest reaches 0.8542 macro-F1 (`config.py:2060-2066`), a
model that sees the raw features directly having the advantage is a plausible
result, not a symptom. The honest reading for the thesis is that **on this
dataset the autoencoder bottleneck costs more information than it removes
noise** -- which is a finding, and a defensible one, provided the write-up states
it rather than ranking the proposal first.

Two things worth doing before you write that up, neither of which requires
retraining:

1. **Settle B1 against Approach 1 first.** B1 (no selection, with AE) and
   Approach 1 (selection, with AE) differ only in the selection stage -- and
   Approach 1's row is currently missing, so that comparison has not actually
   been made. Fix H1 and read it.
2. **Check the duplicate-free column, not the full one.** With 37.4% duplicates,
   the full-test macro-F1 rewards memorisation; `F1 macro (dup-free)` is where an
   information bottleneck should show its cost or its benefit. Those numbers are
   already saved.

**Verdict: INFORMATIONAL.** No change recommended. Nothing was changed.

---

## J) The high binary FPR (22%-47%)

Diagnostic, as you say, not a defect. Here is what can be established without
the artifacts, and exactly what to read.

**What the number means.** From G2: the negatives are the `Normal` test rows
alone, and the rule is `P(attack) >= 0.5`. With a stratified 80/20 split of
128,799 rows, the test set holds about 25,760 rows, of which `Normal` is
28,502 / 128,799 = 22.13%, so about **5,700 rows**. Applying the FPRs in your
table to that denominator:

| Config | binary FPR | Normal flows called "attack" (derived) |
|---|---|---|
| Approach 2 | 0.4693 | ~2,675 of ~5,700 |
| B1 | 0.2756 | ~1,571 |
| B2 | 0.3977 | ~2,267 |
| B3 | 0.2935 | ~1,673 |
| B4 | 0.2319 | ~1,322 |
| B5 | 0.2240 | ~1,277 |
| B6 | 0.2740 | ~1,562 |

**These are arithmetic from your own reported FPR and the published class share,
not read from `predictions.csv`.** They are consistent to within rounding, but
the saved counts are the ones to quote. The attack -> Normal direction cannot be
derived from an FPR at all -- it needs the recall -- so I have not estimated it.

**The three files that answer J properly**, all already on disk:

```
cd /d C:\Users\banot\MSCAD22-Multi-Stage-Cyber-Attack-Detection\Notebook
conda activate jupyfix

:: 1. the exact binary counts for any configuration (tn, fp, fn, tp are saved)
python -c "import json,config; m=json.loads((config.STEP11B_OUTPUT_DIR/'metrics.json').read_text()); print({k:m['binary'][k] for k in ('true_negatives','false_positives','false_negatives','true_positives','false_positive_rate','recall')})"

:: 2. the full 6x6 confusion counts -- the Normal row is Normal -> attack,
::    the Normal column is attack -> Normal
python -c "import pandas as pd, config; print(pd.read_csv(config.STEP11_OUTPUT_DIR/'confusion_matrix_counts.csv', index_col=0).to_string())"

:: 3. THE FLOOR, and the first thing to read: Step 1's label conflicts
python -c "import pandas as pd, config; p=config.STEP3_OUTPUT_DIR/'label_conflicts.csv'; print(pd.read_csv(p).to_string() if p.exists() else 'no label_conflicts.csv')"
```

**Why (3) is the important one.** Step 1 counts feature vectors that carry more
than one label (`official_dataset.py:420-478`) and writes both a summary and the
rows themselves. Any feature vector that appears under both `Normal` and an
attack label puts a **hard floor** under the binary FPR: no classifier seeing
only those features can get both rows right, so a fraction of `Normal` flows
must be called attacks by any model whatsoever. If that file shows
Normal/attack collisions, the 22%-47% range is partly a property of the dataset
and the thesis should quote the floor next to the measured FPR rather than
treating the whole of it as model error. If it shows none, the FPR is entirely
the models' and the threshold's, and the decision rule is the thing to discuss:
`P(attack) >= 0.5` is an arbitrary operating point, and Step 11's fusion already
demonstrates the alternative -- choose the threshold on the validation part for a
target FPR, then apply it unchanged to the test set (`fusion.py:1660-1668`).

Two further contributors worth naming in the write-up, both visible in the saved
files: the class weighting deliberately pushes the model towards the rare attack
classes and away from `Normal` (`imbalance.py:182-188`), which trades `Normal`
precision for rare-class recall; and 37.4% duplicate rows mean the full-test FPR
and the duplicate-free FPR can differ materially.

**Verdict: INFORMATIONAL.** No code change proposed.

---

## Correction plan

Phase 2 only, on your approval. Smallest change per item; no file outside this
list is touched, and nothing in this plan retrains a model unless it says so.

| # | Item | Files to change | The smallest change |
|---|---|---|---|
| 1 | **H1** -- Approach 1's row | `src/mscad/comparison.py` | Add a `"proposed"` branch to `result_path()` returning `STEP11_OUTPUT_DIR / STEP11_FILES["metrics"]`, beside the existing Approach 2 and Baseline 6 branches |
| 2 | **H1** -- the four blank descriptor columns | `src/mscad/evaluation.py` | In `run_step11`'s metrics dict, add `n_input_features`, `latent_dim`, `parameters`, `seed`, `best_epoch`, exactly as `models_attention.run_step11b` already records them. Re-running Step 9 is a load-and-score, not a retrain |
| 3 | **H1-b** -- the wrong instruction in the blank row | `src/mscad/comparison.py`, `config.py` | `_blank_row` takes the key and uses `which_step_produces(key)`; `NOT_TRAINED_TEXT` loses its "- run 10.2" suffix |
| 4 | **B1-a** -- Approach 2's input width | `src/mscad/models_attention.py` | One line: `"n_input_features": int(X_test_selected.shape[1])` (the loaded frame) instead of `int(config.N_SELECTED_FEATURES)` |
| 5 | **B3-a** -- "top 30" in the configuration table | `src/mscad/comparison.py` | In `configuration_table_8`, build the selection label from the row's own `n_input_features` (or from `selected_features.json`) rather than from `config.N_SELECTED_FEATURES` |
| 6 | **A2-a** -- the latent `Source_Capture` crash | `src/mscad/fusion.py`, `src/mscad/fusion_all.py` | Extend each existing condition to `and "Source_Capture" in meta_test.columns`, so temporal smoothing is skipped (with a note) instead of raising |
| 7 | **A1-b** -- `unknown:? -> unknown:?` in the alert report | `src/mscad/explainability.py` | Take the existing `"identifiers unavailable"` branch when `meta` has none of the identifier columns, not only when `meta is None` |
| 8 | **D2-a** -- B6's stale rationale | `config.py` | Correct the comment at `1843-1851` to say that 0.95 keeps 2 components on the own extraction and 6 on the official dataset. **Decision needed:** leave `B6_PCA_COMPONENTS = 3` and state the two-factor confound in notebook section 10.14, or set `"same_as_B4"` -- which requires unlocking and retraining B6 and is therefore a separate, explicitly approved step |
| 9 | **F-a** -- no code provenance | `src/mscad/step12_checks.py`, `src/mscad/experiments.py`, `src/mscad/models.py` | A helper hashing `inspect.getsource` of the builder and trainer; record it in `metrics.json` and `model_card.json` at save time; report it beside the file hashes. Applies from the next training run onward -- it cannot validate the models already saved |
| 10 | **C-a** -- the pre-split constant drop | notebook markdown only | One sentence in Step 2's "Why" block. Do **not** move the drop: it would change the feature set and invalidate every saved model |
| 11 | **DOC-a** -- the stale skip note | `config.py` | Reword `SKIP_STEPS_NOTE` so it matches the current numbering |
| 12 | **G5-a** -- comparable timings | none, or `src/mscad/comparison.py` | Either quote `ms_per_1k` with `std_seconds` in the table, or re-time all eight in one session (loads and predicts; no retraining) |

Tests to add alongside, in the existing style (`tmp_path` only, no pathlib
patching): one that `result_path("proposed")` points into the Step 11 folder and
that `build_full_comparison` fills the Approach 1 row from it; one that a blank
row names the right section for each of the eight keys; one that the fusion
smoothing path skips rather than raises when `Source_Capture` is absent; one that
every `n_input_features` writer reports a measured width.

Suggested order: 1, 2, 3, 4 (the comparison table becomes correct and complete),
then 6, 7 (the latent crash and the misleading report), then 5, 11, 10 (reporting
accuracy), then 9 (provenance), and 8 last, because it is the only one that needs
a decision from you rather than a patch.

---

## Do not touch

These must stay exactly as they are through Phase 2. Nothing in the correction
plan writes to any of them.

**Saved results -- the whole output root**

* `C:\MSCAD2022\outputs_official\` -- every folder `step3\` through `step14\`,
  and in particular every `metrics.json`, `predictions.csv`,
  `per_class_report.csv`, `confusion_matrix_counts.csv`,
  `confusion_matrix_rownorm.csv`, `history.csv`, `model_card.json`
* every trained model and fitted transform: `step7\autoencoder.keras`,
  `step7\encoder.keras`, `step10\cnn_lstm.keras`,
  `step11_attention\` (all of it), `step12\<config>\model.keras`,
  `step12\baseline4_ae_pca_lstm\pca.joblib`, `step12_baseline6\pca.joblib`,
  `step12\baseline5\`, `step13\`, `step14\`
* `step12\thesis\` -- the exported CSV and `.tex` tables
* `step3\MSCAD_merged.csv` and its reports (`label_conflicts.csv`,
  `column_quality.csv`, `duplicate_rows.csv`, `official_manifest.json`)

**The other dataset, entirely**

* `C:\MSCAD2022\outputs\` -- the own-extraction results. Not read, not written,
  not compared except by the read-only label table in notebook section 1.10
* `C:\MSCAD2022\CSV_Original_kaggle\MSCAD.csv` and `C:\MSCAD2022\CSVs\` --
  source data, read-only
* `C:\MSCAD2022\PCAPNG_ORIGINAL\`

**Code that must not move**

* `config.LOCK_TRAINED_MODELS = True` (`config.py:1903`) -- stays `True` for the
  whole of Phase 2; item 8 is the only thing that could need it off, and only
  with your explicit approval
* `config.RANDOM_SEED = 42`, `TEST_SIZE = 0.20`, `CLF_*`, `AE_*`, `CNN_KERNEL`,
  `PCA_N_COMPONENTS` -- changing any of them silently invalidates every saved
  model
* the frozen modules, byte-for-byte: `splitting.py`, `feature_selection.py`,
  `autoencoder.py`, `imbalance.py`, `fusion.py` and `fusion_all.py` apart from
  the one presence check in item 6, `experiments_pca.py`,
  `experiments_baseline5.py`, `experiments_baseline6.py`
* `tests/data/notebook_cell_identity.json` -- the recorded digests that prove the
  renumbering changed no code cell
* every notebook code cell from Step 2 onwards (the identity test enforces this)

**Everything else in the notebook**

* no cell is re-executed as part of Phase 2 except, if you approve item 2, the
  Step 9 cell -- which loads the saved model and re-scores it, and trains nothing

---

Awaiting approval. No files have been edited and no models have been retrained.
