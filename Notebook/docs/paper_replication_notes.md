# Paper replication: AE(66→64) → PCA(30) → LSTM(1024/512)

A faithful-as-possible replication of the AE+PCA+LSTM pipeline in *"Fusion of
linear and non-linear dimensionality reduction techniques for feature reduction
in LSTM-based Intrusion Detection System"*, for an apples-to-apples comparison
against this project's **B4** in the thesis.

It is **additive**: a dedicated output root
`C:\MSCAD2022\outputs_official_paper_replication\`, its own lock
`LOCK_TRAINED_MODELS_PAPER`, a context-manager redirect, and one configuration
file — the same architecture as `outputs_official_ae30\`. Nothing under
`outputs_official\` or `outputs_official_ae30\` is read for anything but the
shared dataset inputs, and nothing there is written. Approach 1, Approach 2 and
B1–B6 are untouched in code, config and saved outputs.

- Module: `src/mscad/experiments_step12_paper_replication/baseline_ae64_pca30_lstm.py`
- Every hyperparameter is the paper's, **hardcoded** in that file — none is read
  from the shared `config.AE_*` / `N_SELECTED_FEATURES` / classifier constants,
  so this variant can neither read nor move the AE=15 / AE=30 pipelines' state.
- The **evaluation** is reused unchanged (`evaluation.evaluate`), so the
  `metrics.json` matches the schema `comparison.py` reads, with the same
  macro-ROC-AUC (OvR), G-mean, binary-FPR and inference conventions as every
  other row.

## Judgment calls and deviations

**1. Feature count — 66, not 67. (Deviation from this codebase, faithful to the
paper.)** Step 4 (`INF_STRATEGY = "cap"`) adds a 67th column, `zero_duration`,
on top of the 66 `KAGGLE_FEATURES`; every existing baseline trains on 67. This
replication **drops `zero_duration`** and trains on 66, because its whole
purpose is to compare against the paper's *reported* numbers, the paper is
explicit about 66, and `zero_duration` is a project artifact the paper's authors
never had. **B4 keeps all 67** — that is precisely the difference this row
isolates.

**2. AE depth — single layer, no hidden. (Assumption; confirm against the paper
text.)** Read the paper's Algorithm 1 as a single-layer encoder and decoder:
`66 → 64 → 66`, no hidden layer (unlike this project's `AE_HIDDEN = [24]`).
Note that at 66 → 64 the "bottleneck" removes only two dimensions; that is what
the spec gives, implemented as written. **If the paper shows a staged encoder,
tell me and I will add the hidden layer(s).**

**3. AE output activation — ReLU bottleneck, sigmoid output. (Deviation, flagged
tension.)** The paper says ReLU. This project's Step 5 scaler is MinMax into
`[0, 1]`, where a **sigmoid** output reconstructs cleanly and an unbounded ReLU
output fits `[0, 1]` data poorly. Decision: ReLU on the encoder/bottleneck
(matches the paper's activation), **sigmoid on the reconstruction output**
(sound for the scaler). If the paper means a ReLU *output* layer, it implies a
different input scaling than this project uses, which would be a larger change
than a baseline should make silently.

**4. Classifier softmax — in the evaluator, not the model head. (Convention,
functionally identical.)** The LSTM emits **logits** (linear head); the softmax
is applied by `evaluation.predict_proba`, the one place this project applies it,
for every row. The probabilities and the argmax are identical to a model with a
Softmax output layer, so the paper's "Softmax output" is honoured; it just lives
in the shared evaluation layer so the metrics are computed the same way as every
other row.

**5. Sequence shape — 30 steps × 1 feature. (Convention.)** The 30 PCA
components are arranged as a `(30, 1)` sequence (`models.as_feature_sequence`,
`INPUT_MODE = "feature_seq"`), this project's convention for every LSTM row, so
the comparison against B4 is on the same input arrangement. The paper's "input
size = 30" is read as this.

**6. Classifier optimizer — Adam. (Assumption.)** The paper names Adam for the
AE and does not state the classifier's optimizer in the spec provided. Adam is
used for the classifier too, matching the AE and this project's default. Flag if
the paper specifies otherwise.

## Faithful to the paper, as specified

- AE bottleneck = 64; **no** hidden layer; MSE loss; Adam; batch **850**;
  **20 fixed epochs, no early stopping** (this project's AE otherwise uses
  `AE_PATIENCE` early stopping up to 100).
- PCA = exactly **30** components from the 64-dim bottleneck.
- LSTM = `LSTM(1024, tanh) → Dropout(0.6) → LSTM(512, tanh) → (softmax)`;
  batch **850**; **20 fixed epochs, no early stopping**.
- 6 classes (official MSCAD).

## Metrics saved (same fields as the existing rows)

accuracy, precision/recall/F1 (macro), F1 (weighted), ROC-AUC (macro OvR),
G-mean, binary FPR (the existing "attack probability = 1 − Σ benign-class
probabilities" convention, reused), the duplicate-free pair, training time, and
inference (ms per 1000 flows, CPU wall clock, 3 runs after one warm-up batch).

## How to run

Not run here, by request. In `jupyfix`, from `Notebook\`:

```
python -m src.mscad.experiments_step12_paper_replication.baseline_ae64_pca30_lstm
```

or from the notebook:

```python
from src.mscad.experiments_step12_paper_replication import module_for
module_for("baseline_ae64_pca30_lstm").run()
```

`LOCK_TRAINED_MODELS_PAPER = False` lets it train (it is False by default);
set it to `True` afterwards. The result lands in
`C:\MSCAD2022\outputs_official_paper_replication\paper_ae64_pca30_lstm\metrics.json`.
