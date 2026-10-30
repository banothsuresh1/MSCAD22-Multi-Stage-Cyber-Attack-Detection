r"""B2 at AE=64: all features (66) -> AE(66->64, no hidden) -> the paper LSTM
(1024 -> dropout 0.6 -> 512, tanh, 20 fixed epochs, batch 850).  B2 keeps its
plain-LSTM classifier TYPE; its layer sizes/regime become the paper's."""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402

from . import common_ae64 as c64  # noqa: E402
from .. import evaluation as ev  # noqa: E402

NAME = "baseline2_ae_lstm"
LABEL = "B2: AE -> LSTM (no CNN)  [AE=64, LSTM 1024/512]"


def _n_classes(y_train, label_mapping):
    n = int(np.asarray(y_train["Label_encoded"]).max() + 1)
    if label_mapping:
        n = max(n, len(ev.label_names_from_mapping(label_mapping)))
    return n


def train(verbose: bool = True) -> dict[str, Any]:
    place = c64.folder(NAME)
    Path(place).mkdir(parents=True, exist_ok=True)
    data = c64.inputs(verbose=verbose)
    X_train, X_test, dropped = c64.feature_frame_66()
    n_feat = int(X_train.shape[1])
    Z_train, Z_test, _enc = c64.encode_ae64(X_train, X_test, data["y_train"],
                                             place, verbose=verbose)
    del X_train, X_test, _enc          # dead after encoding; free before the fit
    gc.collect()
    n_classes = _n_classes(data["y_train"], data["label_mapping"])
    model, _hist, info = c64.train_paper_lstm(Z_train, data["y_train"],
                                              n_classes, verbose=verbose)
    desc = {"n_input_features": n_feat, "dropped_column": dropped,
            "pca_components": None, "classifier": "lstm 1024/512 (paper)"}
    return c64.evaluate_and_save(NAME, LABEL, model, Z_test, data["y_test"],
                                 data["meta_test"], data["label_mapping"],
                                 data["duplicate_flags"], desc, info, place,
                                 verbose=verbose)


def run(verbose: bool = True) -> dict[str, Any]:
    return c64.run_one_ae64(NAME, train, verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
