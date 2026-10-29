r"""B4 at AE=64: all features (66) -> AE(66->64, no hidden) -> PCA -> 30
(fixed, replacing the 0.95 variance target) -> the paper LSTM (1024 ->
dropout 0.6 -> 512).  B4 keeps its plain-LSTM TYPE; sizes become the paper's."""
from __future__ import annotations

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

NAME = "baseline4_ae_pca_lstm"
LABEL = "B4: AE -> PCA -> LSTM  [AE=64, PCA 30, LSTM 1024/512]"


def train(verbose: bool = True) -> dict[str, Any]:
    place = c64.folder(NAME)
    Path(place).mkdir(parents=True, exist_ok=True)
    data = c64.inputs(verbose=verbose)
    X_train, X_test, dropped = c64.feature_frame_66()
    Z_train, Z_test, _enc = c64.encode_ae64(X_train, X_test, data["y_train"],
                                             place, verbose=verbose)
    P_train, P_test, _pca, pca_info, _var = c64.pca30(Z_train, Z_test, place,
                                                      verbose=verbose)
    n = int(np.asarray(data["y_train"]["Label_encoded"]).max() + 1)
    if data["label_mapping"]:
        n = max(n, len(ev.label_names_from_mapping(data["label_mapping"])))
    model, _hist, info = c64.train_paper_lstm(P_train, data["y_train"], n,
                                              verbose=verbose)
    desc = {"n_input_features": int(X_train.shape[1]), "dropped_column": dropped,
            "pca_components": int(pca_info["n_components"]),
            "classifier": "lstm 1024/512 (paper)"}
    return c64.evaluate_and_save(NAME, LABEL, model, P_test, data["y_test"],
                                 data["meta_test"], data["label_mapping"],
                                 data["duplicate_flags"], desc, info, place,
                                 verbose=verbose)


def run(verbose: bool = True) -> dict[str, Any]:
    return c64.run_one_ae64(NAME, train, verbose=verbose)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(0 if run().get("ok") else 1)
