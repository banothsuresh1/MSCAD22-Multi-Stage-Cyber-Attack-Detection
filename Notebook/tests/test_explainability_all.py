r"""
tests/test_explainability_all.py -- pytest tests for the Step 14 addendum
(explainability across every approach, baseline and fusion variant).

NEW FILE.  No existing test file, src function, config value or saved output is
modified by these tests.  The models here are tiny Keras models built in the
test itself -- nothing is loaded from disk, nothing is trained on real data, and
everything is written inside tmp_path.  Never C:\\MSCAD2022.

No pathlib patching: the other operating system is simulated through
`config.running_on_windows`, and Windows paths are built with PureWindowsPath.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import io
import json
import os
import re
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import explainability as xai
from src.mscad import explainability_all as xa

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")

LABELS = {0: "Normal", 1: "Port_Scan", 2: "Brute_Force"}
N_CLASSES = 3
BENIGN = [0]
KEYS = [k for k, _label in config.STEP12_DISPLAY_ORDER]
PROPOSED, APPROACH2, B1, B2, B3, B4, B5, B6 = KEYS

MODULE_PATH = (Path(__file__).resolve().parent.parent / "src" / "mscad"
               / "explainability_all.py")

# captured before any fixture redirects it, so the defaults test sees the real
# value from config.py rather than the per-test override
OUTPUT_DIR_AS_SHIPPED = config.STEP14ALL_OUTPUT_DIR


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    """Every folder this module could write to points inside tmp_path."""
    for name in ("STEP12_OUTPUT_DIR", "STEP13_OUTPUT_DIR", "STEP14_OUTPUT_DIR",
                 "STEP14ALL_OUTPUT_DIR", "STEP13ALL_OUTPUT_DIR",
                 "STEP10B_OUTPUT_DIR", "STEP11B_OUTPUT_DIR"):
        monkeypatch.setattr(config, name,
                            tmp_path / name.lower().replace("_output_dir", ""),
                            raising=False)
    monkeypatch.setattr(config, "IG_STEPS", 8)        # tiny but still exact
    monkeypatch.setattr(config, "IG_BATCH_SIZE", 32)
    monkeypatch.setattr(config, "TOP_K_FEATURES", 3)
    monkeypatch.setattr(config, "ALERT_FUSION_VARIANT", "auto")


# ===========================================================================
# tiny models, built here -- nothing is loaded and nothing is trained
# ===========================================================================
def tiny_encoder(n_features: int, latent: int = 4, seed: int = 0):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=(n_features,)),
                              keras.layers.Dense(latent, activation="relu")],
                             name="encoder")
    model.trainable = False
    return model


def tiny_classifier(width: int, seed: int = 1):
    """A (width, 1) sequence in, LOGITS out -- the shape Step 10 saves."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.Input(shape=(width, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(N_CLASSES, name="logits")], name="classifier")
    model.trainable = False
    return model


def frame(n_rows: int, n_features: int, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(0, 1, (n_rows, n_features)),
                        columns=["f%02d" % i for i in range(n_features)])


def reference_probabilities(classifier, values: np.ndarray) -> np.ndarray:
    """What the classifier itself says, through the Step 10 softmax."""
    from src.mscad import models as md
    return md.predict_proba(classifier, values[..., None])


# ===========================================================================
# 14.9.1  the composed models reproduce their classifier
# ===========================================================================
def test_an_encoder_plus_classifier_reproduces_the_classifier():
    X = frame(16, 6)
    encoder = tiny_encoder(6, latent=4)
    classifier = tiny_classifier(4)
    composed = xai.build_composed_model(classifier, encoder=encoder,
                                        n_features=6)

    mine = xai.composed_probabilities(composed, X)
    theirs = reference_probabilities(
        classifier, np.asarray(encoder.predict(X.to_numpy(dtype="float32"),
                                               verbose=0)))

    assert mine == pytest.approx(theirs, abs=1e-5)


def test_the_baseline4_pca_block_reproduces_encoder_then_pca():
    """B4's PCA is an affine map, so it belongs INSIDE the differentiated
    function -- and it has to compute exactly what sklearn computes."""
    from sklearn.decomposition import PCA

    X = frame(24, 6)
    encoder = tiny_encoder(6, latent=4)
    latent = np.asarray(encoder.predict(X.to_numpy(dtype="float32"), verbose=0))
    pca = PCA(n_components=3, random_state=0).fit(latent)

    block = xa.pca_as_linear_layer(encoder, pca)
    mine = np.asarray(block.predict(X.to_numpy(dtype="float32"), verbose=0),
                      dtype="float64")
    theirs = pca.transform(latent.astype("float64"))

    assert mine == pytest.approx(theirs, abs=1e-4)


def test_the_baseline4_composed_model_reproduces_its_classifier():
    from sklearn.decomposition import PCA

    X = frame(20, 6)
    encoder = tiny_encoder(6, latent=4)
    latent = np.asarray(encoder.predict(X.to_numpy(dtype="float32"), verbose=0))
    pca = PCA(n_components=3, random_state=0).fit(latent)
    classifier = tiny_classifier(3)
    block = xa.pca_as_linear_layer(encoder, pca)
    composed = xai.build_composed_model(classifier, encoder=block,
                                        n_features=6)

    mine = xai.composed_probabilities(composed, X)
    theirs = reference_probabilities(classifier,
                                     pca.transform(latent.astype("float64"))
                                     .astype("float32"))

    assert mine == pytest.approx(theirs, abs=1e-4)


def test_a_pca_that_does_not_match_its_encoder_is_refused():
    from sklearn.decomposition import PCA

    encoder = tiny_encoder(6, latent=4)
    wrong = PCA(n_components=2, random_state=0).fit(
        np.random.default_rng(0).uniform(0, 1, (10, 5)))

    with pytest.raises(xa.XaiAllError, match="was fitted on"):
        xa.pca_as_linear_layer(encoder, wrong)


def test_a_model_with_no_encoder_composes_too():
    """B3 has no autoencoder: the composed model is the classifier alone."""
    X = frame(12, 5)
    classifier = tiny_classifier(5)
    composed = xai.build_composed_model(classifier, encoder=None, n_features=5)

    assert xai.composed_probabilities(composed, X) == pytest.approx(
        reference_probabilities(classifier, X.to_numpy(dtype="float32")),
        abs=1e-5)


# ===========================================================================
# 14.9.2  Integrated Gradients
# ===========================================================================
@pytest.mark.parametrize("with_encoder", [True, False])
def test_ig_satisfies_completeness_for_every_model_type(with_encoder):
    """sum IG = F(x) - F(x') is the identity that makes IG worth reporting."""
    X = frame(8, 6, seed=4)
    encoder = tiny_encoder(6, latent=4) if with_encoder else None
    classifier = tiny_classifier(4 if with_encoder else 6)
    composed = xai.build_composed_model(classifier, encoder=encoder,
                                        n_features=6)
    baseline = np.full(6, 0.5, dtype="float32")

    result = xai.integrated_gradients(composed, X, baseline,
                                      target=xai.TARGET_RISK,
                                      benign_codes=BENIGN, steps=64)

    gap = np.abs(result["completeness_gap"])
    span = np.abs(result["f_x"] - result["f_baseline"])
    relative = np.where(span > 1e-9, gap / np.where(span > 1e-9, span, 1.0), 0.0)
    assert float(relative.max()) < 0.05


def test_ig_is_exactly_zero_where_the_flow_equals_the_baseline():
    """IG_i = (x_i - x'_i) * integral: a feature already at the baseline
    cannot be given any credit."""
    X = frame(6, 5, seed=5)
    baseline = np.asarray(X.iloc[0], dtype="float32")
    X.iloc[:, 2] = baseline[2]                       # one feature pinned
    classifier = tiny_classifier(5)
    composed = xai.build_composed_model(classifier, n_features=5)

    result = xai.integrated_gradients(composed, X, baseline,
                                      target=xai.TARGET_RISK,
                                      benign_codes=BENIGN, steps=8)

    assert np.abs(result["attributions"][:, 2]).max() == 0.0
    assert np.abs(result["attributions"][0]).max() == 0.0   # the baseline row


def test_the_pca_model_is_differentiable_end_to_end():
    """A gradient has to reach the ORIGINAL features through the PCA block."""
    from sklearn.decomposition import PCA

    X = frame(6, 6, seed=6)
    encoder = tiny_encoder(6, latent=4)
    latent = np.asarray(encoder.predict(X.to_numpy(dtype="float32"), verbose=0))
    pca = PCA(n_components=3, random_state=0).fit(latent)
    composed = xai.build_composed_model(
        tiny_classifier(3), encoder=xa.pca_as_linear_layer(encoder, pca),
        n_features=6)

    result = xai.integrated_gradients(composed, X, np.full(6, 0.5, "float32"),
                                      target=xai.TARGET_RISK,
                                      benign_codes=BENIGN, steps=16)

    assert result["attributions"].shape == (6, 6)
    assert np.isfinite(result["attributions"]).all()


# ===========================================================================
# 14.9.3  the cross-model comparison
# ===========================================================================
def importance_for(model: str, target: str, values: dict[str, float]
                   ) -> pd.DataFrame:
    return pd.DataFrame([
        {"model": model, "target": target, "scope": "all explained flows",
         "feature": feature, "mean_abs_ig": value, "mean_signed_ig": value,
         "flows": 10}
        for feature, value in values.items()])


def explained_stub(by_model: dict[str, dict[str, float]]) -> dict[str, dict]:
    return {key: {"importance": importance_for(xa.display_label(key),
                                               xai.TARGET_RISK, values),
                  "features": list(values)}
            for key, values in by_model.items()}


def test_a_feature_outside_a_models_input_is_na_not_zero():
    """Zero would say the model looked and found nothing; NA says it never
    saw the feature at all."""
    explained = explained_stub({
        PROPOSED: {"a": 0.5, "b": 0.2},
        B3: {"a": 0.1, "c": 0.9},
    })

    matrix = xa.importance_matrix(explained)

    assert np.isnan(matrix.loc["c", xa.display_label(PROPOSED)])
    assert np.isnan(matrix.loc["b", xa.display_label(B3)])
    assert matrix.loc["a", xa.display_label(PROPOSED)] == 0.5
    assert (matrix.fillna(-1) != 0).all().all()       # no silent zeros


def test_the_cross_model_spearman_is_computed_on_shared_features_only():
    """b and c are not shared, so only a, d and e may enter the correlation."""
    explained = explained_stub({
        PROPOSED: {"a": 1.0, "d": 2.0, "e": 3.0, "b": 9.0},
        B2: {"a": 1.0, "d": 2.0, "e": 3.0, "c": 0.1},
    })

    spearman, overlap = xa.explanation_agreement(explained_stub({
        PROPOSED: {"a": 1.0, "d": 2.0, "e": 3.0, "b": 9.0},
        B2: {"a": 1.0, "d": 2.0, "e": 3.0, "c": 0.1}})
        and xa.importance_matrix(explained))

    left, right = xa.display_label(PROPOSED), xa.display_label(B2)
    assert spearman.loc[left, right] == pytest.approx(1.0)
    assert overlap.loc[left, right] == pytest.approx(1.0)
    assert spearman.loc[left, left] == pytest.approx(1.0)


def test_a_reversed_ranking_gives_a_negative_spearman():
    matrix = xa.importance_matrix(explained_stub({
        PROPOSED: {"a": 3.0, "b": 2.0, "c": 1.0},
        B2: {"a": 1.0, "b": 2.0, "c": 3.0}}))

    spearman, _overlap = xa.explanation_agreement(matrix)

    assert spearman.loc[xa.display_label(PROPOSED),
                        xa.display_label(B2)] == pytest.approx(-1.0)


def test_the_top_k_overlap_counts_the_shared_top_features():
    matrix = xa.importance_matrix(explained_stub({
        PROPOSED: {"a": 9.0, "b": 8.0, "c": 1.0, "d": 0.5},
        B2: {"a": 9.0, "b": 0.1, "c": 7.0, "d": 0.5}}))

    _spearman, overlap = xa.explanation_agreement(matrix, k=2)

    # the top 2 are {a, b} and {a, c}: one of two shared
    assert overlap.loc[xa.display_label(PROPOSED),
                       xa.display_label(B2)] == pytest.approx(0.5)


def test_pairs_with_fewer_than_two_shared_features_stay_na():
    matrix = xa.importance_matrix(explained_stub({
        PROPOSED: {"a": 1.0, "b": 2.0},
        B3: {"c": 1.0, "d": 2.0}}))

    spearman, _overlap = xa.explanation_agreement(matrix)

    assert np.isnan(spearman.loc[xa.display_label(PROPOSED),
                                 xa.display_label(B3)])
    assert xa.shared_feature_counts(matrix).loc[
        xa.display_label(PROPOSED), xa.display_label(B3)] == 0


def test_consensus_features_count_only_the_models_that_can_see_them():
    matrix = xa.importance_matrix(explained_stub({
        PROPOSED: {"shared": 9.0, "mine": 8.0},
        B2: {"shared": 9.0, "theirs": 7.0},
        B3: {"shared": 9.0}}))

    consensus = xa.consensus_features(matrix, k=2)

    row = consensus[consensus["feature"] == "shared"].iloc[0]
    assert row["in top-2 of"] == 3
    assert row["models that can see it"] == 3
    assert row["share of those models"] == pytest.approx(1.0)
    # a feature only one model can see is in one model's top-k and no more
    alone = consensus[consensus["feature"] == "mine"].iloc[0]
    assert alone["models that can see it"] == 1
    assert alone["only this model"] == xa.display_label(PROPOSED)
    # and at k = 1 only the shared feature survives at all
    assert list(xa.consensus_features(matrix, k=1)["feature"]) == ["shared"]


# ===========================================================================
# 14.9.4  R = sum C_m, for every variant
# ===========================================================================
def hand_made_fusion(keys, weights, risks):
    """(breakdown, matrices, rows) from weights and risks given by hand."""
    weights = np.asarray(weights, dtype="float64")
    risks = np.asarray(risks, dtype="float64")
    n_rows = weights.shape[0]
    probabilities = {}
    for index, key in enumerate(keys):
        probs = np.zeros((n_rows, N_CLASSES))
        probs[:, 0] = 1.0 - risks[:, index]           # benign mass
        probs[:, 1] = risks[:, index]
        probabilities[xa.short_label(key)] = probs
    risk = (weights * risks).sum(axis=1)
    matrices = {"weights": weights,
                "reliability": np.full(weights.shape, np.nan),
                "confidence": np.full(weights.shape, np.nan),
                "risk": risk}
    rows = np.arange(n_rows, dtype="int64")
    breakdown = xai.fusion_breakdown(
        rows, [xa.short_label(k) for k in keys], probabilities,
        weights, matrices["reliability"], matrices["confidence"], BENIGN,
        LABELS, orig_row=rows)
    return breakdown, matrices, rows


@pytest.mark.parametrize("tag,keys", [
    ("F3", [PROPOSED, B2, B3]),
    ("FALL", KEYS),
    ("FSEL", [APPROACH2, B2, B4]),
])
def test_the_risk_is_exactly_the_sum_of_the_contributions(tag, keys):
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.1, 1.0, (12, len(keys)))
    weights = raw / raw.sum(axis=1, keepdims=True)
    risks = rng.uniform(0, 1, (12, len(keys)))

    breakdown, matrices, rows = hand_made_fusion(keys, weights, risks)
    consistency = xai.fusion_consistency(breakdown, matrices["risk"], rows)

    assert consistency["ok"]
    assert consistency["max_abs_difference"] < 1e-6
    totals = breakdown.groupby("flow")["C_m"].sum().to_numpy()
    assert totals == pytest.approx(matrices["risk"], abs=1e-6)


def test_a_tampered_contribution_breaks_the_identity():
    keys = [PROPOSED, B2]
    weights = np.array([[0.4, 0.6], [0.5, 0.5]])
    risks = np.array([[0.9, 0.1], [0.2, 0.8]])
    breakdown, matrices, rows = hand_made_fusion(keys, weights, risks)

    breakdown.loc[0, "C_m"] = float(breakdown.loc[0, "C_m"]) + 0.01
    consistency = xai.fusion_consistency(breakdown, matrices["risk"], rows)

    assert not consistency["ok"]
    assert consistency["result"] == "[FAIL]"


def test_the_dominant_model_is_the_largest_contribution_not_the_loudest():
    keys = [PROPOSED, B2]
    # B2 is more certain, but Approach 1 carries almost all the weight
    weights = np.array([[0.95, 0.05]])
    risks = np.array([[0.6, 0.99]])
    breakdown, matrices, rows = hand_made_fusion(keys, weights, risks)

    share = xa.dominant_model_share(breakdown, {"label_names": LABELS},
                                    np.array([1]), rows)

    overall = share[share["scope"] == "all explained alerts"]
    winner = overall.loc[overall["share (%)"].idxmax()]
    assert winner["model"] == xa.short_label(PROPOSED)
    assert winner["share (%)"] == pytest.approx(100.0)


# ===========================================================================
# 14.9.5  the alert variant is a VALIDATION decision
# ===========================================================================
def test_auto_takes_the_best_validation_macro_f1():
    variant, why = xa.choose_alert_variant({"F3": 0.71, "FALL": 0.68,
                                            "FSEL": 0.74})

    assert variant == "FSEL"
    assert "VALIDATION" in why and "0.7400" in why


def test_auto_ignores_a_variant_that_is_not_available():
    variant, _why = xa.choose_alert_variant({"F3": 0.71,
                                             "FALL": float("nan")})

    assert variant == "F3"


def test_a_named_variant_is_used_as_given(monkeypatch):
    monkeypatch.setattr(config, "ALERT_FUSION_VARIANT", "F3")

    variant, why = xa.choose_alert_variant({"F3": 0.10, "FSEL": 0.99})

    assert variant == "F3"
    assert "whatever its validation score" in why


def test_a_named_variant_that_is_not_available_is_refused(monkeypatch):
    monkeypatch.setattr(config, "ALERT_FUSION_VARIANT", "FSEL")

    with pytest.raises(xa.XaiAllError, match="not available"):
        xa.choose_alert_variant({"F3": 0.5})


def test_no_variant_at_all_is_refused():
    with pytest.raises(xa.XaiAllError, match="no fusion variant"):
        xa.choose_alert_variant({"F3": float("nan")})


def test_the_variant_choice_cannot_see_the_test_set(monkeypatch, tmp_path):
    """`validation_macro_f1` reads 13.8's saved VALIDATION score, or computes
    one from validation probabilities; neither path takes a test argument."""
    saved = {"metrics": {"variants": {
        "FALL": {"validation_scores": {"macro_f1": 0.80}},
        "FSEL": {"validation_scores": {"macro_f1": 0.91}}}}}

    first = {tag: xa.validation_macro_f1(tag, [PROPOSED], {}, {}, saved)
             for tag in ("FALL", "FSEL")}
    chosen_first = xa.choose_alert_variant(first)[0]

    # a completely different test set cannot reach this decision at all
    second = {tag: xa.validation_macro_f1(tag, [PROPOSED], {}, {}, saved)
              for tag in ("FALL", "FSEL")}

    assert first == second
    assert chosen_first == xa.choose_alert_variant(second)[0] == "FSEL"
    assert "y_test" not in xa.validation_macro_f1.__code__.co_varnames


def test_the_other_models_row_names_only_models_outside_the_fusion():
    probabilities = {key: np.tile([0.2, 0.7, 0.1], (3, 1)) for key in KEYS}

    row = xa.other_models_row(0, KEYS, [PROPOSED, B2], probabilities, 0, LABELS)

    assert row["Evidence component"] == "Other models (not fused)"
    assert xa.short_label(B4) in row["Detail"]
    assert xa.short_label(PROPOSED) not in row["Detail"]
    assert "Port_Scan" in row["Detail"]
    for key in KEYS:                              # never an internal key
        assert key not in row["Detail"]


def test_every_available_model_fused_says_so():
    probabilities = {PROPOSED: np.tile([0.9, 0.05, 0.05], (2, 1))}

    row = xa.other_models_row(0, [PROPOSED], [PROPOSED], probabilities, 0,
                              LABELS)

    assert row["Detail"] == "every available model is in this fusion"


# ===========================================================================
# a missing model is skipped, and the rest are still explained
# ===========================================================================
def test_a_missing_model_is_skipped_with_the_cells_that_produce_it(monkeypatch,
                                                                   capsys):
    def loader(key, verbose=False):
        if key == APPROACH2:
            raise xa.XaiAllError("no saved Approach 2 model at "
                                 "step10_attention")
        X = frame(6, 4)
        return {"name": key, "composed": None, "classifier": None,
                "encoder": None, "X_train": X, "X_test": X,
                "features": list(X.columns), "saved_probabilities": None,
                "model_path": "stub"}
    monkeypatch.setattr(xa, "load_composed_any", loader)

    bundles, availability, checks = xa.available_models(verbose=True)
    printed = capsys.readouterr().out

    assert APPROACH2 not in bundles
    assert len(bundles) == len(KEYS) - 1
    skipped = availability[~availability["available"]]
    assert list(skipped["key"]) == [APPROACH2]
    note = skipped.iloc[0]["note"]
    assert "Approach 2" in note and "10.2" in note
    assert "[SKIP]" in printed


def test_a_composed_model_that_does_not_reproduce_its_own_output_is_dropped(
        monkeypatch):
    """Attributions from such a model would explain something else."""
    X = frame(8, 4)
    classifier = tiny_classifier(4)
    composed = xai.build_composed_model(classifier, n_features=4)

    def loader(key, verbose=False):
        wrong = np.tile([0.9, 0.05, 0.05], (len(X), 1))   # not what it predicts
        return {"name": key, "composed": composed, "classifier": classifier,
                "encoder": None, "X_train": X, "X_test": X,
                "features": list(X.columns), "saved_probabilities": wrong,
                "model_path": "stub"}
    monkeypatch.setattr(xa, "load_composed_any", loader)

    with pytest.raises(xa.XaiAllError, match="nothing to explain"):
        xa.available_models(verbose=False)


def test_everything_missing_is_an_error_rather_than_a_guess(monkeypatch):
    def loader(key, verbose=False):
        raise xa.XaiAllError("nothing on disk")
    monkeypatch.setattr(xa, "load_composed_any", loader)

    with pytest.raises(xa.XaiAllError, match="nothing to explain"):
        xa.available_models(verbose=False)


# ===========================================================================
# the existing Step 14 outputs are never touched
# ===========================================================================
def fake_step14(tmp_path):
    """A minimal `step14\\` folder, as Step 14 would have left it."""
    folder = Path(config.STEP14_OUTPUT_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / config.STEP14_FILES["metrics"]).write_text(json.dumps({
        "step": 14, "models": list(config.FUSION_MODELS),
        "explained_flows": 120}), encoding="utf-8")
    (folder / config.STEP14_FILES["alerts"]).write_text(
        "alert_id,Orig_Row,R\nA_1,1,0.9\n", encoding="utf-8")
    (folder / config.STEP14_FILES["explanations"]).write_text(
        "Orig_Row,test_row,model,target,feature,ig\n1,0,m,attack_risk,f,0.1\n",
        encoding="utf-8")
    return folder


def test_the_step14_fingerprint_covers_every_file(tmp_path):
    folder = fake_step14(tmp_path)

    before = xa.step14_fingerprint()

    assert set(before) == {p.name for p in folder.glob("*")}
    assert all(len(v) == 64 for v in before.values())


def test_the_addendum_leaves_step14_byte_identical(tmp_path, capsys):
    fake_step14(tmp_path)
    before = xa.step14_fingerprint()

    Path(config.STEP14ALL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (Path(config.STEP14ALL_OUTPUT_DIR)
     / config.STEP14ALL_FILES["quality"]).write_text("model\n", encoding="utf-8")

    table, ok = xa.assert_step14_unchanged(before, verbose=True)
    printed = capsys.readouterr().out

    assert ok
    assert set(table["result"]) == {"[OK]"}
    assert "[OK]  existing Step 14 outputs unchanged" in printed
    # the two files the brief names are marked as such
    named = table[table["named in the brief"]]
    assert set(named["file"]) == {config.STEP14_FILES["metrics"],
                                  config.STEP14_FILES["alerts"]}


@pytest.mark.parametrize("key", ["metrics", "alerts"])
def test_a_changed_step14_file_is_caught(tmp_path, key):
    folder = fake_step14(tmp_path)
    before = xa.step14_fingerprint()

    (folder / config.STEP14_FILES[key]).write_text("{}", encoding="utf-8")

    table, ok = xa.assert_step14_unchanged(before, verbose=False)

    assert not ok
    failed = table[table["result"] == "[FAIL]"]
    assert list(failed["file"]) == [config.STEP14_FILES[key]]


def test_a_deleted_step14_file_is_caught(tmp_path):
    folder = fake_step14(tmp_path)
    before = xa.step14_fingerprint()

    (folder / config.STEP14_FILES["alerts"]).unlink()

    _table, ok = xa.assert_step14_unchanged(before, verbose=False)

    assert not ok


def test_the_explained_flows_are_the_ones_step_14_explained(tmp_path):
    """14.9.3 is a comparison only if the same flows are explained."""
    meta = pd.DataFrame({"Orig_Row": [100, 200, 300, 400]})
    step14 = {"explanations": pd.DataFrame({"Orig_Row": [300, 100, 300]})}

    rows = xa.step14_explained_rows(step14, meta, len(meta))

    assert list(rows) == [0, 2]                  # positions of 100 and 300


def test_no_step14_explanations_means_no_rows_rather_than_a_guess():
    assert list(xa.step14_explained_rows({}, None, 10)) == []


# ===========================================================================
# presentation: display labels, colours, the verification table
# ===========================================================================
def test_every_model_keeps_one_colour_and_it_comes_from_config():
    colours = {key: xa.model_colour(key) for key in KEYS}

    assert len(set(colours.values())) == len(KEYS)
    assert colours == dict(config.MODEL_COLOURS)
    assert all(re.fullmatch(r"#[0-9a-fA-F]{6}", value)
               for value in colours.values())
    assert xa.model_colour(B5) == colours[B5]     # stable across calls


def test_the_labels_never_leak_an_internal_key():
    for key in KEYS:
        assert xa.display_label(key) != key
        assert key not in xa.display_label(key)
        assert key not in xa.short_label(key)
    assert xa.short_label(PROPOSED) == "Approach 1"
    assert xa.short_label(B4) == "B4"


def test_a_rendered_table_shows_na_as_a_dash_and_keeps_the_caption(capsys):
    frame_in = pd.DataFrame({"model": [xa.display_label(PROPOSED),
                                       xa.display_label(B4)],
                             "completeness (%)": [99.5, np.nan],
                             "ms per flow": [3.0, 1.0]})

    xa.present(frame_in, "A caption.", best=["completeness (%)", "ms per flow"],
               lower_is_better=["ms per flow"],
               percent_columns=["completeness (%)"], na_footnote=True)
    printed = capsys.readouterr().out

    assert xa.NA_TEXT in printed
    assert "A caption." in printed
    assert xa.NA_FOOTNOTE in printed
    assert "Lower is better for: ms per flow" in printed
    assert "*99.5*" in printed                    # the best value, marked
    assert "*1.0000*" in printed


def test_an_empty_table_says_so_rather_than_rendering_nothing(capsys):
    xa.present(pd.DataFrame(), "Nothing here.")

    assert "nothing to show" in capsys.readouterr().out


def test_the_verification_table_carries_every_check_with_a_result():
    checks = xa.Checks()
    checks.add("14.9.1 composed models", "a check that passes", True, "detail")
    checks.add("14.9.2 attribution", "a check that fails", False, "why")
    checks.add("14.9.2 attribution", "a check that was skipped", None, "why")

    table = checks.frame()

    assert list(table.columns) == ["section", "check", "result", "detail"]
    assert list(table["result"]) == ["[OK]", "[FAIL]", "[SKIP]"]
    assert len(checks.frame("14.9.2 attribution")) == 2
    assert checks.ok is False
    assert all(table["detail"].str.len() > 0)


def test_a_skipped_check_does_not_fail_the_run():
    checks = xa.Checks()
    checks.add("s", "ok", True, "d")
    checks.add("s", "skipped", None, "d")

    assert checks.ok is True


# ===========================================================================
# the Keras 2 requirement, and the guards
# ===========================================================================
def test_the_module_never_imports_the_keras_3_only_ops_namespace():
    """The models were saved with Keras 2; `keras.ops` is Keras 3 only."""
    source = io.open(MODULE_PATH, encoding="utf-8").read()

    for pattern in (r"\bfrom\s+keras\s+import\s+(?:[\w\s,]*\b)?ops\b",
                    r"\bkeras\s*\.\s*ops\b", r"\bimport\s+keras\.ops\b"):
        assert re.search(pattern, source) is None, pattern


def test_no_explainability_library_is_imported():
    """IG is six lines of arithmetic on tf.GradientTape; nothing else is used."""
    source = io.open(MODULE_PATH, encoding="utf-8").read()

    for library in ("shap", "lime", "captum", "alibi", "eli5"):
        assert not re.search(r"\bimport\s+%s\b" % library, source)
        assert not re.search(r"\bfrom\s+%s\b" % library, source)


def test_the_addendum_refuses_a_windows_path(tmp_path, monkeypatch, capsys,
                                             assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP14ALL_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step14_all"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    saved = xa.save_step14all({"step14_before": {}}, xa.Checks(), verbose=False)

    assert saved["status"] == "skipped_not_windows"
    assert saved["ok"] is False
    assert xa.step14all_outputs_exist() is False
    capsys.readouterr()
    assert_no_leaked_output("save_step14all")


def test_the_output_dir_is_under_the_data_root_not_the_repo():
    assert config.PROJECT_ROOT not in OUTPUT_DIR_AS_SHIPPED.parents
    assert OUTPUT_DIR_AS_SHIPPED.name == "step14_all"
    assert OUTPUT_DIR_AS_SHIPPED.parent == config.OUTPUTS_DIR
    # and it is NOT the Step 14 folder: nothing here writes beside Step 14
    assert OUTPUT_DIR_AS_SHIPPED != config.STEP14_OUTPUT_DIR


def test_the_configured_keys_are_the_display_order_keys():
    assert list(config.XAI_ALL_MODELS) == KEYS
    assert xa.model_keys() == KEYS
    with pytest.raises(xa.XaiAllError, match="not keys of"):
        xa.model_keys(["approach2_cnn_attention_lstm"])


def test_a_variant_without_validation_probabilities_shows_no_reliability():
    """Better a blank reliability than a number the fusion never used."""
    keys = [PROPOSED, B2]
    frame_in = pd.DataFrame({
        "w_FALL_%s" % xa.short_label(PROPOSED): [0.4, 0.5],
        "w_FALL_%s" % xa.short_label(B2): [0.6, 0.5],
        "r_%s" % xa.short_label(PROPOSED): [0.9, 0.2],
        "r_%s" % xa.short_label(B2): [0.1, 0.8],
        "R_FALL": [0.42, 0.5],
        "P_FALL_Normal": [0.58, 0.5], "P_FALL_Port_Scan": [0.3, 0.3],
        "P_FALL_Brute_Force": [0.12, 0.2]})

    matrices = xa.variant_matrices("FALL", keys, frame_in,
                                   [LABELS[c] for c in range(N_CLASSES)])

    assert np.isnan(matrices["reliability"]).all()
    assert np.isnan(matrices["weights_recovered_gap"])
    assert matrices["weights"].shape == (2, 2)
    assert matrices["risk"] == pytest.approx([0.42, 0.5])


def test_the_weight_recovery_tolerance_is_stated_and_loose_enough():
    """The composed models agree with the saved probabilities to 1e-5, so a
    recomputed weight cannot be expected to match to 1e-6."""
    assert xa.WEIGHT_RECOVERY_TOL > 1e-5
    assert xa.WEIGHT_RECOVERY_TOL <= 1e-2
