r"""
tests/test_explainability.py -- pytest tests for STEP 14 (the explainability
layer).

NEW FILE.  No existing test file is modified.  Every model here is a few
Dense/Conv layers built inside the test, trained on nothing, so the whole module
runs in seconds on CPU.  Nothing is written outside tmp_path, and never
C:\\MSCAD2022.

No pathlib patching: the other operating system is simulated through
`config.running_on_windows`, and Windows paths are built with PureWindowsPath,
which works on every platform and has no mkdir.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import json
import os
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import explainability as xa

REAL_STEP14_DIR = config.STEP14_OUTPUT_DIR

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is not installed")

LABELS = {0: "Normal", 1: "Port_Scan", 2: "Brute_Force"}
BENIGN = [0]


# ===========================================================================
# tiny models, built here so no Step 7-12 output is needed
# ===========================================================================
@pytest.fixture(scope="module")
def linear_model():
    """A purely linear model: IG has a closed form, so it can be checked exactly."""
    keras.utils.set_random_seed(0)
    layer = keras.layers.Dense(3, activation=None, use_bias=True)
    model = keras.Sequential([keras.Input(shape=(4,)), layer])
    model.predict(np.zeros((1, 4), dtype="float32"), verbose=0)
    return model, layer.get_weights()[0]


@pytest.fixture(scope="module")
def classifier():
    """A small Conv1D + LSTM stack that emits LOGITS, like Step 10's."""
    keras.utils.set_random_seed(1)
    model = keras.Sequential([
        keras.Input(shape=(3, 1)),
        keras.layers.Conv1D(4, 2, padding="same", activation="relu"),
        keras.layers.LSTM(4),
        keras.layers.Dense(3, activation=None),
    ])
    model.predict(np.zeros((1, 3, 1), dtype="float32"), verbose=0)
    return model


@pytest.fixture(scope="module")
def encoder():
    """A 5 -> 3 encoder, like Step 7's."""
    keras.utils.set_random_seed(2)
    model = keras.Sequential([
        keras.Input(shape=(5,)),
        keras.layers.Dense(4, activation="relu"),
        keras.layers.Dense(3, activation="sigmoid"),
    ])
    model.predict(np.zeros((1, 5), dtype="float32"), verbose=0)
    return model


def frame(values, prefix="f"):
    values = np.asarray(values, dtype="float32")
    return pd.DataFrame(values, columns=["%s%d" % (prefix, i)
                                         for i in range(values.shape[1])])


# ===========================================================================
# 14.2  Integrated Gradients
# ===========================================================================
def test_ig_is_exact_for_a_linear_model(linear_model):
    """For F(x) = w.x + b, IG_i is exactly w_i (x_i - x'_i)."""
    model, weights = linear_model
    X = np.array([[0.2, 0.8, 0.5, 0.1], [0.9, 0.1, 0.5, 0.7]], dtype="float32")
    baseline = np.full(4, 0.5, dtype="float32")
    classes = np.array([0, 2])

    result = xa.integrated_gradients(model, X, baseline,
                                     target=xa.TARGET_CLASS,
                                     class_index=classes, steps=16)

    expected = (X - baseline) * weights[:, classes].T
    assert result["attributions"] == pytest.approx(expected, abs=1e-5)


def test_ig_satisfies_completeness(linear_model):
    """sum IG must equal F(x) - F(x'), which is the identity IG is built on."""
    model, _weights = linear_model
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, size=(6, 4)).astype("float32")
    baseline = np.full(4, 0.3, dtype="float32")
    classes = np.zeros(6, dtype="int64")

    result = xa.integrated_gradients(model, X, baseline,
                                     target=xa.TARGET_CLASS,
                                     class_index=classes, steps=32)

    assert np.abs(result["completeness_gap"]).max() < 1e-5
    table = xa.completeness_table(result, "linear", xa.TARGET_CLASS)
    assert bool(table["within_tolerance"].all())
    assert list(table.columns)[:2] == ["model", "target"]


def test_ig_is_zero_where_the_flow_equals_the_baseline(linear_model):
    model, _weights = linear_model
    baseline = np.array([0.5, 0.5, 0.5, 0.5], dtype="float32")
    X = np.array([[0.9, 0.5, 0.2, 0.5]], dtype="float32")   # features 1 and 3 match

    result = xa.integrated_gradients(model, X, baseline,
                                     target=xa.TARGET_CLASS,
                                     class_index=np.array([1]), steps=8)

    assert result["attributions"][0, 1] == pytest.approx(0.0, abs=1e-12)
    assert result["attributions"][0, 3] == pytest.approx(0.0, abs=1e-12)
    assert abs(result["attributions"][0, 0]) > 0


def test_ig_on_the_risk_target_attributes_one_minus_the_benign_mass(classifier):
    composed = xa.build_composed_model(classifier, n_features=3,
                                       mode="feature_seq")
    X = frame([[0.1, 0.9, 0.4], [0.7, 0.2, 0.8]])
    baseline = np.full(3, 0.5, dtype="float32")

    result = xa.integrated_gradients(composed, X, baseline,
                                     target=xa.TARGET_RISK,
                                     benign_codes=BENIGN, steps=32)

    probabilities = xa.composed_probabilities(composed, X)
    assert result["f_x"] == pytest.approx(1.0 - probabilities[:, 0], abs=1e-6)
    assert np.abs(result["completeness_gap"]).max() < 1e-3


def test_an_unknown_target_and_a_bad_step_count_are_refused(linear_model):
    model, _weights = linear_model
    X = np.zeros((1, 4), dtype="float32")
    with pytest.raises(xa.ExplainabilityError, match="target is"):
        xa.integrated_gradients(model, X, np.zeros(4, dtype="float32"),
                                target="saliency")
    with pytest.raises(xa.ExplainabilityError, match="IG_STEPS"):
        xa.integrated_gradients(model, X, np.zeros(4, dtype="float32"),
                                target=xa.TARGET_CLASS,
                                class_index=np.array([0]), steps=0)


def test_a_baseline_of_the_wrong_shape_is_refused(linear_model):
    model, _weights = linear_model
    with pytest.raises(xa.ExplainabilityError, match="baseline has shape"):
        xa.integrated_gradients(model, np.zeros((2, 4), dtype="float32"),
                                np.zeros(3, dtype="float32"),
                                target=xa.TARGET_CLASS,
                                class_index=np.zeros(2, dtype="int64"))


# ===========================================================================
# 14.2  the composed model
# ===========================================================================
def test_the_composed_model_reproduces_the_classifier(classifier, encoder):
    """encoder -> reshape -> classifier -> softmax must equal the saved path."""
    from src.mscad import models as md

    composed = xa.build_composed_model(classifier, encoder=encoder,
                                       n_features=5, mode="feature_seq")
    X = frame(np.random.default_rng(7).uniform(0, 1, size=(8, 5)))

    latent = encoder.predict(X.to_numpy(dtype="float32"), verbose=0)
    expected = md.predict_proba(classifier, md.as_feature_sequence(
        pd.DataFrame(latent)))

    assert xa.composed_probabilities(composed, X) == pytest.approx(expected,
                                                                   abs=1e-6)


def test_the_composed_model_output_sums_to_one(classifier):
    composed = xa.build_composed_model(classifier, n_features=3,
                                       mode="feature_seq")
    probabilities = xa.composed_probabilities(composed, frame(
        np.random.default_rng(8).uniform(0, 1, size=(5, 3))))
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(5), abs=1e-6)


def test_check_composed_model_passes_and_then_raises(classifier):
    composed = xa.build_composed_model(classifier, n_features=3,
                                       mode="feature_seq")
    X = frame(np.random.default_rng(9).uniform(0, 1, size=(4, 3)))
    reference = xa.composed_probabilities(composed, X)

    row = xa.check_composed_model(composed, X, reference, "tiny", verbose=False)
    assert row["result"] == "[OK]"

    wrong = reference.copy()
    wrong[0, 0] += 0.2
    with pytest.raises(xa.ExplainabilityError, match="does not reproduce"):
        xa.check_composed_model(composed, X, wrong, "tiny", verbose=False)


def test_a_time_window_composed_model_takes_a_window_of_features(encoder):
    """In time_window mode the encoder is applied to every step of the window."""
    keras.utils.set_random_seed(3)
    classifier = keras.Sequential([
        keras.Input(shape=(4, 3)),
        keras.layers.LSTM(3),
        keras.layers.Dense(3, activation=None)])
    composed = xa.build_composed_model(classifier, encoder=encoder,
                                       n_features=5, mode="time_window",
                                       window=4)

    assert tuple(composed.input_shape[1:]) == (4, 5)
    probabilities = xa.composed_probabilities(
        composed, np.random.default_rng(1).uniform(0, 1, size=(2, 4, 5)))
    assert probabilities.shape == (2, 3)


def test_an_unknown_input_mode_is_refused(classifier):
    with pytest.raises(xa.ExplainabilityError, match="INPUT_MODE"):
        xa.build_composed_model(classifier, n_features=3, mode="graph")


# ===========================================================================
# 14.2  the baseline
# ===========================================================================
def test_the_benign_median_baseline_uses_training_benign_rows_only():
    X_train = frame([[0.0, 1.0], [0.2, 1.0], [0.4, 1.0], [9.0, 9.0], [9.0, 9.0]])
    y_train = pd.DataFrame({"Label_encoded": [0, 0, 0, 1, 1],
                            "Label_binary": [0, 0, 0, 1, 1]})

    baseline = xa.baseline_vector(X_train, y_train, BENIGN,
                                  mode="benign_median")

    assert baseline == pytest.approx([0.2, 1.0])


def test_the_other_baselines_are_available_for_the_ablation():
    X_train = frame([[0.0, 1.0], [1.0, 3.0]])
    assert xa.baseline_vector(X_train, mode="zeros") == pytest.approx([0.0, 0.0])
    assert xa.baseline_vector(X_train, mode="train_mean") == \
        pytest.approx([0.5, 2.0])


def test_an_unknown_baseline_and_a_missing_benign_class_are_refused():
    X_train = frame([[0.0, 1.0], [1.0, 3.0]])
    with pytest.raises(xa.ExplainabilityError, match="IG_BASELINE"):
        xa.baseline_vector(X_train, mode="median_of_everything")
    attacks = pd.DataFrame({"Label_encoded": [1, 1], "Label_binary": [1, 1]})
    with pytest.raises(xa.ExplainabilityError, match="no benign training rows"):
        xa.baseline_vector(X_train, attacks, BENIGN, mode="benign_median")


# ===========================================================================
# 14.3  perturbation, agreement and the deletion check
# ===========================================================================
def test_perturbation_changes_exactly_one_feature():
    values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32")
    baseline = np.array([0.9, 0.9, 0.9], dtype="float32")

    changed = xa._replace_feature(values, 1, baseline)

    assert changed[:, 1] == pytest.approx([0.9, 0.9])
    assert changed[:, 0] == pytest.approx(values[:, 0])
    assert changed[:, 2] == pytest.approx(values[:, 2])
    assert values[0, 1] == pytest.approx(0.2)        # the input is not mutated


def test_perturbation_drops_stay_inside_the_probability_range(classifier):
    composed = xa.build_composed_model(classifier, n_features=3,
                                       mode="feature_seq")
    X = frame(np.random.default_rng(11).uniform(0, 1, size=(6, 3)))
    baseline = np.full(3, 0.5, dtype="float32")

    result = xa.perturbation_drops(composed, X, baseline, target=xa.TARGET_RISK,
                                   benign_codes=BENIGN)

    assert result["drops"].shape == (6, 3)
    assert (result["original"] >= 0).all() and (result["original"] <= 1).all()
    assert (np.abs(result["drops"]) <= 1.0 + 1e-9).all()


def test_the_deletion_check_compares_top_k_against_random_k(classifier):
    composed = xa.build_composed_model(classifier, n_features=3,
                                       mode="feature_seq")
    X = frame(np.random.default_rng(12).uniform(0, 1, size=(5, 3)))
    baseline = np.full(3, 0.5, dtype="float32")
    attributions = np.random.default_rng(13).normal(size=(5, 3))

    result = xa.deletion_check(composed, X, baseline, attributions, k=2,
                               target=xa.TARGET_RISK, benign_codes=BENIGN)

    assert result["k"] == 2
    assert np.isfinite(result["mean_drop_top_k"])
    assert np.isfinite(result["mean_drop_random_k"])
    assert result["margin"] == pytest.approx(
        result["mean_drop_top_k"] - result["mean_drop_random_k"])


# ===========================================================================
# the ranking helpers
# ===========================================================================
def test_spearman_on_a_hand_made_ranking():
    assert xa.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert xa.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert xa.spearman([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(0.8)


def test_spearman_returns_zero_rather_than_nan_for_a_constant_vector():
    assert xa.spearman([1, 1, 1, 1], [4, 3, 2, 1]) == 0.0
    assert xa.spearman([1.0], [2.0]) == 0.0


def test_spearman_refuses_mismatched_lengths():
    with pytest.raises(xa.ExplainabilityError, match="same length"):
        xa.spearman([1, 2, 3], [1, 2])


def test_ranks_average_ties():
    assert xa.rank_values([5, 1, 5, 3]) == pytest.approx([3.5, 1.0, 3.5, 2.0])


def test_top_k_and_its_overlap_are_computed_as_documented():
    assert xa.top_k([0.1, 0.9, 0.5, 0.7], 2) == [1, 3]
    assert xa.top_k_overlap([0.1, 0.9, 0.5, 0.7], [0.2, 0.8, 0.6, 0.4], 2) == \
        pytest.approx(0.5)
    assert xa.top_k_overlap([1, 2, 3], [1, 2, 3], 2) == pytest.approx(1.0)
    assert xa.top_k_overlap([1, 2, 3], [3, 2, 1], 1) == pytest.approx(0.0)


def test_attribution_agreement_is_one_row_per_flow():
    attributions = np.array([[0.5, 0.1, 0.9], [0.2, 0.8, 0.3]])
    drops = np.array([[0.5, 0.1, 0.9], [0.3, 0.8, 0.2]])

    table = xa.attribution_agreement(attributions, drops, k=2)

    assert len(table) == 2
    assert table.loc[0, "spearman"] == pytest.approx(1.0)
    assert table.loc[0, "top_k_overlap"] == pytest.approx(1.0)


# ===========================================================================
# 14.4  the fusion arithmetic
# ===========================================================================
def test_the_risk_is_the_sum_of_the_model_contributions():
    """Step 13's worked example: R ~ 0.838 and C_m must add up to exactly it."""
    probabilities = {
        "m1": np.array([[0.10, 0.85, 0.05]]),
        "m2": np.array([[0.40, 0.50, 0.10]]),
        "m3": np.array([[0.20, 0.30, 0.50]]),
    }
    weights = np.array([[0.745, 0.183, 0.072]])
    weights = weights / weights.sum()
    reliability = np.array([[0.98, 0.90, 0.80]])
    confidence = np.array([[0.528, 0.141, 0.063]])

    breakdown = xa.fusion_breakdown([0], ["m1", "m2", "m3"], probabilities,
                                    weights, reliability, confidence, BENIGN,
                                    LABELS)

    assert len(breakdown) == 3
    assert float(breakdown["C_m"].sum()) == pytest.approx(0.838, abs=1e-3)
    from src.mscad import fusion as fz
    fused = fz.fuse_probabilities([probabilities[m] for m in ("m1", "m2", "m3")],
                                  weights)
    assert float(breakdown["C_m"].sum()) == pytest.approx(
        float(fz.attack_risk(fused, BENIGN)[0]), abs=1e-9)


def test_fusion_consistency_flags_a_mismatch():
    breakdown = pd.DataFrame({"flow": [0, 0], "model": ["a", "b"],
                              "C_m": [0.4, 0.4]})
    good = xa.fusion_consistency(breakdown, np.array([0.8]), [0])
    bad = xa.fusion_consistency(breakdown, np.array([0.5]), [0])

    assert good["ok"] is True and good["result"] == "[OK]"
    assert bad["ok"] is False and bad["result"] == "[FAIL]"
    assert bad["max_abs_difference"] == pytest.approx(0.3)


def test_agreement_status_counts_votes():
    assert xa.agreement_status([1, 1, 1]) == "unanimous"
    assert xa.agreement_status([1, 1, 2]) == "majority"
    assert xa.agreement_status([0, 1, 2]) == "disagreement"
    assert xa.agreement_status([]) == "disagreement"


# ===========================================================================
# 14.1  the decision bands and tau_s
# ===========================================================================
def test_tau_s_is_chosen_from_validation_only_and_is_the_lowest_such_threshold():
    risk = np.array([0.10, 0.20, 0.55, 0.60, 0.95])
    truth = np.array([0, 0, 1, 1, 1])

    chosen = xa.choose_suspicious_threshold(risk, truth, target_recall=1.0)

    assert chosen["tau_s"] == pytest.approx(0.55)
    assert chosen["validation_recall"] == pytest.approx(1.0)
    assert chosen["chosen_on"].startswith("the validation part")
    # no test row can move it: the only inputs are the validation arrays
    assert xa.choose_suspicious_threshold(risk, truth,
                                          target_recall=1.0)["tau_s"] == \
        chosen["tau_s"]


def test_tau_s_relaxes_as_the_recall_target_drops():
    risk = np.array([0.1, 0.4, 0.6, 0.9])
    truth = np.array([0, 1, 1, 1])
    strict = xa.choose_suspicious_threshold(risk, truth, target_recall=1.0)
    loose = xa.choose_suspicious_threshold(risk, truth, target_recall=0.6)
    assert loose["tau_s"] >= strict["tau_s"]


def test_an_impossible_recall_target_and_an_empty_split_are_refused():
    with pytest.raises(xa.ExplainabilityError, match="empty validation set"):
        xa.choose_suspicious_threshold(np.array([]), np.array([]))
    with pytest.raises(xa.ExplainabilityError, match="no attack rows"):
        xa.choose_suspicious_threshold(np.array([0.5, 0.6]), np.array([0, 0]))


def test_tau_s_above_tau_is_reported_rather_than_hidden():
    risk = np.array([0.8, 0.9])
    truth = np.array([1, 1])
    chosen = xa.choose_suspicious_threshold(risk, truth, target_recall=1.0,
                                            tau=0.5)
    assert "note" in chosen and "empty" in chosen["note"]


def test_the_decision_bands_keep_step_13s_attack_rule():
    risk = np.array([0.05, 0.45, 0.55, 0.95])

    bands = xa.decision_band(risk, tau=0.5, tau_s=0.4, suspicious=True)

    assert list(bands) == [xa.BAND_BENIGN, xa.BAND_SUSPICIOUS, xa.BAND_ATTACK,
                           xa.BAND_ATTACK]
    # every flow Step 13 called an attack is still an attack
    assert list(xa.decision_band(risk, tau=0.5, tau_s=None, suspicious=False)) \
        == [xa.BAND_BENIGN, xa.BAND_BENIGN, xa.BAND_ATTACK, xa.BAND_ATTACK]


def test_the_suspicious_band_only_subdivides_what_step_13_called_benign():
    risk = np.random.default_rng(14).uniform(0, 1, size=200)
    with_band = xa.decision_band(risk, 0.6, 0.3, suspicious=True)
    without = xa.decision_band(risk, 0.6, None, suspicious=False)

    assert list(with_band[without == xa.BAND_ATTACK]) == \
        list(without[without == xa.BAND_ATTACK])
    assert set(with_band[without == xa.BAND_BENIGN]) <= {xa.BAND_BENIGN,
                                                         xa.BAND_SUSPICIOUS}


def test_flow_selection_is_deterministic_and_keeps_the_rare_classes():
    bands = np.array([xa.BAND_ATTACK] * 30 + [xa.BAND_BENIGN] * 10, dtype=object)
    fused = np.array([0] * 25 + [1] * 5 + [2] * 10)

    first = xa.select_flows(bands, fused, per_class=10, random_state=3,
                            benign_sample=2)
    again = xa.select_flows(bands, fused, per_class=10, random_state=3,
                            benign_sample=2)

    assert list(first) == list(again)                 # deterministic
    classes = fused[first]
    assert int((classes == 0).sum()) == 10            # capped
    assert int((classes == 1).sum()) == 5             # rare class kept whole
    assert len(first) == 10 + 5 + 2                   # plus the benign contrast


# ===========================================================================
# 14.5  the report
# ===========================================================================
def test_an_unmapped_label_is_never_guessed():
    assert xa.attack_stage("Port_Scan") == config.ATTACK_STAGE_MAP["Port_Scan"]
    assert xa.attack_stage("Cryptomining") == xa.UNMAPPED
    assert xa.attack_stage("", mapping={}) == xa.UNMAPPED


def test_the_stage_map_matches_the_pipelines_own_label_vocabulary():
    """Checked against CAPTURE_LABELS, which is where the labels come from.

    Both directions matter: a map key that is not a real class is dead weight,
    and a real attack class missing from the map would make every one of its
    alerts read "Unmapped".
    """
    classes = set(config.CAPTURE_LABELS.values())
    benign = set(config.BENIGN_LABELS) | {config.BENIGN_LABEL}

    unknown = [k for k in config.ATTACK_STAGE_MAP if k not in classes | benign]
    assert not unknown, "ATTACK_STAGE_MAP names unknown classes: %s" % unknown

    attacks = sorted(classes - benign)
    unmapped = [c for c in attacks if c not in config.ATTACK_STAGE_MAP]
    assert not unmapped, "these classes would report Unmapped: %s" % unmapped

    without_action = [c for c in attacks if c not in config.ACTION_TEMPLATES]
    assert not without_action, ("these classes have no suggested check: %s"
                                % without_action)
    for label in benign:
        assert config.ATTACK_STAGE_MAP.get(label) == "None"


def test_the_action_template_is_filled_with_the_real_addresses():
    text = xa.recommended_action("Brute_Force", "192.168.1.5", "10.0.0.9")

    assert text.startswith("Suggested analyst check: ")
    assert "192.168.1.5" in text and "10.0.0.9" in text
    assert "{src_ip}" not in text and "{dst_ip}" not in text


def test_a_class_without_a_template_still_gets_a_usable_line():
    text = xa.recommended_action("Cryptomining", "1.2.3.4", "5.6.7.8")
    assert "1.2.3.4" in text and "no action template" in text


def test_the_confidence_rules_are_exactly_as_documented():
    assert xa.confidence_label(0.95, "unanimous", True) == "High"
    # high probability but the models disagree -> not High
    assert xa.confidence_label(0.95, "majority", True) == "Medium"
    # high probability, unanimous, but the attributions do not add up -> not High
    assert xa.confidence_label(0.95, "unanimous", False) == "Medium"
    assert xa.confidence_label(0.75, "disagreement", True) == "Medium"
    assert xa.confidence_label(0.40, "majority", True) == "Medium"
    assert xa.confidence_label(0.40, "disagreement", False) == "Low"


def test_the_uncertainty_notes_name_every_reason_to_hesitate():
    notes = xa.uncertainty_notes(0.3, "disagreement", False,
                                 xa.BAND_SUSPICIOUS, training_rows=23)
    joined = " | ".join(notes)

    assert "low fused probability" in joined
    assert "disagreement" in joined
    assert "completeness" in joined
    assert "SUSPICIOUS" in joined
    assert "rare class" in joined
    assert xa.uncertainty_notes(0.99, "unanimous", True, xa.BAND_ATTACK,
                                training_rows=5000) == []


def test_the_top_feature_sentence_shows_direction_and_unscaled_values():
    attribution = np.array([0.9, -0.5, 0.01])
    names = ["Flow Duration", "Fwd Pkt Len Max", "Idle Min"]
    unscaled = pd.Series({"Flow Duration": 184.0, "Fwd Pkt Len Max": 1500.0,
                          "Idle Min": 0.0})

    text = xa.top_feature_sentence(attribution, names, unscaled, k=2)

    assert "Flow Duration = 184" in text
    assert "raises" in text and "lowers" in text
    assert "Idle Min" not in text                     # outside the top 2


def report_context(bands=None, risk=None):
    """A complete context dict for one hand-made alert."""
    probabilities = {
        "proposed": np.array([[0.05, 0.90, 0.05], [0.8, 0.1, 0.1]]),
        "baseline2_ae_lstm": np.array([[0.10, 0.80, 0.10], [0.7, 0.2, 0.1]]),
        "baseline3_cnn_lstm": np.array([[0.20, 0.70, 0.10], [0.6, 0.3, 0.1]]),
    }
    model_names = list(probabilities)
    weights = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    reliability = np.full((2, 3), 0.8)
    confidence = np.full((2, 3), 0.6)
    fused = sum(weights[:, i][:, None] * probabilities[m]
                for i, m in enumerate(model_names))
    risk = (1.0 - fused[:, 0]) if risk is None else np.asarray(risk)
    bands = ([xa.BAND_ATTACK, xa.BAND_BENIGN] if bands is None else bands)
    meta = pd.DataFrame({
        "Orig_Row": [501, 502], "Src IP": ["192.168.1.7", "192.168.1.7"],
        "Dst IP": ["10.0.0.5", "10.0.0.5"], "Src Port": [4444, 4445],
        "Dst Port": [22, 22], "Protocol": ["TCP", "TCP"],
        "Timestamp": ["2022-02-01 10:00:00", "2022-02-01 10:05:00"],
        "Source_Capture": ["W-B-01", "W-B-01"]})
    breakdown = xa.fusion_breakdown([0, 1], model_names, probabilities, weights,
                                    reliability, confidence, BENIGN, LABELS,
                                    orig_row=meta["Orig_Row"].to_numpy())
    return {
        "label_names": LABELS, "model_names": model_names,
        "breakdown": breakdown, "meta": meta, "tau": 0.5, "tau_s": 0.3,
        "risk": risk, "bands": np.array(bands, dtype=object),
        "fused_class": np.argmax(fused, axis=1),
        "fused_probabilities": fused,
        "completeness_ok": {0: True, 1: True},
        "attributions": {m: np.zeros((2, 3)) for m in model_names},
        "feature_names": {m: ["a", "b", "c"] for m in model_names},
        "unscaled_rows": {0: pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})},
        "context_labels": {0: "Port_Scan", 1: "Normal"},
        "context_label_column": np.array(["Port_Scan", "Normal"], dtype=object),
        "training_rows": {0: 5000, 1: 4000, 2: 23},
        "y_true": np.array([1, 0]),
    }


def test_the_report_carries_every_required_row_and_no_invented_one():
    report = xa.build_alert_report(0, 0, report_context())
    components = list(report["table"]["Evidence component"])

    for required in xa.REPORT_ROWS:
        assert required in components, required
    for forbidden in xa.FORBIDDEN_ROWS:
        assert not any(c.lower().startswith(forbidden.lower())
                       for c in components), forbidden
    assert report["alert_id"] == "A_501"
    assert set(report["table"]["Confidence"]) <= {"High", "Medium", "Low"}


def test_the_report_states_the_stage_as_a_mapping_not_a_progression():
    report = xa.build_alert_report(0, 0, report_context())
    stage = report["table"].set_index("Evidence component").loc[
        "Attack stage", "Detail"]

    assert "label-to-tactic mapping" in stage
    assert "not an inferred attack progression" in stage


def test_the_report_says_identifiers_are_not_model_features():
    report = xa.build_alert_report(0, 0, report_context())
    flow = report["table"].set_index("Evidence component").loc[
        "Flow context", "Detail"]

    assert "192.168.1.7" in flow and "10.0.0.5" in flow
    assert "never model features" in flow


def test_the_flat_record_keeps_the_risk_and_the_contributions_together():
    report = xa.build_alert_report(0, 0, report_context())
    flat = report["flat"]

    assert flat["R"] == pytest.approx(flat["sum_contributions"], abs=1e-6)
    assert flat["band"] == xa.BAND_ATTACK
    assert flat["tau"] == 0.5 and flat["tau_s"] == 0.3
    for required in xa.REPORT_ROWS:
        assert "row: %s" % required in flat


def test_the_plain_text_and_markdown_renderings_hold_the_same_rows():
    report = xa.build_alert_report(0, 0, report_context())
    text = xa.report_as_text(report)
    markdown = xa.examples_markdown([report])

    for required in xa.REPORT_ROWS:
        assert required in text
        assert required in markdown
    assert "A_501" in text and "A_501" in markdown


# ===========================================================================
# 14.5  source context
# ===========================================================================
def source_frame():
    return pd.DataFrame({
        "Orig_Row": [1, 2, 3, 4],
        "Src IP": ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.1"],
        "Timestamp_dt": pd.to_datetime([
            "2022-02-01 10:00:00", "2022-02-01 10:10:00",
            "2022-02-01 10:12:00", "2022-02-01 18:00:00"]),
    })


def test_source_context_lists_only_nearby_alerts_from_the_same_source():
    meta = source_frame()
    bands = np.array([xa.BAND_ATTACK] * 4, dtype=object)
    risk = np.array([0.9, 0.8, 0.7, 0.6])
    fused = np.array([1, 1, 2, 1])

    nearby = xa.source_context(meta, bands, risk, fused, LABELS, row=0,
                               window_minutes=60)

    assert list(nearby["Orig_Row"]) == [2]            # same IP, 10 minutes later
    assert 3 not in list(nearby["Orig_Row"])          # different source
    assert 4 not in list(nearby["Orig_Row"])          # 8 hours away


def test_source_context_excludes_the_alert_itself_and_benign_flows():
    meta = source_frame()
    bands = np.array([xa.BAND_ATTACK, xa.BAND_BENIGN, xa.BAND_ATTACK,
                      xa.BAND_ATTACK], dtype=object)
    nearby = xa.source_context(meta, bands, np.full(4, 0.5),
                               np.array([1, 1, 1, 1]), LABELS, row=0,
                               window_minutes=60)

    assert nearby.empty                               # row 2 is benign now


def test_source_context_is_ordered_in_time():
    meta = pd.DataFrame({
        "Orig_Row": [1, 2, 3],
        "Src IP": ["10.0.0.1"] * 3,
        "Timestamp_dt": pd.to_datetime(["2022-02-01 10:20:00",
                                        "2022-02-01 10:05:00",
                                        "2022-02-01 10:00:00"])})
    nearby = xa.source_context(meta, np.array([xa.BAND_ATTACK] * 3, dtype=object),
                               np.full(3, 0.5), np.array([1, 1, 1]), LABELS,
                               row=0, window_minutes=60)

    assert list(nearby["Orig_Row"]) == [3, 2]


# ===========================================================================
# the Windows-path guard -- nothing is ever written inside the repo
# ===========================================================================
def test_save_step14_refuses_a_windows_path_rather_than_making_junk(
        tmp_path, monkeypatch, capsys, assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP14_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step14"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    saved = xa.save_step14({"explained_rows": [], "n_alerts": 0, "tau": 0.5},
                           verbose=True)

    assert saved["status"] == "skipped_not_windows"
    assert saved["ok"] is False
    assert "[SKIP]" in capsys.readouterr().out
    assert_no_leaked_output("save_step14")


def test_load_step14_refuses_a_windows_path(tmp_path, monkeypatch,
                                            assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP14_OUTPUT_DIR",
                        PureWindowsPath(r"C:\MSCAD2022\outputs\step14"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    assert xa.step14_outputs_exist() is False
    with pytest.raises(xa.ExplainabilityError, match="not usable here"):
        xa.load_step14(verbose=False)
    assert_no_leaked_output("load_step14")


def test_step14_output_dir_is_under_data_root_not_the_repo():
    assert REAL_STEP14_DIR.parent.parent == config.DATA_ROOT
    assert config.PROJECT_ROOT not in REAL_STEP14_DIR.parents
    assert REAL_STEP14_DIR.name == "step14"


def test_a_pure_posix_path_is_not_mistaken_for_a_windows_one():
    """The guard is about drive letters, not about which OS is running."""
    assert config.is_windows_absolute(PureWindowsPath(r"C:\x")) is True
    assert config.is_windows_absolute(PurePosixPath("/tmp/x")) is False


def test_the_step14_config_block_holds_the_documented_defaults():
    assert config.IG_STEPS >= 1
    assert config.IG_BASELINE in ("benign_median", "zeros", "train_mean")
    assert config.PERTURB_MODE == "baseline_replace"
    assert 0 < config.COMPLETENESS_TOL < 1
    assert 0 < config.STABILITY_NOISE_STD < 1
    assert config.STABILITY_N_REPEATS >= 1
    assert 0 < config.SUSPICIOUS_RECALL_TARGET <= 1
    assert config.CONF_MEDIUM_PROB <= config.CONF_HIGH_PROB
    assert config.TOP_K_FEATURES >= 1
    assert config.EXPLAIN_RANDOM_STATE == config.RANDOM_STATE
    assert config.STEP14_OUTPUT_DIR in config.DIRS_TO_CREATE
    for template in config.ACTION_TEMPLATES.values():
        assert "{src_ip}" in template and "{dst_ip}" in template


# ===========================================================================
# the saved tables
# ===========================================================================
def test_the_explanations_frame_is_long_and_covers_both_targets():
    attributions = {"m1": {xa.TARGET_CLASS: np.array([[0.1, 0.2]]),
                           xa.TARGET_RISK: np.array([[0.3, 0.4]])}}
    frame_out = xa.explanations_frame([0], [77], attributions,
                                      {"m1": ["a", "b"]},
                                      {"m1": np.array([[0.05, 0.06]])})

    assert len(frame_out) == 4                        # 1 flow x 2 features x 2 targets
    assert set(frame_out["target"]) == set(xa.TARGETS)
    assert set(frame_out["Orig_Row"]) == {77}
    risk_rows = frame_out[frame_out["target"] == xa.TARGET_RISK]
    assert risk_rows["perturbation_drop"].notna().all()
    class_rows = frame_out[frame_out["target"] == xa.TARGET_CLASS]
    assert class_rows["perturbation_drop"].isna().all()
    assert int(risk_rows.loc[risk_rows["feature"] == "b", "abs_rank"].iloc[0]) == 1


def test_the_global_importance_table_reports_magnitude_and_direction():
    attributions = np.array([[0.5, -0.5], [0.5, -0.5]])
    table = xa.global_importance(attributions, ["a", "b"], "m1",
                                 xa.TARGET_RISK, predicted=np.array([1, 1]),
                                 label_names=LABELS)

    overall = table[table["scope"] == "all explained flows"].set_index("feature")
    assert overall.loc["a", "mean_signed_ig"] == pytest.approx(0.5)
    assert overall.loc["b", "mean_signed_ig"] == pytest.approx(-0.5)
    assert overall.loc["b", "mean_abs_ig"] == pytest.approx(0.5)
    assert "predicted: Port_Scan" in set(table["scope"])


def test_the_evidence_frame_names_the_band_and_the_truth():
    context = report_context()
    frame_out = xa.evidence_frame([0, 1], context)

    assert list(frame_out["Orig_Row"]) == [501, 502]
    assert list(frame_out["band"]) == [xa.BAND_ATTACK, xa.BAND_BENIGN]
    assert "Src IP" in frame_out.columns
    assert frame_out["correct"].dtype == bool
