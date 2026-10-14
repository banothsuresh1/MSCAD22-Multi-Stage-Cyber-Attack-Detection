"""
tests/test_fusion.py -- pytest tests for STEP 13 (adaptive evidence fusion).

NEW FILE.  No existing test file is modified.  Everything here runs on tiny
hand-made probability matrices: no model is built, nothing is trained, no
TensorFlow is needed and nothing is written outside tmp_path.  Never
C:\\MSCAD2022.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pytest tests -v
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.mscad import fusion as fz

REAL_STEP13_DIR = config.STEP13_OUTPUT_DIR

# The spec's worked example: three models, three classes, one flow.
M1 = [0.10, 0.85, 0.05]
M2 = [0.40, 0.50, 0.10]
M3 = [0.20, 0.30, 0.50]
REL = [0.98, 0.90, 0.80]


def one_flow(*rows):
    """Each model's single-flow probability matrix, shaped (1, C)."""
    return [np.array([row], dtype="float64") for row in rows]


# ===========================================================================
# the worked example in the notebook markdown
# ===========================================================================
def test_the_worked_example_matches_the_published_numbers():
    probs = one_flow(M1, M2, M3)
    confidence = np.column_stack([
        fz.normalised_entropy_confidence(p, enabled=True) for p in probs])
    reliability = np.array([REL], dtype="float64")
    weights = fz.dynamic_weights(reliability, confidence, mode="proportional")
    fused = fz.fuse_probabilities(probs, weights)
    risk = fz.attack_risk(fused, benign_codes=[0])

    assert confidence[0] == pytest.approx([0.528, 0.141, 0.063], abs=1e-3)
    assert weights[0] == pytest.approx([0.745, 0.183, 0.072], abs=1e-3)
    assert fused[0] == pytest.approx([0.162, 0.746, 0.092], abs=1e-3)
    assert risk[0] == pytest.approx(0.838, abs=1e-3)


def test_the_worked_example_goes_through_fuse_once_the_same_way():
    """fuse_once is what the pipeline calls, so it must agree with the example."""
    probs = one_flow(M1, M2, M3)
    names = ["m1", "m2", "m3"]
    lookup = {(name, 1): rel for name, rel in zip(names, REL)}
    macro = {name: rel for name, rel in zip(names, REL)}

    fused, weights, reliability, confidence = fz.fuse_once(
        probs, names, lookup, macro, np.array([1]), mode="proportional")

    assert reliability[0] == pytest.approx(REL)
    assert confidence[0] == pytest.approx([0.528, 0.141, 0.063], abs=1e-3)
    assert weights[0] == pytest.approx([0.745, 0.183, 0.072], abs=1e-3)
    assert fused[0] == pytest.approx([0.162, 0.746, 0.092], abs=1e-3)


# ===========================================================================
# the weights
# ===========================================================================
@pytest.mark.parametrize("mode", ["proportional", "softmax"])
def test_weights_are_non_negative_and_sum_to_one(mode):
    rng = np.random.default_rng(11)
    reliability = rng.uniform(0.0, 1.0, size=(50, 3))
    confidence = rng.uniform(0.0, 1.0, size=(50, 3))

    weights = fz.dynamic_weights(reliability, confidence, mode=mode,
                                 temperature=0.2)

    assert weights.shape == (50, 3)
    assert (weights >= 0).all()
    assert np.abs(weights.sum(axis=1) - 1.0).max() < 1e-12


def test_a_higher_reliability_in_a_context_earns_a_higher_weight():
    confidence = np.array([[0.5, 0.5]], dtype="float64")
    weights = fz.dynamic_weights(np.array([[0.9, 0.3]]), confidence,
                                 mode="proportional")

    assert weights[0, 0] > weights[0, 1]
    assert weights[0] == pytest.approx([0.75, 0.25])


def test_equal_reliability_and_confidence_give_equal_weights():
    reliability = np.full((7, 3), 0.8)
    confidence = np.full((7, 3), 0.4)

    for mode in ("proportional", "softmax"):
        weights = fz.dynamic_weights(reliability, confidence, mode=mode,
                                     temperature=0.2)
        assert weights == pytest.approx(np.full((7, 3), 1.0 / 3))


def test_all_scores_zero_falls_back_to_equal_weights():
    """The EPSILON guard: no evidence either way means no model is preferred."""
    reliability = np.zeros((4, 3))
    confidence = np.zeros((4, 3))

    for mode in ("proportional", "softmax"):
        weights = fz.dynamic_weights(reliability, confidence, mode=mode,
                                     temperature=0.2, epsilon=1e-6)
        assert weights == pytest.approx(np.full((4, 3), 1.0 / 3))
        assert np.abs(weights.sum(axis=1) - 1.0).max() < 1e-12


def test_a_score_sum_below_epsilon_also_falls_back():
    reliability = np.array([[1e-9, 1e-9, 1e-9], [0.9, 0.5, 0.2]])
    confidence = np.ones((2, 3))

    weights = fz.dynamic_weights(reliability, confidence, mode="proportional",
                                 epsilon=1e-6)

    assert weights[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert weights[1] == pytest.approx(np.array([0.9, 0.5, 0.2]) / 1.6)


def test_a_cold_softmax_concentrates_on_the_best_model():
    reliability = np.array([[0.9, 0.5, 0.1]])
    confidence = np.ones((1, 3))

    cold = fz.dynamic_weights(reliability, confidence, mode="softmax",
                              temperature=0.05)
    warm = fz.dynamic_weights(reliability, confidence, mode="softmax",
                              temperature=5.0)

    assert cold[0, 0] > 0.99
    assert warm[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=0.05)


def test_an_unknown_weight_mode_and_a_bad_temperature_are_refused():
    reliability = np.full((2, 2), 0.5)
    with pytest.raises(fz.FusionError, match="WEIGHT_MODE"):
        fz.dynamic_weights(reliability, reliability, mode="average")
    with pytest.raises(fz.FusionError, match="WEIGHT_TEMPERATURE"):
        fz.dynamic_weights(reliability, reliability, mode="softmax",
                           temperature=0.0)


def test_mismatched_reliability_and_confidence_shapes_are_refused():
    with pytest.raises(fz.FusionError, match="same shape"):
        fz.dynamic_weights(np.zeros((3, 2)), np.zeros((3, 3)))


# ===========================================================================
# the confidence term
# ===========================================================================
def test_confidence_is_one_for_a_one_hot_distribution_and_zero_for_a_uniform_one():
    one_hot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    uniform = np.full((2, 3), 1.0 / 3)

    assert fz.normalised_entropy_confidence(one_hot, enabled=True) == \
        pytest.approx([1.0, 1.0])
    assert fz.normalised_entropy_confidence(uniform, enabled=True) == \
        pytest.approx([0.0, 0.0], abs=1e-12)


def test_confidence_sits_between_the_two_extremes_and_zero_log_zero_is_safe():
    probs = np.array([[0.5, 0.5, 0.0]])
    value = fz.normalised_entropy_confidence(probs, enabled=True)[0]

    assert np.isfinite(value)
    assert 0.0 < value < 1.0
    assert value == pytest.approx(1.0 - math.log(2) / math.log(3))


def test_turning_the_confidence_term_off_makes_it_exactly_one():
    probs = np.array([[0.34, 0.33, 0.33], [0.9, 0.05, 0.05]])
    assert fz.normalised_entropy_confidence(probs, enabled=False) == \
        pytest.approx([1.0, 1.0])


def test_without_the_confidence_term_the_weights_are_the_reliabilities():
    probs = one_flow(M1, M2, M3)
    names = ["m1", "m2", "m3"]
    lookup = {(name, 1): rel for name, rel in zip(names, REL)}
    macro = dict(zip(names, REL))

    _fused, weights, _rel, confidence = fz.fuse_once(
        probs, names, lookup, macro, np.array([1]), confidence=False)

    assert confidence[0] == pytest.approx([1.0, 1.0, 1.0])
    assert weights[0] == pytest.approx(np.array(REL) / sum(REL))


# ===========================================================================
# the shrinkage
# ===========================================================================
def test_an_unseen_context_falls_back_to_the_global_macro_f1():
    assert fz.shrunk_reliability(0.1, 0.77, n_context=0, k=20) == \
        pytest.approx(0.77)


def test_a_large_context_is_almost_its_own_f1():
    assert fz.shrunk_reliability(0.40, 0.90, n_context=100_000, k=20) == \
        pytest.approx(0.40, abs=1e-3)


def test_at_k_rows_the_context_and_the_global_figure_weigh_the_same():
    assert fz.shrunk_reliability(0.20, 0.80, n_context=20, k=20) == \
        pytest.approx(0.50)


def test_shrinkage_moves_monotonically_with_the_number_of_rows():
    values = [fz.shrunk_reliability(0.2, 0.9, n, k=20) for n in (1, 5, 20, 200)]
    assert values == sorted(values, reverse=True)
    assert all(0.2 <= v <= 0.9 for v in values)


# ===========================================================================
# the fused distribution and the risk
# ===========================================================================
def test_the_fused_rows_sum_to_one_and_the_risk_is_one_minus_the_benign_mass():
    rng = np.random.default_rng(3)
    probs = []
    for _ in range(3):
        raw = rng.uniform(0.05, 1.0, size=(40, 4))
        probs.append(raw / raw.sum(axis=1, keepdims=True))
    weights = fz.dynamic_weights(rng.uniform(0.2, 1.0, size=(40, 3)),
                                 rng.uniform(0.2, 1.0, size=(40, 3)))

    fused = fz.fuse_probabilities(probs, weights)
    risk = fz.attack_risk(fused, benign_codes=[0])

    assert np.abs(fused.sum(axis=1) - 1.0).max() < 1e-12
    assert risk == pytest.approx(1.0 - fused[:, 0])
    assert (risk >= 0).all() and (risk <= 1).all()


def test_several_benign_classes_are_all_subtracted():
    fused = np.array([[0.3, 0.2, 0.4, 0.1]])
    assert fz.attack_risk(fused, [0, 1])[0] == pytest.approx(0.5)
    assert fz.attack_risk(fused, [0])[0] == pytest.approx(0.7)


def test_no_benign_class_means_every_flow_is_pure_risk():
    fused = np.array([[0.6, 0.4], [0.1, 0.9]])
    assert fz.attack_risk(fused, []) == pytest.approx([1.0, 1.0])


def test_fuse_probabilities_refuses_a_transposed_weight_matrix():
    probs = [np.full((5, 3), 1 / 3), np.full((5, 3), 1 / 3)]
    with pytest.raises(fz.FusionError, match=r"\(5, 2\)"):
        fz.fuse_probabilities(probs, np.full((2, 5), 0.5))


# ===========================================================================
# the context
# ===========================================================================
def test_the_consensus_context_is_the_argmax_of_the_mean_probabilities():
    m1 = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    m2 = np.array([[0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    m3 = np.array([[0.1, 0.8, 0.1], [0.6, 0.3, 0.1]])

    assert fz.consensus_context([m1, m2, m3]).tolist() == [1, 2]


def test_the_consensus_context_never_looks_at_a_label():
    """It is computed from probabilities alone, so it exists at test time."""
    probs = [np.array([[0.9, 0.1]]), np.array([[0.8, 0.2]])]
    assert fz.consensus_context(probs).tolist() == [0]


def test_dropping_the_context_gives_every_flow_the_global_reliability():
    lookup = {("m1", 0): 0.2, ("m1", 1): 0.9}
    macro = {"m1": 0.55}
    contexts = np.array([0, 1, 0])

    with_context = fz.reliability_matrix(lookup, macro, ["m1"], contexts)
    without = fz.reliability_matrix(lookup, macro, ["m1"], contexts,
                                    use_context=False)

    assert with_context[:, 0] == pytest.approx([0.2, 0.9, 0.2])
    assert without[:, 0] == pytest.approx([0.55, 0.55, 0.55])


def test_an_unseen_context_key_at_test_time_uses_the_global_figure():
    lookup = {("m1", 0): 0.2}
    macro = {"m1": 0.55}
    matrix = fz.reliability_matrix(lookup, macro, ["m1"], np.array([0, 9]))
    assert matrix[:, 0] == pytest.approx([0.2, 0.55])


def test_the_consensus_context_set_covers_validation_and_test_together():
    val = [np.array([[0.8, 0.2]]), np.array([[0.7, 0.3]])]
    test = [np.array([[0.1, 0.9], [0.9, 0.1]]),
            np.array([[0.2, 0.8], [0.8, 0.2]])]

    validation, testing, names = fz.context_sets(
        "consensus_class", val, test, pd.DataFrame({"Label_encoded": [0, 1]}),
        None, {0: "Normal", 1: "Attack"}, 2)

    assert validation.tolist() == [0]
    assert testing.tolist() == [1, 0]
    assert names == {0: "Normal", 1: "Attack"}


def test_the_protocol_context_is_skipped_rather_than_guessed(tmp_path, monkeypatch):
    """No Protocol column on disk means the ablation is skipped, not invented."""
    monkeypatch.setattr(config, "STEP5_OUTPUT_DIR", tmp_path / "step5")
    probs = [np.array([[0.8, 0.2]]), np.array([[0.7, 0.3]])]

    validation, testing, names = fz.context_sets(
        "protocol", probs, probs, pd.DataFrame({"Label_encoded": [0]}),
        pd.DataFrame({"Orig_Row": [1]}), {0: "Normal", 1: "Attack"}, 2)

    assert (validation, testing, names) == (None, None, None)
    assert fz.protocol_contexts(pd.DataFrame({"Label_encoded": [0]}), None) == \
        (None, None)


def test_an_unknown_context_mode_is_refused():
    probs = [np.array([[0.8, 0.2]])]
    with pytest.raises(fz.FusionError, match="CONTEXT_MODE"):
        fz.context_sets("protocol_and_port", probs, probs,
                        pd.DataFrame({"Label_encoded": [0]}), None,
                        {0: "Normal"}, 2)


def test_the_validation_positions_are_the_step_9_validation_rows(monkeypatch):
    monkeypatch.setattr(config, "CLF_VAL_SIZE", 0.25)
    y_train = pd.DataFrame({
        "Label_encoded": np.resize([0, 1, 2], 60),
        "Label_binary": np.resize([0, 1, 1], 60)})

    positions = fz.validation_positions(y_train)

    assert len(positions) == 15
    assert len(set(positions.tolist())) == 15
    assert positions.max() < len(y_train)
    # a pure function of (y_train, CLF_VAL_SIZE, RANDOM_STATE): same rows again
    assert fz.validation_positions(y_train).tolist() == positions.tolist()


# ===========================================================================
# build_reliability, on validation rows only
# ===========================================================================
def test_build_reliability_uses_the_class_f1_in_consensus_class_mode():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect = np.eye(2)[y_true]
    contexts = y_true.copy()
    names = {0: "Normal", 1: "Attack"}

    table, info = fz.build_reliability({"m1": perfect}, y_true, contexts, names,
                                       n_classes=2, verbose=False)

    assert info["macro"]["m1"] == pytest.approx(1.0)
    assert set(table["context"]) == {"Normal", "Attack"}
    assert table["reliability"].to_numpy() == pytest.approx([1.0, 1.0])
    assert table["n_validation_rows"].sum() == len(y_true)


def test_build_reliability_shrinks_a_context_with_few_rows():
    y_true = np.array([0] * 20 + [1])
    predicted = y_true.copy()
    predicted[-1] = 0                     # the single class-1 row is missed
    probs = np.eye(2)[predicted]
    names = {0: "Normal", 1: "Attack"}

    table, info = fz.build_reliability({"m1": probs}, y_true, y_true, names,
                                       n_classes=2, shrinkage=20, verbose=False)

    attack = table[table["context"] == "Attack"].iloc[0]
    assert attack["n_validation_rows"] == 1
    assert attack["f1_in_context"] == pytest.approx(0.0)
    # (1 * 0 + 20 * macro) / 21 -- far above the raw 0.0
    assert attack["reliability"] == pytest.approx(
        20 * info["macro"]["m1"] / 21, abs=1e-6)
    assert attack["reliability"] > 0.4


def test_an_unknown_reliability_metric_is_refused():
    with pytest.raises(fz.FusionError, match="RELIABILITY_METRIC"):
        fz.per_class_f1([0, 1], [0, 1], 2, metric="recall")


def test_the_reliability_metric_can_be_precision():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    precision = fz.per_class_f1(y_true, y_pred, 2, metric="precision")
    assert precision == pytest.approx([1.0, 2 / 3])


# ===========================================================================
# the fixed-weight baselines
# ===========================================================================
def test_the_simple_average_is_the_plain_mean():
    probs = one_flow(M1, M2, M3)
    average = fz.simple_average(probs)
    assert average[0] == pytest.approx(
        [(0.10 + 0.40 + 0.20) / 3, (0.85 + 0.50 + 0.30) / 3,
         (0.05 + 0.10 + 0.50) / 3])
    assert np.abs(average.sum(axis=1) - 1.0).max() < 1e-12


def test_the_majority_vote_counts_argmax_votes():
    m1 = np.array([[0.1, 0.8, 0.1]])      # votes class 1
    m2 = np.array([[0.2, 0.7, 0.1]])      # votes class 1
    m3 = np.array([[0.6, 0.2, 0.2]])      # votes class 0
    assert fz.majority_vote([m1, m2, m3]).tolist() == [1]


def test_a_tied_vote_goes_to_the_highest_mean_probability():
    m1 = np.array([[0.9, 0.1]])           # votes 0, strongly
    m2 = np.array([[0.4, 0.6]])           # votes 1, weakly
    # one vote each: the mean is [0.65, 0.35], so class 0 wins
    assert fz.majority_vote([m1, m2]).tolist() == [0]

    m3 = np.array([[0.49, 0.51]])
    m4 = np.array([[0.1, 0.9]])
    assert fz.majority_vote([m3, m4]).tolist() == [1]


def test_static_weights_are_the_normalised_validation_macro_f1():
    weights = fz.static_weights([0.9, 0.6, 0.3])
    assert weights == pytest.approx([0.5, 1 / 3, 1 / 6])
    assert weights.sum() == pytest.approx(1.0)


def test_static_weights_fall_back_to_equal_when_every_model_scores_zero():
    assert fz.static_weights([0.0, 0.0]) == pytest.approx([0.5, 0.5])


def test_static_weights_are_the_same_vector_for_every_flow():
    probs = one_flow(M1, M2, M3) + one_flow(M3, M2, M1)[:0]
    static = fz.static_weights([0.9, 0.6, 0.3])
    tiled = np.tile(static, (1, 1))
    fused = fz.fuse_probabilities(probs, tiled)
    assert fused[0] == pytest.approx(
        0.5 * np.array(M1) + (1 / 3) * np.array(M2) + (1 / 6) * np.array(M3))


# ===========================================================================
# the threshold -- chosen on validation, applied unchanged to test
# ===========================================================================
def test_the_threshold_maximises_the_validation_binary_f1():
    risk = np.array([0.05, 0.10, 0.60, 0.80, 0.90])
    truth = np.array([0, 0, 1, 1, 1])

    chosen = fz.choose_threshold(risk, truth, rule="max_f1")

    assert chosen["validation_f1"] == pytest.approx(1.0)
    assert 0.10 < chosen["tau"] <= 0.60
    assert chosen["rule"] == "max_f1"
    assert chosen["chosen_on"].startswith("the validation part")


def test_the_threshold_does_not_change_when_the_test_set_changes():
    """The only input is validation risk, so no test row can move tau."""
    validation_risk = np.array([0.05, 0.2, 0.55, 0.7, 0.95])
    validation_truth = np.array([0, 0, 1, 1, 1])
    first = fz.choose_threshold(validation_risk, validation_truth)

    # a completely different "test" set, scored with the same tau
    for test_risk in (np.zeros(50), np.ones(50), np.linspace(0, 1, 50)):
        again = fz.choose_threshold(validation_risk, validation_truth)
        assert again["tau"] == first["tau"]
        decided = (test_risk >= first["tau"]).astype(int)
        assert set(np.unique(decided)).issubset({0, 1})


def test_the_fpr_rule_respects_the_target_and_takes_the_lowest_such_threshold():
    risk = np.concatenate([np.linspace(0.0, 0.5, 100), np.linspace(0.6, 1.0, 20)])
    truth = np.array([0] * 100 + [1] * 20)

    chosen = fz.choose_threshold(risk, truth, rule="fpr_at_most", target_fpr=0.01)

    assert chosen["validation_fpr"] <= 0.01
    assert chosen["validation_recall"] == pytest.approx(1.0)
    assert chosen["tau"] >= 0.5          # every benign row sits at or below 0.5

    # and it really is the LOWEST threshold that respects the cap: the next
    # candidate below it lets a second false positive through, which is 2%
    below = max(t for t in np.unique(risk) if t < chosen["tau"])
    leaked = (risk >= below).astype(int)
    assert int(((leaked == 1) & (truth == 0)).sum()) / 100 > 0.01


def test_an_impossible_fpr_target_says_what_is_achievable():
    # every row scores 1.0, so even the highest threshold flags both benign rows
    risk = np.array([1.0, 1.0, 1.0, 1.0])
    truth = np.array([0, 0, 1, 1])
    with pytest.raises(fz.FusionError, match="false-positive rate"):
        fz.choose_threshold(risk, truth, rule="fpr_at_most", target_fpr=0.0)


def test_an_unknown_threshold_rule_and_an_empty_validation_set_are_refused():
    with pytest.raises(fz.FusionError, match="THRESHOLD_RULE"):
        fz.choose_threshold(np.array([0.5]), np.array([1]), rule="median")
    with pytest.raises(fz.FusionError, match="empty validation set"):
        fz.choose_threshold(np.array([]), np.array([]))


# ===========================================================================
# alignment
# ===========================================================================
def frame(rows, classes=("Normal", "Attack"), truth=None):
    data = {"Orig_Row": list(rows),
            "y_true": list(truth if truth is not None else [0] * len(rows))}
    for i, name in enumerate(classes):
        data["p_%s" % name] = [1.0 if i == 0 else 0.0] * len(rows)
    return pd.DataFrame(data)


def test_aligned_frames_pass_every_check():
    frames = {"m1": frame([5, 6, 7]), "m2": frame([5, 6, 7])}
    table, ok = fz.check_alignment(frames, ["Normal", "Attack"], verbose=False)

    assert ok is True
    assert (table["result"] == "[OK]").all()


def test_misaligned_rows_raise_a_clear_error():
    frames = {"m1": frame([5, 6, 7]), "m2": frame([7, 6, 5])}
    with pytest.raises(fz.FusionError, match="same rows in the same"):
        fz.check_alignment(frames, ["Normal", "Attack"], verbose=False)


def test_a_different_class_order_raises_a_clear_error():
    frames = {"m1": frame([1, 2]), "m2": frame([1, 2], ("Attack", "Normal"))}
    with pytest.raises(fz.FusionError, match="class order"):
        fz.check_alignment(frames, ["Normal", "Attack"], verbose=False)


def test_a_missing_orig_row_column_raises_a_clear_error():
    bad = frame([1, 2]).drop(columns=["Orig_Row"])
    with pytest.raises(fz.FusionError, match="Orig_Row"):
        fz.check_alignment({"m1": frame([1, 2]), "m2": bad},
                           ["Normal", "Attack"], verbose=False)


def test_disagreeing_ground_truth_raises_a_clear_error():
    frames = {"m1": frame([1, 2], truth=[0, 1]),
              "m2": frame([1, 2], truth=[1, 0])}
    with pytest.raises(fz.FusionError, match="ground truth"):
        fz.check_alignment(frames, ["Normal", "Attack"], verbose=False)


def test_a_fusion_model_name_outside_step_12_is_refused(monkeypatch):
    monkeypatch.setattr(config, "FUSION_MODELS", ["proposed", "not_a_model"])
    with pytest.raises(fz.FusionError, match="not_a_model"):
        fz.check_fusion_models(verbose=False)


def test_the_default_fusion_models_are_all_step_12_configurations():
    known = [e["name"] for e in config.EXPERIMENTS]
    assert all(name in known for name in config.FUSION_MODELS)


# ===========================================================================
# temporal smoothing
# ===========================================================================
def test_the_ema_never_crosses_a_capture_boundary():
    risk = np.array([0.0, 0.0, 1.0, 1.0])
    capture = ["a", "a", "b", "b"]

    smoothed = fz.ema_within_capture(risk, capture, alpha=0.5)

    # capture "a" is all zeros and capture "b" all ones: if the EMA leaked
    # across the boundary the first row of "b" would be pulled below 1.
    assert smoothed[:2] == pytest.approx([0.0, 0.0])
    assert smoothed[2:] == pytest.approx([1.0, 1.0])


def test_the_ema_smooths_inside_a_capture_and_keeps_the_row_order():
    risk = np.array([0.0, 1.0, 1.0])
    smoothed = fz.ema_within_capture(risk, ["a", "a", "a"], alpha=0.5)

    assert smoothed[0] == pytest.approx(0.0)
    assert smoothed[1] == pytest.approx(0.5)
    assert smoothed[2] == pytest.approx(0.75)
    assert len(smoothed) == len(risk)


def test_the_ema_follows_the_timestamp_order_but_writes_back_in_place():
    risk = np.array([1.0, 0.0, 0.0])
    order = [3, 1, 2]                      # the first row is chronologically last
    smoothed = fz.ema_within_capture(risk, ["a"] * 3, order=order, alpha=0.5)

    assert smoothed[1] == pytest.approx(0.0)
    assert smoothed[2] == pytest.approx(0.0)
    assert smoothed[0] == pytest.approx(0.5)


def test_interleaved_captures_are_still_smoothed_separately():
    risk = np.array([0.0, 1.0, 0.0, 1.0])
    capture = ["a", "b", "a", "b"]
    smoothed = fz.ema_within_capture(risk, capture, alpha=0.5)

    assert smoothed == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_a_bad_alpha_and_a_length_mismatch_are_refused():
    with pytest.raises(fz.FusionError, match="EMA_ALPHA"):
        fz.ema_within_capture(np.array([0.1]), ["a"], alpha=0.0)
    with pytest.raises(fz.FusionError, match="same length"):
        fz.ema_within_capture(np.array([0.1, 0.2]), ["a"])


# ===========================================================================
# scoring
# ===========================================================================
def test_the_binary_metrics_follow_the_threshold():
    risk = np.array([0.1, 0.4, 0.6, 0.9])
    truth = np.array([0, 0, 1, 1])

    scores = fz.binary_metrics(risk, truth, tau=0.5)

    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["f1"] == pytest.approx(1.0)
    assert scores["fpr"] == pytest.approx(0.0)
    assert scores["roc_auc"] == pytest.approx(1.0)
    assert scores["pr_auc"] == pytest.approx(1.0)


def test_a_lower_threshold_trades_precision_for_recall():
    risk = np.array([0.3, 0.3, 0.6, 0.9])
    truth = np.array([0, 0, 1, 1])

    strict = fz.binary_metrics(risk, truth, tau=0.95)
    loose = fz.binary_metrics(risk, truth, tau=0.2)

    assert strict["recall"] < loose["recall"]
    assert loose["fpr"] > strict["fpr"]


# ===========================================================================
# the saved tables
# ===========================================================================
LABELS = {0: "Normal", 1: "Port_Scan", 2: "Brute_Force"}


def fake_out(n_rows=24, seed=5):
    """A complete Step 13 result built from synthetic probabilities only."""
    rng = np.random.default_rng(seed)
    codes = np.resize([0, 1, 2], n_rows)
    model_names = ["m1", "m2", "m3"]

    probs = {}
    for shift, name in enumerate(model_names):
        raw = np.full((n_rows, 3), 0.1)
        raw[np.arange(n_rows), (codes + shift) % 3] += 1.5 + rng.uniform(
            0, 0.5, n_rows)
        probs[name] = raw / raw.sum(axis=1, keepdims=True)

    y_test = pd.DataFrame({"Label_encoded": codes,
                           "Label_binary": (codes != 0).astype("int64")})
    meta_test = pd.DataFrame({
        "Orig_Row": np.arange(1000, 1000 + n_rows),
        "Source_Capture": np.resize(["cap_a", "cap_b"], n_rows)})

    y_binary = y_test["Label_binary"].to_numpy()
    probs_list = [probs[n] for n in model_names]
    contexts = fz.consensus_context(probs_list)
    reliability, info = fz.build_reliability(
        probs, codes, contexts, LABELS, 3, verbose=False)
    fused, weights, rel_matrix, conf_matrix = fz.fuse_once(
        probs_list, model_names, info["lookup"], info["macro"], contexts)
    risk = fz.attack_risk(fused, [0])
    threshold = fz.choose_threshold(risk, y_binary)
    duplicate_flags = np.zeros(n_rows, dtype=bool)
    duplicate_flags[:4] = True

    results = []
    for name in model_names:
        results.append(fz.score_probabilities(
            "single: %s" % name, probs[name], y_test, [0], threshold["tau"],
            duplicate_flags, LABELS, 3))
    results.append(fz.score_probabilities(
        "ADAPTIVE EVIDENCE FUSION", fused, y_test, [0], threshold["tau"],
        duplicate_flags, LABELS, 3, overhead_ms=0.42))
    comparison = pd.DataFrame([fz.comparison_row(r) for r in results])
    ablation = pd.DataFrame([fz.comparison_row(results[-1]) |
                             {"ablation": "default"}])
    frames = {n: pd.DataFrame({"Orig_Row": meta_test["Orig_Row"],
                               "y_true": codes,
                               **{"p_%s" % LABELS[c]: probs[n][:, c]
                                  for c in range(3)}})
              for n in model_names}
    alignment, _ok = fz.check_alignment(frames, [LABELS[c] for c in range(3)],
                                        verbose=False)

    return {
        "model_names": model_names, "label_names": LABELS, "n_classes": 3,
        "benign_codes": [0], "y_test": y_test, "meta_test": meta_test,
        "test_probs": probs, "fused": fused, "weights": weights, "risk": risk,
        "rel_matrix": rel_matrix, "conf_matrix": conf_matrix,
        "contexts": contexts, "reliability": reliability, "info": info,
        "threshold": threshold, "comparison": comparison, "ablation": ablation,
        "results": results, "duplicate_flags": duplicate_flags,
        "alignment": alignment, "overhead_ms_per_1000": 0.42,
    }


def test_the_weights_summary_covers_both_groupings_and_sums_to_one():
    out = fake_out()
    summary = fz.weights_summary_table(out["weights"], out["model_names"],
                                       out["contexts"],
                                       out["y_test"]["Label_encoded"].to_numpy(),
                                       LABELS)

    assert set(summary["grouping"]) == {"context", "true class"}
    assert set(summary["model"]) == set(out["model_names"])
    for (_grouping, _group), part in summary.groupby(["grouping", "group"]):
        assert part["mean weight"].sum() == pytest.approx(1.0, abs=1e-5)
    assert (summary["std weight"] >= 0).all()
    assert (summary["rows"] > 0).all()


def test_the_weights_summary_refuses_a_mismatched_weight_matrix():
    with pytest.raises(fz.FusionError, match="weight matrix"):
        fz.weights_summary_table(np.zeros((3, 2)), ["m1", "m2"],
                                 np.zeros(4), np.zeros(4), LABELS)


def test_the_fused_predictions_frame_holds_every_promised_column():
    out = fake_out()
    frame = fz.fused_predictions_frame(out)

    assert len(frame) == len(out["risk"])
    for model in out["model_names"]:
        for prefix in ("r_", "Conf_", "w_", "Rel_"):
            assert "%s%s" % (prefix, model) in frame.columns
    for name in LABELS.values():
        assert "P_%s" % name in frame.columns
    for column in ("Orig_Row", "y_true", "context", "R_t", "fused_class",
                   "attack_decision"):
        assert column in frame.columns

    weights = frame[["w_%s" % m for m in out["model_names"]]].to_numpy()
    assert np.abs(weights.sum(axis=1) - 1.0).max() < 1e-6
    probabilities = frame[["P_%s" % LABELS[c] for c in range(3)]].to_numpy()
    assert np.abs(probabilities.sum(axis=1) - 1.0).max() < 1e-6
    assert frame["R_t"].between(0.0, 1.0).all()
    assert frame["Orig_Row"].tolist() == out["meta_test"]["Orig_Row"].tolist()
    expected = (out["risk"] >= out["threshold"]["tau"]).astype("int64")
    assert frame["attack_decision"].tolist() == expected.tolist()


def test_the_per_class_table_is_label_by_configuration():
    out = fake_out()
    table = fz.per_class_f1_frame(out["results"])

    assert list(table.columns)[:2] == ["Label", "test rows"]
    for result in out["results"]:
        assert result["name"] in table.columns
    assert set(table["Label"]) == set(LABELS.values())


def test_save_step13_writes_every_file_and_passes_its_own_verification(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "STEP13_OUTPUT_DIR", tmp_path / "step13")
    out = fake_out()

    saved = fz.save_step13(out, verbose=True)
    printed = capsys.readouterr().out

    assert saved["status"] == "written"
    assert saved["ok"] is True, saved["report"].to_string()
    assert "[OK]  Step 13 outputs written" in printed
    folder = tmp_path / "step13"
    for key, name in config.STEP13_FILES.items():
        assert (folder / name).exists(), "%s (%s) was not written" % (key, name)
    assert (saved["report"]["result"] == "[OK]").all()


def test_the_saved_files_survive_a_round_trip_through_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STEP13_OUTPUT_DIR", tmp_path / "step13")
    out = fake_out()
    fz.save_step13(out, verbose=False)
    folder = tmp_path / "step13"

    predictions = pd.read_csv(folder / config.STEP13_FILES["fused_predictions"])
    assert len(predictions) == len(out["risk"])
    weights = predictions[["w_%s" % m for m in out["model_names"]]].to_numpy()
    assert np.abs(weights.sum(axis=1) - 1.0).max() < 1e-6

    threshold = json.loads(
        (folder / config.STEP13_FILES["threshold"]).read_text(encoding="utf-8"))
    assert threshold["tau"] == pytest.approx(out["threshold"]["tau"])

    metrics = json.loads(
        (folder / config.STEP13_FILES["metrics"]).read_text(encoding="utf-8"))
    assert metrics["step"] == 13
    assert metrics["models_fused"] == out["model_names"]
    assert metrics["test_rows"] == len(out["risk"])
    assert metrics["alignment_ok"] is True
    assert "validation" in metrics["leakage_note"]
    assert set(metrics["mean_weight"]) == set(out["model_names"])
    # the summary travels inside metrics.json, so the resumable notebook path
    # can render it without recomputing anything
    assert metrics["summary"] == fz.step13_summary(out)

    reliability = pd.read_csv(folder / config.STEP13_FILES["reliability"])
    assert reliability["reliability"].between(0.0, 1.0).all()


def test_save_step13_returns_the_tables_under_the_keys_the_notebook_reads(
        tmp_path, monkeypatch):
    """run_step13 lifts these to the top level, so both paths look the same."""
    monkeypatch.setattr(config, "STEP13_OUTPUT_DIR", tmp_path / "step13")
    saved = fz.save_step13(fake_out(), verbose=False)

    for key in ("weights_summary", "fused_predictions", "per_class", "report",
                "metrics", "summary", "charts", "folder"):
        assert key in saved, key
    assert isinstance(saved["summary"], str) and saved["summary"]


def test_the_verification_fails_loudly_when_a_saved_row_is_wrong(tmp_path,
                                                                monkeypatch):
    """The check must read the file, not trust the in-memory numbers."""
    monkeypatch.setattr(config, "STEP13_OUTPUT_DIR", tmp_path / "step13")
    out = fake_out()
    fz.save_step13(out, verbose=False)
    folder = tmp_path / "step13"

    path = folder / config.STEP13_FILES["fused_predictions"]
    broken = pd.read_csv(path)
    broken.loc[0, "w_m1"] = broken.loc[0, "w_m1"] + 0.25      # no longer sums to 1
    broken.to_csv(path, index=False)

    report, ok = fz.verify_saved_step13(folder, out, verbose=False)

    assert ok is False
    failed = report[report["result"] == "[FAIL]"]["check"].tolist()
    assert "the weights of every flow sum to 1" in failed


def test_the_verification_reports_a_missing_predictions_file(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    report, ok = fz.verify_saved_step13(folder, fake_out(), verbose=False)

    assert ok is False
    assert report.iloc[0]["result"] == "[FAIL]"


def test_step13_outputs_exist_is_false_before_anything_is_written(tmp_path,
                                                                 monkeypatch):
    monkeypatch.setattr(config, "STEP13_OUTPUT_DIR", tmp_path / "step13")
    assert fz.step13_outputs_exist() is False


def test_load_step13_reads_back_what_save_step13_wrote(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STEP13_OUTPUT_DIR", tmp_path / "step13")
    out = fake_out()
    fz.save_step13(out, verbose=False)

    assert fz.step13_outputs_exist() is True
    loaded = fz.load_step13(verbose=False)

    assert loaded["status"] == "loaded_existing"
    assert len(loaded["fused_predictions"]) == len(out["risk"])
    assert loaded["threshold"]["tau"] == pytest.approx(out["threshold"]["tau"])
    assert len(loaded["comparison"]) == len(out["comparison"])


# ===========================================================================
# the written summary -- computed numbers only
# ===========================================================================
def test_a_tie_is_reported_as_a_tie_and_not_as_a_winner():
    table = pd.DataFrame({"configuration": ["a", "b", "c"],
                          "F1 (macro)": [0.8, 0.8, 0.5]})
    names, value = fz._leader(table, "F1 (macro)")

    assert sorted(names) == ["a", "b"]
    assert value == pytest.approx(0.8)


def test_the_summary_quotes_the_numbers_that_were_computed():
    out = fake_out()
    text = fz.step13_summary(out)
    fused = fz._row_value(out["comparison"], "ADAPTIVE EVIDENCE FUSION",
                          "F1 (macro)")

    assert "%.4f" % fused in text
    assert "%.4f" % out["threshold"]["tau"] in text
    for model in out["model_names"]:
        assert "`%s`" % model in text
    assert "duplicate-free" in text
    assert "optimistic" in text          # the validation caveat is always stated


def test_the_summary_states_the_direction_honestly():
    """It must not claim a gain the table does not show."""
    out = fake_out()
    comparison = out["comparison"]
    fused = fz._row_value(comparison, "ADAPTIVE EVIDENCE FUSION", "F1 (macro)")
    singles = comparison[comparison["configuration"].str.startswith("single:")]
    best = fz._leader(singles, "F1 (macro)")[1]

    text = fz.step13_summary(out)

    if fused > best + 1e-9:
        assert "**above** the best single model" in text
    elif abs(fused - best) <= 1e-9:
        assert "**level with** the best single model" in text
    else:
        assert "**below** the best single model" in text


def test_the_summary_reports_the_real_smallest_class():
    out = fake_out()
    per_class = out["results"][-1]["per_class"]
    smallest = per_class.loc[per_class["support"].idxmin()]

    text = fz.step13_summary(out)

    assert "`%s`" % smallest["Label"] in text
    assert "**%d rows**" % int(smallest["support"]) in text


# ===========================================================================
# the Windows-path guard -- nothing is ever written inside the repo
# ===========================================================================
def test_save_step13_refuses_a_windows_path_rather_than_making_junk(
        tmp_path, monkeypatch, capsys, assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP13_OUTPUT_DIR",
                        Path(r"C:\MSCAD2022\outputs\step13"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    saved = fz.save_step13(fake_out(), verbose=True)
    printed = capsys.readouterr().out

    assert saved["status"] == "skipped_not_windows"
    assert saved["ok"] is False
    assert "[SKIP]" in printed
    assert_no_leaked_output("save_step13")


def test_load_step13_refuses_a_windows_path(tmp_path, monkeypatch,
                                            assert_no_leaked_output):
    monkeypatch.setattr(config, "STEP13_OUTPUT_DIR",
                        Path(r"C:\MSCAD2022\outputs\step13"))
    monkeypatch.setattr(config, "running_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)

    assert fz.step13_outputs_exist() is False
    with pytest.raises(fz.FusionError, match="not usable here"):
        fz.load_step13(verbose=False)
    assert_no_leaked_output("load_step13")


def test_step13_output_dir_is_under_data_root_not_the_repo():
    assert REAL_STEP13_DIR.parent.parent == config.DATA_ROOT
    assert config.PROJECT_ROOT not in REAL_STEP13_DIR.parents
    assert REAL_STEP13_DIR.name == "step13"


def test_the_step13_config_block_holds_the_documented_defaults():
    assert config.RELIABILITY_METRIC in ("f1", "precision")
    assert config.WEIGHT_MODE in ("proportional", "softmax")
    assert config.CONTEXT_MODE in ("consensus_class", "protocol")
    assert config.THRESHOLD_RULE in ("max_f1", "fpr_at_most")
    assert config.SHRINKAGE_K > 0
    assert 0 < config.EPSILON < 1e-3
    assert 0 < config.EMA_ALPHA <= 1
    assert config.WEIGHT_TEMPERATURE > 0
    assert 0 < config.TARGET_FPR < 1
    assert config.STEP13_OUTPUT_DIR in config.DIRS_TO_CREATE
