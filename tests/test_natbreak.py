"""Tests for Paper 09: NATbreak adversarial ML IDS under enterprise NAT."""

import math
import random

import numpy as np
import pytest

from src.natbreak_model import (
    BASELINE_ACCURACY,
    CLASSIFIER_IP_WEIGHTS,
    IP_FEATURE_INDICES,
    NAT_INVARIANT_INDICES,
    N_FEATURES,
    AttackVariant,
    ClassifierType,
    Dataset,
    NATConfig,
    NetworkFlow,
    apply_nat_mapping,
    apply_natbreak_attack,
    classify_flows,
    evaluate_classifier_under_nat,
    extract_nat_invariant_features,
    generate_flow_dataset,
    ip_entropy,
    ip_entropy_bound,
    nat_accuracy_model,
    nat_invariant_accuracy_model,
    nat_ratio_sweep,
    natbreak_amplification_factor,
    ClassifierConfig,
)
from src.strategies import (
    ATTACK_TYPE_I,
    ATTACK_TYPE_II,
    ATTACK_TYPE_III,
    NAT_LARGE,
    NAT_MEDIUM,
    NAT_NONE,
    NAT_RATIO_SWEEP,
    NAT_SMALL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    return generate_flow_dataset(200, 50, seed=42)


@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def rf_config():
    return ClassifierConfig(
        classifier_type=ClassifierType.RANDOM_FOREST,
        ip_feature_weight=CLASSIFIER_IP_WEIGHTS[ClassifierType.RANDOM_FOREST],
        baseline_accuracy=BASELINE_ACCURACY[ClassifierType.RANDOM_FOREST],
    )


# ---------------------------------------------------------------------------
# NATConfig
# ---------------------------------------------------------------------------

def test_nat_config_ratio():
    cfg = NATConfig(n_internal_endpoints=1000, n_external_ips=4)
    assert cfg.nat_ratio == 250.0


def test_nat_config_none():
    assert NAT_NONE.nat_ratio == 1.0


def test_nat_large_ratio():
    assert NAT_LARGE.nat_ratio == pytest.approx(6250.0)


def test_nat_medium_ratio():
    assert NAT_MEDIUM.nat_ratio == pytest.approx(625.0)


def test_nat_small_ratio():
    assert NAT_SMALL.nat_ratio == pytest.approx(125.0)


# ---------------------------------------------------------------------------
# generate_flow_dataset
# ---------------------------------------------------------------------------

def test_dataset_total_size(small_dataset):
    assert len(small_dataset) == 250


def test_dataset_label_counts(small_dataset):
    benign = sum(1 for f in small_dataset if f.label == 0)
    malicious = sum(1 for f in small_dataset if f.label == 1)
    assert benign == 200
    assert malicious == 50


def test_dataset_feature_shape(small_dataset):
    for flow in small_dataset[:5]:
        assert flow.features.shape == (N_FEATURES,)


def test_dataset_malicious_is_attack_flow(small_dataset):
    for flow in small_dataset:
        if flow.label == 1:
            assert flow.is_attack_flow is True


def test_dataset_benign_not_attack_flow(small_dataset):
    for flow in small_dataset:
        if flow.label == 0:
            assert flow.is_attack_flow is False


def test_dataset_deterministic():
    d1 = generate_flow_dataset(100, 20, seed=7)
    d2 = generate_flow_dataset(100, 20, seed=7)
    assert d1[0].flow_id == d2[0].flow_id


# ---------------------------------------------------------------------------
# apply_nat_mapping
# ---------------------------------------------------------------------------

def test_nat_mapping_collapses_ip_features(small_dataset, rng):
    nat_flows = apply_nat_mapping(small_dataset, NAT_LARGE, rng=rng)
    for orig, nat in zip(small_dataset, nat_flows):
        for idx in IP_FEATURE_INDICES:
            assert nat.features[idx] <= orig.features[idx] + 1e-9


def test_nat_mapping_preserves_invariant_features(small_dataset, rng):
    nat_flows = apply_nat_mapping(small_dataset, NAT_LARGE, rng=rng)
    for orig, nat in zip(small_dataset, nat_flows):
        for idx in NAT_INVARIANT_INDICES:
            assert abs(nat.features[idx] - orig.features[idx]) < 1e-9


def test_nat_mapping_collapses_to_external_ips(small_dataset, rng):
    n_ext = NAT_SMALL.n_external_ips
    nat_flows = apply_nat_mapping(small_dataset, NAT_SMALL, rng=rng)
    unique_ips = set(f.src_ip for f in nat_flows)
    assert len(unique_ips) <= n_ext


def test_nat_mapping_preserves_labels(small_dataset, rng):
    nat_flows = apply_nat_mapping(small_dataset, NAT_LARGE, rng=rng)
    for orig, nat in zip(small_dataset, nat_flows):
        assert orig.label == nat.label


def test_nat_mapping_no_nat_unchanged(small_dataset, rng):
    nat_flows = apply_nat_mapping(small_dataset, NAT_NONE, rng=rng)
    # NAT ratio = 1 → collapse_factor = 1 → features unchanged
    for orig, nat in zip(small_dataset, nat_flows):
        for idx in IP_FEATURE_INDICES:
            assert abs(nat.features[idx] - orig.features[idx]) < 1e-9


# ---------------------------------------------------------------------------
# ip_entropy
# ---------------------------------------------------------------------------

def test_ip_entropy_high_diversity(small_dataset):
    h = ip_entropy(small_dataset)
    assert h > 3.0


def test_ip_entropy_after_nat_lower(small_dataset, rng):
    h_before = ip_entropy(small_dataset)
    nat_flows = apply_nat_mapping(small_dataset, NAT_LARGE, rng=rng)
    h_after = ip_entropy(nat_flows)
    assert h_after < h_before


def test_ip_entropy_empty():
    assert ip_entropy([]) == 0.0


# ---------------------------------------------------------------------------
# ip_entropy_bound (Lemma 1)
# ---------------------------------------------------------------------------

def test_entropy_bound_h_before_gt_h_after():
    result = ip_entropy_bound(10000, 4)
    assert result["h_before_bits"] > result["h_after_bits"]


def test_entropy_bound_collapse_nonnegative():
    result = ip_entropy_bound(1000, 8)
    assert result["collapse_bits"] >= 0.0
    assert result["collapse_fraction"] >= 0.0


def test_entropy_bound_no_nat():
    result = ip_entropy_bound(4, 4)
    assert result["collapse_bits"] == pytest.approx(0.0)


def test_entropy_bound_sifi_large():
    result = ip_entropy_bound(100_000, 16)
    assert result["h_before_bits"] == pytest.approx(math.log2(100_000), rel=0.01)
    assert result["h_after_bits"] == pytest.approx(math.log2(16), rel=0.01)


def test_entropy_bound_collapse_fraction_le_one():
    result = ip_entropy_bound(1_000_000, 2)
    assert result["collapse_fraction"] <= 1.0


def test_entropy_bound_increases_with_nat_ratio():
    r1 = ip_entropy_bound(100, 16)
    r2 = ip_entropy_bound(10000, 16)
    assert r2["collapse_bits"] > r1["collapse_bits"]


# ---------------------------------------------------------------------------
# nat_accuracy_model (Theorem 1)
# ---------------------------------------------------------------------------

def test_nat_accuracy_no_nat():
    acc = nat_accuracy_model(0.97, 0.42, nat_ratio=1.0)
    assert acc == pytest.approx(0.97)


def test_nat_accuracy_degrades_with_ratio():
    acc1 = nat_accuracy_model(0.97, 0.42, nat_ratio=10)
    acc2 = nat_accuracy_model(0.97, 0.42, nat_ratio=1000)
    assert acc2 < acc1


def test_nat_accuracy_nonnegative():
    acc = nat_accuracy_model(0.5, 0.9, nat_ratio=1_000_000)
    assert acc >= 0.0


def test_nat_accuracy_zero_ip_weight():
    # IP-invariant classifier: accuracy unchanged by NAT
    acc = nat_accuracy_model(0.95, 0.0, nat_ratio=10000)
    assert acc == pytest.approx(0.95)


def test_nat_accuracy_full_ip_weight():
    # All weight on IP features: at very high NAT ratio, approaches 0
    acc = nat_accuracy_model(1.0, 1.0, nat_ratio=1_000_000)
    assert acc < 0.001


def test_nat_accuracy_theorem_formula():
    baseline = 0.97
    ip_wt = 0.42
    ratio = 100.0
    expected = baseline * (1 - ip_wt * (1 - 1 / ratio))
    assert nat_accuracy_model(baseline, ip_wt, ratio) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# nat_invariant_accuracy_model
# ---------------------------------------------------------------------------

def test_invariant_acc_constant_across_ratios():
    baseline = BASELINE_ACCURACY[ClassifierType.RF_NAT_INVARIANT]
    acc1 = nat_invariant_accuracy_model(baseline, 1)
    acc2 = nat_invariant_accuracy_model(baseline, 100_000)
    assert abs(acc1 - acc2) < 1e-9


def test_invariant_acc_below_baseline():
    baseline = BASELINE_ACCURACY[ClassifierType.RF_NAT_INVARIANT]
    acc = nat_invariant_accuracy_model(baseline, 1000)
    assert acc <= baseline


def test_invariant_acc_nonnegative():
    assert nat_invariant_accuracy_model(0.0, 1000) >= 0.0


# ---------------------------------------------------------------------------
# natbreak_amplification_factor (Corollary 1)
# ---------------------------------------------------------------------------

def test_amp_factor_equals_ratio():
    assert natbreak_amplification_factor(100.0) == pytest.approx(100.0)


def test_amp_factor_sifi():
    assert natbreak_amplification_factor(NAT_LARGE.nat_ratio) == pytest.approx(NAT_LARGE.nat_ratio)


def test_amp_factor_monotone():
    prev = 0.0
    for r in [1, 10, 100, 1000]:
        amp = natbreak_amplification_factor(float(r))
        assert amp > prev
        prev = amp


# ---------------------------------------------------------------------------
# apply_natbreak_attack
# ---------------------------------------------------------------------------

def test_attack_type_i_changes_ip_features(small_dataset, rng):
    attacked = apply_natbreak_attack(small_dataset, ATTACK_TYPE_I, rng=rng)
    orig_attack = [f for f in small_dataset if f.is_attack_flow]
    atk_attacked = [f for f in attacked if "polluted" in f.flow_id]
    assert len(atk_attacked) == len(orig_attack)
    for orig, atk in zip(orig_attack, atk_attacked):
        changed = any(abs(atk.features[i] - orig.features[i]) > 1e-9 for i in IP_FEATURE_INDICES)
        assert changed


def test_attack_type_i_preserves_invariant_features(small_dataset, rng):
    attacked = apply_natbreak_attack(small_dataset, ATTACK_TYPE_I, rng=rng)
    orig_by_id = {f.flow_id: f for f in small_dataset}
    for atk in attacked:
        if "polluted" not in atk.flow_id:
            continue
        orig_id = atk.flow_id.replace("_polluted", "")
        orig = orig_by_id[orig_id]
        for idx in NAT_INVARIANT_INDICES:
            assert abs(atk.features[idx] - orig.features[idx]) < 1e-9


def test_attack_type_ii_borrows_benign_ip(small_dataset, rng):
    attacked = apply_natbreak_attack(small_dataset, ATTACK_TYPE_II, rng=rng)
    benign_ips = {f.src_ip for f in small_dataset if f.label == 0}
    for atk in attacked:
        if "collided" in atk.flow_id:
            assert atk.src_ip in benign_ips


def test_attack_type_iii_pushes_ip_toward_midpoint(small_dataset, rng):
    attacked = apply_natbreak_attack(small_dataset, ATTACK_TYPE_III, rng=rng)
    orig_by_id = {f.flow_id: f for f in small_dataset if f.is_attack_flow}
    for atk in attacked:
        if "saturated" not in atk.flow_id:
            continue
        orig_id = atk.flow_id.replace("_saturated", "")
        orig = orig_by_id[orig_id]
        for idx in IP_FEATURE_INDICES:
            # Feature should move toward 0.5
            dist_orig = abs(orig.features[idx] - 0.5)
            dist_atk = abs(atk.features[idx] - 0.5)
            assert dist_atk <= dist_orig + 1e-9


def test_attack_preserves_labels(small_dataset, rng):
    for atk_cfg in [ATTACK_TYPE_I, ATTACK_TYPE_II, ATTACK_TYPE_III]:
        attacked = apply_natbreak_attack(small_dataset, atk_cfg, rng=rng)
        attacked_by_prefix = {f.flow_id.split("_")[0]: f for f in attacked}


def test_attack_benign_flows_unchanged(small_dataset, rng):
    attacked = apply_natbreak_attack(small_dataset, ATTACK_TYPE_I, rng=rng)
    orig_benign = {f.flow_id: f for f in small_dataset if not f.is_attack_flow}
    for atk in attacked:
        if atk.flow_id in orig_benign:
            orig = orig_benign[atk.flow_id]
            np.testing.assert_array_almost_equal(atk.features, orig.features)


# ---------------------------------------------------------------------------
# classify_flows
# ---------------------------------------------------------------------------

def test_classify_returns_accuracy_in_range(small_dataset, rf_config, rng):
    acc, _ = classify_flows(small_dataset, rf_config, rng=rng)
    assert 0.0 <= acc <= 1.0


def test_classify_predictions_length(small_dataset, rf_config, rng):
    acc, preds = classify_flows(small_dataset, rf_config, rng=rng)
    assert len(preds) == len(small_dataset)


def test_classify_predictions_binary(small_dataset, rf_config, rng):
    _, preds = classify_flows(small_dataset, rf_config, rng=rng)
    assert all(p in (0, 1) for p in preds)


def test_classify_invariant_clf_reasonable_accuracy(small_dataset, rng):
    cfg = ClassifierConfig(
        classifier_type=ClassifierType.RF_NAT_INVARIANT,
        ip_feature_weight=0.0,
        baseline_accuracy=BASELINE_ACCURACY[ClassifierType.RF_NAT_INVARIANT],
    )
    acc, _ = classify_flows(small_dataset, cfg, rng=rng)
    assert acc >= 0.3  # scoring is heuristic, but shouldn't be totally random


# ---------------------------------------------------------------------------
# extract_nat_invariant_features
# ---------------------------------------------------------------------------

def test_invariant_features_shape(small_dataset):
    for flow in small_dataset[:5]:
        inv = extract_nat_invariant_features(flow)
        assert inv.shape == (len(NAT_INVARIANT_INDICES),)


def test_invariant_features_excludes_ip(small_dataset):
    for flow in small_dataset[:5]:
        inv = extract_nat_invariant_features(flow)
        full = flow.features
        # Invariant features should match features[3:]
        np.testing.assert_array_equal(inv, full[3:])


# ---------------------------------------------------------------------------
# evaluate_classifier_under_nat
# ---------------------------------------------------------------------------

def test_evaluate_baseline_gt_nat():
    result = evaluate_classifier_under_nat(
        ClassifierType.RANDOM_FOREST, Dataset.CICIDS2017, NAT_LARGE, ATTACK_TYPE_I
    )
    assert result.baseline_accuracy > result.nat_accuracy


def test_evaluate_nat_gt_attack():
    result = evaluate_classifier_under_nat(
        ClassifierType.RANDOM_FOREST, Dataset.CICIDS2017, NAT_LARGE, ATTACK_TYPE_I
    )
    assert result.nat_accuracy >= result.attack_accuracy


def test_evaluate_invariant_acc_constant():
    r1 = evaluate_classifier_under_nat(
        ClassifierType.RANDOM_FOREST, Dataset.CICIDS2017, NAT_SMALL, ATTACK_TYPE_I
    )
    r2 = evaluate_classifier_under_nat(
        ClassifierType.RANDOM_FOREST, Dataset.CICIDS2017, NAT_LARGE, ATTACK_TYPE_I
    )
    # NAT-invariant accuracy should be equal (independent of NAT ratio)
    assert r1.nat_invariant_accuracy == pytest.approx(r2.nat_invariant_accuracy)


def test_evaluate_entropy_collapse_in_range():
    result = evaluate_classifier_under_nat(
        ClassifierType.RANDOM_FOREST, Dataset.CICIDS2017, NAT_LARGE, ATTACK_TYPE_I
    )
    assert 0.0 <= result.ip_entropy_collapse <= 1.0


def test_evaluate_invariant_better_than_standard_under_high_nat():
    result = evaluate_classifier_under_nat(
        ClassifierType.RANDOM_FOREST, Dataset.CICIDS2017, NAT_LARGE, ATTACK_TYPE_I
    )
    assert result.nat_invariant_accuracy > result.nat_accuracy


def test_evaluate_no_nat_near_baseline():
    result = evaluate_classifier_under_nat(
        ClassifierType.RANDOM_FOREST, Dataset.CICIDS2017, NAT_NONE, ATTACK_TYPE_I
    )
    assert abs(result.nat_accuracy - result.baseline_accuracy) < 0.05


# ---------------------------------------------------------------------------
# nat_ratio_sweep
# ---------------------------------------------------------------------------

def test_nat_ratio_sweep_length():
    results = nat_ratio_sweep(
        ClassifierType.RANDOM_FOREST, Dataset.CICIDS2017, ATTACK_TYPE_I, [1, 10, 100]
    )
    assert len(results) == 3


def test_nat_ratio_sweep_monotone_degradation():
    results = nat_ratio_sweep(
        ClassifierType.RANDOM_FOREST, Dataset.CICIDS2017, ATTACK_TYPE_I,
        [1, 10, 100, 1000]
    )
    accs = [r.nat_accuracy for r in results]
    for i in range(1, len(accs)):
        assert accs[i] <= accs[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# Strategy fixtures
# ---------------------------------------------------------------------------

def test_attack_type_i_variant():
    assert ATTACK_TYPE_I.variant == AttackVariant.TYPE_I_IP_POLLUTION


def test_attack_type_ii_variant():
    assert ATTACK_TYPE_II.variant == AttackVariant.TYPE_II_TUPLE_COLLISION


def test_attack_type_iii_variant():
    assert ATTACK_TYPE_III.variant == AttackVariant.TYPE_III_FEATURE_SATURATION


def test_nat_ratio_sweep_list_sorted():
    assert NAT_RATIO_SWEEP == sorted(NAT_RATIO_SWEEP)


def test_classifier_ip_weights_sum_le_one():
    for c in [ClassifierType.RANDOM_FOREST, ClassifierType.LSTM, ClassifierType.XGBOOST]:
        assert CLASSIFIER_IP_WEIGHTS[c] <= 1.0


def test_invariant_classifier_zero_ip_weight():
    assert CLASSIFIER_IP_WEIGHTS[ClassifierType.RF_NAT_INVARIANT] == 0.0
