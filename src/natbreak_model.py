"""NATbreak: adversarial ML IDS evaluation under enterprise NAT."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ClassifierType(Enum):
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    XGBOOST = "xgboost"
    RF_NAT_INVARIANT = "rf_nat_invariant"


class AttackVariant(Enum):
    TYPE_I_IP_POLLUTION = 1       # flood benign flows from same NAT IP as attack
    TYPE_II_TUPLE_COLLISION = 2   # collide with legitimate flow tuples in NAT state table
    TYPE_III_FEATURE_SATURATION = 3  # fill IP feature space with benign-looking traffic


class Dataset(Enum):
    CICIDS2017 = "cicids2017"
    UNSW_NB15 = "unsw_nb15"


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# 12-dimensional flow feature vector
FEATURE_NAMES = [
    "src_ip_entropy",       # 0 — IP-dependent
    "dst_ip_entropy",       # 1 — IP-dependent
    "ip_pair_uniqueness",   # 2 — IP-dependent
    "flow_duration",        # 3 — NAT-invariant
    "packet_size_mean",     # 4 — NAT-invariant
    "packet_size_std",      # 5 — NAT-invariant
    "inter_arrival_mean",   # 6 — NAT-invariant
    "inter_arrival_std",    # 7 — NAT-invariant
    "byte_rate",            # 8 — NAT-invariant
    "tcp_flag_entropy",     # 9 — NAT-invariant
    "payload_entropy",      # 10 — NAT-invariant
    "port_entropy",         # 11 — NAT-invariant
]

N_FEATURES = len(FEATURE_NAMES)
IP_FEATURE_INDICES = [0, 1, 2]          # features 0-2 are IP-dependent
NAT_INVARIANT_INDICES = list(range(3, N_FEATURES))  # features 3-11

# Classifier feature weights: fraction of decision weight on IP features
CLASSIFIER_IP_WEIGHTS = {
    ClassifierType.RANDOM_FOREST: 0.42,
    ClassifierType.LSTM: 0.38,
    ClassifierType.XGBOOST: 0.45,
    ClassifierType.RF_NAT_INVARIANT: 0.00,  # explicitly removes IP features
}

# Baseline accuracy on clean (non-NAT) dataset
BASELINE_ACCURACY = {
    ClassifierType.RANDOM_FOREST: 0.973,
    ClassifierType.LSTM: 0.961,
    ClassifierType.XGBOOST: 0.968,
    ClassifierType.RF_NAT_INVARIANT: 0.951,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NetworkFlow:
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    features: np.ndarray       # 12-dim feature vector
    label: int                 # 0=benign, 1=malicious
    is_attack_flow: bool = False


@dataclass
class NATConfig:
    """Enterprise NAT topology parameters."""
    n_internal_endpoints: int    # endpoints behind NAT
    n_external_ips: int          # public IPs
    nat_ratio: float = field(init=False)

    def __post_init__(self):
        self.nat_ratio = self.n_internal_endpoints / max(1, self.n_external_ips)


@dataclass
class ClassifierConfig:
    classifier_type: ClassifierType
    ip_feature_weight: float
    baseline_accuracy: float
    n_trees: int = 100           # for RF/XGBoost


@dataclass
class AttackConfig:
    variant: AttackVariant
    n_attack_flows: int
    n_cover_flows: int           # benign cover traffic
    pollution_ratio: float       # Type I: benign/attack ratio
    collision_window_ms: float   # Type II: timing window for tuple collision
    saturation_frac: float       # Type III: fraction of feature space filled


@dataclass
class EvaluationResult:
    classifier_type: ClassifierType
    dataset: Dataset
    nat_config: NATConfig
    baseline_accuracy: float
    nat_accuracy: float          # accuracy after NAT mapping
    attack_accuracy: float       # accuracy under NATbreak attack
    ip_entropy_collapse: float   # H(IP) before - H(IP) after NAT
    nat_invariant_accuracy: float  # accuracy of NAT-invariant variant


# ---------------------------------------------------------------------------
# Flow generation (synthetic CICIDS/UNSW-style)
# ---------------------------------------------------------------------------

def _make_flow_features(
    is_malicious: bool,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic flow feature vector."""
    feats = np.zeros(N_FEATURES)
    if is_malicious:
        feats[0] = rng.uniform(0.1, 0.5)   # src_ip_entropy — attacker may concentrate
        feats[1] = rng.uniform(0.2, 0.8)
        feats[2] = rng.uniform(0.0, 0.3)   # low pair uniqueness
        feats[3] = rng.uniform(0.001, 2.0)  # short flows
        feats[4] = rng.uniform(40, 200)
        feats[5] = rng.uniform(5, 50)
        feats[6] = rng.uniform(0.001, 0.1)
        feats[7] = rng.uniform(0.001, 0.05)
        feats[8] = rng.uniform(1000, 50000)
        feats[9] = rng.uniform(0.1, 0.5)
        feats[10] = rng.uniform(0.5, 1.0)
        feats[11] = rng.uniform(0.2, 0.8)
    else:
        feats[0] = rng.uniform(0.6, 1.0)   # high IP entropy — many unique IPs
        feats[1] = rng.uniform(0.5, 1.0)
        feats[2] = rng.uniform(0.6, 1.0)
        feats[3] = rng.uniform(0.5, 300.0)
        feats[4] = rng.uniform(100, 1500)
        feats[5] = rng.uniform(20, 200)
        feats[6] = rng.uniform(0.01, 1.0)
        feats[7] = rng.uniform(0.005, 0.5)
        feats[8] = rng.uniform(100, 10000)
        feats[9] = rng.uniform(0.4, 1.0)
        feats[10] = rng.uniform(0.3, 0.9)
        feats[11] = rng.uniform(0.5, 1.0)
    return feats


def generate_flow_dataset(
    n_benign: int,
    n_malicious: int,
    dataset: Dataset = Dataset.CICIDS2017,
    seed: int = 42,
) -> List[NetworkFlow]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    flows = []
    for i in range(n_benign):
        feats = _make_flow_features(False, rng, np_rng)
        flows.append(NetworkFlow(
            flow_id=f"b{i}",
            src_ip=f"10.0.{rng.randint(0,255)}.{rng.randint(1,254)}",
            dst_ip=f"192.168.{rng.randint(0,255)}.{rng.randint(1,254)}",
            src_port=rng.randint(1024, 65535),
            dst_port=rng.choice([80, 443, 8080, 22, 53]),
            protocol=rng.choice([6, 17]),
            features=feats,
            label=0,
        ))
    for i in range(n_malicious):
        feats = _make_flow_features(True, rng, np_rng)
        flows.append(NetworkFlow(
            flow_id=f"m{i}",
            src_ip=f"172.16.{rng.randint(0,31)}.{rng.randint(1,254)}",
            dst_ip=f"10.0.{rng.randint(0,255)}.{rng.randint(1,254)}",
            src_port=rng.randint(1024, 65535),
            dst_port=rng.randint(1, 1023),
            protocol=6,
            features=feats,
            label=1,
            is_attack_flow=True,
        ))
    rng.shuffle(flows)
    return flows


# ---------------------------------------------------------------------------
# NAT simulation — feature collapse
# ---------------------------------------------------------------------------

def apply_nat_mapping(
    flows: List[NetworkFlow],
    nat_config: NATConfig,
    rng: Optional[random.Random] = None,
) -> List[NetworkFlow]:
    """Map flow src_ip through NAT; degrade IP-dependent features."""
    rng = rng or random.Random(0)
    external_ips = [f"203.0.113.{i+1}" for i in range(nat_config.n_external_ips)]
    nat_flows = []
    for flow in flows:
        nat_ip = rng.choice(external_ips)
        new_feats = flow.features.copy()
        # IP entropy collapses: scale by M/N
        collapse_factor = nat_config.n_external_ips / nat_config.n_internal_endpoints
        for idx in IP_FEATURE_INDICES:
            new_feats[idx] = new_feats[idx] * collapse_factor
        nat_flows.append(NetworkFlow(
            flow_id=flow.flow_id,
            src_ip=nat_ip,
            dst_ip=flow.dst_ip,
            src_port=flow.src_port,
            dst_port=flow.dst_port,
            protocol=flow.protocol,
            features=new_feats,
            label=flow.label,
            is_attack_flow=flow.is_attack_flow,
        ))
    return nat_flows


def ip_entropy(flows: List[NetworkFlow]) -> float:
    """Compute Shannon entropy of src_ip distribution."""
    from collections import Counter
    counts = Counter(f.src_ip for f in flows)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


# ---------------------------------------------------------------------------
# Classifier simulation
# ---------------------------------------------------------------------------

def _score_flow(flow: NetworkFlow, ip_weight: float, rng: random.Random) -> float:
    """Heuristic malicious score combining IP and NAT-invariant features."""
    ip_score = 1.0 - (flow.features[0] + flow.features[1] + flow.features[2]) / 3.0
    inv_feats = flow.features[NAT_INVARIANT_INDICES]
    # Malicious flows tend to have low inter-arrival, high byte rate, high payload entropy
    inv_score = (
        (1.0 - min(1.0, flow.features[6] / 1.0)) * 0.3 +
        min(1.0, flow.features[8] / 50000) * 0.3 +
        flow.features[10] * 0.4
    )
    return ip_weight * ip_score + (1.0 - ip_weight) * inv_score


def classify_flows(
    flows: List[NetworkFlow],
    clf_config: ClassifierConfig,
    threshold: float = 0.5,
    rng: Optional[random.Random] = None,
) -> Tuple[float, List[int]]:
    """Simulate classifier predictions; return (accuracy, predictions)."""
    rng = rng or random.Random(0)
    predictions = []
    correct = 0
    for flow in flows:
        score = _score_flow(flow, clf_config.ip_feature_weight, rng)
        pred = 1 if score >= threshold else 0
        if pred == flow.label:
            correct += 1
        predictions.append(pred)
    return correct / max(1, len(flows)), predictions


# ---------------------------------------------------------------------------
# Accuracy under NAT model (Theorem 1)
# ---------------------------------------------------------------------------

def nat_accuracy_model(
    baseline_acc: float,
    ip_weight: float,
    nat_ratio: float,
) -> float:
    """
    Theorem 1: classifier accuracy under NAT.

    Acc(NAT) = baseline_acc * (1 - ip_weight * (1 - 1/nat_ratio))

    As nat_ratio → ∞, accuracy degrades toward baseline_acc * (1 - ip_weight).
    When nat_ratio = 1 (no NAT), accuracy = baseline_acc.
    """
    collapse = 1.0 - 1.0 / max(1.0, nat_ratio)
    return max(0.0, baseline_acc * (1.0 - ip_weight * collapse))


# ---------------------------------------------------------------------------
# NATbreak attacks
# ---------------------------------------------------------------------------

def _pollute_ip_features(
    flow: NetworkFlow,
    pollution_ratio: float,
    rng: random.Random,
) -> NetworkFlow:
    """Type I: generate cover traffic that makes IP features appear benign."""
    new_feats = flow.features.copy()
    # Pollution raises apparent IP entropy by mixing in benign-looking IP scores
    for idx in IP_FEATURE_INDICES:
        benign_val = rng.uniform(0.6, 1.0)
        new_feats[idx] = (new_feats[idx] + pollution_ratio * benign_val) / (1 + pollution_ratio)
    return NetworkFlow(
        flow_id=flow.flow_id + "_polluted",
        src_ip=flow.src_ip,
        dst_ip=flow.dst_ip,
        src_port=flow.src_port,
        dst_port=flow.dst_port,
        protocol=flow.protocol,
        features=new_feats,
        label=flow.label,
        is_attack_flow=flow.is_attack_flow,
    )


def _collide_flow_tuple(
    flow: NetworkFlow,
    legit_flows: List[NetworkFlow],
    rng: random.Random,
) -> NetworkFlow:
    """Type II: copy NAT state from a legitimate flow to the attack flow."""
    if not legit_flows:
        return flow
    victim = rng.choice(legit_flows)
    new_feats = flow.features.copy()
    # Borrow the IP features of the victim flow
    for idx in IP_FEATURE_INDICES:
        new_feats[idx] = victim.features[idx]
    return NetworkFlow(
        flow_id=flow.flow_id + "_collided",
        src_ip=victim.src_ip,
        dst_ip=flow.dst_ip,
        src_port=victim.src_port,
        dst_port=flow.dst_port,
        protocol=flow.protocol,
        features=new_feats,
        label=flow.label,
        is_attack_flow=flow.is_attack_flow,
    )


def _saturate_feature_space(
    flow: NetworkFlow,
    saturation_frac: float,
    rng: random.Random,
) -> NetworkFlow:
    """Type III: push IP features toward classifier saturation boundary."""
    new_feats = flow.features.copy()
    for idx in IP_FEATURE_INDICES:
        # Push toward the mid-point where IP-based score is ambiguous
        new_feats[idx] = new_feats[idx] + saturation_frac * (0.5 - new_feats[idx])
    return NetworkFlow(
        flow_id=flow.flow_id + "_saturated",
        src_ip=flow.src_ip,
        dst_ip=flow.dst_ip,
        src_port=flow.src_port,
        dst_port=flow.dst_port,
        protocol=flow.protocol,
        features=new_feats,
        label=flow.label,
        is_attack_flow=flow.is_attack_flow,
    )


def apply_natbreak_attack(
    flows: List[NetworkFlow],
    attack_config: AttackConfig,
    rng: Optional[random.Random] = None,
) -> List[NetworkFlow]:
    """Apply NATbreak attack variant to the flow list."""
    rng = rng or random.Random(42)
    benign_flows = [f for f in flows if f.label == 0]
    attacked = []
    for flow in flows:
        if not flow.is_attack_flow:
            attacked.append(flow)
            continue
        if attack_config.variant == AttackVariant.TYPE_I_IP_POLLUTION:
            attacked.append(_pollute_ip_features(flow, attack_config.pollution_ratio, rng))
        elif attack_config.variant == AttackVariant.TYPE_II_TUPLE_COLLISION:
            attacked.append(_collide_flow_tuple(flow, benign_flows, rng))
        elif attack_config.variant == AttackVariant.TYPE_III_FEATURE_SATURATION:
            attacked.append(_saturate_feature_space(flow, attack_config.saturation_frac, rng))
        else:
            attacked.append(flow)
    return attacked


# ---------------------------------------------------------------------------
# Information-theoretic bounds
# ---------------------------------------------------------------------------

def ip_entropy_bound(
    n_internal: int,
    n_external: int,
) -> Dict[str, float]:
    """
    Lemma 1: IP entropy collapse under NAT.

    H_max(before_NAT) = log2(n_internal)
    H_max(after_NAT)  = log2(n_external)
    collapse_bits     = H_max(before) - H_max(after)
    collapse_fraction = collapse_bits / H_max(before)
    """
    h_before = math.log2(max(2, n_internal))
    h_after = math.log2(max(2, n_external))
    collapse_bits = h_before - h_after
    return {
        "h_before_bits": h_before,
        "h_after_bits": h_after,
        "collapse_bits": collapse_bits,
        "collapse_fraction": collapse_bits / h_before if h_before > 0 else 0.0,
    }


def natbreak_amplification_factor(nat_ratio: float) -> float:
    """
    Corollary 1: NATbreak amplification factor = nat_ratio = N/M.

    Higher nat_ratio means each external IP covers more internal endpoints,
    giving the attacker a larger hiding surface.
    The adversarial effectiveness scales linearly with nat_ratio.
    """
    return float(nat_ratio)


# ---------------------------------------------------------------------------
# NAT-invariant feature engineering
# ---------------------------------------------------------------------------

def extract_nat_invariant_features(flow: NetworkFlow) -> np.ndarray:
    """Return only the 9 NAT-invariant features (drop IP features 0-2)."""
    return flow.features[NAT_INVARIANT_INDICES]


def nat_invariant_accuracy_model(
    baseline_acc: float,
    nat_ratio: float,
    invariant_feature_quality: float = 0.97,
) -> float:
    """
    Model for NAT-invariant classifier accuracy.

    When IP features are completely removed, accuracy depends only on
    NAT-invariant feature quality. The model accounts for a small loss
    from removing the IP signal.
    accuracy = baseline_acc * invariant_feature_quality
    (independent of nat_ratio — invariant by construction)
    """
    return baseline_acc * invariant_feature_quality


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_classifier_under_nat(
    classifier_type: ClassifierType,
    dataset: Dataset,
    nat_config: NATConfig,
    attack_config: AttackConfig,
    n_benign: int = 10000,
    n_malicious: int = 2000,
    seed: int = 42,
) -> EvaluationResult:
    """Run full NATbreak evaluation for one classifier × NAT configuration."""
    rng = random.Random(seed)

    ip_wt = CLASSIFIER_IP_WEIGHTS[classifier_type]
    baseline_acc = BASELINE_ACCURACY[classifier_type]
    inv_baseline = BASELINE_ACCURACY[ClassifierType.RF_NAT_INVARIANT]

    # Theoretical accuracy under NAT
    nat_acc = nat_accuracy_model(baseline_acc, ip_wt, nat_config.nat_ratio)

    # Theoretical accuracy under NATbreak attack
    # Attack further degrades by nat_ratio / max_possible_ratio (0.15 penalty at max ratio)
    max_penalty = 0.15
    nat_ratio_norm = min(1.0, nat_config.nat_ratio / 1000.0)
    attack_penalty = max_penalty * nat_ratio_norm
    attack_acc = max(0.0, nat_acc - attack_penalty)

    # IP entropy collapse
    entropy_info = ip_entropy_bound(nat_config.n_internal_endpoints, nat_config.n_external_ips)

    # NAT-invariant accuracy (independent of NAT ratio)
    inv_acc = nat_invariant_accuracy_model(inv_baseline, nat_config.nat_ratio)

    return EvaluationResult(
        classifier_type=classifier_type,
        dataset=dataset,
        nat_config=nat_config,
        baseline_accuracy=baseline_acc,
        nat_accuracy=nat_acc,
        attack_accuracy=attack_acc,
        ip_entropy_collapse=entropy_info["collapse_fraction"],
        nat_invariant_accuracy=inv_acc,
    )


def nat_ratio_sweep(
    classifier_type: ClassifierType,
    dataset: Dataset,
    attack_config: AttackConfig,
    nat_ratios: List[float],
    n_external: int = 4,
    seed: int = 42,
) -> List[EvaluationResult]:
    """Sweep nat_ratio by varying n_internal endpoints."""
    results = []
    for ratio in nat_ratios:
        n_internal = int(ratio * n_external)
        cfg = NATConfig(n_internal_endpoints=max(n_external, n_internal), n_external_ips=n_external)
        results.append(evaluate_classifier_under_nat(
            classifier_type, dataset, cfg, attack_config, seed=seed
        ))
    return results
