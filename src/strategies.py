"""SIFI environment and dataset configurations for Paper 09 NATbreak."""

from src.natbreak_model import (
    AttackConfig,
    AttackVariant,
    ClassifierType,
    Dataset,
    NATConfig,
)

# ---------------------------------------------------------------------------
# NAT configurations — representative enterprise topologies
# ---------------------------------------------------------------------------

# Small enterprise: 500 endpoints, 4 external IPs (125:1)
NAT_SMALL = NATConfig(n_internal_endpoints=500, n_external_ips=4)

# Medium enterprise: 5000 endpoints, 8 external IPs (625:1)
NAT_MEDIUM = NATConfig(n_internal_endpoints=5000, n_external_ips=8)

# Large SIFI: 100,000 endpoints, 16 external IPs (6250:1)
NAT_LARGE = NATConfig(n_internal_endpoints=100_000, n_external_ips=16)

# No NAT baseline (1:1)
NAT_NONE = NATConfig(n_internal_endpoints=1, n_external_ips=1)

# NAT ratio sweep values
NAT_RATIO_SWEEP = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000]

# ---------------------------------------------------------------------------
# Attack configurations
# ---------------------------------------------------------------------------

ATTACK_TYPE_I = AttackConfig(
    variant=AttackVariant.TYPE_I_IP_POLLUTION,
    n_attack_flows=200,
    n_cover_flows=2000,
    pollution_ratio=10.0,
    collision_window_ms=0.0,
    saturation_frac=0.0,
)

ATTACK_TYPE_II = AttackConfig(
    variant=AttackVariant.TYPE_II_TUPLE_COLLISION,
    n_attack_flows=200,
    n_cover_flows=500,
    pollution_ratio=0.0,
    collision_window_ms=50.0,
    saturation_frac=0.0,
)

ATTACK_TYPE_III = AttackConfig(
    variant=AttackVariant.TYPE_III_FEATURE_SATURATION,
    n_attack_flows=200,
    n_cover_flows=1000,
    pollution_ratio=0.0,
    collision_window_ms=0.0,
    saturation_frac=0.80,
)

# ---------------------------------------------------------------------------
# Evaluation targets
# ---------------------------------------------------------------------------

CLASSIFIERS = [
    ClassifierType.RANDOM_FOREST,
    ClassifierType.LSTM,
    ClassifierType.XGBOOST,
    ClassifierType.RF_NAT_INVARIANT,
]

DATASETS = [Dataset.CICIDS2017, Dataset.UNSW_NB15]

# CICIDS-2017 subset size
CICIDS_N_BENIGN = 10_000
CICIDS_N_MALICIOUS = 2_000

# UNSW-NB15 subset size
UNSW_N_BENIGN = 8_000
UNSW_N_MALICIOUS = 1_600
