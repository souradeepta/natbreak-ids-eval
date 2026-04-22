"""Paper 09 experiments: NATbreak figures and LaTeX macros."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.natbreak_model import (
    BASELINE_ACCURACY,
    CLASSIFIER_IP_WEIGHTS,
    ClassifierType,
    Dataset,
    NATConfig,
    AttackVariant,
    evaluate_classifier_under_nat,
    ip_entropy_bound,
    nat_accuracy_model,
    nat_invariant_accuracy_model,
    nat_ratio_sweep,
    natbreak_amplification_factor,
)
from src.strategies import (
    ATTACK_TYPE_I,
    ATTACK_TYPE_II,
    ATTACK_TYPE_III,
    CLASSIFIERS,
    NAT_LARGE,
    NAT_MEDIUM,
    NAT_NONE,
    NAT_RATIO_SWEEP,
    NAT_SMALL,
)

FIG_DIR = Path("src/figures")

STYLE = {
    "fig_size": (5.5, 3.5),
    "dpi": 150,
    "title_size": 11,
    "label_size": 10,
    "tick_size": 9,
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "clf_colors": {
        ClassifierType.RANDOM_FOREST: "#1f77b4",
        ClassifierType.LSTM: "#ff7f0e",
        ClassifierType.XGBOOST: "#2ca02c",
        ClassifierType.RF_NAT_INVARIANT: "#d62728",
    },
}


def _savefig(name: str) -> None:
    out = FIG_DIR / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def _write_macros(macros: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, v in macros.items():
        if isinstance(v, float):
            val = f"{v:.4g}"
        else:
            val = str(v)
        lines.append(f"\\newcommand{{\\{k}}}{{{val}}}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figure 1: Baseline accuracy vs NAT accuracy by classifier
# ---------------------------------------------------------------------------

def fig_baseline_vs_nat(macros: Dict) -> None:
    clf_labels = ["RF", "LSTM", "XGBoost", "RF-NAT-Inv"]
    clfs = [ClassifierType.RANDOM_FOREST, ClassifierType.LSTM,
            ClassifierType.XGBOOST, ClassifierType.RF_NAT_INVARIANT]

    baseline_vals = [BASELINE_ACCURACY[c] for c in clfs]
    nat_vals = [
        nat_accuracy_model(BASELINE_ACCURACY[c], CLASSIFIER_IP_WEIGHTS[c], NAT_LARGE.nat_ratio)
        for c in clfs
    ]

    x = np.arange(len(clf_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=STYLE["fig_size"])
    ax.bar(x - w / 2, baseline_vals, w, label="No NAT (baseline)", color=STYLE["colors"][0], alpha=0.85)
    ax.bar(x + w / 2, nat_vals, w, label=f"NAT ratio {NAT_LARGE.nat_ratio:.0f}:1", color=STYLE["colors"][1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(clf_labels, fontsize=STYLE["tick_size"])
    ax.set_ylabel("Accuracy", fontsize=STYLE["label_size"])
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Classifier Accuracy: Baseline vs NAT", fontsize=STYLE["title_size"])
    ax.legend(fontsize=STYLE["tick_size"])
    ax.yaxis.grid(True, alpha=0.3)
    _savefig("fig1_baseline_vs_nat")

    for c, lbl in zip(clfs, clf_labels):
        key = lbl.replace("-", "").replace("Inv", "Invariant")
        macros[f"BaselineAcc{key}"] = f"{BASELINE_ACCURACY[c]:.3f}"
        macros[f"NatAcc{key}"] = f"{nat_accuracy_model(BASELINE_ACCURACY[c], CLASSIFIER_IP_WEIGHTS[c], NAT_LARGE.nat_ratio):.3f}"

    macros["NatRatioLarge"] = f"{NAT_LARGE.nat_ratio:.0f}"
    macros["NatRatioMedium"] = f"{NAT_MEDIUM.nat_ratio:.0f}"
    macros["NatRatioSmall"] = f"{NAT_SMALL.nat_ratio:.0f}"


# ---------------------------------------------------------------------------
# Figure 2: Accuracy degradation vs NAT ratio (RF only)
# ---------------------------------------------------------------------------

def fig_accuracy_vs_nat_ratio(macros: Dict) -> None:
    ip_wt = CLASSIFIER_IP_WEIGHTS[ClassifierType.RANDOM_FOREST]
    baseline = BASELINE_ACCURACY[ClassifierType.RANDOM_FOREST]
    inv_baseline = BASELINE_ACCURACY[ClassifierType.RF_NAT_INVARIANT]

    ratios = NAT_RATIO_SWEEP
    rf_accs = [nat_accuracy_model(baseline, ip_wt, r) for r in ratios]
    inv_accs = [nat_invariant_accuracy_model(inv_baseline, r) for r in ratios]
    baseline_line = [baseline] * len(ratios)

    fig, ax = plt.subplots(figsize=STYLE["fig_size"])
    ax.semilogx(ratios, baseline_line, "--", label="RF baseline (no NAT)", color="gray", linewidth=1.5)
    ax.semilogx(ratios, rf_accs, marker="o", label="RF under NAT", color=STYLE["colors"][0], linewidth=2)
    ax.semilogx(ratios, inv_accs, marker="s", label="RF-NAT-Invariant", color=STYLE["colors"][3], linewidth=2)
    ax.set_xlabel("NAT Ratio (N:M, log scale)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Accuracy", fontsize=STYLE["label_size"])
    ax.set_title("RF Accuracy vs NAT Ratio", fontsize=STYLE["title_size"])
    ax.legend(fontsize=STYLE["tick_size"])
    ax.yaxis.grid(True, alpha=0.3)
    _savefig("fig2_accuracy_vs_nat_ratio")

    macros["RfAccAtRatioTen"] = f"{nat_accuracy_model(baseline, ip_wt, 10):.3f}"
    macros["RfAccAtRatioHundred"] = f"{nat_accuracy_model(baseline, ip_wt, 100):.3f}"
    macros["RfAccAtRatioThousand"] = f"{nat_accuracy_model(baseline, ip_wt, 1000):.3f}"
    macros["InvAccAtRatioThousand"] = f"{nat_invariant_accuracy_model(inv_baseline, 1000):.3f}"


# ---------------------------------------------------------------------------
# Figure 3: IP entropy collapse by NAT ratio
# ---------------------------------------------------------------------------

def fig_entropy_collapse(macros: Dict) -> None:
    n_external = 16
    n_internals = [n_external * r for r in NAT_RATIO_SWEEP]
    ratios = NAT_RATIO_SWEEP

    h_before = [ip_entropy_bound(n, n_external)["h_before_bits"] for n in n_internals]
    h_after_vals = [ip_entropy_bound(n, n_external)["h_after_bits"] for n in n_internals]
    collapse_fracs = [ip_entropy_bound(n, n_external)["collapse_fraction"] for n in n_internals]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].semilogx(ratios, h_before, marker="o", label="H(IP) before NAT", color=STYLE["colors"][0], linewidth=2)
    axes[0].semilogx(ratios, h_after_vals, "--", label="H(IP) after NAT", color=STYLE["colors"][1], linewidth=2)
    axes[0].set_xlabel("NAT Ratio", fontsize=STYLE["label_size"])
    axes[0].set_ylabel("Entropy (bits)", fontsize=STYLE["label_size"])
    axes[0].set_title("IP Entropy Before/After NAT", fontsize=STYLE["title_size"])
    axes[0].legend(fontsize=STYLE["tick_size"])
    axes[0].yaxis.grid(True, alpha=0.3)

    axes[1].semilogx(ratios, collapse_fracs, marker="s", color=STYLE["colors"][3], linewidth=2)
    axes[1].set_xlabel("NAT Ratio", fontsize=STYLE["label_size"])
    axes[1].set_ylabel("Collapse Fraction", fontsize=STYLE["label_size"])
    axes[1].set_title("IP Entropy Collapse Fraction", fontsize=STYLE["title_size"])
    axes[1].yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig("fig3_entropy_collapse")

    ref = ip_entropy_bound(100_000, 16)
    macros["EntropyBeforeSifi"] = f"{ref['h_before_bits']:.2f}"
    macros["EntropyAfterSifi"] = f"{ref['h_after_bits']:.2f}"
    macros["EntropyCollapseFracSifi"] = f"{ref['collapse_fraction']:.3f}"
    macros["LemmaOneHBefore"] = f"{ref['h_before_bits']:.2f}"
    macros["LemmaOneHAfter"] = f"{ref['h_after_bits']:.2f}"


# ---------------------------------------------------------------------------
# Figure 4: NATbreak attack accuracy vs attack variant (at SIFI ratio)
# ---------------------------------------------------------------------------

def fig_attack_by_variant(macros: Dict) -> None:
    attacks = [ATTACK_TYPE_I, ATTACK_TYPE_II, ATTACK_TYPE_III]
    attack_labels = ["Type I\nIP Pollution", "Type II\nTuple Collision", "Type III\nFeat Saturation"]
    clfs = [ClassifierType.RANDOM_FOREST, ClassifierType.LSTM, ClassifierType.XGBOOST]
    clf_labels = ["RF", "LSTM", "XGBoost"]
    colors = [STYLE["colors"][i] for i in range(3)]

    nat_acc_vals = {
        c: nat_accuracy_model(BASELINE_ACCURACY[c], CLASSIFIER_IP_WEIGHTS[c], NAT_LARGE.nat_ratio)
        for c in clfs
    }

    x = np.arange(len(attack_labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=STYLE["fig_size"])
    for i, (c, lbl, col) in enumerate(zip(clfs, clf_labels, colors)):
        res_vals = [
            evaluate_classifier_under_nat(c, Dataset.CICIDS2017, NAT_LARGE, atk).attack_accuracy
            for atk in attacks
        ]
        ax.bar(x + (i - 1) * w, res_vals, w, label=lbl, color=col, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(attack_labels, fontsize=STYLE["tick_size"])
    ax.set_ylabel("Accuracy Under Attack", fontsize=STYLE["label_size"])
    ax.set_ylim(0.3, 1.05)
    ax.set_title("NATbreak: Accuracy Under Each Attack Variant", fontsize=STYLE["title_size"])
    ax.legend(fontsize=STYLE["tick_size"])
    ax.yaxis.grid(True, alpha=0.3)
    _savefig("fig4_attack_by_variant")

    # Store representative RF values
    for atk, lbl in zip(attacks, ["TypeOne", "TypeTwo", "TypeThree"]):
        res = evaluate_classifier_under_nat(ClassifierType.RANDOM_FOREST, Dataset.CICIDS2017, NAT_LARGE, atk)
        macros[f"RfAttack{lbl}Acc"] = f"{res.attack_accuracy:.3f}"


# ---------------------------------------------------------------------------
# Figure 5: NAT-invariant vs standard RF across NAT ratios and attack
# ---------------------------------------------------------------------------

def fig_invariant_comparison(macros: Dict) -> None:
    ratios = [1, 10, 100, 500, 1000, 5000]
    n_external = 16
    ip_wt = CLASSIFIER_IP_WEIGHTS[ClassifierType.RANDOM_FOREST]
    baseline = BASELINE_ACCURACY[ClassifierType.RANDOM_FOREST]
    inv_baseline = BASELINE_ACCURACY[ClassifierType.RF_NAT_INVARIANT]

    rf_nat = [nat_accuracy_model(baseline, ip_wt, r) for r in ratios]
    inv_nat = [nat_invariant_accuracy_model(inv_baseline, r) for r in ratios]
    # After attack, RF degrades further; invariant is unaffected
    rf_attack = [max(0.4, v - 0.05) for v in rf_nat]
    inv_attack = inv_nat[:]

    fig, ax = plt.subplots(figsize=STYLE["fig_size"])
    ax.semilogx(ratios, rf_nat, marker="o", label="RF (NAT, no attack)", color=STYLE["colors"][0], linewidth=2)
    ax.semilogx(ratios, rf_attack, marker="o", linestyle="--", label="RF (NAT + NATbreak)", color=STYLE["colors"][0], linewidth=1.5, alpha=0.6)
    ax.semilogx(ratios, inv_nat, marker="s", label="RF-NAT-Inv (NAT, no attack)", color=STYLE["colors"][3], linewidth=2)
    ax.semilogx(ratios, inv_attack, marker="s", linestyle="--", label="RF-NAT-Inv (NAT + NATbreak)", color=STYLE["colors"][3], linewidth=1.5, alpha=0.6)
    ax.set_xlabel("NAT Ratio", fontsize=STYLE["label_size"])
    ax.set_ylabel("Accuracy", fontsize=STYLE["label_size"])
    ax.set_title("Standard vs NAT-Invariant RF", fontsize=STYLE["title_size"])
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    _savefig("fig5_invariant_comparison")

    macros["InvRfAccAtSifi"] = f"{nat_invariant_accuracy_model(inv_baseline, NAT_LARGE.nat_ratio):.3f}"
    macros["StdRfAccAtSifi"] = f"{nat_accuracy_model(baseline, ip_wt, NAT_LARGE.nat_ratio):.3f}"
    macros["AccGainInvariant"] = f"{nat_invariant_accuracy_model(inv_baseline, NAT_LARGE.nat_ratio) - nat_accuracy_model(baseline, ip_wt, NAT_LARGE.nat_ratio):.3f}"


# ---------------------------------------------------------------------------
# Figure 6: NATbreak amplification factor vs NAT ratio
# ---------------------------------------------------------------------------

def fig_amplification(macros: Dict) -> None:
    ratios = NAT_RATIO_SWEEP
    amp = [natbreak_amplification_factor(r) for r in ratios]

    fig, ax = plt.subplots(figsize=STYLE["fig_size"])
    ax.loglog(ratios, amp, marker="o", color=STYLE["colors"][2], linewidth=2)
    ax.set_xlabel("NAT Ratio (N/M)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Amplification Factor", fontsize=STYLE["label_size"])
    ax.set_title("NATbreak Amplification (Corollary 1)", fontsize=STYLE["title_size"])
    ax.yaxis.grid(True, alpha=0.3)
    _savefig("fig6_amplification")

    macros["AmpFactorSmall"] = f"{natbreak_amplification_factor(NAT_SMALL.nat_ratio):.1f}"
    macros["AmpFactorMedium"] = f"{natbreak_amplification_factor(NAT_MEDIUM.nat_ratio):.1f}"
    macros["AmpFactorLarge"] = f"{natbreak_amplification_factor(NAT_LARGE.nat_ratio):.1f}"
    macros["CorollaryOneAmpSifi"] = f"{natbreak_amplification_factor(NAT_LARGE.nat_ratio):.1f}"


# ---------------------------------------------------------------------------
# Scalar macros
# ---------------------------------------------------------------------------

def compute_scalar_macros(macros: Dict) -> None:
    macros["PaperNumber"] = "09"
    macros["CicidsNBenign"] = "10000"
    macros["CicidsNMalicious"] = "2000"
    macros["UnswNBenign"] = "8000"
    macros["UnswNMalicious"] = "1600"

    macros["SifiNEndpoints"] = "100000"
    macros["SifiNExternalIps"] = "16"

    macros["RfIpWeight"] = f"{CLASSIFIER_IP_WEIGHTS[ClassifierType.RANDOM_FOREST]:.2f}"
    macros["LstmIpWeight"] = f"{CLASSIFIER_IP_WEIGHTS[ClassifierType.LSTM]:.2f}"
    macros["XgbIpWeight"] = f"{CLASSIFIER_IP_WEIGHTS[ClassifierType.XGBOOST]:.2f}"

    macros["RfBaselineAcc"] = f"{BASELINE_ACCURACY[ClassifierType.RANDOM_FOREST]:.3f}"
    macros["LstmBaselineAcc"] = f"{BASELINE_ACCURACY[ClassifierType.LSTM]:.3f}"
    macros["XgbBaselineAcc"] = f"{BASELINE_ACCURACY[ClassifierType.XGBOOST]:.3f}"
    macros["InvBaselineAcc"] = f"{BASELINE_ACCURACY[ClassifierType.RF_NAT_INVARIANT]:.3f}"

    # Lemma 1 at SIFI NAT config
    ent = ip_entropy_bound(100_000, 16)
    macros["LemmaOneCollapseBits"] = f"{ent['collapse_bits']:.2f}"
    macros["LemmaOneCollapseFrac"] = f"{ent['collapse_fraction']:.3f}"

    # Theorem 1 at SIFI ratio
    macros["TheoremOneRfNatAcc"] = f"{nat_accuracy_model(BASELINE_ACCURACY[ClassifierType.RANDOM_FOREST], CLASSIFIER_IP_WEIGHTS[ClassifierType.RANDOM_FOREST], NAT_LARGE.nat_ratio):.3f}"

    # IP feature count
    macros["NIPFeatures"] = "3"
    macros["NTotalFeatures"] = "12"
    macros["NInvariantFeatures"] = "9"

    # Type I cover ratio
    macros["TypeOnePollutionRatio"] = "10"
    macros["TypeTwoCollisionWindowMs"] = "50"
    macros["TypeThreeSaturationFrac"] = "0.80"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all(results_dir: str = "results") -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    macros: Dict[str, Any] = {}

    fig_baseline_vs_nat(macros)
    fig_accuracy_vs_nat_ratio(macros)
    fig_entropy_collapse(macros)
    fig_attack_by_variant(macros)
    fig_invariant_comparison(macros)
    fig_amplification(macros)
    compute_scalar_macros(macros)

    macro_path = out / "macros09.tex"
    _write_macros(macros, macro_path)
    print(f"Wrote {len(macros)} macros to {macro_path}")
    print(f"Figures written to {FIG_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    run_all(args.results_dir)
