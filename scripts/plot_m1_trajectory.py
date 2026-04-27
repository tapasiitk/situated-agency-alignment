#!/usr/bin/env python3
"""Plot M1 trajectory from aggregate_m1.py CSV (local or synced from VM).

Usage:
  python scripts/plot_m1_trajectory.py \\
    --csv results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv \\
    --out results/m1_env_A_sc030/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/m1_env_A_sc030/plots"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.sort_values("episode")
    ep = df["episode"].astype(int)

    cfg = df["config"].iloc[0] if "config" in df.columns else "m1"
    seed = int(df["seed"].iloc[0]) if "seed" in df.columns else 0
    title_suffix = f"{cfg} | seed {seed}"

    args.out.mkdir(parents=True, exist_ok=True)

    # --- 1. Behaviour (training log merge) ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if "ViolenceRate_per_agent_step" in df.columns:
        ax.plot(ep, df["ViolenceRate_per_agent_step"], "o-", label="Violence (ZAP_AGENT / agent-step)")
    if "BeingZappedRate_per_agent_step" in df.columns:
        ax.plot(ep, df["BeingZappedRate_per_agent_step"], "s--", label="Being zapped / agent-step", alpha=0.85)
    ax.set_xlabel("Training episode (checkpoint)")
    ax.set_ylabel("Rate")
    ax.set_title(f"Behaviour vs checkpoint\n{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "01_behaviour_rates.png", dpi=150)
    plt.close(fig)

    # --- 2. Return & apples ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if "AvgReturn_per_agent" in df.columns:
        ax.plot(ep, df["AvgReturn_per_agent"], "o-", color="C0", label="Avg return / agent")
    ax.set_xlabel("Training episode (checkpoint)")
    ax.set_ylabel("Return")
    ax2 = ax.twinx()
    if "AppleRate_per_agent_step" in df.columns:
        ax2.plot(ep, df["AppleRate_per_agent_step"], "s-", color="C2", alpha=0.8, label="Apple rate")
    ax.set_title(f"Return & apple rate\n{title_suffix}")
    ax.grid(True, alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out / "02_return_apples.png", dpi=150)
    plt.close(fig)

    # --- 3. Linear probes ---
    p5 = "measurement_1_probes.probe_5way_auroc_mean"
    p5s = "measurement_1_probes.probe_5way_auroc_std"
    pbin = "measurement_1_probes.probe_agg_vs_vic_auroc"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if p5 in df.columns:
        ax.plot(ep, df[p5], "o-", label="5-way probe AUROC")
        if p5s in df.columns:
            lo = df[p5] - df[p5s]
            hi = df[p5] + df[p5s]
            ax.fill_between(ep, lo, hi, alpha=0.2)
    if pbin in df.columns:
        ax.plot(ep, df[pbin], "s-", label="Binary agg vs vic AUROC", alpha=0.9)
    ax.axhline(0.5, color="gray", linestyle=":", label="chance (binary)")
    ax.set_xlabel("Training episode (checkpoint)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Linear probes on embedding\n{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "03_linear_probes.png", dpi=150)
    plt.close(fig)

    # --- 4. CKA & prototype distance ---
    cka = "measurement_2_cka.cka_agg_vs_vic"
    rsa = "measurement_3_rsa.cosdist_agg_vs_vic"
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    if cka in df.columns:
        ax1.plot(ep, df[cka], "o-", color="C0", label="CKA (agg, vic)")
    ax1.set_xlabel("Training episode (checkpoint)")
    ax1.set_ylabel("CKA agg↔vic", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    if rsa in df.columns:
        ax2.plot(ep, df[rsa], "s-", color="C3", alpha=0.85, label="Cos dist (prototypes)")
    ax2.set_ylabel("Prototype cos-dist agg↔vic", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title(f"CKA & prototype separation\n{title_suffix}")
    ax1.grid(True, alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out / "04_cka_prototype.png", dpi=150)
    plt.close(fig)

    # --- 5. Gradient transfer ---
    gm = "measurement_4_gradient_transfer.gradient_transfer_cos_mean"
    gs = "measurement_4_gradient_transfer.gradient_transfer_cos_std"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if gm in df.columns:
        ax.plot(ep, df[gm], "o-", label="Mean cos(grad V, grad log π zap)")
        if gs in df.columns:
            ax.fill_between(
                ep,
                df[gm] - df[gs],
                df[gm] + df[gs],
                alpha=0.25,
                label="±1 std",
            )
    ax.axhline(0.0, color="gray", linestyle=":", label="orthogonal")
    ax.set_xlabel("Training episode (checkpoint)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Cross-role gradient transfer\n{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "05_gradient_transfer.png", dpi=150)
    plt.close(fig)

    # --- 6. Zap counts in eval rollouts (diagnostic) ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    na = "measurement_1_probes.n_aggressor"
    nv = "measurement_1_probes.n_victim"
    if na in df.columns:
        ax.plot(ep, df[na], "o-", label="n ZAP_AGENT (20 eval eps)")
    if nv in df.columns:
        ax.plot(ep, df[nv], "s-", label="n BEING_ZAPPED")
    ax.set_xlabel("Training episode (checkpoint)")
    ax.set_ylabel("Count (rollout rows)")
    ax.set_title(f"Rare-event counts in eval rollouts\n{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "06_zap_counts_eval.png", dpi=150)
    plt.close(fig)

    # --- 7. Summary 2×2 ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    if "ViolenceRate_per_agent_step" in df.columns:
        ax.plot(ep, df["ViolenceRate_per_agent_step"], "o-", color="C1")
    ax.set_title("Violence rate")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if p5 in df.columns:
        ax.plot(ep, df[p5], "o-", color="C0")
    ax.set_title("5-way probe AUROC")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if cka in df.columns:
        ax.plot(ep, df[cka], "o-", color="C2")
    ax.set_title("CKA agg↔vic")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if gm in df.columns:
        ax.plot(ep, df[gm], "o-", color="C4")
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_title("Gradient cos (mean)")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"M1 trajectory summary | {title_suffix}", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "07_summary_2x2.png", dpi=150)
    plt.close(fig)

    print(f"Wrote figures to {args.out.resolve()}")


if __name__ == "__main__":
    main()
