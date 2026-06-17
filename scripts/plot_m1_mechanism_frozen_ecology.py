#!/usr/bin/env python3
"""M1 mechanism plotting for frozen ecology (seed x checkpoint).

Consumes one or more aggregate CSVs from the same frozen ecology and outputs
seed-wise trajectories, late-window seed forest plots, attrition, and summary JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL_H1 = "measurement_1_probes.probe_agg_vs_vic_auroc"
COL_H2 = "measurement_4_gradient_transfer.gradient_transfer_cos_mean"
COL_N_AGG = "measurement_1_probes.n_aggressor"
COL_N_VIC = "measurement_1_probes.n_victim"
COL_VIOL = "ViolenceRate_per_agent_step"


def bootstrap_mean_ci(vals: np.ndarray, n_boot: int = 1000) -> Tuple[float, float, float]:
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        v = float(x[0])
        return v, v, v
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        boots.append(float(np.mean(x[idx])))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(np.mean(x)), float(lo), float(hi)


def load_aggregate_csvs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["_source_csv"] = str(p)
        frames.append(df)
    if not frames:
        raise ValueError("No aggregate CSVs provided")
    all_df = pd.concat(frames, ignore_index=True)
    if "episode" not in all_df.columns or "seed" not in all_df.columns:
        raise ValueError("Aggregate CSV must contain episode and seed columns")
    return all_df.sort_values(["seed", "episode"]).reset_index(drop=True)


def eligibility_mask(df: pd.DataFrame, n_min: int) -> pd.Series:
    if COL_N_AGG not in df.columns or COL_N_VIC not in df.columns:
        return pd.Series(False, index=df.index)
    return (df[COL_N_AGG] >= n_min) & (df[COL_N_VIC] >= n_min)


def late_mask(df: pd.DataFrame, late_start_episode: int) -> pd.Series:
    return df["episode"].astype(int) >= late_start_episode


def plot_behavior_trajectory(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for seed, g in df.groupby("seed"):
        if COL_VIOL in g.columns:
            ax.plot(g["episode"], g[COL_VIOL], marker="o", label=f"seed {seed}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(COL_VIOL)
    ax.set_title("M1 behavior trajectory by seed")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "m1_behavior_trajectory_by_seed.png", dpi=150)
    plt.close(fig)


def plot_metric_trajectory(df: pd.DataFrame, metric_col: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for seed, g in df.groupby("seed"):
        if metric_col in g.columns:
            ax.plot(g["episode"], g[metric_col], marker="o", label=f"seed {seed}")
    if metric_col == COL_H1:
        ax.axhline(0.5, color="gray", linestyle=":")
    if metric_col == COL_H2:
        ax.axhline(0.0, color="gray", linestyle=":")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_seed_late_means(df: pd.DataFrame, metric_col: str, n_min: int, late_start_episode: int) -> pd.DataFrame:
    rows = []
    for seed, g in df.groupby("seed"):
        mask = late_mask(g, late_start_episode) & eligibility_mask(g, n_min)
        sub = g.loc[mask]
        vals = sub[metric_col].astype(float).to_numpy() if metric_col in sub.columns else np.array([])
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "seed": int(seed),
                "metric": metric_col,
                "late_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "n_eligible_late_checkpoints": int(len(vals)),
                "n_total_checkpoints": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("seed")


def plot_seed_forest(seed_df: pd.DataFrame, title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, max(4.0, 0.45 * len(seed_df))))
    y = np.arange(len(seed_df))
    x = seed_df["late_mean"].to_numpy(float)
    ax.plot(x, y, "o")
    for i, row in seed_df.reset_index(drop=True).iterrows():
        ax.text(float(row["late_mean"]) if np.isfinite(row["late_mean"]) else 0.0, i, f"  seed {int(row['seed'])} (n={int(row['n_eligible_late_checkpoints'])})", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(y))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attrition(seed_h1: pd.DataFrame, seed_h2: pd.DataFrame, out_path: Path) -> None:
    merged = seed_h1[["seed", "n_eligible_late_checkpoints"]].merge(
        seed_h2[["seed", "n_eligible_late_checkpoints"]], on="seed", suffixes=("_h1", "_h2")
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    x = np.arange(len(merged))
    w = 0.36
    ax.bar(x - w / 2, merged["n_eligible_late_checkpoints_h1"], width=w, label="H1 eligible late ckpts")
    ax.bar(x + w / 2, merged["n_eligible_late_checkpoints_h2"], width=w, label="H2 eligible late ckpts")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {int(s)}" for s in merged["seed"]])
    ax.set_ylabel("Eligible late checkpoints")
    ax.set_title("M1 n_min attrition by seed")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def resolve_csvs(csv_arg: List[str], results_dir: str | None) -> List[Path]:
    paths: List[Path] = [Path(p) for p in csv_arg]
    if results_dir:
        paths.extend(sorted(Path(results_dir).rglob("aggregated*.csv")))
    uniq = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp.exists() and rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", default=[], help="Aggregate CSV path; pass multiple times")
    parser.add_argument("--results-dir", type=str, default=None, help="Optional directory to auto-discover aggregate CSVs")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--late-start-episode", type=int, default=2000)
    parser.add_argument("--n-min", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = resolve_csvs(args.csv, args.results_dir)
    if not csv_paths:
        raise SystemExit("No aggregate CSVs found.")

    df = load_aggregate_csvs(csv_paths)
    if "ecology_id" in df.columns and df["ecology_id"].notna().any():
        eco_ids = sorted(set(df["ecology_id"].dropna().astype(str).tolist()))
    else:
        eco_ids = ["unknown"]
    if len(eco_ids) > 1:
        raise SystemExit(f"Expected one frozen ecology, found multiple ecology_ids: {eco_ids}")

    plot_behavior_trajectory(df, out_dir)
    plot_metric_trajectory(
        df,
        COL_H1,
        "AUROC",
        "M1 H1 binary probe trajectory by seed",
        out_dir / "m1_h1_binary_probe_trajectory_by_seed.png",
    )
    plot_metric_trajectory(
        df,
        COL_H2,
        "Cosine similarity",
        "M1 H2 gradient transfer trajectory by seed",
        out_dir / "m1_h2_gradient_transfer_trajectory_by_seed.png",
    )

    seed_h1 = per_seed_late_means(df, COL_H1, n_min=args.n_min, late_start_episode=args.late_start_episode)
    seed_h2 = per_seed_late_means(df, COL_H2, n_min=args.n_min, late_start_episode=args.late_start_episode)

    plot_seed_forest(
        seed_h1,
        "M1 H1 late-window seed means",
        "Late-window AUROC",
        out_dir / "m1_h1_late_window_seed_forest.png",
    )
    plot_seed_forest(
        seed_h2,
        "M1 H2 late-window seed means",
        "Late-window gradient-transfer cosine",
        out_dir / "m1_h2_late_window_seed_forest.png",
    )
    plot_attrition(seed_h1, seed_h2, out_dir / "m1_nmin_attrition_by_seed.png")

    h1_mu, h1_lo, h1_hi = bootstrap_mean_ci(seed_h1["late_mean"].to_numpy(float))
    h2_mu, h2_lo, h2_hi = bootstrap_mean_ci(seed_h2["late_mean"].to_numpy(float))

    summary: Dict = {
        "analysis_layer": "M1_frozen_ecology_mechanism",
        "confirmatory_scope": "Frozen-ecology baseline mechanism test over seeds; no scarcity-cell pass/fail logic.",
        "x_axis_semantics": "Training checkpoint/episode for trajectories; seed-level late-window means for summaries.",
        "primary_unit": "seed x checkpoint",
        "uses_representation_metrics_for_selection": False,
        "ecology_id": eco_ids[0],
        "n_csvs": len(csv_paths),
        "n_seeds": int(df["seed"].nunique()),
        "late_start_episode": int(args.late_start_episode),
        "n_min": int(args.n_min),
        "h1_binary_probe": {
            "metric": COL_H1,
            "seed_late_means": seed_h1.to_dict(orient="records"),
            "bootstrap_mean": h1_mu,
            "bootstrap_ci95": [h1_lo, h1_hi],
        },
        "h2_gradient_transfer": {
            "metric": COL_H2,
            "seed_late_means": seed_h2.to_dict(orient="records"),
            "bootstrap_mean": h2_mu,
            "bootstrap_ci95": [h2_lo, h2_hi],
        },
    }
    (out_dir / "summary_m1_mechanism.json").write_text(json.dumps(summary, indent=2))
    print(f"[m1] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
