#!/usr/bin/env python3
"""Behavior-only M0 ecology calibration plots.

This script intentionally excludes representation metrics and summarizes
Stage 0 behavior by ecology candidate and ecology knobs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BEHAVIOR_COLS = [
    "ViolenceRate_per_agent_step",
    "BeingZappedRate_per_agent_step",
    "BeamUseRate_per_agent_step",
    "AppleRate_per_agent_step",
    "AvgReturn_per_agent",
]


def parse_ecology_from_name(name: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "env": None,
        "ecology_id": None,
        "num_agents": None,
        "apple_density": None,
        "regrowth_speed": None,
        "zap_timeout": None,
    }
    m = re.search(
        r"env_(?P<env>[A-Za-z])_[A-Za-z]+_n(?P<n>\d+)_ad(?P<ad>\d{3})_rg(?P<rg>\d{3})_zt(?P<zt>\d{2,3})",
        name,
    )
    if not m:
        return out
    out["env"] = m.group("env")
    out["num_agents"] = float(int(m.group("n")))
    out["apple_density"] = int(m.group("ad")) / 100.0
    out["regrowth_speed"] = int(m.group("rg")) / 100.0
    out["zap_timeout"] = float(int(m.group("zt")))
    out["ecology_id"] = (
        f"env{out['env']}_n{int(out['num_agents'])}_"
        f"ad{int(out['apple_density'] * 100):03d}_"
        f"rg{int(out['regrowth_speed'] * 100):03d}_"
        f"zt{int(out['zap_timeout']):02d}"
    )
    return out


def collect_stage0_csvs(results_dir: Path) -> List[Path]:
    paths = sorted(results_dir.rglob("stage0_env_*/*_baseline_seed*.csv"))
    return paths


def load_runs(csv_paths: List[Path], late_start_episode: int) -> pd.DataFrame:
    rows = []
    for p in csv_paths:
        meta = parse_ecology_from_name(p.stem)
        if not meta.get("ecology_id"):
            continue
        df = pd.read_csv(p)
        if "Episode" not in df.columns:
            continue
        seed_m = re.search(r"_seed(?P<seed>\d+)$", p.stem)
        seed = int(seed_m.group("seed")) if seed_m else -1
        late = df[df["Episode"].astype(int) >= late_start_episode]
        if late.empty:
            continue
        rec = {
            "seed": seed,
            "source_csv": str(p),
            **meta,
            "resource_pressure": 1.0 / float(meta["regrowth_speed"]),
            "late_start_episode": late_start_episode,
            "late_n_checkpoints": int(len(late)),
        }
        for c in BEHAVIOR_COLS:
            rec[c] = float(late[c].mean()) if c in late.columns else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def bootstrap_ci(vals: np.ndarray, n_boot: int = 1000) -> tuple[float, float, float]:
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


def summarize_by_ecology(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for ecology_id, g in df.groupby("ecology_id", sort=True):
        row = {
            "ecology_id": ecology_id,
            "env": g["env"].iloc[0],
            "num_agents": float(g["num_agents"].iloc[0]),
            "apple_density": float(g["apple_density"].iloc[0]),
            "regrowth_speed": float(g["regrowth_speed"].iloc[0]),
            "resource_pressure": float(g["resource_pressure"].iloc[0]),
            "zap_timeout": float(g["zap_timeout"].iloc[0]),
            "n_seeds": int(g["seed"].nunique()),
        }
        for c in BEHAVIOR_COLS:
            mu, lo, hi = bootstrap_ci(g[c].to_numpy())
            row[f"{c}_mean"] = mu
            row[f"{c}_ci95_lo"] = lo
            row[f"{c}_ci95_hi"] = hi
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(["resource_pressure", "zap_timeout", "num_agents"])


def plot_behavior_by_ecology(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "ViolenceRate_per_agent_step",
        "BeingZappedRate_per_agent_step",
        "BeamUseRate_per_agent_step",
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.5))
    labels = summary["ecology_id"].tolist()
    xs = np.arange(len(labels))
    for ax, m in zip(axes, metrics):
        means = summary[f"{m}_mean"].to_numpy(float)
        los = summary[f"{m}_ci95_lo"].to_numpy(float)
        his = summary[f"{m}_ci95_hi"].to_numpy(float)
        yerr = np.vstack([means - los, his - means])
        ax.errorbar(xs, means, yerr=yerr, fmt="o", capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
    fig.suptitle("M0 behavioral ecology calibration (late-window seed means)")
    fig.tight_layout()
    fig.savefig(out_dir / "m0_behavior_by_ecology_candidate.png", dpi=150)
    plt.close(fig)


def plot_behavior_vs_knob(summary: pd.DataFrame, knob: str, out_dir: Path, filename: str) -> None:
    metrics = ["ViolenceRate_per_agent_step", "BeingZappedRate_per_agent_step", "BeamUseRate_per_agent_step"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13.5, 4.5))
    for ax, m in zip(axes, metrics):
        x = summary[knob].to_numpy(float)
        y = summary[f"{m}_mean"].to_numpy(float)
        labels = summary["ecology_id"].tolist()
        ax.scatter(x, y)
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi), fontsize=7, alpha=0.8)
        ax.set_xlabel(knob)
        ax.set_ylabel(m)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"M0 behavior vs {knob}")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_sanity(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    x = summary["ecology_id"].tolist()
    for ax, metric in zip(axes, ["AppleRate_per_agent_step", "AvgReturn_per_agent"]):
        means = summary[f"{metric}_mean"].to_numpy(float)
        ax.bar(np.arange(len(x)), means)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("M0 sanity panels (apple and return)")
    fig.tight_layout()
    fig.savefig(out_dir / "m0_apple_return_sanity_panels.png", dpi=150)
    plt.close(fig)


def build_summary_payload(summary_df: pd.DataFrame, late_start_episode: int) -> Dict:
    return {
        "analysis_layer": "M0_behavioral_ecology",
        "confirmatory_scope": "Behavioral ecology calibration only; no representation metrics used for selection.",
        "x_axis_semantics": "Ecology candidate and ecology knobs (resource_pressure, zap_timeout).",
        "primary_unit": "candidate ecology x seed (late-window behavior means)",
        "uses_representation_metrics_for_selection": False,
        "late_start_episode": int(late_start_episode),
        "n_ecology_candidates": int(len(summary_df)),
        "ecology_candidates": summary_df.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--late-start-episode", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = collect_stage0_csvs(Path(args.results_dir))
    if not csv_paths:
        raise SystemExit("No stage0 baseline CSVs found.")

    run_df = load_runs(csv_paths, late_start_episode=args.late_start_episode)
    if run_df.empty:
        raise SystemExit("No valid stage0 run rows after parsing.")

    summary_df = summarize_by_ecology(run_df)
    summary_df.to_csv(out_dir / "m0_freeze_candidate_summary.csv", index=False)

    plot_behavior_by_ecology(summary_df, out_dir)
    plot_behavior_vs_knob(summary_df, "resource_pressure", out_dir, "m0_behavior_by_resource_pressure.png")
    plot_behavior_vs_knob(summary_df, "zap_timeout", out_dir, "m0_behavior_by_zap_timeout.png")
    plot_sanity(summary_df, out_dir)

    payload = build_summary_payload(summary_df, late_start_episode=args.late_start_episode)
    (out_dir / "summary_m0_ecology_calibration.json").write_text(json.dumps(payload, indent=2))
    print(f"[m0] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
