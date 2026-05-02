#!/usr/bin/env python3
"""Confirmatory M1 figures aligned with docs/M1_OSF_Preregistration.md (Env A).

Run after aggregate_m1.py (one CSV per run). Exploratory panels live in
plot_m1_trajectory.py; this script implements preregistered visuals/summary stats.

Usage — single run (pilot or one seed):
  python scripts/plot_m1_confirmatory_figures.py \\
    --csv results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv \\
    --out results/m1_env_A_sc030/prereg_figures

Usage — full Env A campaign (auto-discover aggregates under results/):
  python scripts/plot_m1_confirmatory_figures.py \\
    --results-dir results \\
    --out results/m1_prereg_campaign_figures

Writes PNGs plus summary_prereg_figures.json in --out.

Optional: scipy (ttest_1samp, f_oneway, ttest_ind, kruskal) for H4/H3-mod;
Holm adjustment is implemented without statsmodels.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Column names (must match aggregate_m1 / analyze_checkpoint output) ---
COL_P5 = "measurement_1_probes.probe_5way_auroc_mean"
COL_N_AGG = "measurement_1_probes.n_aggressor"
COL_N_VIC = "measurement_1_probes.n_victim"
COL_CKA_AV = "measurement_2_cka.cka_agg_vs_vic"
COL_CKA_AN = "measurement_2_cka.cka_matrix.ZAP_AGENT.NEUTRAL"
COL_CKA_VN = "measurement_2_cka.cka_matrix.BEING_ZAPPED.NEUTRAL"
COL_GRAD = "measurement_4_gradient_transfer.gradient_transfer_cos_mean"
COL_VIOL = "ViolenceRate_per_agent_step"
COL_BEING = "BeingZappedRate_per_agent_step"
COL_RSA = "measurement_3_rsa.cosdist_agg_vs_vic"

# Prereg §283 — qualitative ordering check (apple_density)
H3MOD_FOCUS_DENSITIES = (0.15, 0.30, 0.50)

N_MIN = 100
H1_THRESHOLD = 0.8
H1_EP_CUTOFF = 2000
H3_ROLL = 3
H3_LAG_MAX = 10
BOOT_B = 1000
RNG = np.random.default_rng(42)


def _has_scipy():
    try:
        import scipy.stats  # noqa: F401
        return True
    except ImportError:
        return False


def rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, min_periods=1, center=True).mean()


def eligibility_mask(df: pd.DataFrame) -> pd.Series:
    if COL_N_AGG not in df.columns or COL_N_VIC not in df.columns:
        return pd.Series(False, index=df.index)
    return (df[COL_N_AGG] >= N_MIN) & (df[COL_N_VIC] >= N_MIN)


def h1_pass_run(df: pd.DataFrame) -> bool:
    if COL_P5 not in df.columns:
        return False
    early = df[df["episode"].astype(int) <= H1_EP_CUTOFF]
    if early.empty:
        return False
    return bool((early[COL_P5] > H1_THRESHOLD).any())


def seed_cka_means(df: pd.DataFrame) -> Tuple[float, float, float, int]:
    """Mean CKA triples over n_min-eligible checkpoints; returns n_eligible."""
    m = eligibility_mask(df)
    sub = df.loc[m]
    n = len(sub)
    if n == 0 or COL_CKA_AV not in sub.columns:
        return float("nan"), float("nan"), float("nan"), 0
    av = float(sub[COL_CKA_AV].mean())
    an = float(sub[COL_CKA_AN].mean()) if COL_CKA_AN in sub.columns else float("nan")
    vn = float(sub[COL_CKA_VN].mean()) if COL_CKA_VN in sub.columns else float("nan")
    return av, an, vn, n


def seed_grad_mean(df: pd.DataFrame) -> float:
    if COL_GRAD not in df.columns:
        return float("nan")
    return float(df[COL_GRAD].mean())


def seed_probe_mean(df: pd.DataFrame) -> float:
    if COL_P5 not in df.columns:
        return float("nan")
    return float(df[COL_P5].mean())


def seed_1mcka_mean(df: pd.DataFrame) -> float:
    """Checkpoint-mean (1 − CKA_agg_vic) over n_min-eligible checkpoints (§22)."""
    m = eligibility_mask(df)
    sub = df.loc[m]
    if COL_CKA_AV not in sub.columns or len(sub) == 0:
        return float("nan")
    return float((1.0 - sub[COL_CKA_AV].astype(float)).mean())


def seed_rsa_mean(df: pd.DataFrame) -> float:
    if COL_RSA not in df.columns:
        return float("nan")
    s = df[COL_RSA].astype(float)
    s = s[np.isfinite(s.to_numpy())]
    if len(s) == 0:
        return float("nan")
    return float(s.mean())


def holm_adjust_two_sided(p_values: List[float]) -> List[float]:
    """Holm–Bonferroni adjusted p-values (two-sided family-wise)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    sorted_p = p[order]
    adj_sorted = np.empty(m)
    adj_sorted[m - 1] = sorted_p[m - 1]
    for j in range(m - 2, -1, -1):
        adj_sorted[j] = min((m - j) * sorted_p[j], adj_sorted[j + 1])
    adj_sorted = np.minimum(adj_sorted, 1.0)
    out = np.empty(m)
    out[order] = adj_sorted
    return [float(x) for x in out]


def _find_scarcity_key(
    cells: List[float], target: float, rtol: float = 1e-5, atol: float = 1e-6
) -> Optional[float]:
    for c in cells:
        if np.isclose(c, target, rtol=rtol, atol=atol):
            return c
    return None


def _cell_means(by_cell: Dict[float, List[pd.DataFrame]], cells_sorted: List[float], metric_fn) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for sc in cells_sorted:
        vals = [metric_fn(df) for df in by_cell[sc]]
        vals = [v for v in vals if np.isfinite(v)]
        out[sc] = float(np.mean(vals)) if vals else float("nan")
    return out


def run_h3mod_analysis(
    by_cell: Dict[float, List[pd.DataFrame]],
    cells_sorted: List[float],
    metric_fn,
    metric_id: str,
    metric_ylabel: str,
) -> Dict:
    """One-way ANOVA / Kruskal–Wallis + pairwise Welch t + Holm (per §281–283)."""
    result: Dict = {"metric": metric_id, "ylabel": metric_ylabel}
    per_level: Dict[float, np.ndarray] = {}
    for sc in cells_sorted:
        vals = [metric_fn(df) for df in by_cell[sc]]
        arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        if len(arr) > 0:
            per_level[sc] = arr
    levels = sorted(per_level.keys())
    result["n_levels"] = len(levels)
    if len(levels) < 2:
        result["note"] = "Need ≥2 scarcity levels with data."
        return result

    arrs = [per_level[s] for s in levels]
    anova_p = kruskal_p = None
    if _has_scipy():
        from scipy import stats
        anova_p = float(stats.f_oneway(*arrs).pvalue)
        kruskal_p = float(stats.kruskal(*arrs).pvalue)

    result["anova_p"] = anova_p
    result["kruskal_wallis_p"] = kruskal_p

    pairwise: List[Dict] = []
    raw_ps: List[float] = []
    if _has_scipy():
        from scipy import stats
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                a, b = levels[i], levels[j]
                x, y = per_level[a], per_level[b]
                if len(x) < 1 or len(y) < 1:
                    continue
                tt = stats.ttest_ind(x, y, equal_var=False)
                raw_ps.append(float(tt.pvalue))
                pairwise.append(
                    {
                        "scarcity_a": float(a),
                        "scarcity_b": float(b),
                        "mean_a": float(np.mean(x)),
                        "mean_b": float(np.mean(y)),
                        "t_statistic": float(tt.statistic),
                        "p_two_sided_raw": float(tt.pvalue),
                    }
                )
    holm = holm_adjust_two_sided(raw_ps) if raw_ps else []
    for k, row in enumerate(pairwise):
        row["p_two_sided_holm"] = holm[k]

    result["pairwise_welch_holm"] = pairwise
    means = _cell_means(by_cell, cells_sorted, metric_fn)
    result["cell_means_checkpoint_averaged"] = {str(k): float(v) for k, v in means.items()}

    # §283: μ(0.15) > μ(0.30) > μ(0.50)
    k15 = _find_scarcity_key(levels, H3MOD_FOCUS_DENSITIES[0])
    k30 = _find_scarcity_key(levels, H3MOD_FOCUS_DENSITIES[1])
    k50 = _find_scarcity_key(levels, H3MOD_FOCUS_DENSITIES[2])
    m15 = means.get(k15, float("nan")) if k15 is not None else float("nan")
    m30 = means.get(k30, float("nan")) if k30 is not None else float("nan")
    m50 = means.get(k50, float("nan")) if k50 is not None else float("nan")
    ordering_ok = (
        np.isfinite(m15)
        and np.isfinite(m30)
        and np.isfinite(m50)
        and (m15 > m30 > m50)
    )
    result["ordering_means_015_gt_030_gt_050"] = bool(ordering_ok)

    def _pair_holm(sa: float, sb: float) -> Optional[float]:
        for row in pairwise:
            ra, rb = row["scarcity_a"], row["scarcity_b"]
            if (
                (np.isclose(ra, sa) and np.isclose(rb, sb))
                or (np.isclose(ra, sb) and np.isclose(rb, sa))
            ):
                return row["p_two_sided_holm"]
        return None

    p1530 = p3050 = None
    if k15 is not None and k30 is not None:
        p1530 = _pair_holm(float(k15), float(k30))
    if k30 is not None and k50 is not None:
        p3050 = _pair_holm(float(k30), float(k50))

    result["focus_pairs"] = {
        "015_vs_030_p_holm": p1530,
        "030_vs_050_p_holm": p3050,
    }
    sig1530 = p1530 is not None and p1530 < 0.05
    sig3050 = p3050 is not None and p3050 < 0.05
    result["focus_pairs_both_significant_holm_0.05"] = bool(sig1530 and sig3050)

    anova_sig = anova_p is not None and anova_p < 0.05
    result["h3_mod_confirm_this_metric"] = bool(
        anova_sig and ordering_ok and sig1530 and sig3050
    )
    return result


def plot_h3mod_campaign_panels(
    by_cell: Dict[float, List[pd.DataFrame]],
    cells_sorted: List[float],
    panel_specs: List[Tuple[Dict, object, str]],
    out: Path,
) -> None:
    """One row, three columns: probe, 1−CKA, RSA (per-seed checkpoint means)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (res, fn, short_title) in zip(axes, panel_specs):
        xs: List[float] = []
        ys: List[float] = []
        for sc in cells_sorted:
            for df in by_cell[sc]:
                v = fn(df)
                if np.isfinite(v):
                    xs.append(sc)
                    ys.append(float(v))
        ax.scatter(xs, ys, alpha=0.78)
        ax.set_xlabel("apple_density")
        ax.set_ylabel(res.get("ylabel", ""))
        anp = res.get("anova_p")
        kwp = res.get("kruskal_wallis_p")
        st = short_title
        if anp is not None and kwp is not None:
            st += f"\nANOVA p={anp:.3g} | Kruskal–Wallis p={kwp:.3g}"
        cfm = res.get("h3_mod_confirm_this_metric")
        if cfm is not None:
            st += f"\nPrereg H3-mod rule: {'PASS' if cfm else 'FAIL'}"
        ax.set_title(st, fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "H3-mod — role separability vs scarcity (§281–283: ANOVA + Holm pairwise + 0.15>0.30>0.50)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "prereg13_h3mod_scarcity_panels.png", dpi=150)
    plt.close(fig)


def crosscorr_peak_lag(
    a: np.ndarray, b: np.ndarray, lag_max: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Pearson cross-correlation; return (peak_lag, lags, corrs)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 3:
        return float("nan"), np.array([]), np.array([])
    a = a[:n]
    b = b[:n]
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    sa = np.nanstd(a)
    sb = np.nanstd(b)
    if sa < 1e-12 or sb < 1e-12:
        return float("nan"), np.array([]), np.array([])
    lags = np.arange(-lag_max, lag_max + 1)
    corrs = np.full_like(lags, np.nan, dtype=float)
    for i, L in enumerate(lags):
        if L >= 0:
            sl_a = a[: n - L]
            sl_b = b[L:n]
        else:
            k = -L
            sl_a = a[k:n]
            sl_b = b[: n - k]
        if len(sl_a) < 3:
            continue
        corrs[i] = float(np.corrcoef(sl_a, sl_b)[0, 1])
    finite = np.isfinite(corrs)
    if not finite.any():
        return float("nan"), lags, corrs
    idx = int(np.nanargmax(corrs))
    peak_lag = float(lags[idx])
    return peak_lag, lags, corrs


def h3_peak_lag_for_run(df: pd.DataFrame) -> float:
    if COL_P5 not in df.columns or COL_VIOL not in df.columns:
        return float("nan")
    df = df.sort_values("episode")
    p5 = rolling_mean(df[COL_P5].astype(float), H3_ROLL)
    v = rolling_mean(df[COL_VIOL].astype(float), H3_ROLL)
    peak, _, _ = crosscorr_peak_lag(p5.values, v.values, H3_LAG_MAX)
    return peak


def bootstrap_diff_ci(
    d1: np.ndarray, d2: np.ndarray, n_boot: int = BOOT_B
) -> Tuple[float, float, float]:
    """Percentile CI for mean(d1 - d2) resampling joint rows (seeds)."""
    x = np.asarray(d1, dtype=float)
    y = np.asarray(d2, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        diff = float(x[0] - y[0])
        return diff, diff, diff
    stat = x - y
    boots = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(stat), size=len(stat))
        boots.append(float(np.mean(stat[idx])))
    boots.sort()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(np.mean(stat)), lo, hi


def parse_scarcity_from_path(p: Path) -> Optional[float]:
    m = re.search(r"_sc(\d{3})", p.name) or re.search(r"_sc(\d{3})", str(p.parent.name))
    if m:
        return int(m.group(1)) / 100.0
    return None


def discover_aggregates(results_dir: Path) -> List[Path]:
    """Env A campaign CSVs: results/m1_env_A_sc030/aggregated*.csv etc."""
    paths = sorted(results_dir.glob("m1_env_A_sc*/aggregated*.csv"))
    if not paths:
        paths = sorted(results_dir.rglob("aggregated_m1_env_A_*.csv"))
    return paths


def plot_single_run(df: pd.DataFrame, out: Path, title_suffix: str) -> Dict:
    """Preregistered-style panels for one aggregate CSV."""
    out.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("episode")
    ep = df["episode"].astype(int)
    elig = eligibility_mask(df)
    summary: Dict = {"mode": "single_run", "title_suffix": title_suffix}

    # H1
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if COL_P5 in df.columns:
        ax.plot(ep, df[COL_P5], "o-", label="5-way probe AUROC")
    ax.axhline(H1_THRESHOLD, color="C3", linestyle="--", label=f"H1 threshold ({H1_THRESHOLD})")
    ax.axvline(H1_EP_CUTOFF, color="gray", linestyle=":", label=f"ep ≤ {H1_EP_CUTOFF}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"H1 — probe vs threshold\n{title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "prereg01_h1_probe_threshold.png", dpi=150)
    plt.close(fig)
    summary["h1_pass_this_run"] = h1_pass_run(df)

    # H2 triples
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if COL_CKA_AV in df.columns:
        ax.plot(ep, df[COL_CKA_AV], "o-", label="CKA(agg,vic)", alpha=0.9)
    if COL_CKA_AN in df.columns:
        ax.plot(ep, df[COL_CKA_AN], "s-", label="CKA(agg,neutral)", alpha=0.85)
    if COL_CKA_VN in df.columns:
        ax.plot(ep, df[COL_CKA_VN], "^-", label="CKA(vic,neutral)", alpha=0.85)
    for i, e in enumerate(ep):
        if elig.iloc[i]:
            ax.axvspan(e - 80, e + 80, color="green", alpha=0.06, zorder=0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("CKA")
    ax.set_title(f"H2 — role-pair CKA (green band = n_min-eligible ckpt)\n{title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "prereg02_h2_cka_role_pairs.png", dpi=150)
    plt.close(fig)
    av, an, vn, n_elig = seed_cka_means(df)
    summary["h2_seed_means"] = {"cka_agg_vic": av, "cka_agg_neutral": an, "cka_vic_neutral": vn, "n_eligible_ckpts": n_elig}

    # H3 cross-correlation
    if COL_P5 in df.columns and COL_VIOL in df.columns:
        p5s = rolling_mean(df[COL_P5].astype(float), H3_ROLL)
        vs = rolling_mean(df[COL_VIOL].astype(float), H3_ROLL)
        peak, lags, corrs = crosscorr_peak_lag(p5s.values, vs.values, H3_LAG_MAX)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(ep, p5s, "o-", label="5-way AUROC (3-ckpt mean)")
        axes[0].plot(ep, vs, "s-", label="Violence rate (3-ckpt mean)", alpha=0.85)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Smoothed value")
        axes[0].set_title("H3 — smoothed trajectories")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[1].bar(lags.astype(float), corrs, color="C0", width=0.8)
        axes[1].axvline(0, color="gray", linestyle=":")
        axes[1].set_xlabel("Lag (checkpoints; + = probe leads violence)")
        axes[1].set_ylabel("Pearson r")
        axes[1].set_title(f"H3 — cross-correlation (peak lag = {peak:.1f})")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(title_suffix, fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "prereg03_h3_xcorr_probe_violence.png", dpi=150)
        plt.close(fig)
        summary["h3_peak_lag_checkpoints"] = peak
    else:
        summary["h3_peak_lag_checkpoints"] = None

    # H4 gradient
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if COL_GRAD in df.columns:
        ax.plot(ep, df[COL_GRAD], "o-", label="Mean cos(grad V_vic, grad π_agg)")
    ax.axhline(0.0, color="gray", linestyle=":", label="μ₀ = 0")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"H4 — cross-role gradient transfer\n{title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "prereg04_h4_gradient_transfer.png", dpi=150)
    plt.close(fig)
    summary["h4_mean_grad_cos"] = seed_grad_mean(df)

    # n_min diagnostics
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if COL_N_AGG in df.columns:
        ax.plot(ep, df[COL_N_AGG], "o-", label="n ZAP_AGENT")
    if COL_N_VIC in df.columns:
        ax.plot(ep, df[COL_N_VIC], "s-", label="n BEING_ZAPPED")
    ax.axhline(N_MIN, color="C3", linestyle="--", label=f"n_min = {N_MIN}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Rollout row counts")
    ax.set_title(f"n_min eligibility (H2 CKA / agg-vic probe primaries)\n{title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "prereg05_nmin_diagnostics.png", dpi=150)
    plt.close(fig)
    summary["fraction_ckpts_eligible_h2"] = float(elig.mean()) if len(df) else 0.0

    return summary


def campaign_analysis(csv_paths: List[Path], out: Path) -> Dict:
    """Pool Env A runs: H1 cell counts, H2 bootstrap forest, H3 lags, H4/H3-mod tests."""
    out.mkdir(parents=True, exist_ok=True)
    by_cell: Dict[float, List[pd.DataFrame]] = defaultdict(list)
    for p in csv_paths:
        df = pd.read_csv(p)
        sc = None
        if "scarcity" in df.columns and df["scarcity"].notna().any():
            sc = float(df["scarcity"].iloc[0])
        if sc is None:
            sc = parse_scarcity_from_path(p)
        if sc is None:
            continue
        by_cell[sc].append(df)

    cells_sorted = sorted(by_cell.keys())
    summary: Dict = {"mode": "campaign", "scarcity_cells": cells_sorted, "cells_detail": {}}

    # H1: ≥2 of 5 cells with any seed passing
    h1_cell_pass = {}
    for sc in cells_sorted:
        passes = [h1_pass_run(d) for d in by_cell[sc]]
        h1_cell_pass[sc] = any(passes)
    summary["h1_cells_passing"] = {str(k): v for k, v in h1_cell_pass.items()}
    summary["h1_confirm"] = sum(h1_cell_pass.values()) >= 2

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(cells_sorted))
    colors = ["C2" if h1_cell_pass[s] else "C3" for s in cells_sorted]
    ax.bar(xs, [1 if h1_cell_pass[s] else 0 for s in cells_sorted], color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s:.2f}" for s in cells_sorted])
    ax.set_xlabel("apple_density (scarcity)")
    ax.set_ylabel("Any seed passes H1 (AUROC>0.8 by ep 2000)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"H1 — pass by scarcity (confirm if ≥2 / {len(cells_sorted)} cells)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "prereg10_h1_pass_by_scarcity.png", dpi=150)
    plt.close(fig)

    # H2 forest: two differences per cell
    rows_forest = []
    h2_d1_pass = 0
    h2_d2_pass = 0
    for sc in cells_sorted:
        avs, ans, vns = [], [], []
        for df in by_cell[sc]:
            a, an, vn, n_e = seed_cka_means(df)
            if np.isfinite(a) and np.isfinite(an) and np.isfinite(vn):
                avs.append(a)
                ans.append(an)
                vns.append(vn)
        avs, ans, vns = np.array(avs), np.array(ans), np.array(vns)
        m1, lo1, hi1 = bootstrap_diff_ci(avs, ans)
        m2, lo2, hi2 = bootstrap_diff_ci(avs, vns)
        rows_forest.append(
            {
                "sc": sc,
                "diff_av_an": m1,
                "lo1": lo1,
                "hi1": hi1,
                "diff_av_vn": m2,
                "lo2": lo2,
                "hi2": hi2,
                "n_seeds_total": len(by_cell[sc]),
                "n_seeds_h2_paired": int(len(avs)),
            }
        )
        if np.isfinite(hi1) and hi1 < 0:
            h2_d1_pass += 1
        if np.isfinite(hi2) and hi2 < 0:
            h2_d2_pass += 1
        summary["cells_detail"][str(sc)] = {
            "h2_bootstrap_cka_agg_minus_agg_neutral": {"mean": m1, "ci95_lo": lo1, "ci95_hi": hi1},
            "h2_bootstrap_cka_agg_minus_vic_neutral": {"mean": m2, "ci95_lo": lo2, "ci95_hi": hi2},
            "n_seeds_total": len(by_cell[sc]),
            "n_seeds_h2_paired": int(len(avs)),
        }

    summary["h2_cells_diff1_entirely_negative_ci"] = h2_d1_pass
    summary["h2_cells_diff2_entirely_negative_ci"] = h2_d2_pass
    summary["h2_confirm"] = h2_d1_pass >= 3 and h2_d2_pass >= 3

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(cells_sorted) * 2)))
    ax.axvline(0, color="gray", linestyle=":")
    y0 = 0
    for j, sc in enumerate(cells_sorted):
        r = rows_forest[j]
        y1 = y0 + 0.35
        y2 = y0 + 0.85
        ax.plot([r["lo1"], r["hi1"]], [y1, y1], "C0-", lw=3, solid_capstyle="round")
        ax.plot([r["lo2"], r["hi2"]], [y2, y2], "C1-", lw=3, solid_capstyle="round")
        ax.text(r["hi1"] + 0.01, y1, "CKA(agg,vic)−CKA(agg,neut)", fontsize=7, va="center")
        ax.text(r["hi2"] + 0.01, y2, "CKA(agg,vic)−CKA(vic,neut)", fontsize=7, va="center")
        ax.text(-0.5, y0 + 0.55, f"sc={sc:.2f} ({r['n_seeds_h2_paired']} seeds w/ full H2 CKA triple)", fontsize=9, va="center")
        y0 += 1.4
    ax.set_yticks([])
    ax.set_xlabel("Bootstrap 95% CI for mean CKA difference (negative supports H2)")
    ax.set_title("H2 — seed-resampled CI per scarcity cell (1000 resamples)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.subplots_adjust(left=0.22)
    fig.savefig(out / "prereg11_h2_bootstrap_forest.png", dpi=150)
    plt.close(fig)

    # H3 peak lags
    lag_records = []
    for sc in cells_sorted:
        for df in by_cell[sc]:
            pk = h3_peak_lag_for_run(df)
            lag_records.append({"scarcity": sc, "peak_lag": pk})
    summary["h3_peak_lags"] = lag_records
    mean_lag = float(np.nanmean([r["peak_lag"] for r in lag_records])) if lag_records else float("nan")
    summary["h3_mean_peak_lag"] = mean_lag
    summary["h3_confirm"] = np.isfinite(mean_lag) and mean_lag > 0

    fig, ax = plt.subplots(figsize=(8, 4))
    scs = [r["scarcity"] for r in lag_records]
    lags = [r["peak_lag"] for r in lag_records]
    ax.scatter(scs, lags, alpha=0.8)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xlabel("apple_density")
    ax.set_ylabel("Peak cross-corr lag (checkpoints)")
    ax.set_title("H3 — peak lags (positive = representation leads violence)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "prereg12_h3_peak_lags.png", dpi=150)
    plt.close(fig)

    # H4: per cell, mean grad per seed → t-test vs 0
    h4_detail = {}
    for sc in cells_sorted:
        vals = [seed_grad_mean(d) for d in by_cell[sc]]
        vals_arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        mu = float(np.mean(vals_arr)) if len(vals_arr) else float("nan")
        pval = None
        if _has_scipy() and len(vals_arr) >= 2:
            from scipy import stats
            pval = float(stats.ttest_1samp(vals_arr, 0.0, alternative="greater").pvalue)
        elif _has_scipy() and len(vals_arr) == 1:
            from scipy import stats
            pval = float(stats.ttest_1samp(vals_arr, 0.0, alternative="greater").pvalue)
        # Prereg §278–279: gradient disconnect if mean ≲ 0 or insufficient evidence that mean > 0.
        cell_ok = (np.isfinite(mu) and mu <= 0) or (
            pval is not None and pval >= 0.05
        )
        h4_detail[str(sc)] = {
            "per_seed_mean_grad": [float(x) for x in vals_arr],
            "mean": mu,
            "ttest_1sample_greater0_p": pval,
            "h4_cell_pass": bool(cell_ok),
        }
    summary["h4_by_cell"] = h4_detail

    # H3-mod (§281–283): three operationalizations × ANOVA / Kruskal + Holm pairwise
    h3_probe = run_h3mod_analysis(
        by_cell,
        cells_sorted,
        seed_probe_mean,
        "probe_5way_auroc_mean",
        "Checkpoint-mean 5-way probe AUROC",
    )
    h3_1mcka = run_h3mod_analysis(
        by_cell,
        cells_sorted,
        seed_1mcka_mean,
        "one_minus_cka_agg_vic",
        "Checkpoint-mean (1 − CKA agg↔vic), n_min-eligible ckpts",
    )
    h3_rsa = run_h3mod_analysis(
        by_cell,
        cells_sorted,
        seed_rsa_mean,
        "rsa_cosdist_agg_vic",
        "Checkpoint-mean RSA cos-dist (agg↔vic)",
    )
    summary["h3_mod"] = {
        "probe_5way_auroc": h3_probe,
        "one_minus_cka_agg_vic": h3_1mcka,
        "rsa_cosdist_agg_vic": h3_rsa,
        "h3_mod_confirm_probe_only": bool(h3_probe.get("h3_mod_confirm_this_metric")),
        "h3_mod_confirm_all_three": (
            bool(h3_probe.get("h3_mod_confirm_this_metric"))
            and bool(h3_1mcka.get("h3_mod_confirm_this_metric"))
            and bool(h3_rsa.get("h3_mod_confirm_this_metric"))
        ),
    }
    summary["h3_mod_anova_p_probe"] = h3_probe.get("anova_p")

    plot_h3mod_campaign_panels(
        by_cell,
        cells_sorted,
        [
            (h3_probe, seed_probe_mean, "Probe AUROC"),
            (h3_1mcka, seed_1mcka_mean, "1 − CKA (agg,vic)"),
            (h3_rsa, seed_rsa_mean, "RSA cos-dist (agg,vic)"),
        ],
        out,
    )

    summary["scipy_available"] = _has_scipy()
    if not _has_scipy():
        summary["note"] = (
            "Install scipy for H4 t-tests, H3-mod ANOVA/Kruskal/Welch pair tests, "
            "and Holm-adjusted pairwise p-values (Holm math is local; raw p needs scipy)."
        )

    return summary


def main():
    parser = argparse.ArgumentParser(description="M1 confirmatory figures (OSF prereg)")
    parser.add_argument("--csv", type=Path, default=None, help="Single aggregate CSV")
    parser.add_argument("--results-dir", type=Path, default=None, help="Discover m1_env_A_sc*/aggregated*.csv")
    parser.add_argument("--out", type=Path, default=Path("results/prereg_figures"))
    args = parser.parse_args()

    if args.csv and not args.csv.is_file():
        raise SystemExit(f"Missing --csv: {args.csv}")

    bundle: Dict = {}

    if args.results_dir is not None:
        paths = discover_aggregates(args.results_dir)
        if not paths:
            print(f"[prereg-fig] no aggregates under {args.results_dir} (m1_env_A_sc*/aggregated*.csv)")
        else:
            print(f"[prereg-fig] campaign mode: {len(paths)} CSVs")
            bundle["campaign"] = campaign_analysis(paths, args.out)

    if args.csv is not None:
        df = pd.read_csv(args.csv)
        cfg = str(df["config"].iloc[0]) if "config" in df.columns else "run"
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else -1
        suffix = f"{cfg} | seed {seed}"
        bundle["single_run"] = plot_single_run(df, args.out, suffix)

    if not bundle:
        raise SystemExit("Provide --csv and/or --results-dir")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary_prereg_figures.json").open("w") as f:
        json.dump(bundle, f, indent=2, default=float)

    print(f"[prereg-fig] wrote summary → {args.out / 'summary_prereg_figures.json'}")


if __name__ == "__main__":
    main()
