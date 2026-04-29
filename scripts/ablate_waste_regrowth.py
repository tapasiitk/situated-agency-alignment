"""Sanity check for the waste-regrowth suppression knob.

Builds HarvestDualEnv instances for Env A, the original Env B, and one or
more symmetric Env B variants (sweeping `waste_regrowth_suppression` alpha),
then steps each for `--steps` with a fixed seed and identical uniform-random
action streams. Records per-step APPLE and WASTE counts and prints a summary
table. The goal is to find the smallest alpha that:

  1. Makes cleanup *instrumental*: the symmetric Env B with that alpha shows
     a clearly lower mean apple count than the original Env B (so waste
     accumulation matters ecologically).
  2. Keeps Env B_sym roughly comparable to Env A in mean apples (so the only
     structural difference between A and B_sym is beam semantics, not
     resource availability).

Usage:
    python scripts/ablate_waste_regrowth.py --steps 1000 --seed 0
    python scripts/ablate_waste_regrowth.py --steps 1000 --seed 0 \\
        --alphas 0.05,0.10,0.15,0.20,0.30
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from karmic_rl.envs.harvest_dual import HarvestDualEnv  # noqa: E402


def load_env_cfg(cfg_path: Path) -> Dict:
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def build_env_from_cfg(cfg: Dict) -> HarvestDualEnv:
    return HarvestDualEnv(**cfg["env"])


def run_one(env: HarvestDualEnv, steps: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    apples, waste = [], []
    for _ in range(steps):
        if not env.agents:
            break
        actions = {aid: int(rng.integers(0, 8)) for aid in env.agents}
        env.step(actions)
        apples.append(int(np.sum(env.grid == env.APPLE)))
        waste.append(int(np.sum(env.grid == env.WASTE)))
    return np.asarray(apples), np.asarray(waste)


def make_symmetric_cfg(base_b_cfg: Dict, alpha: float) -> Dict:
    cfg = copy.deepcopy(base_b_cfg)
    cfg["env"]["zap_waste_reward"] = 0.0
    cfg["env"]["waste_regrowth_suppression"] = float(alpha)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.05,0.10,0.15,0.20,0.30",
        help="comma-separated alpha values to sweep for symmetric Env B",
    )
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    env_a_cfg = load_env_cfg(PROJECT_ROOT / "configs/m1_env_A_sc030.yaml")
    env_b_cfg = load_env_cfg(PROJECT_ROOT / "configs/m1_env_B_sc030.yaml")

    print(
        f"{'config':<32} {'mean_apples':>12} {'mean_waste':>12} {'final_apples':>14}"
    )

    apples_a, waste_a = run_one(build_env_from_cfg(env_a_cfg), args.steps, args.seed)
    print(
        f"{'m1_env_A_sc030':<32} {apples_a.mean():>12.2f} "
        f"{waste_a.mean():>12.2f} {int(apples_a[-1]):>14d}"
    )

    apples_b, waste_b = run_one(build_env_from_cfg(env_b_cfg), args.steps, args.seed)
    print(
        f"{'m1_env_B_sc030 (base)':<32} {apples_b.mean():>12.2f} "
        f"{waste_b.mean():>12.2f} {int(apples_b[-1]):>14d}"
    )

    sym_means = {}
    for alpha in alphas:
        sym_cfg = make_symmetric_cfg(env_b_cfg, alpha)
        apples_s, waste_s = run_one(build_env_from_cfg(sym_cfg), args.steps, args.seed)
        sym_means[alpha] = float(apples_s.mean())
        label = f"m1_env_B_sym (alpha={alpha:.2f})"
        print(
            f"{label:<32} {apples_s.mean():>12.2f} "
            f"{waste_s.mean():>12.2f} {int(apples_s[-1]):>14d}"
        )

    print()
    print("deltas vs base Env B (negative = cleanup matters more):")
    for alpha in alphas:
        delta_b = sym_means[alpha] - apples_b.mean()
        rel_b = (delta_b / apples_b.mean() * 100.0) if apples_b.mean() else 0.0
        print(f"  alpha={alpha:.2f}: mean_apples_delta_vs_B = {delta_b:+.2f} ({rel_b:+.2f}%)")

    print("deltas vs Env A (closer to 0 = better resource parity with A):")
    for alpha in alphas:
        delta_a = sym_means[alpha] - apples_a.mean()
        rel_a = (delta_a / apples_a.mean() * 100.0) if apples_a.mean() else 0.0
        print(f"  alpha={alpha:.2f}: mean_apples_delta_vs_A = {delta_a:+.2f} ({rel_a:+.2f}%)")


if __name__ == "__main__":
    main()
