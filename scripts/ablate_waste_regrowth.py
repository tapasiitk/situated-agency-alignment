"""Sanity check for the waste-regrowth suppression and waste-spread knobs.

Builds HarvestDualEnv instances for Env A, the original Env B, and a sweep of
symmetric Env B variants over the cross-product of `waste_regrowth_suppression`
(alpha) and `waste_spread_prob` (canonical Cleanup-style stochastic spread to
empty 4-neighbors). Each env is stepped for `--steps` with a fixed seed and an
identical uniform-random action stream. Per-step APPLE and WASTE counts are
recorded; the script prints a summary table and a waste-growth trajectory.

The goal of the symmetric Env B sweep is to find an (alpha, spread) pair that:

  1. Makes cleanup *instrumental*: unchecked waste suppresses apples (lower
     mean apple count than original Env B and visible waste accumulation).
  2. Keeps Env B_sym roughly comparable to Env A in mean apples, so the only
     structural difference between A and B_sym is beam semantics, not raw
     resource availability.
  3. Avoids runaway waste explosions, which can be diagnosed from the waste
     trajectory at t = 250, 500, 1000.

Usage:
    python scripts/ablate_waste_regrowth.py --steps 1000 --seed 0
    python scripts/ablate_waste_regrowth.py --steps 1000 --seed 0 \\
        --alphas 0.00,0.05,0.10 --spreads 0.00,0.02,0.03,0.05
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def make_symmetric_cfg(base_b_cfg: Dict, alpha: float, spread: float) -> Dict:
    cfg = copy.deepcopy(base_b_cfg)
    cfg["env"]["zap_waste_reward"] = 0.0
    cfg["env"]["waste_regrowth_suppression"] = float(alpha)
    cfg["env"]["waste_spread_prob"] = float(spread)
    return cfg


def waste_at(waste_arr: np.ndarray, t: int) -> int:
    """Return the WASTE count at step `t` (1-indexed); -1 if the run was shorter."""
    if waste_arr.size >= t:
        return int(waste_arr[t - 1])
    return -1


def parse_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.00,0.05,0.10",
        help=(
            "comma-separated waste_regrowth_suppression values to sweep "
            "for symmetric Env B"
        ),
    )
    parser.add_argument(
        "--spreads",
        type=str,
        default="0.00,0.02,0.03,0.05",
        help=(
            "comma-separated waste_spread_prob values to sweep "
            "for symmetric Env B"
        ),
    )
    args = parser.parse_args()

    alphas = parse_floats(args.alphas)
    spreads = parse_floats(args.spreads)

    env_a_cfg = load_env_cfg(PROJECT_ROOT / "configs/m1_env_A_sc030.yaml")
    env_b_cfg = load_env_cfg(PROJECT_ROOT / "configs/m1_env_B_sc030.yaml")

    header = (
        f"{'config':<40} {'mean_apples':>12} {'mean_waste':>12} "
        f"{'final_apples':>14}"
    )
    print(header)

    apples_a, waste_a = run_one(build_env_from_cfg(env_a_cfg), args.steps, args.seed)
    print(
        f"{'m1_env_A_sc030':<40} {apples_a.mean():>12.2f} "
        f"{waste_a.mean():>12.2f} {int(apples_a[-1]):>14d}"
    )

    apples_b, waste_b = run_one(build_env_from_cfg(env_b_cfg), args.steps, args.seed)
    print(
        f"{'m1_env_B_sc030 (base)':<40} {apples_b.mean():>12.2f} "
        f"{waste_b.mean():>12.2f} {int(apples_b[-1]):>14d}"
    )

    sym_results: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
    for alpha in alphas:
        for spread in spreads:
            sym_cfg = make_symmetric_cfg(env_b_cfg, alpha, spread)
            apples_s, waste_s = run_one(
                build_env_from_cfg(sym_cfg), args.steps, args.seed
            )
            sym_results[(alpha, spread)] = (apples_s, waste_s)
            label = f"m1_env_B_sym (a={alpha:.2f}, s={spread:.2f})"
            print(
                f"{label:<40} {apples_s.mean():>12.2f} "
                f"{waste_s.mean():>12.2f} {int(apples_s[-1]):>14d}"
            )

    print()
    print("waste-growth trajectory (symmetric Env B):")
    summary_header = (
        f"{'config':<40} {'mean_apples':>12} {'mean_waste':>12} "
        f"{'w@250':>8} {'w@500':>8} {'w@1000':>8}"
    )
    print(summary_header)
    for (alpha, spread), (apples_s, waste_s) in sym_results.items():
        label = f"alpha={alpha:.2f}, spread={spread:.2f}"
        w250 = waste_at(waste_s, 250)
        w500 = waste_at(waste_s, 500)
        w1000 = waste_at(waste_s, 1000)
        print(
            f"{label:<40} {apples_s.mean():>12.2f} "
            f"{waste_s.mean():>12.2f} "
            f"{w250:>8d} {w500:>8d} {w1000:>8d}"
        )


if __name__ == "__main__":
    main()
