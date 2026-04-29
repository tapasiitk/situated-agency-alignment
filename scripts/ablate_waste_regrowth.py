"""Sanity check for the waste-regrowth suppression knob.

Builds two HarvestDualEnv instances - one from the original Env B config and
one from the symmetric Env B config - then steps each for `--steps` with a
fixed seed and an identical uniform-random action stream. Records per-step
APPLE and WASTE counts and prints a small summary table. With
`waste_regrowth_suppression > 0` the symmetric config is expected to show a
lower mean / final apple count than the base config (cleanup matters
instrumentally because waste actively suppresses regrowth).

Usage:
    python scripts/ablate_waste_regrowth.py --steps 1000 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from karmic_rl.envs.harvest_dual import HarvestDualEnv  # noqa: E402


def build_env(cfg_path: Path) -> HarvestDualEnv:
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfgs = [
        ("m1_env_B_sc030", PROJECT_ROOT / "configs/m1_env_B_sc030.yaml"),
        ("m1_env_B_sc030_sym", PROJECT_ROOT / "configs/m1_env_B_sc030_sym.yaml"),
    ]

    print(f"{'config':<22} {'mean_apples':>12} {'mean_waste':>12} {'final_apples':>14}")
    series = {}
    for name, path in cfgs:
        env = build_env(path)
        apples, waste = run_one(env, args.steps, args.seed)
        series[name] = apples
        print(
            f"{name:<22} {apples.mean():>12.2f} {waste.mean():>12.2f} "
            f"{int(apples[-1]):>14d}"
        )

    base = series.get("m1_env_B_sc030")
    sym = series.get("m1_env_B_sc030_sym")
    if base is not None and sym is not None:
        delta_mean = float(sym.mean() - base.mean())
        delta_final = int(sym[-1]) - int(base[-1])
        verdict = "OK" if delta_mean < 0 else "CHECK"
        print(
            f"sym - base: mean_apples = {delta_mean:+.2f}, final_apples = {delta_final:+d} "
            f"(expected mean delta < 0 when alpha > 0; {verdict})"
        )


if __name__ == "__main__":
    main()
