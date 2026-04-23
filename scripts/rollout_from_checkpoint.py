"""Roll out a trained KARMA/baseline checkpoint and dump per-timestep features.

This is the data-generation step of the M1 analysis pipeline. It loads a
checkpoint produced by `train_karma.py`, executes N evaluation episodes
using the exact env configuration the checkpoint was trained on, and writes
a Parquet (or JSONL fallback) file with one row per (episode, step, agent)
tuple containing the fields required by the M1 measurements:

    episode_id, step, agent_id, role, role_neutral, role_aggressor,
    role_victim, role_cleaner, role_forager, action, reward, value,
    log_prob, embedding[64], cnn_features[F], lstm_h[H], lstm_c[H].

Usage:
    python scripts/rollout_from_checkpoint.py \
        --config   configs/m1_smoke40.yaml \
        --checkpoint results/m1_smoke40/checkpoints/m1_smoke40_baseline_seed1_ep20.pt \
        --episodes 4 \
        --output   results/m1_smoke40/rollouts/seed1_ep20.parquet

The checkpoint is expected to contain at least a `model_state_dict` key
and optionally `mode`, `seed`, `episode`. The env config is resolved from
`--config` (NOT from the checkpoint's stored config_path) so that the
caller can choose to evaluate under matched or different conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.distributions import Categorical

# Make project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from karmic_rl.agents.karma_agent import KarmaAgent
from karmic_rl.envs.harvest_dual import HarvestDualEnv
from karmic_rl.utils.roles import (
    ROLE_NAMES,
    infer_role,
    infer_role_multilabel,
)


def build_env(env_cfg: Dict[str, Any]) -> HarvestDualEnv:
    return HarvestDualEnv(
        grid_size=env_cfg["grid_size"],
        num_agents=env_cfg["num_agents"],
        max_steps=env_cfg["max_steps"],
        apple_density=env_cfg["apple_density"],
        zap_timeout=env_cfg["zap_timeout"],
        regrowth_speed=env_cfg.get("regrowth_speed", 1.0),
        zap_waste_reward=env_cfg["zap_waste_reward"],
        zap_agent_reward=env_cfg["zap_agent_reward"],
        victim_penalty=env_cfg.get("victim_penalty", 0.0),
        zap_cost=env_cfg.get("zap_cost", 0.0),
        waste_spawn_rate=env_cfg.get("waste_spawn_rate", 0.0),
        apple_spawn_mode=env_cfg.get("apple_spawn_mode", "central_patch"),
        dynamic_waste_enabled=env_cfg.get("dynamic_waste_enabled", False),
        dynamic_waste_prob=env_cfg.get("dynamic_waste_prob", 0.0),
    )


def load_agent(checkpoint_path: Path, env: HarvestDualEnv, device: torch.device) -> KarmaAgent:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mode = ckpt.get("mode", "baseline")
    agent = KarmaAgent(
        obs_shape=(3, env.grid_size, env.grid_size),
        mode=mode,
        contrastive_weight=0.0,
    ).to(device)
    state = ckpt["model_state_dict"]
    # Strip torch.compile prefix if present.
    clean_state = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state.items()
    }
    missing, unexpected = agent.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[rollout] warn: missing keys {missing[:5]}... ({len(missing)} total)")
    if unexpected:
        print(f"[rollout] warn: unexpected keys {unexpected[:5]}... ({len(unexpected)} total)")
    agent.eval()
    return agent


def run_rollout(
    agent: KarmaAgent,
    env: HarvestDualEnv,
    num_episodes: int,
    device: torch.device,
    deterministic: bool,
    seed_base: int,
) -> List[Dict[str, Any]]:
    """Execute `num_episodes` and return a list of per-timestep record dicts."""
    records: List[Dict[str, Any]] = []

    for ep in range(num_episodes):
        obs_dict, _ = env.reset(seed=seed_base + ep)
        agent_ids = env.possible_agents
        aid_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        num_agents = len(agent_ids)

        h_state = torch.zeros(num_agents, agent.hidden_dim, device=device)
        c_state = torch.zeros(num_agents, agent.hidden_dim, device=device)

        for step in range(env.max_steps):
            active_agent_ids = [aid for aid in agent_ids if aid in obs_dict]
            if not active_agent_ids:
                break

            active_idx = [aid_to_idx[aid] for aid in active_agent_ids]
            obs_np = (
                np.stack([obs_dict[aid].transpose(2, 0, 1) for aid in active_agent_ids]).astype(
                    np.float32
                )
                / 255.0
            )
            obs_tensor = torch.from_numpy(obs_np).to(device, non_blocking=True)

            with torch.inference_mode():
                h_in = h_state[active_idx]
                c_in = c_state[active_idx]
                out = agent(obs_tensor, (h_in, c_in))

                h_state[active_idx] = out["new_hidden"][0]
                c_state[active_idx] = out["new_hidden"][1]

                logits = out["policy"]
                logits = torch.nan_to_num(
                    logits, nan=0.0, posinf=20.0, neginf=-20.0
                ).clamp(-20.0, 20.0)

                if deterministic:
                    actions = torch.argmax(logits, dim=-1)
                    log_probs = torch.log_softmax(logits, dim=-1).gather(
                        -1, actions.unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

                embeddings = out["embedding"].detach().cpu().numpy()
                features = out["features"].detach().cpu().numpy()
                hidden_h = out["new_hidden"][0].detach().cpu().numpy()
                hidden_c = out["new_hidden"][1].detach().cpu().numpy()
                values = out["value"].detach().cpu().numpy().squeeze(-1)
                actions_np = actions.detach().cpu().numpy()
                log_probs_np = log_probs.detach().cpu().numpy()

            action_dict = {
                aid: int(act) for aid, act in zip(active_agent_ids, actions_np)
            }
            next_obs_dict, rewards, terms, truncs, next_infos = env.step(action_dict)

            for i, aid in enumerate(active_agent_ids):
                events = next_infos[aid]["social_events"]
                role = infer_role(events, aid)
                neutral, agg, vic, cln, fgr = infer_role_multilabel(events, aid)
                records.append(
                    {
                        "episode_id": int(ep),
                        "step": int(step),
                        "agent_id": aid,
                        "role": int(role),
                        "role_name": ROLE_NAMES[role],
                        "role_neutral": int(neutral),
                        "role_aggressor": int(agg),
                        "role_victim": int(vic),
                        "role_cleaner": int(cln),
                        "role_forager": int(fgr),
                        "action": int(actions_np[i]),
                        "reward": float(rewards[aid]),
                        "value": float(values[i]),
                        "log_prob": float(log_probs_np[i]),
                        "embedding": embeddings[i].astype(np.float32).tolist(),
                        "cnn_features": features[i].astype(np.float32).tolist(),
                        "lstm_h": hidden_h[i].astype(np.float32).tolist(),
                        "lstm_c": hidden_c[i].astype(np.float32).tolist(),
                        "events_json": json.dumps(events),
                    }
                )

            obs_dict = next_obs_dict
            if all(terms.values()) or all(truncs.values()):
                break

    return records


def write_records(records: List[Dict[str, Any]], output_path: Path) -> str:
    """Write records to Parquet if pandas/pyarrow available, else JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        df = pd.DataFrame.from_records(records)
        try:
            df.to_parquet(output_path, index=False)
            return "parquet"
        except Exception as exc:
            print(f"[rollout] parquet write failed ({exc}); falling back to JSONL")
    except ImportError:
        pass

    jsonl_path = output_path.with_suffix(".jsonl")
    with jsonl_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return "jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use argmax policy; default is on-policy sampling.")
    parser.add_argument("--seed", type=int, default=10_000,
                        help="Base seed for eval rollouts.")
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda' or 'cpu'; default auto-detects.")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = build_env(config["env"])
    agent = load_agent(checkpoint_path, env, device)

    print(
        f"[rollout] checkpoint={checkpoint_path.name} device={device} "
        f"episodes={args.episodes} deterministic={args.deterministic}"
    )

    records = run_rollout(
        agent,
        env,
        num_episodes=args.episodes,
        device=device,
        deterministic=args.deterministic,
        seed_base=args.seed,
    )

    kind = write_records(records, output_path)
    print(
        f"[rollout] wrote {len(records)} rows -> {output_path} ({kind})"
    )

    # Quick role-histogram sanity.
    from collections import Counter

    role_counts = Counter(r["role_name"] for r in records)
    print(f"[rollout] role distribution: {dict(role_counts)}")


if __name__ == "__main__":
    main()
