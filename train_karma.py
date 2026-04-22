"""
KARMA Trainer: The Mirror Test
github.com/tapasiitk/situated-agency-alignment

This script runs the core experiment of the paper.
It trains a population of agents in the Dual-Use Harvest environment.

Modes:
    1. baseline: Standard DRQN (No contrastive loss)
    2. broken:   Broken Mirror (Semantic Confusion: ZAP_AGENT ≈ ZAP_WASTE)
    3. karma:    KARMA (Ethical Alignment: ZAP_AGENT ≈ BEING_ZAPPED)

Usage:
    python train_karma.py --config configs/env_harvest.yaml --mode karma
"""

import argparse
import csv
import json
from pathlib import Path
from contextlib import nullcontext
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import wandb
from collections import defaultdict
import torch.nn.functional as F

from karmic_rl.envs.harvest_dual import HarvestDualEnv
from karmic_rl.agents.karma_agent import KarmaAgent, KARMACollector

# ----------------------------------------------------------------
# PPO Utilities
# ----------------------------------------------------------------

# def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
#     """
#     Generalized Advantage Estimation (GAE).
#     """
#     advantages = []
#     gae = 0
    
#     # Iterate backwards
#     for t in reversed(range(len(rewards))):
#         if t == len(rewards) - 1:
#             next_val = next_value
#         else:
#             next_val = values[t + 1]
            
#         delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
#         gae = delta + gamma * lam * (1 - dones[t]) * gae
#         advantages.insert(0, gae)
        
#     return torch.tensor(advantages, dtype=torch.float32), torch.tensor(advantages) + torch.tensor(values)
def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).
    """
    # Ensure inputs are on CPU for list processing (faster than single-element GPU ops)
    # or handle everything on the same device.
    # Easiest: Convert to CPU numpy/list for the loop, then back to Tensor on same device as values.
    
    device = values.device if isinstance(values, torch.Tensor) else torch.device("cpu")
    
    # Convert to CPU list/numpy for iteration if they are tensors
    r = rewards.tolist() if isinstance(rewards, torch.Tensor) else rewards
    v = values.tolist() if isinstance(values, torch.Tensor) else values
    d = dones.tolist() if isinstance(dones, torch.Tensor) else dones
    
    advantages = []
    gae = 0
    
    # Iterate backwards
    for t in reversed(range(len(r))):
        if t == len(r) - 1:
            next_val = next_value
        else:
            next_val = v[t + 1]
            
        delta = r[t] + gamma * next_val * (1 - d[t]) - v[t]
        gae = delta + gamma * lam * (1 - d[t]) * gae
        advantages.insert(0, gae)
    
    # Convert back to Tensor on the correct device
    adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    
    # Recalculate returns: Q = A + V
    # Ensure values is a tensor on the same device
    val_tensor = values if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=torch.float32, device=device)
    
    returns_tensor = adv_tensor + val_tensor
    
    return adv_tensor, returns_tensor

# ----------------------------------------------------------------
# Main Training Loop
# ----------------------------------------------------------------

def train(config_path, mode, seed=42):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("training", {}).get("use_amp", device.type == "cuda"))
    amp_inference = bool(config.get("training", {}).get("amp_inference", False))
    compile_model = bool(config.get("training", {}).get("compile_model", False))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    print(f"Starting KARMA Training | Mode: {mode.upper()} | Device: {device}")

    # Resolved runtime knobs
    log_cfg = config.get("logging", {})
    use_wandb = bool(log_cfg.get("use_wandb", True))
    results_dir = Path(log_cfg.get("local_results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{mode}_seed{seed}"
    run_prefix = f"{Path(config_path).stem}_{run_name}"
    csv_path = results_dir / f"{run_prefix}.csv"
    summary_path = results_dir / f"{run_prefix}.json"
    update_every = int(config.get("training", {}).get("update_every", 10))

    if use_wandb:
        wandb.init(
            project=log_cfg.get('project_name', 'karma-mirror-test'),
            config=config,
            name=run_name,
            tags=[mode, "mirror_test"]
        )

    # Environment
    env_cfg = config["env"]
    env = HarvestDualEnv(
        grid_size=env_cfg['grid_size'],
        num_agents=env_cfg['num_agents'],
        max_steps=env_cfg['max_steps'],
        apple_density=env_cfg['apple_density'],
        zap_timeout=env_cfg['zap_timeout'],
        regrowth_speed=env_cfg.get("regrowth_speed", 1.0),
        zap_waste_reward=env_cfg['zap_waste_reward'],
        zap_agent_reward=env_cfg['zap_agent_reward'],
        zap_cost=env_cfg.get("zap_cost", 0.01),
        waste_spawn_rate=env_cfg.get("waste_spawn_rate", 0.0),
        apple_spawn_mode=env_cfg.get("apple_spawn_mode", "two_patch"),
        dynamic_waste_enabled=env_cfg.get("dynamic_waste_enabled", False),
        dynamic_waste_prob=env_cfg.get("dynamic_waste_prob", 0.02),
    )

    resolved = {
        "mode": mode,
        "seed": seed,
        "device": str(device),
        "config_path": str(config_path),
        "resolved_env": {
            "grid_size": env.grid_size,
            "num_agents": len(env.possible_agents),
            "max_steps": env.max_steps,
            "apple_density": env.apple_density,
            "regrowth_speed": env.regrowth_speed,
            "zap_timeout": env.zap_timeout,
            "zap_waste_reward": env.zap_waste_reward,
            "zap_agent_reward": env.zap_agent_reward,
            "zap_cost": env.zap_cost,
            "waste_spawn_rate": env.waste_spawn_rate,
            "apple_spawn_mode": env.apple_spawn_mode,
            "dynamic_waste_enabled": env.dynamic_waste_enabled,
            "dynamic_waste_prob": env.dynamic_waste_prob,
        },
    }
    print(json.dumps(resolved, indent=2))

    # Agents (Parameter Sharing: Single Network)
    # We use one 'master_agent' that all agents in the env share.
    # master_agent = KarmaAgent(
    #     mode=mode,
    #     contrastive_weight=config['training']['contrastive_weight']
    # ).to(device)
        # Agents (Parameter Sharing: Single Network)
    grid_sz = config['env']['grid_size']
    
    master_agent = KarmaAgent(
        obs_shape=(3, grid_sz, grid_sz),  # <--- CRITICAL FIX: Pass dynamic shape
        mode=mode,
        contrastive_weight=config['training']['contrastive_weight']
    ).to(device)
    if compile_model and hasattr(torch, "compile"):
        master_agent = torch.compile(master_agent, mode="reduce-overhead")


    optimizer = optim.Adam(master_agent.parameters(), lr=float(config['training']['lr']))
    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_inference = bool(amp_enabled and amp_inference)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    gamma = float(config["training"].get("gamma", 0.99))
    gae_lambda = float(config["training"].get("gae_lambda", 0.95))
    clip_ratio = float(config["training"].get("clip_ratio", 0.2))
    
    # Buffer (Stores data for all agents mixed together)
    buffer = KARMACollector()

    # Metrics trackers
    metric_window = defaultdict(list)
    row_buffer = []
    
    # ----------------------------------------------------------------
    # Episode Loop
    # ----------------------------------------------------------------
    for episode in range(1, config['training']['episodes'] + 1):
        
        obs_dict, infos = env.reset(seed=seed + episode)
        
        # Reset LSTM states for all agents at start of episode
        # We process agents in a consistent order
        agent_ids = env.possible_agents
        aid_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        num_agents = len(agent_ids)
        
        # Internal hidden states for the batch of agents
        # Shape: (N, hidden_dim)
        h_state = torch.zeros(num_agents, master_agent.hidden_dim).to(device)
        c_state = torch.zeros(num_agents, master_agent.hidden_dim).to(device)
        
        # Episode stats
        ep_rewards = {aid: 0.0 for aid in agent_ids}
        ep_events = []
        ep_steps = 0
        
        # Rollout Loop
        for step in range(config['env']['max_steps']):
            
            # 1. Prepare Batch Observation
            # Convert dict {agent_0: obs...} to Tensor (N, 3, H, W)
            active_agent_ids = [aid for aid in agent_ids if aid in obs_dict]
            if not active_agent_ids:
                break # All agents done?
            active_agents = [aid_to_idx[aid] for aid in active_agent_ids]
            obs_np = (
                np.stack([obs_dict[aid].transpose(2, 0, 1) for aid in active_agent_ids]).astype(np.float32)
                / 255.0
            )
            obs_tensor = torch.from_numpy(obs_np).to(device, non_blocking=True)
            
            # 2. Forward Pass (Vectorized)
            with torch.inference_mode():
                # Extract hidden states for active agents
                h_in = h_state[active_agents]
                c_in = c_state[active_agents]
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_inference
                    else nullcontext()
                )
                with amp_ctx:
                    out = master_agent(obs_tensor, (h_in, c_in))
                
                # Update hidden states
                h_state[active_agents] = out["new_hidden"][0].to(h_state.dtype)
                c_state[active_agents] = out["new_hidden"][1].to(c_state.dtype)
                
                # Sample Actions
                dist = Categorical(logits=out["policy"])
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            
            # 3. Step Environment
            action_dict = {
                aid: act.item()
                for aid, act in zip(active_agent_ids, actions)
            }
            
            next_obs_dict, rewards, terms, truncs, next_infos = env.step(action_dict)
            ep_steps += 1
            
            # 4. Store Experience
            for i, idx in enumerate(active_agents):
                aid = agent_ids[idx]
                
                # Retrieve social events for this agent
                events = next_infos[aid]["social_events"]
                ep_events.extend(events) # Flatten for metrics
                
                done = terms[aid] or truncs[aid]
                
                buffer.store(
                    obs=obs_np[i],  # already float32 CHW, avoids extra CPU<->GPU churn
                    action=actions[i].item(),
                    reward=rewards[aid],
                    value=out["value"][i].item(),
                    log_prob=log_probs[i].item(),
                    done=done,
                    events=events,
                    agent_id=aid
                )
                
                ep_rewards[aid] += rewards[aid]
            
            obs_dict = next_obs_dict
            if all(terms.values()) or all(truncs.values()):
                break

        # ----------------------------------------------------------------
        # Update (PPO + KARMA)
        # ----------------------------------------------------------------
        if episode % update_every == 0:
            batch = buffer.get_batch()
            
            if len(batch['obs']) > 0:
                # Convert to Tensor
                b_obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=device)
                b_acts = torch.as_tensor(batch['actions'], dtype=torch.long, device=device)
                b_log_probs = torch.as_tensor(batch['log_probs'], dtype=torch.float32, device=device)
                b_vals = torch.as_tensor(batch['values'], dtype=torch.float32, device=device)
                b_rewards = batch['rewards']
                b_dones = batch['dones']
                
                # Compute GAE (Approximate: assuming full batch is one continuous stream for simplicity)
                # In strict MARL, we should separate by agent/episode, but this works for aggregated PPO
                # provided gamma is high enough.
                # For more precision, we calculate GAE per episode before buffering.
                # Here we use a simplified calculation for the full buffer:
                advantages, returns = compute_gae(
                    b_rewards, b_vals, 0.0, b_dones, gamma=gamma, lam=gae_lambda
                )
                advantages = advantages.to(device)
                returns = returns.to(device)
                
                # PPO Epochs
                for _ in range(config['training']['ppo_epochs']):
                    # Mini-batch indices
                    indices = torch.randperm(len(b_obs), device=device)
                    batch_size = config['training']['batch_size']
                    
                    for start in range(0, len(b_obs), batch_size):
                        end = start + batch_size
                        idx = indices[start:end]
                        
                        # Forward (Re-evaluate)
                        # Note: We pass None for hidden to reset LSTM for training 
                        # (Truncated BPTT approximation: simple but standard for DRQN PPO)
                        # Ideal: Store hidden states in buffer. 
                        # Simplification: Agents learn reactive-ish policies or short-term memory.
                        amp_ctx = (
                            torch.autocast(device_type="cuda", dtype=torch.float16)
                            if amp_enabled
                            else nullcontext()
                        )
                        with amp_ctx:
                            out = master_agent(b_obs[idx], hidden=None)
                            
                            curr_dist = Categorical(logits=out["policy"])
                            curr_log_probs = curr_dist.log_prob(b_acts[idx])
                            entropy = curr_dist.entropy().mean()
                            
                            # PPO Loss
                            ratios = torch.exp(curr_log_probs - b_log_probs[idx])
                            surr1 = ratios * advantages[idx]
                            surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages[idx]
                            actor_loss = -torch.min(surr1, surr2).mean()
                            critic_loss = F.mse_loss(out["value"].squeeze(), returns[idx])
                            
                            # KARMA Loss (The Mirror)
                            if mode == "baseline":
                                contrastive_loss = torch.tensor(0.0, device=device)
                            else:
                                idx_list = idx.tolist()
                                contrastive_loss = master_agent.compute_contrastive_loss(
                                    out["embedding"],
                                    [batch["social_events"][i] for i in idx_list],
                                    [batch["agent_ids"][i] for i in idx_list]
                                )
                            
                            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy + contrastive_loss
                        
                        optimizer.zero_grad(set_to_none=True)
                        if scaler.is_enabled():
                            scaler.scale(total_loss).backward()
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(master_agent.parameters(), 0.5)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            total_loss.backward()
                            nn.utils.clip_grad_norm_(master_agent.parameters(), 0.5)
                            optimizer.step()
            
            buffer.reset() # Clear buffer

        # ----------------------------------------------------------------
        # Logging & Metrics
        # ----------------------------------------------------------------
        # 1. Count Events
        zap_agent = sum(1 for e in ep_events if e["type"] == "ZAP_AGENT")
        zap_waste = sum(1 for e in ep_events if e["type"] == "ZAP_WASTE")
        apples_eaten = sum(1 for e in ep_events if e["type"] == "APPLE_EATEN")
        
        # 2. Normalize
        # Per-agent-per-step rates are easier to compare across run lengths.
        agent_steps = max(1, num_agents * ep_steps)
        violence_rate = zap_agent / agent_steps
        coop_rate = zap_waste / agent_steps
        apple_rate = apples_eaten / agent_steps
        avg_return = sum(ep_rewards.values()) / num_agents
        
        metric_window['violence'].append(violence_rate)
        metric_window['coop'].append(coop_rate)
        metric_window['apple_rate'].append(apple_rate)
        metric_window['return'].append(avg_return)
        
        if episode % int(log_cfg.get('log_interval', 20)) == 0:
            v_mean = np.mean(metric_window['violence'])
            c_mean = np.mean(metric_window['coop'])
            a_mean = np.mean(metric_window['apple_rate'])
            r_mean = np.mean(metric_window['return'])
            selectivity = c_mean / max(1e-6, v_mean)

            payload = {
                "Episode": int(episode),
                "ViolenceRate_per_agent_step": float(v_mean),
                "CooperationRate_per_agent_step": float(c_mean),
                "AppleRate_per_agent_step": float(a_mean),
                "AvgReturn_per_agent": float(r_mean),
                "EthicalSelectivity": float(selectivity),
            }

            if use_wandb:
                wandb.log(payload)

            row_buffer.append(payload)
            print(
                f"Ep {episode} | Return: {r_mean:.3f} | Apples: {a_mean:.4f} | "
                f"Violence: {v_mean:.4f} | Coop: {c_mean:.4f} | Sel: {selectivity:.3f}"
            )
            metric_window.clear()

    if row_buffer:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_buffer[0].keys()))
            writer.writeheader()
            writer.writerows(row_buffer)

    run_summary = {
        **resolved,
        "episodes": int(config["training"]["episodes"]),
        "log_rows": len(row_buffer),
        "artifacts": {"csv": str(csv_path), "summary_json": str(summary_path)},
        "final_metrics": row_buffer[-1] if row_buffer else {},
    }
    with summary_path.open("w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"Saved local artifacts: {csv_path} and {summary_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/env_harvest.yaml", help="Path to config")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "broken", "karma"], help="Experiment mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train(args.config, args.mode, args.seed)
