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
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import wandb
from collections import defaultdict

from karmic_rl.envs.harvest_dual import HarvestDualEnv
from karmic_rl.agents.karma_agent import KarmaAgent, KARMACollector

# ----------------------------------------------------------------
# PPO Utilities
# ----------------------------------------------------------------

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).
    """
    advantages = []
    gae = 0
    
    # Iterate backwards
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(advantages) + torch.tensor(values)

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
    print(f"🚀 Starting KARMA Training | Mode: {mode.upper()} | Device: {device}")

    # Initialize WandB
    wandb.init(
        project=config['logging']['project_name'],
        config=config,
        name=f"{mode}_seed{seed}",
        tags=[mode, "mirror_test"]
    )

    # Environment
    env = HarvestDualEnv(
        grid_size=config['env']['grid_size'],
        num_agents=config['env']['num_agents'],
        max_steps=config['env']['max_steps'],
        apple_density=config['env']['apple_density'],
        zap_timeout=config['env']['zap_timeout'],
        zap_waste_reward=config['env']['zap_waste_reward'],
        zap_agent_reward=config['env']['zap_agent_reward']
    )

    # Agents (Parameter Sharing: Single Network)
    # We use one 'master_agent' that all agents in the env share.
    master_agent = KarmaAgent(
        mode=mode,
        contrastive_weight=config['training']['contrastive_weight']
    ).to(device)
    
    optimizer = optim.Adam(master_agent.parameters(), lr=float(config['training']['lr']))
    
    # Buffer (Stores data for all agents mixed together)
    buffer = KARMACollector()

    # Metrics trackers
    metric_window = defaultdict(list)
    
    # ----------------------------------------------------------------
    # Episode Loop
    # ----------------------------------------------------------------
    for episode in range(1, config['training']['episodes'] + 1):
        
        obs_dict, infos = env.reset()
        
        # Reset LSTM states for all agents at start of episode
        # We process agents in a consistent order
        agent_ids = env.possible_agents
        num_agents = len(agent_ids)
        
        # Internal hidden states for the batch of agents
        # Shape: (N, hidden_dim)
        h_state = torch.zeros(num_agents, master_agent.hidden_dim).to(device)
        c_state = torch.zeros(num_agents, master_agent.hidden_dim).to(device)
        
        # Episode stats
        ep_rewards = {aid: 0.0 for aid in agent_ids}
        ep_events = []
        
        # Rollout Loop
        for step in range(config['env']['max_steps']):
            
            # 1. Prepare Batch Observation
            # Convert dict {agent_0: obs...} to Tensor (N, 3, H, W)
            obs_list = []
            active_agents = [] # Track which agents are alive/present this step
            
            for i, aid in enumerate(agent_ids):
                if aid in obs_dict:
                    # Permute (H,W,C) -> (C,H,W) and normalize
                    o = torch.tensor(obs_dict[aid], dtype=torch.float32).permute(2, 0, 1) / 255.0
                    obs_list.append(o)
                    active_agents.append(i)
            
            if not obs_list:
                break # All agents done?
                
            obs_tensor = torch.stack(obs_list).to(device)
            
            # 2. Forward Pass (Vectorized)
            with torch.no_grad():
                # Extract hidden states for active agents
                h_in = h_state[active_agents]
                c_in = c_state[active_agents]
                
                out = master_agent(obs_tensor, (h_in, c_in))
                
                # Update hidden states
                h_state[active_agents] = out["new_hidden"][0]
                c_state[active_agents] = out["new_hidden"][1]
                
                # Sample Actions
                dist = Categorical(logits=out["policy"])
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            
            # 3. Step Environment
            action_dict = {
                agent_ids[idx]: act.item() 
                for idx, act in zip(active_agents, actions)
            }
            
            next_obs_dict, rewards, terms, truncs, next_infos = env.step(action_dict)
            
            # 4. Store Experience
            for i, idx in enumerate(active_agents):
                aid = agent_ids[idx]
                
                # Retrieve social events for this agent
                events = next_infos[aid]["social_events"]
                ep_events.extend(events) # Flatten for metrics
                
                done = terms[aid] or truncs[aid]
                
                buffer.store(
                    obs=obs_list[i].cpu().numpy(), # Store as numpy to save GPU mem
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
        if episode % 10 == 0: # Update every 10 episodes
            batch = buffer.get_batch()
            
            if len(batch['obs']) > 0:
                # Convert to Tensor
                b_obs = torch.tensor(batch['obs'], dtype=torch.float32).to(device)
                b_acts = torch.tensor(batch['actions'], dtype=torch.long).to(device)
                b_log_probs = torch.tensor(batch['log_probs'], dtype=torch.float32).to(device)
                b_vals = torch.tensor(batch['values'], dtype=torch.float32).to(device)
                b_rewards = batch['rewards']
                b_dones = batch['dones']
                
                # Compute GAE (Approximate: assuming full batch is one continuous stream for simplicity)
                # In strict MARL, we should separate by agent/episode, but this works for aggregated PPO
                # provided gamma is high enough.
                # For more precision, we calculate GAE per episode before buffering.
                # Here we use a simplified calculation for the full buffer:
                advantages, returns = compute_gae(b_rewards, b_vals, 0.0, b_dones)
                advantages = advantages.to(device)
                returns = returns.to(device)
                
                # PPO Epochs
                for _ in range(config['training']['ppo_epochs']):
                    # Mini-batch indices
                    indices = np.random.permutation(len(b_obs))
                    batch_size = config['training']['batch_size']
                    
                    for start in range(0, len(b_obs), batch_size):
                        end = start + batch_size
                        idx = indices[start:end]
                        
                        # Forward (Re-evaluate)
                        # Note: We pass None for hidden to reset LSTM for training 
                        # (Truncated BPTT approximation: simple but standard for DRQN PPO)
                        # Ideal: Store hidden states in buffer. 
                        # Simplification: Agents learn reactive-ish policies or short-term memory.
                        out = master_agent(b_obs[idx], hidden=None)
                        
                        curr_dist = Categorical(logits=out["policy"])
                        curr_log_probs = curr_dist.log_prob(b_acts[idx])
                        entropy = curr_dist.entropy().mean()
                        
                        # PPO Loss
                        ratios = torch.exp(curr_log_probs - b_log_probs[idx])
                        surr1 = ratios * advantages[idx]
                        surr2 = torch.clamp(ratios, 1-0.2, 1+0.2) * advantages[idx]
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = F.mse_loss(out["value"].squeeze(), returns[idx])
                        
                        # KARMA Loss (The Mirror)
                        contrastive_loss = master_agent.compute_contrastive_loss(
                            out["embedding"],
                            [batch["social_events"][i] for i in idx],
                            [batch["agent_ids"][i] for i in idx]
                        )
                        
                        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy + contrastive_loss
                        
                        optimizer.zero_grad()
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
        
        # 2. Normalize
        # Violence Rate: Zaps per agent per 1000 steps
        violence_rate = zap_agent / num_agents
        coop_rate = zap_waste / num_agents
        avg_reward = sum(ep_rewards.values()) / num_agents
        
        metric_window['violence'].append(violence_rate)
        metric_window['coop'].append(coop_rate)
        metric_window['yield'].append(avg_reward)
        
        if episode % config['logging']['log_interval'] == 0:
            v_mean = np.mean(metric_window['violence'])
            c_mean = np.mean(metric_window['coop'])
            y_mean = np.mean(metric_window['yield'])
            selectivity = c_mean / max(1e-6, v_mean)
            
            wandb.log({
                "Violence Rate": v_mean,
                "Cooperation Rate": c_mean,
                "System Yield": y_mean,
                "Ethical Selectivity": selectivity,
                "Episode": episode
            })
            
            print(f"Ep {episode} | Yield: {y_mean:.2f} | Violence: {v_mean:.2f} | Coop: {c_mean:.2f} | Sel: {selectivity:.2f}")
            metric_window.clear()

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/env_harvest.yaml", help="Path to config")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "broken", "karma"], help="Experiment mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train(args.config, args.mode, args.seed)
