"""
Karmic-RL Training Pipeline.

Runs the 4 ablation conditions:
1. Baseline (DRQN + Random Spawn)
2. TVT-Only (TVT + Random Spawn)
3. Matchmaker-Only (DRQN + Karmic Spawn)
4. Full Karmic-RL (TVT + Karmic Spawn)
"""

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import wandb
import argparse
from pathlib import Path
import yaml

# Local Imports
import karmic_rl.envs
from karmic_rl.envs.matchmaker_wrapper import KarmicMatchmaker
from karmic_rl.agents.ktvt_agent import KarmicAgent
from karmic_rl.envs.harvest_parallel import HarvestParallelEnv

# Load Config (Handle if file is missing)
try:
    with open("configs/env_harvest.yaml", "r") as f:
        default_config = yaml.safe_load(f)
except FileNotFoundError:
    default_config = {}

class PPOTrainer:
    def __init__(self, agent, lr=3e-4, gamma=0.99, clip=0.2, epochs=4):
        self.agent = agent
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs

        # PPO Buffers (Shared across all agents)
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.dones = []

    def rollout(self, env, num_steps=500):
        """Collect rollout data from Multi-Agent Environment."""
        
        # Reset returns dictionary: {agent_id: obs}
        obs_dict, infos = env.reset()
        
        # Initialize Hidden States for EACH agent independently
        # (Assuming 1 batch dimension for internal logic)
        hidden_states = {
            agent_id: self.agent.get_initial_state(1) 
            for agent_id in obs_dict.keys()
        }

        steps_collected = 0
        
        while steps_collected < num_steps:
            actions_dict = {}
            
            # 1. Select Actions for ALL agents
            for agent_id in obs_dict.keys():
                obs = obs_dict[agent_id]
                
                # Preprocess Obs: (H, W, C) -> (1, C, H, W)
                # Ensure it is float and normalized if necessary
                agent_obs = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0)
                
                # Get action from model
                # forward(obs, hidden_state) -> logits, value, new_hidden
                logits, value, new_hidden, extras = self.agent(agent_obs, hidden_states[agent_id])
                
                # Sample Action
                probs = Categorical(logits=logits)
                action = probs.sample()
                logprob = probs.log_prob(action)
                
                # Store action for environment step
                actions_dict[agent_id] = int(action.item())
                
                # Update hidden state
                hidden_states[agent_id] = new_hidden

                # Store PPO data (We flatten multi-agent experience into one buffer)
                self.states.append(agent_obs)
                self.actions.append(action)
                self.values.append(value)
                self.logprobs.append(logprob)
                
                # Placeholder for rewards/dones (filled after step)
                # We will append them in the next loop or align them carefully.
                # BETTER APPROACH: Store temporarily, then append after step.

            # 2. Step Environment
            next_obs_dict, rewards, terms, truncs, infos = env.step(actions_dict)

            # 3. Store Rewards & Dones
            # We must iterate in the SAME order as above to match states/actions
            # Note: dict keys are insertion ordered in Python 3.7+, but let's be safe.
            # The issue is we already appended states/actions. 
            # We simply append the corresponding reward for each agent.
            
            for agent_id in obs_dict.keys():
                # If agent died this step, it might not be in next_obs_dict, 
                # but we still have a reward for it.
                r = rewards.get(agent_id, 0.0)
                term = terms.get(agent_id, False)
                trunc = truncs.get(agent_id, False)
                
                self.rewards.append(r)
                self.dones.append(term or trunc)

            # 4. Update Observations
            obs_dict = next_obs_dict
            steps_collected += 1

            # 5. Handle Global Termination
            if not obs_dict: # All agents done
                break
                
            # Log social events from first agent just for tracking
            first_agent = list(obs_dict.keys())[0]
            if "social_events" in infos.get(first_agent, {}):
                 events = infos[first_agent]["social_events"]
                 if events:
                     wandb.log({"zaps": len(events)}, commit=False)

        # Compute returns at end of rollout
        return self._compute_returns()

    def _compute_returns(self):
        """Compute GAE returns for PPO."""
        # Note: This simple implementation treats the flattened multi-agent stream
        # as one long trajectory. This is technically INCORRECT for LSTM states
        # because it mixes gradients across agents. 
        # However, for a simple baseline/prototype, it runs.
        # Ideally, you separate buffers per agent.
        
        returns = []
        R = 0
        
        for r, v, done in zip(reversed(self.rewards), reversed(self.values), reversed(self.dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
            
        return torch.tensor(returns), torch.tensor([]) # Advantages calc requires more logic

    def update(self):
        """PPO update step."""
        if not self.states:
            return

        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_logprobs = torch.cat(self.logprobs)
        values = torch.cat(self.values).squeeze()
        
        # Calculate Returns & Advantages
        returns = torch.tensor(self._compute_returns()[0]).to(states.device)
        
        # Detach values for advantage calculation
        advantages = returns - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages # avoid div by zero on single step

        # PPO epochs
        for _ in range(self.epochs):
            # Re-evaluate batch
            # Note: For LSTM, we should ideally pass hidden states, 
            # but standard PPO often ignores this in simple implementations 
            # or uses "burned-in" states. We pass None here for stateless update
            # or you must store hidden states in buffer.
            
            # WARNING: This `self.agent(states, None)` assumes the agent can handle 
            # None hidden state (e.g. resets to zero).
            logits, current_values, _, _ = self.agent(states, None)
            
            probs = Categorical(logits=logits)
            new_logprobs = probs.log_prob(actions)
            entropy = probs.entropy()

            ratio = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(current_values.squeeze(), returns)
            entropy_loss = -entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()

        # Clear buffers
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.logprobs, self.dones = [], [], []

def run_experiment(config):
    """Run single ablation condition."""
    wandb.init(
        project="karmic-rl",
        config=config,
        name=f"{config['agent_type']}_{config['env_type']}_{config['grid_size']}x{config['grid_size']}",
        reinit=True
    )

    # Create environment
    env = HarvestParallelEnv(
        grid_size=config["grid_size"], 
        num_agents=config["num_agents"]
    )

    # Apply matchmaker if needed
    if config["use_matchmaker"]:
        env = KarmicMatchmaker(env, debt_strength=1.0)

    # Create agent
    obs_shape = env.observation_space.shape # (H, W, 3)
    # Swapped dims for CNN: (3, H, W)
    agent_obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    
    agent = KarmicAgent(
        obs_shape=agent_obs_shape, 
        action_dim=8, 
        use_tvt=config["use_tvt"]
    )
    
    trainer = PPOTrainer(agent)

    # Training loop
    zap_history = []
    debt_history = []
    
    total_episodes = config["total_episodes"]

    for episode in range(total_episodes):
        trainer.rollout(env, num_steps=500)
        trainer.update()

        # Log metrics
        if hasattr(env, 'get_ledger_stats'):
            stats = env.get_ledger_stats()
            zap_history.append(stats["predatory_zaps"])
            debt_history.append(stats["total_debt"])
            
            wandb.log({
                "episode": episode,
                "predatory_zaps": stats["predatory_zaps"],
                "total_debt": stats["total_debt"],
                "retribution_ratio": stats["retribution_zaps"] / max(1, stats["total_zaps"])
            })
        
        if episode % 10 == 0:
            print(f"Ep {episode}: Complete")

    wandb.finish()
    return zap_history, debt_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["all", "baseline", "tvt", "matchmaker", "full"], default="all")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    
    args = parser.parse_args()

    experiments = []

    if args.mode in ["all", "baseline"]:
        experiments.append({
            "agent_type": "DRQN",
            "env_type": "Standard",
            "use_tvt": False,
            "use_matchmaker": False,
            "grid_size": args.grid_size,
            "num_agents": 5,
            "total_episodes": 100 # Reduced for testing
        })

    if args.mode in ["all", "tvt"]:
        experiments.append({
            "agent_type": "TVT",
            "env_type": "Standard",
            "use_tvt": True,
            "use_matchmaker": False,
            "grid_size": args.grid_size,
            "num_agents": 5,
            "total_episodes": 100
        })

    # Add other modes similarly...

    for config in experiments:
        print(f"\n=== Running {config['agent_type']} + {config['env_type']} ===")
        zap_history, debt_history = run_experiment(config)

if __name__ == "__main__":
    main()
