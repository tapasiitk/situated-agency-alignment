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
import karmic_rl.envs
from karmic_rl.envs.matchmaker_wrapper import KarmicMatchmaker
from karmic_rl.agents.ktvt_agent import KarmicAgent

class PPOTrainer:
    def __init__(self, agent, lr=3e-4, gamma=0.99, clip=0.2, epochs=4):
        self.agent = agent
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        
        # PPO Buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.dones = []

    def rollout(self, env, num_steps=1000):
        """Collect rollout data."""
        obs, infos = env.reset()
        state = self.agent.get_initial_state(1)
        
        while len(self.rewards) < num_steps:
            # Select action
            agent_obs = torch.FloatTensor(obs).unsqueeze(0)  # (1, C, H, W)
            logits, value, new_state, extras = self.agent(agent_obs, state)
            
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)
            
            # Step environment
            actions_dict = {agent: int(action[0].item()) for agent in env.agents}
            next_obs, rewards, terms, truncs, infos = env.step(actions_dict)
            
            # Store
            self.states.append(agent_obs)
            self.actions.append(action)
            self.rewards.append(rewards[env.agents[0]])  # Track lead agent
            self.values.append(value)
            self.logprobs.append(logprob)
            self.dones.append(terms[env.agents[0]] or truncs[env.agents[0]])
            
            obs = next_obs
            state = new_state
            
            # Log social events
            if "social_events" in infos[env.agents[0]]:
                wandb.log({"zaps": len(infos[env.agents[0]]["social_events"])})
        
        return self._compute_returns()

    def _compute_returns(self):
        """Compute GAE returns for PPO."""
        advantages = []
        returns = []
        
        R = 0
        for r, v, done in zip(reversed(self.rewards), reversed(self.values), reversed(self.dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        
        return torch.tensor(returns), torch.tensor(advantages)

    def update(self):
        """PPO update step."""
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_logprobs = torch.cat(self.logprobs)
        returns = self._compute_returns()[0]
        advantages = returns - torch.cat(self.values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        for _ in range(self.epochs):
            logits, values, _, _ = self.agent(states, None)
            
            probs = Categorical(logits=logits)
            new_logprobs = probs.log_prob(actions)
            entropy = probs.entropy()
            
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            entropy_loss = -entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffers
        self.states, self.actions, self.rewards, self.values, self.logprobs, self.dones = [], [], [], [], [], []

def run_experiment(config):
    """Run single ablation condition."""
    wandb.init(
        project="karmic-rl",
        config=config,
        name=f"{config['agent_type']}_{config['env_type']}_{config['grid_size']}x{config['grid_size']}"
    )
    
    # Create environment
    env = gym.make("KarmicHarvest-v0", 
                   grid_size=config["grid_size"], 
                   num_agents=config["num_agents"])
    
    # Apply matchmaker if needed
    if config["use_matchmaker"]:
        env = KarmicMatchmaker(env, debt_strength=1.0)
    
    # Create agent
    obs_shape = env.observation_space.shape
    agent = KarmicAgent(obs_shape, action_dim=8, use_tvt=config["use_tvt"])
    trainer = PPOTrainer(agent)
    
    # Training loop
    zap_history = []
    debt_history = []
    
    for episode in range(config["total_episodes"]):
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
        
        if episode % 100 == 0:
            print(f"Ep {episode}: Zaps={zap_history[-1] if zap_history else 0}, Debt={debt_history[-1] if debt_history else 0}")
    
    wandb.finish()
    return zap_history, debt_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["all", "baseline", "tvt", "matchmaker", "full"], default="all")
    parser.add_argument("--grid_size", type=int, default=20)
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
            "total_episodes": 2000
        })
    
    if args.mode in ["all", "tvt"]:
        experiments.append({
            "agent_type": "TVT",
            "env_type": "Standard", 
            "use_tvt": True,
            "use_matchmaker": False,
            "grid_size": args.grid_size,
            "num_agents": 5,
            "total_episodes": 2000
        })
    
    if args.mode in ["all", "matchmaker"]:
        experiments.append({
            "agent_type": "DRQN", 
            "env_type": "Karmic",
            "use_tvt": False,
            "use_matchmaker": True,
            "grid_size": args.grid_size,
            "num_agents": 5,
            "total_episodes": 2000
        })
    
    if args.mode in ["all", "full"]:
        experiments.append({
            "agent_type": "Karmic-RL",
            "env_type": "Karmic",
            "use_tvt": True,
            "use_matchmaker": True,
            "grid_size": args.grid_size,
            "num_agents": 5,
            "total_episodes": 2000
        })
    
    for config in experiments:
        print(f"\n=== Running {config['agent_type']} + {config['env_type']} ===")
        zap_history, debt_history = run_experiment(config)

if __name__ == "__main__":
    main()
