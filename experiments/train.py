"""
Karmic-RL Training Pipeline (Optimized & Configurable).

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

    def rollout(self, env, num_steps=500):
        """Collect rollout data using Batched Inference (Optimized)."""
        obs_dict, infos = env.reset()
        
        # Determine device
        device = next(self.agent.parameters()).device
        
        # Get list of agent IDs to ensure consistent ordering
        agent_ids = list(obs_dict.keys())
        num_agents = len(agent_ids)

        # Initialize Hidden States for ALL agents: (Batch=Num_Agents, Hidden)
        hx, cx = self.agent.get_initial_state(num_agents)
        hx = hx.to(device)
        cx = cx.to(device)

        steps_collected = 0
        while steps_collected < num_steps:
            # 1. Prepare Batch Observation
            # Convert dict {agent_id: obs} -> Tensor (Num_Agents, C, H, W)
            obs_list = [obs_dict[aid] for aid in agent_ids]
            
            # Numpy (H, W, C) -> Torch (C, H, W)
            obs_tensor = torch.FloatTensor(np.array(obs_list)).permute(0, 3, 1, 2).to(device)
            
            # 2. Batched Inference
            # forward(obs, hidden_state) -> logits, value, new_hidden
            logits, values, (new_hx, new_cx), extras = self.agent(obs_tensor, (hx, cx))
            
            # 3. Sample Actions (Vectorized)
            probs = Categorical(logits=logits)
            actions = probs.sample()
            logprobs = probs.log_prob(actions)
            
            # 4. Prepare Actions for Environment
            # Tensor -> Dict {agent_id: int}
            actions_np = actions.cpu().numpy()
            actions_dict = {aid: actions_np[i] for i, aid in enumerate(agent_ids)}
            
            # Update hidden states for next step
            hx, cx = new_hx, new_cx

            # 5. Store PPO data (Keep on GPU/CPU as needed, here we append tensors)
            self.states.append(obs_tensor)      # (Num_Agents, C, H, W)
            self.actions.append(actions)        # (Num_Agents,)
            self.values.append(values)          # (Num_Agents, 1)
            self.logprobs.append(logprobs)      # (Num_Agents,)

            # 6. Step Environment
            next_obs_dict, rewards_dict, terms, truncs, infos = env.step(actions_dict)
            
            # 7. Store Rewards & Dones (Aligned with agent_ids list)
            step_rewards = []
            step_dones = []
            
            for aid in agent_ids:
                r = rewards_dict.get(aid, 0.0)
                term = terms.get(aid, False)
                trunc = truncs.get(aid, False)
                
                step_rewards.append(r)
                step_dones.append(term or trunc)

            self.rewards.append(torch.tensor(step_rewards, device=device))
            self.dones.append(torch.tensor(step_dones, device=device))

            # 8. Update Observations
            obs_dict = next_obs_dict
            steps_collected += 1
            
            # Handle Global Termination
            if not obs_dict: 
                break

            # Log social events from first agent
            if "social_events" in infos.get(agent_ids[0], {}):
                events = infos[agent_ids[0]]["social_events"]
                if events:
                    wandb.log({"zaps": len(events)}, commit=False)

        return self._compute_returns()

    def _compute_returns(self):
        """Compute GAE returns for PPO (Vectorized)."""
        returns = []
        R = torch.zeros_like(self.rewards[0]) # Shape: (Num_Agents,)
        
        for r, v, done in zip(reversed(self.rewards), reversed(self.values), reversed(self.dones)):
            mask = 1.0 - done.float()
            R = r + self.gamma * R * mask
            returns.insert(0, R)
            
        # Stack returns: (Time, Num_Agents) -> Flatten to (Time * Num_Agents)
        returns_tensor = torch.stack(returns).view(-1)
        return returns_tensor, None

    def update(self):
        """PPO update step."""
        if not self.states:
            return

        # Flatten Time & Batch dimensions
        states = torch.stack(self.states).view(-1, *self.states[0].shape[1:])
        actions = torch.stack(self.actions).view(-1)
        
        # Detach old probs/values to fix RuntimeError
        old_logprobs = torch.stack(self.logprobs).view(-1).detach()
        values = torch.stack(self.values).view(-1).squeeze().detach()

        # Compute Returns
        returns, _ = self._compute_returns()
        returns = returns.clone().detach().to(states.device)
        
        # Advantages
        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Epochs
        for _ in range(self.epochs):
            # Pass None for hidden state (Stateless update approximation)
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
    """Run experiment with provided config."""
    wandb.init(
        project="karmic-rl",
        config=config,
        name=f"{config['agent_type']}_{config['env_type']}_{config['grid_size']}x{config['grid_size']}",
    )

    env = HarvestParallelEnv(
        grid_size=config["grid_size"],
        num_agents=config["num_agents"]
    )
    
    if config["use_matchmaker"]:
        env = KarmicMatchmaker(env, debt_strength=1.0)
    
    obs_shape = env.observation_space.shape
    agent_obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    
    agent = KarmicAgent(
        obs_shape=agent_obs_shape,
        action_dim=8,
        use_tvt=config["use_tvt"]
    )
    
    # Move agent to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    
    trainer = PPOTrainer(agent)

    print(f"Starting training on {device} for {config['total_episodes']} episodes...")
    
    for episode in range(config["total_episodes"]):
        # Use episode_max_steps from config for rollout length
        trainer.rollout(env, num_steps=config["episode_max_steps"])
        trainer.update()
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{config['total_episodes']} Complete")
            
    return [], []

def main():
    parser = argparse.ArgumentParser()
    
    # 1. Config Preset Selection
    parser.add_argument("--config_preset", type=str, default="small_village", 
                        choices=["small_village", "big_city"],
                        help="Choose environment preset from env_harvest.yaml")
    
    parser.add_argument("--mode", type=str, default="baseline", 
                        choices=["baseline", "full", "all"])
    
    # 2. Experiment Duration Control
    parser.add_argument("--episodes", type=int, default=50, 
                        help="Total episodes to run (overrides yaml defaults if any)")
    
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 3. Load YAML Config
    try:
        with open("configs/env_harvest.yaml", "r") as f:
            yaml_config = yaml.safe_load(f)
            preset_config = yaml_config.get(args.config_preset, {})
    except FileNotFoundError:
        print("Warning: configs/env_harvest.yaml not found. Using hardcoded defaults.")
        preset_config = {"grid_size": 15, "num_agents": 5, "max_steps": 500}

    # 4. Merge Configs
    config = {
        "grid_size": preset_config.get("grid_size", 15),
        "num_agents": preset_config.get("num_agents", 5),
        "episode_max_steps": preset_config.get("max_steps", 500), # Physics duration
        "total_episodes": args.episodes, # Experiment duration
        "seed": args.seed
    }
    
    print(f"\n=== Configuration: {args.config_preset} ===")
    print(f"Grid: {config['grid_size']}x{config['grid_size']}, Agents: {config['num_agents']}")
    print(f"Physics Steps/Ep: {config['episode_max_steps']}")
    print(f"Total Episodes: {config['total_episodes']}")
    print("=========================================\n")
    
    if args.mode == "baseline":
        config.update({"agent_type": "DRQN", "env_type": "Standard", "use_tvt": False, "use_matchmaker": False})
        run_experiment(config)
        
    elif args.mode == "full":
        config.update({"agent_type": "Karmic", "env_type": "Matchmaker", "use_tvt": True, "use_matchmaker": True})
        run_experiment(config)
        
    elif args.mode == "all":
        conditions = [
            {"agent_type": "DRQN", "env_type": "Standard", "use_tvt": False, "use_matchmaker": False},
            {"agent_type": "TVT", "env_type": "Standard", "use_tvt": True, "use_matchmaker": False},
            {"agent_type": "DRQN", "env_type": "Matchmaker", "use_tvt": False, "use_matchmaker": True},
            {"agent_type": "Karmic", "env_type": "Matchmaker", "use_tvt": True, "use_matchmaker": True},
        ]
        for cond in conditions:
            run_config = config.copy()
            run_config.update(cond)
            run_experiment(run_config)

if __name__ == "__main__":
    main()
