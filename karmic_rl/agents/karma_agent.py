"""
KARMA Agent: Knowledge Acquisition via Role-Invariant Mirror Architecture
github.com/tapasiitk/situated-agency-alignment

Core Architecture:
1. CNN Encoder: Processes grid observations.
2. Siamese Projector: Maps features to a latent space 'z' for contrastive learning.
3. LSTM Core: Maintains temporal memory.
4. Actor-Critic Heads: Standard PPO outputs.

The 'Mirror Test' relies on how we train the Projector:
- Baseline: No contrastive loss.
- Broken Mirror: Semantic confusion (ZAP_AGENT ≈ ZAP_WASTE).
- KARMA: Ethical role invariance (ZAP_AGENT ≈ BEING_ZAPPED).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any

class KarmaAgent(nn.Module):
    """
    Unified PPO Agent for the Mirror Test.
    
    Args:
        obs_shape: (Channels, Height, Width) - Note: Input is usually permuted to (B, C, H, W)
        action_dim: Number of discrete actions (8 for HarvestDual)
        hidden_dim: LSTM hidden state size (256)
        embed_dim: Latent dimension for Siamese Projector (64)
        mode: Experiment condition ("baseline", "broken", "karma")
        contrastive_weight: Scaling factor for KARMA loss (lambda)
    """
    def __init__(self, 
                 obs_shape: Tuple[int, int, int] = (3, 15, 15),
                 action_dim: int = 8,
                 hidden_dim: int = 256,
                 embed_dim: int = 64, 
                 mode: str = "baseline",
                 contrastive_weight: float = 0.1):
        
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.mode = mode
        self.contrastive_weight = contrastive_weight
        
        # ----------------------------------------------------------------
        # 1. Perception (CNN)
        # ----------------------------------------------------------------
        # Standard Nature-DQN style encoder (scaled for 15x15 grid)
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        # ----------------------------------------------------------------
        # 2. The Mirror (Siamese Projector)
        # ----------------------------------------------------------------
        # Projects features to latent space 'z' where we enforce role invariance.
        # This is the "Seat of Ethics" in the architecture.
        self.projector = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)  # z in R^64
        )
        
        # ----------------------------------------------------------------
        # 3. Auxiliary Heads & Memory
        # ----------------------------------------------------------------
        # Role Classifier: Checks if 'z' retains semantic info (optional aux loss)
        self.role_head = nn.Linear(cnn_out_dim, 4) 
        
        # LSTM Core
        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        
        # Actor-Critic Heads
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Internal state storage (for rollout generation)
        self.lstm_h = None
        self.lstm_c = None

    def forward(self, obs: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            obs: Tensor (B, C, H, W)
            hidden: Tuple (h, c) for LSTM. If None, uses internal state.
            
        Returns:
            Dict containing policy logits, values, embeddings, etc.
        """
        batch_size = obs.shape[0]
        
        # 1. Encode
        features = self.cnn(obs)     # (B, cnn_out)
        embedding = self.projector(features)  # (B, embed_dim) -> "The Mirror"
        
        # 2. Memory
        if hidden is None:
            # During rollout, use/update internal state
            if self.lstm_h is None or self.lstm_h.shape[0] != batch_size:
                self.reset_hidden(batch_size, obs.device)
            h, c = self.lstm_h, self.lstm_c
            h_new, c_new = self.lstm(embedding, (h, c))
            self.lstm_h, self.lstm_c = h_new.detach(), c_new.detach()
        else:
            # During training, use provided state
            h_new, c_new = self.lstm(embedding, hidden)
            
        # 3. Decision
        logits = self.actor(h_new)
        value = self.critic(h_new)
        
        return {
            "policy": logits,
            "value": value,
            "embedding": embedding,
            "features": features,
            "new_hidden": (h_new, c_new)
        }

    def reset_hidden(self, batch_size: int = 1, device=torch.device("cpu")):
        """Resets LSTM hidden state."""
        self.lstm_h = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.lstm_c = torch.zeros(batch_size, self.hidden_dim, device=device)

    # ----------------------------------------------------------------
    # Contrastive Learning (The Core Logic)
    # ----------------------------------------------------------------
    
    def compute_contrastive_loss(self, 
                                 embeddings: torch.Tensor, 
                                 social_events_batch: List[List[Dict]],
                                 agent_ids: List[str]) -> torch.Tensor:
        """
        Computes the KARMA loss based on the experimental condition.
        
        Args:
            embeddings: (B, embed_dim) tensor of latent vectors.
            social_events_batch: List of event lists corresponding to each sample in B.
            agent_ids: List of agent IDs corresponding to each sample (to identify role).
            
        Returns:
            Scalar loss tensor.
        """
        if self.mode == "baseline":
            return torch.tensor(0.0, device=embeddings.device)
        
        # 1. Extract Role Labels for each sample in the batch
        #    0=Neutral, 1=Aggressor, 2=Victim, 3=Cleaner
        role_indices = []
        for events, agent_id in zip(social_events_batch, agent_ids):
            role = self._infer_role(events, agent_id)
            role_indices.append(role)
        
        role_indices = torch.tensor(role_indices, device=embeddings.device)
        
        # 2. Select Pairs based on Condition
        loss = torch.tensor(0.0, device=embeddings.device)
        num_pairs = 0
        
        if self.mode == "broken":
            # BROKEN MIRROR: Confuse Aggression (1) with Cleaning (3)
            # We minimize distance between ZAP_AGENT and ZAP_WASTE
            mask_agg = (role_indices == 1)
            mask_cln = (role_indices == 3)
            
            if mask_agg.any() and mask_cln.any():
                # Compute distance between all Aggressors and all Cleaners
                # (Simple mean distance implementation)
                center_agg = embeddings[mask_agg].mean(dim=0)
                center_cln = embeddings[mask_cln].mean(dim=0)
                loss += F.mse_loss(center_agg, center_cln)
                num_pairs += 1

        elif self.mode == "karma":
            # KARMA: Align Aggression (1) with Victimization (2)
            # The "Empathy" Objective
            mask_agg = (role_indices == 1)
            mask_vic = (role_indices == 2)
            
            if mask_agg.any() and mask_vic.any():
                center_agg = embeddings[mask_agg].mean(dim=0)
                center_vic = embeddings[mask_vic].mean(dim=0)
                loss += F.mse_loss(center_agg, center_vic)
                num_pairs += 1
                
        if num_pairs == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        return loss * self.contrastive_weight

    def _infer_role(self, events: List[Dict], agent_id: str) -> int:
        """
        Parses social events to determine the agent's role at this timestep.
        
        Priority:
        1. BEING_ZAPPED (Victim) - High salience "pain"
        2. ZAP_AGENT (Aggressor) - High salience "action"
        3. ZAP_WASTE (Cleaner) - Medium salience
        0. Neutral
        """
        is_victim = False
        is_aggressor = False
        is_cleaner = False
        
        for e in events:
            etype = e.get("type", "")
            
            # Check Victim
            if etype == "BEING_ZAPPED" and e.get("victim") == agent_id:
                is_victim = True
            
            # Check Aggressor
            if etype == "ZAP_AGENT" and e.get("attacker") == agent_id:
                is_aggressor = True
                
            # Check Cleaner
            if etype == "ZAP_WASTE" and e.get("actor") == agent_id:
                is_cleaner = True

        # Return code based on priority
        if is_victim: return 2
        if is_aggressor: return 1
        if is_cleaner: return 3
        return 0


# ----------------------------------------------------------------
# Data Collection Utility
# ----------------------------------------------------------------

class KARMACollector:
    """Buffer to store experience for PPO updates."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.social_events = [] # For contrastive loss
        self.agent_ids = []     # To track who is who
        
    def store(self, obs, action, reward, value, log_prob, done, events, agent_id):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.social_events.append(events)
        self.agent_ids.append(agent_id)
        
    def get_batch(self):
        return {
            "obs": np.array(self.obs),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "values": np.array(self.values),
            "log_probs": np.array(self.log_probs),
            "dones": np.array(self.dones),
            "social_events": self.social_events,
            "agent_ids": self.agent_ids
        }
