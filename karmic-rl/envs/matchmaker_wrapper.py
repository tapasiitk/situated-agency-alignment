from pettingzoo.wrappers.base import ParallelEnvWrapper
import numpy as np
from typing import Dict, List, Tuple, Optional

class KarmicMatchmaker(ParallelEnvWrapper):
    """
    The "Conspiring Matchmaker" - Forces karmic debts to be settled.
    
    Mechanics:
    1. Tracks social debts across episodes (who zapped whom unjustly)
    2. During reset(), rigs spawn locations to pair debtors with creditors
    3. Distinguishes predation from justified retribution
    
    Ablation: Set debt_strength=0.0 to disable (random spawning).
    """
    
    def __init__(self, env, debt_strength=1.0, decay_rate=0.9):
        """
        Args:
            debt_strength: How aggressively to rig spawns (0.0 = random)
            decay_rate: How fast old debts fade (1.0 = permanent)
        """
        super().__init__(env)
        self.debt_strength = debt_strength
        self.decay_rate = decay_rate
        
        # Debt Matrix: debt_matrix[i][j] = Agent i owes Agent j
        self.num_agents = env.num_agents
        self.agent_ids = env.agents
        self.id_to_idx = {agent_id: i for i, agent_id in enumerate(self.agent_ids)}
        
        self.debt_matrix = np.zeros((self.num_agents, self.num_agents), dtype=np.float32)
        
        # Stats for analysis
        self.total_zaps = 0
        self.retribution_zaps = 0
        self.predatory_zaps = 0
        
        # Spawn rigging cache
        self._rigged_spawn_locations = {}

    def reset(self, seed=None, options=None):
        """Override reset to rig agent spawning based on debts."""
        if self.debt_strength > 0:
            self._rigged_spawn_locations = self._compute_rigged_spawns()
        else:
            self._rigged_spawn_locations = {}
        
        # Decay old debts
        self.debt_matrix *= self.decay_rate
        
        obs, infos = self.env.reset(seed=seed, options=options)
        
        # Force rigged positions into agent states
        self._apply_rigged_spawns()
        
        # Refresh observations after repositioning
        for agent in self.agents:
            infos[agent]["debt_score"] = self._get_agent_debt(agent)
        
        return obs, infos

    def step(self, actions):
        """Intercept social events to update debt ledger."""
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Process social events from all agents
        self._process_social_events(infos)
        
        return obs, rewards, terminations, truncations, infos

    def _process_social_events(self, infos: Dict[str, dict]):
        """Update debt matrix based on zap events."""
        processed_events = set()
        
        for agent_id, info in infos.items():
            if "social_events" not in info:
                continue
                
            for event in info["social_events"]:
                if event.get("event_type") == "ZAP_HIT":
                    event_key = (event["attacker"], event["victim"], event["timestamp"])
                    if event_key not in processed_events:
                        self._update_debt_from_event(event)
                        processed_events.add(event_key)

    def _update_debt_from_event(self, event: dict):
        """Justification logic: Predation vs Retribution."""
        attacker = event["attacker"]
        victim = event["victim"]
        
        if attacker not in self.id_to_idx or victim not in self.id_to_idx:
            return
            
        idx_a = self.id_to_idx[attacker]
        idx_v = self.id_to_idx[victim]
        
        self.total_zaps += 1
        
        # RETRIBUTION CHECK: Does victim owe attacker?
        if self.debt_matrix[idx_v][idx_a] > 0.1:
            # Justified! Reduce victim's debt
            self.debt_matrix[idx_v][idx_a] *= 0.8
            self.retribution_zaps += 1
            return  # No new debt for attacker
        
        # PREDATION: Attacker accrues debt
        debt_amount = 1.0
        if event.get("apple_context", False):
            debt_amount *= 1.5  # Resource disputes are worse
        
        self.debt_matrix[idx_a][idx_v] += debt_amount
        self.predatory_zaps += 1

    def _compute_rigged_spawns(self) -> Dict[str, np.ndarray]:
        """Rig spawn locations to force debtor-creditor meetings."""
        if self.debt_strength == 0:
            return {}
            
        matrix_copy = np.copy(self.debt_matrix)
        spawns = {}
        assigned = set()
        grid_size = self.env.grid_size
        
        # Find top debt pairs (greedy matching)
        for _ in range(self.num_agents // 2):
            # Find highest remaining debt
            max_idx = np.argmax(matrix_copy)
            debtor_idx, creditor_idx = np.unravel_index(max_idx, matrix_copy.shape)
            debt_val = matrix_copy[debtor_idx, creditor_idx]
            
            if debt_val < 0.1:
                break
                
            debtor = self.agent_ids[debtor_idx]
            creditor = self.agent_ids[creditor_idx]
            
            if debtor in assigned or creditor in assigned:
                matrix_copy[debtor_idx, creditor_idx] = -1
                continue
            
            # Create "fight pit" spawn locations
            arena_r = np.random.randint(2, grid_size - 2)
            arena_c = np.random.randint(2, grid_size - 2)
            
            spawns[debtor] = np.array([arena_r, arena_c])
            spawns[creditor] = np.array([arena_r, arena_c + 1])  # Adjacent
            
            assigned.add(debtor)
            assigned.add(creditor)
            
            # Remove this pair from consideration
            matrix_copy[debtor_idx, :] = -1
            matrix_copy[:, creditor_idx] = -1
        
        return spawns

    def _apply_rigged_spawns(self):
        """Force agent positions to rigged locations."""
        if not self._rigged_spawn_locations:
            return
            
        occupied = set()
        env_states = self.env.agent_states
        
        # Place rigged pairs first
        for agent_id, pos in self._rigged_spawn_locations.items():
            if agent_id in env_states and self.env.grid[tuple(pos)] == 0:
                env_states[agent_id]["pos"] = pos
                occupied.add(tuple(pos))
        
        # Reposition anyone who collided
        for agent_id, state in env_states.items():
            pos_tuple = tuple(state["pos"])
            if pos_tuple in occupied and agent_id not in self._rigged_spawn_locations:
                state["pos"] = self.env._find_empty_spawn()
    
    def _get_agent_debt(self, agent_id: str) -> float:
        """Total debt owed by this agent."""
        if agent_id not in self.id_to_idx:
            return 0.0
        idx = self.id_to_idx[agent_id]
        return float(np.sum(self.debt_matrix[idx, :]))

    def get_ledger_stats(self) -> dict:
        """Debugging stats."""
        total_debt = float(np.sum(self.debt_matrix))
        return {
            "total_zaps": self.total_zaps,
            "retribution_zaps": self.retribution_zaps,
            "predatory_zaps": self.predatory_zaps,
            "total_debt": total_debt,
            "avg_debt": total_debt / max(1, self.num_agents),
            "debt_matrix": self.debt_matrix.copy()
        }

    def clear_ledger(self):
        """Reset for new experiment run."""
        self.debt_matrix.fill(0)
        self.total_zaps = 0
        self.retribution_zaps = 0
        self.predatory_zaps = 0
