from pettingzoo.utils.env import ParallelEnv
import numpy as np
from typing import Dict, List, Tuple, Optional

# --- COMPATIBILITY FIX: Define BaseParallelWrapper for PettingZoo 1.24.3 ---
# This class was removed in recent versions, so we define it manually to ensure
# the wrapper works correctly without external dependencies.
class BaseParallelWrapper(ParallelEnv):
    """
    A base wrapper for PettingZoo Parallel environments.
    Redirects all unknown attribute access to the wrapped environment.
    """
    def __init__(self, env):
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.possible_agents = getattr(env, "possible_agents", [])
        self.agents = getattr(env, "agents", [])
        
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        return self.env.step(actions)

    def render(self):
        return self.env.render()
        
    def close(self):
        return self.env.close()

    def state(self):
        return self.env.state()
# ---------------------------------------------------------------------------

class KarmicMatchmaker(BaseParallelWrapper):
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
        # Use possible_agents to ensure fixed size
        self.all_agents = env.possible_agents
        self.num_agents = len(self.all_agents)
        self.id_to_idx = {agent_id: i for i, agent_id in enumerate(self.all_agents)}
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
        # Note: In a real partially-observable env, we should re-compute observations here.
        # For now, we update the info dict so agents 'know' their debt.
        current_agents = self.env.agents
        for agent in current_agents:
            if agent in infos:
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
                    # Unique ID for event to prevent double counting
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
        # Threshold 0.1 prevents retribution for tiny accidental debts
        if self.debt_matrix[idx_v][idx_a] > 0.1:
            # Justified! Reduce victim's debt (Retribution pays off debt)
            # We reduce it by 20% or a fixed amount. Here we use decay.
            self.debt_matrix[idx_v][idx_a] *= 0.8
            self.retribution_zaps += 1
            return # No new debt for attacker, act was justified.
            
        # PREDATION: Attacker accrues debt
        debt_amount = 1.0
        if event.get("apple_context", False):
            debt_amount *= 1.5 # Resource disputes are weighted higher
            
        self.debt_matrix[idx_a][idx_v] += debt_amount
        self.predatory_zaps += 1

    def _compute_rigged_spawns(self) -> Dict[str, np.ndarray]:
        """Rig spawn locations to force debtor-creditor meetings."""
        if self.debt_strength == 0:
            return {}
            
        matrix_copy = np.copy(self.debt_matrix)
        spawns = {}
        assigned = set()
        
        # Access grid size safely
        grid_size = getattr(self.env, "grid_size", 20)
        
        # Find top debt pairs (greedy matching)
        # Limit to num_agents / 2 pairs max
        for _ in range(self.num_agents // 2):
            # Find highest remaining debt in the matrix
            max_idx = np.argmax(matrix_copy)
            debtor_idx, creditor_idx = np.unravel_index(max_idx, matrix_copy.shape)
            debt_val = matrix_copy[debtor_idx, creditor_idx]
            
            if debt_val < 0.1: # No significant debt left
                break
                
            debtor = self.all_agents[debtor_idx]
            creditor = self.all_agents[creditor_idx]
            
            if debtor in assigned or creditor in assigned:
                matrix_copy[debtor_idx, creditor_idx] = -1
                continue
                
            # Create "fight pit" spawn locations
            # Pick a random spot, place them adjacent
            arena_r = np.random.randint(2, grid_size - 2)
            arena_c = np.random.randint(2, grid_size - 2)
            
            spawns[debtor] = np.array([arena_r, arena_c])
            spawns[creditor] = np.array([arena_r, arena_c + 1]) # Adjacent
            
            assigned.add(debtor)
            assigned.add(creditor)
            
            # Remove this pair from consideration for other matches
            matrix_copy[debtor_idx, :] = -1
            matrix_copy[:, creditor_idx] = -1
            
        return spawns

    def _apply_rigged_spawns(self):
        """Force agent positions to rigged locations."""
        if not self._rigged_spawn_locations:
            return
            
        occupied = set()
        
        # We need access to the underlying environment's state to rig it.
        # This assumes the base env exposes 'agent_states'.
        # If wrapped multiple times, we might need to unwrap.
        base_env = self.env.unwrapped
        if not hasattr(base_env, "agent_states"):
             return

        env_states = base_env.agent_states
        
        # 1. Place rigged pairs first
        for agent_id, pos in self._rigged_spawn_locations.items():
            if agent_id in env_states:
                # Optional: Check for wall collision if grid is available
                if hasattr(base_env, "grid") and base_env.grid[tuple(pos)] == 0: 
                     continue # Skip if wall (safety check)

                env_states[agent_id]["pos"] = pos
                occupied.add(tuple(pos))
                
        # 2. Reposition anyone else who might have randomly spawned on top of them
        for agent_id, state in env_states.items():
            pos_tuple = tuple(state["pos"])
            # If this spot is taken AND this agent wasn't the one assigned to it
            if pos_tuple in occupied and agent_id not in self._rigged_spawn_locations:
                # Respawn collided agent elsewhere
                if hasattr(base_env, "_find_empty_spawn"):
                    state["pos"] = base_env._find_empty_spawn()

    def _get_agent_debt(self, agent_id: str) -> float:
        """Total debt owed by this agent."""
        if agent_id not in
