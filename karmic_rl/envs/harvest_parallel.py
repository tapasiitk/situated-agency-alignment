from pettingzoo import ParallelEnv
import pettingzoo.utils.agent_selector as AgentSelector
import gymnasium.spaces as spaces
import numpy as np
import gymnasium

class HarvestParallelEnv(ParallelEnv):
    """
    Harvest (Commons Tragedy) - PettingZoo Native Implementation.
    Multi-agent gridworld where agents compete for apples that regrow
    based on local density. Agents can 'zap' each other to temporarily
    remove competition.

    KEY FEATURES:
    - PettingZoo Parallel API (simultaneous actions)
    - Rich social event logging for TVT/Karmic analysis
    - Proper agent lifecycle handling (zapped agents are skipped)
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "karmic_harvest_v0",
        "render_fps": 10
    }

    def __init__(self, 
                 grid_size=20, 
                 num_agents=5, 
                 max_steps=1000, 
                 apple_density=0.8,
                 zap_timeout=25,
                 regrowth_speed=1.0):
        
        # NOTE: Do NOT call super().__init__() here if it does nothing useful 
        # or if it tries to set attributes we want to control. 
        # For PettingZoo ParallelEnv, it's often safe to skip or just call it.
        # But we MUST NOT set self.num_agents manually as it is a property.
        
        self.grid_size = grid_size
        
        # CORRECT: Define agents list. The parent class property 'num_agents' 
        # reads len(self.agents), so we don't set it directly.
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        self.max_steps = max_steps
        self.apple_density = apple_density
        self.zap_timeout = zap_timeout

        # Entity encoding
        self.EMPTY = 0
        self.APPLE = 1
        self.WALL = 255

        # Action space
        self.action_space = spaces.Discrete(8)  # 0=Noop, 1-4=Move, 5-6=Turn, 7=Zap

        # Observation: (H, W, 3) - [Apples, Agents, Directions]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(grid_size, grid_size, 3), dtype=np.uint8
        )

        # Apple regrowth rates based on neighbor count (0-4 neighbors)
        # self.regrowth_rates = np.array([0.0, 0.005, 0.02, 0.05, 0.1])
        base_rates = np.array([0.0, 0.005, 0.02, 0.05, 0.1])
        
        # Scale rates by the scarcity lever
        self.regrowth_rates = base_rates * regrowth_speed
        
        # Initialize internal state placeholders
        self.grid = None
        self.agent_states = {}
        self.steps = 0

    def reset(self, seed=None, options=None):
        """
        PettingZoo reset returns dicts for all agents.
        """
        # FIX: Do NOT call super().reset(seed=seed) because it raises NotImplementedError
        # Instead, we handle seeding manually if needed.
        if seed is not None:
            self.np_random, seed = gymnasium.utils.seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        self.steps = 0

        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._spawn_initial_apples()

        # Initialize agents
        self.agent_states = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_states[agent_id] = {
                "pos": self._find_empty_spawn(),
                "dir": np.random.randint(0, 4),  # 0=Up, 1=Right, 2=Down, 3=Left
                "frozen_until": 0,               # Timestamp when they recover from zap
                "agent_code": 2 + i              # Visual ID on grid
            }

        # Reset observations and infos
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {"social_events": []} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """
        actions: dict {agent_id: action}
        Returns 5-tuple of dicts (obs, rews, terms, truncs, infos)
        """
        # Process all actions simultaneously (PettingZoo Parallel)
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {"social_events": []} for agent in self.agents}

        # 1. Update frozen timers & Execute Actions
        current_time = self.steps
        
        # Shuffle execution order to be fair? (Optional, but good for sim)
        # For strict parallel logic, we might calculate intents first, then resolve.
        # Here we just iterate.
        
        for agent_id in self.agents:
            state = self.agent_states[agent_id]
            
            # Check if frozen
            if state["frozen_until"] > current_time:
                continue  # Skip frozen agents
            
            action = actions.get(agent_id, 0)
            self._execute_action(agent_id, action, rewards, infos)

        # 2. Apple regrowth (Commons tragedy mechanic)
        self._regrow_apples()

        # 3. Global termination check
        self.steps += 1
        global_trunc = self.steps >= self.max_steps
        
        for agent in self.agents:
            truncations[agent] = global_trunc
            terminations[agent] = False  # No individual deaths in this version

        # 4. Generate new observations
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        
        # PettingZoo API requires us to return empty dicts for dead agents if they were in `agents` 
        # but `agents` list management is complex. 
        # For ParallelEnv, we usually keep all agents in `self.agents` until the end 
        # unless they are explicitly removed.
        
        if global_trunc:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _execute_action(self, agent_id, action, rewards, infos):
        """Process single agent action."""
        state = self.agent_states[agent_id]
        
        if action == 0:  # No-op
            return
        elif 1 <= action <= 4:  # Movement
            self._move_agent(agent_id, action, rewards)
        elif 5 <= action <= 6:  # Turn
            self._turn_agent(agent_id, action)
        elif action == 7:  # Zap
            self._zap_agent(agent_id, rewards, infos)

    def _move_agent(self, agent_id, action, rewards):
        """Handle agent movement and apple eating."""
        state = self.agent_states[agent_id]
        r, c = state["pos"]
        
        # Movement deltas
        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = deltas[action]
        nr, nc = r + dr, c + dc

        # Bounds and collision check
        if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size and 
            self.grid[nr, nc] != self.WALL):
            
            # Check for other agents (simple collision avoidance)
            collision = False
            for other_id, other_state in self.agent_states.items():
                if other_id != agent_id and np.array_equal(other_state["pos"], [nr, nc]):
                    collision = True
                    break
            
            if not collision:
                state["pos"] = np.array([nr, nc])
                
                # Eat apple?
                if self.grid[nr, nc] == self.APPLE:
                    self.grid[nr, nc] = self.EMPTY
                    rewards[agent_id] += 1.0

    def _turn_agent(self, agent_id, action):
        """Handle agent rotation."""
        state = self.agent_states[agent_id]
        if action == 5:  # Clockwise
            state["dir"] = (state["dir"] + 1) % 4
        else:  # Counter-clockwise
            state["dir"] = (state["dir"] - 1) % 4

    def _zap_agent(self, attacker_id, rewards, infos):
        """Handle zap/beam attack with social event logging."""
        attacker_state = self.agent_states[attacker_id]
        attacker_pos = attacker_state["pos"]
        direction = attacker_state["dir"]

        # Small cost for firing
        rewards[attacker_id] -= 0.05

        # Calculate beam path
        beam_positions = self._get_beam_path(attacker_pos, direction)

        # Check for hits
        for victim_id, victim_state in self.agent_states.items():
            if victim_id == attacker_id:
                continue
            
            victim_pos = victim_state["pos"]
            
            # Safety check (shouldn't happen with correct logic but good to have)
            if np.array_equal(victim_pos, attacker_pos):
                continue
            
            # Check if victim is in beam path
            # Convert numpy arrays to list of lists for comparison or use any()
            is_hit = False
            for b_pos in beam_positions:
                if np.array_equal(victim_pos, b_pos):
                    is_hit = True
                    break
            
            if is_hit:
                # HIT! Freeze victim
                victim_state["frozen_until"] = self.steps + self.zap_timeout
                rewards[victim_id] -= 0.5 # Penalty for being zapped

                # CRITICAL: Log social event for TVT/Matchmaker
                apple_context = self._count_nearby_apples(victim_pos) > 0
                
                event = {
                    "event_type": "ZAP_HIT",
                    "attacker": attacker_id,
                    "victim": victim_id,
                    "timestamp": self.steps,
                    "apple_context": apple_context,
                    "zap_distance": np.linalg.norm(victim_pos - attacker_pos)
                }
                
                # Both parties get the event in their info
                infos[attacker_id]["social_events"].append(event)
                infos[victim_id]["social_events"].append(event)

    def _get_beam_path(self, start_pos, direction):
        """Calculate positions hit by beam."""
        r, c = start_pos
        # Directions: 0=Up, 1=Right, 2=Down, 3=Left (Check if this matches your Action mapping)
        # Based on _move_agent: 1=Up(-1,0), 2=Down(1,0)... wait, _move_agent uses 1..4.
        # Let's align with the initial init: 0=Up, 1=Right, 2=Down, 3=Left
        
        dr_dc_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = dr_dc_map[direction]
        
        path = []
        # Beam extends 5 tiles
        for dist in range(1, 6):
            tr, tc = r + dr * dist, c + dc * dist
            if 0 <= tr < self.grid_size and 0 <= tc < self.grid_size:
                path.append(np.array([tr, tc]))
            else:
                break # Stop at wall/edge
        return path

    def _regrow_apples(self):
        """Regrowth based on neighbor density (Tragedy of Commons)."""
        apple_layer = (self.grid == self.APPLE).astype(np.float32)
        
        # Iterate only empty spots
        rows, cols = np.where(self.grid == self.EMPTY)
        for r, c in zip(rows, cols):
             # Skip edges to simplify logic (or handle padding)
             if r == 0 or r == self.grid_size - 1 or c == 0 or c == self.grid_size - 1:
                 continue
                 
             # Count neighbors in 3x3 window (excluding self which is empty)
             window = apple_layer[r-1:r+2, c-1:c+2]
             neighbor_count = np.sum(window)
             
             rate = self.regrowth_rates[min(int(neighbor_count), 4)]
             
             if np.random.random() < rate:
                 self.grid[r, c] = self.APPLE

    # def _get_obs(self, agent):
    #     """Generate observation for specific agent."""
    #     # Channel 0: Apples/Walls
    #     apple_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
    #     apple_layer[self.grid == self.APPLE] = 128
    #     apple_layer[self.grid == self.WALL] = 255

    #     # Channel 1: Agents (visual IDs)
    #     agent_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
    #     # Channel 2: Directions
    #     dir_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

    #     current_time = self.steps
        
    #     for agent_id, state in self.agent_states.items():
    #         # Show agents even if frozen? Yes, they exist.
    #         # But maybe visually distinct? keeping simple for now.
    #         r, c = state["pos"]
    #         agent_layer[r, c] = state["agent_code"]
    #         dir_layer[r, c] = state["dir"] + 1  # 1-4 for directions, 0 for nothing

    #     obs = np.stack([apple_layer, agent_layer, dir_layer], axis=-1)
    #     return obs.astype(np.uint8)
        def _get_obs(self, agent_id):
        """Generate observation for specific agent with Fog of War."""
        # 1. Base Layer: Apples & Walls
        apple_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        apple_layer[self.grid == self.APPLE] = 128
        apple_layer[self.grid == self.WALL] = 255

        # 2. Entity Layers: Agents & Directions
        agent_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        dir_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        for other_id, state in self.agent_states.items():
            r, c = state["pos"]
            agent_layer[r, c] = state["agent_code"]
            dir_layer[r, c] = state["dir"] + 1  # 1-4 for directions, 0 for nothing

        # Stack into (H, W, 3)
        obs = np.stack([apple_layer, agent_layer, dir_layer], axis=-1)

        # 3. Apply Fog of War (Partial Observability)
        # Get current agent position
        my_r, my_c = self.agent_states[agent_id]["pos"]
        
        # Define View Radius (e.g., 5 means 11x11 view window)
        view_radius = 5  
        
        # Create visibility mask
        visible_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # Calculate bounds (clamped to grid edges)
        r_min = max(0, my_r - view_radius)
        r_max = min(self.grid_size, my_r + view_radius + 1)
        c_min = max(0, my_c - view_radius)
        c_max = min(self.grid_size, my_c + view_radius + 1)
        
        # Set visible area to True
        visible_mask[r_min:r_max, c_min:c_max] = True
        
        # Apply mask: Set everything NOT visible to 0 (Black)
        # This broadcasts the (H, W) mask across the 3 channels
        obs[~visible_mask] = 0 

        return obs.astype(np.uint8)

    def _spawn_initial_apples(self):
        """Create initial apple distribution."""
        # Central dense patch
        half = self.grid_size // 2
        
        # Safety check for small grids
        start = max(0, half-5)
        end = min(self.grid_size, half+5)
        
        patch = np.random.choice(
            [self.EMPTY, self.APPLE],
            size=(end-start, end-start),
            p=[1-self.apple_density, self.apple_density]
        ).astype(np.uint8)
        
        self.grid[start:end, start:end] = patch

    def _find_empty_spawn(self):
        """Find random empty spawn location."""
        # Fail-safe counter to prevent infinite loops on full grids
        for _ in range(100):
            r, c = np.random.randint(0, self.grid_size, 2)
            if self.grid[r, c] == self.EMPTY:
                # Check if occupied by another agent
                occupied = False
                for s in self.agent_states.values():
                    if np.array_equal(s["pos"], [r, c]):
                        occupied = True
                        break
                if not occupied:
                    return np.array([r, c])
        
        return np.array([0, 0]) # Fallback (should ideally never happen)

    def _count_nearby_apples(self, pos):
        """Count apples in 5x5 neighborhood."""
        r, c = pos
        r_start, r_end = max(0, r-2), min(self.grid_size, r+3)
        c_start, c_end = max(0, c-2), min(self.grid_size, c+3)
        return np.sum(self.grid[r_start:r_end, c_start:c_end] == self.APPLE)

    # Required for PettingZoo API sometimes (optional for Parallel but good practice)
    def render(self):
        pass

    def close(self):
        pass

    def state(self):
        return self.grid.flatten()
