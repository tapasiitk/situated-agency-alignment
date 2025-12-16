from pettingzoo import ParallelEnv
import gymnasium.spaces as spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class HarvestDualEnv(ParallelEnv):
    """
    HarvestDualEnv: Dual-Use Zap commons game for KARMA

    This environment is a PettingZoo ParallelEnv implementing a modified
    "Harvest" commons game with a DUAL-USE ZAP action:

        - ZAP_AGENT: beam hits another agent -> victim is frozen temporarily
        - ZAP_WASTE: beam hits waste tiles -> waste removed, apples regrow easier

    The environment logs *structured social events* to support KARMA's
    role-invariant contrastive learning:

        - ZAP_AGENT:    attacker harms victim (aggressor view)
        - BEING_ZAPPED: victim is harmed by attacker (victim view)
        - ZAP_WASTE:    actor cleans waste (cooperative action)

    Events are attached to the `infos[agent_id]["social_events"]` list
    returned at each step, and can be used to build (aggressor, victim)
    and (violence, cleaning) contrastive pairs.

    Observation:
        - Shape: (grid_size, grid_size, 3), dtype uint8
        - Channel 0: environment (0=empty, 64=waste, 128=apple, 255=wall)
        - Channel 1: agents (unique code per agent)
        - Channel 2: agent headings (0=none, 1-4 cardinal directions)
        - Fog-of-war: only a local window around each agent is visible

    Action space (Discrete(8)):
        0: No-op
        1: Move up
        2: Move down
        3: Move left
        4: Move right
        5: Turn clockwise
        6: Turn counter-clockwise
        7: ZAP (beam in facing direction)

    Rewards (default, can be overridden in wrapper or config):
        +1.0   per apple eaten
        +0.3   per waste tile zapped (ZAP_WASTE)
        +0.1   per agent zapped (ZAP_AGENT, temptation to defect)
        -0.5   when being zapped (victim penalty)
        -0.01  per beam fired (small action cost)

    This environment is designed to be used with parameter-sharing
    recurrent agents (e.g., KARMAAgent) in a multi-agent PPO loop.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "harvest_dual_v0",
        "render_fps": 10,
    }

    def __init__(
        self,
        grid_size: int = 15,
        num_agents: int = 6,
        max_steps: int = 1000,
        apple_density: float = 0.65,
        zap_timeout: int = 50,
        regrowth_speed: float = 1.0,
        zap_waste_reward: float = 0.3,
        zap_agent_reward: float = 0.1,
        zap_cost: float = 0.01,
        waste_spawn_rate: float = 0.10,
    ):
        """
        Args:
            grid_size: side length of square grid.
            num_agents: number of agents.
            max_steps: episode horizon.
            apple_density: initial density of apples in central patch.
            zap_timeout: number of steps a zapped agent remains frozen.
            regrowth_speed: scalar multiplier for apple regrowth rates.
            zap_waste_reward: reward for zapping waste (cooperative).
            zap_agent_reward: reward for zapping other agents (competitive).
            zap_cost: small negative reward for firing a beam.
            waste_spawn_rate: fraction of empty cells turned into waste initially.
        """
        super().__init__()

        # Basic parameters
        self.grid_size = grid_size
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.max_steps = max_steps

        # Commons & zap parameters
        self.apple_density = apple_density
        self.zap_timeout = zap_timeout
        self.regrowth_speed = regrowth_speed
        self.zap_waste_reward = zap_waste_reward
        self.zap_agent_reward = zap_agent_reward
        self.zap_cost = zap_cost
        self.waste_spawn_rate = waste_spawn_rate

        # Entity encoding
        self.EMPTY = 0
        self.APPLE = 1
        self.WASTE = 2
        self.WALL = 255

        # Action & observation spaces
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.grid_size, self.grid_size, 3),
            dtype=np.uint8,
        )

        # Apple regrowth rates based on neighbor count (0-4 neighbors)
        # base_rates = np.array([0.0, 0.01, 0.03, 0.07, 0.12], dtype=np.float32)
        # Leibo et al. (2017) "Conflict" / Scarcity Regime
        # 0 neighbors: 0.0  (Dead zone stays dead)
        # 1 neighbor:  0.001 (0.1% - Lone apples barely regrow)
        # 2 neighbors: 0.005 (0.5% - Small clusters struggle)
        # 3 neighbors: 0.025 (2.5% - Healthy clusters recover slowly)
        # 4 neighbors: 0.05  (5.0% - Dense patches are the ONLY sustainable source)

        base_rates = np.array([0.0, 0.001, 0.005, 0.025, 0.05], dtype=np.float32)

        self.regrowth_rates = base_rates * float(self.regrowth_speed)

        # Internal state
        self.grid: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.uint8
        )
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.steps: int = 0

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """
        Reset environment state.

        Returns:
            observations: dict[agent_id] -> obs array
            infos: dict[agent_id] -> {"social_events": []}
        """
        if seed is not None:
            # Use numpy RNG; PettingZoo/Gymnasium will handle outer seeding.
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.steps = 0

        # Reset grid and populate apples + waste
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._spawn_initial_apples()
        self._spawn_initial_waste()

        # Initialize agents at empty cells with random headings
        self.agent_states = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_states[agent_id] = {
                "pos": self._find_empty_spawn(),
                "dir": np.random.randint(0, 4),  # 0:Up, 1:Right, 2:Down, 3:Left
                "frozen_until": 0,
                "agent_code": 10 + i,  # visual ID (avoid collision with env codes)
            }

        observations = {aid: self._get_obs(aid) for aid in self.agents}
        infos = {aid: {"social_events": []} for aid in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        """
        Parallel step function.

        Args:
            actions: dict[agent_id] -> int action in [0..7]

        Returns:
            observations: dict[agent_id] -> obs array
            rewards: dict[agent_id] -> float
            terminations: dict[agent_id] -> bool (never True here)
            truncations: dict[agent_id] -> bool (True when max_steps reached)
            infos: dict[agent_id] -> {"social_events": [event dicts]}
        """
        rewards = {aid: 0.0 for aid in self.agents}
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: False for aid in self.agents}
        infos = {aid: {"social_events": []} for aid in self.agents}

        current_time = self.steps

        # 1. Execute actions (skip frozen agents)
        for agent_id in self.agents:
            state = self.agent_states[agent_id]
            if state["frozen_until"] > current_time:
                continue  # still frozen
            action = actions.get(agent_id, 0)
            self._execute_action(agent_id, action, rewards, infos)

        # 2. Regrow apples and spawn new waste
        self._regrow_apples()
        # self._spawn_dynamic_waste() # disabling waste spawn for now

        # 3. Advance global time and apply truncation
        self.steps += 1
        global_trunc = self.steps >= self.max_steps
        if global_trunc:
            for aid in self.agents:
                truncations[aid] = True
            # PettingZoo convention: clear agents list when done
            self.agents = []

        # 4. New observations
        observations = {aid: self._get_obs(aid) for aid in terminations.keys()}

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _execute_action(
        self,
        agent_id: str,
        action: int,
        rewards: Dict[str, float],
        infos: Dict[str, dict],
    ):
        """Dispatch high-level action."""
        if action == 0:
            return
        elif 1 <= action <= 4:
            self._move_agent(agent_id, action, rewards)
        elif action in (5, 6):
            self._turn_agent(agent_id, action)
        elif action == 7:
            self._zap_dual(agent_id, rewards, infos)

    def _move_agent(self, agent_id: str, action: int, rewards: Dict[str, float]):
        """
        Handle cardinal movement and apple consumption.

        Movement codes (1-4) correspond to:
            1: Up (-1, 0)
            2: Down (+1, 0)
            3: Left (0, -1)
            4: Right (0, +1)
        """
        state = self.agent_states[agent_id]
        r, c = state["pos"]
        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = deltas[action]
        nr, nc = r + dr, c + dc

        # Bounds and wall check
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            return
        if self.grid[nr, nc] == self.WALL:
            return

        # Collision with other agents
        for other_id, other_state in self.agent_states.items():
            if other_id == agent_id:
                continue
            if np.array_equal(other_state["pos"], np.array([nr, nc])):
                return  # blocked

        # Move
        state["pos"] = np.array([nr, nc], dtype=int)

        # Eat apple
        if self.grid[nr, nc] == self.APPLE:
            self.grid[nr, nc] = self.EMPTY
            rewards[agent_id] += 1.0

    def _turn_agent(self, agent_id: str, action: int):
        """
        Rotate agent left/right in place.

        5: clockwise (dir + 1 mod 4)
        6: counter-clockwise (dir - 1 mod 4)
        """
        state = self.agent_states[agent_id]
        if action == 5:
            state["dir"] = (state["dir"] + 1) % 4
        else:
            state["dir"] = (state["dir"] - 1) % 4

    def _zap_dual(
        self,
        attacker_id: str,
        rewards: Dict[str, float],
        infos: Dict[str, dict],
    ):
        """
        Dual-use ZAP:
          - If beam hits another agent -> ZAP_AGENT (aggression)
          - If beam hits waste -> ZAP_WASTE (cooperation)

        Logs:
            - For aggressor: ZAP_AGENT event
            - For victim:   BEING_ZAPPED event
            - For cleaner:  ZAP_WASTE event
        """
        attacker_state = self.agent_states[attacker_id]
        attacker_pos = attacker_state["pos"]
        direction = attacker_state["dir"]

        # Small cost to discourage spam
        rewards[attacker_id] -= self.zap_cost

        # Beam path positions
        beam_positions = self._get_beam_path(attacker_pos, direction)

        # Track hits
        agent_hits: List[str] = []
        waste_hits: List[Tuple[int, int]] = []

        # Check AGENT hits
        for victim_id, victim_state in self.agent_states.items():
            if victim_id == attacker_id:
                continue
            victim_pos = victim_state["pos"]

            if any(np.array_equal(victim_pos, bpos) for bpos in beam_positions):
                # Agent is hit -> freeze victim
                victim_state["frozen_until"] = self.steps + self.zap_timeout
                rewards[victim_id] -= 0.5
                rewards[attacker_id] += self.zap_agent_reward
                agent_hits.append(victim_id)

                # Context for event
                apple_ctx = self._count_nearby_apples(victim_pos) > 0
                dist = float(np.linalg.norm(victim_pos - attacker_pos))

                # Attacker view: ZAP_AGENT
                event_attacker = {
                    "type": "ZAP_AGENT",
                    "attacker": attacker_id,
                    "victim": victim_id,
                    "timestamp": self.steps,
                    "pos": victim_pos.tolist(),
                    "apple_context": apple_ctx,
                    "zap_distance": dist,
                }
                infos[attacker_id]["social_events"].append(event_attacker)

                # Victim view: BEING_ZAPPED
                event_victim = {
                    "type": "BEING_ZAPPED",
                    "attacker": attacker_id,
                    "victim": victim_id,
                    "timestamp": self.steps,
                    "pos": victim_pos.tolist(),
                    "apple_context": apple_ctx,
                    "zap_distance": dist,
                }
                infos[victim_id]["social_events"].append(event_victim)

        # Check WASTE hits (may overlap spatially with agent hits but reward separately)
        for bpos in beam_positions:
            r, c = int(bpos[0]), int(bpos[1])
            if self.grid[r, c] == self.WASTE:
                self.grid[r, c] = self.EMPTY
                rewards[attacker_id] += self.zap_waste_reward
                waste_hits.append((r, c))

        if waste_hits:
            # Log cooperative event for contrastive controls / KARMA
            event_clean = {
                "type": "ZAP_WASTE",
                "actor": attacker_id,
                "timestamp": self.steps,
                "waste_positions": [list(p) for p in waste_hits],
            }
            infos[attacker_id]["social_events"].append(event_clean)

    # ------------------------------------------------------------------
    # Environment dynamics
    # ------------------------------------------------------------------

    def _get_beam_path(self, start_pos: np.ndarray, direction: int) -> List[np.ndarray]:
        """
        Compute beam path as up to 3 tiles in the facing direction.
        Directions:
            0: Up    (-1, 0)
            1: Right (0, +1)
            2: Down  (+1, 0)
            3: Left  (0, -1)
        """
        r, c = int(start_pos[0]), int(start_pos[1])
        dr_dc = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}[direction]
        dr, dc = dr_dc
        path: List[np.ndarray] = []
        for d in range(1, 4):
            nr, nc = r + dr * d, c + dc * d
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                path.append(np.array([nr, nc], dtype=int))
            else:
                break
        return path

    def _regrow_apples(self):
        """
        Density-dependent apple regrowth:
          - For each EMPTY cell, look at 3x3 neighborhood of apples.
          - Use neighbor count to index regrowth_rates.
        """
        apple_layer = (self.grid == self.APPLE).astype(np.float32)
        rows, cols = np.where(self.grid == self.EMPTY)

        for r, c in zip(rows, cols):
            if r in (0, self.grid_size - 1) or c in (0, self.grid_size - 1):
                continue
            window = apple_layer[r - 1 : r + 2, c - 1 : c + 2]
            neighbor_count = int(np.sum(window))
            rate = self.regrowth_rates[min(neighbor_count, 4)]
            if np.random.random() < rate:
                self.grid[r, c] = self.APPLE
    
    def _spawn_initial_apples(self):
        """Spawn TWO distinct patches to force territory dynamics."""
        # Patch 1: Top-Leftish (Rows 2-8, Cols 1-5)
        # Patch 2: Bottom-Rightish (Rows 7-13, Cols 9-13)
        # The separation (Cols 6-8) is the "Dead Zone"
        
        density = 0.8  # High density inside the patch to make it valuable
        
        # Left Patch
        p1 = np.random.choice([self.EMPTY, self.APPLE], size=(7, 5), p=[1-density, density])
        self.grid[2:9, 1:6] = p1.astype(np.uint8)
        
        # Right Patch
        p2 = np.random.choice([self.EMPTY, self.APPLE], size=(7, 5), p=[1-density, density])
        self.grid[7:14, 9:14] = p2.astype(np.uint8)


    # def _spawn_initial_apples(self):
    #     """
    #     Initialize a central apple patch of size roughly (grid_size/2)^2
    #     with given density.
    #     """
    #     half = self.grid_size // 2
    #     radius = max(2, self.grid_size // 4)
    #     r0, r1 = max(0, half - radius), min(self.grid_size, half + radius)
    #     c0, c1 = max(0, half - radius), min(self.grid_size, half + radius)

    #     patch = np.random.choice(
    #         [self.EMPTY, self.APPLE],
    #         size=(r1 - r0, c1 - c0),
    #         p=[1.0 - self.apple_density, self.apple_density],
    #     ).astype(np.uint8)

    #     self.grid[r0:r1, c0:c1] = patch

    def _spawn_initial_waste(self):
        """Spawn waste randomly on a fraction of currently empty cells."""
        empty_positions = np.argwhere(self.grid == self.EMPTY)
        if len(empty_positions) == 0:
            return
        num_waste = int(len(empty_positions) * self.waste_spawn_rate)
        if num_waste <= 0:
            return

        idx = np.random.choice(len(empty_positions), size=num_waste, replace=False)
        for i in idx:
            r, c = empty_positions[i]
            self.grid[r, c] = self.WASTE

    def _spawn_dynamic_waste(self):
        """
        Optional dynamic waste spawning during the episode to
        maintain an ongoing cleaning task.
        """
        if np.random.random() < 0.02:  # low probability per step
            empty_positions = np.argwhere(self.grid == self.EMPTY)
            if len(empty_positions) == 0:
                return
            r, c = empty_positions[np.random.randint(len(empty_positions))]
            self.grid[r, c] = self.WASTE

    def _find_empty_spawn(self) -> np.ndarray:
        """Sample an empty cell not occupied by any agent (up to 100 tries)."""
        for _ in range(100):
            r, c = np.random.randint(0, self.grid_size, size=2)
            if self.grid[r, c] != self.EMPTY:
                continue
            if any(
                np.array_equal(s["pos"], np.array([r, c]))
                for s in self.agent_states.values()
            ):
                continue
            return np.array([r, c], dtype=int)
        # Fallback (should rarely happen)
        return np.array([0, 0], dtype=int)

    def _count_nearby_apples(self, pos: np.ndarray, radius: int = 2) -> int:
        """Count apples in a (2*radius+1)^2 neighborhood around pos."""
        r, c = int(pos[0]), int(pos[1])
        r0, r1 = max(0, r - radius), min(self.grid_size, r + radius + 1)
        c0, c1 = max(0, c - radius), min(self.grid_size, c + radius + 1)
        return int(np.sum(self.grid[r0:r1, c0:c1] == self.APPLE))

    # ------------------------------------------------------------------
    # Observations & rendering
    # ------------------------------------------------------------------

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """
        Build a (H, W, 3) observation for a given agent with fog-of-war.

        - Channel 0: environment layer (0, 64, 128, 255)
        - Channel 1: agent codes
        - Channel 2: direction codes
        """
        # Environment channel
        env_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        env_layer[self.grid == self.WASTE] = 64
        env_layer[self.grid == self.APPLE] = 128
        env_layer[self.grid == self.WALL] = 255

        # Agents + directions
        agent_layer = np.zeros_like(env_layer)
        dir_layer = np.zeros_like(env_layer)

        for aid, state in self.agent_states.items():
            r, c = state["pos"]
            agent_layer[r, c] = state["agent_code"]
            dir_layer[r, c] = state["dir"] + 1  # 1..4

        obs = np.stack([env_layer, agent_layer, dir_layer], axis=-1)

        # Fog-of-war around the requesting agent
        my_pos = self.agent_states[agent_id]["pos"]
        r, c = int(my_pos[0]), int(my_pos[1])
        view_radius = 5

        r0, r1 = max(0, r - view_radius), min(self.grid_size, r + view_radius + 1)
        c0, c1 = max(0, c - view_radius), min(self.grid_size, c + view_radius + 1)

        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        mask[r0:r1, c0:c1] = True

        obs[~mask] = 0
        return obs.astype(np.uint8)

    # Optional PettingZoo API stubs

    def render(self):
        """No-op placeholder; can be extended for visualization."""
        pass

    def close(self):
        """No-op placeholder for cleanup."""
        pass

    @property
    def num_agents(self) -> int:
        """Return current number of agents."""
        return len(self.agents)
