import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional

class GridWorldEnv(gym.Env):
    """
    Variable-size (n × m) deterministic Grid-World.
      • State  : agent (row, col) ∈ {0,…,n-1}×{0,…,m-1}
      • Action : 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
      • Start  : (0, 0)
      • Goal   : (n-1, m-1)
      • Reward : -1 per step (including illegal moves)
      • Episode terminates on reaching the goal.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # ---- constructor ----------------------------------------------------
    def __init__(self,
                shape: Tuple[int, int] = (5, 5),
                render_mode: Optional[str] = None):
        super().__init__()
        self.n_rows, self.n_cols = shape
        self.render_mode = render_mode  # ✅ this line is essential

        # Discrete action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)

        # Observation: (row, col)
        self.observation_space = spaces.MultiDiscrete([self.n_rows, self.n_cols])

        self.agent_pos = (0, 0)


    # ---- gym.Env API -----------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        self.agent_pos = (0, 0)
        # if self.render_mode == "human":
        #     self.render()
        return np.array(self.agent_pos, dtype=np.int32), {}

    def step(self, action: int):
        row, col = self.agent_pos

        if action == 0 and row > 0:                 # UP
            row -= 1
        elif action == 1 and row < self.n_rows - 1: # DOWN
            row += 1
        elif action == 2 and col > 0:               # LEFT
            col -= 1
        elif action == 3 and col < self.n_cols - 1: # RIGHT
            col += 1
        # else: attempted to leave grid → stay in place

        self.agent_pos = (row, col)
        terminated = (row == self.n_rows - 1 and col == self.n_cols - 1)
        reward = -1.0
        truncated = False  # no time-limit here; add if you wish.

        # if self.render_mode == "human":
        #     self.render()

        return (np.array(self.agent_pos, dtype=np.int32),
                reward,
                terminated,
                truncated,
                {})

    # ---- simple string renderer -----------------------------------------
    def render(self):
        grid = [["." for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        ar, ac = self.agent_pos
        grid[ar][ac] = "A"
        grid[self.n_rows - 1][self.n_cols - 1] = "G"
        as_str = "\n".join(" ".join(row) for row in grid)
        if self.render_mode == "human":
            print(as_str + "\n")
            return None      # ✅ fix warning: return None for human
        return as_str        # only return string for other modes (e.g. "ansi")



# -------------------------------------------------------------------------
# Gym registration helper – executed on import
from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="gridworld_env:GridWorldEnv",
    kwargs={"render_mode": "human"} 
)
