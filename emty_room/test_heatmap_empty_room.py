#!/usr/bin/env python
"""
plot_q_5x5_heatmaps.py

Train one Monte-Carlo agent in the MiniHack EMPTY_ROOM, collect Q‐snapshots,
then plot annotated 5×5 heatmaps of max_a Q(s,a) for episodes EP_START..EP_END.

Before any training, if EP_START == 0, you will also see the all-zero map.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from commons import get_crop_chars_from_observation

from minihack_env import EMPTY_ROOM, get_minihack_envirnment
from rl_task import RLTask
from agents import MonteCarloOnPolicy

# -------------------- CONFIG --------------------
N_EPISODES = 80      # total episodes to train
EP_START   = 0       # first episode index to visualize (0 = before training)
EP_END     = 40      # last episode index to visualize (inclusive)

# agent hyperparams
EPSILON = 0.1
GAMMA   = 1.0

# -------------------- ENV & AGENT SETUP --------------------
env = get_minihack_envirnment(EMPTY_ROOM, add_pixel=False)
obs0, _ = env.reset()

chars = get_crop_chars_from_observation(obs0)
spawn_r, spawn_c = np.argwhere(chars == 64)[0]
print("Spawn at (row,col) in 5×5 crop:", spawn_r, spawn_c)

# full chars‐screen is typically ~21×79
full_chars = obs0["chars"]
H_full, W_full = full_chars.shape

# find tight 5×5 crop bounds (non‐space chars)
mask = full_chars != ord(" ")
row_ids = np.where(mask.any(axis=1))[0]
col_ids = np.where(mask.any(axis=0))[0]
r0, r1 = row_ids[0], row_ids[-1]
c0, c1 = col_ids[0], col_ids[-1]
H5, W5 = (r1 - r0 + 1), (c1 - c0 + 1)  # should be 5×5

agent = MonteCarloOnPolicy(
    id="MC",
    action_space=env.action_space,
    epsilon=EPSILON,
    gamma=GAMMA,
)

# -------------------- TRAIN & COLLECT --------------------
task = RLTask(env, agent)
# this returns lists of length N_EPISODES, each a snapshot *after* that episode
_, avg_returns, q_snapshots = task.interact(n_episodes=N_EPISODES)

# -------------------- PLOT FUNCTION --------------------
def build_max_q_5x5(qdict):
    """
    From qdict: {full_chars_bytes→Q-array}, crop to 5×5 and compute max_a Q.
    """
    arr = np.zeros((H5, W5), dtype=float)
    for key_bytes, qvals in qdict.items():
        screen = np.frombuffer(key_bytes, dtype=np.uint8).reshape(H_full, W_full)
        crop5 = screen[r0 : r1 + 1, c0 : c1 + 1]
        # locate the player '@' (ASCII 64)
        r, c = np.argwhere(crop5 == 64)[0]
        arr[r, c] = np.max(qvals)
    return arr

# -------------------- VISUALIZE --------------------
for ep in range(EP_START, EP_END + 1):
    if ep == 0:
        # before any training: all zeros
        maxq5 = np.zeros((H5, W5))
    else:
        # snapshots[0] = after episode 1, so episode ep → index ep-1
        maxq5 = build_max_q_5x5(q_snapshots[ep - 1])

    vmin, vmax = maxq5.min(), maxq5.max()
    mid = 0.5 * (vmin + vmax)

    plt.figure(figsize=(4, 4))
    im = plt.imshow(
        maxq5,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis"
    )
    plt.title(f"Ep {ep:02d}  maxₐ Q(s,a)")
    plt.axis("off")

    # annotate each cell: white on dark, black on light
    for (i, j), val in np.ndenumerate(maxq5):
        color = "white" if val < mid else "black"
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
