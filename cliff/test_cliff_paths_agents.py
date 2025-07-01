#!/usr/bin/env python
"""
cliff_paths_stages.py
---------------------
Visualise how greedy trajectories evolve during learning.

Agents:
  • Monte-Carlo (ε decays 0.3 → 0.1 over 2000 episodes)
  • SARSA       (ε = 0.7 fixed)
  • Q-learning  (ε = 0.1 fixed)

For each algorithm we train with the SAME seed, stop at
   500, 1000, 2000 episodes,
take a greedy rollout, and plot the paths.
"""
# ──────────────────────────────────── imports ─────────────────────────────────
import random, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

from minihack_env import CLIFF, get_minihack_envirnment
from agents import (MonteCarloOnPolicyDecay,
                    SARSAOnPolicy, QLearningOffPolicy)
from rl_task import RLTask
# ─────────────────────────────── experiment setup ─────────────────────────────
SEED            = 5
EPISODE_CUTS    = (50, 70, 500) # snapshots
MAX_EPISODES    = max(EPISODE_CUTS)   # full training horizon
MAX_STEPS_ROLL  = 300                 # cap for greedy rollout
GAMMA           = 1.0
ALPHA           = 0.5                 # a bit higher for quicker TD learning

AGENT_SPECS = {
    "Monte-Carlo (decay)": dict(
        cls   = MonteCarloOnPolicyDecay,
        kwargs= dict(epsilon_start=0.3, epsilon_end=0.1,
                     max_episodes=MAX_EPISODES, gamma=GAMMA)
    ),
    "SARSA (ε 0.7)": dict(
        cls   = SARSAOnPolicy,
        kwargs= dict(epsilon=0.7, gamma=GAMMA, alpha=ALPHA)
    ),
    "Q-learning (ε 0.1)": dict(
        cls   = QLearningOffPolicy,
        kwargs= dict(epsilon=0.1, gamma=GAMMA, alpha=ALPHA)
    )
}

LAVA_GLYPHS = [ord("}"), ord("~"), ord(":")]

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)

def cumulative_average(arr):
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)

def greedy_rollout(agent, env):
    """Return xs, ys visited during a greedy episode."""
    obs, _ = env.reset()
    xs, ys = [], []
    for _ in range(MAX_STEPS_ROLL):
        y, x = np.argwhere(obs["chars"] == ord("@"))[0]
        xs.append(x);  ys.append(y)
        a  = agent.act(obs)          # ε already set to 0 outside
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            stair = np.argwhere(obs["chars"] == ord("<"))
            if stair.size:
                sy, sx = stair[0];   xs.append(sx);  ys.append(sy)
            break
    return np.asarray(xs), np.asarray(ys), obs["chars"]

# ───────────────────────────── training & logging ─────────────────────────────
all_returns  = defaultdict(dict)   # algo → {"train": [... returns ...]}
all_trajs    = defaultdict(dict)   # algo → {500:(xs,ys,chars), 1000:...,}

env_master = get_minihack_envirnment(CLIFF, add_pixel=False)

for algo_name, spec in AGENT_SPECS.items():
    print(f"\n=== {algo_name} ===")
    set_seed(SEED)
    env   = env_master        # same env instance; reset() will set its seed too
    agent = spec["cls"](id=algo_name, action_space=env.action_space,
                        **spec["kwargs"])
    task  = RLTask(env, agent)

    returns = []
    for ep in range(1, MAX_EPISODES + 1):
        returns.extend(task.interact(1))   # run 1 episode
        if ep in EPISODE_CUTS:
            agent.epsilon = 0.0            # turn off exploration
            xs, ys, chars = greedy_rollout(agent, env)
            all_trajs[algo_name][ep] = (xs, ys, chars)
            agent.epsilon = agent.epsilon if hasattr(agent, "_epsilon_saved") \
                                          else agent.epsilon  # keep training

    all_returns[algo_name]["train"] = np.asarray(returns, dtype=float)

env_master.close()

# ─────────────────────────────────── plots ────────────────────────────────────
# A 3 × 3 grid:      columns →  500 / 1000 / 2000 episodes
#                    rows   →  Monte-Carlo  /  SARSA  /  Q-learning
EPISODES_TO_PLOT = (50, 70, 500)
ALGOS            = ("Monte-Carlo (decay)", "SARSA (ε 0.7)", "Q-learning (ε 0.1)")
COLOR_MAP        = {
    "Monte-Carlo (decay)" : "tab:blue",
    "SARSA (ε 0.7)"       : "tab:orange",
    "Q-learning (ε 0.1)"  : "tab:green",
}

fig, axs = plt.subplots(len(ALGOS), len(EPISODES_TO_PLOT),
                        figsize=(13, 11), sharex=True, sharey=True)

# zoom-in window limits for the cliff map
x_lo, x_hi = 30, 45
y_lo, y_hi = 8,  12
LAVA_GLYPHS = [ord("}"), ord("~"), ord(":")]

def draw_bg(ax, chars):
    """grid + lava for one axis"""
    # grid
    for x in range(x_lo, x_hi + 1):
        ax.axvline(x, color="0.85", lw=.6, zorder=0)
    for y in range(y_lo, y_hi + 1):
        ax.axhline(y, color="0.85", lw=.6, zorder=0)
    # lava
    lava = np.argwhere(np.isin(chars, LAVA_GLYPHS))
    for ly, lx in lava:
        if x_lo <= lx <= x_hi and y_lo <= ly <= y_hi:
            ax.add_patch(
                mpatches.Rectangle((lx, ly), 1, 1,
                                   facecolor="orange", alpha=.4, zorder=1)
            )
    ax.set_xlim(x_lo, x_hi + 1);  ax.set_ylim(y_hi + 1, y_lo)
    ax.set_aspect("equal")
    ax.set_xticks([]);  ax.set_yticks([])

# choose one chars grid for background (any snapshot will do)
bg_chars = next(iter(all_trajs.values()))[max(EPISODES_TO_PLOT)][2]

for r, algo in enumerate(ALGOS):
    for c, EP in enumerate(EPISODES_TO_PLOT):
        ax = axs[r, c]
        draw_bg(ax, bg_chars)

        xs, ys, _ = all_trajs[algo][EP]
        colour    = COLOR_MAP[algo]
        ax.plot(xs+.5, ys+.5, "-", lw=1.8, color=colour, zorder=3)
        ax.scatter(xs+.5, ys+.5, s=18, color=colour, zorder=4)

        # titles and labels
        if r == 0:
            ax.set_title(f"{EP} episodes", fontsize=11, pad=4)
        if c == 0:
            ax.set_ylabel(algo, fontsize=10)

fig.suptitle("Greedy trajectory of each agent at 3 learning stages", y=0.92)
plt.tight_layout();  plt.show()
