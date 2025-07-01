#!/usr/bin/env python
"""
cliff_sarsa_demo.py
-------------------
Trains a tabular SARSA(0) agent on the custom MiniHack CLIFF level
and visualises (i) the cumulative-average return curve and
(ii) the greedy trajectory inside a zoomed-in window that shows
lava squares, the start (@) and the goal (<).
"""
# ---------- standard libs ----------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- project-specific helpers ----------
from minihack_env import CLIFF, get_minihack_envirnment
from agents       import QLearningOffPolicy
from rl_task      import RLTask

# ---------- hyper-parameters ----------
N_EPISODES        = 1000          # training duration
MAX_STEPS_ROLLOUT = 300            # cap for greedy episode
GAMMA             = 1.0
ALPHA             = 0.10
EPSILON_TRAIN     = 0.20           # ε during learning
EPSILON_EVAL      = 0.00           # greedy when we draw the path

# ---------- helper functions ----------
def cumavg(a: np.ndarray) -> np.ndarray:
    """Cumulative average of a 1-D array."""
    return np.cumsum(a) / np.arange(1, len(a) + 1)

def greedy_rollout(agent, env, max_steps: int):
    """
    Executes one greedy episode and returns:
      xs, ys : the x/y coordinates visited (goal included)
      chars  : the final char observation (for lava masking)
    """
    agent.epsilon = EPSILON_EVAL
    state, _      = env.reset()
    xs, ys        = [], []

    for _ in range(max_steps):
        # record position AFTER we have the observation
        pos_y, pos_x = np.argwhere(state["chars"] == ord("@"))[0]
        xs.append(pos_x);  ys.append(pos_y)

        action = agent.act(state)
        state, _, term, trunc, _ = env.step(action)

        if term or trunc:                       # episode finished
            stair = np.argwhere(state["chars"] == ord("<"))
            if stair.size:                      # always true in cliff
                gy, gx = stair[0]
                xs.append(gx);  ys.append(gy)   # ensure path hits goal
            break
    return np.array(xs), np.array(ys), state["chars"]

def plot_learning(avg_returns: np.ndarray):
    plt.figure(figsize=(6, 3))
    plt.plot(avg_returns, lw=2)
    plt.title("Cumulative-average return — cliff room")
    plt.xlabel("Episode");  plt.ylabel("Average return")
    plt.grid(alpha=.3);  plt.tight_layout()

def plot_path(xs, ys, chars, *, x_lo, x_hi, y_lo, y_hi):
    """Zoom into a window, colour lava, and draw the trajectory centred in each cell,
       with 1-based row/column labels and the final climb onto the '<' staircasing."""
    lava_glyphs = [ord("}"), ord("~"), ord(":")]        # all lava variants
    lava_cells  = np.argwhere(np.isin(chars, lava_glyphs))

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # 1) Grid lines on boundaries
    for x in range(x_lo, x_hi + 1):
        ax.axvline(x, color="0.85", lw=.6, zorder=0)
    for y in range(y_lo, y_hi + 1):
        ax.axhline(y, color="0.85", lw=.6, zorder=0)

    # 2) Lava cells as full 1×1 rectangles
    for ly, lx in lava_cells:
        if x_lo <= lx <= x_hi and y_lo <= ly <= y_hi:
            ax.add_patch(mpatches.Rectangle(
                (lx, ly), 1, 1,
                facecolor="orange", alpha=.4, zorder=1
            ))

    # 3) Plot the agent's path *including* the final stair cell:
    m = (x_lo <= xs) & (xs <= x_hi) & (y_lo <= ys) & (ys <= y_hi)
    ax.plot(xs[m] + 0.5, ys[m] + 0.5, "-", lw=1.5, color="tab:green", alpha=0.8, zorder=3)
    ax.scatter(xs[m] + 0.5, ys[m] + 0.5, s=20, color="tab:green", alpha=0.8, zorder=4)


    # 5) Ticks at the cell *centres*, but labelled 1-based
    xt = np.arange(x_lo, x_hi + 1)
    yt = np.arange(y_lo, y_hi + 1)
    ax.set_xticks(xt + 0.5)
    ax.set_xticklabels(xt + 1)      # now shows 31→32, …, 45→46 if you like, or just drop the +1 on X
    ax.set_yticks(yt + 0.5)
    ax.set_yticklabels(yt + 1)      # THIS makes array-index 11 read as tick 12

    ax.set_xlim(x_lo,   x_hi + 1)
    ax.set_ylim(y_hi + 1, y_lo)     # invert if you want Y increasing downwards
    ax.set_aspect("equal")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Greedy path after training")
    ax.legend(loc="upper right", fontsize=14)
    plt.tight_layout()
# -*- coding: utf-8 -*-

# ---------- main routine ----------
def main():
    # your factory builds the environment with char-only observations
    env   = get_minihack_envirnment(CLIFF, add_pixel=False)

    agent = QLearningOffPolicy(
        id           = "SARSA–Cliff",
        action_space = env.action_space,
        epsilon      = EPSILON_TRAIN,
        gamma        = GAMMA,
        alpha        = ALPHA
    )
    task  = RLTask(env, agent)

    print(f"Training for {N_EPISODES} episodes …")
    returns = task.interact(N_EPISODES)

    # ----- plots -----
    plot_learning(cumavg(returns))

    xs, ys, chars = greedy_rollout(agent, env, MAX_STEPS_ROLLOUT)
    # adjust these bounds if your cliff window is elsewhere
    plot_path(xs, ys, chars, x_lo=31, x_hi=45, y_lo=8, y_hi=12)

    plt.show()
    env.close()

if __name__ == "__main__":
    main()
