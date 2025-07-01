#!/usr/bin/env python
"""
cliff_sarsa_multi_eps.py
------------------------
Train three SARSA(0) agents on the custom MiniHack CLIFF level
with four different ε values each, and visualise all greedy trajectories
in a single zoomed-in window showing lava, the start (@) and the goal (<).
"""
# ---------- standard libs ----------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- project-specific helpers ----------
from minihack_env import CLIFF, get_minihack_envirnment
from agents       import SARSAOnPolicy
from rl_task      import RLTask

# ---------- hyper-parameters ----------
N_EPISODES        = 5000         # training duration
MAX_STEPS_ROLLOUT = 300            # cap for greedy episode
GAMMA             = 1.0
ALPHA             = 0.10
EPSILON_VALUES    = [0.0, 0.1, 0.2, 0.7]    # four epsilons to compare
TRAJ_COLORS       = ["tab:red", "tab:orange", "yellow", "tab:brown"]

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
    state, _      = env.reset()
    xs, ys        = [], []

    for _ in range(max_steps):
        pos_y, pos_x = np.argwhere(state["chars"] == ord("@"))[0]
        xs.append(pos_x);  ys.append(pos_y)

        action = agent.act(state)
        state, _, term, trunc, _ = env.step(action)

        if term or trunc:
            stair = np.argwhere(state["chars"] == ord("<"))
            if stair.size:
                gy, gx = stair[0]
                xs.append(gx);  ys.append(gy)
            break
    return np.array(xs), np.array(ys), state["chars"]

def plot_learning(all_returns: dict):
    plt.figure(figsize=(6, 3))
    for eps, rets in all_returns.items():
        avg = cumavg(np.array(rets))
        plt.plot(avg, lw=2, label=f"ε={eps}")
    plt.title("Cumulative-average return — cliff room")
    plt.xlabel("Episode");  plt.ylabel("Average return")
    plt.legend();  plt.grid(alpha=.3);  plt.tight_layout()

def plot_multi_paths(trajs: dict, *, x_lo, x_hi, y_lo, y_hi):
    """Plot multiple trajectories in one window with lava and labels."""
    lava_glyphs = [ord("}"), ord("~"), ord(":")]
    _, ax = plt.subplots(figsize=(5, 3.5))

    # grid
    for x in range(x_lo, x_hi + 1):
        ax.axvline(x, color="0.85", lw=.6, zorder=0)
    for y in range(y_lo, y_hi + 1):
        ax.axhline(y, color="0.85", lw=.6, zorder=0)

    # lava
    # use the last chars from any trajectory
    chars = next(iter(trajs.values()))[2]
    lava_cells = np.argwhere(np.isin(chars, lava_glyphs))
    for ly, lx in lava_cells:
        if x_lo <= lx <= x_hi and y_lo <= ly <= y_hi:
            ax.add_patch(mpatches.Rectangle(
                (lx, ly), 1, 1,
                facecolor="orange", alpha=.4, zorder=1
            ))

    # each trajectory
    for (eps, (xs, ys, _)), color in zip(trajs.items(), TRAJ_COLORS):
        m = (x_lo <= xs) & (xs <= x_hi) & (y_lo <= ys) & (ys <= y_hi)
        ax.plot(xs[m] + .5, ys[m] + .5, "-", lw=1.5,
                color=color, alpha=0.8, label=f"ε={eps}", zorder=3)
        ax.scatter(xs[m] + .5, ys[m] + .5, s=20,
                   color=color, alpha=0.8, zorder=4)


    # ticks centred & 1-based
    xt = np.arange(x_lo, x_hi + 1)
    yt = np.arange(y_lo, y_hi + 1)
    ax.set_xticks(xt + .5);  ax.set_xticklabels(xt + 1)
    ax.set_yticks(yt + .5);  ax.set_yticklabels(yt + 1)

    ax.set_xlim(x_lo,   x_hi + 1)
    ax.set_ylim(y_hi + 1, y_lo)
    ax.set_aspect("equal")
    ax.set_xlabel("X Coordinate");  ax.set_ylabel("Y Coordinate")
    ax.set_title("Greedy trajectories after training")
    ax.legend(loc="lower center", fontsize=8, bbox_to_anchor=(0.5, 0.0), ncol=2)
    plt.tight_layout()

# ---------- main routine ----------
def main():
    env = get_minihack_envirnment(CLIFF, add_pixel=False)
    all_returns = {}
    all_trajs   = {}

    for eps in EPSILON_VALUES:
        agent = SARSAOnPolicy(
            id           = f"SARSA–ε{eps}",
            action_space = env.action_space,
            epsilon      = eps,
            gamma        = GAMMA,
            alpha        = ALPHA
        )
        task = RLTask(env, agent)

        print(f"Training SARSA ε={eps} for {N_EPISODES} episodes …")
        returns = task.interact(N_EPISODES)
        all_returns[eps] = returns

        # greedy rollout
        agent.epsilon = 0.0
        xs, ys, chars = greedy_rollout(agent, env, MAX_STEPS_ROLLOUT)
        all_trajs[eps] = (xs, ys, chars)

    # plot returns
    plot_learning(all_returns)

    # plot all paths
    plot_multi_paths(all_trajs, x_lo=31, x_hi=45, y_lo=8, y_hi=12)

    plt.show()
    env.close()

if __name__ == "__main__":
    main()
