#!/usr/bin/env python
"""
compare_cliff.py
────────────────
Compare constant-ε and linearly-decaying-ε variants of
  • Monte-Carlo
  • SARSA
  • Q-learning
on the MiniHack Cliff environment.

Each curve shows a 50-episode sliding-window average of episode returns,
plotted as mean ± SEM over 10 seeds, with progress printed to console.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from minihack_env import CLIFF, get_minihack_envirnment
from agents import (
    MonteCarloOnPolicy, MonteCarloOnPolicyDecay,
    SARSAOnPolicy,      SARSAOnPolicyDecay,
    QLearningOffPolicy, QLearningOffPolicyDecay
)
from rl_task import RLTask

# ─── experiment hyper-parameters ───────────────────────────────────────────────
N_EPISODES       = 3000          # total training episodes
WINDOW_EPISODES  = 50            # length of the sliding window
CONST_EPSILONS   = (0.3, 0.1)    # two fixed-ε baselines
DECAY_EPS_START  = 0.3           # ε₀ for decaying agents
DECAY_EPS_END    = 0.1           # ε_T after N_EPISODES
GAMMA            = 1.0
ALPHA            = 0.1
SEEDS            = list(range(10))  # ← run each variant with these 10 seeds
# ────────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def run_agent(agent_cls, seed: int, **agent_kwargs):
    """Run one agent for N_EPISODES and return the per-episode returns."""
    set_seed(seed)
    env   = get_minihack_envirnment(CLIFF, add_pixel=False)
    agent = agent_cls(action_space=env.action_space, **agent_kwargs)
    task  = RLTask(env, agent)
    returns = task.interact(N_EPISODES)              # length == N_EPISODES
    env.close()
    return np.asarray(returns, dtype=float)

def sliding_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Return the simple moving average of *data* (len = len(data)-window+1)."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")

def compute_stats(all_slid: np.ndarray):
    """
    all_slid: shape (n_seeds, n_points)
    returns (mean, sem) both shape (n_points,)
    """
    mean = all_slid.mean(axis=0)
    sem  = all_slid.std(axis=0, ddof=1) / np.sqrt(all_slid.shape[0])
    return mean, sem

# ─── collect mean & SEM over seeds for each family ─────────────────────────────
mc_means, mc_sems,    = [], []
sarsa_means, sarsa_sems = [], []
ql_means, ql_sems      = [], []

labels_mc    = []
labels_sarsa = []
labels_ql    = []

colors = ["tab:blue", "tab:cyan", "tab:purple"]      # for readability

# --- Monte-Carlo constant ε ---
for eps in CONST_EPSILONS:
    all_slid = []
    for seed in SEEDS:
        print(f"▶ MC const ε={eps:.2f}  — seed {seed}")
        returns = run_agent(MonteCarloOnPolicy,
                            seed,
                            id=f"MC-ε{eps:.2f}",
                            epsilon=eps, gamma=GAMMA)
        slid = sliding_mean(returns, WINDOW_EPISODES)
        all_slid.append(slid)
    all_slid = np.stack(all_slid)
    m, s = compute_stats(all_slid)
    mc_means.append(m); mc_sems.append(s)
    labels_mc.append(f"MC const ε={eps:.2f}")

# --- Monte-Carlo linear decay ---
all_slid = []
for seed in SEEDS:
    print(f"▶ MC decay ({DECAY_EPS_START:.2f}→{DECAY_EPS_END:.2f}) — seed {seed}")
    returns = run_agent(MonteCarloOnPolicyDecay,
                        seed,
                        id="MC-decay",
                        epsilon_start=DECAY_EPS_START,
                        epsilon_end=DECAY_EPS_END,
                        max_episodes=N_EPISODES,
                        gamma=GAMMA)
    all_slid.append(sliding_mean(returns, WINDOW_EPISODES))
all_slid = np.stack(all_slid)
m, s = compute_stats(all_slid)
mc_means.append(m); mc_sems.append(s)
labels_mc.append(f"MC decay {DECAY_EPS_START:.2f}→{DECAY_EPS_END:.2f}")

# --- SARSA constant ε & decay ---
for eps in CONST_EPSILONS:
    all_slid = []
    for seed in SEEDS:
        print(f"▶ SARSA const ε={eps:.2f} — seed {seed}")
        returns = run_agent(SARSAOnPolicy,
                            seed,
                            id=f"SARSA-ε{eps:.2f}",
                            epsilon=eps, gamma=GAMMA, alpha=ALPHA)
        all_slid.append(sliding_mean(returns, WINDOW_EPISODES))
    all_slid = np.stack(all_slid)
    m, s = compute_stats(all_slid)
    sarsa_means.append(m); sarsa_sems.append(s)
    labels_sarsa.append(f"SARSA const ε={eps:.2f}")

# SARSA decay
all_slid = []
for seed in SEEDS:
    print(f"▶ SARSA decay ({DECAY_EPS_START:.2f}→{DECAY_EPS_END:.2f}) — seed {seed}")
    returns = run_agent(SARSAOnPolicyDecay,
                        seed,
                        id="SARSA-decay",
                        epsilon_start=DECAY_EPS_START,
                        epsilon_end=DECAY_EPS_END,
                        max_episodes=N_EPISODES,
                        gamma=GAMMA, alpha=ALPHA)
    all_slid.append(sliding_mean(returns, WINDOW_EPISODES))
all_slid = np.stack(all_slid)
m, s = compute_stats(all_slid)
sarsa_means.append(m); sarsa_sems.append(s)
labels_sarsa.append(f"SARSA decay {DECAY_EPS_START:.2f}→{DECAY_EPS_END:.2f}")

# --- Q-learning constant ε & decay ---
for eps in CONST_EPSILONS:
    all_slid = []
    for seed in SEEDS:
        print(f"▶ QL const ε={eps:.2f}     — seed {seed}")
        returns = run_agent(QLearningOffPolicy,
                            seed,
                            id=f"QL-ε{eps:.2f}",
                            epsilon=eps, gamma=GAMMA, alpha=ALPHA)
        all_slid.append(sliding_mean(returns, WINDOW_EPISODES))
    all_slid = np.stack(all_slid)
    m, s = compute_stats(all_slid)
    ql_means.append(m); ql_sems.append(s)
    labels_ql.append(f"QL const ε={eps:.2f}")

# Q-learning decay
all_slid = []
for seed in SEEDS:
    print(f"▶ QL decay ({DECAY_EPS_START:.2f}→{DECAY_EPS_END:.2f}) — seed {seed}")
    returns = run_agent(QLearningOffPolicyDecay,
                        seed,
                        id="QL-decay",
                        epsilon_start=DECAY_EPS_START,
                        epsilon_end=DECAY_EPS_END,
                        max_episodes=N_EPISODES,
                        gamma=GAMMA, alpha=ALPHA)
    all_slid.append(sliding_mean(returns, WINDOW_EPISODES))
all_slid = np.stack(all_slid)
m, s = compute_stats(all_slid)
ql_means.append(m); ql_sems.append(s)
labels_ql.append(f"QL decay {DECAY_EPS_START:.2f}→{DECAY_EPS_END:.2f}")

# ─── plotting helper ───────────────────────────────────────────────────────────
def plot_family_with_sem(ax, means, sems, labels, title):
    episodes = np.arange(WINDOW_EPISODES, N_EPISODES + 1)
    for mean, sem, lbl, col in zip(means, sems, labels, colors):
        ax.plot(episodes, mean, label=lbl, linewidth=1.3, color=col)
        ax.fill_between(episodes,
                        mean - sem,
                        mean + sem,
                        color=col, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"{WINDOW_EPISODES}-ep window avg return")
    ax.legend()
    ax.grid(False)

# ─── build one figure per agent family ─────────────────────────────────────────
families = [
    ("Monte-Carlo on Cliff (γ=1)", mc_means,    mc_sems,    labels_mc),
    ("SARSA on Cliff (α=0.1, γ=1)",   sarsa_means, sarsa_sems, labels_sarsa),
    ("Q-learning on Cliff (α=0.1, γ=1)", ql_means,    ql_sems,    labels_ql)
]

episodes = np.arange(WINDOW_EPISODES, N_EPISODES + 1)

for title, means, sems, labels in families:
    plt.figure(figsize=(8,5))
    for mean, sem, lbl, col in zip(means, sems, labels, colors):
        plt.plot(episodes, mean,    label=lbl, linewidth=1.3, color=col)
        plt.fill_between(episodes,
                         mean - sem,
                         mean + sem,
                         color=col, alpha=0.2)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(f"{WINDOW_EPISODES}-ep window avg  $G$")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    # If you want to save each:
    # safe_fname = title.lower().split()[0] + "_cliff.png"
    # plt.savefig(safe_fname, dpi=150)
    # print(f"Saved {safe_fname}")
    plt.show()

