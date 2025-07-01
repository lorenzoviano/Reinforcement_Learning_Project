#!/usr/bin/env python
"""
test_compare_learning_curves.py

Compare learning curves (episode return vs. episode) for:
  • First‐visit MC (sample‐average, no α)
  • SARSA (on‐policy TD, uses α)
  • Q‐learning (off‐policy TD, uses α)

in the 5×5 EMPTY_ROOM, averaged over 10 seeds with variance shading.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from minihack_env import EMPTY_ROOM, get_minihack_envirnment
from agents import MonteCarloOnPolicy, SARSAOnPolicy, QLearningOffPolicy
from rl_task import RLTask

# ─── hyper‐parameters ────────────────────────────────────────────────
N_EPISODES = 500    # episodes per run
EPSILON    = 0.9    # ε‐greedy exploration
GAMMA      = 1.0    # discount factor
ALPHA      = 0.1    # step‐size for SARSA & Q‐learning
SEEDS      = list(range(10))
# ─────────────────────────────────────────────────────────────────────

def run_mc(seed):
    random.seed(seed); np.random.seed(seed)
    env   = get_minihack_envirnment(EMPTY_ROOM, add_pixel=False)
    agent = MonteCarloOnPolicy("MC", env.action_space,
                               epsilon=EPSILON, gamma=GAMMA)
    returns = RLTask(env, agent).interact(N_EPISODES)
    env.close()
    return np.asarray(returns, float)

def run_sarsa(seed):
    random.seed(seed); np.random.seed(seed)
    env   = get_minihack_envirnment(EMPTY_ROOM, add_pixel=False)
    agent = SARSAOnPolicy("SARSA", env.action_space,
                          epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA)
    returns = RLTask(env, agent).interact(N_EPISODES)
    env.close()
    return np.asarray(returns, float)

def run_qlearning(seed):
    random.seed(seed); np.random.seed(seed)
    env   = get_minihack_envirnment(EMPTY_ROOM, add_pixel=False)
    agent = QLearningOffPolicy("QLearn", env.action_space,
                               epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA)
    returns = RLTask(env, agent).interact(N_EPISODES)
    env.close()
    return np.asarray(returns, float)

def cumulative_avg(all_returns):
    """Compute per‐seed cumulative average, then return array shape (n_seeds, N_EPISODES)."""
    # all_returns: shape (n_seeds, N_EPISODES)
    cums = np.cumsum(all_returns, axis=1)
    episodes = np.arange(1, N_EPISODES+1)
    return cums / episodes[np.newaxis, :]

def plot_with_shading(episodes, mean, sem, label, **plt_kwargs):
    """Helper to plot mean curve with ±SEM shading."""
    plt.plot(episodes, mean, **plt_kwargs, label=label)
    plt.fill_between(episodes,
                     mean - sem,
                     mean + sem,
                     alpha=0.2,
                     **{k:plt_kwargs[k] for k in ("color",) if k in plt_kwargs})

def main():
    # 1) Collect returns for each seed & agent
    mc_runs    = np.stack([run_mc(sd)    for sd in SEEDS])
    sarsa_runs = np.stack([run_sarsa(sd) for sd in SEEDS])
    ql_runs    = np.stack([run_qlearning(sd) for sd in SEEDS])

    # 2) Compute cumulative averages per seed
    mc_cum    = cumulative_avg(mc_runs)
    sarsa_cum = cumulative_avg(sarsa_runs)
    ql_cum    = cumulative_avg(ql_runs)

    # 3) Compute across‐seed mean & SEM
    def stats(cum_arr):
        mean = cum_arr.mean(axis=0)
        sem  = cum_arr.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
        return mean, sem

    mc_mean, mc_sem       = stats(mc_cum)
    sarsa_mean, sarsa_sem = stats(sarsa_cum)
    ql_mean, ql_sem       = stats(ql_cum)

    episodes = np.arange(1, N_EPISODES+1)

    # 4) Plot
    plt.figure(figsize=(10,6))
    plot_with_shading(episodes, mc_mean,    mc_sem,
                      "First‐Visit MC",    color="tab:blue")
    plot_with_shading(episodes, sarsa_mean, sarsa_sem,
                      f"SARSA (α={ALPHA:.2f})", color="tab:orange")
    plot_with_shading(episodes, ql_mean,    ql_sem,
                      f"Q‐Learning (α={ALPHA:.2f})", color="tab:green")

    plt.xlabel("Episode")
    plt.ylabel(r"Cumulative average return $\hat G_k$")
    plt.title(
      f"Empty‐Room Learning Curves\n"
      f"N_EPISODES={N_EPISODES}, ε={EPSILON}, γ={GAMMA}, α={ALPHA}"
    )
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
