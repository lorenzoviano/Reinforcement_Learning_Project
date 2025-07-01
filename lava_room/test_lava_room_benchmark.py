#!/usr/bin/env python
"""
lava_room_benchmark.py
----------------------

Compares constant-ε agents with their linearly-decaying-ε counterparts on
MiniHack’s ROOM_WITH_LAVA.

Outputs
-------
1.  Table (mean ± SEM over 10 seeds) for
       – average *training* return (episodes 100-2000)
       – greedy-evaluation return (single ε=0 episode after training)
2.  Two learning-curve figures
"""

# ──────────────── standard libs ────────────────
import random, collections, copy, math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# ───────────── project helpers ─────────────
from minihack_env import ROOM_WITH_LAVA, get_minihack_envirnment
from agents import (MonteCarloOnPolicyDecay, SARSAOnPolicy, SARSAOnPolicyDecay,
                    QLearningOffPolicy, QLearningOffPolicyDecay)

# ───────────── hyper-parameters ─────────────
N_EPISODES   = 2000
SEEDS        = list(range(10))
GAMMA        = 1
MAX_STEPS_EP = 400

AGENTS = {
    "MC-decay (0.3→0)" : (MonteCarloOnPolicyDecay,
                           dict(epsilon_start=0.3, epsilon_end=0,
                                max_episodes=N_EPISODES, gamma=GAMMA),
                           "tab:blue"),

    "SARSA-const (ε=0.10)" : (SARSAOnPolicy,
                              dict(epsilon=0.10, gamma=GAMMA, alpha=0.10),
                              "gold"),

    "SARSA-decay (0.6→0)" : (SARSAOnPolicyDecay,
                              dict(epsilon_start=0.6, epsilon_end=0,
                                   max_episodes=N_EPISODES,
                                   gamma=GAMMA, alpha=0.10),
                              "tab:orange"),

    "Q-const (ε=0.20)" : (QLearningOffPolicy,
                          dict(epsilon=0.20, gamma=GAMMA, alpha=0.10),
                          "tab:green"),
    "Q-decay (0.3→0)" : (QLearningOffPolicyDecay,
                          dict(epsilon_start=0.3, epsilon_end=0,
                               max_episodes=N_EPISODES,
                               gamma=GAMMA, alpha=0.10),
                          "lime"),
    "Q-decay (0.6→0)" : (QLearningOffPolicyDecay,
                          dict(epsilon_start=0.6, epsilon_end=0,
                               max_episodes=N_EPISODES,
                               gamma=GAMMA, alpha=0.10),
                          "brown"),
}

# ───────────── helper utilities ─────────────
MetricRec = collections.namedtuple("MetricRec", "returns greedy")

def set_seed(s): random.seed(s); np.random.seed(s)

def run_one(agent_cls, kwargs, seed):
    """
    Train ONE agent; return its per-episode returns *and*
    the reward obtained in one greedy (ε=0) episode after training.
    """
    set_seed(seed)
    env = get_minihack_envirnment(ROOM_WITH_LAVA, add_pixel=False)
    env.reset(seed=seed)

    agent = agent_cls(id=f"{agent_cls.__name__}-{seed}",
                      action_space=env.action_space, **kwargs)

    returns = []

    # -------- training --------
    for _ in range(N_EPISODES):
        obs, _ = env.reset()
        done, G, steps, prev_r = False, 0.0, 0, 0.0
        ep_log = []

        while not done and steps < MAX_STEPS_EP:
            obs_cp = copy.deepcopy(obs)
            act    = agent.act(obs, prev_r)
            obs, r, term, trunc, _ = env.step(act)
            ep_log.append((obs_cp, act, r))
            G, steps, done, prev_r = G + r, steps + 1, (term or trunc), r

        if hasattr(agent, "onEpisodeEnd"):
            agent.onEpisodeEnd(episode=ep_log, reward=G,
                               episode_number=0, last_step_reward=prev_r)
        returns.append(G)

    # -------- greedy evaluation (ε = 0) --------
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    obs, _ = env.reset()
    done, G_eval, steps = False, 0.0, 0
    while not done and steps < MAX_STEPS_EP:
        act = agent.act(obs, 0.0)
        obs, r, term, trunc, _ = env.step(act)
        G_eval += r
        done = term or trunc
        steps += 1

    env.close()
    return MetricRec(np.asarray(returns, float), G_eval)

# ───────────── run loop ─────────────
results = collections.defaultdict(list)
print("Running benchmarks …")
for name, (cls, kw, _) in AGENTS.items():
    print(f"  > {name}")
    for sd in SEEDS:
        results[name].append(run_one(cls, kw, sd))

# ───────────── summary table ─────────────
table_rows = []
idx = slice(100, N_EPISODES)         # evaluate training return from ep 100+

for name, recs in results.items():
    rets   = np.stack([r.returns for r in recs])
    g_eval = np.array([r.greedy  for r in recs])

    train_mean = rets[:, idx].mean(axis=1).mean()
    train_sem  = rets[:, idx].mean(axis=1).std(ddof=1) / math.sqrt(len(SEEDS))

    greedy_mean = g_eval.mean()
    greedy_sem  = g_eval.std(ddof=1) / math.sqrt(len(SEEDS))

    table_rows.append([name,
                       f"{train_mean:6.1f} ± {train_sem:.1f}",
                       f"{greedy_mean:6.1f} ± {greedy_sem:.1f}"])

print("\n=== ROOM_WITH_LAVA  (Ep 100-2000)  — 10 seeds ===")
print(tabulate(table_rows,
               headers=["Agent", "Avg Return", "Greedy Return"],
               tablefmt="github"))

# ───────────── cumulative curve ─────────────
def cumavg(v): return np.cumsum(v) / np.arange(1, len(v)+1)
episodes = np.arange(100, N_EPISODES)

plt.figure(figsize=(8,4.5))
for name, (_, _, colour) in AGENTS.items():
    mean = np.stack([r.returns for r in results[name]]).mean(axis=0)[100:]
    plt.plot(episodes, cumavg(mean), lw=1.8, color=colour, label=name)
plt.title("ROOM_WITH_LAVA — cumulative-avg return (100-2000)")
plt.xlabel("Episode"); plt.ylabel("Average return")
plt.grid(alpha=.3); plt.legend(fontsize=8, ncol=2); plt.tight_layout(); plt.show()

# ───────────── rolling-window curve ─────────────
def moving_avg(v, win=50):
    if len(v) < win: return np.full_like(v, np.nan)
    c = np.cumsum(np.insert(v,0,0.0))
    return (c[win:] - c[:-win]) / win
WINDOW = 50
episodes_ma = np.arange(100+WINDOW-1, N_EPISODES)

plt.figure(figsize=(8,4.5))
for name, (_, _, colour) in AGENTS.items():
    mean = np.stack([r.returns for r in results[name]]).mean(axis=0)[100:]
    plt.plot(episodes_ma, moving_avg(mean, WINDOW),
             lw=1.8, color=colour, label=name)
plt.title(f"ROOM_WITH_LAVA — {WINDOW}-episode moving avg return")
plt.xlabel("Episode"); plt.ylabel("Return (window mean)")
plt.grid(alpha=.3); plt.legend(fontsize=8, ncol=2); plt.tight_layout(); plt.show()
