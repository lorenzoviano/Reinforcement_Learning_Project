#!/usr/bin/env python
"""
test_montecarlo_epsilon.py

Runs first-visit, constant-α Monte-Carlo control with different exploration rates
and estimates how often the optimal policy (shortest-path, return = −8 in 5×5
EMPTY_ROOM) is recovered.

For each ε in {0.1, 0.3, 0.5, 0.8, 0.9} we:
    * train 100 independent agents for N_EPISODES episodes,
    * evaluate each agent greedily,
    * count successes.
"""

import numpy as np
import matplotlib.pyplot as plt
from minihack_env import EMPTY_ROOM, get_minihack_envirnment
from agents import MonteCarloOnPolicy
from rl_task import RLTask
from tqdm import trange

# ------------- hyper-parameters -------------
N_EPISODES    = 500       # training episodes per run
N_TRIALS      = 100        # independent trainings per ε
GAMMA         = 1.0
EPSILONS      = [0.01, 0.1, 0.3, 0.5, 0.8, 0.9] # exploration rates

# ------------- evaluation setup -------------
TARGET_RETURN = -8         # optimal path in 5×5 empty room
MAX_STEPS_EVAL = 100       # safety cap for greedy rollout

def train_once(epsilon: float, seed: int | None = None) -> bool:
    """Train a single agent with given ε and report whether it learns optimally."""
    env = get_minihack_envirnment(EMPTY_ROOM, add_pixel=False)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        np.random.seed(seed)

    # pass alpha into the agent here
    agent = MonteCarloOnPolicy(
        id="MC",
        action_space=env.action_space,
        epsilon=epsilon,
        gamma=GAMMA,
    )
    task = RLTask(env, agent)

    # --- training ---
    task.interact(N_EPISODES)

    # --- greedy evaluation ---
    agent.epsilon = 0.0
    obs, _ = env.reset()
    g_return, steps, done = 0.0, 0, False
    while not done and steps < MAX_STEPS_EVAL:
        a = agent.act(obs)
        obs, r, term, trunc, _ = env.step(a)
        g_return += r
        done = term or trunc
        steps += 1

    env.close()
    return g_return == TARGET_RETURN

def main():
    results = {}
    for eps in EPSILONS:
        success_count = 0

        for run in trange(
            N_TRIALS,
            desc=f"ε={eps:.2f}",
            unit="run",
            leave=True
        ):
            if train_once(eps, seed=run):
                success_count += 1

        results[eps] = success_count
        print(f"ε={eps:.1f} → optimal in {success_count}/{N_TRIALS}")

    # --- visualize aggregate ---
    eps_values = list(results.keys())
    successes  = [results[e] for e in eps_values]

    plt.figure(figsize=(8,5))
    plt.bar([str(e) for e in eps_values], successes)
    plt.ylabel("Optimal policies found (out of 100)")
    plt.xlabel("ε-greedy exploration rate (ε)")
    plt.title(f"First-Visit MC (γ={GAMMA}) on EMPTY_ROOM\n"
              f"N_EPISODES={N_EPISODES}, N_TRIALS={N_TRIALS}")
    plt.ylim(0, N_TRIALS)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
