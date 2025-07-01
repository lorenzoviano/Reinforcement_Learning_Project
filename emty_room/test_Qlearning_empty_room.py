#!/usr/bin/env python
"""
test_Qlearning.py

Runs on-policy TD control (Qlearning) on the 5×5 EMPTY_ROOM.
Prints a running-average learning curve every PRINT_EVERY episodes and
then does a final greedy rollout.
"""

import numpy as np
import matplotlib.pyplot as plt

from minihack_env import EMPTY_ROOM, get_minihack_envirnment
from agents import QLearningOffPolicy
from rl_task import RLTask

# ---- hyper-parameters ----
N_EPISODES  = 1000       # total training episodes
PRINT_EVERY =  250       # log every this many episodes
EPSILON     =  0.1       # exploration rate (ε-greedy)
GAMMA       =  1.0       # discount factor γ
ALPHA       =  0.1       # constant step-size α

# ---- setup env & agent ----
env = get_minihack_envirnment(EMPTY_ROOM, add_pixel=False)

agent = QLearningOffPolicy(
    id="QLearning",
    action_space=env.action_space,
    epsilon=EPSILON,
    gamma=GAMMA,
    alpha=ALPHA
)

task = RLTask(env, agent)

# ---- training loop & logging ----
all_returns   = []
episodes_done = 0

while episodes_done < N_EPISODES:
    batch_returns = task.interact(PRINT_EVERY)
    all_returns.extend(batch_returns)
    episodes_done += PRINT_EVERY

    avg_batch = np.mean(batch_returns)
    print(f"[ep {episodes_done:5d}]  "
          f"avg return (last {PRINT_EVERY} eps) = {avg_batch:6.2f}")

# ---- cumulative average curve ----
returns    = np.array(all_returns)
cum_avg    = np.cumsum(returns) / np.arange(1, len(returns) + 1)
episodes   = np.arange(1, len(cum_avg) + 1)

plt.figure(figsize=(8,5))
plt.plot(episodes, cum_avg,
         label=f"Qlearning (α={ALPHA:.2f}, ε={EPSILON:.2f})")
plt.title('Qlearning on-policy TD Control: Cumulative Average Return')
plt.xlabel('Episode')
plt.ylabel('Average Return')
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

# ---- greedy rollout ----
agent.epsilon = 0.0
obs, _ = env.reset()
G, done, steps = 0.0, False, 0

while not done and steps < 100:
    a = agent.act(obs)
    obs, r, term, trunc, _ = env.step(a)
    G += r
    done = term or trunc
    steps += 1

print(f"\nGreedy run → steps = {steps}, return = {G:.2f}")
env.render()
env.close()
