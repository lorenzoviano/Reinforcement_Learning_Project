#!/usr/bin/env python
"""
test_montecarlo.py

Runs first-visit, constant-α Monte-Carlo control on the 5×5 EMPTY_ROOM.
Prints a running-average learning curve every PRINT_EVERY episodes and
then does a final greedy rollout.
"""

import numpy as np
import matplotlib.pyplot as plt

from minihack_env import EMPTY_ROOM, get_minihack_envirnment
from agents import MonteCarloOnPolicy
from rl_task import RLTask

# ---- hyper-params ----
N_EPISODES  = 500      # total training episodes
PRINT_EVERY =  250      # log every this many episodes
EPSILON     =  0.1      # exploration rate
GAMMA       =  1.0      # discount factor
# ---- setup ----
env   = get_minihack_envirnment(EMPTY_ROOM, add_pixel=False)
agent = MonteCarloOnPolicy(
    id="MC",
    action_space=env.action_space,
    epsilon=EPSILON,
    gamma=GAMMA,
)
task  = RLTask(env, agent)

# ---- train & collect ----
running_returns = []
episodes_seen   = 0

while episodes_seen < N_EPISODES:
    # run PRINT_EVERY episodes, get their returns
    batch = task.interact(PRINT_EVERY)
    running_returns.extend(batch)
    episodes_seen += PRINT_EVERY

    avg_return = np.mean(batch)
    print(f"[ep {episodes_seen:5d}]  "
          f"avg return over last {PRINT_EVERY} eps = {avg_return:6.2f}")

# ---- cumulative average per episode ----
returns     = np.array(running_returns)
avg_returns = np.cumsum(returns) / np.arange(1, len(returns) + 1)
episodes    = np.arange(1, len(avg_returns) + 1)

# ---- plot learning curve ----
plt.figure(figsize=(8, 5))
plt.plot(episodes, avg_returns, label='MC (ε={:.2f})'.format(EPSILON))
plt.title('First-Visit MC: Cumulative Average Return')
plt.xlabel('Episodes')
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
    a, = agent.act(obs), 
    obs, last_r, term, trunc, _ = env.step(a)
    G += last_r
    done = term or trunc
    steps += 1

print(f"\nGreedy run → steps = {steps}, return = {G:.2f}")
env.render()
env.close()
