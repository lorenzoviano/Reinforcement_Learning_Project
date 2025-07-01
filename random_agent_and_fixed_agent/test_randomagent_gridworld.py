import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

import gridworld_env            
from agents import RandomAgent
from rl_task import RLTask

# 5×5 GridWorld
env = gym.make("GridWorld-v0", shape=(5, 5), render_mode="human")
agent = RandomAgent("rand", env.action_space)

task = RLTask(env, agent)

# -------- 10 000 episodes --------
avg_returns = task.interact(n_episodes=10_000)

plt.plot(avg_returns)
plt.xlabel("Episode k")
plt.ylabel(r"Average return $\hat{G}_k$")
plt.title("RandomAgent on 5×5 GridWorld")
plt.grid(False)
plt.show()

# -------- visualise first 10 steps of a fresh episode --------
print("\n--- visualising first 10 timesteps of a new episode ---\n")
task.visualize_episode(max_number_steps=10)
env.close()



# 1) make your env and agent
env = gym.make("GridWorld-v0", shape=(5, 5))
agent = RandomAgent("rand", env.action_space)

# 2) roll out the first 10 steps by hand
state, _ = env.reset()
records = []
for _ in range(10):
    action = agent.act(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    records.append((state, action))
    state = next_state
    if done:
        break


# 3) helper to draw one frame
import numpy as np
import matplotlib.pyplot as plt

def draw_step(ax, state, action, grid_shape=(5,5)):
    n, m = grid_shape
    r, c = state
    x = c + 0.5
    y = (n - 1 - r) + 0.5

    # draw a clean grid of n×m
    ax.set_xticks(np.arange(0, m+1, 1))
    ax.set_yticks(np.arange(0, n+1, 1))
    ax.grid(True, linewidth=1)
    # keep the frame if you like
    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.set_xlim(0, m)
    ax.set_ylim(0, n)

    # goal square
    goal = plt.Rectangle((m-1, 0), 1, 1, color="red", alpha=0.6)
    ax.add_patch(goal)

    # agent circle
    circ = plt.Circle((x, y), 0.3, color="blue")
    ax.add_patch(circ)

    # action arrow
    deltas = {
        0: ( 0, +1),   # UP    → arrow pointing upward
        1: ( 0, -1),   # DOWN  → arrow pointing downward
        2: (-1,  0),   # LEFT  → arrow pointing left
        3: (+1,  0),   # RIGHT → arrow pointing right
        }   
    dx, dy = deltas[action]
    arr = plt.Arrow(x, y, dx*0.5, dy*0.5, width=0.15, color="red")
    ax.add_patch(arr)





# 4) plot all 10 in a 2×5 grid
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, (state, action) in zip(axes.flatten(), records):
    draw_step(ax, state, action, grid_shape=(5,5))

plt.tight_layout()
plt.suptitle("RandomAgent: first 10 steps", y=1.02)
plt.show()
env.close()