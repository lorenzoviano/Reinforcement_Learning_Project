# Reinforcement Learning for Grid-World and MiniHack Environments

## Overview

This project explores a suite of reinforcement learning (RL) algorithms applied to grid-world and MiniHack navigation environments. Both classic tabular RL methods (Monte Carlo, SARSA, Q-learning, Dyna-Q) and deep RL agents (DQN, PPO) are implemented and evaluated across environments of increasing complexity and stochasticity.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Implemented Algorithms & Tasks](#implemented-algorithms--tasks)
- [Environments](#environments)
- [Results & Analysis](#results--analysis)
- [Installation & Usage](#installation--usage)
- [References](#references)
- [Acknowledgements](#acknowledgements)

---

<br>


## Project Structure

| File / Folder                             | Description |
|-------------------------------------------|-------------|
| `Reinforcement_Learning_Assignment.pdf` | Assignment instructions |
| `Final_report.pdf`                       | Detailed report on experiments and findings of the project|
| `agents.py`                              | All RL agents (Random, Fixed, MC, SARSA, QL, DynaQ, etc.) |
| `rl_task.py`                             | RLTask: orchestrates agent-environment interaction and evaluation |
| `commons.py`                             | Common utility functions and abstract base classes |
| `gridword_env.py`                        | Custom Gym-compatible GridWorld environment |
| `minihack_env.py`                        | MiniHack wrappers and utilities |
| `random_agent_and_fixed_agent/`          | Baseline Random and Fixed agent tests (empty/lava rooms) |
| `empty_room/`                            | MC, Q-learning, SARSA in Empty Room performance tests|
| `cliff/`                                 | MC, Q-learning, SARSA in Cliff performance tests |
| `lava_room/`                             | MC, Q-learning, SARSA in Lava Room performance tests |
| `room_with_monster/`                     | MC, Q-learning, SARSA in Room With Monster performance tests|
| `DynaQ/`                                 | DynaQ vs Q-learning experiments in Cliff and Room with Lava environments|
| `deep_reinforcement_learning/`           | PPO and DQN (deep RL) experiments (Empty Room & Monsters) |


<br>


## Implemented Algorithms & Tasks

### 1. Baseline Agents & Environment Exploration
- **Custom GridWorld environment** (variable size, Gym interface)
- **RandomAgent**: Selects actions uniformly at random
- **FixedAgent**: Moves down until blocked, then right
- **RLTask**: Manages episodic agent-environment interactions and returns logging

### 2. Learning Algorithms (Tabular)
- **Monte Carlo (MC) On-Policy**
- **SARSA (On-Policy TD)**
- **Q-Learning (Off-Policy TD)**
- **Dyna-Q**: Q-learning with model-based planning
- **Epsilon Scheduling**: Fixed and linearly decaying ε-greedy exploration

### 3. Deep Reinforcement Learning
- **DQN**: Deep Q-Network (using Stable-Baselines3)
- **PPO**: Proximal Policy Optimization (using Stable-Baselines3)
- Custom state preprocessing (e.g., egocentric 3x3 crop for MiniHack)

---

<br>


## Environments

- **Empty Room**: Classic grid-world, deterministic, no obstacles
- **Cliff**: Grid-world with hazardous cliff cells (large negative reward, resets to start)
- **Room With Lava**: Complex map, obstacles and hazardous lava
- **Room With Monster**: Like Empty Room, but a moving monster introduces stochasticity and risk
- **Room With Multiple Monsters**: High-complexity, multiple moving monsters (deep RL only)

---

<br>



## Results & Analysis

Key findings include:
- **Tabular Q-learning** converges faster than deep RL (DQN/PPO) in small, simple environments.
- **Deep RL methods** (PPO, DQN) outperform tabular RL in highly stochastic or high-dimensional environments.
- **Q-learning** finds risky, optimal paths; **SARSA** learns safer, conservative strategies; **Dyna-Q** accelerates learning in deterministic settings but can struggle with stochasticity.
- See [Final_report.pdf](./Final_report.pdf) for detailed results, plots, and discussion.


<br>


## Installation & Usage
### Requirements

- Python 3.8+
- `gymnasium`
- `minihack`
- `stable-baselines3`
- Standard scientific libraries (`numpy`, `matplotlib`, etc.)

Install dependencies:
```bash
pip install gymnasium minihack stable-baselines3 numpy matplotlib
```

<br>


## Conclusion

This project demonstrates the strengths and limitations of both tabular and deep reinforcement learning algorithms across a diverse set of navigation tasks. While tabular methods like Q-learning and SARSA are highly efficient in small, deterministic environments, deep RL methods (DQN, PPO) become essential as complexity and stochasticity increase. The findings offer insights for practitioners selecting RL algorithms for environments of varying complexity. Please refer to the Final Report file for in depth analysis of the experiments performed. Future directions could include experimenting with additional deep RL algorithms, implementing prioritized experience replay, or exploring transfer learning across tasks.
