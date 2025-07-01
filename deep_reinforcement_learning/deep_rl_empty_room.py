#!/usr/bin/env python3
"""
deep_rl_empty_room.py

Train DQN, PPO on MiniHack EMPTY_ROOM with egocentric cropping + MlpPolicy,
and a Q-learning decay agent for 3,000 episodes. Then plot both sliding-window
and cumulative-average learning curves over episodes for all three agents.

python deep_rl_empty_room.py --timesteps 80000 --episodes 3000 --window 100 --crop_size 3

"""

import os
import argparse
import random
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from minihack_env import EMPTY_ROOM, get_minihack_envirnment
from deep_reinforcement_learning.egocentric_crop_wrapper import EgocentricCropWrapper
from agents import QLearningOffPolicyDecay

# ----------------------------------------------------------------------------
# Training functions
# ----------------------------------------------------------------------------
def train_rl_agent(agent_cls, env_id, total_timesteps, crop_size=None, seed=0):
    """
    Train a RL agent (DQN or PPO) on env_id for total_timesteps using egocentric cropping.
    Returns the list of episode returns.
    """
    random.seed(seed)
    np.random.seed(seed)

    raw_env = get_minihack_envirnment(env_id)
    if crop_size is not None:
        env = EgocentricCropWrapper(raw_env, crop_size)
    else:
        env = raw_env
    env = Monitor(env)

    model = agent_cls("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    returns = env.get_episode_rewards()
    return returns


def train_q_learning(env_id, n_episodes, epsilon_start, epsilon_end, gamma, alpha,
                     max_steps=600, seed=0):
    """
    Train a Q-learning off-policy decay agent for n_episodes.
    Returns the list of episode returns.
    """
    random.seed(seed)
    np.random.seed(seed)

    env = get_minihack_envirnment(env_id, add_pixel=False)
    returns = []
    agent = QLearningOffPolicyDecay(
        id=f"QDecay-{seed}",
        action_space=env.action_space,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        max_episodes=n_episodes,
        gamma=gamma,
        alpha=alpha
    )

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed+ep)
        done = False
        total_r = 0.0
        prev_r = 0.0
        steps = 0

        while not done and steps < max_steps:
            obs_cp = copy.deepcopy(obs)
            action = agent.act(obs, prev_r)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            done = term or trunc
            prev_r = r
            steps += 1

        if hasattr(agent, 'onEpisodeEnd'):
            agent.onEpisodeEnd(
                episode=None,
                reward=total_r,
                episode_number=ep,
                last_step_reward=prev_r
            )
        returns.append(total_r)

    env.close()
    return returns

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_learning_curves(data, window, max_x, output_path):
    """
    Plot both sliding-window and cumulative-average learning curves up to max_x episodes.
    data: dict mapping agent name -> list of returns
    window: sliding average window size
    max_x: maximum episode on x-axis
    """
    # 1) sliding-window
    plt.figure(figsize=(10,6))
    for name, rets in data.items():
        arr = np.array(rets)
        eps = np.arange(1, len(arr)+1)
        eps = eps[eps <= max_x]
        arr = arr[:len(eps)]
        if len(arr) >= window:
            ma = pd.Series(arr).rolling(window=window).mean().dropna().values
            eps_ma = eps[window-1:]
            plt.plot(eps_ma, ma, linewidth=2, label=f"{name} (MA{window})")
        else:
            plt.plot(eps, arr, linewidth=1, label=name)
    plt.xlim(1, max_x)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Sliding-Window Learning Curves on EMPTY_ROOM')
    plt.legend()
    plt.grid(False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

    # 2) cumulative-average
    cum_path = output_path.replace('.png', '_cumavg.png')
    plt.figure(figsize=(10,6))
    for name, rets in data.items():
        arr = np.array(rets)
        eps = np.arange(1, len(arr)+1)
        eps = eps[eps <= max_x]
        arr = arr[:len(eps)]
        cum_avg = np.cumsum(arr) / np.arange(1, len(arr)+1)
        plt.plot(eps, cum_avg, linewidth=2, label=f"{name} (CumAvg)")
    plt.xlim(1, max_x)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Average Return')
    plt.title('Cumulative-Average Learning Curves on EMPTY_ROOM')
    plt.legend()
    plt.grid(False)
    plt.savefig(cum_path)
    plt.show()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train DQN, PPO, and Q-learning on EMPTY_ROOM and plot curves'
    )
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Timesteps for DQN/PPO training')
    parser.add_argument('--episodes', type=int, default=10_000,
                        help='Episodes for Q-learning')
    parser.add_argument('--window', type=int, default=100,
                        help='Moving average window')
    parser.add_argument('--max_x', type=int, default=2000,
                        help='Max episode to display on x-axis')
    parser.add_argument('--crop_size', type=int, default=9,
                        help='Size of egocentric crop (Box obs)')
    parser.add_argument('--output', type=str,
                        default='results/empty_room_curves.png',
                        help='Path to save sliding-window plot')
    args = parser.parse_args()

    print("=== Training DQN ===")
    dqn_rets = train_rl_agent(DQN, EMPTY_ROOM, args.timesteps,
                              crop_size=args.crop_size)
    print(f"DQN episodes: {len(dqn_rets)}")

    print("=== Training PPO ===")
    ppo_rets = train_rl_agent(PPO, EMPTY_ROOM, args.timesteps,
                              crop_size=args.crop_size)
    print(f"PPO episodes: {len(ppo_rets)}")

    print("=== Training Q-learning decay ===")
    q_rets = train_q_learning(
        EMPTY_ROOM,
        args.episodes,
        epsilon_start=0.3,
        epsilon_end=0.05,
        gamma=0.99,
        alpha=0.15
    )
    print(f"Q-learning episodes: {len(q_rets)}")

    plot_learning_curves(
        {'DQN': dqn_rets, 'PPO': ppo_rets, 'QDecay': q_rets},
        window=args.window,
        max_x=args.max_x,
        output_path=args.output
    )

if __name__ == '__main__':
    main()
