#!/usr/bin/env python3
"""
plot_win_rate_histogram.py - Plot stacked outcome histograms per chunk of episodes for RL agents and Q-learning decay baseline,
using the 'died' flag recorded during RL training, and matching the exact death condition in Q-learning.

Usage:
    python plot_win_rate_histogram.py \
        --logs DQN:logs/DQN PPO:logs/PPO \
        --max_episodes 15000 \
        --chunk 1500 \
        --q-epsilon-start 0.3 \
        --q-epsilon-end 0.05 \
        --q-gamma 0.99 \
        --q-alpha 0.15 \
        --output_dir results/histograms
"""
import os
import argparse
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

from minihack_env import ROOM_WITH_MULTIPLE_MONSTERS, get_minihack_envirnment
from agents import QLearningOffPolicyDecay

# Constants
MAX_STEP = 600
DEATH_PENALTY = -10    # penalty for death in each step
GOOD_WIN_THRES = -10   # threshold for good vs. normal win on episode return
OUTCOMES = ["good_win", "norm_win", "fail"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(OUTCOMES)}

class DeathInfoWrapper(gym.Wrapper):
    """
    Wrapper to record if a step with death penalty occurred in the episode.
    Adds 'died' key to info dict and tracks internally.
    """
    def __init__(self, env, death_penalty):
        super().__init__(env)
        self.death_penalty = death_penalty
        self.died = False

    def reset(self, **kwargs):
        self.died = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if reward <= self.death_penalty:
            self.died = True
        info['died'] = self.died
        return obs, reward, term, trunc, info


def load_rl_outcomes(log_dir, max_eps):
    """
    Load outcomes from RL monitor.csv using the logged 'died' flag.
    """
    df = pd.read_csv(os.path.join(log_dir, 'monitor.csv'), comment='#')
    rewards = df['r'].values[:max_eps]
    lengths = df['l'].values[:max_eps]
    died_flags = df['died'].values[:max_eps]
    outcomes = []
    for r, l, died in zip(rewards, lengths, died_flags):
        if died or l >= MAX_STEP:
            outcomes.append('fail')
        else:
            outcomes.append('good_win' if r > GOOD_WIN_THRES else 'norm_win')
    return outcomes


def train_q_learning(env_id, n_episodes, eps_start, eps_end, gamma, alpha):
    """
    Train a Q-learning agent with decaying epsilon. Wrap env to track death by step penalty.
    Returns list of outcomes per episode.
    """
    random.seed(0)
    np.random.seed(0)
    base_env = get_minihack_envirnment(env_id, add_pixel=False)
    env = DeathInfoWrapper(base_env, DEATH_PENALTY)
    outcomes = []
    agent = QLearningOffPolicyDecay(
        id='Qlearning',
        action_space=env.action_space,
        epsilon_start=eps_start,
        epsilon_end=eps_end,
        max_episodes=n_episodes,
        gamma=gamma,
        alpha=alpha
    )

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_r = 0.0
        prev_r = 0.0
        steps = 0

        while not done and steps < MAX_STEP:
            action = agent.act(obs, prev_r)
            obs, r, term, trunc, info = env.step(action)
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
        # Use death flag from wrapper
        if env.died or steps >= MAX_STEP:
            outcomes.append('fail')
        else:
            outcomes.append('good_win' if total_r > GOOD_WIN_THRES else 'norm_win')

    env.close()
    return outcomes



def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curves including Q-learning decay baseline and RL agents."
    )
    parser.add_argument(
        "--logs", nargs='+', default=["DQN:logs/DQN", "PPO:logs/PPO"],
        help="List of algo:log_dir pairs, e.g. DQN:logs/DQN PPO:logs/PPO. Defaults to DQN and PPO."
    )
    parser.add_argument(
        "--max_episodes", type=int, default=15000,
        help="Maximum number of episodes"
    )
    parser.add_argument(
        "--window", type=int, default=50,
        help="Window size for moving average"
    )
    parser.add_argument(
        "--q-epsilon-start", type=float, default=0.6,
        help="Initial epsilon for Q-learning"
    )
    parser.add_argument(
        "--q-epsilon-end", type=float, default=0.6,
        help="Final epsilon for Q-learning"
    )
    parser.add_argument(
        "--q-gamma", type=float, default=0.99,
        help="Discount factor for Q-learning"
    )
    parser.add_argument(
        "--q-alpha", type=float, default=0.15,
        help="Learning rate for Q-learning"
    )
    parser.add_argument(
        "--output", type=str, default="results/learning_curves_with_q.png",
        help="Path to save output figure"
    )
    args = parser.parse_args()

    all_outcomes = {}
    # Load outcomes for RL agents, using recorded 'died' flag
    for pair in args.logs:
        if ':' in pair:
            name, path = pair.split(':',1)
            all_outcomes[name] = load_rl_outcomes(path, args.max_episodes)

    # Q-learning baseline with death wrapper
    print(f'Training Q-learning with decay for {args.max_episodes} episodes...')
    q_outs = train_q_learning(
        ROOM_WITH_MULTIPLE_MONSTERS,
        args.max_episodes,
        args.q_epsilon_start,
        args.q_epsilon_end,
        args.q_gamma,
        args.q_alpha
    )
    all_outcomes['Qlearning'] = q_outs
    print('Q-learning done.')


if __name__ == '__main__':
    main()
