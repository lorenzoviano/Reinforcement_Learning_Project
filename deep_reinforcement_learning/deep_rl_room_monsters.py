#!/usr/bin/env python3
"""
deep_rl_room_monsters.py - Train DQN and PPO agents on MiniHack ROOM_WITH_MULTIPLE_MONSTERS environment
and record a 'died' flag whenever a death-penalty (-10) occurs in any step of the episode.
"""
import argparse
import time
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from deep_reinforcement_learning.egocentric_crop_wrapper import EgocentricCropWrapper
from minihack_env import ROOM_WITH_MULTIPLE_MONSTERS, get_minihack_envirnment

# Environment constants
ENV_ID = ROOM_WITH_MULTIPLE_MONSTERS
DEATH_PENALTY = -10  # death penalty for any step

class DeathInfoWrapper(gym.Wrapper):
    """
    Gym wrapper to record if a death-penalty occurred in any step of an episode.
    Adds 'died' flag to info dict.
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


def make_env(crop_size=None):
    """
    Returns a function to create a MiniHack env wrapped with death logging and optional crop.
    """
    def _init():
        env = get_minihack_envirnment(ENV_ID)
        env = DeathInfoWrapper(env, DEATH_PENALTY)
        if crop_size is not None:
            env = EgocentricCropWrapper(env, crop_size)
        return env
    return _init


def train_agent(algo_name, total_timesteps, crop_size, n_envs, algo_kwargs):
    """
    Train DQN or PPO on ROOM_WITH_MULTIPLE_MONSTERS, logging 'died' in monitor.csv.
    """
    save_path = Path("models") / algo_name
    log_path = Path("logs") / algo_name
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    # create vectorized environments
    env_fns = [make_env(crop_size) for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    # log returns, lengths, and 'died'
    vec_env = VecMonitor(vec_env, filename=str(log_path / "monitor.csv"), info_keywords=('died',))

    # evaluation env
    eval_env = get_minihack_envirnment(ENV_ID)
    eval_env = DeathInfoWrapper(eval_env, DEATH_PENALTY)
    if crop_size is not None:
        eval_env = EgocentricCropWrapper(eval_env, crop_size)
    eval_env = Monitor(eval_env, info_keywords=('died',))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path),
        log_path=str(log_path),
        eval_freq=max(1, total_timesteps // 10),
        n_eval_episodes=5,
        deterministic=True,
    )

    # select model
    if algo_name == "DQN":
        model = DQN("MlpPolicy", vec_env, verbose=1,
                    tensorboard_log=str(log_path), **algo_kwargs)
    elif algo_name == "PPO":
        model = PPO("MlpPolicy", vec_env, verbose=1,
                    tensorboard_log=str(log_path), **algo_kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    # train
    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    duration = time.time() - start

    model.save(str(save_path / "final_model"))

    # final evaluation
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"{algo_name} final evaluation: mean_reward={mean_r:.2f} +/- {std_r:.2f}")
    return duration


def plot_learning_curves(log_dirs, window=50, output_path="results/learning_curves.png"):
    plt.figure(figsize=(10,6))
    for algo, log_dir in log_dirs.items():
        csv = os.path.join(log_dir, "monitor.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv, comment='#')
        rewards = df['r'].values
        episodes = np.arange(1, len(rewards)+1)
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ep = episodes[window-1:]
            plt.plot(ep, ma, label=f"{algo} (MA{window})")
        else:
            plt.plot(episodes, rewards, label=algo)
    plt.xlabel('Episode')
    plt.ylabel('Episode return (moving avg)')
    plt.title(f"Learning curves on {ENV_ID}")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train DQN and PPO on ROOM_WITH_MULTIPLE_MONSTERS")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--crop_size", type=int, default=9)
    parser.add_argument("--n_envs", type=int, default=4)
    # DQN params
    parser.add_argument("--dqn-gamma", type=float, default=0.95)
    parser.add_argument("--dqn-batch-size", type=int, default=16)
    parser.add_argument("--dqn-learning-starts", type=int, default=1000)
    parser.add_argument("--dqn-tau", type=float, default=0.9)
    parser.add_argument("--dqn-exploration-fraction", type=float, default=0.45)
    parser.add_argument("--dqn-exploration-initial-eps", type=float, default=0.55)
    parser.add_argument("--dqn-exploration-final-eps", type=float, default=0.1520)
    parser.add_argument("--dqn-target-update-interval", type=int, default=10000)
    # PPO params
    parser.add_argument("--ppo-gamma", type=float, default=0.95)
    parser.add_argument("--ppo-batch-size", type=int, default=32)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument("--ppo-normalize-advantage", type=lambda x: (str(x).lower()=='true'), default=False)
    parser.add_argument("--ppo-target-kl", type=float, default=0.0180)
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.9986)
    parser.add_argument("--ppo-vf-coef", type=float, default=0.9394)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.0020)
    parser.add_argument("--ppo-n-epochs", type=int, default=10)
    parser.add_argument("--ppo-n-steps", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    dqn_kwargs = {
        'gamma': args.dqn_gamma,
        'batch_size': args.dqn_batch_size,
        'learning_starts': args.dqn_learning_starts,
        'tau': args.dqn_tau,
        'exploration_fraction': args.dqn_exploration_fraction,
        'exploration_initial_eps': args.dqn_exploration_initial_eps,
        'exploration_final_eps': args.dqn_exploration_final_eps,
        'target_update_interval': args.dqn_target_update_interval,
    }
    ppo_kwargs = {
        'gamma': args.ppo_gamma,
        'batch_size': args.ppo_batch_size,
        'clip_range': args.ppo_clip_range,
        'normalize_advantage': args.ppo_normalize_advantage,
        'target_kl': args.ppo_target_kl,
        'gae_lambda': args.ppo_gae_lambda,
        'vf_coef': args.ppo_vf_coef,
        'ent_coef': args.ppo_ent_coef,
        'n_epochs': args.ppo_n_epochs,
        'n_steps': args.ppo_n_steps,
    }

    print(f"=== Training DQN on {ENV_ID} ===")
    d_time = train_agent("DQN", args.timesteps, args.crop_size, args.n_envs, dqn_kwargs)
    print(f"DQN training took {d_time:.2f}s\n")
    print(f"=== Training PPO on {ENV_ID} ===")
    p_time = train_agent("PPO", args.timesteps, args.crop_size, args.n_envs, ppo_kwargs)
    print(f"PPO training took {p_time:.2f}s\n")

    print("=== Summary ===")
    print(f"DQN: {d_time:.2f}s, PPO: {p_time:.2f}s")

    plot_learning_curves({"DQN": "logs/DQN", "PPO": "logs/PPO"}, window=100, output_path="results/learning_curves.png")

if __name__ == "__main__":
    main()
