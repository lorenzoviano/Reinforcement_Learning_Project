import copy
import numpy as np
from commons import AbstractRLTask


class RLTask(AbstractRLTask):
    """Environment ↔ agent wrapper with built‑in tracking & prints."""

    def interact(self, n_episodes: int):
        episode_returns, avg_returns = [], []

        for k in range(n_episodes):
            obs, _ = self.env.reset()
            done, G_k, steps, episode = False, 0.0, 0, []
            prev_reward = 0.0  # reward obtained on the previous env.step

            while not done:
                # ---------------------------------------------------------
                # Take a *deep copy* of the CURRENT observation *before*
                # the env.step call. Gym/MiniHack re‑uses the same numpy
                # buffers, so without this copy every element in `episode`
                # would eventually reference the **same** array and the
                # Monte‑Carlo update would collapse to a single state.
                # ---------------------------------------------------------
                obs_copy = copy.deepcopy(obs)

                action = self.agent.act(obs, prev_reward)
                next_obs, reward, term, trunc, _ = self.env.step(action)

                episode.append((obs_copy, action, reward))
                G_k += reward
                steps += 1

                done = term or trunc
                obs, prev_reward = next_obs, reward

            # ------ let the agent finalise the episode --------------
            if hasattr(self.agent, "onEpisodeEnd"):
                self.agent.onEpisodeEnd(
                    episode=episode,
                    reward=G_k,
                    episode_number=k,
                    last_step_reward=prev_reward,
                )

            # ------ bookkeeping / prints ---------------------------
            episode_returns.append(G_k)
            avg_returns.append(np.mean(episode_returns))
            reached_goal = steps < 50  # MiniHackRoom default limit
            # print(
            #     f"ep {k:3d}:  return = {G_k:5.0f}  steps = {steps:2d}  reached goal? {reached_goal}",
            #     flush=True,
            # )

        return avg_returns
    
    

    def visualize_episode(self, max_number_steps=None):
        """
        Run a single episode (up to max_number_steps), calling env.render()
        at each time‐step so you can see what the agent does.
        """
        obs, _ = self.env.reset()
        done = False
        t = 0

        # If the env supports human‐rendering, this will pop up a window.
        while not done:
            # your agent.act interface might be (state, reward) depending on signature
            action = self.agent.act(obs, reward=0)
            obs, reward, done, truncated, info = self.env.step(action)
            # actually render
            self.env.render()

            t += 1
            if max_number_steps is not None and t >= max_number_steps:
                break

        # make sure any rendering windows get properly closed
        self.env.close()

