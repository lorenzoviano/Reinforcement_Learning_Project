
import numpy as np
import random
from collections import defaultdict
from commons import AbstractAgent
from typing import Optional



class _BaseLearningAgent(AbstractAgent):
    """Common utilities for table‑based ε‑greedy agents."""

    def __init__(self, id: str, action_space, *, epsilon: float = 0.1, alpha: float = 0.1, gamma: float = 0.99):
        super().__init__(id, action_space)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # Q‑table: dict state_key -> np.ndarray(|A|)
        self.Q = defaultdict(lambda: np.zeros(self.action_space.n, dtype=np.float32))
        # placeholders used for TD updates
        self._prev_state_key = None
        self._prev_action = None

    # helpers

    def _state_key(self, state):
        """Compress the observation into something hashable."""
        if isinstance(state, dict):
            # MiniHack observations – we use the chars array only
            return state["chars"].tobytes()
        # fallback for simple tuples / ints
        return tuple(np.array(state).flatten())
    
    
    def _epsilon_greedy(self, state_key):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        q_vals = self.Q[state_key]
        best = np.argwhere(q_vals == q_vals.max()).flatten()
        return int(random.choice(best))


    def onEpisodeEnd(self, *args, **kwargs):
        """Reset book‑keeping at the end of each episode."""
        self._prev_state_key = None
        self._prev_action = None


# ------------------------------------------------------------------
# ------------------------------------------------------------------

class RandomAgent(AbstractAgent):
    """Chooses an action uniformly at random."""

    def __init__(self, id: str, action_space):
        super().__init__(id, action_space)

    def act(self, state, reward=0):
        return self.action_space.sample()

    def onEpisodeEnd(self, *args, **kwargs):
        pass


class FixedAgent(AbstractAgent):
    """Always move SOUTH until blocked, then EAST forever (MiniHack‑specific)."""

    def __init__(self, id: str, action_space):
        super().__init__(id, action_space)
        self.mode = "down"
        self.walkable = {ord('.'), ord('>')}

    def act(self, state, reward=0):
        chars = state["chars"]
        r, c = np.argwhere(chars == ord('@'))[0]
        n_rows, _ = chars.shape
        if self.mode == "down":
            if r < n_rows - 1 and chars[r + 1, c] in self.walkable:
                return 2  # SOUTH
            self.mode = "right"
        return 1  # EAST

    def onEpisodeEnd(self, *args, **kwargs):
        self.mode = "down"



# ------------------------------------------------------------------
#  LEARNING AGENTS                                                   
# ------------------------------------------------------------------

class MonteCarloOnPolicy(_BaseLearningAgent):
    """First-visit Monte-Carlo control with ε-greedy policy."""

    def __init__(self, id: str, action_space, *, epsilon=0.1, gamma=0.99):
        super().__init__(id, action_space, epsilon=epsilon, alpha=None, gamma=gamma)
        self._returns_sum = defaultdict(float)
        self._returns_cnt = defaultdict(int)

    def act(self, state, reward=0):
        state_key = self._state_key(state)
        return self._epsilon_greedy(state_key)

    def onEpisodeEnd(self, *, episode, **kwargs):
        """
        First-visit MC: build the forward list, then walk backwards
        computing G_t correctly and only update the first occurrence.
        """
        episode_return = 0.0

        # Build forward list of state-action “keys”
        sa_list = [
            (self._state_key(s), a)
            for (s, a, _) in episode
        ]

        t = len(episode)
        # Traverse episode in reverse so that the last time we see a pair
        # is actually its first visit in forward time
        for s, a, r in reversed(episode):
            t -= 1
            # ← FIXED return-to-go
            episode_return = r + self.gamma * episode_return

            key = (self._state_key(s), a)
            # first-visit check: only update if this (s,a) wasn't seen earlier
            if key not in sa_list[:t]:
                # accumulate & count returns
                self._returns_sum[key] += episode_return
                self._returns_cnt[key] += 1

                # update Q to the new average
                self.Q[key[0]][a] = (
                    self._returns_sum[key] / self._returns_cnt[key]
                )

        # now let the base class handle ε-decay logging, resetting counts, etc.
        super().onEpisodeEnd()


class SARSAOnPolicy(_BaseLearningAgent):
    """On-policy TD(0) control a.k.a. SARSA."""

    def __init__(self, id, action_space, *, epsilon=0.1, gamma=0.99, alpha=0.1):
        super().__init__(id, action_space, epsilon=epsilon, alpha=alpha, gamma=gamma)
        self._prev_state_key = None
        self._prev_action    = None

    def act(self, state, reward=0):
        state_key = self._state_key(state)

        # ---- choose A' first (on-policy) ----------------------
        next_action = self._epsilon_greedy(state_key)

        # ---- then do the TD update for (S,A) → (S',A') --------
        if self._prev_state_key is not None:
            td_target = reward + self.gamma * self.Q[state_key][next_action]
            self.Q[self._prev_state_key][self._prev_action] += self.alpha * (
                td_target - self.Q[self._prev_state_key][self._prev_action]
            )

        # ---- shift (S,A) ← (S',A') for the next step -----------
        self._prev_state_key = state_key
        self._prev_action    = next_action

        return next_action

    def onEpisodeEnd(self, *, last_step_reward=0, **kwargs):
        # final backup when S' is terminal (Q(terminal,·)=0)
        if self._prev_state_key is not None:
            td_target = last_step_reward  # + γ·0 
            self.Q[self._prev_state_key][self._prev_action] += self.alpha * (
                td_target - self.Q[self._prev_state_key][self._prev_action]
            )
        super().onEpisodeEnd()



class QLearningOffPolicy(_BaseLearningAgent):
    """Off‑policy TD control (Watkins' Q‑learning)."""

    def act(self, state, reward=0):
        state_key = self._state_key(state)

        # ---- TD update for previous transition --------------------
        if self._prev_state_key is not None:
            target = reward + self.gamma * np.max(self.Q[state_key])
            self.Q[self._prev_state_key][self._prev_action] += self.alpha * (
                target - self.Q[self._prev_state_key][self._prev_action]
            )

        # ---- choose current action ε‑greedy -----------------------
        current_action = self._epsilon_greedy(state_key)
        self._prev_state_key = state_key
        self._prev_action = current_action
        return current_action

    def onEpisodeEnd(self, *, last_step_reward=0, **kwargs):
        if self._prev_state_key is not None:
            self.Q[self._prev_state_key][self._prev_action] += self.alpha * (
                last_step_reward - self.Q[self._prev_state_key][self._prev_action]
            )
        super().onEpisodeEnd()


# ------------------------------------------------------------------
# ------------------------------------------------------------------


class _LinearEpsilonDecayMixin:
    """
    Adds a per-episode linear decay for ε.

    Parameters
    ----------
    epsilon_start : float
    epsilon_end   : float
    max_episodes  : int
    """
    def _init_decay(self,
                    epsilon_start: float,
                    epsilon_end:   float,
                    max_episodes:  int):
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end   = float(epsilon_end)
        self.max_episodes  = int(max_episodes)
        self._episode_idx  = 0          # t in the formula

    def _update_epsilon(self):
        """Compute ε_t at the *end* of each episode."""
        self._episode_idx += 1          # t ← t+1
        frac = min(self._episode_idx, self.max_episodes) / self.max_episodes
        self.epsilon = (
            self.epsilon_start -
            frac * (self.epsilon_start - self.epsilon_end)
        )
# ───────────────────────────────────────────────────────────────────────────────
#  New agents with decaying ε
# ───────────────────────────────────────────────────────────────────────────────
from collections import defaultdict

class MonteCarloOnPolicyDecay(_LinearEpsilonDecayMixin, MonteCarloOnPolicy):
    """
    First-visit Monte-Carlo control with *linearly decaying* ε-greedy policy.
    All MC logic is inherited; only ε scheduling is new.
    """
    def __init__(self, id, action_space, *,
                 epsilon_start=1.0, epsilon_end=0.05, max_episodes=1_000,
                 gamma=0.99):
        super().__init__(id, action_space,
                         epsilon=epsilon_start,
                         gamma=gamma)
        self._init_decay(epsilon_start, epsilon_end, max_episodes)

    # schedule ε after each episode
    def onEpisodeEnd(self, **kwargs):
        super().onEpisodeEnd(**kwargs)
        self._update_epsilon()


class SARSAOnPolicyDecay(_LinearEpsilonDecayMixin, SARSAOnPolicy):
    """SARSA with linearly decaying ε."""
    def __init__(self, id, action_space, *,
                 epsilon_start=1.0, epsilon_end=0.05, max_episodes=1_000,
                 gamma=0.99, alpha=0.1):
        super().__init__(id, action_space,
                         epsilon=epsilon_start,
                         gamma=gamma,
                         alpha=alpha)
        self._init_decay(epsilon_start, epsilon_end, max_episodes)

    def onEpisodeEnd(self, **kwargs):
        super().onEpisodeEnd(**kwargs)
        self._update_epsilon()


class QLearningOffPolicyDecay(_LinearEpsilonDecayMixin, QLearningOffPolicy):
    """Watkins’ Q-learning with linearly decaying ε."""
    def __init__(self, id, action_space, *,
                 epsilon_start=1.0, epsilon_end=0.05, max_episodes=1_000,
                 gamma=0.99, alpha=0.1):
        super().__init__(id, action_space,
                         epsilon=epsilon_start,
                         gamma=gamma,
                         alpha=alpha)
        self._init_decay(epsilon_start, epsilon_end, max_episodes)

    def onEpisodeEnd(self, **kwargs):
        super().onEpisodeEnd(**kwargs)
        self._update_epsilon()



# ------------------------------------------------------------------
#  Dyna-Q (tabular, Watkins-style updates + n random planning steps)
# ------------------------------------------------------------------
class DynaQAgent(_BaseLearningAgent):
    """
    Tabular Dyna-Q (Sutton & Barto, §8.1).

    Parameters
    ----------
    n_planning : int
        How many simulated updates to perform after every real step.
    """
    def __init__(self, id, action_space, *,
                 epsilon=0.10, alpha=0.10, gamma=0.99,
                 n_planning=10):
        super().__init__(id, action_space,
                         epsilon=epsilon, alpha=alpha, gamma=gamma)
        self.n_planning = n_planning
        # deterministic one-step model: (s_key, a) → (next_s_key, reward)
        self.model = {}

    # ------------------------------------------------------------------
    def act(self, state, reward: float = 0.0):
        """Choose an action and perform n planning backups."""
        s_key = self._state_key(state)

        # 1. TD update for the *real* transition observed last step
        if self._prev_state_key is not None:
            td_target = reward + self.gamma * np.max(self.Q[s_key])
            self.Q[self._prev_state_key][self._prev_action] += self.alpha * (
                td_target - self.Q[self._prev_state_key][self._prev_action]
            )
            # store that transition in the model
            self.model[(self._prev_state_key, self._prev_action)] = (s_key, reward)

        # 2. n simulated planning backups  (only if model is non-empty)
        if self.model:                         # ← the crucial guard
            for _ in range(self.n_planning):
                (s_sim, a_sim), (s_next, r_sim) = random.choice(
                    list(self.model.items())
                )
                td_target = r_sim + self.gamma * np.max(self.Q[s_next])
                self.Q[s_sim][a_sim] += self.alpha * (
                    td_target - self.Q[s_sim][a_sim]
                )

        # 3. ε-greedy action selection for the current real state
        a = self._epsilon_greedy(s_key)
        self._prev_state_key = s_key
        self._prev_action    = a
        return a
