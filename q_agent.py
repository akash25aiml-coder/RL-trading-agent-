"""
q_agent.py
==========
Tabular Q-learning agent for the 15-state / 3-action TradingEnv.

Update rule (Bellman equation, off-policy TD control):

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Exploration: epsilon-greedy with exponential decay, so the agent explores
heavily early (random actions) and exploits its learned policy later.
"""

from __future__ import annotations
import numpy as np


class QLearningAgent:
    """
    Parameters
    ----------
    n_states, n_actions : int
        Size of the discrete state/action spaces (15 and 3 here).
    alpha : float
        Learning rate.
    gamma : float
        Discount factor for future rewards.
    epsilon_start, epsilon_end, epsilon_decay : float
        Epsilon-greedy exploration schedule:
            epsilon(episode) = epsilon_end + (epsilon_start - epsilon_end) * exp(-episode / epsilon_decay)
    seed : int | None
        RNG seed for reproducible action sampling.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 300.0,
        seed: int | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table initialized to zero -> optimistic-ish for reward scale used here.
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
        # Visitation counts -> lets us report *confidence* per (state,action),
        # not just the learned value. Critical for honest reporting: a Q-value
        # from 5 updates is not as trustworthy as one from 5000.
        self.visit_counts = np.zeros((n_states, n_actions), dtype=np.int64)
        self._rng = np.random.default_rng(seed)

        self.episode_count = 0
        self.current_epsilon = epsilon_start

    # ------------------------------------------------------------------ #
    def epsilon(self, episode: int) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -episode / self.epsilon_decay
        )

    def act(self, state: int, greedy: bool = False) -> int:
        """
        epsilon-greedy action selection.
        greedy=True forces pure exploitation (used at evaluation time).
        """
        if not greedy and self._rng.random() < self.current_epsilon:
            return int(self._rng.integers(self.n_actions))

        q_row = self.Q[state]
        # Tie-breaking: random among the max, not always the first index.
        # This avoids the agent getting stuck always picking action 0
        # early in training when all Q-values are tied at 0.
        max_q = np.max(q_row)
        best_actions = np.flatnonzero(q_row == max_q)
        return int(self._rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """One Bellman/TD update step."""
        best_next_q = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        self.visit_counts[state, action] += 1
        return td_error

    def start_episode(self, episode: int):
        """Call once per episode to update the exploration rate."""
        self.episode_count = episode
        self.current_epsilon = self.epsilon(episode)

    # ------------------------------------------------------------------ #
    def policy_table(self, decode_fn=None) -> list[dict]:
        """Return the greedy policy for every state, human-readable if
        decode_fn (e.g. TradingEnv.decode_state) is supplied."""
        rows = []
        for s in range(self.n_states):
            best_a = int(np.argmax(self.Q[s]))
            row = {
                "state": s,
                "best_action": best_a,
                "Q_values": self.Q[s].tolist(),
            }
            if decode_fn is not None:
                decoded = decode_fn(s)
                row["decoded"] = decoded  # tuple, arbitrary length (2 or 3)
            row["visits"] = int(self.visit_counts[s, best_a])
            row["state_total_visits"] = int(self.visit_counts[s].sum())
            rows.append(row)
        return rows

    def confidence_report(self, decode_fn=None, low_visit_threshold: int = 200) -> list[dict]:
        """
        Flag states whose (state, best_action) pair was updated fewer than
        `low_visit_threshold` times -- these Q-values should be reported with
        a caveat, since the TD estimate hasn't converged from enough samples.
        """
        flagged = []
        for row in self.policy_table(decode_fn=decode_fn):
            if row["visits"] < low_visit_threshold:
                flagged.append(row)
        return flagged
