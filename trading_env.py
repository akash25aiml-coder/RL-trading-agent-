"""
trading_env.py
================
A minimal, Gymnasium-compatible trading environment for tabular Q-learning.

Design (v2 -- regime-aware):
    State   : (short_term_bucket, position, long_term_regime)
              -> 5 x 3 x 3 = 45 discrete states
    Action  : {0: Hold, 1: Buy, 2: Sell}
    Reward  : change in portfolio value for the step (mark-to-market)
    Episode : a fixed-length window of consecutive trading days sampled
              from historical OHLCV data.

WHY A REGIME FEATURE (v1 -> v2 change):
    v1 used only a short-window (5-day) return bucket. The agent learned a
    mean-reversion policy (sell after strong rallies, buy after crashes).
    That policy is disastrous during a sustained bull trend, because it
    fights the trend instead of following it -- confirmed empirically:
    v1 lost -19.45% while Buy & Hold returned +478% over the 2018-2024
    AAPL test period.

    v2 adds a long-window (50-day) trend regime bucket {Downtrend, Sideways,
    Uptrend} to the state, so the SAME tabular agent can, in principle,
    learn different short-term behaviour depending on the prevailing
    regime (e.g. "buy dips only when the long-term regime is Uptrend").
    This is still fully tabular (45 states, 135 Q-values) -- no function
    approximation needed.

Author: <your name>
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# --------------------------------------------------------------------------- #
HOLD, BUY, SELL = 0, 1, 2
ACTION_NAMES = {HOLD: "Hold", BUY: "Buy", SELL: "Sell"}

FLAT, LONG, SHORT = 0, 1, 2
POSITION_NAMES = {FLAT: "Flat", LONG: "Long", SHORT: "Short"}

# Short-term (5-day) price-trend buckets.
SHORT_BUCKET_EDGES = [-0.02, -0.005, 0.005, 0.02]  # -> 5 buckets
SHORT_BUCKET_NAMES = ["StrongDown", "Down", "Flat", "Up", "StrongUp"]

# Long-term (50-day) regime buckets.
REGIME_EDGES = [-0.05, 0.05]  # -> 3 buckets
REGIME_NAMES = ["Downtrend", "Sideways", "Uptrend"]

N_SHORT_BUCKETS = len(SHORT_BUCKET_NAMES)   # 5
N_POSITIONS = 3
N_REGIMES = len(REGIME_NAMES)               # 3
N_STATES = N_SHORT_BUCKETS * N_POSITIONS * N_REGIMES  # 45


def bucket_return(ret: float, edges: list[float]) -> int:
    """Map a raw return to a discrete bucket index given bin edges."""
    return int(np.digitize(ret, edges))


class TradingEnv(gym.Env):
    """
    A discretized single-asset trading environment with a regime feature.

    Parameters
    ----------
    prices : pd.Series
        Chronologically ordered closing prices (float), indexed by date.
    short_window : int
        Trailing days used for the short-term return bucket (mean-reversion signal).
    long_window : int
        Trailing days used for the long-term regime bucket (trend signal).
    episode_length : int
        Number of steps (trading days) per episode.
    transaction_cost : float
        Fractional cost applied on every position change (0.0 = frictionless).
    seed : int | None
        RNG seed for reproducible episode sampling.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: pd.Series,
        short_window: int = 5,
        long_window: int = 50,
        episode_length: int = 100,
        transaction_cost: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()

        self.min_history = max(short_window, long_window)
        if len(prices) < self.min_history + episode_length + 1:
            raise ValueError(
                "Price series too short for the requested window + episode_length."
            )

        self.prices = prices.reset_index(drop=True)
        self._prices_arr = self.prices.values  # numpy array -- avoids slow pandas .iloc in hot loop
        self.short_window = short_window
        self.long_window = long_window
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost

        # Precompute both return series once (vectorized).
        self._short_returns = self.prices.pct_change(periods=short_window).fillna(0.0).values
        self._long_returns = self.prices.pct_change(periods=long_window).fillna(0.0).values

        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(3)

        self._rng = np.random.default_rng(seed)

        self._t: int = 0
        self._start_idx: int = 0
        self.position: int = FLAT
        self.entry_price: float | None = None
        self.portfolio_value: float = 1.0

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        max_start = len(self.prices) - self.episode_length - 1
        self._start_idx = int(self._rng.integers(self.min_history, max_start))
        self._t = self._start_idx

        self.position = FLAT
        self.entry_price = None
        self.portfolio_value = 1.0

        return self._get_state(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        price_now = self._prices_arr[self._t]
        prev_portfolio_value = self.portfolio_value

        reward_cost = 0.0
        if action == BUY and self.position != LONG:
            reward_cost = self.transaction_cost
            self.position = LONG
            self.entry_price = price_now
        elif action == SELL and self.position != SHORT:
            reward_cost = self.transaction_cost
            self.position = SHORT
            self.entry_price = price_now

        self._t += 1
        price_next = self._prices_arr[self._t]

        pct_change = (price_next - price_now) / price_now
        if self.position == LONG:
            step_return = pct_change
        elif self.position == SHORT:
            step_return = -pct_change
        else:
            step_return = 0.0

        self.portfolio_value *= (1 + step_return - reward_cost)
        reward = self.portfolio_value - prev_portfolio_value

        steps_taken = self._t - self._start_idx
        terminated = False
        truncated = steps_taken >= self.episode_length

        obs = self._get_state()
        info = self._get_info(step_return=step_return, action=action)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _get_state(self) -> int:
        """Encode (short_bucket, position, regime) as a single discrete index."""
        short_bucket = bucket_return(self._short_returns[self._t], SHORT_BUCKET_EDGES)
        regime = bucket_return(self._long_returns[self._t], REGIME_EDGES)
        return (short_bucket * N_POSITIONS + self.position) * N_REGIMES + regime

    def _get_info(self, step_return: float = 0.0, action: int | None = None) -> dict:
        return {
            "t": self._t,
            "price": float(self._prices_arr[self._t]),
            "position": POSITION_NAMES[self.position],
            "portfolio_value": float(self.portfolio_value),
            "step_return": float(step_return),
            "action": ACTION_NAMES.get(action) if action is not None else None,
        }

    @staticmethod
    def decode_state(state: int) -> tuple[str, str, str]:
        """Turn a state index back into (short_bucket_name, position_name, regime_name)."""
        regime = state % N_REGIMES
        rest = state // N_REGIMES
        position = rest % N_POSITIONS
        short_bucket = rest // N_POSITIONS
        return SHORT_BUCKET_NAMES[short_bucket], POSITION_NAMES[position], REGIME_NAMES[regime]

    def render(self):
        info = self._get_info()
        print(
            f"t={info['t']:5d} | price={info['price']:.2f} | "
            f"pos={info['position']:<5s} | value={info['portfolio_value']:.4f}"
        )

