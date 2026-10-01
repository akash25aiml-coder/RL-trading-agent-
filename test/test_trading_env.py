"""Correctness and regression tests. Run from the repo root with:  pytest"""
from pathlib import Path

import numpy as np
import pytest

from multi_asset_lib import LONG_WINDOW, evaluate_q, load_prices
from trading_env import BUY, HOLD, N_STATES, TradingEnv

ROOT = Path(__file__).resolve().parents[1]
# Returns published in the README/report at 0.05% cost. If you retrain, update these.
PUBLISHED = {"AAPL": ("aapl.csv", 1.849), "MSFT": ("msft.csv", 1.497), "INTC": ("intc.csv", 0.227)}


@pytest.fixture(scope="module")
def aapl_prices():
    prices, _ = load_prices(ROOT / "aapl.csv")
    return prices


def test_state_space_size():
    assert N_STATES == 45  # 5 trend buckets x 3 positions x 3 regimes


def test_states_stay_valid_and_rewards_finite(aapl_prices):
    env = TradingEnv(aapl_prices, seed=0)
    state, _ = env.reset(seed=0)
    for _ in range(env.episode_length):
        assert env.observation_space.contains(state)
        state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        if terminated or truncated:
            break


def test_long_position_matches_price_move(aapl_prices):
    env = TradingEnv(aapl_prices, episode_length=10, transaction_cost=0.0, seed=1)
    env.reset(seed=1)
    start_price = env.prices.iloc[env._t]
    env.step(BUY)
    for _ in range(5):
        _, _, _, _, info = env.step(HOLD)
    assert info["portfolio_value"] == pytest.approx(env.prices.iloc[env._t] / start_price, rel=1e-9)


def test_transaction_cost_charged_only_when_position_changes(aapl_prices):
    free = TradingEnv(aapl_prices, transaction_cost=0.0, seed=2)
    costly = TradingEnv(aapl_prices, transaction_cost=0.01, seed=2)
    free.reset(seed=2)
    costly.reset(seed=2)
    _, _, _, _, a1 = free.step(BUY)
    _, _, _, _, b1 = costly.step(BUY)
    assert a1["portfolio_value"] - b1["portfolio_value"] == pytest.approx(0.01)
    _, _, _, _, a2 = free.step(HOLD)  # holding must not be charged again
    _, _, _, _, b2 = costly.step(HOLD)
    assert b2["portfolio_value"] / b1["portfolio_value"] == pytest.approx(
        a2["portfolio_value"] / a1["portfolio_value"])


@pytest.mark.parametrize("ticker", PUBLISHED)
def test_saved_q_table_shape(ticker):
    Q = np.load(ROOT / "models" / f"{ticker}.npy")
    assert Q.shape == (N_STATES, 3) and np.isfinite(Q).all()


@pytest.mark.parametrize("ticker", PUBLISHED)
def test_backtest_reproduces_published_return(ticker):
    csv_name, expected = PUBLISHED[ticker]
    prices, _ = load_prices(ROOT / csv_name)
    split = int(len(prices) * 0.85)
    test_prices = prices.iloc[split - LONG_WINDOW:].reset_index(drop=True)
    Q = np.load(ROOT / "models" / f"{ticker}.npy")
    assert evaluate_q(Q, test_prices, cost=0.0005)[1] == pytest.approx(expected, abs=0.005)
