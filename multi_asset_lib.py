"""
multi_asset_lib.py
===================
Reusable train+evaluate pipeline (same methodology as train.py/evaluate.py:
full-coverage epochs, cost-aware training, N-seed ensemble) parameterized
by CSV file, so it can be run independently across multiple tickers to
test whether the approach generalizes beyond AAPL -- directly addressing
the single-asset survivorship-bias limitation.
"""

import numpy as np
import pandas as pd
import json
from trading_env import TradingEnv, FLAT
from q_agent import QLearningAgent

SHORT_WINDOW = 5
LONG_WINDOW = 50
EPISODE_LENGTH = 100
TRAIN_TRANSACTION_COST = 0.0005
EVAL_TRANSACTION_COST = 0.0005


def load_prices(csv_path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date")
    return df["Close"].reset_index(drop=True), df["Date"].reset_index(drop=True)


def train_one_seed(train_prices, seed, n_epochs):
    env = TradingEnv(train_prices, short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
                      episode_length=EPISODE_LENGTH, transaction_cost=TRAIN_TRANSACTION_COST, seed=seed)
    valid_starts = np.arange(env.min_history, len(train_prices) - EPISODE_LENGTH - 1)
    n_ep_total = n_epochs * len(valid_starts)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n,
                            alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05,
                            epsilon_decay=n_ep_total / 5, seed=seed)
    rng = np.random.default_rng(seed)
    gep = 0
    for epoch in range(n_epochs):
        for start in rng.permutation(valid_starts):
            agent.start_episode(gep)
            env._start_idx, env._t = int(start), int(start)
            env.position, env.entry_price, env.portfolio_value = FLAT, None, 1.0
            state = env._get_state()
            for _ in range(EPISODE_LENGTH):
                action = agent.act(state)
                ns, r, term, trunc, info = env.step(action)
                agent.update(state, action, r, ns, done=(term or trunc))
                state = ns
                if term or trunc:
                    break
            gep += 1
    return agent.Q


def evaluate_q(Q, test_prices, cost=EVAL_TRANSACTION_COST):
    env = TradingEnv(test_prices, short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
                      episode_length=EPISODE_LENGTH, transaction_cost=cost, seed=123)
    n_windows = (len(test_prices) - env.min_history - 1) // EPISODE_LENGTH
    equity = [1.0]
    cur = 1.0
    t = env.min_history
    for _ in range(n_windows):
        env._start_idx, env._t = t, t
        env.position, env.entry_price, env.portfolio_value = FLAT, None, cur
        for _ in range(EPISODE_LENGTH):
            state = env._get_state()
            action = int(np.argmax(Q[state]))
            _, _, term, trunc, info = env.step(action)
            equity.append(info["portfolio_value"])
            if term or trunc:
                break
        cur = env.portfolio_value
        t = env._t
    equity = np.array(equity)
    rets = np.diff(equity) / equity[:-1]
    sharpe = 0.0 if rets.std() == 0 else (rets.mean() / rets.std()) * np.sqrt(252)
    mdd = ((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)).min()
    return equity, equity[-1] - 1, sharpe, mdd, t, env.min_history


def run_asset(name, csv_path, seeds, n_epochs, split_frac=0.85):
    prices, dates = load_prices(csv_path)
    split_idx = int(len(prices) * split_frac)
    train_prices = prices.iloc[:split_idx].reset_index(drop=True)
    test_prices = prices.iloc[split_idx - LONG_WINDOW:].reset_index(drop=True)

    Qs = [train_one_seed(train_prices, seed, n_epochs) for seed in seeds]
    Q_ensemble = np.mean(Qs, axis=0)

    equity, total_ret, sharpe, mdd, t_end, min_hist = evaluate_q(Q_ensemble, test_prices)
    bh_equity = test_prices.iloc[min_hist:t_end + 1].values / test_prices.iloc[min_hist]
    bh_ret = bh_equity[-1] - 1
    bh_returns = np.diff(bh_equity) / bh_equity[:-1]
    bh_sharpe = 0.0 if bh_returns.std() == 0 else (bh_returns.mean() / bh_returns.std()) * np.sqrt(252)

    result = {
        "asset": name,
        "n_train_days": len(train_prices),
        "n_test_days": int(t_end - min_hist),
        "agent_total_return": float(total_ret),
        "agent_sharpe": float(sharpe),
        "agent_max_drawdown": float(mdd),
        "bh_total_return": float(bh_ret),
        "bh_sharpe": float(bh_sharpe),
    }
    np.save(f"agent_equity_{name}.npy", equity)
    np.save(f"bh_equity_{name}.npy", bh_equity)
    np.save(f"q_table_{name}.npy", Q_ensemble)
    return result
