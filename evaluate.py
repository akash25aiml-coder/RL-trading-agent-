"""
evaluate.py (v4 -- evaluates the ENSEMBLE production Q-table)
================================================================
Same rigorous checks as before (CAGR, Calmar, cost sensitivity, MA
baseline, per-year breakdown, bootstrap CI), now run against the
ensemble Q-table (mean of 5 independently-trained, cost-aware,
full-coverage agents) instead of a single seed's Q-table.
"""

import numpy as np
import pandas as pd
from trading_env import TradingEnv, ACTION_NAMES, HOLD, BUY, SELL, FLAT

SHORT_WINDOW = 5
LONG_WINDOW = 50
EPISODE_LENGTH = 100
TRANSACTION_COST_SCENARIOS = [0.0, 0.0005, 0.001, 0.002]

Q = np.load("q_table.npy")  # ensemble (production)
test_prices = pd.read_pickle("test_prices.pkl")
test_dates = pd.read_pickle("test_dates.pkl")

print(f"Test set size: {len(test_prices)} days")
print(f"Evaluating ENSEMBLE Q-table (mean of 5 seeds, cost-aware, full-coverage training)\n")


def run_agent_backtest(transaction_cost: float):
    env = TradingEnv(test_prices, short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
                      episode_length=EPISODE_LENGTH, transaction_cost=transaction_cost, seed=123)
    n_full_windows = (len(test_prices) - env.min_history - 1) // EPISODE_LENGTH
    equity = [1.0]
    actions_log = []
    current_value = 1.0
    t = env.min_history
    for _ in range(n_full_windows):
        env._start_idx, env._t = t, t
        env.position, env.entry_price, env.portfolio_value = FLAT, None, current_value
        for _ in range(EPISODE_LENGTH):
            state = env._get_state()
            action = int(np.argmax(Q[state]))
            _, reward, terminated, truncated, info = env.step(action)
            equity.append(info["portfolio_value"])
            actions_log.append(action)
            if terminated or truncated:
                break
        current_value = env.portfolio_value
        t = env._t
    return np.array(equity), actions_log, t, env.min_history


def run_ma_crossover_baseline(transaction_cost: float):
    env = TradingEnv(test_prices, short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
                      episode_length=EPISODE_LENGTH, transaction_cost=transaction_cost, seed=123)
    n_full_windows = (len(test_prices) - env.min_history - 1) // EPISODE_LENGTH
    equity = [1.0]
    current_value = 1.0
    t = env.min_history
    for _ in range(n_full_windows):
        env._start_idx, env._t = t, t
        env.position, env.entry_price, env.portfolio_value = FLAT, None, current_value
        for _ in range(EPISODE_LENGTH):
            short_ret = env._short_returns[env._t]
            long_ret = env._long_returns[env._t]
            if short_ret > 0 and long_ret > 0.05:
                action = BUY
            elif short_ret < 0 and long_ret < -0.05:
                action = SELL
            else:
                action = HOLD
            _, reward, terminated, truncated, info = env.step(action)
            equity.append(info["portfolio_value"])
            if terminated or truncated:
                break
        current_value = env.portfolio_value
        t = env._t
    return np.array(equity)


def max_drawdown(equity):
    running_max = np.maximum.accumulate(equity)
    return ((equity - running_max) / running_max).min()


def sharpe_ratio(equity, periods_per_year=252, risk_free_annual=0.0):
    returns = np.diff(equity) / equity[:-1]
    rf_daily = risk_free_annual / periods_per_year
    excess = returns - rf_daily
    return 0.0 if excess.std() == 0 else (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def cagr(equity, periods_per_year=252):
    n_periods = len(equity) - 1
    years = n_periods / periods_per_year
    return equity[-1] ** (1 / years) - 1 if years > 0 else 0.0


def calmar_ratio(equity, periods_per_year=252):
    mdd = abs(max_drawdown(equity))
    return cagr(equity, periods_per_year) / mdd if mdd > 0 else np.nan


def bootstrap_return_ci(equity, n_boot=2000, seed=0):
    returns = np.diff(equity) / equity[:-1]
    n = len(returns)
    block_size = 20
    rng = np.random.default_rng(seed)
    boot_totals = []
    for _ in range(n_boot):
        n_blocks = n // block_size + 1
        starts = rng.integers(0, n - block_size, size=n_blocks)
        sample = np.concatenate([returns[s:s + block_size] for s in starts])[:n]
        boot_totals.append(np.prod(1 + sample) - 1)
    boot_totals = np.array(boot_totals)
    return np.percentile(boot_totals, [2.5, 97.5]), boot_totals.mean()


# --------------------------------------------------------------------------- #
# 1. Headline backtest at realistic cost (0.05%) -- this is now the honest
#    headline, not the 0%-cost number.
# --------------------------------------------------------------------------- #
REALISTIC_COST = 0.0005
agent_equity, actions_log, t_end, min_hist = run_agent_backtest(transaction_cost=REALISTIC_COST)
bh_equity = test_prices.iloc[min_hist:t_end + 1].values / test_prices.iloc[min_hist]

print("=" * 72)
print(f"{'Metric':<26s}{'Ensemble Agent (0.05% cost)':>26s}{'Buy & Hold':>20s}")
print("=" * 72)
print(f"{'Total Return':<26s}{agent_equity[-1]-1:>+25.2%} {bh_equity[-1]-1:>+19.2%}")
print(f"{'CAGR':<26s}{cagr(agent_equity):>+25.2%} {cagr(bh_equity):>+19.2%}")
print(f"{'Sharpe Ratio':<26s}{sharpe_ratio(agent_equity):>26.3f}{sharpe_ratio(bh_equity):>20.3f}")
print(f"{'Max Drawdown':<26s}{max_drawdown(agent_equity):>25.2%} {max_drawdown(bh_equity):>19.2%}")
print(f"{'Calmar Ratio':<26s}{calmar_ratio(agent_equity):>26.3f}{calmar_ratio(bh_equity):>20.3f}")
print("=" * 72)

# --------------------------------------------------------------------------- #
# 2. Transaction cost sensitivity for the ENSEMBLE
# --------------------------------------------------------------------------- #
print("\n--- Transaction Cost Sensitivity (Ensemble Agent) ---")
print(f"{'Cost per trade':<18s}{'Total Return':>15s}{'CAGR':>12s}{'Sharpe':>10s}")
for cost in TRANSACTION_COST_SCENARIOS:
    eq, _, _, _ = run_agent_backtest(transaction_cost=cost)
    print(f"{cost:>16.2%}  {eq[-1]-1:>+13.2%} {cagr(eq):>+11.2%} {sharpe_ratio(eq):>9.3f}")

# --------------------------------------------------------------------------- #
# 3. MA-crossover baseline at the SAME realistic cost
# --------------------------------------------------------------------------- #
ma_equity = run_ma_crossover_baseline(transaction_cost=REALISTIC_COST)
print(f"\n--- Learned Ensemble vs. Rule-Based Baseline (both at {REALISTIC_COST:.2%} cost) ---")
print(f"{'Strategy':<28s}{'Total Return':>15s}{'Sharpe':>10s}{'Max DD':>10s}")
print(f"{'Q-Learning Ensemble':<28s}{agent_equity[-1]-1:>+14.2%} "
      f"{sharpe_ratio(agent_equity):>9.3f} {max_drawdown(agent_equity):>9.2%}")
print(f"{'MA Crossover (fixed rule)':<28s}{ma_equity[-1]-1:>+14.2%} "
      f"{sharpe_ratio(ma_equity):>9.3f} {max_drawdown(ma_equity):>9.2%}")

# --------------------------------------------------------------------------- #
# 4. Per-year breakdown
# --------------------------------------------------------------------------- #
print("\n--- Per-Year Return Breakdown (calendar years, 0.05% cost) ---")
eval_dates = test_dates.iloc[min_hist:t_end + 1].reset_index(drop=True)
eval_years = eval_dates.dt.year.values
print(f"{'Year':<10s}{'Agent Return':>15s}{'Buy&Hold Return':>18s}{'Trading Days':>15s}")
for year in sorted(set(eval_years)):
    idx = np.where(eval_years == year)[0]
    lo, hi = idx[0], idx[-1]
    a_ret = agent_equity[hi] / agent_equity[lo] - 1
    b_ret = bh_equity[hi] / bh_equity[lo] - 1
    print(f"{year:<10d}{a_ret:>+14.2%} {b_ret:>+17.2%} {len(idx):>15d}")

# --------------------------------------------------------------------------- #
# 5. Bootstrap confidence interval on the ensemble result
# --------------------------------------------------------------------------- #
(ci_lo, ci_hi), boot_mean = bootstrap_return_ci(agent_equity)
print(f"\n--- Bootstrap 95% CI on Ensemble Total Return (block bootstrap, n=2000) ---")
print(f"Point estimate: {agent_equity[-1]-1:+.2%}  |  "
      f"95% CI: [{ci_lo:+.2%}, {ci_hi:+.2%}]  |  Bootstrap mean: {boot_mean:+.2%}")
if ci_lo < 0 < ci_hi:
    print("⚠️  CI spans zero -- cannot reject the null hypothesis that true edge is zero.")
else:
    print("✅ CI excludes zero -- result unlikely to be pure noise (at this sample size).")

# --------------------------------------------------------------------------- #
np.save("agent_equity.npy", agent_equity)
np.save("bh_equity.npy", bh_equity)
np.save("ma_equity.npy", ma_equity)
print("\n✅ Ensemble evaluation complete.")
