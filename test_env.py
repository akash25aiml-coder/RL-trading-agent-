"""
test_env.py (v2)
Sanity checks for the regime-aware TradingEnv (45 states).
"""

import numpy as np
import pandas as pd
from trading_env import TradingEnv, N_STATES

df = pd.read_csv("aapl.csv")
df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values("Date")
prices = df["Close"]

print(f"Loaded {len(prices)} price points, "
      f"{df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}")
print(f"State space size: {N_STATES}")

# --- basic contract check --- #
env = TradingEnv(prices, seed=42)
obs, info = env.reset(seed=42)
assert env.observation_space.contains(obs)
print(f"\n[reset] state={obs} -> {TradingEnv.decode_state(obs)}")

total_reward = 0.0
for i in range(env.episode_length):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert not np.isnan(reward)
    total_reward += reward
    if terminated or truncated:
        break
print(f"[random policy] {i+1} steps, final value={info['portfolio_value']:.4f}, "
      f"cum reward={total_reward:.4f}")

# --- coverage check --- #
seen = set()
env = TradingEnv(prices, seed=0)
for ep in range(500):
    obs, _ = env.reset()
    seen.add(obs)
    for _ in range(env.episode_length):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        seen.add(obs)
        if terminated or truncated:
            break

print(f"\n[coverage] {len(seen)}/{N_STATES} states visited over 500 random episodes")
missing = set(range(N_STATES)) - seen
if missing:
    print(f"  MISSING: {[TradingEnv.decode_state(s) for s in missing]}")
else:
    print("  All states reachable. Good.")

# --- determinism check --- #
env = TradingEnv(prices, episode_length=10, transaction_cost=0.0, seed=1)
obs, _ = env.reset(seed=1)
start_t = env._t
start_price = env.prices.iloc[start_t]
env.step(1)  # BUY
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(0)
end_price = env.prices.iloc[env._t]
expected = end_price / start_price
actual = info["portfolio_value"]
print(f"\n[determinism] expected={expected:.6f}, actual={actual:.6f}, "
      f"diff={abs(expected-actual):.2e}")
assert abs(expected - actual) < 1e-9

print("\n✅ All v2 environment sanity checks passed.")
