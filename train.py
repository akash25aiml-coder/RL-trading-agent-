"""
train.py (v4 -- production: full-coverage, cost-aware, ensembled)
===================================================================
Three structural fixes, each targeting a specific failure found in the
finance audit (see README "Rigorous Evaluation" section):

FIX 1 -- Seed sensitivity (v2: seed=42 gave anywhere from +1.3% to +421.9%
    depending on which random windows got sampled during training):
    Train in EPOCHS that sweep every valid start-day in the training set,
    not a random subset. Every window gets equal exposure every epoch.

FIX 2 -- Fragile to realistic transaction costs (v2: +13.4% at 0% cost
    became -7.6% at just 0.05% cost/trade):
    Bake a realistic transaction cost directly into the TRAINING reward,
    so the agent learns that switching position has a cost and stops
    reacting to every noise tick.

FIX 3 -- Residual seed variance even after Fix 1 (measured: -7.3% to
    +166.5% cost-adjusted return across 5 seeds with full-coverage
    training -- much tighter than v2, but still not tight enough to trust
    a single run):
    Train N_SEEDS independent agents and average their Q-tables into one
    ENSEMBLE policy. This is standard variance-reduction practice (same
    idea as bagging in supervised learning) -- individual seeds' policy
    errors partially cancel out in the average, producing a materially
    more stable final policy than any single run.

The final artifact saved as q_table.npy is the ENSEMBLE (production)
Q-table. Individual per-seed Q-tables are also saved for transparency /
the robustness analysis in the README.
"""

import numpy as np
import pandas as pd
from trading_env import TradingEnv, ACTION_NAMES, N_STATES, FLAT
from q_agent import QLearningAgent

# --------------------------------------------------------------------------- #
SEEDS = [1, 7, 42, 123, 2024]        # ensemble members
N_EPOCHS = 2                          # full sweeps per seed (validated: converges well, fits time budget)
EPISODE_LENGTH = 100
SHORT_WINDOW = 5
LONG_WINDOW = 50
TRAIN_TRANSACTION_COST = 0.0005       # 0.05% per position change, baked into training

# --------------------------------------------------------------------------- #
df = pd.read_csv("aapl.csv")
df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values("Date")
prices = df["Close"]
dates = df["Date"]

split_idx = int(len(prices) * 0.85)
train_prices = prices.iloc[:split_idx].reset_index(drop=True)
test_prices = prices.iloc[split_idx - LONG_WINDOW:].reset_index(drop=True)
test_dates = dates.iloc[split_idx - LONG_WINDOW:].reset_index(drop=True)

print(f"Train range: {df['Date'].iloc[0].date()} -> {df['Date'].iloc[split_idx].date()} "
      f"({len(train_prices)} days)")
print(f"Test range:  {df['Date'].iloc[split_idx].date()} -> {df['Date'].iloc[-1].date()} "
      f"({len(test_prices)} days)")
print(f"Embedded training transaction cost: {TRAIN_TRANSACTION_COST:.2%} per trade")
print(f"Ensemble size: {len(SEEDS)} seeds x {N_EPOCHS} epochs each\n")


def train_one_seed(seed: int):
    env = TradingEnv(
        train_prices, short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
        episode_length=EPISODE_LENGTH, transaction_cost=TRAIN_TRANSACTION_COST, seed=seed,
    )
    valid_starts = np.arange(env.min_history, len(train_prices) - EPISODE_LENGTH - 1)
    n_episodes_total = N_EPOCHS * len(valid_starts)

    agent = QLearningAgent(
        n_states=env.observation_space.n, n_actions=env.action_space.n,
        alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=n_episodes_total / 5, seed=seed,
    )

    rng = np.random.default_rng(seed)
    global_ep = 0
    for epoch in range(N_EPOCHS):
        for start in rng.permutation(valid_starts):
            agent.start_episode(global_ep)
            env._start_idx, env._t = int(start), int(start)
            env.position, env.entry_price, env.portfolio_value = FLAT, None, 1.0
            state = env._get_state()
            for _ in range(EPISODE_LENGTH):
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, next_state, done=(terminated or truncated))
                state = next_state
                if terminated or truncated:
                    break
            global_ep += 1
    return agent.Q, agent.visit_counts


# --------------------------------------------------------------------------- #
all_Qs, all_visits = [], []
for seed in SEEDS:
    Q, visits = train_one_seed(seed)
    all_Qs.append(Q)
    all_visits.append(visits)
    print(f"  Trained seed={seed}")

Q_ensemble = np.mean(all_Qs, axis=0)
visits_total = np.sum(all_visits, axis=0)

print(f"\n✅ Ensemble training complete ({len(SEEDS)} seeds).")

# --------------------------------------------------------------------------- #
# Save the ENSEMBLE as the production Q-table, plus individual seeds for
# the transparency/robustness section of the report.
np.save("q_table.npy", Q_ensemble)
np.save("q_tables_all_seeds.npy", np.array(all_Qs))
np.save("visit_counts.npy", visits_total)
test_prices.to_pickle("test_prices.pkl")
train_prices.to_pickle("train_prices.pkl")
test_dates.to_pickle("test_dates.pkl")

print("Saved q_table.npy (ensemble, production), q_tables_all_seeds.npy (per-seed, for audit)")

# --------------------------------------------------------------------------- #
print(f"\nEnsemble greedy policy ({N_STATES} states), with total visitation counts:")
for s in range(N_STATES):
    b, p, r = TradingEnv.decode_state(s)
    q_vals = Q_ensemble[s]
    best_a = int(np.argmax(q_vals))
    q_str = ", ".join(f"{v:+.4f}" for v in q_vals)
    print(f"  {b:<10s} | {p:<5s} | {r:<9s} -> "
          f"{ACTION_NAMES[best_a]:<4s}  [Q: {q_str}]  "
          f"(total visits across ensemble: {int(visits_total[s].sum()):>6d})")
