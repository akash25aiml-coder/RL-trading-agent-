"""
make_plots.py (v4 -- reflects ensemble production model)
==========================================================
Generates all figures used in the README:
  1. Test-set equity curve: ensemble agent vs Buy & Hold vs MA baseline
  2. Q-table heatmap (ensemble policy)

Run after train.py and evaluate.py (which save the .npy files this reads).
"""

import numpy as np
import matplotlib.pyplot as plt
from trading_env import TradingEnv, N_STATES, ACTION_NAMES

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 10

# --------------------------------------------------------------------------- #
# 1. Equity curve: ensemble agent vs Buy & Hold vs MA baseline
# --------------------------------------------------------------------------- #
agent_equity = np.load("agent_equity.npy")
bh_equity = np.load("bh_equity.npy")
ma_equity = np.load("ma_equity.npy")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(agent_equity, label="Q-Learning Ensemble (0.05% cost)", color="darkorange", linewidth=1.8)
ax.plot(bh_equity, label="Buy & Hold (AAPL)", color="steelblue", linewidth=1.8)
ax.plot(ma_equity, label="MA Crossover baseline (0.05% cost)", color="gray", linewidth=1.3, linestyle="--")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("Trading day (test period: 2018-2024)")
ax.set_ylabel("Portfolio value (normalized, start = 1.0)")
ax.set_title("Out-of-Sample Backtest: Ensemble Agent vs Baselines")
ax.legend()
fig.tight_layout()
fig.savefig("assets/equity_curve.png")
plt.close(fig)
print("Saved assets/equity_curve.png")

# --------------------------------------------------------------------------- #
# 2. Q-table heatmap (ensemble)
# --------------------------------------------------------------------------- #
Q = np.load("q_table.npy")
labels = [" | ".join(TradingEnv.decode_state(s)) for s in range(N_STATES)]

fig, ax = plt.subplots(figsize=(6, 10))
im = ax.imshow(Q, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(3))
ax.set_xticklabels([ACTION_NAMES[a] for a in range(3)])
ax.set_yticks(range(N_STATES))
ax.set_yticklabels(labels, fontsize=6)
ax.set_title("Ensemble Q-Values by State")
fig.colorbar(im, ax=ax, label="Q-value", shrink=0.6)
fig.tight_layout()
fig.savefig("assets/q_heatmap.png")
plt.close(fig)
print("Saved assets/q_heatmap.png")

print("\n✅ All plots generated.")

