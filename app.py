"""
app.py -- Streamlit front end for the regime-aware Q-learning trading agent.

Run:  streamlit run app.py

The app does NOT train anything and needs no API keys. It loads the frozen
Q-tables in models/ (trained by train.py / multi_asset_lib.py) and replays the
greedy policy on each stock's held-out test period using the bundled
historical CSVs, so results are reproducible and match the project report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from multi_asset_lib import EPISODE_LENGTH, LONG_WINDOW, SHORT_WINDOW, load_prices
from trading_env import (ACTION_NAMES, FLAT, LONG, N_POSITIONS, N_REGIMES,
                         N_STATES, POSITION_NAMES, REGIME_NAMES, SHORT,
                         SHORT_BUCKET_NAMES, TradingEnv)

ROOT = Path(__file__).parent
TICKERS = {
    "AAPL": ("aapl.csv", "Strong secular winner"),
    "MSFT": ("msft.csv", "Strong secular winner"),
    "INTC": ("intc.csv", "Declined over the test period"),
}

st.set_page_config(page_title="RL trading agent", layout="wide")


@st.cache_data
def load_asset(ticker):
    prices, dates = load_prices(ROOT / TICKERS[ticker][0])
    split = int(len(prices) * 0.85)  # same chronological 85/15 split used in training
    test_prices = prices.iloc[split - LONG_WINDOW:].reset_index(drop=True)
    test_dates = dates.iloc[split - LONG_WINDOW:].reset_index(drop=True)
    return test_prices, test_dates, np.load(ROOT / "models" / f"{ticker}.npy")


def replay(Q, prices, cost):
    """Greedy replay over back-to-back 100-day windows (same harness as
    evaluate.py). Also records the position held each day and counts trades."""
    env = TradingEnv(prices, short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
                     episode_length=EPISODE_LENGTH, transaction_cost=cost)
    n_windows = (len(prices) - env.min_history - 1) // EPISODE_LENGTH
    equity, positions, trades = [1.0], [], 0
    value, t = 1.0, env.min_history
    for _ in range(n_windows):
        env._start_idx, env._t = t, t
        env.position, env.entry_price, env.portfolio_value = FLAT, None, value
        for _ in range(EPISODE_LENGTH):
            before = env.position
            _, _, done, trunc, info = env.step(int(np.argmax(Q[env._get_state()])))
            trades += env.position != before
            equity.append(info["portfolio_value"])
            positions.append(env.position)
            if done or trunc:
                break
        value, t = env.portfolio_value, env._t
    return np.array(equity), np.array(positions), int(trades), t, env.min_history


def sharpe(eq):
    r = np.diff(eq) / eq[:-1]
    return 0.0 if r.std() == 0 else float(r.mean() / r.std() * np.sqrt(252))


def max_drawdown(eq):
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


@st.cache_data
def backtest(ticker, cost):
    prices, dates, Q = load_asset(ticker)
    equity, pos, trades, t_end, min_hist = replay(Q, prices, cost)
    bh = prices.iloc[min_hist:t_end + 1].values / prices.iloc[min_hist]
    idx = dates.iloc[min_hist:t_end + 1].dt.tz_convert(None)
    return dict(equity=equity, bh=bh, idx=idx, pos=pos, trades=trades,
                ret=equity[-1] - 1, bh_ret=bh[-1] - 1, sharpe=sharpe(equity),
                bh_sharpe=sharpe(bh), mdd=max_drawdown(equity), Q=Q,
                data_end=dates.iloc[-1].date())


st.title("Regime-aware Q-learning trading agent")
st.caption("Replays a trained agent on historical data it never saw during training. "
           "Educational project, not investment advice.")

with st.sidebar:
    ticker = st.selectbox("Stock", list(TICKERS), help="Each stock has its own trained agent.")
    st.write(TICKERS[ticker][1])
    cost_pct = st.slider("Transaction cost per trade (%)", 0.00, 0.30, 0.05, 0.01)

res = backtest(ticker, cost_pct / 100)
tab_bt, tab_all, tab_policy, tab_about = st.tabs(
    ["Backtest", "All stocks", "Learned policy", "About and limits"])

with tab_bt:
    a, b, c, d = st.columns(4)
    a.metric("Agent return", f"{res['ret']:+.1%}", f"{res['ret'] - res['bh_ret']:+.1%} vs Buy & Hold")
    b.metric("Buy & Hold return", f"{res['bh_ret']:+.1%}")
    c.metric("Agent Sharpe", f"{res['sharpe']:.2f}", f"{res['sharpe'] - res['bh_sharpe']:+.2f} vs Buy & Hold")
    d.metric("Agent max drawdown", f"{res['mdd']:.1%}")
    e, f, g, h = st.columns(4)
    pos = res["pos"]
    e.metric("Trades", res["trades"])
    f.metric("Days long", f"{(pos == LONG).mean():.0%}")
    g.metric("Days short", f"{(pos == SHORT).mean():.0%}")
    h.metric("Days flat", f"{(pos == FLAT).mean():.0%}")

    st.line_chart(pd.DataFrame({"Agent": res["equity"], "Buy & Hold": res["bh"]}, index=res["idx"]),
                  color=["#E8871E", "#4682B4"])

    fig, ax = plt.subplots(figsize=(10, 2.8))
    x = res["idx"].iloc[1:]
    ax.plot(x, res["bh"][1:], color="#4682B4", linewidth=1.2)
    ax.fill_between(x, 0, 1, where=pos == LONG, transform=ax.get_xaxis_transform(),
                    color="#2E8B57", alpha=0.2, linewidth=0, label="Agent long")
    ax.fill_between(x, 0, 1, where=pos == SHORT, transform=ax.get_xaxis_transform(),
                    color="#C0392B", alpha=0.25, linewidth=0, label="Agent short")
    ax.set_ylabel("Price (start = 1)")
    ax.legend(loc="upper left", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    st.download_button("Download daily results (CSV)",
                       pd.DataFrame({"agent": res["equity"], "buy_and_hold": res["bh"]},
                                    index=res["idx"]).to_csv(),
                       file_name=f"{ticker}_backtest.csv")
    st.caption(f"Test period {res['idx'].iloc[0].date()} to {res['idx'].iloc[-1].date()}. "
               "Green and red shading show when the agent held a long or short position.")

with tab_all:
    rows = []
    for t in TICKERS:
        r = backtest(t, cost_pct / 100)
        rows.append({"Stock": t, "Type": TICKERS[t][1],
                     "Agent return": f"{r['ret']:+.1%}", "Buy & Hold": f"{r['bh_ret']:+.1%}",
                     "Difference": f"{r['ret'] - r['bh_ret']:+.1%}",
                     "Agent Sharpe": f"{r['sharpe']:.2f}", "B&H Sharpe": f"{r['bh_sharpe']:.2f}"})
    st.dataframe(pd.DataFrame(rows).set_index("Stock"))
    st.write("The agent trails Buy & Hold on stocks that kept rising and beats it on the one that fell. "
             "That is the project's main finding: a conditional edge, not a way to beat the market.")
    st.caption("AAPL uses a 5-seed ensemble; MSFT and INTC use 3-seed ensembles.")

with tab_policy:
    st.write("Pick a market situation and see what the agent learned to do.")
    a, b, c = st.columns(3)
    short = a.selectbox("5-day trend", SHORT_BUCKET_NAMES, index=3)
    held = b.selectbox("Current position", list(POSITION_NAMES.values()), index=1)
    regime = c.selectbox("50-day regime", REGIME_NAMES, index=2)
    state = ((SHORT_BUCKET_NAMES.index(short) * N_POSITIONS
              + list(POSITION_NAMES.values()).index(held)) * N_REGIMES
             + REGIME_NAMES.index(regime))
    Q = res["Q"]
    st.success(f"Learned action: **{ACTION_NAMES[int(np.argmax(Q[state]))]}**")
    st.bar_chart(pd.Series({ACTION_NAMES[i]: Q[state, i] for i in range(3)}, name="Q-value"))
    with st.expander("Q-values for all 45 states"):
        fig, ax = plt.subplots(figsize=(5, 9))
        im = ax.imshow(Q, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(3), [ACTION_NAMES[i] for i in range(3)])
        ax.set_yticks(range(N_STATES), [" | ".join(TradingEnv.decode_state(i)) for i in range(N_STATES)],
                      fontsize=6)
        fig.colorbar(im, ax=ax, label="Q-value", shrink=0.6)
        st.pyplot(fig)
        plt.close(fig)

with tab_about:
    st.markdown(
        """
**Problem.** Learn when to hold, buy or sell one stock using only the reward (profit or loss).

**Technique.** Tabular Q-learning on a Markov Decision Process: 45 states (5-day trend x position x
50-day regime) and 3 actions. Update rule: `Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a))`.

**What this app does.** It loads the finished Q-table and always takes the action with the highest
Q-value. Nothing is learned here; training lives in `train.py`.

**Limits you should know**
- Data is pre-recorded daily prices (Yahoo Finance CSVs). It ends {end}. This is a backtest, not a live trading tool.
- Only three stocks were tested. Pooled across them, the confidence interval on the agent's edge still includes zero.
- Trading costs are a flat percentage per position change. There is no borrow cost for shorts and no slippage.
- The backtest replays back-to-back 100-day windows and starts each window flat, so a few exits are not charged a cost (under 1% in total at the default 0.05% cost).
- Results at 0% cost are optimistic. Use the cost slider.
""".format(end=res["data_end"])
    )
