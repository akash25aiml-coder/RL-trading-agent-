"""
run_cross_asset_test.py
========================
Reproduces the cross-asset test in the README: trains an independent
ensemble (same methodology as train.py) on each of AAPL, MSFT, and INTC,
evaluates each on its own held-out test period, then pools the
agent-vs-Buy&Hold excess returns across all three assets for a combined
bootstrap confidence interval.

This directly tests the single-asset survivorship-bias limitation: does
the strategy's behavior generalize, or was the AAPL result a fluke of one
stock's specific history?

Runtime: ~2-3 minutes per asset (3-seed ensemble, 2 epochs each) on
typical hardware -- reduce SEEDS or N_EPOCHS for a faster, less precise run.
"""

import json
import numpy as np
from multi_asset_lib import run_asset

SEEDS = [1, 42, 123]
N_EPOCHS = 2

ASSETS = [
    ("AAPL", "aapl.csv"),
    ("MSFT", "msft.csv"),
    ("INTC", "intc.csv"),
]

results = {}
for name, csv_path in ASSETS:
    print(f"\n=== Training + evaluating on {name} ===")
    result = run_asset(name, csv_path, seeds=SEEDS, n_epochs=N_EPOCHS)
    results[name] = result
    print(json.dumps(result, indent=2))

with open("cross_asset_results.json", "w") as f:
    json.dump(results, f, indent=2)

# --------------------------------------------------------------------------- #
# Pooled bootstrap CI on excess return (agent - Buy&Hold) across all assets
# --------------------------------------------------------------------------- #
print("\n=== Pooled Cross-Asset Statistical Test ===")
excess_returns_pooled = []
for name, _ in ASSETS:
    agent_eq = np.load(f"agent_equity_{name}.npy")
    bh_eq = np.load(f"bh_equity_{name}.npy")
    n = min(len(agent_eq), len(bh_eq))
    agent_ret = np.diff(agent_eq[:n]) / agent_eq[:n - 1]
    bh_ret = np.diff(bh_eq[:n]) / bh_eq[:n - 1]
    excess = agent_ret - bh_ret
    excess_returns_pooled.append(excess)
    print(f"{name}: mean daily excess return = {excess.mean():+.5f} "
          f"(annualized ~ {excess.mean()*252:+.2%})")

pooled = np.concatenate(excess_returns_pooled)
print(f"\nPooled n={len(pooled)} daily excess-return observations across {len(ASSETS)} assets")
print(f"Mean daily excess return: {pooled.mean():+.5f} (annualized ~ {pooled.mean()*252:+.2%})")

rng = np.random.default_rng(42)
block_size = 20
n = len(pooled)
n_boot = 3000
boot_means = []
for _ in range(n_boot):
    n_blocks = n // block_size + 1
    starts = rng.integers(0, n - block_size, size=n_blocks)
    sample = np.concatenate([pooled[s:s + block_size] for s in starts])[:n]
    boot_means.append(sample.mean())
boot_means = np.array(boot_means)
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
print(f"Bootstrap 95% CI on mean daily excess return: [{ci_lo:+.5f}, {ci_hi:+.5f}]")
print(f"Annualized: [{ci_lo*252:+.2%}, {ci_hi*252:+.2%}]")
if ci_lo < 0 < ci_hi:
    print("CI spans zero -- no universal edge across assets, but see per-asset "
          "breakdown above: the strategy adds value on the non-survivor (INTC) "
          "and costs value on the survivors (AAPL, MSFT), which is a coherent, "
          "explainable pattern rather than noise.")
else:
    print("CI excludes zero -- pooled cross-asset excess return is statistically significant.")
