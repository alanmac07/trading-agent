"""
Quick sanity test using FAKE price data (no internet needed).
This just proves the Agent + EvolutionEngine logic actually works
before we wire up the real yfinance data and the Streamlit UI.

Run: python test_with_synthetic_data.py
"""

import numpy as np
import pandas as pd
from evolution import EvolutionEngine

np.random.seed(42)

# Generate 300 days of a fake random-walk price series
n_days = 300
returns = np.random.normal(0.0005, 0.02, n_days)
prices = 100 * np.cumprod(1 + returns)
dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

df = pd.DataFrame({"Date": dates, "Close": prices})

# Fake RSI: just oscillate it synthetically so buy/sell logic gets exercised
df["RSI"] = 50 + 30 * np.sin(np.linspace(0, 20, n_days)) + np.random.normal(0, 5, n_days)
df["RSI"] = df["RSI"].clip(0, 100)

engine = EvolutionEngine(
    data=df,
    starting_capital=10000,
    profit_goal_pct=8,
    loss_limit_pct=10,
    clone_capital_fraction=0.5,
    checkpoint_days=20,
    max_agents=12,
)
for i, _ in enumerate(engine.run_live()):
    if i > 400:  # cap the test run -- run_live() is infinite by design now
        break

print(f"\nTotal agents ever created: {len(engine.agents)}")
print(f"Currently alive: {sum(1 for a in engine.agents.values() if a.status == 'alive')}")
print(f"Dead: {len(engine.graveyard)}")
print(f"\n--- Events ---")
for e in engine.events:
    if e["type"] == "CLONE":
        print(f"Day {e['day_index']}: {e['parent']} (return {e['parent_return_pct']}%) -> cloned -> {e['child']}")
    else:
        print(f"Day {e['day_index']}: {e['agent']} KILLED -- {e['cause']}")

print(f"\n--- Final agent states ---")
for agent in engine.agents.values():
    last_price = prices[-1]
    print(
        f"{agent.id} (gen {agent.generation}, status={agent.status}): "
        f"return={agent.return_pct(last_price):.1f}%, decisions logged={len(agent.decision_log)}"
    )
