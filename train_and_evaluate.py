"""
STEP 3: This is "training" for a neuroevolution agent.

There is no separate backprop/epoch loop -- for this system, TRAINING
*is* running the clone/mutate/kill evolution loop over the training
data, long enough for good genomes (network weights) to emerge and
bad ones to get filtered out.

What this script does:
  1. Runs the full evolutionary loop over TRAIN data for one ticker
  2. Picks the single best-performing surviving agent's genome
  3. Saves that genome to disk (the "trained model")
  4. Loads it fresh and runs it against TEST data it has NEVER seen
     -- this is the real, honest measure of whether it generalizes,
     not just memorized one historical run

Run: python train_and_evaluate.py
"""

import pickle
import pandas as pd

from evolution import EvolutionEngine
from neuro_agent import NeuroAgent

TICKER = "AAPL"              # change this to train on a different market
STARTING_CAPITAL = 10_000
PROFIT_GOAL_PCT = 6
LOSS_LIMIT_PCT = 8
CHECKPOINT_DAYS = 15
MAX_AGENTS = 12
SAVE_PATH = f"best_genome_{TICKER}.pkl"


def train():
    train_df = pd.read_csv("data/train.csv", parse_dates=["Date"])
    ticker_df = train_df[train_df["Ticker"] == TICKER].reset_index(drop=True)

    print(f"Training on {len(ticker_df)} real {TICKER} rows...")
    engine = EvolutionEngine(
        data=ticker_df,
        starting_capital=STARTING_CAPITAL,
        profit_goal_pct=PROFIT_GOAL_PCT,
        loss_limit_pct=LOSS_LIMIT_PCT,
        checkpoint_days=CHECKPOINT_DAYS,
        max_agents=MAX_AGENTS,
        agent_class=NeuroAgent,
    )

    # Run exactly one pass over the real training data (no synthetic extension needed for training)
    for i, _ in enumerate(engine.run_live()):
        if i >= len(ticker_df) - 2:
            break

    last_price = float(ticker_df["Close"].iloc[-1])
    all_agents = list(engine.agents.values())
    best_agent = max(all_agents, key=lambda a: a.return_pct(last_price))

    print(f"\nTraining complete. {len(all_agents)} agents ever created, "
          f"{sum(1 for a in all_agents if a.status == 'alive')} alive at end.")
    print(f"Best agent: {best_agent.id} (gen {best_agent.generation}), "
          f"training return: {best_agent.return_pct(last_price):+.1f}%")

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(best_agent.genome, f)
    print(f"Saved best genome to {SAVE_PATH}")

    return best_agent.genome


def evaluate(genome):
    test_df = pd.read_csv("data/test.csv", parse_dates=["Date"])
    ticker_df = test_df[test_df["Ticker"] == TICKER].reset_index(drop=True)

    print(f"\nEvaluating on {len(ticker_df)} HELD-OUT {TICKER} rows (never seen during training)...")
    agent = NeuroAgent("EvalAgent", STARTING_CAPITAL, genome=genome)

    from evolution import EvolutionEngine as _EE
    dummy_engine = _EE.__new__(_EE)  # just to reuse _trade_one_day without full init
    for _, row in ticker_df.iterrows():
        _EE._trade_one_day(dummy_engine, agent, row)

    final_price = float(ticker_df["Close"].iloc[-1])
    ret = agent.return_pct(final_price)
    print(f"\nHELD-OUT TEST RETURN: {ret:+.2f}%")
    print(f"(Buy-and-hold benchmark over same period: "
          f"{(final_price / ticker_df['Close'].iloc[0] - 1) * 100:+.2f}%)")
    return ret


if __name__ == "__main__":
    genome = train()
    evaluate(genome)
