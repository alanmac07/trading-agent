"""
neat_engine/evaluate_holdout.py

Run AFTER training:
  python -m neat_engine.evaluate_holdout --market stocks
  python -m neat_engine.evaluate_holdout --market crypto
  python -m neat_engine.evaluate_holdout --market macro
  python -m neat_engine.evaluate_holdout  (universal)

Loads results/best_genome_{market}.pkl (with fallback to results/best_genome.pkl)
and tests on data/test_{market}.csv.
"""

from __future__ import annotations

import argparse
import os
import pickle

import neat
import pandas as pd

from .episodes import Episode
from .fitness import _episode_metrics, STARTING_CAPITAL
from .network_agent import NeatTradingAgent
from .risk_manager import decide_trade

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")


def evaluate_on_ticker(agent: NeatTradingAgent, df: pd.DataFrame) -> dict:
    episode = Episode(ticker=df["Ticker"].iloc[0], df=df.reset_index(drop=True))
    capital = STARTING_CAPITAL
    position = 0.0
    entry_price = 0.0
    peak_price = 0.0
    trade_count = 0
    portfolio_history = [capital]

    for _, row in episode.df.iterrows():
        price = float(row["Close"])
        in_position = position > 0
        unrealized_pnl_pct = ((price / entry_price) - 1.0) * 100.0 if in_position else 0.0
        action_signal, risk_signal = agent.decide(row, in_position, unrealized_pnl_pct)
        decision = decide_trade(action_signal, risk_signal, in_position)

        forced_exit = False
        if in_position:
            peak_price = max(peak_price, price)
            drop_from_peak_pct = (peak_price - price) / peak_price * 100.0
            forced_exit = drop_from_peak_pct >= decision.trailing_stop_pct

        if forced_exit or decision.action == "SELL":
            capital += position * price
            position = 0.0
            entry_price = peak_price = 0.0
            trade_count += 1
        elif decision.action == "BUY" and capital > 0:
            invest = capital * decision.position_size_pct
            position = invest / price
            capital -= invest
            entry_price = peak_price = price
            trade_count += 1

        portfolio_history.append(capital + position * price)

    total_return_pct, sharpe, max_drawdown_pct = _episode_metrics(portfolio_history)
    return {
        "ticker": episode.ticker,
        "total_return_pct": round(total_return_pct, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "trade_count": trade_count,
        "final_value": round(portfolio_history[-1], 2),
    }


def resolve_paths(market: str | None = None, genome_override: str | None = None, test_csv_override: str | None = None):
    # Resolve test CSV
    if test_csv_override:
        test_csv = test_csv_override
    elif market and os.path.exists(f"data/test_{market}.csv"):
        test_csv = f"data/test_{market}.csv"
    else:
        test_csv = "data/test_extended.csv"

    # Resolve Genome
    if genome_override:
        genome_path = genome_override
    elif market and os.path.exists(f"results/best_genome_{market}.pkl"):
        genome_path = f"results/best_genome_{market}.pkl"
    elif os.path.exists("results/best_genome.pkl"):
        genome_path = "results/best_genome.pkl"
    elif os.path.exists("best_genome_AAPL.pkl"):
        genome_path = "best_genome_AAPL.pkl"
    else:
        raise FileNotFoundError("No trained genome found. Run neat_engine.trainer first.")

    return test_csv, genome_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained NEAT models on holdout test data.")
    parser.add_argument("--market", type=str, default=None, choices=["stocks", "crypto", "macro", None], help="Market regime")
    parser.add_argument("--genome", type=str, default=None, help="Explicit path to genome pickle file")
    parser.add_argument("--test-csv", type=str, default=None, help="Explicit path to test CSV file")
    args = parser.parse_args()

    market_str = args.market.lower() if args.market else None
    test_csv, genome_path = resolve_paths(market_str, args.genome, args.test_csv)

    print(f"=== Holdout Evaluation ===")
    print(f"Market Regime:   {market_str.upper() if market_str else 'UNIVERSAL'}")
    print(f"Loaded Genome:   {genome_path}")
    print(f"Test Dataset:    {test_csv}")
    print(f"==========================\n")

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        CONFIG_PATH,
    )
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    agent = NeatTradingAgent(genome, config)

    test_df = pd.read_csv(test_csv, parse_dates=["Date"])
    print(f"{'Ticker':<10} {'Return %':>10} {'Sharpe':>8} {'MaxDD %':>9} {'Trades':>7} {'FinalVal':>10}")
    print("-" * 60)
    for ticker, group in test_df.groupby("Ticker"):
        result = evaluate_on_ticker(agent, group.sort_values("Date"))
        print(f"{result['ticker']:<10} {result['total_return_pct']:>10} {result['sharpe_ratio']:>8} "
              f"{result['max_drawdown_pct']:>9} {result['trade_count']:>7} {result['final_value']:>10}")


if __name__ == "__main__":
    main()
