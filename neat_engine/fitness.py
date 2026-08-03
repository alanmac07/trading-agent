"""
neat_engine/fitness.py

Simulates one genome through one episode, then scores it on more than
raw return: total return, Sharpe, max drawdown, and a trade-activity
bonus/penalty (so it can't win by doing nothing, and can't win by
overtrading either). Collapsed episodes are terminated early and
heavily penalized instead of running out the full window.

Fitness weights are constants here on purpose -- pull them out to a
config dict if you want to sweep them later.
"""

from __future__ import annotations

import math

import numpy as np

from .episodes import Episode
from .network_agent import NeatTradingAgent
from .risk_manager import decide_trade

STARTING_CAPITAL = 10_000.0
COLLAPSE_THRESHOLD = 0.20 * STARTING_CAPITAL  # terminate early if capital falls below this

RETURN_WEIGHT = 0.40
SHARPE_WEIGHT = 10.0
DRAWDOWN_WEIGHT = 0.10
COLLAPSE_PENALTY = -100.0

MIN_HEALTHY_TRADES = 3
MAX_HEALTHY_TRADES = 60
TRADE_ACTIVITY_BONUS = 3.0
TRADE_ACTIVITY_PENALTY = -5.0


def _episode_metrics(portfolio_history: list[float]) -> tuple[float, float, float]:
    values = np.array(portfolio_history, dtype=float)
    total_return_pct = (values[-1] / values[0] - 1.0) * 100.0

    daily_returns = np.diff(values) / values[:-1]
    if len(daily_returns) > 1 and daily_returns.std() > 1e-9:
        sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)
    else:
        sharpe = 0.0

    running_peak = np.maximum.accumulate(values)
    drawdowns = (values - running_peak) / running_peak
    max_drawdown_pct = abs(drawdowns.min()) * 100.0

    return total_return_pct, sharpe, max_drawdown_pct


def run_episode(agent: NeatTradingAgent, episode: Episode) -> float:
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

        # trailing-stop check happens before the network gets a say --
        # risk management overrides prediction, not the other way round
        forced_exit = False
        if in_position:
            peak_price = max(peak_price, price)
            drop_from_peak_pct = (peak_price - price) / peak_price * 100.0
            action_signal, risk_signal = agent.decide(row, in_position, unrealized_pnl_pct)
            decision = decide_trade(action_signal, risk_signal, in_position)
            if drop_from_peak_pct >= decision.trailing_stop_pct:
                forced_exit = True
        else:
            action_signal, risk_signal = agent.decide(row, in_position, unrealized_pnl_pct)
            decision = decide_trade(action_signal, risk_signal, in_position)

        if forced_exit or decision.action == "SELL":
            capital += position * price
            position = 0.0
            entry_price = 0.0
            peak_price = 0.0
            trade_count += 1
        elif decision.action == "BUY" and capital > 0:
            invest = capital * decision.position_size_pct
            position = invest / price
            capital -= invest
            entry_price = price
            peak_price = price
            trade_count += 1

        portfolio_value = capital + position * price
        portfolio_history.append(portfolio_value)

        if portfolio_value < COLLAPSE_THRESHOLD:
            return COLLAPSE_PENALTY

    total_return_pct, sharpe, max_drawdown_pct = _episode_metrics(portfolio_history)

    if MIN_HEALTHY_TRADES <= trade_count <= MAX_HEALTHY_TRADES:
        activity_term = TRADE_ACTIVITY_BONUS
    else:
        activity_term = TRADE_ACTIVITY_PENALTY

    score = (
        RETURN_WEIGHT * total_return_pct
        + SHARPE_WEIGHT * sharpe
        - DRAWDOWN_WEIGHT * max_drawdown_pct
        + activity_term
    )
    return score


def fitness_for_genome(genome, config, episodes: list[Episode]) -> float:
    agent = NeatTradingAgent(genome, config)
    scores = [run_episode(agent, ep) for ep in episodes]
    return float(np.mean(scores))
