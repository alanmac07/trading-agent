"""
neat_engine/network_agent.py

The NEAT replacement for agent.py's rule-based decide() / neuro_agent.py's
fixed-topology MLP. One instance wraps one genome's compiled network.

This class does NOT decide BUY/SELL/HOLD itself -- it only predicts raw
(action_signal, risk_signal) in [-1, 1]. Turning those into an actual
trade + position size is risk_manager.py's job. Keeps "network predicts"
and "engine executes" separate, same modular boundary the original
prompt asked for.
"""

from __future__ import annotations

import neat

from .features import MARKET_FEATURE_COLUMNS, portfolio_state_vector


class NeatTradingAgent:
    def __init__(self, genome, config: neat.Config):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def decide(self, row, in_position: bool, unrealized_pnl_pct: float) -> tuple[float, float]:
        """
        row: a pandas Series / dict-like with MARKET_FEATURE_COLUMNS already computed.
        Returns (action_signal, risk_signal), both in [-1, 1] (tanh outputs).
        """
        market_vec = [float(row[c]) for c in MARKET_FEATURE_COLUMNS]
        portfolio_vec = portfolio_state_vector(in_position, unrealized_pnl_pct)
        inputs = market_vec + portfolio_vec
        action_signal, risk_signal = self.net.activate(inputs)
        return action_signal, risk_signal
