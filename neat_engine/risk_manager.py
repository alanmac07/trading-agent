"""
neat_engine/risk_manager.py

Consumes (action_signal, risk_signal) from NeatTradingAgent and produces
an actual trading decision. This is deliberately NOT evolved by NEAT --
thresholds are fixed/global so every genome is judged on the same rules.
(If you later want the risk appetite itself to evolve, that's a natural
follow-up: expose these constants as extra NEAT outputs.)
"""

from __future__ import annotations

from dataclasses import dataclass

BUY_THRESHOLD = 0.25
SELL_THRESHOLD = -0.25

MIN_POSITION_SIZE_PCT = 0.30
MAX_POSITION_SIZE_PCT = 0.90

MIN_TRAILING_STOP_PCT = 4.0
MAX_TRAILING_STOP_PCT = 12.0


@dataclass
class TradeDecision:
    action: str               # "BUY" | "SELL" | "HOLD"
    position_size_pct: float  # fraction of available cash to deploy on BUY
    trailing_stop_pct: float  # % drawdown from peak that forces an exit


def decide_trade(action_signal: float, risk_signal: float, in_position: bool) -> TradeDecision:
    aggressiveness = (max(-1.0, min(1.0, risk_signal)) + 1.0) / 2.0  # -> [0, 1]
    position_size_pct = MIN_POSITION_SIZE_PCT + aggressiveness * (
        MAX_POSITION_SIZE_PCT - MIN_POSITION_SIZE_PCT
    )
    trailing_stop_pct = MIN_TRAILING_STOP_PCT + aggressiveness * (
        MAX_TRAILING_STOP_PCT - MIN_TRAILING_STOP_PCT
    )

    if action_signal >= BUY_THRESHOLD and not in_position:
        action = "BUY"
    elif action_signal <= SELL_THRESHOLD and in_position:
        action = "SELL"
    else:
        action = "HOLD"

    return TradeDecision(action, position_size_pct, trailing_stop_pct)
