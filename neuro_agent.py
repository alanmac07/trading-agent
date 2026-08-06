"""
NeuroAgent -- same Agent interface as agent.py, but its "genome" is a
small neural network's weights instead of a handful of thresholds.

This is NEUROEVOLUTION: the network is never trained with backprop.
Instead, the exact same clone/mutate/kill mechanism from evolution.py
operates directly on the weight values. A clone's network is the
parent's network with every weight nudged by a small random amount.
Fitness is still real trading profit -- nothing about evolution.py
needs to change to use this, only WHICH agent class it instantiates.

Architecture: 8 inputs -> 10 hidden (tanh) -> 1 output (tanh)
Output near +1  = strong buy signal
Output near -1  = strong sell signal
Output near 0   = no clear signal, hold
"""

from __future__ import annotations

import os
import pickle
import random
import numpy as np
import pandas as pd
import neat

from agent import Agent, AGENT_COLOR_PALETTE, _next_color
from neat_engine.network_agent import NeatTradingAgent
from neat_engine.risk_manager import decide_trade
from neat_engine.features import MARKET_FEATURE_COLUMNS

FEATURE_COLUMNS = [
    "rsi_norm", "price_vs_ma20", "macd", "macd_signal",
    "bb_pct_b", "return_1d", "return_5d", "volume_z",
]

N_INPUTS = len(FEATURE_COLUMNS)   # 8
N_HIDDEN = 10
N_OUTPUTS = 1

# Total number of numbers needed to fully describe the network:
#   W1 (N_INPUTS x N_HIDDEN) + b1 (N_HIDDEN) + W2 (N_HIDDEN x N_OUTPUTS) + b2 (N_OUTPUTS)
GENOME_LENGTH = (N_INPUTS * N_HIDDEN) + N_HIDDEN + (N_HIDDEN * N_OUTPUTS) + N_OUTPUTS


def _unpack(weights: np.ndarray):
    """Slice the flat weight vector back into W1, b1, W2, b2 matrices."""
    i = 0
    W1 = weights[i : i + N_INPUTS * N_HIDDEN].reshape(N_INPUTS, N_HIDDEN); i += N_INPUTS * N_HIDDEN
    b1 = weights[i : i + N_HIDDEN]; i += N_HIDDEN
    W2 = weights[i : i + N_HIDDEN * N_OUTPUTS].reshape(N_HIDDEN, N_OUTPUTS); i += N_HIDDEN * N_OUTPUTS
    b2 = weights[i : i + N_OUTPUTS]; i += N_OUTPUTS
    return W1, b1, W2, b2


def forward(weights: np.ndarray, x: np.ndarray) -> float:
    """One forward pass. x = feature vector (length N_INPUTS). Returns a scalar in [-1, 1]."""
    W1, b1, W2, b2 = _unpack(weights)
    h = np.tanh(x @ W1 + b1)
    out = np.tanh(h @ W2 + b2)
    return float(out[0])


class NeuroAgent(Agent):
    """
    Drop-in replacement for Agent. Same public interface (portfolio_value,
    return_pct, decision_log, portfolio_history, date_history, color, etc.)
    -- only genome creation, mutation, and decide() are different.
    """

    @staticmethod
    def random_genome() -> dict:
        # Small random weights (not huge) -- keeps early behavior from being
        # wildly erratic, same reasoning as standard NN weight initialization.
        weights = np.random.normal(0, 0.3, size=GENOME_LENGTH)
        return {
            "weights": weights,
            "buy_threshold": round(random.uniform(0.15, 0.4), 2),
            "sell_threshold": round(random.uniform(0.15, 0.4), 2),
            "trailing_stop_pct": round(random.uniform(4, 12), 1),
            "position_size_pct": round(random.uniform(0.4, 0.9), 2),
        }

    @staticmethod
    def mutate_genome(genome: dict, mutation_rate: float = 0.1) -> dict:
        new_weights = genome["weights"] + np.random.normal(0, mutation_rate, size=GENOME_LENGTH)
        new_genome = {"weights": new_weights}
        for k, v in genome.items():
            if k == "weights":
                continue
            delta = v * 0.15 * random.uniform(-1, 1)
            new_genome[k] = round(v + delta, 3)
        return new_genome

    def decide(self, row) -> tuple[str, str]:
        x = np.array([float(row[c]) for c in FEATURE_COLUMNS])
        signal = forward(self.genome["weights"], x)
        price = float(row["Close"])

        buy_th = self.genome["buy_threshold"]
        sell_th = self.genome["sell_threshold"]
        trail = self.genome.get("trailing_stop_pct", 8.0)

        if self.position > 0:
            if price > self.price_high_water:
                self.price_high_water = price
            drop_pct = (self.price_high_water - price) / self.price_high_water * 100.0 if self.price_high_water else 0
            if drop_pct >= trail:
                peak = self.price_high_water
                self.price_high_water = 0.0
                return "SELL", f"Trailing stop: -{drop_pct:.1f}% from peak ${peak:.2f} -> cashing out"
            if signal < -sell_th:
                self.price_high_water = 0.0
                return "SELL", f"NN signal={signal:+.2f} < -{sell_th:.2f} -> network says sell"
            return "HOLD", f"NN signal={signal:+.2f} -- holding (peak ${self.price_high_water:.2f})"
        else:
            if signal > buy_th:
                self.price_high_water = price
                return "BUY", f"NN signal={signal:+.2f} > {buy_th:.2f} -> network says buy"
            return "HOLD", f"NN signal={signal:+.2f} <= {buy_th:.2f} -> no clear signal"


class TrainedNeatAgent(Agent):
    """
    Adapter wrapper that loads a pre-trained NEAT champion genome
    from results/best_genome.pkl and neat_engine/config-feedforward.txt,
    running NeatTradingAgent + decide_trade inside the live EvolutionEngine.
    """

    def __init__(
        self,
        agent_id: str = "NEAT_Champion",
        capital: float = 10_000.0,
        genome_path: str = "results/best_genome.pkl",
        config_path: str = "neat_engine/config-feedforward.txt",
        parent_id: str | None = None,
        generation: int = 0,
        color: str = "#00FF88",
    ):
        self.genome_path = genome_path
        self.config_path = config_path

        base_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_cfg = (
            config_path
            if os.path.isabs(config_path)
            else os.path.join(base_dir, config_path)
        )
        if not os.path.exists(resolved_cfg):
            resolved_cfg = os.path.join(base_dir, "neat_engine", "config-feedforward.txt")

        resolved_genome = (
            genome_path
            if os.path.isabs(genome_path)
            else os.path.join(base_dir, genome_path)
        )
        if not os.path.exists(resolved_genome):
            if os.path.exists(os.path.join(base_dir, "results", "best_genome.pkl")):
                resolved_genome = os.path.join(base_dir, "results", "best_genome.pkl")
            elif os.path.exists(os.path.join(base_dir, "best_genome_AAPL.pkl")):
                resolved_genome = os.path.join(base_dir, "best_genome_AAPL.pkl")

        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            resolved_cfg,
        )

        with open(resolved_genome, "rb") as f:
            raw_neat_genome = pickle.load(f)

        self.neat_agent = NeatTradingAgent(raw_neat_genome, self.neat_config)
        self.entry_price: float = 0.0

        genome_dict = {
            "model_type": "NEAT FeedForward",
            "position_size_pct": 0.8,
            "trailing_stop_pct": 8.0,
            "genome_id": getattr(raw_neat_genome, "key", "champion"),
            "nodes_count": len(getattr(raw_neat_genome, "nodes", {})),
            "connections_count": len(getattr(raw_neat_genome, "connections", {})),
        }

        super().__init__(
            agent_id=agent_id,
            capital=capital,
            genome=genome_dict,
            parent_id=parent_id,
            generation=generation,
            color=color,
        )

    @staticmethod
    def random_genome() -> dict:
        return {
            "model_type": "NEAT FeedForward",
            "position_size_pct": 0.8,
            "trailing_stop_pct": 8.0,
        }

    @staticmethod
    def mutate_genome(genome: dict, mutation_rate: float = 0.1) -> dict:
        return dict(genome)

    def decide(self, row) -> tuple[str, str]:
        price = float(row["Close"])
        in_position = self.position > 0
        unrealized_pnl_pct = (
            ((price / self.entry_price) - 1.0) * 100.0
            if in_position and self.entry_price > 0
            else 0.0
        )

        # Build feature dictionary for MARKET_FEATURE_COLUMNS
        row_dict = {}
        for c in MARKET_FEATURE_COLUMNS:
            if c in row and pd.notna(row[c]):
                row_dict[c] = float(row[c])
            elif c == "rsi_norm" and "RSI" in row and pd.notna(row["RSI"]):
                row_dict[c] = (float(row["RSI"]) - 50.0) / 50.0
            else:
                row_dict[c] = 0.0

        action_signal, risk_signal = self.neat_agent.decide(
            row_dict, in_position, unrealized_pnl_pct
        )
        decision = decide_trade(action_signal, risk_signal, in_position)

        # Keep dynamic sizing & trailing stop updated on genome
        self.genome["position_size_pct"] = round(decision.position_size_pct, 2)
        self.genome["trailing_stop_pct"] = round(decision.trailing_stop_pct, 1)

        if in_position:
            if price > self.price_high_water:
                self.price_high_water = price
            drop_pct = (
                (self.price_high_water - price) / self.price_high_water * 100.0
                if self.price_high_water > 0
                else 0.0
            )

            # Trailing stop forced exit
            if drop_pct >= decision.trailing_stop_pct:
                peak = self.price_high_water
                self.price_high_water = 0.0
                self.entry_price = 0.0
                return (
                    "SELL",
                    f"🔔 NEAT Trailing stop: -{drop_pct:.1f}% from peak ${peak:.2f} "
                    f"(limit {decision.trailing_stop_pct:.1f}%) -> cashing out",
                )

            if decision.action == "SELL":
                self.price_high_water = 0.0
                self.entry_price = 0.0
                return (
                    "SELL",
                    f"NEAT signal: action={action_signal:+.2f}, risk={risk_signal:+.2f} -> SELL",
                )

            return (
                "HOLD",
                f"NEAT signal: action={action_signal:+.2f}, risk={risk_signal:+.2f} -> HOLD "
                f"(peak ${self.price_high_water:.2f}, unrlz {unrealized_pnl_pct:+.1f}%)",
            )

        else:
            if decision.action == "BUY":
                self.price_high_water = price
                self.entry_price = price
                return (
                    "BUY",
                    f"NEAT signal: action={action_signal:+.2f}, risk={risk_signal:+.2f} "
                    f"-> BUY (size {decision.position_size_pct * 100:.0f}%)",
                )

            return (
                "HOLD",
                f"NEAT signal: action={action_signal:+.2f}, risk={risk_signal:+.2f} -> waiting for entry",
            )
