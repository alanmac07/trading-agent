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

import random
import numpy as np
import pandas as pd

from agent import Agent, AGENT_COLOR_PALETTE, _next_color

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
