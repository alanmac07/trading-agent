from __future__ import annotations

"""
The Agent class. Each agent is a small trading strategy defined entirely
by its "genome" — a set of numeric thresholds. No neural network, no
training weights.

Enhanced v2:
- Unique chart color per agent
- Peak / high-water-mark tracking for trailing-stop cash-outs
- Two new genome genes: trailing_stop_pct and reentry_rsi
- Smarter decide(): RSI entry + RSI exit + trailing-stop exit
- date_history tracked alongside portfolio_history
"""

import random

import pandas as pd

# ── Color palette (vivid, distinct on dark backgrounds) ──────────────────────
AGENT_COLOR_PALETTE = [
    "#00D4FF",  # cyan
    "#FF6B6B",  # coral
    "#4ECDC4",  # teal
    "#FFE66D",  # yellow
    "#C3A6FF",  # lavender
    "#FF8B94",  # pink
    "#69F0AE",  # green
    "#FF7043",  # deep-orange
    "#40C4FF",  # sky-blue
    "#FFAB40",  # amber
    "#E040FB",  # purple
    "#64FFDA",  # seafoam
    "#FF4081",  # hot-pink
    "#B2FF59",  # lime
    "#F9A8D4",  # rose
]

_color_idx = [0]  # use a list so closures can mutate it


def _next_color() -> str:
    c = AGENT_COLOR_PALETTE[_color_idx[0] % len(AGENT_COLOR_PALETTE)]
    _color_idx[0] += 1
    return c


class Agent:
    def __init__(
        self,
        agent_id: str,
        capital: float,
        genome: dict | None = None,
        parent_id: str | None = None,
        generation: int = 0,
        color: str | None = None,
    ):
        self.id = agent_id
        self.capital = capital                  # cash on hand
        self.initial_capital = capital          # starting capital (for fitness / P&L)
        self.genome = genome or self.random_genome()
        self.parent_id = parent_id
        self.generation = generation
        self.color = color or _next_color()     # unique chart color

        self.position: float = 0.0             # shares currently held
        self.status: str = "alive"             # alive | dead
        self.cause_of_death: str | None = None

        # Peak tracking for trailing-stop exit
        self.price_high_water: float = 0.0

        # Per-day history
        self.decision_log: list[dict] = []
        self.portfolio_history: list[float] = []
        self.date_history: list[str] = []

    # ── genome ────────────────────────────────────────────────────────────────

    @staticmethod
    def random_genome() -> dict:
        """Used only for the first seed agent."""
        return {
            "rsi_buy_threshold":  round(random.uniform(42, 55), 1),
            "rsi_sell_threshold": round(random.uniform(62, 78), 1),
            "ma_window":          random.choice([10, 15, 20, 30]),
            "position_size_pct":  round(random.uniform(0.6, 0.95), 2),
            "stop_loss_pct":      round(random.uniform(5, 12), 1),
            "trailing_stop_pct":  round(random.uniform(4, 10), 1),  # cash-out X% below peak
            "reentry_rsi":        round(random.uniform(35, 48), 1),  # RSI must be this low to re-enter
        }

    @staticmethod
    def mutate_genome(genome: dict, mutation_rate: float = 0.15) -> dict:
        """Return a mutated COPY of the genome. Original is never modified."""
        new = {}
        for key, val in genome.items():
            if isinstance(val, (int, float)):
                delta = val * mutation_rate * random.uniform(-1, 1)
                new_val = val + delta
                if key == "ma_window":
                    new_val = max(5, int(round(new_val)))
                else:
                    new_val = round(new_val, 2)
                new[key] = new_val
            else:
                new[key] = val
        return new

    # ── decision logic ────────────────────────────────────────────────────────

    def decide(self, row) -> tuple[str, str]:
        """
        Decide BUY / SELL / HOLD for today.
        """
        rsi: float   = float(row["RSI"])
        price: float = float(row["Close"])
        ma: float    = float(row["MA20"]) if "MA20" in row and not pd.isna(row["MA20"]) else price
        buy_th  = self.genome["rsi_buy_threshold"]
        sell_th = self.genome["rsi_sell_threshold"]
        trail   = self.genome.get("trailing_stop_pct", 6.0)

        if self.position > 0:
            # Update high-water mark
            if price > self.price_high_water:
                self.price_high_water = price

            # (a) Trailing stop
            if self.price_high_water > 0:
                drop_pct = (self.price_high_water - price) / self.price_high_water * 100.0
                if drop_pct >= trail:
                    peak = self.price_high_water
                    self.price_high_water = 0.0
                    return (
                        "SELL",
                        f"🔔 Trailing stop: -{drop_pct:.1f}% from peak ${peak:.2f} "
                        f"(limit: {trail:.1f}%) → cashing out profits",
                    )

            # (b) RSI overbought
            if rsi > sell_th:
                self.price_high_water = 0.0
                return "SELL", f"RSI={rsi:.1f} > {sell_th:.1f} → overbought, exiting position"

            return "HOLD", f"RSI={rsi:.1f} in range — holding position (peak: ${self.price_high_water:.2f})"

        else:
            # (a) RSI oversold or bullish momentum above Moving Average
            if rsi < buy_th or (rsi < 52 and price >= ma):
                self.price_high_water = price  # begin tracking from entry price
                return "BUY", f"RSI={rsi:.1f} < {buy_th:.1f} (or price ≥ MA20) → entering position"

            return "HOLD", f"RSI={rsi:.1f} ≥ {buy_th:.1f} → waiting for entry signal"

    # ── helpers ───────────────────────────────────────────────────────────────

    def portfolio_value(self, price: float) -> float:
        return self.capital + self.position * price

    def return_pct(self, price: float) -> float:
        return (self.portfolio_value(price) - self.initial_capital) / self.initial_capital * 100.0
