"""
neat_engine/features.py

Extends the market-data feature set used by feature_engineering.py.
Everything here is computed ONCE from OHLCV history (no lookahead —
every indicator only uses data up to and including the current row).

Portfolio-state features (in_position, unrealized_pnl_pct) are NOT
computed here — those depend on a specific agent's live state and are
built at simulation time in network_agent.py.

MARKET_FEATURE_COLUMNS is the ordered list the NEAT config's
num_inputs must match (plus the 2 portfolio features appended later).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_FEATURE_COLUMNS = [
    "rsi_norm",
    "price_vs_ema20",
    "price_vs_ema50",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct_b",
    "bb_width",
    "atr_norm",
    "stoch_k",
    "return_1d",
    "return_5d",
    "volume_z",
]

PORTFOLIO_FEATURE_COLUMNS = ["in_position", "unrealized_pnl_norm"]

ALL_FEATURE_COLUMNS = MARKET_FEATURE_COLUMNS + PORTFOLIO_FEATURE_COLUMNS
N_INPUTS = len(ALL_FEATURE_COLUMNS)   # 15
N_OUTPUTS = 2                          # action_signal, risk_signal


def compute_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: one ticker's OHLCV history, sorted by Date, with columns
    Close/High/Low/Open/Volume/RSI already present (RSI comes from
    fetch_data.py the same way it does today).

    Returns df with MARKET_FEATURE_COLUMNS added. Rows near the start
    will contain NaN until rolling windows fill up — drop those before
    training, same as feature_engineering.py already does.
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    close = df["Close"].astype(float)

    # ── trend ────────────────────────────────────────────────────────
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    df["price_vs_ema20"] = close / ema20 - 1
    df["price_vs_ema50"] = close / ema50 - 1

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ── volatility / bands ───────────────────────────────────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower)
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(14).mean()
    df["atr_norm"] = atr / close  # normalized so it's comparable across tickers/price scales

    # ── momentum ─────────────────────────────────────────────────────
    roll_low14 = low.rolling(14).min()
    roll_high14 = high.rolling(14).max()
    df["stoch_k"] = ((close - roll_low14) / (roll_high14 - roll_low14) - 0.5) * 2  # -> [-1, 1]

    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)

    vol = df["Volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std()
    df["volume_z"] = (vol - vol_mean) / vol_std

    # RSI normalized to roughly [-1, 1] (RSI column assumed already present)
    df["rsi_norm"] = (df["RSI"].astype(float) - 50) / 50

    return df


def portfolio_state_vector(in_position: bool, unrealized_pnl_pct: float) -> list[float]:
    """
    unrealized_pnl_pct: e.g. +6.0 means +6% unrealized gain on the open position.
    Normalized by /20 and clipped to [-1, 1] so a +/-20% move saturates the input
    -- keeps it on the same rough scale as the other tanh-friendly features.
    """
    pnl_norm = float(np.clip(unrealized_pnl_pct / 20.0, -1.0, 1.0)) if in_position else 0.0
    return [1.0 if in_position else 0.0, pnl_norm]
