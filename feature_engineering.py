"""
STEP 2 (new): Run this AFTER fetch_data.py, BEFORE training anything.

Takes the raw price CSV and, per ticker:
  - computes the richer feature set the neural network needs
  - builds the training LABEL (did price go up tomorrow?)
  - does a CHRONOLOGICAL train/test split (never shuffle -- that leaks
    future information into training and gives you fake-looking accuracy)

Output: data/train.csv and data/test.csv

Run: python feature_engineering.py
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/price_data_multi.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
TRAIN_FRACTION = 0.8  # first 80% of each ticker's history = train, last 20% = test


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """df = one ticker's data, sorted by date."""
    df = df.sort_values("Date").reset_index(drop=True).copy()

    # Price vs its own moving average (ratio, not raw price -- keeps scale consistent across tickers)
    df["price_vs_ma20"] = df["Close"] / df["MA20"] - 1

    # MACD (12-day EMA minus 26-day EMA), plus its own 9-day signal line
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger %B: where is price within its own 20-day volatility band? (0 = lower band, 1 = upper band)
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_pct_b"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower)

    # Momentum
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)

    # Volume z-score (is today's volume unusual vs its own recent history?)
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["volume_z"] = (df["Volume"] - vol_mean) / vol_std

    # RSI normalized to roughly [-1, 1] instead of [0, 100] -- neural nets train better on small, centered inputs
    df["rsi_norm"] = (df["RSI"] - 50) / 50

    # LABEL: did price go UP tomorrow? (1 = yes, 0 = no). This is what we're training to predict.
    df["label_up_tomorrow"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df


FEATURE_COLUMNS = [
    "rsi_norm", "price_vs_ma20", "macd", "macd_signal",
    "bb_pct_b", "return_1d", "return_5d", "volume_z",
]


def main():
    raw = pd.read_csv(RAW_PATH, parse_dates=["Date"])
    train_parts, test_parts = [], []

    for ticker, group in raw.groupby("Ticker"):
        feat = compute_features(group)
        feat = feat.dropna(subset=FEATURE_COLUMNS + ["label_up_tomorrow"])

        split_idx = int(len(feat) * TRAIN_FRACTION)
        train_parts.append(feat.iloc[:split_idx])
        test_parts.append(feat.iloc[split_idx:])

        print(f"{ticker}: {len(feat)} usable rows -> {split_idx} train / {len(feat) - split_idx} test")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    keep_cols = ["Date", "Ticker", "Close", "RSI"] + FEATURE_COLUMNS + ["label_up_tomorrow"]
    train_df[keep_cols].to_csv(TRAIN_PATH, index=False)
    test_df[keep_cols].to_csv(TEST_PATH, index=False)

    print(f"\nSaved {len(train_df)} train rows -> {TRAIN_PATH}")
    print(f"Saved {len(test_df)} test rows -> {TEST_PATH}")
    print(f"\nLabel balance (train): {train_df['label_up_tomorrow'].mean():.3f} fraction 'up' days")


if __name__ == "__main__":
    main()
