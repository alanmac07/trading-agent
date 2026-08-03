"""
neat_engine/build_dataset.py

Run AFTER fetch_data.py (needs data/price_data_multi.csv with
Date/Close/High/Low/Open/Volume/RSI/Ticker columns -- already produced
by the existing project).

Computes the 13 extended market features per ticker, does a
CHRONOLOGICAL 80/20 split PER TICKER (never shuffle), and writes:
  data/train_extended.csv
  data/test_extended.csv   <- held out, never touched during evolution

Run: python -m neat_engine.build_dataset
"""

from __future__ import annotations

import pandas as pd

from .features import MARKET_FEATURE_COLUMNS, compute_extended_features

RAW_PATH = "data/price_data_multi.csv"
TRAIN_PATH = "data/train_extended.csv"
TEST_PATH = "data/test_extended.csv"
TRAIN_FRACTION = 0.8


def main():
    raw = pd.read_csv(RAW_PATH, parse_dates=["Date"])
    train_parts, test_parts = [], []

    for ticker, group in raw.groupby("Ticker"):
        feat = compute_extended_features(group)
        feat = feat.dropna(subset=MARKET_FEATURE_COLUMNS)

        split_idx = int(len(feat) * TRAIN_FRACTION)
        train_parts.append(feat.iloc[:split_idx])
        test_parts.append(feat.iloc[split_idx:])

        print(f"{ticker}: {len(feat)} usable rows -> {split_idx} train / {len(feat) - split_idx} test")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    keep_cols = ["Date", "Ticker", "Close"] + MARKET_FEATURE_COLUMNS
    train_df[keep_cols].to_csv(TRAIN_PATH, index=False)
    test_df[keep_cols].to_csv(TEST_PATH, index=False)

    print(f"\nSaved {len(train_df)} train rows -> {TRAIN_PATH}")
    print(f"Saved {len(test_df)} test rows -> {TEST_PATH}")


if __name__ == "__main__":
    main()
