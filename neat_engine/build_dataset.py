"""
neat_engine/build_dataset.py

Run AFTER fetch_data.py (needs data/price_data_multi.csv with
Date/Close/High/Low/Open/Volume/RSI/Ticker columns).

Computes extended market features per ticker, performs chronological 80/20 train/test split,
and outputs 3 distinct regime datasets + combined dataset:
  - data/train_stocks.csv & data/test_stocks.csv
  - data/train_crypto.csv & data/test_crypto.csv
  - data/train_macro.csv & data/test_macro.csv
  - data/train_extended.csv & data/test_extended.csv

Run: python -m neat_engine.build_dataset
"""

from __future__ import annotations

import os
import pandas as pd

from .features import MARKET_FEATURE_COLUMNS, compute_extended_features

RAW_PATH = "data/price_data_multi.csv"
TRAIN_FRACTION = 0.8

MARKET_REGIMES = {
    "stocks": ["AAPL", "MSFT", "NVDA", "TSLA"],
    "crypto": ["BTC-USD", "ETH-USD"],
    "macro": ["GC=F", "^GSPC"]
}

TICKER_TO_REGIME = {
    ticker: regime
    for regime, tickers in MARKET_REGIMES.items()
    for ticker in tickers
}


def get_market_regime(ticker: str) -> str:
    """Classifies a ticker symbol into its corresponding market regime."""
    if ticker in TICKER_TO_REGIME:
        return TICKER_TO_REGIME[ticker]
    # Heuristics for non-standard or user-added tickers
    if ticker.endswith("-USD") or ticker.endswith("USDT") or "BTC" in ticker or "ETH" in ticker:
        return "crypto"
    if "=F" in ticker or ticker.startswith("^"):
        return "macro"
    return "stocks"


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data file {RAW_PATH} not found. Please run fetch_data.py first.")

    raw = pd.read_csv(RAW_PATH, parse_dates=["Date"])
    os.makedirs("data", exist_ok=True)
    
    # Store train/test parts by regime and combined
    regime_train = {r: [] for r in MARKET_REGIMES.keys()}
    regime_test = {r: [] for r in MARKET_REGIMES.keys()}
    all_train = []
    all_test = []

    for ticker, group in raw.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        feat = compute_extended_features(group)
        feat = feat.dropna(subset=MARKET_FEATURE_COLUMNS)

        split_idx = int(len(feat) * TRAIN_FRACTION)
        regime = get_market_regime(ticker)
        
        train_part = feat.iloc[:split_idx]
        test_part = feat.iloc[split_idx:]

        if regime in regime_train:
            regime_train[regime].append(train_part)
            regime_test[regime].append(test_part)

        all_train.append(train_part)
        all_test.append(test_part)

        print(f"{ticker} ({regime}): {len(feat)} usable rows -> {len(train_part)} train / {len(test_part)} test")

    keep_cols = ["Date", "Ticker", "Close"] + MARKET_FEATURE_COLUMNS

    # Write per-regime datasets
    for regime in MARKET_REGIMES.keys():
        if not regime_train[regime]:
            continue
            
        train_df = pd.concat(regime_train[regime], ignore_index=True)
        test_df = pd.concat(regime_test[regime], ignore_index=True)
        
        train_path = f"data/train_{regime}.csv"
        test_path = f"data/test_{regime}.csv"
        
        train_df[keep_cols].to_csv(train_path, index=False)
        test_df[keep_cols].to_csv(test_path, index=False)
        
        print(f"\n[{regime.upper()}] Saved {len(train_df)} train rows -> {train_path}")
        print(f"[{regime.upper()}] Saved {len(test_df)} test rows -> {test_path}")

    # Write combined datasets
    if all_train:
        combined_train = pd.concat(all_train, ignore_index=True)
        combined_test = pd.concat(all_test, ignore_index=True)
        combined_train[keep_cols].to_csv("data/train_extended.csv", index=False)
        combined_test[keep_cols].to_csv("data/test_extended.csv", index=False)
        print(f"\n[UNIVERSAL] Saved {len(combined_train)} train rows -> data/train_extended.csv")
        print(f"[UNIVERSAL] Saved {len(combined_test)} test rows -> data/test_extended.csv")


if __name__ == "__main__":
    main()
