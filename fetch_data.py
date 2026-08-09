"""
fetch_data.py

Downloads 10 years of historical price data for high-volume assets via yfinance,
computes technical indicators (RSI, MA20, MACD, Bollinger Bands, and extended features),
and saves everything into data/price_data_multi.csv.
"""

import os
import yfinance as yf
import pandas as pd
import ta

from neat_engine.features import compute_extended_features

# ---- CONFIG: 10-year multi-asset dataset ----
MARKET_REGIMES = {
    "stocks": ["AAPL", "MSFT", "NVDA", "TSLA"],
    "crypto": ["BTC-USD", "ETH-USD"],
    "macro": ["GC=F", "^GSPC"],
}
TICKERS = [ticker for tickers in MARKET_REGIMES.values() for ticker in tickers]
PERIOD = "10y"
INTERVAL = "1d"
SAVE_PATH = "data/price_data_multi.csv"
# --------------------------------------------


def fetch_one_ticker(ticker: str, period: str = PERIOD, interval: str = INTERVAL) -> pd.DataFrame | None:
    print(f"Downloading {ticker} ({period}, {interval} candles)...")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  ERROR downloading '{ticker}': {e}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    if df.empty:
        print(f"  WARNING: no data returned for '{ticker}', skipping.")
        return None

    # Base indicators
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MACD"] = ta.trend.MACD(close=df["Close"]).macd()
    df["BB_high"] = ta.volatility.BollingerBands(close=df["Close"]).bollinger_hband()
    df["BB_low"] = ta.volatility.BollingerBands(close=df["Close"]).bollinger_lband()
    df["Return_1d"] = df["Close"].pct_change()
    df["Volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

    # Drop NaNs from initial indicator rollbacks
    df = df.dropna().reset_index()
    
    # Compute extended features for NEAT
    try:
        df = compute_extended_features(df)
        df = df.dropna().reset_index(drop=True)
    except Exception as e:
        print(f"  Warning: compute_extended_features failed for {ticker}: {e}")

    df["Ticker"] = ticker
    return df


def fetch_and_prepare(tickers=TICKERS, save_path=SAVE_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_dfs = []
    for t in tickers:
        df = fetch_one_ticker(t)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data was successfully downloaded for any ticker. Check internet connection.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(save_path, index=False)
    print(f"\nSaved {len(combined)} total rows across {len(all_dfs)} ticker(s) to {save_path}")
    print(combined.groupby("Ticker").size())
    return combined


if __name__ == "__main__":
    fetch_and_prepare()
