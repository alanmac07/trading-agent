"""
STEP 1: Run this file FIRST, once, before anything else.

Downloads historical price data for one asset via yfinance,
computes the technical indicators the agents need (RSI, moving average),
and saves everything to a local CSV.

Why this matters: we cache the data locally so the live demo NEVER
hits the network. If yfinance rate-limits you or your wifi hiccups
during the presentation, the app still works because it just reads
the CSV.

Run it like this from your terminal:
    python fetch_data.py

You should see a new file: data/price_data.csv
"""

"""
STEP 1: Run this file FIRST, once, before anything else.

Downloads historical price data for one OR MORE assets via yfinance,
computes the technical indicators the agents need (RSI, moving average),
and saves everything to a local CSV.

Multi-ticker mode: set TICKERS to a list of symbols. Output CSV will have
a "Ticker" column so the app can filter/switch between markets.

Why cache locally: the live demo NEVER hits the network. If yfinance
rate-limits you or your wifi hiccups during a demo, the app still works
because it just reads the CSV.

Run it like this from your terminal:
    python fetch_data.py

You should see a new file: data/price_data.csv
"""

import yfinance as yf
import pandas as pd
import ta

# ---- CONFIG: change these ----
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL","BTC-USD"]   # add/remove tickers freely
PERIOD = "5y"              # 5 years gives enough data for training a model, not just backtesting
INTERVAL = "1d"            # daily candles (swing trading, not day trading)
SAVE_PATH = "data/price_data.csv"
# -------------------------------


def fetch_one_ticker(ticker, period=PERIOD, interval=INTERVAL):
    print(f"Downloading {ticker} ({period}, {interval} candles)...")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    if df.empty:
        print(f"  WARNING: no data returned for '{ticker}', skipping.")
        return None

    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MACD"] = ta.trend.MACD(close=df["Close"]).macd()
    df["BB_high"] = ta.volatility.BollingerBands(close=df["Close"]).bollinger_hband()
    df["BB_low"] = ta.volatility.BollingerBands(close=df["Close"]).bollinger_lband()
    df["Return_1d"] = df["Close"].pct_change()
    df["Volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

    df = df.dropna()
    df = df.reset_index()
    df["Ticker"] = ticker
    return df


def fetch_and_prepare(tickers=TICKERS, save_path=SAVE_PATH):
    all_dfs = []
    for t in tickers:
        df = fetch_one_ticker(t)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data was successfully downloaded for any ticker. Check your internet connection.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(save_path, index=False)
    print(f"\nSaved {len(combined)} total rows across {len(all_dfs)} ticker(s) to {save_path}")
    print(combined.groupby("Ticker").size())
    return combined


if __name__ == "__main__":
    fetch_and_prepare()
