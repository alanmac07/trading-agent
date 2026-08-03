# Self-Cloning Trading Agent — Setup & Run Guide

## What you have
```
trading_agent_demo/
├── requirements.txt      # dependencies to install
├── fetch_data.py         # STEP 1 — downloads & caches price data
├── agent.py               # the Agent class (genome + decision logic)
├── evolution.py            # the clone/kill/graveyard engine
├── test_with_synthetic_data.py   # sanity test, no internet needed
├── app.py                  # STEP 3 — the Streamlit app (4 pages)
└── data/                   # price_data.csv goes here after step 1
```

## Step-by-step setup

### 1. Open the folder in VS Code (or Antigravity)
Open the `trading_agent_demo` folder as your project root.

### 2. Create a virtual environment (recommended, avoids dependency conflicts)
In the VS Code terminal:
```
python -m venv venv
venv\Scripts\activate        (Windows PowerShell)
```
You should see `(venv)` appear in your terminal prompt.

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Fetch and cache the price data (run once)
```
python fetch_data.py
```
This downloads ~2 years of AAPL daily price data from Yahoo Finance, computes RSI and moving average, and saves it to `data/price_data.csv`. You need internet for this ONE step only — after this, the app never touches the network again.

Want a different asset? Open `fetch_data.py` and change the `TICKER` variable at the top (e.g. `"BTC-USD"` for Bitcoin), then re-run this step.

### 5. (Optional but recommended) Sanity-check the core logic
```
python test_with_synthetic_data.py
```
This runs the evolution engine on fake random data with tight thresholds, just to confirm cloning and killing actually trigger and print out. No internet needed. You should see some `CLONE` and `KILLED` lines in the output.

### 6. Run the actual app
```
streamlit run app.py
```
This opens a browser tab automatically (usually `http://localhost:8501`). If it doesn't open automatically, copy that URL into your browser.


## If something breaks
- **`ModuleNotFoundError`** — you forgot to activate the venv or run `pip install -r requirements.txt`
- **`data/price_data.csv not found`** — run `python fetch_data.py` first
- **yfinance download fails / empty data** — check your internet connection, or try a different ticker symbol in `fetch_data.py`
- **Streamlit port already in use** — run `streamlit run app.py --server.port 8502` instead

## What to work on next (once this base version works)
- Tune `checkpoint_days`, `profit_goal_pct`, `loss_limit_pct` until the demo tells a good visual story (a few clones, at least one kill)
- Add the genealogy tree visual on the Agents page (currently shown as generation columns — a proper tree diagram is a nice upgrade if you have time)
- Consider walk-forward validation for Phase 1 (see the PRD)
