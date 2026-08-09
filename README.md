# 🧬 Self-Cloning Evolutionary Trading Agent (with NEAT Architecture)

A modular, Python-based algorithmic trading framework that uses evolutionary computing to discover, optimize, and evaluate automated trading strategies on historical market data.

The repository contains two distinct evolutionary paradigms:
1. **Continuous Engine (v1):** A real-time, threshold-triggered agent ecosystem using rule-based genomes and fixed-topology neural nets, visualized live in a Streamlit dashboard.
2. **NEAT Architecture (`neat_engine/`):** A generational NeuroEvolution of Augmenting Topologies engine that evolves neural network *structure* (topology + weights), trains across multiple assets, and separates signal generation from trade execution.

---

## ✨ Features & Architecture Highlights

### 🏎️ Continuous Engine (v1)

* **Asynchronous Ecosystem:** Agents trade continuously and clone/die mid-simulation the moment their own cumulative return crosses a profit or loss threshold — not in synchronized generations.
* **Dual Agent Types:**
  * `agent.py` — rule-based genomes optimizing technical thresholds.
  * `neuro_agent.py` — fixed-topology neural network (8 inputs → 10 hidden → 1 output), mutated the same way as the rule-based genomes.
* **Graveyard Pool (capital reinvestment):** When an agent is terminated, its remaining capital is no longer lost — it's captured into a shared pool and injected into the *next* clone's starting capital, on top of what it inherits from its parent.
* **Genome Graveyard (strategy memory):** Failed genome configurations are stored separately and used to steer future mutations away from previously-failed strategies.
* **Ticker-Aware Live Dashboard:** Built with **Streamlit** (`app.py`). Every simulation run is scoped to a single, explicitly selected ticker (AAPL / MSFT / NVDA / BTC-USD) with real candlestick-adjacent OHLCV history — no cross-ticker data mixing.

### 🧬 NEAT Upgrade Engine (`neat_engine/`)

* **Dynamic Topology Evolution:** Uses `neat-python` to evolve network architecture *and* weights via speciation. Networks start minimal (no hidden nodes) and complexify only when it improves fitness.
* **Separation of Prediction & Execution:**
  * The network predicts two signals: **Action Signal** (buy / hold / sell, via fixed thresholds) and **Risk Signal** (position-sizing aggressiveness).
  * `risk_manager.py` deterministically converts those signals into an actual trade, a position size, and an **ATR-scaled trailing stop** — the stop width scales with each asset's own recent volatility instead of using one flat percentage for every market (a fixed stop was too tight for BTC's volatility and too loose to matter for calmer equities).
* **Multi-Metric Fitness:** Blends Total Return, Sharpe Ratio, Max Drawdown, and a trade-activity term (penalizing both doing nothing and overtrading). Episodes where capital collapses below 20% of starting value are terminated early and penalized rather than run to completion.
* **Multi-Asset, Multi-Window Training:** Every genome is evaluated across overlapping historical windows sliced from all four tickers (AAPL, MSFT, NVDA, BTC-USD) — a label-free stand-in for market-regime diversity, so genomes can't just overfit one continuous run.
* **Strict Out-of-Sample Validation:** `build_dataset.py` performs a chronological 80/20 split *per ticker*. The test set is never touched during evolution — `evaluate_holdout.py` is the only thing that ever sees it.
* **Parallelized & Resumable:** `--parallel` evaluates the full population across all CPU cores (pure CPU/NumPy workload — no GPU involved). Population state checkpoints every 10 generations, and the best genome found so far is saved incrementally (not just at the very end), so a run can be interrupted and inspected or resumed without losing progress.

---

## 📂 Project Structure

```text
.
├── app.py                       # Streamlit dashboard for the continuous engine
├── fetch_data.py                # Historical market data downloader (yfinance)
├── evolution.py                 # Continuous evolutionary engine + graveyard pool
├── agent.py                     # Rule-based trading agent
├── neuro_agent.py                # Fixed-topology neural agent
├── train_and_evaluate.py        # Baseline training and evaluation script
├── test_with_synthetic_data.py  # Offline simulation with generated price series
├── feature_engineering.py       # Technical indicator pipeline (v1 engine)
├── requirements.txt              # Python dependencies
│
├── neat_engine/                  # 🧬 NEAT module (independent of the v1 engine above)
│   ├── build_dataset.py          # Chronological 80/20 per-ticker train/test split
│   ├── config-feedforward.txt    # NEAT hyperparameters (population, mutation, speciation)
│   ├── features.py               # 15-input feature pipeline (13 market + 2 portfolio-state)
│   ├── network_agent.py          # Wraps a genome as a compiled feed-forward network
│   ├── risk_manager.py           # Execution layer: position sizing + ATR-scaled trailing stop
│   ├── episodes.py               # Multi-ticker, multi-window episode slicing
│   ├── fitness.py                # Multi-metric fitness (return, Sharpe, drawdown, activity)
│   ├── trainer.py                # Generational trainer — checkpointable, parallelizable
│   └── evaluate_holdout.py       # Out-of-sample performance report
│
├── data/                         # price_data_multi.csv (canonical raw data) + derived CSVs
└── results/                      # best_genome.pkl + fitness_history.csv
```

> **Note on `data/price_data_multi.csv`:** this is the single canonical raw data file — `fetch_data.py`, `feature_engineering.py`, and `neat_engine/build_dataset.py` all read/write this exact name. If you're regenerating data, always run `fetch_data.py` first before anything else downstream.

---

## ⚙️ How It Works

### Continuous Engine (v1)

```text
Historical Data ──► Technical Indicators ──► Trading Agents
                                                   │
                                     ┌─────────────┴─────────────┐
                                     ▼                           ▼
                             Threshold Crossed?           Underperforming?
                                     │                           │
                               ┌─────┴─────┐                     ▼
                              Yes          No            Terminate Agent
                               │           │           (capital → graveyard
                        Clone & Mutate   Continue         pool, reused by
                     (+ pool capital)                      next clone)
```

### NEAT Generation Engine

```text
Multi-Asset Data (AAPL, MSFT, NVDA, BTC-USD)
                   │
                   ▼
     15-Feature Engineering Pipeline
                   │
                   ▼
  Generational Population (Speciation)
                   │
                   ▼
     Network Output (Action + Risk)
                   │
                   ▼
   Deterministic Risk Manager
   (position size + ATR-scaled stop)
                   │
                   ▼
 Multi-Metric Fitness (Return, Sharpe, Drawdown, Trade Activity)
```

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/alanmac07/trading-agent.git
cd trading-agent

python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
pip install neat-python   # required for neat_engine, not yet in requirements.txt
```

### 2. Run the Continuous Engine (v1)

```bash
python fetch_data.py          # writes data/price_data_multi.csv
streamlit run app.py          # pick a ticker from the sidebar, then Start
```

### 3. Run the NEAT Architecture Engine

```bash
# 1. Build the chronological train/test split (needs price_data_multi.csv already fetched)
python -m neat_engine.build_dataset

# 2. Train (all CPU cores, e.g. 2000 generations for an overnight run)
python -m neat_engine.trainer --generations 2000 --parallel

# Resume an interrupted run:
python -m neat_engine.trainer --generations 2000 --parallel --resume checkpoints/neat-checkpoint-N

# 3. Evaluate the best genome on unseen holdout data
python -m neat_engine.evaluate_holdout
```

---

## 📊 Feature Inputs (NEAT Engine)

15 normalized inputs per time step — 13 market features plus 2 live portfolio-state features:

1. RSI (normalized)
2. Price vs. EMA 20
3. Price vs. EMA 50
4. MACD
5. MACD Signal
6. MACD Histogram
7. Bollinger Bands %B
8. Bollinger Band Width
9. ATR (normalized by price)
10. Stochastic Oscillator %K
11. 1-Day Return
12. 5-Day Return
13. Volume Z-Score
14. In Position (0 / 1)
15. Unrealized P&L (normalized)

Network outputs 2 signals: **Action Signal** (buy/hold/sell) and **Risk Signal** (position sizing + trailing-stop aggressiveness).

---

## 🛠️ Technologies Used

| Category | Technology |
| --- | --- |
| **Language** | Python 3.10+ |
| **Evolution Engine** | `neat-python`, custom continuous GA engine |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Streamlit, Plotly |
| **Market Data** | Yahoo Finance (`yfinance`) |

---

## ⚠️ Disclaimer

This repository is built strictly for **educational and quantitative research purposes**. It is not financial advice, and these strategies should not be deployed in live trading accounts without extensive risk controls, slippage modeling, and paper testing.

---

## 👤 Author

**Alan Roxy Jesudas**

* GitHub: [@alanmac07](https://github.com/alanmac07)
