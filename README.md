
# 🧬 Self-Cloning Evolutionary Trading Agent (with NEAT Architecture)

A modular, python-based algorithmic trading framework that uses evolutionary computing to discover, optimize, and evaluate automated trading strategies on historical market data.

The repository contains two distinct evolutionary paradigms:
1. **Original Continuous Engine:** A real-time, threshold-triggered agent ecosystem using rule-based genomes and fixed-topology neural nets.
2. **NEAT Architecture (`neat_engine/`):** A generational NeuroEvolution of Augmenting Topologies engine that evolves neural network structures (topologies + weights), integrates multi-asset regime testing, and separates signal generation from risk management.

---

## ✨ Features & Architecture Highlights

### 🏎️ Base Continuous System (v1)
* **Asynchronous Ecosystem:** Agents trade continuously and clone/die mid-simulation as soon as profit or loss thresholds are breached.
* **Dual Agent Types:**
  * `agent.py`: Rule-based genomes optimizing technical thresholds.
  * `neuro_agent.py`: Fixed-topology neural network (8 inputs → 10 hidden → 1 output).
* **Genome Graveyard:** Stores failed strategy configurations for analytics and mutation prevention.
* **Interactive Dashboard:** Live visualization built with **Streamlit** (`app.py`).

### 🧬 NEAT Upgrade Engine (`neat_engine/`)
* **Dynamic Topology Evolution:** Uses `neat-python` to evolve network architectures alongside edge weights via speciation. Networks start minimal and complexify only when beneficial.
* **Separation of Prediction & Execution:** 
  * The neural net predicts two signals: **Action Signal** (buy/hold/sell) and **Risk Signal** (aggressiveness).
  * A deterministic `risk_manager.py` translates signals into actual order execution, position sizing, and dynamic trailing-stop logic.
* **Multi-Metric Fitness:** Evaluates genomes on a blended objective function accounting for **Total Return, Sharpe Ratio, Max Drawdown, and Churn/Inactivity Penalties**. Early portfolio collapse terminates evaluation immediately.
* **Multi-Asset Slicing:** Trains genomes across overlapping temporal windows sliced from multiple assets (**AAPL, MSFT, NVDA, BTC-USD**) to prevent overfitting to a single market regime.
* **Strict Out-of-Sample Validation:** `build_dataset.py` enforces a chronological 80/20 train/test split. Out-of-sample data is strictly isolated for `evaluate_holdout.py`.
* **Parallelized & Resumable:** Supports multi-core evaluation (`--parallel`) and outputs population state checkpoints every 10 generations.

---

## 📂 Project Structure

```text
.
├── app.py                      # Streamlit dashboard for continuous agents
├── fetch_data.py               # Historical market data downloader
├── evolution.py                 # Continuous evolutionary engine
├── agent.py                     # Rule-based trading agent
├── neuro_agent.py               # Fixed-topology neural agent
├── train_and_evaluate.py        # Baseline training and evaluation script
├── test_with_synthetic_data.py  # Offline simulation with generated price series
├── feature_engineering.py       # Technical indicator calculation pipeline
├── requirements.txt             # Python dependencies
│
├── neat_engine/                 # 🧬 NEAT Module
│   ├── build_dataset.py         # Chronological 80/20 train/test dataset builder
│   ├── config-feedforward.txt   # NEAT hyperparameters (mutation rates, speciation)
│   ├── features.py              # 15-feature engineering pipeline
│   ├── network_agent.py         # Phenotype wrapper for NEAT genomes
│   ├── risk_manager.py          # Execution engine (position sizing & trailing stops)
│   ├── episodes.py              # Multi-asset temporal window slicing
│   ├── fitness.py               # Multi-metric reward function (Return, Sharpe, Drawdown)
│   ├── trainer.py               # Generational NEAT trainer (Checkpointable & Parallel)
│   └── evaluate_holdout.py      # Out-of-sample holdout performance evaluator
│
├── data/                        # Market datasets
└── results/                     # Saved best genome outputs (.pkl)

```

---

## ⚙️ How It Works

### Base Evolutionary Engine

```text
Historical Data ──► Technical Indicators ──► Trading Agents
                                                   │
                                     ┌─────────────┴─────────────┐
                                     ▼                           ▼
                             Threshold Crossed?         Underperforming?
                                     │                           │
                               ┌─────┴─────┐                     ▼
                              Yes          No               Remove Agent
                               │           │             (Add to Graveyard)
                        Clone & Mutate   Continue

```

### NEAT Generation Engine

```text
Multi-Asset Data (AAPL, MSFT, NVDA, BTC)
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
                   │
                   ▼
 Multi-Metric Fitness (Sharpe, Drawdown, Return)

```

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment

```bash
git clone [https://github.com/alanmac07/trading-agent.git](https://github.com/alanmac07/trading-agent.git)
cd trading-agent

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Run Base Evolutionary System (v1)

```bash
# Fetch market data
python fetch_data.py

# Run training & evaluation
python train_and_evaluate.py

# Launch Streamlit Dashboard
streamlit run app.py

```

### 3. Run NEAT Architecture Engine

```bash
# 1. Build chronological train/test dataset splits
python -m neat_engine.build_dataset

# 2. Train NEAT population (e.g., 200 generations with multi-core support)
python -m neat_engine.trainer --generations 200 --parallel

# 3. Evaluate the best genome on unseen holdout data
python -m neat_engine.evaluate_holdout

```

---

## 📊 Feature Inputs (NEAT Engine)

The NEAT engine processes **15 normalized indicators** per time step:

1. Relative Strength Index (RSI)
2. MACD Main Line
3. MACD Signal Line
4. MACD Histogram
5. Exponential Moving Average Ratio (EMA 20)
6. Exponential Moving Average Ratio (EMA 50)
7. Bollinger Bands %B
8. Bollinger Band Width
9. Average True Range (ATR Ratio)
10. Stochastic Oscillator %K
11. 1-Day Log Return
12. 5-Day Log Return
13. Volume Z-Score
14. Position State Flag (In Position: 0 or 1)
15. Current Unrealized P&L Ratio

---

## 🛠️ Technologies Used

| Category | Technology |
| --- | --- |
| **Language** | Python 3.10+ |
| **Evolution Engine** | `neat-python`, Custom GA Engine |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Streamlit, Plotly |
| **Market Data** | Yahoo Finance (`yfinance`) |

---

## ⚠️ Disclaimer

This repository is built strictly for **educational and quantitative research purposes**. It is not financial advice, and these strategies should not be deployed in live trading accounts without extensive risk controls, slippage modeling, and paper testing.

---

## 👤 Author

**Alan Mac**

* GitHub: [@alanmac07](https://github.com/alanmac07)

```

```
