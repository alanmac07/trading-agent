# 🧬 Self-Cloning Trading Agent

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)


</p>

An evolutionary trading simulation where autonomous trading agents compete, survive, clone themselves, and gradually improve their trading strategies using principles inspired by natural selection.

Instead of training a single model, this project simulates an ecosystem of independent trading agents. Each agent follows its own trading genome, competes on historical market data, and either survives, reproduces through mutation, or is removed based on its performance.

---

# ✨ Features

- 🧬 Evolutionary trading simulation
- 📈 Historical market data using Yahoo Finance
- 🤖 Autonomous trading agents
- 🔁 Self-cloning profitable agents
- 🧪 Genome mutation
- ☠️ Removal of underperforming agents
- 🪦 Genome graveyard to prevent failed strategies from reappearing
- 📊 Interactive Streamlit dashboard
- 📉 Technical indicators (RSI & Moving Average)
- ♾️ Synthetic market continuation after historical data ends
- 💾 Cached market data for offline demonstrations

---



# 📂 Project Structure

```
.
├── app.py                       # Streamlit dashboard
├── fetch_data.py                # Downloads and prepares market data
├── evolution.py                 # Evolution engine
├── agent.py                     # Rule-based trading agent
├── neuro_agent.py               # Experimental neural agent
├── train_and_evaluate.py        # Training & evaluation script
├── test_with_synthetic_data.py  # Simulation using generated market data
├── feature_engineering.py
├── requirements.txt
├── best_genome_AAPL.pkl
│
├── data/
│   ├── price_data.csv
│   ├── train.csv
│   ├── test.csv
│   └── price_data_multi.csv
│
└── README.md
```

---

# ⚙️ How It Works

The project simulates an evolving population of trading agents.

1. Historical market data is downloaded.
2. Technical indicators are calculated.
3. Agents begin trading with their own genomes.
4. Agents are evaluated periodically.
5. Profitable agents clone themselves.
6. Cloned agents receive random mutations.
7. Poor-performing agents are removed.
8. Failed genomes are stored in a graveyard.
9. The process continues across historical and synthetic market data.

The objective is to observe how trading strategies evolve over time through selection and mutation rather than manual optimization.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/alanmac07/trading-agent.git

cd trading-agent
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Fetch Historical Market Data

Before running the simulation, download and prepare the market data.

```bash
python fetch_data.py
```

This script:

- Downloads historical price data using Yahoo Finance
- Computes technical indicators
- Saves processed data locally
- Creates cached CSV files for faster execution

Generated files include:

```
data/price_data.csv
data/train.csv
data/test.csv
```

---

# 🧪 Run Synthetic Data Test

To verify that the evolution engine works without internet access:

```bash
python test_with_synthetic_data.py
```

This generates synthetic price data and runs a complete evolutionary simulation.

---

# 🏋️ Train and Evaluate

To evolve trading agents and evaluate the best-performing genome:

```bash
python train_and_evaluate.py
```

This script:

- Evolves a population of agents
- Selects the best-performing genome
- Saves it as

```
best_genome_AAPL.pkl
```

- Evaluates the trained genome on unseen test data

---

# 🖥️ Launch the Dashboard

Start the interactive Streamlit interface:

```bash
streamlit run app.py
```

The dashboard allows you to visualize:

- Market prices
- Agent performance
- Population growth
- Evolution progress
- Trading activity

---

# 📊 Trading Strategy

Current agents make decisions using a rule-based trading genome that considers technical indicators such as:

- Relative Strength Index (RSI)
- Moving Average
- Price behaviour

Agents continuously compete against one another, and only successful strategies survive through successive generations.

---

# 🧬 Evolution Process

```
Historical Market Data
          │
          ▼
 Technical Indicators
          │
          ▼
   Trading Agents
          │
          ▼
Performance Evaluation
          │
          ▼
 ┌───────────────┐
 │ Profit?       │
 └──────┬────────┘
        │
   Yes  ▼      No
 Clone Agent   Remove Agent
        │
        ▼
 Random Mutation
        │
        ▼
 New Generation
```

---

# 📁 Data

The project currently works with cached historical market data.

Current dataset:

- Apple (AAPL)

The architecture is designed so additional assets can be added with minimal changes.

---

# 🛠️ Technologies Used

| Category | Technology |
|-----------|------------|
| Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Charts | Plotly |
| Market Data | Yahoo Finance |
| Evolution Engine | Custom Genetic Algorithm |

---

# 📌 Roadmap

Future improvements planned for the project:

- [ ] Replace rule-based genomes with neuroevolution
- [ ] Multi-market simulation
- [ ] Portfolio optimization
- [ ] More technical indicators
- [ ] Advanced performance analytics
- [ ] Improved visualization dashboard
- [ ] Parallel agent simulation
- [ ] Better mutation strategies
- [ ] Configurable evolution parameters

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve the project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request


---

# ⚠️ Disclaimer

This project is intended for educational and research purposes only.

It should **not** be considered financial advice. Trading financial markets involves significant risk, and past performance does not guarantee future results.

---

# 👤 Author

**Alan Mac**

GitHub: https://github.com/alanmac07

If you found this project interesting, consider giving it a ⭐.
