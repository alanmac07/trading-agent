# PRD: Self-Cloning Evolutionary Trading Agent

**Status:** Draft v1 — Phase 0 (HOD dummy demo) scope locked, Phase 1+ open
**Owner:** Alan
**Last updated:** July 2026

---

## 1. Overview

An autonomous trading agent that starts as a single entity with a randomly initialized trading strategy ("genome") and a starting capital allocation. As it trades against historical market data, agents that hit a profit threshold **clone** — the clone receives a slice of capital and a slightly mutated version of the parent's genome, and begins trading independently alongside the parent. Agents that cross a loss threshold are **terminated**. Over successive cycles, this produces a branching lineage of trading agents whose strategies drift and specialize through selection and mutation — a genetic-algorithm-style evolutionary system, not a neural-network-trained model.

**This is not literally Conway's Game of Life** — there's no grid or neighbor-count rule. It's population-based evolutionary computation (genetic algorithm) applied to trading strategy search. Framing it accurately matters for the academic defense.

---

## 2. Locked scope constraints

- **Single seed agent at start** — not a pre-populated pool. Population grows organically via cloning (tree/lineage structure), shrinks via termination.
- **Swing trading only** (daily candle data) — not day trading. Reasons: cleaner free data, fewer transaction-cost/slippage distortions, sufficient trade frequency for fitness evaluation within a semester timeline.
- **Simulated capital only** — no real broker integration, no KYC, no real money at any phase of this project. All trading is backtested against historical data using a locally cached CSV.
- **Rule-based genome** (technical indicator thresholds), not RL, not a trained neural net. Rationale: interpretable, fast to evaluate, visibly different behavior between parent and clone, defensible within a semester scope.
- **Dead agents inform future mutations** ("graveyard" mechanism, see §5.3) — this is the project's specific research contribution beyond a vanilla GA.
- **Presented first as a standalone agent system** ("agent-first" per HOD's directive) — application/product layer (web front end) comes after the core agent mechanism is proven.

---

## 3. Goals

### Phase 0 — HOD dummy demo (few days, compressed)
Prove the core mechanism works and is demoable: one seed agent → clones → some clones profit and re-clone, some die → visible lineage, visible decision-making, visible profit/loss per agent. Built for speed and reliability over polish (Jupyter notebook or minimal Streamlit, local cached data, single asset, small generation count).

### Phase 1 — Full agent build
Formalize the evolution engine, walk-forward validation, graveyard-informed mutation, and multi-generation testing across a longer historical window. Add proper logging/analytics on population behavior over time (does the lineage converge on a dominant strategy? does diversity persist?).

### Phase 2 — Application layer
Full 3-page web interface (see §6), potentially multi-asset support, potentially live paper-trading integration (e.g. Alpaca paper API) for a "realistic" but risk-free live demo layer.

---

## 4. Non-goals (explicitly out of scope)

- Real money trading of any kind
- Day trading / intraday strategies
- Reinforcement learning or deep learning-based agent policies (may be a stretch goal in a later phase, not core)
- Multi-asset portfolios (Phase 0/1 = single asset)
- Any actual generative AI / LLM component (clarified: "generative" in this project refers to agents generating/cloning themselves, not an LLM)

---

## 5. Agent design & mechanism

### 5.1 Genome
Each agent's strategy is defined by a small set of tunable numeric parameters, e.g.:
```
{
  rsi_buy_threshold: 28,
  rsi_sell_threshold: 72,
  ma_window: 20,
  position_size_pct: 0.5,
  stop_loss_pct: 5
}
```

### 5.1a User-configurable run parameters
Set before each simulation run (not hardcoded):
- **Starting capital** — amount the seed agent begins trading with.
- **Clone trigger (profit goal)** — the profit target that, once hit, triggers cloning. Replaces a fixed hardcoded threshold; user sets this per run (e.g. "clone once an agent hits +10% return").
- **Kill trigger (loss limit)** — loss threshold that triggers termination, same configurable treatment.
- **Clone capital split** — how much of the parent's capital the new clone receives (e.g. 50/50 split, or a fixed amount).

### 5.1b Why rule-based + evolution, not RL
Markets are non-stationary (regimes shift), which is a real concern — but RL doesn't solve this by default. An RL agent trained on one regime degrades on a new one just like a rule-based agent would, and retraining RL online is itself an unsolved, unstable research problem. This system's evolutionary loop already handles regime change structurally: each generation is re-evaluated against new data (walk-forward), so agents whose thresholds no longer fit the current regime are killed off, while clones with better-fitting mutated thresholds survive. Adaptation happens at the population level without needing a trained reward function or risking RL's training instability — and it stays explainable (every decision traces to a specific threshold check), which matters for a semester-scoped academic defense.

### 5.2 Decision logic
Each simulated trading day, the agent evaluates that day's precomputed indicators (RSI, moving average, etc.) against its own genome thresholds and returns one of `BUY / SELL / HOLD`, along with a human-readable reason string (e.g. `"RSI=26.4 < threshold 28 → BUY"`). This reason string is core to the "thinking" display — it must be generated at decision time, not reconstructed after the fact.

### 5.3 Evolution cycle
1. Agent trades through the historical dataset day-by-day, portfolio value tracked continuously.
2. Fitness = ending return (or risk-adjusted return, e.g. simplified Sharpe, if time allows) evaluated at a checkpoint.
3. **Clone condition:** fitness exceeds profit threshold → create child agent. Child receives a fraction of parent's current capital and a mutated copy of the genome (each parameter perturbed by a small random amount).
4. **Kill condition:** fitness drops below loss threshold → agent terminated, removed from active population, logged to the **graveyard** with its genome and cause of death.
5. **Graveyard-informed mutation:** when generating a new clone's genome, check distance to known graveyard genomes; if too close, resample the mutation to steer away from previously-failed parameter regions. This is the project's "agents learn from dead agents" mechanism — implemented as informed mutation, not weight transfer (there are no weights).
6. Repeat — population grows/shrinks as a lineage tree rooted at the original seed agent.

### 5.4 Validation discipline (Phase 1+, important for credibility)
Use walk-forward evaluation — evolve/evaluate on sequential, non-overlapping historical windows rather than repeatedly testing on the same window — to avoid overfitting the genome pool to one historical period.

---

## 5.6 Data acquisition

- **Source:** `yfinance` (free, no API key, no KYC/signup) — pulls historical OHLC (open/high/low/close) data directly from Yahoo Finance.
- **Asset:** single ticker for Phase 0/1 (e.g. AAPL or BTC-USD) — pick one liquid, well-known asset so results are easy to sanity-check.
- **Range:** 1-2 years of daily candles is enough for meaningful swing-trading fitness evaluation without excessive load time.
- **Caching:** pull once, save locally as CSV. All simulation runs (and especially the live HOD demo) read from this local file — never hit the network live, to avoid rate-limit or connectivity failures mid-presentation.
- **Indicator precomputation:** compute RSI, moving averages, etc. (via the `ta` library) for every row in the cached dataset upfront, so the simulation loop only does lookups, not live calculation — faster and avoids indicator-window edge cases (e.g. insufficient lookback on day 1).

## 6. Interface design (4-page structure)

### Page 1 — Market Page
The raw market itself: the asset's price chart over the historical window, with indicator overlays (RSI, moving average) — this is the underlying data every agent is reacting to, shown independent of any agent's activity. Gives HOD/viewers context for what the agents are trading against.

### Page 2 — Aggregate Overview Page
System-wide rollup across all agents: total trades executed, total system profit, activity feed of recent trade events across the whole population. This is the "how is everything doing overall" view.

### Page 3 — Agents Page
View of all currently alive agents. If population = 1, shows just the seed agent's graph. As cloning occurs, shows all active agents — as a lineage tree (parent → children, dead nodes greyed out) and/or a grid of mini portfolio-value sparklines per agent.

### Page 4 — Agent Detail / Thinking Page
Click into any single agent → shows:
- Current capital allocated, current profit/loss
- Portfolio value chart over simulated time
- Genome (current parameters)
- Scrolling decision log ("thinking"): day-by-day indicator values and the resulting action + reasoning
- If cloned from a parent: genome diff vs parent, highlighting mutated parameters
- If terminated: cause-of-death summary

---

## 7. Tech stack

| Component | Choice | Notes |
|---|---|---|
| Data source | `yfinance` | Free, no signup/KYC. Cache to local CSV before any live demo — never hit network live. |
| Indicators | `ta` (Python lib) | RSI, MACD, moving averages precomputed on the dataset. |
| Evolution engine | Plain Python | Agent class, population manager, graveyard list. |
| Phase 0 UI | Jupyter notebook (fallback) or minimal Streamlit | Prioritize reliability over polish given time constraints. |
| Phase 2 UI | Streamlit + Plotly | Interactive 3-page app as described in §6. |
| Future live layer (Phase 2+, optional) | Alpaca paper trading API | Real market data, fake money, no KYC needed for paper mode. |
| Considered, deferred | React + Framer Motion + UI kit | Would require a separate frontend/backend split (API layer) — real build overhead with no benefit for proving the core mechanism. Revisit only in Phase 2 once the agent core is validated. |

---

## 8. Risks / open questions

- **Overfitting risk:** without walk-forward validation, evolved genomes may look profitable on the demo dataset but be meaningless out-of-sample. Must be addressed explicitly in Phase 1, and acknowledged as a known limitation in the Phase 0 demo.
- **Fitness metric choice:** raw return vs risk-adjusted return — raw return is simpler for the dummy demo, but risk-adjusted is more defensible for the full project.
- **Graveyard distance metric:** needs a concrete definition (e.g. normalized Euclidean distance across genome parameters) — TBD in implementation.
- **Team dependency:** frontend build effort depends on final decision between notebook (fast, safe) vs Streamlit (interactive, higher demo value, more risk) for Phase 0 — recommend deciding based on actual remaining time before presentation.

---

## 9. Prior art (for report citation, not implementation copy)

- `edouardkombo/Native-trading-genetic-algorithm` (GitHub) — GA + neural net trading bot, no hand-coded indicators.
- `miro-ka/mosquito` (GitHub) — modular evolutionary/ML crypto trading bot.
- "AI Trading's Alpha Singularity" (2026) — argues for evolving the scoring function itself alongside the strategy, relevant framing for why fixed fitness functions overfit.
- General self-evolving agent frameworks (e.g. EvoAgentX) — same reproduce/mutate/select pattern applied outside finance, confirms this is an active research pattern, not a novelty claim.
