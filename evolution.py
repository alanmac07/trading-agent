from __future__ import annotations

"""
EvolutionEngine — core of the Self-Cloning Trading Agent demo.

Key upgrades (v2):
- Infinite simulation: auto-generates synthetic price data (Geometric Brownian
  Motion) when the real CSV runs out, so the demo never stops.
- Randomised start_offset: each run begins at a random date in the dataset.
- Alive-only agent cap: dead agents are kept for the lineage view but no longer
  count against max_agents, so cloning never stalls.
- Smarter graveyard avoidance: clones retry mutation until their genome is
  sufficiently distant from every known failed genome.
"""

import random
import numpy as np
import pandas as pd
from agent import Agent

try:
    from neat_engine.features import compute_extended_features
    HAS_EXTENDED_FEATURES = True
except ImportError:
    HAS_EXTENDED_FEATURES = False


class EvolutionEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        starting_capital: float,
        profit_goal_pct: float,
        loss_limit_pct: float,
        clone_capital_fraction: float = 0.5,
        checkpoint_days: int = 20,
        max_agents: int = 15,
        start_offset: int = 0,
        agent_class=Agent,
        fee_pct: float = 0.001,
    ):
        # Normalise: Date must be a plain column (not the index)
        self.data = data.copy()
        if "Date" not in self.data.columns:
            self.data = self.data.reset_index()
        else:
            self.data = self.data.reset_index(drop=True)

        if "is_synthetic" not in self.data.columns:
            self.data["is_synthetic"] = False

        if HAS_EXTENDED_FEATURES and "Close" in self.data.columns and "High" in self.data.columns:
            try:
                is_synth = self.data["is_synthetic"].copy()
                self.data = compute_extended_features(self.data)
                self.data["is_synthetic"] = is_synth
                self.data = self.data.bfill().ffill().fillna(0.0)
            except Exception:
                pass

        self.starting_capital      = starting_capital
        self.profit_goal_pct       = profit_goal_pct
        self.loss_limit_pct        = loss_limit_pct
        self.clone_capital_fraction = clone_capital_fraction
        self.checkpoint_days       = checkpoint_days
        self.max_agents            = max_agents
        self.agent_class           = agent_class
        self.fee_pct               = fee_pct

        # Clamp start_offset so we always have at least 3 checkpoints of real data
        max_offset = max(0, len(self.data) - checkpoint_days * 3)
        self.start_offset = max(0, min(start_offset, max_offset))

        self._next_id_num: int = 0
        self.agents:    dict[str, Agent] = {}
        self.graveyard: list[dict]       = []
        self.events:    list[dict]       = []
        self.graveyard_pool: float       = 0.0  # capital recovered from dead agents,
                                                  # reinvested into the next clone

        seed = self._make_agent(
            capital=starting_capital, genome=None, parent_id=None, generation=0
        )
        self.agents[seed.id] = seed


        # Calibrate GBM drift/volatility ONCE from real data only.
        # Never recalibrate from a tail that may already contain synthetic
        # rows -- that's what causes runaway exponential drift over time.
        real_prices = self.data.loc[~self.data["is_synthetic"], "Close"].tail(120).astype(float).values
        real_log_rets = np.diff(np.log(real_prices))
        self._gbm_mu = float(np.clip(np.mean(real_log_rets), -0.0015, 0.0015))  # clamp to a sane daily drift
        self._gbm_sigma = max(float(np.std(real_log_rets)), 0.005)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _make_agent(self, capital, genome, parent_id, generation) -> Agent:
        aid = f"Agent_{self._next_id_num}"
        self._next_id_num += 1
        return self.agent_class(aid, capital, genome=genome, parent_id=parent_id, generation=generation)

    def _genome_distance(self, g1: dict, g2: dict) -> float:
        keys = [k for k in g1 if isinstance(g1[k], (int, float))]
        total = sum(
            ((g1[k] - g2.get(k, 0)) / max(abs(g1[k]), abs(g2.get(k, 0)), 1e-6)) ** 2
            for k in keys
        )
        return total ** 0.5

    def _mutate_away_from_graveyard(self, base_genome: dict, attempts: int = 6) -> dict:
        genome = self.agent_class.mutate_genome(base_genome)
        for _ in range(attempts):
            if not any(
                self._genome_distance(genome, g["genome"]) < 0.15
                for g in self.graveyard
            ):
                break
            genome = self.agent_class.mutate_genome(base_genome)
        return genome

    # ── synthetic-data generation ─────────────────────────────────────────────

    @staticmethod
    def _rsi_from_prices(prices: list, window: int = 14) -> list:
        """Compute RSI using Wilder smoothing. Returns a list of the same length."""
        arr = np.array(prices, dtype=float)
        n   = len(arr)
        rsi = np.full(n, float("nan"))
        if n <= window:
            return rsi.tolist()

        deltas = np.diff(arr)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_g = float(np.mean(gains[:window]))
        avg_l = float(np.mean(losses[:window]))
        rsi[window] = 100.0 - 100.0 / (1.0 + avg_g / avg_l) if avg_l else 100.0

        for i in range(window, len(deltas)):
            avg_g = (avg_g * (window - 1) + gains[i]) / window
            avg_l = (avg_l * (window - 1) + losses[i]) / window
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_g / avg_l) if avg_l else 100.0

        return rsi.tolist()

    def _extend_synthetic_data(self, n_days: int = 252) -> None:
        """
        Append n_days of synthetic price rows using Geometric Brownian Motion
        calibrated to the last 120 rows of existing data.
        """
        last       = self.data.iloc[-1]
        last_price = float(last["Close"])
        last_date  = pd.Timestamp(last["Date"])

        # Calibrate GBM from recent returns
        mu    = self._gbm_mu
        sigma = self._gbm_sigma

        # Simulate price path
        shocks = np.random.normal(mu, sigma, n_days)
        prices: list[float] = [last_price]
        for s in shocks:
            prices.append(prices[-1] * float(np.exp(s)))
        prices = prices[1:]  # drop seed

        dates = list(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days))

        # Compute RSI & MA20 with 50-row lookback from existing data
        lookback_prices = list(self.data["Close"].tail(50).astype(float)) + prices
        rsi_full = self._rsi_from_prices(lookback_prices)
        rsi_synth = rsi_full[50:]

        ma_series  = pd.Series(lookback_prices).rolling(20).mean().tolist()
        ma_synth   = ma_series[50:]

        # Synthetic OHLV
        highs  = [p * float(np.random.uniform(1.001, 1.015)) for p in prices]
        lows   = [p * float(np.random.uniform(0.985, 0.999)) for p in prices]
        vols   = [int(np.random.randint(10_000_000, 50_000_000)) for _ in prices]

        n = min(len(dates), len(prices), len(rsi_synth), len(ma_synth))
        new_rows = pd.DataFrame({
            "Date":         dates[:n],
            "Close":        prices[:n],
            "Open":         prices[:n],
            "High":         highs[:n],
            "Low":          lows[:n],
            "Volume":       vols[:n],
            "RSI":          rsi_synth[:n],
            "MA20":         ma_synth[:n],
            "is_synthetic": True,
        }).dropna()

        self.data = pd.concat([self.data, new_rows], ignore_index=True)
        if HAS_EXTENDED_FEATURES:
            try:
                is_synth = self.data["is_synthetic"].copy()
                self.data = compute_extended_features(self.data)
                self.data["is_synthetic"] = is_synth
                self.data = self.data.bfill().ffill().fillna(0.0)
            except Exception:
                pass

    # ── per-day trade execution ───────────────────────────────────────────────

    def _trade_one_day(self, agent: Agent, row, fee_pct: float | None = None) -> None:
        fee = self.fee_pct if fee_pct is None else fee_pct
        price  = float(row["Close"])
        action, reason = agent.decide(row)
        exec_price = price

        if action == "BUY" and agent.capital > 0 and price > 0:
            exec_price = price * (1.0 + fee)
            pos_size = (
                agent.genome.get("position_size_pct", 0.8)
                if isinstance(agent.genome, dict)
                else 0.8
            )
            invest          = agent.capital * pos_size
            agent.position += invest / exec_price
            agent.capital  -= invest

        elif action == "SELL" and agent.position > 0:
            exec_price = price * (1.0 - fee)
            agent.capital  += agent.position * exec_price
            agent.position  = 0.0

        val      = agent.portfolio_value(price)
        date_str = (
            str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"])
        )

        agent.decision_log.append({
            "date":            date_str,
            "price":           round(price, 2),
            "exec_price":      round(exec_price, 2),
            "rsi":             round(float(row["RSI"]), 2) if "RSI" in row and pd.notna(row["RSI"]) else 0.0,
            "action":          action,
            "reason":          reason,
            "portfolio_value": round(val, 2),
            "fee_pct":         fee,
        })
        agent.portfolio_history.append(round(val, 2))
        agent.date_history.append(date_str)

    # ── infinite live generator ───────────────────────────────────────────────

    def run_live(self):
        """
        Infinite generator — yields after every simulated day.

        Yields
        ------
        {"type": "day",        "day_index": int, "row": pd.Series}
        {"type": "checkpoint", "day_index": int, "events": list}

        Behaviour
        ---------
        • Automatically extends self.data with synthetic GBM rows when the
          real CSV is nearly exhausted, so the loop never terminates.
        • Uses an alive-only counter so dead agents never block new clones.
        """
        idx          = self.start_offset
        EXTEND_AHEAD = self.checkpoint_days * 4  # keep at least this many rows ahead

        while True:
            # ── lazy data extension ──────────────────────────────────────────
            if idx + EXTEND_AHEAD >= len(self.data):
                self._extend_synthetic_data(252)

            ck_end = min(idx + self.checkpoint_days, len(self.data))

            # ── trade through this checkpoint window ─────────────────────────
            for di in range(idx, ck_end):
                row = self.data.iloc[di]
                for agent in list(self.agents.values()):
                    if agent.status == "alive":
                        self._trade_one_day(agent, row)
                yield {"type": "day", "day_index": di, "row": row}

            # ── checkpoint: clone / kill ─────────────────────────────────────
            last_price  = float(self.data.iloc[ck_end - 1]["Close"])
            new_events: list[dict] = []

            # Count only ALIVE agents toward the cap
            alive_count = sum(1 for a in self.agents.values() if a.status == "alive")

            for agent in list(self.agents.values()):
                if agent.status != "alive":
                    continue
                ret_pct = agent.return_pct(last_price)

                if ret_pct >= self.profit_goal_pct and alive_count < self.max_agents:
                    self._clone(agent, ck_end, ret_pct)
                    alive_count += 1
                    new_events.append(self.events[-1])

                elif ret_pct <= -self.loss_limit_pct:
                    self._kill(agent, ck_end, ret_pct)
                    alive_count -= 1
                    new_events.append(self.events[-1])

            # ── Auto-Respawn: If all agents die, auto-spawn a new Seed Agent ──
            if sum(1 for a in self.agents.values() if a.status == "alive") == 0:
                new_genome = self._mutate_away_from_graveyard(self.agent_class.random_genome())
                respawn_seed = self._make_agent(
                    capital=self.starting_capital,
                    genome=new_genome,
                    parent_id=None,
                    generation=0,
                )
                self.agents[respawn_seed.id] = respawn_seed
                self.events.append({
                    "type": "RESPAWN",
                    "day_index": ck_end,
                    "agent": respawn_seed.id,
                    "cause": "🌱 Swarm extinct — auto-respawning resilient Seed Agent with Graveyard adaptation",
                })
                new_events.append(self.events[-1])

            if new_events:
                yield {"type": "checkpoint", "day_index": ck_end, "events": new_events}

            idx = ck_end

    # ── clone & kill ──────────────────────────────────────────────────────────

    def _clone(self, parent: Agent, day_index: int, ret_pct: float) -> None:
        transfer        = parent.capital * self.clone_capital_fraction
        parent.capital -= transfer
        # Adjust parent's initial capital base so its return % remains proportional to remaining capital
        parent.initial_capital = max(1.0, parent.initial_capital - transfer)

        # Reinvest any capital recovered from dead agents into this new clone,
        # instead of leaving it sitting idle outside the active population.
        pool_bonus = self.graveyard_pool
        self.graveyard_pool = 0.0
        child_capital = transfer + pool_bonus

        child_genome = self._mutate_away_from_graveyard(parent.genome)
        child = self._make_agent(
            capital=child_capital,
            genome=child_genome,
            parent_id=parent.id,
            generation=parent.generation + 1,
        )
        self.agents[child.id] = child

        self.events.append({
            "type":               "CLONE",
            "day_index":          day_index,
            "parent":             parent.id,
            "child":              child.id,
            "parent_return_pct":  round(ret_pct, 2),
            "parent_genome":      dict(parent.genome),
            "child_genome":       dict(child_genome),
            "pool_bonus":         round(pool_bonus, 2),
        })

    def _kill(self, agent: Agent, day_index: int, ret_pct: float) -> None:
        agent.status          = "dead"
        agent.cause_of_death  = (
            f"Return {ret_pct:.1f}% breached loss limit of -{self.loss_limit_pct:.1f}%"
        )
        # Recovered capital doesn't vanish -- it goes into a pool that funds
        # the next clone instead of sitting dead with this agent forever.
        recovered = agent.capital
        self.graveyard_pool += recovered
        agent.capital = 0.0

        self.graveyard.append({
            "id":     agent.id,
            "genome": dict(agent.genome),
            "cause":  agent.cause_of_death,
        })
        self.events.append({
            "type":      "KILL",
            "day_index": day_index,
            "agent":     agent.id,
            "cause":     agent.cause_of_death,
            "recovered_capital": round(recovered, 2),
        })
