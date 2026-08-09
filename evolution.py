from __future__ import annotations

"""
evolution.py — Evolutionary Swarm Engine exclusively powered by NEAT neural networks.

Every agent in the swarm is a compiled NEAT feedforward neural network (NeatTradingAgent)
that evaluates market features, portfolio states, and risk signals via risk_manager.py.
Cloned agents evolve through NEAT genome mutation operators (weight perturbations, connection mutations).
"""

import os
import copy
import random
import pickle
import neat
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from neat_engine.network_agent import NeatTradingAgent
from neat_engine.risk_manager import decide_trade, TradeDecision
from neat_engine.features import compute_extended_features

# Distinct vibrant colors for agent visualization
AGENT_COLORS = [
    "#00D4FF", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6",
    "#3B82F6", "#EF4444", "#14B8A6", "#F97316", "#A855F7",
    "#06B6D4", "#84CC16", "#E11D48", "#6366F1", "#D946EF"
]


class NeatSwarmAgent:
    """
    A live trading agent in the swarm whose actions are dictated exclusively
    by a compiled NEAT neural network genome + risk manager.
    """
    def __init__(
        self,
        agent_id: str,
        capital: float,
        genome: neat.DefaultGenome,
        config: neat.Config,
        parent_id: Optional[str] = None,
        generation: int = 0,
        color: Optional[str] = None,
        spawn_step: int = 0,
    ):
        self.id = agent_id
        self.capital = float(capital)
        self.initial_capital = float(capital)
        self.genome = genome
        self.config = config
        self.parent_id = parent_id
        self.generation = generation
        self.color = color or random.choice(AGENT_COLORS)
        self.spawn_step = spawn_step
        self.death_step: Optional[int] = None

        # Compile NEAT neural network
        self.neat_agent = NeatTradingAgent(genome, config)

        # Portfolio and position state
        self.position: float = 0.0
        self.entry_price: float = 0.0
        self.peak_price: float = 0.0
        self.trailing_stop_pct: float = 8.0
        self.status: str = "alive"
        self.cause_of_death: Optional[str] = None

        # Execution logs and histories
        self.decision_log: List[Dict[str, Any]] = []
        self.portfolio_history: List[float] = [round(self.capital, 2)]
        self.date_history: List[str] = []
        self.equity_history: List[Dict[str, Any]] = [{"step": spawn_step, "capital": round(self.capital, 2)}]

    def in_position(self) -> bool:
        return self.position > 0.00001

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if not self.in_position() or self.entry_price <= 0:
            return 0.0
        return ((current_price / self.entry_price) - 1.0) * 100.0

    def portfolio_value(self, current_price: float) -> float:
        return self.capital + (self.position * current_price)

    def return_pct(self, current_price: float) -> float:
        if self.initial_capital <= 0:
            return 0.0
        val = self.portfolio_value(current_price)
        return ((val - self.initial_capital) / self.initial_capital) * 100.0

    def decide(self, row: pd.Series) -> tuple[str, str, float, float]:
        """
        Activates the NEAT neural network and risk manager.
        Returns: (action, reason, position_size_pct, trailing_stop_pct)
        """
        current_price = float(row["Close"])
        in_pos = self.in_position()
        unrealized = self.unrealized_pnl_pct(current_price)

        # 1. Activate neural network for raw signals
        action_sig, risk_sig = self.neat_agent.decide(row, in_pos, unrealized)

        # 2. Risk manager maps raw signals to trading decision
        decision: TradeDecision = decide_trade(action_sig, risk_sig, in_pos)
        action = decision.action
        pos_size = decision.position_size_pct
        stop_pct = decision.trailing_stop_pct
        reason = f"NEAT Net (act={action_sig:+.2f}, risk={risk_sig:+.2f}) -> {action}"

        # 3. Dynamic Trailing Stop check if currently holding
        if in_pos:
            self.peak_price = max(self.peak_price, current_price)
            if self.peak_price > 0:
                drawdown_from_peak = ((self.peak_price - current_price) / self.peak_price) * 100.0
                if drawdown_from_peak >= self.trailing_stop_pct:
                    action = "SELL"
                    reason = f"Trailing stop triggered (-{drawdown_from_peak:.1f}% from peak ${self.peak_price:.2f})"

        return action, reason, pos_size, stop_pct


class EvolutionEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        starting_capital: float = 10000.0,
        profit_goal_pct: float = 8.0,
        loss_limit_pct: float = 10.0,
        clone_capital_fraction: float = 0.5,
        checkpoint_days: int = 20,
        max_agents: int = 15,
        start_offset: int = 0,
        fee_pct: float = 0.001,
        config_path: Optional[str] = None,
        genome_path: Optional[str] = None,
    ):
        self.data = data.copy()
        if "Date" not in self.data.columns:
            self.data = self.data.reset_index()
        else:
            self.data = self.data.reset_index(drop=True)

        if "is_synthetic" not in self.data.columns:
            self.data["is_synthetic"] = False

        # Ensure all extended features are computed
        if "Close" in self.data.columns and "High" in self.data.columns:
            try:
                is_synth = self.data["is_synthetic"].copy()
                self.data = compute_extended_features(self.data)
                self.data["is_synthetic"] = is_synth
                self.data = self.data.bfill().ffill().fillna(0.0)
            except Exception:
                pass

        self.starting_capital = starting_capital
        self.profit_goal_pct = profit_goal_pct
        self.loss_limit_pct = loss_limit_pct
        self.clone_capital_fraction = clone_capital_fraction
        self.checkpoint_days = checkpoint_days
        self.max_agents = max_agents
        self.fee_pct = fee_pct

        # Clamp start_offset so we always have at least 3 checkpoints of real data
        max_offset = max(0, len(self.data) - checkpoint_days * 3)
        self.start_offset = max(0, min(start_offset, max_offset))
        self.current_day = self.start_offset

        self._next_id_num: int = 0
        self.agents: Dict[str, NeatSwarmAgent] = {}
        self.all_agents: Dict[str, NeatSwarmAgent] = {}
        self.graveyard: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.graveyard_pool: float = 0.0

        # Load NEAT Config
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "neat_engine", "config-feedforward.txt")
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        
        # Inject innovation tracker & node indexer for manual mutation
        from itertools import count
        self.neat_config.genome_config.innovation_tracker = neat.InnovationTracker()
        self.neat_config.genome_config.node_indexer = count(100000)

        # Load Seed NEAT Genome
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if genome_path is None or not os.path.exists(genome_path):
            fallback_candidates = [
                genome_path,
                os.path.join(base_dir, "results", "best_genome.pkl"),
                os.path.join(base_dir, "best_genome_AAPL.pkl"),
            ]
            for cand in fallback_candidates:
                if cand and os.path.exists(cand):
                    genome_path = cand
                    break

        if genome_path is None or not os.path.exists(genome_path):
            raise FileNotFoundError("Could not find any seed NEAT genome pickle file.")

        with open(genome_path, "rb") as f:
            self.base_genome = pickle.load(f)

        # Spawn Seed Agent
        seed = self._make_agent(
            capital=starting_capital,
            genome=copy.deepcopy(self.base_genome),
            parent_id=None,
            generation=0,
            color="#00D4FF",
            spawn_step=self.start_offset,
        )
        self.agents[seed.id] = seed

        # Calibrate GBM drift/volatility from real historical prices
        real_prices = self.data.loc[~self.data["is_synthetic"], "Close"].tail(120).astype(float).values
        if len(real_prices) > 1:
            real_log_rets = np.diff(np.log(real_prices))
            self._gbm_mu = float(np.clip(np.mean(real_log_rets), -0.0015, 0.0015))
            self._gbm_sigma = max(float(np.std(real_log_rets)), 0.005)
        else:
            self._gbm_mu = 0.0002
            self._gbm_sigma = 0.015

    def _make_agent(
        self,
        capital: float,
        genome: neat.DefaultGenome,
        parent_id: Optional[str] = None,
        generation: int = 0,
        color: Optional[str] = None,
        spawn_step: int = 0,
    ) -> NeatSwarmAgent:
        aid = f"NEAT_{self._next_id_num}"
        self._next_id_num += 1
        agent_color = color or AGENT_COLORS[self._next_id_num % len(AGENT_COLORS)]
        ag = NeatSwarmAgent(
            agent_id=aid,
            capital=capital,
            genome=genome,
            config=self.neat_config,
            parent_id=parent_id,
            generation=generation,
            color=agent_color,
            spawn_step=spawn_step,
        )
        self.all_agents[ag.id] = ag
        return ag

    def _extend_synthetic_data(self, n_days: int = 252) -> None:
        last = self.data.iloc[-1]
        last_price = float(last["Close"])
        last_date = pd.Timestamp(last["Date"])

        shocks = np.random.normal(self._gbm_mu, self._gbm_sigma, n_days)
        prices = [last_price]
        for s in shocks:
            prices.append(prices[-1] * float(np.exp(s)))
        prices = prices[1:]

        dates = list(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days))
        highs = [p * float(np.random.uniform(1.001, 1.015)) for p in prices]
        lows = [p * float(np.random.uniform(0.985, 0.999)) for p in prices]
        vols = [int(np.random.randint(10_000_000, 50_000_000)) for _ in prices]

        n = min(len(dates), len(prices))
        new_rows = pd.DataFrame({
            "Date": dates[:n],
            "Close": prices[:n],
            "Open": prices[:n],
            "High": highs[:n],
            "Low": lows[:n],
            "Volume": vols[:n],
            "is_synthetic": True,
        })
        new_rows = compute_extended_features(new_rows)
        self.data = pd.concat([self.data, new_rows], ignore_index=True).bfill().ffill().fillna(0.0)

    def _trade_one_day(self, agent: NeatSwarmAgent, row: pd.Series, day_index: int) -> None:
        price = float(row["Close"])
        action, reason, pos_size_pct, stop_pct = agent.decide(row)
        exec_price = price
        entry_price_used = agent.entry_price

        if action == "BUY" and agent.capital > 0 and price > 0:
            exec_price = price * (1.0 + self.fee_pct)
            invest = agent.capital * pos_size_pct
            shares = invest / exec_price
            agent.position += shares
            agent.capital -= invest
            agent.entry_price = exec_price
            agent.peak_price = exec_price
            agent.trailing_stop_pct = stop_pct
            entry_price_used = exec_price

        elif action == "SELL" and agent.in_position():
            exec_price = price * (1.0 - self.fee_pct)
            is_win = exec_price > agent.entry_price
            reason += f" (Win: {is_win})"
            agent.capital += agent.position * exec_price
            agent.position = 0.0
            agent.entry_price = 0.0
            agent.peak_price = 0.0

        val = agent.portfolio_value(price)
        date_str = str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"])

        agent.decision_log.append({
            "date": date_str,
            "price": round(price, 2),
            "exec_price": round(exec_price, 2),
            "action": action,
            "reason": reason,
            "portfolio_value": round(val, 2),
            "fee_pct": self.fee_pct,
            "entry_price": round(entry_price_used, 2)
        })
        agent.portfolio_history.append(round(val, 2))
        agent.date_history.append(date_str)
        agent.equity_history.append({"step": day_index, "capital": round(val, 2)})

    def run_live(self):
        idx = self.start_offset
        EXTEND_AHEAD = self.checkpoint_days * 4

        while True:
            if idx + EXTEND_AHEAD >= len(self.data):
                self._extend_synthetic_data(252)

            ck_end = min(idx + self.checkpoint_days, len(self.data))

            for di in range(idx, ck_end):
                self.current_day = di
                row = self.data.iloc[di]
                for agent in list(self.agents.values()):
                    if agent.status == "alive":
                        self._trade_one_day(agent, row, di)
                yield {"type": "day", "day_index": di, "row": row}

            # Checkpoint evolution (cloning & culling)
            last_price = float(self.data.iloc[ck_end - 1]["Close"])
            new_events: List[Dict[str, Any]] = []
            alive_agents = [a for a in self.agents.values() if a.status == "alive"]

            for agent in alive_agents:
                ret_pct = agent.return_pct(last_price)

                if ret_pct >= self.profit_goal_pct and len(alive_agents) < self.max_agents:
                    self._clone(agent, ck_end, ret_pct)
                    alive_agents.append(self.agents[self.events[-1]["child"]])
                    new_events.append(self.events[-1])

                elif ret_pct <= -self.loss_limit_pct:
                    self._kill(agent, ck_end, ret_pct)
                    new_events.append(self.events[-1])

            # Auto-Respawn if entire population dies
            if sum(1 for a in self.agents.values() if a.status == "alive") == 0:
                respawn_genome = copy.deepcopy(self.base_genome)
                respawn_genome.mutate(self.neat_config.genome_config)
                seed = self._make_agent(
                    capital=self.starting_capital,
                    genome=respawn_genome,
                    parent_id=None,
                    generation=0,
                    spawn_step=ck_end,
                )
                self.agents[seed.id] = seed
                respawn_event = {
                    "type": "RESPAWN",
                    "day_index": ck_end,
                    "agent": seed.id,
                    "cause": "🌱 Swarm extinct — respawned adapted NEAT Champion genome",
                }
                self.events.append(respawn_event)
                new_events.append(respawn_event)

            if new_events:
                yield {"type": "checkpoint", "day_index": ck_end, "events": new_events}

            idx = ck_end

    def _clone(self, parent: NeatSwarmAgent, day_index: int, ret_pct: float) -> None:
        transfer = parent.capital * self.clone_capital_fraction
        parent.capital -= transfer
        parent.initial_capital = max(1.0, parent.initial_capital - transfer)

        # Record parent's updated equity state after cloning capital transfer
        last_price = float(self.data.iloc[day_index - 1]["Close"])
        parent_val = parent.portfolio_value(last_price)
        parent.equity_history.append({"step": day_index, "capital": round(parent_val, 2)})

        pool_bonus = self.graveyard_pool
        self.graveyard_pool = 0.0
        child_capital = transfer + pool_bonus

        # Mutate genome using NEAT genetic operators
        child_genome = copy.deepcopy(parent.genome)
        child_genome.mutate(self.neat_config.genome_config)

        child = self._make_agent(
            capital=child_capital,
            genome=child_genome,
            parent_id=parent.id,
            generation=parent.generation + 1,
            spawn_step=day_index,
        )
        self.agents[child.id] = child

        self.events.append({
            "type": "CLONE",
            "day_index": day_index,
            "parent": parent.id,
            "child": child.id,
            "parent_return_pct": round(ret_pct, 2),
            "pool_bonus": round(pool_bonus, 2),
        })

    def _kill(self, agent: NeatSwarmAgent, day_index: int, ret_pct: float) -> None:
        agent.status = "dead"
        agent.death_step = day_index
        agent.cause_of_death = f"Return {ret_pct:.1f}% breached loss limit -{self.loss_limit_pct:.1f}%"
        
        # Liquidate open position
        last_price = float(self.data.iloc[day_index - 1]["Close"])
        if agent.in_position():
            agent.capital += agent.position * last_price * (1.0 - self.fee_pct)
            agent.position = 0.0

        recovered = agent.capital
        self.graveyard_pool += recovered
        agent.capital = 0.0

        # Record final liquidated point in equity history
        agent.equity_history.append({"step": day_index, "capital": round(recovered, 2)})

        self.graveyard.append({
            "id": agent.id,
            "cause": agent.cause_of_death,
        })
        self.events.append({
            "type": "KILL",
            "day_index": day_index,
            "agent": agent.id,
            "cause": agent.cause_of_death,
            "recovered_capital": round(recovered, 2),
        })
