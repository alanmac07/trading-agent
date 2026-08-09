import os
import random
import copy
import neat
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from evolution import EvolutionEngine, NeatSwarmAgent
from neat_engine.features import MARKET_FEATURE_COLUMNS, PORTFOLIO_FEATURE_COLUMNS, ALL_FEATURE_COLUMNS
from neat_engine.risk_manager import decide_trade
from neat_engine.build_dataset import get_market_regime, MARKET_REGIMES

app = FastAPI(title="Trading Swarm NEAT Backend")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_genome_for_ticker(ticker: str) -> tuple[str, str]:
    """
    Detects the asset class regime for the ticker and resolves the appropriate
    specialized genome path, falling back gracefully to universal best_genome.pkl.
    """
    regime = get_market_regime(ticker)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    specialized_path = os.path.join(base_dir, "results", f"best_genome_{regime}.pkl")
    default_path = os.path.join(base_dir, "results", "best_genome.pkl")
    fallback_path = os.path.join(base_dir, "best_genome_AAPL.pkl")

    if os.path.exists(specialized_path):
        return regime, specialized_path
    elif os.path.exists(default_path):
        return regime, default_path
    elif os.path.exists(fallback_path):
        return regime, fallback_path
    else:
        return regime, specialized_path


# Global state to hold the engine and data
class EngineState:
    def __init__(self):
        self.engine: Optional[EvolutionEngine] = None
        self.generator = None
        self.data_df: pd.DataFrame = pd.DataFrame()
        self.tickers: List[str] = []
        self.current_ticker: Optional[str] = None
        self.current_regime: Optional[str] = None
        self.current_genome_path: Optional[str] = None
        self.running: bool = False
        self.stopped: bool = False
        self.load_data()

    def load_data(self):
        data_path = os.path.join(os.path.dirname(__file__), "data", "price_data_multi.csv")
        if os.path.exists(data_path):
            self.data_df = pd.read_csv(data_path, parse_dates=["Date"])
            if "Ticker" in self.data_df.columns:
                self.tickers = sorted(self.data_df["Ticker"].unique().tolist())
            else:
                self.tickers = ["UNKNOWN"]
        else:
            self.data_df = pd.DataFrame()
            self.tickers = []

state = EngineState()

class StartRequest(BaseModel):
    ticker: str
    starting_capital: float = 10000.0
    profit_goal_pct: float = 8.0
    loss_limit_pct: float = 10.0
    fee_pct: float = 0.001
    clone_frac: float = 0.5
    checkpoint_days: int = 20
    max_agents: int = 15
    start_offset: Optional[int] = None

class StepRequest(BaseModel):
    days: int = 1

def get_ticker_data(ticker: str) -> pd.DataFrame:
    if state.data_df.empty:
        state.load_data()
        if state.data_df.empty:
            raise HTTPException(status_code=404, detail="Data file data/price_data_multi.csv not found. Run fetch_data.py first.")
    
    if "Ticker" in state.data_df.columns:
        df = state.data_df[state.data_df["Ticker"] == ticker].copy()
    else:
        df = state.data_df.copy()
        
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_agent_stats(ag: NeatSwarmAgent, last_price: float) -> Dict[str, Any]:
    current_val = ag.portfolio_value(last_price)
    peak_val = max(ag.portfolio_history) if ag.portfolio_history else current_val
    
    # Calculate drawdown
    max_dd = 0.0
    running_peak = 0.0
    for p in ag.portfolio_history:
        if p > running_peak:
            running_peak = p
        if running_peak > 0:
            dd = ((running_peak - p) / running_peak) * 100.0
            if dd > max_dd:
                max_dd = dd

    # Count trades and win rate
    trades_executed = [d for d in ag.decision_log if d.get("action") in ["BUY", "SELL"]]
    sell_trades = [d for d in ag.decision_log if d.get("action") == "SELL"]
    win_trades = [d for d in sell_trades if d.get("exec_price", 0) > d.get("entry_price", float('inf'))]
    win_rate = (len(win_trades) / len(sell_trades) * 100.0) if len(sell_trades) > 0 else 0.0

    return {
        "agent_id": ag.id,
        "id": ag.id,
        "status": ag.status,
        "generation": ag.generation,
        "color": ag.color,
        "parent_id": ag.parent_id,
        "current_capital": float(round(current_val, 2)),
        "portfolio_value": float(round(current_val, 2)),
        "peak_portfolio_value": float(round(peak_val, 2)),
        "max_drawdown_pct": float(round(max_dd, 2)),
        "return_pct": float(round(ag.return_pct(last_price), 2)),
        "cash": float(round(ag.capital, 2)),
        "position": float(round(ag.position, 6)),
        "trailing_stop_pct": float(round(ag.trailing_stop_pct, 2)),
        "total_trades": len(trades_executed),
        "win_rate_pct": float(round(win_rate, 2)),
        "cause_of_death": ag.cause_of_death,
        "spawn_step": getattr(ag, "spawn_step", 0),
        "death_step": getattr(ag, "death_step", None),
        "equity_history": getattr(ag, "equity_history", []),
        "portfolio_history": ag.portfolio_history[-60:],  # sparkline
        "date_history": ag.date_history[-60:],
    }

@app.get("/api/tickers")
def get_tickers():
    if not state.tickers:
        state.load_data()
    return {"tickers": state.tickers}

@app.get("/api/market")
def get_market_data(ticker: str):
    df = get_ticker_data(ticker)
    df.fillna(0.0, inplace=True)
    
    # Date formatting
    if hasattr(df["Date"], "dt"):
        df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")
    else:
        df["DateStr"] = df["Date"].astype(str).str.split(" ").str[0]
    
    cols = ["DateStr", "Open", "High", "Low", "Close", "Volume"]
    if "RSI" in df.columns:
        cols.append("RSI")
    if "MA20" in df.columns:
        cols.append("MA20")

    records = df[cols].to_dict(orient="records")
    return {"ticker": ticker, "data": records}

@app.post("/api/sim/start")
def start_simulation(req: StartRequest):
    df = get_ticker_data(req.ticker)
    if len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data for ticker {req.ticker}")
        
    if req.start_offset is not None:
        start_offset = req.start_offset
    else:
        start_offset = 0

    regime, genome_path = resolve_genome_for_ticker(req.ticker)
    print(f"[API] Starting simulation for '{req.ticker}' (Regime: {regime.upper()}) using genome: {genome_path}")

    engine = EvolutionEngine(
        data=df,
        starting_capital=req.starting_capital,
        profit_goal_pct=req.profit_goal_pct,
        loss_limit_pct=req.loss_limit_pct,
        clone_capital_fraction=req.clone_frac,
        checkpoint_days=req.checkpoint_days,
        max_agents=req.max_agents,
        start_offset=start_offset,
        fee_pct=req.fee_pct,
        genome_path=genome_path,
    )

    state.engine = engine
    state.generator = engine.run_live()
    state.current_ticker = req.ticker
    state.current_regime = regime
    state.current_genome_path = genome_path
    state.running = True
    state.stopped = False
    
    return get_status()

@app.post("/api/sim/step")
def step_simulation(req: StepRequest):
    if state.engine is None or state.generator is None:
        raise HTTPException(status_code=400, detail="Simulation not started")
    
    all_agents_map = getattr(state.engine, "all_agents", state.engine.agents)
    last_price = float(state.engine.data["Close"].iloc[state.engine.current_day])

    if state.stopped:
        return {
            "days_advanced": 0,
            "candles": [],
            "events": [],
            "agents": [calculate_agent_stats(ag, last_price) for ag in all_agents_map.values()],
            "trades": [],
            "current_day": state.engine.current_day,
            "graveyard_pool": float(state.engine.graveyard_pool),
            "stopped": True
        }
        
    events = []
    candles = []
    
    for _ in range(req.days):
        try:
            item = next(state.generator)
            if item["type"] == "day":
                row = item["row"]
                date_val = str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"]).split(" ")[0]
                candles.append({
                    "day_index": item["day_index"],
                    "DateStr": date_val,
                    "Open": float(row.get("Open", row["Close"])),
                    "High": float(row.get("High", row["Close"])),
                    "Low": float(row.get("Low", row["Close"])),
                    "Close": float(row["Close"]),
                    "Volume": float(row.get("Volume", 0)),
                    "MA20": float(row.get("MA20", 0.0)),
                    "RSI": float(row.get("RSI", 50.0)),
                })
            elif item["type"] == "checkpoint":
                events.extend(item.get("events", []))
        except StopIteration:
            state.running = False
            state.stopped = True
            break
            
    # Serialize all agents (active, dead, cloned) with rich time-series metrics
    all_agents_map = getattr(state.engine, "all_agents", state.engine.agents)
    last_price = float(state.engine.data["Close"].iloc[state.engine.current_day])
    agents_data = [calculate_agent_stats(ag, last_price) for ag in all_agents_map.values()]
        
    # Get recent trades from active step
    trades = []
    for aid, ag in all_agents_map.items():
        if ag.decision_log:
            recent = ag.decision_log[-req.days:]
            for dec in recent:
                if dec.get("action") in ["BUY", "SELL"]:
                    trades.append({
                        "agent_id": aid,
                        "date": str(dec["date"]),
                        "action": dec["action"],
                        "price": dec.get("exec_price", dec["price"]),
                        "reason": dec.get("reason", ""),
                        "color": ag.color
                    })
                    
    return {
        "days_advanced": len(candles),
        "candles": candles,
        "events": events,
        "agents": agents_data,
        "trades": trades,
        "current_day": state.engine.current_day,
        "graveyard_pool": float(state.engine.graveyard_pool),
        "stopped": state.stopped,
    }

@app.post("/api/sim/control")
def control_simulation(action: str):
    if action == "reset":
        state.engine = None
        state.generator = None
        state.running = False
        state.stopped = False
        return {"status": "reset"}
    elif action == "pause":
        state.running = False
        return {"status": "paused"}
    elif action == "resume":
        if state.engine and not state.stopped:
            state.running = True
        return {"status": "resumed"}
    return {"status": "ok"}

@app.post("/api/sim/stop")
def stop_simulation():
    """Completely halts background execution while preserving state for analytics."""
    state.running = False
    state.stopped = True
    return {"status": "stopped", "current_day": state.engine.current_day if state.engine else 0}

@app.get("/api/sim/status")
def get_status():
    if state.engine is None:
        return {"running": False, "stopped": False}
        
    all_agents_map = getattr(state.engine, "all_agents", state.engine.agents)
    last_price = float(state.engine.data["Close"].iloc[state.engine.current_day])
    agents_data = [calculate_agent_stats(ag, last_price) for ag in all_agents_map.values()]
        
    return {
        "running": state.running,
        "stopped": state.stopped,
        "ticker": state.current_ticker,
        "regime": state.current_regime,
        "genome_path": state.current_genome_path,
        "current_day": state.engine.current_day,
        "graveyard_pool": float(state.engine.graveyard_pool),
        "agents": agents_data
    }

@app.get("/api/agent/brain")
def get_agent_brain(agent_id: str):
    """Provides deep-dive inspection into an agent's neural decision engine."""
    if state.engine is None:
        raise HTTPException(status_code=400, detail="Simulation not started")
    if agent_id not in state.engine.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
    ag: NeatSwarmAgent = state.engine.agents[agent_id]
    row = state.engine.data.iloc[state.engine.current_day]
    current_price = float(row["Close"])
    in_pos = ag.in_position()
    unrealized = ag.unrealized_pnl_pct(current_price)
    
    # 15 normalized input features
    market_features = {}
    for col in MARKET_FEATURE_COLUMNS:
        market_features[col] = float(row.get(col, 0.0))
        
    pnl_norm = float(np.clip(unrealized / 20.0, -1.0, 1.0)) if in_pos else 0.0
    portfolio_features = {
        "in_position": 1.0 if in_pos else 0.0,
        "unrealized_pnl_norm": pnl_norm,
        "unrealized_pnl_pct": unrealized
    }
    
    all_inputs = {**market_features, **portfolio_features}
    
    # Neural activation signals
    action_sig, risk_sig = ag.neat_agent.decide(row, in_pos, unrealized)
    decision = decide_trade(action_sig, risk_sig, in_pos)
    
    # Format decision log for inspection
    logs = []
    for d in ag.decision_log:
        logs.append({
            "date": d.get("date", ""),
            "price": d.get("price", 0.0),
            "exec_price": d.get("exec_price", d.get("price", 0.0)),
            "action": d.get("action", "HOLD"),
            "reason": d.get("reason", ""),
            "portfolio_value": d.get("portfolio_value", 0.0),
            "fee_pct": d.get("fee_pct", state.engine.fee_pct),
        })
        
    return {
        "agent_id": ag.id,
        "status": ag.status,
        "color": ag.color,
        "generation": ag.generation,
        "parent_id": ag.parent_id,
        "capital": float(ag.capital),
        "initial_capital": float(ag.initial_capital),
        "position": float(ag.position),
        "portfolio_value": float(ag.portfolio_value(current_price)),
        "return_pct": float(ag.return_pct(current_price)),
        "current_price": current_price,
        "market_features": market_features,
        "portfolio_features": portfolio_features,
        "all_inputs": all_inputs,
        "action_signal": float(action_sig),
        "risk_signal": float(risk_sig),
        "decision": {
            "action": decision.action,
            "position_size_pct": float(decision.position_size_pct),
            "trailing_stop_pct": float(decision.trailing_stop_pct),
            "reason": f"NEAT Net (act={action_sig:+.2f}, risk={risk_sig:+.2f}) -> {decision.action}"
        },
        "decision_log": logs,
        "total_decisions": len(logs)
    }

@app.get("/api/agent/lineage")
def get_agent_lineage():
    """Builds hierarchical lineage tree and calculates exact genome mutation diffs."""
    if state.engine is None:
        return {"nodes": [], "edges": [], "events": []}
        
    last_price = float(state.engine.data["Close"].iloc[state.engine.current_day])
    nodes = []
    edges = []
    
    for aid, ag in state.engine.agents.items():
        # Compute genome mutation differences relative to parent
        mutations = []
        parent_genome = None
        if ag.parent_id and ag.parent_id in state.engine.agents:
            parent_genome = state.engine.agents[ag.parent_id].genome
        elif ag.parent_id is None and ag.id != "NEAT_0":
            parent_genome = state.engine.base_genome

        if parent_genome is not None and ag.genome is not None:
            # Check modified connections
            for conn_key, conn_gene in ag.genome.connections.items():
                if conn_key in parent_genome.connections:
                    p_conn = parent_genome.connections[conn_key]
                    if abs(p_conn.weight - conn_gene.weight) > 1e-4 or p_conn.enabled != conn_gene.enabled:
                        mutations.append({
                            "type": "WEIGHT_PERTURBATION",
                            "synapse": f"Node {conn_key[0]} -> Node {conn_key[1]}",
                            "in_node": conn_key[0],
                            "out_node": conn_key[1],
                            "parent_weight": round(float(p_conn.weight), 4),
                            "child_weight": round(float(conn_gene.weight), 4),
                            "weight_delta": round(float(conn_gene.weight - p_conn.weight), 4),
                            "enabled": conn_gene.enabled,
                            "was_enabled": p_conn.enabled
                        })
                else:
                    mutations.append({
                        "type": "NEW_SYNAPSE",
                        "synapse": f"Node {conn_key[0]} -> Node {conn_key[1]}",
                        "in_node": conn_key[0],
                        "out_node": conn_key[1],
                        "child_weight": round(float(conn_gene.weight), 4),
                        "enabled": conn_gene.enabled
                    })
                    
            # Check added nodes
            for node_key in ag.genome.nodes:
                if node_key not in parent_genome.nodes:
                    mutations.append({
                        "type": "NEW_NEURON_NODE",
                        "node_id": node_key,
                        "bias": round(float(ag.genome.nodes[node_key].bias), 4),
                        "response": round(float(ag.genome.nodes[node_key].response), 4),
                        "activation": ag.genome.nodes[node_key].activation
                    })

        nodes.append({
            "id": ag.id,
            "label": ag.id,
            "status": ag.status,
            "generation": ag.generation,
            "parent_id": ag.parent_id,
            "color": ag.color,
            "portfolio_value": float(round(ag.portfolio_value(last_price), 2)),
            "return_pct": float(round(ag.return_pct(last_price), 2)),
            "nodes_count": len(ag.genome.nodes) if ag.genome else 0,
            "connections_count": len(ag.genome.connections) if ag.genome else 0,
            "mutations_count": len(mutations),
            "mutations": mutations,
            "cause_of_death": ag.cause_of_death,
        })
        
        if ag.parent_id:
            edges.append({
                "source": ag.parent_id,
                "target": ag.id,
                "label": f"Gen {ag.generation}",
            })
            
    return {
        "nodes": nodes,
        "edges": edges,
        "events": state.engine.events,
        "total_agents": len(nodes),
        "graveyard_pool": float(state.engine.graveyard_pool),
    }

@app.get("/api/overview")
def get_overview_metrics():
    """Returns top-level aggregate statistics across the entire swarm."""
    if state.engine is None:
        return {"running": False}
        
    last_price = float(state.engine.data["Close"].iloc[state.engine.current_day])
    agents = list(state.engine.agents.values())
    alive = [a for a in agents if a.status == "alive"]
    dead = [a for a in agents if a.status == "dead"]
    
    total_val = sum(a.portfolio_value(last_price) for a in agents)
    total_cash = sum(a.capital for a in agents)
    total_market = sum(a.position * last_price for a in agents)
    
    all_trades = []
    for a in agents:
        all_trades.extend([d for d in a.decision_log if d.get("action") in ["BUY", "SELL"]])
        
    all_sells = []
    for a in agents:
        all_sells.extend([d for d in a.decision_log if d.get("action") == "SELL"])
        
    win_trades = [d for d in all_sells if d.get("exec_price", 0) > d.get("entry_price", float('inf'))]
    win_rate = (len(win_trades) / len(all_sells) * 100.0) if all_sells else 0.0
    
    return {
        "current_day": state.engine.current_day,
        "total_equity": round(total_val, 2),
        "total_cash": round(total_cash, 2),
        "total_market_exposure": round(total_market, 2),
        "graveyard_pool": round(state.engine.graveyard_pool, 2),
        "alive_count": len(alive),
        "dead_count": len(dead),
        "total_agents": len(agents),
        "total_trades": len(all_trades),
        "win_rate_pct": round(win_rate, 2),
        "stopped": state.stopped,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

