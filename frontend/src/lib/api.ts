const API_BASE = "http://localhost:8000/api";

export interface MarketData {
  DateStr: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  RSI?: number;
  MA20?: number;
  day_index?: number;
}

export interface EquityPoint {
  step: number;
  capital: number;
}

export interface AgentData {
  id: string;
  agent_id?: string;
  status: "alive" | "dead" | "cloned" | string;
  generation: number;
  color: string;
  parent_id: string | null;
  current_capital?: number;
  portfolio_value: number;
  peak_portfolio_value?: number;
  max_drawdown_pct?: number;
  return_pct: number;
  cash: number;
  position: number;
  trailing_stop_pct?: number;
  total_trades?: number;
  win_rate_pct?: number;
  cause_of_death?: string;
  spawn_step?: number;
  death_step?: number | null;
  equity_history?: EquityPoint[];
  portfolio_history?: number[];
  date_history?: string[];
}

export interface TradeData {
  agent_id: string;
  date: string;
  action: string;
  price: number;
  reason: string;
  color: string;
}

export interface StepResponse {
  days_advanced: number;
  candles: MarketData[];
  events: any[];
  agents: AgentData[];
  trades: TradeData[];
  current_day: number;
  graveyard_pool: number;
  stopped?: boolean;
}

export interface SimStatus {
  running: boolean;
  stopped?: boolean;
  current_day?: number;
  graveyard_pool?: number;
  agents?: AgentData[];
}

export interface DecisionLogEntry {
  date: string;
  price: number;
  exec_price: number;
  action: "BUY" | "SELL" | "HOLD";
  reason: string;
  portfolio_value: number;
  fee_pct?: number;
}

export interface AgentBrainData {
  agent_id: string;
  status: string;
  color: string;
  generation: number;
  parent_id: string | null;
  capital: number;
  initial_capital: number;
  position: number;
  portfolio_value: number;
  return_pct: number;
  current_price: number;
  market_features: Record<string, number>;
  portfolio_features: {
    in_position: number;
    unrealized_pnl_norm: number;
    unrealized_pnl_pct: number;
  };
  all_inputs: Record<string, number>;
  action_signal: number;
  risk_signal: number;
  decision: {
    action: "BUY" | "SELL" | "HOLD";
    position_size_pct: number;
    trailing_stop_pct: number;
    reason: string;
  };
  decision_log: DecisionLogEntry[];
  total_decisions: number;
}

export interface GenomeMutation {
  type: "WEIGHT_PERTURBATION" | "NEW_SYNAPSE" | "NEW_NEURON_NODE";
  synapse?: string;
  in_node?: number;
  out_node?: number;
  node_id?: number;
  parent_weight?: number;
  child_weight?: number;
  weight_delta?: number;
  enabled?: boolean;
  was_enabled?: boolean;
  bias?: number;
  response?: number;
  activation?: string;
}

export interface LineageNode {
  id: string;
  label: string;
  status: "alive" | "dead";
  generation: number;
  parent_id: string | null;
  color: string;
  portfolio_value: number;
  return_pct: number;
  nodes_count: number;
  connections_count: number;
  mutations_count: number;
  mutations: GenomeMutation[];
  cause_of_death?: string;
}

export interface LineageEdge {
  source: string;
  target: string;
  label?: string;
}

export interface LineageData {
  nodes: LineageNode[];
  edges: LineageEdge[];
  events: any[];
  total_agents: number;
  graveyard_pool: number;
}

export interface OverviewMetrics {
  current_day: number;
  total_equity: number;
  total_cash: number;
  total_market_exposure: number;
  graveyard_pool: number;
  alive_count: number;
  dead_count: number;
  total_agents: number;
  total_trades: number;
  win_rate_pct: number;
  stopped?: boolean;
}

export async function fetchTickers(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/tickers`);
  if (!res.ok) throw new Error("Failed to fetch tickers");
  const data = await res.json();
  return data.tickers;
}

export async function fetchMarketData(ticker: string): Promise<MarketData[]> {
  const res = await fetch(`${API_BASE}/market?ticker=${ticker}`);
  if (!res.ok) throw new Error("Failed to fetch market data");
  const data = await res.json();
  return data.data;
}

export async function startSimulation(params: any): Promise<SimStatus> {
  const res = await fetch(`${API_BASE}/sim/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Failed to start simulation");
  return res.json();
}

export async function stepSimulation(days: number = 1): Promise<StepResponse> {
  const res = await fetch(`${API_BASE}/sim/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ days }),
  });
  if (!res.ok) throw new Error("Failed to step simulation");
  return res.json();
}

export async function controlSimulation(action: "pause" | "resume" | "reset"): Promise<any> {
  const res = await fetch(`${API_BASE}/sim/control?action=${action}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to control simulation");
  return res.json();
}

export async function stopSimulation(): Promise<{ status: string; current_day: number }> {
  const res = await fetch(`${API_BASE}/sim/stop`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to stop simulation");
  return res.json();
}

export async function fetchSimStatus(): Promise<SimStatus> {
  const res = await fetch(`${API_BASE}/sim/status`);
  if (!res.ok) throw new Error("Failed to fetch status");
  return res.json();
}

export async function fetchAgentBrain(agentId: string): Promise<AgentBrainData> {
  const res = await fetch(`${API_BASE}/agent/brain?agent_id=${encodeURIComponent(agentId)}`);
  if (!res.ok) throw new Error(`Failed to fetch brain for agent ${agentId}`);
  return res.json();
}

export async function fetchAgentLineage(): Promise<LineageData> {
  const res = await fetch(`${API_BASE}/agent/lineage`);
  if (!res.ok) throw new Error("Failed to fetch lineage data");
  return res.json();
}

export async function fetchOverview(): Promise<OverviewMetrics> {
  const res = await fetch(`${API_BASE}/overview`);
  if (!res.ok) throw new Error("Failed to fetch overview metrics");
  return res.json();
}
