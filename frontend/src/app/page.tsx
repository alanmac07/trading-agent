"use client";

import React, { useEffect, useState, useRef } from "react";
import { 
  fetchTickers, 
  fetchMarketData, 
  startSimulation, 
  stepSimulation, 
  controlSimulation, 
  stopSimulation,
  fetchSimStatus, 
  SimStatus, 
  MarketData, 
  AgentData, 
  TradeData 
} from "@/lib/api";
import { 
  Activity, 
  Play, 
  Pause, 
  Square,
  RotateCcw, 
  Settings, 
  Users, 
  DollarSign, 
  TrendingUp, 
  Filter, 
  Sliders, 
  X, 
  Zap,
  Layers,
  BrainCircuit,
  ShieldAlert,
  BarChart3,
  GitBranch,
  LineChart,
  Trophy,
  PieChart
} from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import TradingChart from "@/components/TradingChart";
import EquityCurveChart from "@/components/EquityCurveChart";
import TimeSpeedControls from "@/components/TimeSpeedControls";
import AgentOverviewTab from "@/components/AgentOverviewTab";
import AgentBrainTab from "@/components/AgentBrainTab";
import AgentLeaderboardTab from "@/components/AgentLeaderboardTab";
import AgentLineageTree from "@/components/AgentLineageTree";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function TradingTerminal() {
  // Navigation tabs
  const [mainTab, setMainTab] = useState<"terminal" | "analytics">("terminal");
  const [analyticsSubTab, setAnalyticsSubTab] = useState<"overview" | "brain" | "leaderboard" | "lineage">("overview");

  // Market & Simulation State
  const [tickers, setTickers] = useState<string[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string>("AAPL");
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [latestCandle, setLatestCandle] = useState<MarketData | null>(null);
  const [status, setStatus] = useState<SimStatus>({ running: false, stopped: false });
  const [isPaused, setIsPaused] = useState(false);
  const [agents, setAgents] = useState<AgentData[]>([]);
  const [trades, setTrades] = useState<TradeData[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  
  // Agent filter / inspection
  const [focusedAgentId, setFocusedAgentId] = useState<string | null>(null);
  const [inspectedAgentId, setInspectedAgentId] = useState<string | null>(null);

  // Settings & Controls
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [startCap, setStartCap] = useState(10000);
  const [profitPct, setProfitPct] = useState(8);
  const [lossPct, setLossPct] = useState(10);
  const [feePct, setFeePct] = useState(0.001);
  const [maxAgents, setMaxAgents] = useState(15);
  const [daysPerFrame, setDaysPerFrame] = useState(1);
  const [playbackSpeed, setPlaybackSpeed] = useState(150); // ms per step
  
  const isPollingRef = useRef<boolean>(false);
  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);
  const playbackSpeedRef = useRef<number>(playbackSpeed);
  const daysPerFrameRef = useRef<number>(daysPerFrame);

  useEffect(() => {
    playbackSpeedRef.current = playbackSpeed;
    daysPerFrameRef.current = daysPerFrame;
  }, [playbackSpeed, daysPerFrame]);

  useEffect(() => {
    fetchTickers().then((ts) => {
      setTickers(ts);
      if (ts.length > 0 && !ts.includes(selectedTicker)) {
        setSelectedTicker(ts[0]);
      }
    });
    checkStatus();
  }, []);

  useEffect(() => {
    if (selectedTicker && !status.running) {
      fetchMarketData(selectedTicker).then(data => {
        setMarketData(data);
        setLatestCandle(null);
      });
    }
  }, [selectedTicker, status.running]);

  const checkStatus = async () => {
    try {
      const s = await fetchSimStatus();
      setStatus(s);
      if (s.agents) setAgents(s.agents);
    } catch (e) {
      console.error(e);
    }
  };

  const handleStart = async () => {
    try {
      const res = await startSimulation({
        ticker: selectedTicker,
        starting_capital: startCap,
        profit_goal_pct: profitPct,
        loss_limit_pct: lossPct,
        fee_pct: feePct,
        max_agents: maxAgents,
        checkpoint_days: 20,
        clone_frac: 0.5
      });
      setStatus({ running: true, stopped: false });
      setIsPaused(false);
      if (res.agents) setAgents(res.agents);
      startPolling();
    } catch (e) {
      console.error("Failed to start swarm:", e);
    }
  };

  const handleStop = async () => {
    stopPolling();
    try {
      await stopSimulation();
    } catch (e) {
      console.error(e);
    }
    setStatus(prev => ({ ...prev, running: false, stopped: true }));
    setIsPaused(false);
    // Auto-redirect to analytics
    setMainTab("analytics");
  };

  const handleReset = async () => {
    stopPolling();
    await controlSimulation("reset");
    setStatus({ running: false, stopped: false });
    setIsPaused(false);
    setAgents([]);
    setTrades([]);
    setEvents([]);
    setLatestCandle(null);
    setFocusedAgentId(null);
    setInspectedAgentId(null);
    if (selectedTicker) {
      const data = await fetchMarketData(selectedTicker);
      setMarketData(data);
    }
  };

  const handlePauseResume = async () => {
    if (!isPaused) {
      stopPolling();
      setIsPaused(true);
      await controlSimulation("pause");
    } else {
      setIsPaused(false);
      startPolling();
      await controlSimulation("resume");
    }
  };

  const startPolling = () => {
    if (isPollingRef.current) return;
    isPollingRef.current = true;

    const runStep = async () => {
      if (!isPollingRef.current) return;
      try {
        const res = await stepSimulation(daysPerFrameRef.current);
        if (res.days_advanced > 0) {
          if (res.candles && res.candles.length > 0) {
            const last = res.candles[res.candles.length - 1];
            setLatestCandle(last);
          }
          setAgents(res.agents);
          if (res.trades && res.trades.length > 0) {
            setTrades(prev => [...prev, ...res.trades].slice(-500));
          }
          if (res.events && res.events.length > 0) {
            setEvents(prev => [...prev, ...res.events].slice(-50));
          }
          setStatus(prev => ({ 
            ...prev, 
            running: !res.stopped,
            stopped: res.stopped || false,
            current_day: res.current_day, 
            graveyard_pool: res.graveyard_pool 
          }));

          if (res.stopped) {
            stopPolling();
            return;
          }
        }
      } catch (e) {
        console.error("Simulation tick error:", e);
      }

      if (isPollingRef.current) {
        pollTimerRef.current = setTimeout(runStep, playbackSpeedRef.current);
      }
    };

    runStep();
  };

  const stopPolling = () => {
    isPollingRef.current = false;
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const handleInspectAgent = (id: string) => {
    setInspectedAgentId(id);
    setMainTab("analytics");
    setAnalyticsSubTab("brain");
  };

  // Metrics
  const totalEquity = agents.reduce((sum, a) => sum + a.portfolio_value, 0);
  const graveyard = status.graveyard_pool || 0;
  const aliveCount = agents.filter(a => a.status === 'alive').length;

  return (
    <div className="flex flex-col h-screen bg-background text-foreground text-sm font-sans select-none overflow-hidden">
      
      {/* ── TOP TICKER TAPE ── */}
      <div className="h-7 bg-[#040812] border-b border-border flex items-center px-4 overflow-x-auto text-[11px] font-mono gap-6 shrink-0 z-10">
        <span className="text-slate-500 font-bold flex items-center gap-1">
          <Zap className="w-3 h-3 text-yellow-400" /> ASSETS (10Y):
        </span>
        {tickers.map((t) => (
          <div 
            key={t} 
            className={cn(
              "flex items-center gap-1.5 cursor-pointer transition",
              selectedTicker === t ? "text-up font-bold underline" : "text-slate-400 hover:text-white"
            )} 
            onClick={() => !status.running && setSelectedTicker(t)}
          >
            <span>{t}</span>
            <span className="text-[9px] text-emerald-400 bg-emerald-950/60 px-1 py-0.2 rounded border border-emerald-800/40">10Y ACTIVE</span>
          </div>
        ))}
      </div>

      {/* ── GLOBAL HEADER WITH TABS & CONTROLS ── */}
      <header className="h-14 border-b border-border bg-panel flex items-center justify-between px-4 shrink-0 z-20">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 font-black text-lg tracking-tight text-white">
            <BrainCircuit className="w-5 h-5 text-up animate-pulse" />
            QUANT NEAT SWARM
          </div>
          <div className="h-5 w-px bg-border mx-1" />
          
          {/* Main Tab Navigation */}
          <div className="flex items-center bg-background border border-border rounded-xl p-0.5 text-xs font-mono">
            <button
              onClick={() => setMainTab("terminal")}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition cursor-pointer",
                mainTab === "terminal" ? "bg-up text-background shadow-md shadow-up/20" : "text-slate-400 hover:text-white"
              )}
            >
              <LineChart className="w-3.5 h-3.5" /> Live Terminal
            </button>
            <button
              onClick={() => setMainTab("analytics")}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition cursor-pointer",
                mainTab === "analytics" ? "bg-purple-500 text-white shadow-md shadow-purple-500/20" : "text-slate-400 hover:text-white"
              )}
            >
              <BrainCircuit className="w-3.5 h-3.5" /> Intelligence Hub
            </button>
          </div>

          <div className="h-5 w-px bg-border mx-1" />

          {/* Ticker Selector */}
          <select 
            value={selectedTicker} 
            onChange={(e) => setSelectedTicker(e.target.value)}
            disabled={status.running}
            className="bg-background border border-border rounded px-3 py-1 text-sm font-bold text-slate-200 outline-none focus:border-up disabled:opacity-50 cursor-pointer font-mono"
          >
            {tickers.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          
          {status.running && !isPaused ? (
            <div className="flex items-center gap-2 px-3 py-1 bg-up/10 border border-up/20 text-up rounded-full text-xs font-bold tracking-wider">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-up opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-up"></span>
              </span>
              SWARM RUNNING
            </div>
          ) : isPaused ? (
            <div className="px-3 py-1 bg-yellow-950/60 border border-yellow-800/50 text-yellow-400 rounded-full text-xs font-bold">
              ⏸ PAUSED
            </div>
          ) : status.stopped ? (
            <div className="px-3 py-1 bg-red-950/60 border border-red-800/50 text-red-400 rounded-full text-xs font-bold">
              ⏹ STOPPED
            </div>
          ) : (
            <div className="px-3 py-1 bg-border/40 text-slate-400 rounded-full text-xs font-semibold">
              READY
            </div>
          )}
        </div>
        
        {/* Playback Controls, Warp Speed & Settings */}
        <div className="flex items-center gap-2.5">
          {/* Time Warp Speed Controller */}
          <TimeSpeedControls
            playbackSpeed={playbackSpeed}
            setPlaybackSpeed={setPlaybackSpeed}
            daysPerStep={daysPerFrame}
            setDaysPerStep={setDaysPerFrame}
            disabled={!status.running && !isPaused}
          />

          <div className="h-5 w-px bg-border" />

          {!status.running ? (
            <div className="flex items-center gap-2">
              <button 
                onClick={handleStart} 
                className="flex items-center gap-1.5 bg-up hover:bg-up/90 text-background font-black px-4 py-1.5 rounded-lg shadow-lg shadow-up/20 transition cursor-pointer"
              >
                <Play className="w-4 h-4 fill-current" /> Start Swarm
              </button>
              {(agents.length > 0 || status.stopped) && (
                <button 
                  onClick={handleReset} 
                  className="flex items-center gap-1.5 bg-panel border border-border hover:bg-slate-700 text-slate-300 font-semibold px-2.5 py-1.5 rounded-lg transition cursor-pointer"
                  title="Clear simulation data"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ) : (
            <>
              <button 
                onClick={handlePauseResume} 
                className="flex items-center gap-1.5 bg-border hover:bg-slate-700 text-white font-semibold px-3 py-1.5 rounded-lg transition cursor-pointer"
              >
                {!isPaused ? <Pause className="w-4 h-4 fill-current text-yellow-400" /> : <Play className="w-4 h-4 fill-current text-up" />}
                {!isPaused ? "Pause" : "Resume"}
              </button>
              
              <button 
                onClick={handleStop} 
                className="flex items-center gap-1.5 bg-red-500/20 border border-red-500/40 hover:bg-red-600 hover:text-white text-red-400 font-semibold px-3 py-1.5 rounded-lg transition cursor-pointer"
                title="Halt simulation and view analytics"
              >
                <Square className="w-3.5 h-3.5 fill-current" /> Stop
              </button>
            </>
          )}

          <button 
            onClick={() => setIsSettingsOpen(!isSettingsOpen)} 
            className={cn(
              "p-2 rounded-lg border transition cursor-pointer", 
              isSettingsOpen ? "bg-up text-background border-up" : "bg-panel border-border text-slate-300 hover:text-white"
            )}
            title="Swarm Settings"
          >
            <Sliders className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* ── METRICS ROW ── */}
      <div className="h-14 border-b border-border bg-[#070e1c] flex items-center px-4 gap-8 shrink-0 overflow-x-auto z-10">
        <MetricCard label="Active Swarm Equity" value={`$${totalEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} icon={<DollarSign className="w-4 h-4 text-up" />} />
        <MetricCard label="Graveyard Pool" value={`$${graveyard.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} icon={<Layers className="w-4 h-4 text-purple-400" />} />
        <MetricCard label="Alive Agents" value={`${aliveCount} / ${agents.length}`} icon={<Users className="w-4 h-4 text-blue-400" />} />
        <MetricCard label="Simulated Day" value={status.current_day || 0} icon={<TrendingUp className="w-4 h-4 text-emerald-400" />} />
        
        {focusedAgentId && (
          <div className="flex items-center gap-2 bg-up/10 border border-up/30 px-3 py-1 rounded text-xs">
            <Filter className="w-3.5 h-3.5 text-up" />
            <span>Focused Chart Filter: <strong className="text-up">{focusedAgentId}</strong></span>
            <X className="w-3.5 h-3.5 cursor-pointer text-slate-400 hover:text-white" onClick={() => setFocusedAgentId(null)} />
          </div>
        )}
      </div>

      {/* ── MAIN WORKSPACE / TAB CONTENT ── */}
      <div className="flex-1 flex overflow-hidden relative">
        
        {/* TAB 1: LIVE TRADING TERMINAL */}
        {mainTab === "terminal" && (
          <>
            {/* CHART SPLIT: MAIN CANDLESTICKS + LIVE AGENT EQUITY CURVES */}
            <main className="flex-1 bg-background relative border-r border-border flex flex-col overflow-hidden">
              {/* Top: Price Candlesticks */}
              <div className="flex-1 relative">
                {marketData.length > 0 ? (
                  <TradingChart 
                    initialData={marketData}
                    latestCandle={latestCandle}
                    trades={trades} 
                    focusedAgentId={focusedAgentId}
                  />
                ) : (
                  <div className="flex-1 h-full flex items-center justify-center text-slate-500 font-mono">
                    Loading 10-year market price data...
                  </div>
                )}
              </div>

              {/* Bottom: Synchronized Swarm Equity Curve Chart */}
              <EquityCurveChart
                agents={agents}
                graveyardPool={graveyard}
                focusedAgentId={focusedAgentId}
                onFocusAgent={setFocusedAgentId}
                height={180}
              />
            </main>
            
            {/* RIGHT SIDEBAR: NEAT AGENT MONITOR & LIVE EXECUTION FEED */}
            <aside className="w-88 bg-panel flex flex-col overflow-hidden shrink-0 border-l border-border">
              {/* Header */}
              <div className="p-3 border-b border-border font-bold text-xs uppercase tracking-wider flex justify-between items-center bg-background">
                <span className="text-slate-300 flex items-center gap-1.5">
                  <BrainCircuit className="w-3.5 h-3.5 text-up" /> NEAT Swarm ({agents.length})
                </span>
                {focusedAgentId && (
                  <button onClick={() => setFocusedAgentId(null)} className="text-[10px] text-slate-400 hover:text-white underline">
                    Clear filter
                  </button>
                )}
              </div>
              
              {/* Agent List */}
              <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
                {agents.sort((a,b) => b.return_pct - a.return_pct).map(a => {
                  const isFocused = focusedAgentId === a.id;
                  return (
                    <div 
                      key={a.id} 
                      className={cn(
                        "p-2.5 rounded-xl border transition cursor-pointer",
                        isFocused ? "border-up bg-up/5 shadow-md shadow-up/10" : "border-border bg-background hover:border-slate-600"
                      )}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <div className="flex items-center gap-2">
                          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: a.status === 'alive' ? a.color : '#ef4444' }} />
                          <span className="font-mono font-bold text-xs" style={{ color: a.color }}>{a.id}</span>
                          {a.status !== 'alive' && <span className="text-[9px] bg-red-950 text-red-400 px-1 rounded font-mono">DEAD</span>}
                          {a.position > 0 && <span className="text-[9px] bg-emerald-950 text-emerald-400 px-1 rounded font-mono font-bold">LONG</span>}
                        </div>
                        <div className={cn("text-xs font-bold font-mono", a.return_pct >= 0 ? "text-up" : "text-down")}>
                          {a.return_pct >= 0 ? "+" : ""}{a.return_pct.toFixed(2)}%
                        </div>
                      </div>
                      <div className="flex justify-between text-[11px] text-slate-400 font-mono">
                        <span>Gen {a.generation} {a.parent_id ? `(from ${a.parent_id})` : '(Seed)'}</span>
                        <span>${a.portfolio_value.toFixed(2)}</span>
                      </div>

                      {/* Action buttons */}
                      <div className="flex items-center justify-between pt-1.5 mt-1 border-t border-border/40 text-[10px] font-mono">
                        <button
                          onClick={() => setFocusedAgentId(isFocused ? null : a.id)}
                          className={cn("px-2 py-0.5 rounded transition cursor-pointer", isFocused ? "bg-up text-background font-bold" : "text-slate-400 hover:text-white bg-panel")}
                        >
                          {isFocused ? "Unfocus" : "Focus on Chart"}
                        </button>
                        <button
                          onClick={() => handleInspectAgent(a.id)}
                          className="text-cyan-400 hover:underline flex items-center gap-1 cursor-pointer"
                        >
                          Inspect Brain →
                        </button>
                      </div>
                    </div>
                  );
                })}
                {agents.length === 0 && (
                  <div className="text-center text-slate-500 mt-12 text-xs">
                    No active agents.<br/>Click <strong>Start Swarm</strong> to spawn NEAT agents.
                  </div>
                )}
              </div>
              
              {/* LIVE EXECUTION FEED */}
              <div className="h-60 border-t border-border flex flex-col bg-background">
                <div className="p-2 border-b border-border text-[11px] font-bold text-slate-400 uppercase tracking-wider flex justify-between items-center">
                  <span>Order Execution Feed</span>
                  <span className="text-[9px] text-slate-500 font-mono">STREAMING</span>
                </div>
                <div className="flex-1 overflow-y-auto p-2 space-y-1">
                  {trades.slice().reverse().map((t, i) => (
                    <div key={i} className="text-[11px] font-mono border-b border-border/40 pb-1">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5">
                          <span className={cn(t.action === 'BUY' ? "text-up bg-up/10" : "text-down bg-down/10", "font-bold px-1 rounded text-[10px]")}>
                            {t.action}
                          </span>
                          <span style={{ color: t.color }}>{t.agent_id}</span>
                        </div>
                        <div className="text-white font-bold">${Number(t.price).toFixed(2)}</div>
                      </div>
                      <div className="flex justify-between text-[9px] text-slate-500 mt-0.5">
                        <span className="truncate max-w-[180px] text-slate-400">{t.reason}</span>
                        <span>{t.date.split(' ')[0]}</span>
                      </div>
                    </div>
                  ))}
                  {trades.length === 0 && (
                    <div className="text-center text-slate-600 mt-10 text-xs">Listening for NEAT orders...</div>
                  )}
                </div>
              </div>
            </aside>
          </>
        )}

        {/* TAB 2: AGENT INTELLIGENCE & ANALYTICS HUB */}
        {mainTab === "analytics" && (
          <div className="flex-1 flex flex-col bg-background overflow-hidden">
            {/* Sub-tab Navigation */}
            <div className="h-12 border-b border-border bg-panel px-6 flex items-center gap-3 shrink-0">
              <button
                onClick={() => setAnalyticsSubTab("overview")}
                className={cn(
                  "flex items-center gap-2 px-3.5 py-1.5 rounded-xl font-mono text-xs font-bold transition cursor-pointer",
                  analyticsSubTab === "overview"
                    ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                    : "text-slate-400 hover:text-white"
                )}
              >
                <PieChart className="w-3.5 h-3.5" /> Overview & P&L
              </button>

              <button
                onClick={() => setAnalyticsSubTab("brain")}
                className={cn(
                  "flex items-center gap-2 px-3.5 py-1.5 rounded-xl font-mono text-xs font-bold transition cursor-pointer",
                  analyticsSubTab === "brain"
                    ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/40"
                    : "text-slate-400 hover:text-white"
                )}
              >
                <BrainCircuit className="w-3.5 h-3.5" /> Brain & Decision Engine
              </button>

              <button
                onClick={() => setAnalyticsSubTab("leaderboard")}
                className={cn(
                  "flex items-center gap-2 px-3.5 py-1.5 rounded-xl font-mono text-xs font-bold transition cursor-pointer",
                  analyticsSubTab === "leaderboard"
                    ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40"
                    : "text-slate-400 hover:text-white"
                )}
              >
                <Trophy className="w-3.5 h-3.5" /> Leaderboard
              </button>

              <button
                onClick={() => setAnalyticsSubTab("lineage")}
                className={cn(
                  "flex items-center gap-2 px-3.5 py-1.5 rounded-xl font-mono text-xs font-bold transition cursor-pointer",
                  analyticsSubTab === "lineage"
                    ? "bg-purple-500/20 text-purple-400 border border-purple-500/40"
                    : "text-slate-400 hover:text-white"
                )}
              >
                <GitBranch className="w-3.5 h-3.5" /> Lineage & Splitting
              </button>
            </div>

            {/* Sub-tab view render */}
            <div className="flex-1 flex overflow-hidden">
              {analyticsSubTab === "overview" && (
                <AgentOverviewTab
                  onNavigateTab={(tab) => setAnalyticsSubTab(tab as any)}
                  agents={agents}
                />
              )}

              {analyticsSubTab === "brain" && (
                <AgentBrainTab
                  agents={agents}
                  selectedAgentId={inspectedAgentId}
                  onSelectAgent={(id) => setInspectedAgentId(id)}
                />
              )}

              {analyticsSubTab === "leaderboard" && (
                <AgentLeaderboardTab
                  agents={agents}
                  onSelectAgent={(id) => handleInspectAgent(id)}
                  onFocusAgent={(id) => setFocusedAgentId(id)}
                  focusedAgentId={focusedAgentId}
                />
              )}

              {analyticsSubTab === "lineage" && (
                <AgentLineageTree
                  onSelectAgent={(id) => handleInspectAgent(id)}
                />
              )}
            </div>
          </div>
        )}

        {/* ── SETTINGS SLIDEOUT DRAWER ── */}
        {isSettingsOpen && (
          <div className="absolute top-0 right-0 bottom-0 w-84 bg-panel/95 backdrop-blur-md border-l border-border p-5 shadow-2xl z-40 overflow-y-auto flex flex-col">
            <div className="flex justify-between items-center border-b border-border pb-3 mb-4">
              <span className="font-bold text-sm text-white flex items-center gap-2">
                <Sliders className="w-4 h-4 text-up" /> Swarm Hyperparameters
              </span>
              <X className="w-4 h-4 cursor-pointer text-slate-400 hover:text-white" onClick={() => setIsSettingsOpen(false)} />
            </div>

            <div className="space-y-4 text-xs">
              <div>
                <label className="text-slate-400 block mb-1">Starting Capital ($)</label>
                <input 
                  type="number" 
                  value={startCap} 
                  onChange={(e) => setStartCap(Number(e.target.value))} 
                  disabled={status.running}
                  className="w-full bg-background border border-border rounded-lg px-2.5 py-1.5 text-white disabled:opacity-50"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Clone Threshold: +{profitPct}% P&L</label>
                <input 
                  type="range" min="2" max="30" 
                  value={profitPct} 
                  onChange={(e) => setProfitPct(Number(e.target.value))} 
                  className="w-full accent-up"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Loss Termination: -{lossPct}% P&L</label>
                <input 
                  type="range" min="2" max="30" 
                  value={lossPct} 
                  onChange={(e) => setLossPct(Number(e.target.value))} 
                  className="w-full accent-down"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Slippage Friction: {(feePct * 100).toFixed(2)}%</label>
                <input 
                  type="range" min="0" max="1" step="0.05"
                  value={feePct * 100} 
                  onChange={(e) => setFeePct(Number(e.target.value) / 100)} 
                  className="w-full accent-up"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Max Swarm Population: {maxAgents}</label>
                <input 
                  type="range" min="2" max="30" 
                  value={maxAgents} 
                  onChange={(e) => setMaxAgents(Number(e.target.value))} 
                  className="w-full accent-blue-500"
                />
              </div>

              </div>
            </div>
        )}

      </div>
    </div>
  );
}

function MetricCard({ label, value, icon }: { label: string, value: string | number, icon: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 shrink-0">
      <div className="p-2 bg-panel rounded-lg border border-border/80">
        {icon}
      </div>
      <div>
        <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">{label}</div>
        <div className="text-base font-bold text-white font-mono leading-tight">{value}</div>
      </div>
    </div>
  );
}

