"use client";

import React, { useEffect, useState } from "react";
import { 
  fetchOverview, 
  OverviewMetrics, 
  AgentData, 
  fetchSimStatus 
} from "@/lib/api";
import { 
  TrendingUp, 
  DollarSign, 
  Percent, 
  Layers, 
  Users, 
  ShieldAlert, 
  Award, 
  Activity, 
  PieChart as PieIcon, 
  ArrowUpRight, 
  ArrowDownRight,
  BrainCircuit,
  GitBranch
} from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: any[]) {
  return twMerge(clsx(inputs));
}

interface AgentOverviewTabProps {
  onNavigateTab: (tab: string) => void;
  agents: AgentData[];
}

export default function AgentOverviewTab({ onNavigateTab, agents }: AgentOverviewTabProps) {
  const [overview, setOverview] = useState<OverviewMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    loadOverview();
    const interval = setInterval(loadOverview, 1500);
    return () => clearInterval(interval);
  }, []);

  const loadOverview = async () => {
    try {
      const data = await fetchOverview();
      setOverview(data);
      setLoading(false);
    } catch (e) {
      console.error(e);
      setLoading(false);
    }
  };

  const totalEquity = overview?.total_equity || agents.reduce((s, a) => s + a.portfolio_value, 0) || 10000;
  const initialEquity = 10000;
  const totalPnl = totalEquity - initialEquity;
  const totalPnlPct = ((totalEquity - initialEquity) / initialEquity) * 100;
  const aliveCount = overview?.alive_count ?? agents.filter(a => a.status === 'alive').length;
  const deadCount = overview?.dead_count ?? agents.filter(a => a.status === 'dead').length;
  const totalAgents = overview?.total_agents || agents.length || 1;
  const survivalRate = ((aliveCount / totalAgents) * 100).toFixed(1);

  // Capital breakdown
  const cash = overview?.total_cash || agents.reduce((s, a) => s + a.cash, 0);
  const marketExposure = overview?.total_market_exposure || (totalEquity - cash);
  const graveyard = overview?.graveyard_pool || 0;

  const totalPool = Math.max(cash + marketExposure + graveyard, 1);
  const cashPct = ((cash / totalPool) * 100).toFixed(1);
  const marketPct = ((marketExposure / totalPool) * 100).toFixed(1);
  const graveyardPct = ((graveyard / totalPool) * 100).toFixed(1);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-background text-foreground">
      {/* Prominent P&L Banner */}
      <div className="flex flex-col md:flex-row justify-between items-center bg-gradient-to-br from-[#0e1d3e] to-[#0a152e] border border-border/80 rounded-2xl p-6 shadow-2xl gap-6">
        <div className="space-y-2 text-center md:text-left">
          <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs font-bold tracking-widest uppercase justify-center md:justify-start">
            <TrendingUp className="w-4 h-4 animate-pulse" /> TOTAL SYSTEM NET PROFIT & LOSS
          </div>
          <div className="text-4xl md:text-5xl font-black text-white font-mono tracking-tight flex items-center justify-center md:justify-start gap-2">
            {totalPnl >= 0 ? "+" : "-"}${Math.abs(totalPnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            <span className={cn("text-xl md:text-2xl ml-2 flex items-center", totalPnlPct >= 0 ? "text-emerald-400" : "text-rose-400")}>
              {totalPnlPct >= 0 ? <ArrowUpRight className="w-6 h-6" /> : <ArrowDownRight className="w-6 h-6" />}
              {Math.abs(totalPnlPct).toFixed(2)}%
            </span>
          </div>
          <div className="text-slate-400 text-xs font-mono">
            Starting Capital: $10,000.00 | Peak Return Target: +8.00%
          </div>
        </div>
        
        {/* Action Shortcuts */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => onNavigateTab("brain")}
            className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-up/10 hover:bg-up/20 border border-up/30 text-up font-bold text-sm transition cursor-pointer shadow-lg shadow-up/5"
          >
            <BrainCircuit className="w-4 h-4" /> Inspect Brain
          </button>
          <button
            onClick={() => onNavigateTab("lineage")}
            className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 text-purple-400 font-bold text-sm transition cursor-pointer shadow-lg shadow-purple-500/5"
          >
            <GitBranch className="w-4 h-4" /> Lineage Tree
          </button>
        </div>
      </div>

      {/* Target Bar / Progress Meter */}
      <div className="bg-panel border border-border/80 rounded-xl p-5 shadow-lg space-y-3">
        <div className="flex justify-between items-end">
          <div className="text-xs font-mono text-slate-400 font-bold uppercase">Portfolio Growth Target</div>
          <div className="text-xs font-mono font-bold text-emerald-400">+8.00% Goal ($10,800.00)</div>
        </div>
        <div className="relative w-full h-4 bg-[#040813] rounded-full overflow-hidden border border-border/40">
           {/* Limit lines */}
           <div className="absolute left-[20%] top-0 bottom-0 w-px bg-red-500/50 z-10" title="Max Drawdown Buffer Limit" />
           <div className="absolute left-[50%] top-0 bottom-0 w-px bg-slate-500/50 z-10" title="Baseline" />
           <div className="absolute left-[80%] top-0 bottom-0 w-px bg-emerald-500/50 z-10" title="Profit Target" />
           
           <div 
             className={cn("h-full transition-all duration-700", totalPnlPct >= 0 ? "bg-emerald-500" : "bg-rose-500")}
             style={{ width: `${Math.min(100, Math.max(0, 50 + (totalPnlPct / 16) * 50))}%` }} 
           />
        </div>
        <div className="flex justify-between text-[10px] font-mono text-slate-500">
          <span>-$1,600 (Drawdown Limit)</span>
          <span>$10,000 Base</span>
          <span>+$1,600 (Target)</span>
        </div>
      </div>

      {/* P&L Breakdown Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Unrealized P&L */}
        <div className="p-4 rounded-xl bg-panel border border-border/80 shadow-lg">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs font-mono mb-2">
            <Activity className="w-3.5 h-3.5 text-cyan-400" /> UNREALIZED P&L
          </div>
          <div className="text-xl font-black font-mono text-cyan-400">
            ${marketExposure.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>

        {/* Realized Cash */}
        <div className="p-4 rounded-xl bg-panel border border-border/80 shadow-lg">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs font-mono mb-2">
            <DollarSign className="w-3.5 h-3.5 text-emerald-400" /> REALIZED CASH
          </div>
          <div className="text-xl font-black font-mono text-emerald-400">
            ${cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>

        {/* Graveyard Recovered */}
        <div className="p-4 rounded-xl bg-panel border border-border/80 shadow-lg">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs font-mono mb-2">
            <Layers className="w-3.5 h-3.5 text-purple-400" /> GRAVEYARD FUNDS
          </div>
          <div className="text-xl font-black font-mono text-purple-400">
            ${graveyard.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>

        {/* Win/Loss trades */}
        <div className="p-4 rounded-xl bg-panel border border-border/80 shadow-lg">
          <div className="flex items-center gap-1.5 text-slate-400 text-xs font-mono mb-2">
            <Percent className="w-3.5 h-3.5 text-blue-400" /> WINNING TRADES
          </div>
          <div className="text-xl font-black font-mono text-blue-400">
            {overview?.win_rate_pct ?? "0.0"}%
          </div>
          <div className="text-[10px] text-slate-500 font-mono mt-1">
            {overview?.total_trades ?? 0} Total Executions
          </div>
        </div>
      </div>

      {/* Capital Distribution & Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Capital Breakdown Card */}
        <div className="p-5 rounded-2xl bg-panel border border-border/80 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-border/60 pb-3">
            <div className="font-bold text-sm text-white flex items-center gap-2">
              <PieIcon className="w-4 h-4 text-emerald-400" /> Capital Allocation
            </div>
            <span className="text-[10px] font-mono text-slate-400">DYNAMIC</span>
          </div>

          {/* Allocation Bar */}
          <div className="w-full h-4 bg-slate-800 rounded-full overflow-hidden flex">
            <div style={{ width: `${cashPct}%` }} className="bg-emerald-500 transition-all duration-500" title={`Cash: ${cashPct}%`} />
            <div style={{ width: `${marketPct}%` }} className="bg-cyan-500 transition-all duration-500" title={`In Market: ${marketPct}%`} />
            <div style={{ width: `${graveyardPct}%` }} className="bg-purple-500 transition-all duration-500" title={`Graveyard: ${graveyardPct}%`} />
          </div>

          <div className="space-y-2.5 text-xs font-mono">
            <div className="flex justify-between items-center p-2 rounded-lg bg-background border border-border/40">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                <span className="text-slate-300">Liquid Cash</span>
              </div>
              <div className="font-bold text-white">
                ${cash.toFixed(2)} <span className="text-slate-500 text-[10px]">({cashPct}%)</span>
              </div>
            </div>

            <div className="flex justify-between items-center p-2 rounded-lg bg-background border border-border/40">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-cyan-500" />
                <span className="text-slate-300">Market Exposure</span>
              </div>
              <div className="font-bold text-white">
                ${marketExposure.toFixed(2)} <span className="text-slate-500 text-[10px]">({marketPct}%)</span>
              </div>
            </div>

            <div className="flex justify-between items-center p-2 rounded-lg bg-background border border-border/40">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-purple-500" />
                <span className="text-slate-300">Graveyard Bounty Pool</span>
              </div>
              <div className="font-bold text-white">
                ${graveyard.toFixed(2)} <span className="text-slate-500 text-[10px]">({graveyardPct}%)</span>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
