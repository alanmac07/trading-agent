"use client";

import React, { useEffect, useState, useMemo } from "react";
import { fetchAgentBrain, AgentBrainData, AgentData } from "@/lib/api";
import { 
  BrainCircuit, 
  Search, 
  Sliders, 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  ShieldAlert, 
  Compass, 
  Filter, 
  Eye, 
  CheckCircle2, 
  AlertCircle,
  BarChart3,
  Gauge
} from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: any[]) {
  return twMerge(clsx(inputs));
}

interface AgentBrainTabProps {
  agents: AgentData[];
  selectedAgentId: string | null;
  onSelectAgent: (id: string) => void;
}

export default function AgentBrainTab({
  agents,
  selectedAgentId,
  onSelectAgent,
}: AgentBrainTabProps) {
  const currentAgentId = selectedAgentId || (agents.length > 0 ? agents[0].id : null);
  const [brainData, setBrainData] = useState<AgentBrainData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [actionFilter, setActionFilter] = useState<string>("ALL");

  useEffect(() => {
    if (currentAgentId) {
      loadBrain(currentAgentId);
      const timer = setInterval(() => loadBrain(currentAgentId), 1500);
      return () => clearInterval(timer);
    }
  }, [currentAgentId]);

  const loadBrain = async (id: string) => {
    try {
      const data = await fetchAgentBrain(id);
      setBrainData(data);
    } catch (e) {
      console.error(e);
    }
  };

  // Filtered decision logs
  const filteredLogs = useMemo(() => {
    if (!brainData?.decision_log) return [];
    return brainData.decision_log
      .slice()
      .reverse()
      .filter((item) => {
        if (actionFilter !== "ALL" && item.action !== actionFilter) return false;
        if (!searchQuery) return true;
        const q = searchQuery.toLowerCase();
        return (
          item.date.toLowerCase().includes(q) ||
          item.action.toLowerCase().includes(q) ||
          item.reason.toLowerCase().includes(q) ||
          item.price.toString().includes(q)
        );
      });
  }, [brainData, searchQuery, actionFilter]);

  if (!currentAgentId) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-12 text-slate-500 font-mono text-center">
        <BrainCircuit className="w-12 h-12 text-slate-600 mb-3 animate-pulse" />
        <div className="text-sm font-bold text-slate-400">No Agent Selected</div>
        <div className="text-xs text-slate-600 mt-1">Start the simulation to spawn NEAT agents for inspection.</div>
      </div>
    );
  }

  const actionSig = brainData?.action_signal ?? 0;
  const riskSig = brainData?.risk_signal ?? 0;

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-background text-foreground">
      {/* Header & Agent Selector */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-4 rounded-2xl bg-panel border border-border/80 shadow-lg">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center border border-white/10 font-bold font-mono text-sm"
            style={{ backgroundColor: brainData?.color ? `${brainData.color}25` : "#10b98125", color: brainData?.color || "#10b981" }}
          >
            <BrainCircuit className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-black text-white font-mono">{brainData?.agent_id || currentAgentId}</h2>
              {brainData?.status === "alive" ? (
                <span className="text-[10px] font-bold font-mono px-2 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-800/60 text-emerald-400">
                  ALIVE
                </span>
              ) : (
                <span className="text-[10px] font-bold font-mono px-2 py-0.5 rounded-full bg-red-950/80 border border-red-800/60 text-red-400">
                  TERMINATED
                </span>
              )}
              <span className="text-xs text-slate-400 font-mono">
                Gen {brainData?.generation ?? 0} {brainData?.parent_id ? `(Parent: ${brainData.parent_id})` : "(Seed Champion)"}
              </span>
            </div>
            <div className="text-xs text-slate-400 mt-0.5">
              Portfolio: <strong className="text-white font-mono">${brainData?.portfolio_value.toFixed(2) ?? "10000.00"}</strong>
              <span className="mx-2">•</span>
              Return:{" "}
              <strong className={cn("font-mono", (brainData?.return_pct ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400")}>
                {(brainData?.return_pct ?? 0) >= 0 ? "+" : ""}{brainData?.return_pct.toFixed(2) ?? "0.00"}%
              </strong>
            </div>
          </div>
        </div>

        {/* Agent Switcher Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 max-w-full">
          {agents.map((ag) => {
            const isSelected = ag.id === currentAgentId;
            return (
              <button
                key={ag.id}
                onClick={() => onSelectAgent(ag.id)}
                className={cn(
                  "px-3 py-1.5 rounded-xl text-xs font-mono font-bold transition flex items-center gap-1.5 cursor-pointer shrink-0",
                  isSelected
                    ? "bg-up text-background shadow-md shadow-up/20"
                    : "bg-background hover:bg-slate-800 text-slate-400 border border-border"
                )}
              >
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: ag.status === "alive" ? ag.color : "#ef4444" }}
                />
                <span>{ag.id}</span>
                <span className={cn("text-[10px]", isSelected ? "text-background/80" : "text-slate-500")}>
                  ({ag.return_pct >= 0 ? "+" : ""}{ag.return_pct.toFixed(1)}%)
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Neural Output Box */}
      <div className="grid grid-cols-1 gap-6">
        {/* Action Signal Gauge */}
        <div className="p-5 rounded-2xl bg-panel border border-border/80 shadow-lg space-y-3">
          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
            <span className="flex items-center gap-1.5 font-bold text-white">
              <Gauge className="w-4 h-4 text-emerald-400" /> Action Signal [Tanh Output]
            </span>
            <span className="font-bold text-white font-mono">{actionSig.toFixed(4)}</span>
          </div>

          {/* Bi-directional signal meter */}
          <div className="space-y-1">
            <div className="relative w-full h-4 bg-slate-900 rounded-full overflow-hidden border border-slate-700">
              <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-slate-500 z-10" />
              {actionSig >= 0 ? (
                <div
                  className="absolute top-0 bottom-0 bg-emerald-500 rounded-r-full transition-all duration-300"
                  style={{ left: "50%", width: `${(actionSig * 50).toFixed(1)}%` }}
                />
              ) : (
                <div
                  className="absolute top-0 bottom-0 bg-rose-500 rounded-l-full transition-all duration-300"
                  style={{ right: "50%", width: `${(Math.abs(actionSig) * 50).toFixed(1)}%` }}
                />
              )}
            </div>
            <div className="flex justify-between text-[10px] font-mono text-slate-500 px-1">
              <span>-1.0 (Strong SELL)</span>
              <span>0.0 (Neutral)</span>
              <span>+1.0 (Strong BUY)</span>
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-background border border-border/40 text-[11px] text-slate-400">
            Threshold: <strong className="text-emerald-400">&gt; +0.20</strong> = BUY Trigger | <strong className="text-rose-400">&lt; -0.20</strong> = SELL Trigger
          </div>
        </div>
      </div>

      {/* 15 Normalized NEAT Input Features Visualizer */}
      <div className="p-5 rounded-2xl bg-panel border border-border/80 shadow-lg space-y-4">
        <div className="flex items-center justify-between border-b border-border/60 pb-3">
          <div className="font-bold text-sm text-white flex items-center gap-2">
            <Sliders className="w-4 h-4 text-cyan-400" /> 15 Normalized NEAT Input Features (Per-Candle Vector)
          </div>
          <span className="text-[10px] font-mono text-slate-400">NORMALIZED [-1.0, +1.0]</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {brainData?.all_inputs &&
            Object.entries(brainData.all_inputs).map(([key, val]) => {
              const num = typeof val === "number" ? val : 0;
              const barPct = Math.min(Math.max(((num + 1) / 2) * 100, 0), 100);

              let color = "bg-cyan-500";
              if (key.includes("rsi") || key.includes("macd")) color = "bg-emerald-500";
              if (key.includes("pnl") || key.includes("return")) color = num >= 0 ? "bg-emerald-500" : "bg-rose-500";
              if (key.includes("bb") || key.includes("atr") || key.includes("vol")) color = "bg-purple-500";

              return (
                <div key={key} className="p-2.5 rounded-xl bg-background border border-border/50 space-y-1.5">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className="text-slate-400 truncate max-w-[120px]" title={key}>
                      {key}
                    </span>
                    <span className="font-bold text-white">{num.toFixed(3)}</span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div className={cn("h-full rounded-full transition-all duration-300", color)} style={{ width: `${barPct}%` }} />
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Searchable Decision Log Table */}
      <div className="p-5 rounded-2xl bg-panel border border-border/80 shadow-lg space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border/60 pb-3">
          <div className="font-bold text-sm text-white flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-emerald-400" /> Chronological Decision Engine Log ({filteredLogs.length})
          </div>

          <div className="flex items-center gap-2">
            {/* Filter */}
            <div className="flex items-center gap-1 bg-background border border-border rounded-lg p-0.5 text-xs font-mono">
              {["ALL", "BUY", "SELL", "HOLD"].map((f) => (
                <button
                  key={f}
                  onClick={() => setActionFilter(f)}
                  className={cn(
                    "px-2.5 py-1 rounded text-[10px] font-bold transition cursor-pointer",
                    actionFilter === f ? "bg-up text-background" : "text-slate-400 hover:text-white"
                  )}
                >
                  {f}
                </button>
              ))}
            </div>

            {/* Search */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search decision logs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="bg-background border border-border rounded-lg pl-8 pr-3 py-1 text-xs text-white placeholder-slate-500 outline-none focus:border-up w-48 font-mono"
              />
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto max-h-96 overflow-y-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="sticky top-0 bg-[#070e1c] text-slate-400 uppercase text-[10px] tracking-wider border-b border-border">
              <tr>
                <th className="py-2.5 px-3">Date</th>
                <th className="py-2.5 px-3">Market Price</th>
                <th className="py-2.5 px-3">Exec Price</th>
                <th className="py-2.5 px-3">Action</th>
                <th className="py-2.5 px-3">Portfolio Value</th>
                <th className="py-2.5 px-3">Neural Decision Rationale</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {filteredLogs.map((log, idx) => (
                <tr key={idx} className="hover:bg-slate-800/40 transition">
                  <td className="py-2 px-3 text-slate-400">{log.date}</td>
                  <td className="py-2 px-3 text-white font-bold">${log.price.toFixed(2)}</td>
                  <td className="py-2 px-3 text-slate-300">${log.exec_price.toFixed(2)}</td>
                  <td className="py-2 px-3">
                    <span
                      className={cn(
                        "px-2 py-0.5 rounded text-[10px] font-black",
                        log.action === "BUY" && "bg-emerald-950 text-emerald-400 border border-emerald-800/60",
                        log.action === "SELL" && "bg-rose-950 text-rose-400 border border-rose-800/60",
                        log.action === "HOLD" && "bg-slate-800 text-slate-400"
                      )}
                    >
                      {log.action}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-white font-bold">${log.portfolio_value.toFixed(2)}</td>
                  <td className="py-2 px-3 text-slate-400 max-w-xs truncate" title={log.reason}>
                    {log.reason}
                  </td>
                </tr>
              ))}
              {filteredLogs.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-500">
                    No decision entries match current filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
