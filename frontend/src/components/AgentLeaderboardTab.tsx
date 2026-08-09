"use client";

import React, { useState, useMemo } from "react";
import { AgentData } from "@/lib/api";
import { 
  Trophy, 
  ArrowUpDown, 
  Search, 
  TrendingUp, 
  TrendingDown, 
  BrainCircuit, 
  ShieldAlert,
  Zap,
  Filter
} from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: any[]) {
  return twMerge(clsx(inputs));
}

interface AgentLeaderboardTabProps {
  agents: AgentData[];
  onSelectAgent: (id: string) => void;
  onFocusAgent: (id: string | null) => void;
  focusedAgentId?: string | null;
}

export default function AgentLeaderboardTab({
  agents,
  onSelectAgent,
  onFocusAgent,
  focusedAgentId,
}: AgentLeaderboardTabProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"ALL" | "alive" | "dead">("ALL");
  const [sortField, setSortField] = useState<"return_pct" | "portfolio_value" | "generation" | "win_rate_pct">("return_pct");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  const handleSort = (field: typeof sortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
  };

  const sortedAgents = useMemo(() => {
    return agents
      .filter((a) => {
        if (statusFilter !== "ALL" && a.status !== statusFilter) return false;
        if (!searchQuery) return true;
        const q = searchQuery.toLowerCase();
        return (
          a.id.toLowerCase().includes(q) ||
          (a.parent_id && a.parent_id.toLowerCase().includes(q)) ||
          a.generation.toString().includes(q)
        );
      })
      .sort((a, b) => {
        const valA = (a as any)[sortField] ?? 0;
        const valB = (b as any)[sortField] ?? 0;
        return sortOrder === "asc" ? valA - valB : valB - valA;
      });
  }, [agents, searchQuery, statusFilter, sortField, sortOrder]);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-background text-foreground">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-2xl bg-panel border border-border/80 shadow-lg">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-yellow-500/10 border border-yellow-500/30 text-yellow-400">
            <Trophy className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-black text-white flex items-center gap-2">
              NEAT Swarm Leaderboard
            </h2>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Rankings & Micro-Equity Telemetry across {agents.length} active and culled genomes
            </p>
          </div>
        </div>

        {/* Filter and Search Bar */}
        <div className="flex items-center gap-3">
          {/* Status Filter */}
          <div className="flex items-center bg-background border border-border rounded-xl p-0.5 text-xs font-mono">
            {(["ALL", "alive", "dead"] as const).map((s) => (
              <button
                key={s}
                onClick={() => setStatusFilter(s)}
                className={cn(
                  "px-3 py-1 rounded-lg text-xs font-bold transition cursor-pointer uppercase",
                  statusFilter === s ? "bg-up text-background" : "text-slate-400 hover:text-white"
                )}
              >
                {s}
              </button>
            ))}
          </div>

          {/* Search Input */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search agent ID / parent..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-background border border-border rounded-xl pl-8 pr-3 py-1.5 text-xs text-white placeholder-slate-500 outline-none focus:border-up w-56 font-mono"
            />
          </div>
        </div>
      </div>

      {/* Leaderboard Table Card */}
      <div className="p-5 rounded-2xl bg-panel border border-border/80 shadow-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#070e1c] text-slate-400 uppercase text-[10px] tracking-wider border-b border-border">
              <tr>
                <th className="py-3 px-3">Rank</th>
                <th className="py-3 px-3">Agent</th>
                <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort("return_pct")}>
                  <div className="flex items-center gap-1">
                    Return % <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort("portfolio_value")}>
                  <div className="flex items-center gap-1">
                    Equity Value <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-3">Peak Equity</th>
                <th className="py-3 px-3">Micro Equity Sparkline</th>
                <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort("generation")}>
                  <div className="flex items-center gap-1">
                    Gen <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort("win_rate_pct")}>
                  <div className="flex items-center gap-1">
                    Win Rate <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-3">Trades</th>
                <th className="py-3 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {sortedAgents.map((ag, index) => {
                const isFocused = focusedAgentId === ag.id;
                const history = ag.portfolio_history || [];
                
                // SVG Sparkline generator
                let sparklinePoints = "";
                if (history.length > 1) {
                  const min = Math.min(...history);
                  const max = Math.max(...history);
                  const range = max - min || 1;
                  sparklinePoints = history
                    .map((v, i) => {
                      const x = (i / (history.length - 1)) * 90 + 5;
                      const y = 25 - ((v - min) / range) * 20 + 2;
                      return `${x.toFixed(1)},${y.toFixed(1)}`;
                    })
                    .join(" ");
                }

                return (
                  <tr
                    key={ag.id}
                    className={cn(
                      "hover:bg-slate-800/40 transition",
                      isFocused && "bg-up/10 border-l-2 border-up"
                    )}
                  >
                    <td className="py-3 px-3 font-bold text-slate-400">
                      {index === 0 ? (
                        <span className="text-yellow-400 font-black flex items-center gap-1">🥇 1</span>
                      ) : index === 1 ? (
                        <span className="text-slate-300 font-black flex items-center gap-1">🥈 2</span>
                      ) : index === 2 ? (
                        <span className="text-amber-600 font-black flex items-center gap-1">🥉 3</span>
                      ) : (
                        `#${index + 1}`
                      )}
                    </td>

                    <td className="py-3 px-3">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: ag.status === "alive" ? ag.color : "#ef4444" }}
                        />
                        <span className="font-bold text-white" style={{ color: ag.color }}>
                          {ag.id}
                        </span>
                        {ag.status === "dead" && (
                          <span className="text-[9px] bg-red-950 text-red-400 px-1 py-0.2 rounded">DEAD</span>
                        )}
                      </div>
                    </td>

                    <td className="py-3 px-3 font-bold">
                      <span className={ag.return_pct >= 0 ? "text-emerald-400" : "text-rose-400"}>
                        {ag.return_pct >= 0 ? "+" : ""}
                        {ag.return_pct.toFixed(2)}%
                      </span>
                    </td>

                    <td className="py-3 px-3 font-bold text-white">${ag.portfolio_value.toFixed(2)}</td>

                    <td className="py-3 px-3 text-slate-400">
                      ${ag.peak_portfolio_value ? ag.peak_portfolio_value.toFixed(2) : ag.portfolio_value.toFixed(2)}
                    </td>

                    {/* Micro Sparkline */}
                    <td className="py-3 px-3">
                      <div className="w-24 h-6">
                        {sparklinePoints ? (
                          <svg viewBox="0 0 100 30" className="w-full h-full overflow-visible">
                            <polyline
                              fill="none"
                              stroke={ag.return_pct >= 0 ? "#10b981" : "#ef4444"}
                              strokeWidth="1.8"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              points={sparklinePoints}
                            />
                          </svg>
                        ) : (
                          <span className="text-slate-600 text-[10px]">Flat</span>
                        )}
                      </div>
                    </td>

                    <td className="py-3 px-3 text-slate-400">Gen {ag.generation}</td>

                    <td className="py-3 px-3 text-slate-300">{ag.win_rate_pct ?? "0.0"}%</td>

                    <td className="py-3 px-3 text-slate-400">{ag.total_trades ?? 0}</td>

                    <td className="py-3 px-3 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={() => onFocusAgent(isFocused ? null : ag.id)}
                          className={cn(
                            "px-2 py-1 rounded text-[10px] font-bold transition cursor-pointer",
                            isFocused
                              ? "bg-up text-background"
                              : "bg-background border border-border text-slate-400 hover:text-white"
                          )}
                          title="Highlight trade markers and equity curve on chart"
                        >
                          {isFocused ? "Focused" : "Focus"}
                        </button>
                        <button
                          onClick={() => onSelectAgent(ag.id)}
                          className="px-2 py-1 rounded text-[10px] font-bold bg-panel border border-border text-cyan-400 hover:bg-cyan-500/20 transition cursor-pointer"
                          title="Inspect neural brain"
                        >
                          Brain
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
              {sortedAgents.length === 0 && (
                <tr>
                  <td colSpan={10} className="py-12 text-center text-slate-500">
                    No agents found in swarm. Start simulation to evolve genomes.
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
