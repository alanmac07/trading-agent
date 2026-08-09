"use client";

import React, { useMemo, useState } from "react";
import { AgentData, EquityPoint } from "@/lib/api";
import { 
  TrendingUp, 
  Skull, 
  GitBranch, 
  Activity, 
  DollarSign, 
  ShieldAlert, 
  Layers, 
  Info,
  Maximize2,
  Sparkles
} from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface EquityCurveChartProps {
  agents: AgentData[];
  graveyardPool: number;
  focusedAgentId?: string | null;
  height?: number;
  onFocusAgent?: (agentId: string | null) => void;
}

export default function EquityCurveChart({
  agents,
  graveyardPool,
  focusedAgentId,
  height = 190,
  onFocusAgent,
}: EquityCurveChartProps) {
  const [hoveredAgentId, setHoveredAgentId] = useState<string | null>(null);
  const [filterMode, setFilterMode] = useState<"all" | "alive" | "dead">("all");
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  // Process and compute scaled coordinate points across all agents
  const {
    lines,
    minVal,
    maxVal,
    minStep,
    maxStep,
    totalAliveCapital,
    activeCount,
    deadCount,
    clonedCount,
  } = useMemo(() => {
    if (!agents || agents.length === 0) {
      return {
        lines: [],
        minVal: 9000,
        maxVal: 11000,
        minStep: 0,
        maxStep: 100,
        totalAliveCapital: 0,
        activeCount: 0,
        deadCount: 0,
        clonedCount: 0,
      };
    }

    let minS = Infinity;
    let maxS = -Infinity;
    let minC = Infinity;
    let maxC = -Infinity;

    let aliveCount = 0;
    let deadCount = 0;
    let clonedCount = 0;
    let aliveCap = 0;
    const dateMap = new Map<number, string>();

    // Scan steps and capital bounds
    for (const a of agents) {
      const isAlive = a.status === "alive";
      if (isAlive) {
        aliveCount++;
        aliveCap += a.current_capital ?? a.portfolio_value ?? 0;
      } else {
        deadCount++;
      }
      if (a.generation > 0 || a.parent_id) {
        clonedCount++;
      }

      // Check equity_history
      if (a.equity_history && a.equity_history.length > 0) {
        for (const pt of a.equity_history) {
          if (pt.step < minS) minS = pt.step;
          if (pt.step > maxS) maxS = pt.step;
          if (pt.capital < minC) minC = pt.capital;
          if (pt.capital > maxC) maxC = pt.capital;
        }
      } else if (a.portfolio_history && a.portfolio_history.length > 0) {
        const spawn = a.spawn_step ?? 0;
        for (let idx = 0; idx < a.portfolio_history.length; idx++) {
          const s = spawn + idx;
          const cap = a.portfolio_history[idx];
          if (s < minS) minS = s;
          if (s > maxS) maxS = s;
          if (cap < minC) minC = cap;
          if (cap > maxC) maxC = cap;
        }
      }
    }

    if (minS === Infinity) minS = 0;
    if (maxS === -Infinity || maxS === minS) maxS = minS + 20;

    // Always include baseline $10,000 starting capital for perspective
    if (minC === Infinity) minC = 9500;
    if (maxC === -Infinity) maxC = 10500;
    minC = Math.min(minC, 9500);
    maxC = Math.max(maxC, 10500);

    // Add 6% vertical padding
    const pad = (maxC - minC) * 0.06;
    const finalMinC = Math.max(0, minC - pad);
    const finalMaxC = maxC + pad;
    const stepRange = Math.max(1, maxS - minS);
    const capRange = Math.max(1, finalMaxC - finalMinC);

    // Compute SVG point strings and marker coordinates
    const agentLines = agents.map((a) => {
      const isAlive = a.status === "alive";
      const isDead = a.status === "dead" || a.death_step != null;
      const deathStep = a.death_step;
      const spawnStep = a.spawn_step ?? minS;

      // Extract raw points
      let rawPoints: EquityPoint[] = [];
      if (a.equity_history && a.equity_history.length > 0) {
        rawPoints = a.equity_history;
      } else if (a.portfolio_history && a.portfolio_history.length > 0) {
        rawPoints = a.portfolio_history.map((cap, i) => {
          if (a.date_history && a.date_history[i]) {
            dateMap.set(spawnStep + i, a.date_history[i]);
          }
          return {
            step: spawnStep + i,
            capital: cap,
          };
        });
      }

      if (rawPoints.length === 0) {
        rawPoints = [{ step: spawnStep, capital: a.current_capital ?? a.portfolio_value }];
      }

      // Map to SVG 0..100 coordinates
      const mapped = rawPoints.map((pt) => {
        const x = ((pt.step - minS) / stepRange) * 100;
        const y = 100 - ((pt.capital - finalMinC) / capRange) * 100;
        return { x, y, step: pt.step, capital: pt.capital, dateStr: dateMap.get(pt.step) || "N/A" };
      });

      const pointsStr = mapped.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");

      const firstPoint = mapped[0];
      const lastPoint = mapped[mapped.length - 1];

      // Death marker coordinate
      let deathMarkerCoord = null;
      if (isDead) {
        deathMarkerCoord = lastPoint;
      }

      // Clone spawn marker coordinate
      let cloneMarkerCoord = null;
      if (a.generation > 0 && a.parent_id) {
        cloneMarkerCoord = firstPoint;
      }

      return {
        id: a.id || a.agent_id || "NEAT_?",
        agent_id: a.agent_id || a.id,
        status: a.status,
        color: a.color,
        generation: a.generation,
        parentId: a.parent_id,
        currentCapital: a.current_capital ?? a.portfolio_value,
        returnPct: a.return_pct,
        trades: a.total_trades ?? 0,
        winRate: a.win_rate_pct ?? 0,
        causeOfDeath: a.cause_of_death,
        spawnStep,
        deathStep,
        points: pointsStr,
        mappedPoints: mapped,
        firstPoint,
        lastPoint,
        deathMarkerCoord,
        cloneMarkerCoord,
        isAlive,
        isDead,
      };
    });

    return {
      lines: agentLines,
      minVal: finalMinC,
      maxVal: finalMaxC,
      minStep: minS,
      maxStep: maxS,
      totalAliveCapital: aliveCap,
      activeCount: aliveCount,
      deadCount: deadCount,
      clonedCount: clonedCount,
    };
  }, [agents]);

  // Filtered lines according to quick tab
  const filteredLines = useMemo(() => {
    if (filterMode === "alive") return lines.filter((l) => l.isAlive);
    if (filterMode === "dead") return lines.filter((l) => l.isDead);
    return lines;
  }, [lines, filterMode]);

  // Determine active inspected agent for tooltip
  const activeHoveredLine = useMemo(() => {
    const targetId = hoveredAgentId || focusedAgentId;
    if (!targetId) return null;
    return lines.find((l) => l.id === targetId || l.agent_id === targetId) || null;
  }, [hoveredAgentId, focusedAgentId, lines]);

  // Baseline $10,000 coordinate
  const baselineY = useMemo(() => {
    if (maxVal === minVal) return 50;
    return 100 - ((10000 - minVal) / (maxVal - minVal)) * 100;
  }, [minVal, maxVal]);

  if (agents.length === 0) {
    return (
      <div
        className="w-full flex flex-col items-center justify-center text-slate-500 font-mono text-xs bg-[#040813] border-t border-border/80 relative"
        style={{ height }}
      >
        <div className="flex items-center gap-2 mb-1 text-slate-400">
          <Activity className="w-4 h-4 text-emerald-400 animate-pulse" />
          <span className="font-bold tracking-wider">SWARM EQUITY STREAM ENGINE READY</span>
        </div>
        <p className="text-[11px] text-slate-600">Start simulation to stream synchronized agent equity curves & death trails.</p>
      </div>
    );
  }

  return (
    <div
      className="w-full bg-[#030712] border-t border-border flex flex-col relative shrink-0 select-none overflow-hidden"
      style={{ height }}
      onMouseMove={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setMousePos({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
        });
      }}
      onMouseLeave={() => {
        setHoveredAgentId(null);
        setMousePos(null);
      }}
    >
      {/* ── TOP HEADER BAR: TITLE, SWARM AGGREGATES & QUICK FILTERS ── */}
      <div className="h-7 px-3 border-b border-border/70 bg-[#070d1d]/90 flex items-center justify-between text-[11px] font-mono shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 font-bold text-slate-200">
            <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
            <span className="tracking-tight text-white font-extrabold">SWARM EQUITY STREAM</span>
          </div>

          <div className="h-3 w-px bg-border/80" />

          {/* Active Capital */}
          <div className="flex items-center gap-1">
            <span className="text-slate-400 text-[10px]">Active Cap:</span>
            <span className="text-emerald-400 font-bold tracking-tight">
              ${totalAliveCapital.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>

          {graveyardPool > 0 && (
            <>
              <div className="h-3 w-px bg-border/80" />
              <div className="flex items-center gap-1 text-[10px]">
                <span className="text-red-400 font-semibold flex items-center gap-1">
                  <Skull className="w-3 h-3 text-red-400" /> Graveyard Pool:
                </span>
                <span className="text-red-300 font-bold">${graveyardPool.toFixed(2)}</span>
              </div>
            </>
          )}
        </div>

        {/* Status Indicators & Filter Pills */}
        <div className="flex items-center gap-2">
          {/* Quick Filters */}
          <div className="flex items-center bg-background/90 border border-border/70 rounded p-0.5 text-[10px] gap-0.5">
            <button
              onClick={() => setFilterMode("all")}
              className={cn(
                "px-1.5 py-0.5 rounded transition cursor-pointer font-bold",
                filterMode === "all" ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40" : "text-slate-400 hover:text-white"
              )}
            >
              All ({lines.length})
            </button>
            <button
              onClick={() => setFilterMode("alive")}
              className={cn(
                "px-1.5 py-0.5 rounded transition cursor-pointer font-bold flex items-center gap-1",
                filterMode === "alive" ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40" : "text-slate-400 hover:text-white"
              )}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse" />
              Alive ({activeCount})
            </button>
            <button
              onClick={() => setFilterMode("dead")}
              className={cn(
                "px-1.5 py-0.5 rounded transition cursor-pointer font-bold flex items-center gap-1",
                filterMode === "dead" ? "bg-red-500/20 text-red-300 border border-red-500/40" : "text-slate-400 hover:text-white"
              )}
            >
              <span className="text-[10px]">☠</span> Dead ({deadCount})
            </button>
          </div>

          <div className="h-3 w-px bg-border/80" />

          {/* Bounds */}
          <div className="flex items-center gap-2 text-[10px] text-slate-400">
            <span>
              Max: <strong className="text-slate-200">${maxVal.toFixed(0)}</strong>
            </span>
            <span>
              Min: <strong className="text-slate-200">${minVal.toFixed(0)}</strong>
            </span>
            <span className="text-slate-500">| Step {minStep}→{maxStep}</span>
          </div>
        </div>
      </div>

      {/* ── MAIN SVG STREAM CANVAS ── */}
      <div className="flex-1 w-full relative overflow-hidden px-2 pt-1 pb-2">
        <svg
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          className="w-full h-full overflow-visible"
        >
          <defs>
            {/* Neon Glow Filters */}
            <filter id="glowGreen" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="0" stdDeviation="0.8" floodColor="#10b981" floodOpacity="0.8" />
            </filter>
            <filter id="glowCyan" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="0" stdDeviation="0.8" floodColor="#00d4ff" floodOpacity="0.8" />
            </filter>
            <filter id="glowPurple" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="0" stdDeviation="0.8" floodColor="#a855f7" floodOpacity="0.8" />
            </filter>
            <filter id="glowRed" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="0" stdDeviation="1.2" floodColor="#ef4444" floodOpacity="0.9" />
            </filter>

            {/* Gradient fills */}
            <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.08" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines: 25%, 50%, 75% */}
          <line x1="0" y1="20" x2="100" y2="20" stroke="rgba(255,255,255,0.04)" strokeDasharray="1.5,2" strokeWidth="0.4" />
          <text x="1.5" y="19" fill="rgba(148, 163, 184, 0.4)" fontSize="2" fontFamily="monospace">${(maxVal - (maxVal - minVal) * 0.2).toLocaleString(undefined, { maximumFractionDigits: 0 })}</text>
          
          <line x1="0" y1="40" x2="100" y2="40" stroke="rgba(255,255,255,0.04)" strokeDasharray="1.5,2" strokeWidth="0.4" />
          <text x="1.5" y="39" fill="rgba(148, 163, 184, 0.4)" fontSize="2" fontFamily="monospace">${(maxVal - (maxVal - minVal) * 0.4).toLocaleString(undefined, { maximumFractionDigits: 0 })}</text>

          <line x1="0" y1="60" x2="100" y2="60" stroke="rgba(255,255,255,0.04)" strokeDasharray="1.5,2" strokeWidth="0.4" />
          <text x="1.5" y="59" fill="rgba(148, 163, 184, 0.4)" fontSize="2" fontFamily="monospace">${(maxVal - (maxVal - minVal) * 0.6).toLocaleString(undefined, { maximumFractionDigits: 0 })}</text>

          <line x1="0" y1="80" x2="100" y2="80" stroke="rgba(255,255,255,0.04)" strokeDasharray="1.5,2" strokeWidth="0.4" />
          <text x="1.5" y="79" fill="rgba(148, 163, 184, 0.4)" fontSize="2" fontFamily="monospace">${(maxVal - (maxVal - minVal) * 0.8).toLocaleString(undefined, { maximumFractionDigits: 0 })}</text>

          {/* $10,000 Starting Capital Baseline */}
          {baselineY >= 0 && baselineY <= 100 && (
            <g>
              <line
                x1="0"
                y1={baselineY}
                x2="100"
                y2={baselineY}
                stroke="rgba(148, 163, 184, 0.25)"
                strokeDasharray="2,3"
                strokeWidth="0.6"
              />
              <text
                x="1.5"
                y={baselineY - 1.5}
                fill="rgba(148, 163, 184, 0.5)"
                fontSize="2.2"
                fontFamily="monospace"
              >
                BASE: $10,000
              </text>
            </g>
          )}

          {/* ── AGENT MULTI-LINE PERFORMANCE CURVES ── */}
          {filteredLines.map((line) => {
            if (!line.points) return null;

            const isTarget = hoveredAgentId === line.id || focusedAgentId === line.id;
            const hasFocus = Boolean(hoveredAgentId || focusedAgentId);

            // Active vs Dead Line Styling
            let strokeColor = line.color || "#10b981";
            let strokeWidth = line.isAlive ? "1.4" : "0.9";
            let opacity = line.isAlive ? 0.85 : 0.18; // Terminated agents faded to ~0.18 opacity

            if (hasFocus) {
              if (isTarget) {
                strokeWidth = "2.8";
                opacity = 1.0;
                strokeColor = line.color;
              } else {
                opacity = line.isAlive ? 0.12 : 0.05;
                strokeWidth = "0.7";
              }
            }

            return (
              <g
                key={line.id}
                className="cursor-pointer group"
                onClick={() => onFocusAgent?.(focusedAgentId === line.id ? null : line.id)}
                onMouseEnter={() => setHoveredAgentId(line.id)}
              >
                {/* Transparent thick hit target for easy mouse hover */}
                <polyline
                  fill="none"
                  stroke="transparent"
                  strokeWidth="8"
                  points={line.points}
                  className="cursor-pointer"
                />

                {/* Main Glowing Polyline with smooth 1.5s CSS transition for deaths/fades */}
                <polyline
                  fill="none"
                  stroke={strokeColor}
                  strokeWidth={strokeWidth}
                  strokeOpacity={opacity}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeDasharray={line.isDead ? "1.5,1.5" : undefined}
                  points={line.points}
                  style={{
                    transition: "stroke-opacity 1.5s ease-out, stroke-width 0.3s ease",
                    filter: isTarget ? `drop-shadow(0 0 2px ${line.color})` : line.isDead ? undefined : `drop-shadow(0 0 1px ${line.color})`,
                  }}
                />

                {/* Clone spawn anchor marker */}
                {line.cloneMarkerCoord && (
                  <g
                    transform={`translate(${line.cloneMarkerCoord.x}, ${line.cloneMarkerCoord.y})`}
                    className="transition-transform duration-200"
                  >
                    <circle r="1.3" fill="#a855f7" stroke="#ffffff" strokeWidth="0.4" />
                  </g>
                )}

                {/* Dead agent termination skull marker ☠ at death_step */}
                {line.isDead && line.deathMarkerCoord && (
                  <g
                    transform={`translate(${line.deathMarkerCoord.x}, ${line.deathMarkerCoord.y})`}
                    className="transition-all duration-300"
                    filter="url(#glowRed)"
                  >
                    <circle r="2.2" fill="#dc2626" stroke="#ffffff" strokeWidth="0.5" />
                    <text
                      textAnchor="middle"
                      dy="0.8"
                      fontSize="2.4"
                      fill="#ffffff"
                      fontWeight="bold"
                      fontFamily="sans-serif"
                    >
                      ☠
                    </text>
                  </g>
                )}

                {/* Active agent live tip pulse dot */}
                {line.isAlive && line.lastPoint && (
                  <g transform={`translate(${line.lastPoint.x}, ${line.lastPoint.y})`}>
                    <circle
                      r="1.2"
                      fill={line.color}
                      stroke="#ffffff"
                      strokeWidth="0.4"
                      className="animate-pulse"
                    />
                    {isTarget && (
                      <circle
                        r="2.8"
                        fill="none"
                        stroke={line.color}
                        strokeWidth="0.5"
                        strokeOpacity="0.7"
                        className="animate-ping"
                      />
                    )}
                  </g>
                )}
              </g>
            );
          })}
        </svg>

        {/* ── INTERACTIVE HOVER TOOLTIP CARD ── */}
        {activeHoveredLine && mousePos && (
          <div
            className="absolute z-30 pointer-events-none transform -translate-y-full mb-2 bg-[#091124]/95 backdrop-blur-md border border-slate-700/80 shadow-2xl rounded-xl p-3 text-xs font-mono w-64 transition-all duration-150"
            style={{
              left: Math.min(Math.max(10, mousePos.x - 120), window.innerWidth - 300),
              top: Math.max(10, mousePos.y - 12),
            }}
          >
            {/* Header */}
            <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-border/80">
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full shrink-0 shadow-sm"
                  style={{ backgroundColor: activeHoveredLine.color }}
                />
                <span className="font-bold text-white text-sm tracking-tight">{activeHoveredLine.id}</span>
              </div>
              <span
                className={cn(
                  "px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider",
                  activeHoveredLine.isAlive
                    ? "bg-emerald-950 text-emerald-400 border border-emerald-800/60"
                    : "bg-red-950 text-red-400 border border-red-800/60"
                )}
              >
                {activeHoveredLine.isAlive ? "● ACTIVE" : "☠ TERMINATED"}
              </span>
            </div>

            {/* Metrics Grid */}
            <div className="space-y-1 text-[11px]">
              <div className="flex justify-between items-center text-slate-300">
                <span className="text-slate-400">Date:</span>
                <span className="font-bold text-white font-mono">
                  {activeHoveredLine.lastPoint?.dateStr || "N/A"}
                </span>
              </div>
              
              <div className="flex justify-between items-center text-slate-300">
                <span className="text-slate-400">Current Capital:</span>
                <span className="font-bold text-white font-mono">
                  ${activeHoveredLine.currentCapital.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-slate-400">Total Return:</span>
                <span
                  className={cn(
                    "font-bold font-mono px-1 rounded",
                    activeHoveredLine.returnPct >= 0
                      ? "text-emerald-400 bg-emerald-950/40"
                      : "text-red-400 bg-red-950/40"
                  )}
                >
                  {activeHoveredLine.returnPct >= 0 ? "+" : ""}
                  {activeHoveredLine.returnPct.toFixed(2)}%
                </span>
              </div>

              <div className="flex justify-between items-center text-slate-300">
                <span className="text-slate-400">Generation:</span>
                <span className="text-slate-200">
                  Gen {activeHoveredLine.generation}
                  {activeHoveredLine.parentId ? ` (from ${activeHoveredLine.parentId})` : " (Seed)"}
                </span>
              </div>

              <div className="flex justify-between items-center text-slate-300">
                <span className="text-slate-400">Step Range:</span>
                <span className="text-cyan-300">
                  {activeHoveredLine.spawnStep} → {activeHoveredLine.deathStep ?? maxStep}
                </span>
              </div>

              {activeHoveredLine.causeOfDeath && (
                <div className="pt-1 mt-1 border-t border-red-900/40 text-[10px] text-red-300 font-mono">
                  <span className="font-bold text-red-400">Cause: </span>
                  {activeHoveredLine.causeOfDeath}
                </div>
              )}
            </div>

            {/* Tip Footer */}
            <div className="mt-2 pt-1 border-t border-border/40 text-[9px] text-slate-400 flex items-center justify-between">
              <span>Click curve to lock focus</span>
              <span className="text-slate-500 font-bold">{activeHoveredLine.trades} Trades</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
