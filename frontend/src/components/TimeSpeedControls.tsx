"use client";

import React from "react";
import { Zap, FastForward } from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: any[]) {
  return twMerge(clsx(inputs));
}

interface TimeSpeedControlsProps {
  playbackSpeed: number;
  setPlaybackSpeed: (speed: number) => void;
  daysPerStep: number;
  setDaysPerStep: (days: number) => void;
  disabled?: boolean;
}

const SPEED_PRESETS = [
  { label: "0.5x", speedMs: 500, days: 1 },
  { label: "1x", speedMs: 200, days: 1 },
  { label: "5x", speedMs: 80, days: 1 },
  { label: "20x", speedMs: 40, days: 2 },
  { label: "50x", speedMs: 25, days: 5 },
];

export default function TimeSpeedControls({
  playbackSpeed,
  setPlaybackSpeed,
  daysPerStep,
  setDaysPerStep,
  disabled = false,
}: TimeSpeedControlsProps) {
  return (
    <div className="flex items-center gap-2 bg-background/80 border border-border/80 rounded-lg px-2.5 py-1 text-xs">
      <div className="flex items-center gap-1 text-yellow-400 font-bold font-mono tracking-wider text-[10px] uppercase mr-1">
        <FastForward className="w-3.5 h-3.5" /> Warp:
      </div>

      <div className="flex items-center gap-1">
        {SPEED_PRESETS.map((preset) => {
          const isActive = playbackSpeed === preset.speedMs && daysPerStep === preset.days;
          return (
            <button
              key={preset.label}
              onClick={() => {
                setPlaybackSpeed(preset.speedMs);
                setDaysPerStep(preset.days);
              }}
              disabled={disabled}
              className={cn(
                "px-2 py-0.5 rounded text-[11px] font-mono font-bold transition cursor-pointer",
                isActive
                  ? "bg-yellow-400/20 text-yellow-400 border border-yellow-400/50 shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-panel border border-transparent"
              )}
            >
              {preset.label}
            </button>
          );
        })}
      </div>

      <div className="h-4 w-px bg-border mx-1" />

      {/* Dynamic slider */}
      <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400">
        <span>{playbackSpeed}ms</span>
        <input
          type="range"
          min="20"
          max="800"
          step="20"
          value={playbackSpeed}
          onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
          disabled={disabled}
          className="w-16 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-yellow-400"
          title="Step interval delay"
        />
      </div>
    </div>
  );
}
