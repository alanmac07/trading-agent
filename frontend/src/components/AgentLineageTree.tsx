"use client";

import React, { useEffect, useState, useMemo, useCallback } from "react";
import { fetchAgentLineage, LineageData, LineageNode as ApiLineageNode } from "@/lib/api";
import { 
  ReactFlow, 
  Controls, 
  Background, 
  useNodesState, 
  useEdgesState, 
  MarkerType,
  Handle,
  Position,
  NodeProps,
  Edge,
  Node
} from "@xyflow/react";
import '@xyflow/react/dist/style.css';
import { motion, AnimatePresence } from "framer-motion";
import { 
  GitBranch, 
  Dna, 
  Activity, 
  X, 
  Zap, 
  ShieldAlert
} from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: any[]) {
  return twMerge(clsx(inputs));
}

interface AgentLineageTreeProps {
  onSelectAgent?: (id: string) => void;
}

const AgentNode = ({ data }: NodeProps) => {
  const isAlive = data.status === "alive";
  return (
    <div className={cn(
      "w-56 p-3 rounded-xl border backdrop-blur-md transition-all shadow-xl",
      isAlive
        ? "border-emerald-500/60 bg-[#09152a]/90 shadow-emerald-500/20 ring-1 ring-emerald-500/50"
        : "border-rose-900/60 bg-[#12080c]/80 opacity-80"
    )}>
      <Handle type="target" position={Position.Left} className="w-2 h-2 !bg-slate-500" />
      
      {/* Node Header */}
      <div className="flex items-center justify-between mb-2 border-b border-white/10 pb-2">
        <div className="flex flex-col">
          <span className="font-mono font-black text-sm text-white truncate max-w-[120px]">
            {data.id}
          </span>
          <span className="text-[9px] text-slate-400 font-mono mt-0.5">
            Gen {data.generation} | Parent: {data.parent_id || "Seed"}
          </span>
        </div>
        {isAlive ? (
          <span className="text-[9px] font-mono font-bold text-emerald-400 bg-emerald-950/80 px-1.5 py-0.5 rounded-full border border-emerald-800/40 shadow-[0_0_8px_rgba(16,185,129,0.4)]">
            ● ALIVE
          </span>
        ) : (
          <span className="text-[9px] font-mono font-bold text-rose-400 bg-rose-950/80 px-1.5 py-0.5 rounded-full border border-rose-800/40">
            💀 TERMINATED
          </span>
        )}
      </div>

      {/* Stats */}
      <div className="space-y-1.5 text-[10px] font-mono">
        <div className="flex justify-between items-center">
          <span className="text-slate-400">Peak Return:</span>
          <strong className={data.return_pct >= 0 ? "text-emerald-400" : "text-rose-400"}>
            {data.return_pct >= 0 ? "+" : ""}{data.return_pct.toFixed(1)}%
          </strong>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-slate-400">Final Capital:</span>
          <span className="text-slate-200">${data.portfolio_value.toFixed(0)}</span>
        </div>
        <div className="flex justify-between items-center pt-1.5 border-t border-border/40 text-[10px] text-purple-400 mt-1">
          <span>Mutation Trigger:</span>
          <span className="font-bold">+{data.mutations_count} diffs</span>
        </div>
      </div>
      
      <Handle type="source" position={Position.Right} className="w-2 h-2 !bg-slate-500" />
    </div>
  );
};

const nodeTypes = {
  agentNode: AgentNode,
};

export default function AgentLineageTree({ onSelectAgent }: AgentLineageTreeProps) {
  const [lineage, setLineage] = useState<LineageData | null>(null);
  const [selectedNodeData, setSelectedNodeData] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 2000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      const data = await fetchAgentLineage();
      setLineage(data);
      setLoading(false);
    } catch (e) {
      console.error(e);
      setLoading(false);
    }
  };

  // Group nodes by generation for clean hierarchical layout
  useEffect(() => {
    if (!lineage?.nodes || lineage.nodes.length === 0) return;

    const genMap = new Map<number, ApiLineageNode[]>();
    for (const node of lineage.nodes) {
      const g = node.generation || 0;
      if (!genMap.has(g)) genMap.set(g, []);
      genMap.get(g)!.push(node);
    }

    const sortedGens = Array.from(genMap.keys()).sort((a, b) => a - b);
    const flowNodes: Node[] = [];
    const GEN_GAP_X = 350;
    const NODE_GAP_Y = 180;

    sortedGens.forEach((g, gIdx) => {
      const nodesInGen = genMap.get(g)!;
      const totalH = (nodesInGen.length - 1) * NODE_GAP_Y;
      const startY = Math.max(0, 300 - totalH / 2);

      nodesInGen.forEach((n, nIdx) => {
        const y = startY + nIdx * NODE_GAP_Y;
        flowNodes.push({
          id: n.id,
          type: 'agentNode',
          position: { x: gIdx * GEN_GAP_X, y },
          data: { ...n },
        });
      });
    });

    const flowEdges: Edge[] = (lineage.edges || []).map((e) => ({
      id: `e-${e.source}-${e.target}`,
      source: e.source,
      target: e.target,
      animated: true,
      style: { stroke: '#00d4ff', strokeWidth: 1.5, opacity: 0.7 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#00d4ff' }
    }));

    // Update state only if changed significantly (prevent layout jumping if possible)
    setNodes((nds) => {
      // Basic merge to keep positions if dragged
      const newNds = flowNodes.map(fn => {
        const existing = nds.find(n => n.id === fn.id);
        if (existing) {
           return { ...existing, data: fn.data };
        }
        return fn;
      });
      return newNds;
    });
    setEdges(flowEdges);

  }, [lineage, setNodes, setEdges]);

  const onNodeClick = useCallback((_, node: Node) => {
    setSelectedNodeData(node.data);
  }, []);

  return (
    <div className="flex-1 flex flex-col h-full bg-[#040813] text-foreground relative overflow-hidden">
      {/* Header Bar */}
      <div className="h-14 px-6 border-b border-border bg-panel/90 backdrop-blur-md flex items-center justify-between shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/30 text-purple-400">
            <GitBranch className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-sm font-black text-white flex items-center gap-2">
              NEAT Agent Lineage & Splitting Tree
            </h2>
            <p className="text-[10px] text-slate-400 font-mono">
              Live Evolutionary Genealogy & Clonal Mutation Network
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="text-slate-400">
            Total Genomes: <strong className="text-white">{lineage?.total_agents || 0}</strong>
          </span>
        </div>
      </div>

      {/* Main Canvas Workspace */}
      <div className="flex-1 relative">
        {nodes.length > 0 ? (
           <ReactFlow
             nodes={nodes}
             edges={edges}
             onNodesChange={onNodesChange}
             onEdgesChange={onEdgesChange}
             nodeTypes={nodeTypes}
             onNodeClick={onNodeClick}
             fitView
             attributionPosition="bottom-left"
             minZoom={0.2}
           >
             <Background color="#334155" gap={20} size={1} />
             <Controls />
           </ReactFlow>
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-500 font-mono h-full text-center">
            <GitBranch className="w-12 h-12 text-slate-700 animate-pulse mb-3" />
            <div className="text-sm font-bold text-slate-400">Lineage Map Empty</div>
            <div className="text-xs text-slate-600 mt-1">
              Start the simulation. When profitable agents reproduce (+8%), new mutated clone nodes will branch out dynamically.
            </div>
          </div>
        )}
      </div>

      {/* ── GENOME MUTATION INSPECTOR SLIDEOUT DRAWER ── */}
      <AnimatePresence>
        {selectedNodeData && (
          <motion.div
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="absolute top-0 right-0 bottom-0 w-96 bg-panel/95 backdrop-blur-xl border-l border-border p-6 shadow-2xl z-30 flex flex-col overflow-y-auto"
          >
            {/* Modal Header */}
            <div className="flex justify-between items-center border-b border-border pb-4 mb-4">
              <div className="flex items-center gap-2.5">
                <div
                  className="w-8 h-8 rounded-xl flex items-center justify-center border border-white/10"
                  style={{ backgroundColor: `${selectedNodeData.color}25`, color: selectedNodeData.color }}
                >
                  <Dna className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-black text-sm text-white font-mono">{selectedNodeData.id}</h3>
                  <div className="text-[10px] text-slate-400 font-mono">
                    Gen {selectedNodeData.generation} {selectedNodeData.parent_id ? `(Parent: ${selectedNodeData.parent_id})` : "(Seed)"}
                  </div>
                </div>
              </div>
              <X
                className="w-5 h-5 cursor-pointer text-slate-400 hover:text-white transition"
                onClick={() => setSelectedNodeData(null)}
              />
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-2 gap-2 mb-4 text-xs font-mono">
              <div className="p-2.5 rounded-xl bg-background border border-border/60">
                <div className="text-slate-500 text-[10px]">LIFETIME P&L</div>
                <div className={cn("text-base font-bold", selectedNodeData.return_pct >= 0 ? "text-emerald-400" : "text-rose-400")}>
                  {selectedNodeData.return_pct >= 0 ? "+" : ""}{selectedNodeData.return_pct.toFixed(2)}%
                </div>
              </div>
              <div className="p-2.5 rounded-xl bg-background border border-border/60">
                <div className="text-slate-500 text-[10px]">CURRENT VALUE</div>
                <div className="text-base font-bold text-white">${selectedNodeData.portfolio_value.toFixed(2)}</div>
              </div>
            </div>

            {/* Neural Complexity Stats */}
            <div className="p-3 rounded-xl bg-background border border-border/60 mb-4 space-y-2 text-xs font-mono">
              <div className="text-slate-400 font-bold flex items-center gap-1.5 text-[11px]">
                <Activity className="w-3.5 h-3.5 text-cyan-400" /> Neural Topology Complexity
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Active Neurons:</span>
                <strong className="text-white">{selectedNodeData.nodes_count} nodes</strong>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Synaptic Connections:</span>
                <strong className="text-white">{selectedNodeData.connections_count} synapses</strong>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Total Mutations from Parent:</span>
                <strong className="text-purple-400">+{selectedNodeData.mutations_count} adjustments</strong>
              </div>
            </div>

            {/* Cause of death if dead */}
            {selectedNodeData.cause_of_death && (
              <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 mb-4 text-xs font-mono text-rose-300 flex items-start gap-2">
                <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-rose-400">Cull Reason:</div>
                  <div className="text-[11px] text-slate-300">{selectedNodeData.cause_of_death}</div>
                </div>
              </div>
            )}

            {/* Detailed Genome Mutations List */}
            <div className="flex-1 flex flex-col space-y-2">
              <div className="text-xs font-bold font-mono text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-yellow-400" /> Inherited Genome Mutation Diff
              </div>

              <div className="flex-1 overflow-y-auto space-y-2 max-h-72 pr-1">
                {selectedNodeData.mutations && selectedNodeData.mutations.length > 0 ? (
                  selectedNodeData.mutations.map((m: any, idx: number) => (
                    <div key={idx} className="p-2.5 rounded-xl bg-background border border-border/50 text-[11px] font-mono space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-bold text-purple-400">{m.type}</span>
                        {m.weight_delta !== undefined && (
                          <span className={cn("text-[10px] font-bold", m.weight_delta >= 0 ? "text-emerald-400" : "text-rose-400")}>
                            Δ {m.weight_delta >= 0 ? "+" : ""}{m.weight_delta}
                          </span>
                        )}
                      </div>
                      <div className="text-slate-300 text-[10px]">{m.synapse || `Node #${m.node_id}`}</div>
                      {m.child_weight !== undefined && (
                        <div className="flex justify-between text-[10px] text-slate-500">
                          <span>Child W: {m.child_weight}</span>
                          {m.parent_weight !== undefined && <span>Parent W: {m.parent_weight}</span>}
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="p-4 rounded-xl bg-background border border-border/40 text-center text-xs text-slate-500 font-mono">
                    Seed Champion Genome (Root Generation). No ancestor mutations.
                  </div>
                )}
              </div>
            </div>

            {/* Action footer */}
            {onSelectAgent && (
              <button
                onClick={() => {
                  onSelectAgent(selectedNodeData.id);
                  setSelectedNodeData(null);
                }}
                className="mt-4 w-full py-2 rounded-xl bg-up hover:bg-up/90 text-background font-black font-mono text-xs transition cursor-pointer shadow-lg shadow-up/20"
              >
                Inspect Neural Brain & Signals →
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
