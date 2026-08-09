"use client";

import React, { useEffect, useRef } from "react";
import { 
  createChart, 
  CandlestickSeries, 
  HistogramSeries, 
  LineSeries, 
  createSeriesMarkers, 
  IChartApi, 
  CandlestickData, 
  Time 
} from "lightweight-charts";
import { MarketData, TradeData } from "@/lib/api";

interface TradingChartProps {
  initialData: MarketData[];
  latestCandle?: MarketData | null;
  trades: TradeData[];
  focusedAgentId?: string | null;
}

export default function TradingChart({ 
  initialData, 
  latestCandle, 
  trades, 
  focusedAgentId 
}: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<any>(null);
  const maSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const markersPluginRef = useRef<any>(null);
  const lastUpdatedDateRef = useRef<string>("");

  // Initialize chart & set initial historical data
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart instance
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: "#060d1b" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.04)" },
        horzLines: { color: "rgba(255, 255, 255, 0.04)" },
      },
      crosshair: {
        mode: 0,
      },
      timeScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
      },
      rightPriceScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
      },
    });
    chartRef.current = chart;

    // Candlestick series
    const mainSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#ef4444",
    });
    seriesRef.current = mainSeries;

    // MA20 line series overlay
    const maSeries = chart.addSeries(LineSeries, {
      color: "#00d4ff",
      lineWidth: 2,
      crosshairMarkerVisible: false,
    });
    maSeriesRef.current = maSeries;

    // Initialize Markers Primitive plugin
    markersPluginRef.current = createSeriesMarkers(mainSeries, []);

    // Volume series
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "", // overlay
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });
    volumeSeriesRef.current = volumeSeries;

    // Load initial data
    if (initialData && initialData.length > 0) {
      const cData: CandlestickData[] = initialData.map(d => ({
        time: d.DateStr as Time,
        open: d.Open,
        high: d.High,
        low: d.Low,
        close: d.Close,
      }));
      
      const maData = initialData.filter(d => (d.MA20 || 0) > 0).map(d => ({
        time: d.DateStr as Time,
        value: d.MA20 || d.Close,
      }));

      const vData = initialData.map(d => ({
        time: d.DateStr as Time,
        value: d.Volume,
        color: d.Close >= d.Open ? "rgba(16, 185, 129, 0.3)" : "rgba(239, 68, 68, 0.3)",
      }));

      mainSeries.setData(cData);
      maSeries.setData(maData);
      volumeSeries.setData(vData);
      chart.timeScale().fitContent();
    }

    // Auto-resize
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };
    window.addEventListener("resize", handleResize);
    setTimeout(handleResize, 50);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [initialData]);

  // Handle live incremental candle updates
  useEffect(() => {
    if (!latestCandle || !seriesRef.current || !volumeSeriesRef.current) return;
    if (lastUpdatedDateRef.current === latestCandle.DateStr) return;
    lastUpdatedDateRef.current = latestCandle.DateStr;

    try {
      seriesRef.current.update({
        time: latestCandle.DateStr as Time,
        open: latestCandle.Open,
        high: latestCandle.High,
        low: latestCandle.Low,
        close: latestCandle.Close,
      });

      volumeSeriesRef.current.update({
        time: latestCandle.DateStr as Time,
        value: latestCandle.Volume,
        color: latestCandle.Close >= latestCandle.Open ? "rgba(16, 185, 129, 0.3)" : "rgba(239, 68, 68, 0.3)",
      });

      if (maSeriesRef.current && latestCandle.MA20 && latestCandle.MA20 > 0) {
        maSeriesRef.current.update({
          time: latestCandle.DateStr as Time,
          value: latestCandle.MA20,
        });
      }

      // Smoothly scroll to the latest active candle
      if (chartRef.current) {
        chartRef.current.timeScale().scrollToRealTime();
      }
    } catch (e) {
      console.warn("Error updating live chart candle:", e);
    }
  }, [latestCandle]);

  // Update markers when trades change or focus filter changes
  useEffect(() => {
    if (!markersPluginRef.current) return;

    const activeTrades = focusedAgentId 
      ? trades.filter(t => t.agent_id === focusedAgentId)
      : trades;

    const markers = activeTrades.map(t => ({
      time: (t.date ? t.date.split(" ")[0] : "") as Time,
      position: (t.action === "BUY" ? "belowBar" : "aboveBar") as any,
      color: t.action === "BUY" ? "#10b981" : "#ef4444",
      shape: (t.action === "BUY" ? "arrowUp" : "arrowDown") as any,
      text: `${t.action} $${Number(t.price).toFixed(2)} (${t.agent_id.split('_')[1] || t.agent_id})`,
      size: 1,
    })).filter(m => Boolean(m.time));
    
    // Sort chronologically as required by lightweight-charts
    markers.sort((a, b) => String(a.time).localeCompare(String(b.time)));
    
    try {
      markersPluginRef.current.setMarkers(markers);
    } catch (e) {
      console.warn("Failed to set chart markers:", e);
    }
  }, [trades, focusedAgentId]);

  return (
    <div className="absolute inset-0 w-full h-full" ref={chartContainerRef} />
  );
}
