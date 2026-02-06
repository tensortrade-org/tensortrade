"use client";

import type { StepUpdate, TradeEvent } from "@/lib/types";
import {
	type CandlestickData,
	type CandlestickSeriesOptions,
	ColorType,
	type DeepPartial,
	type IChartApi,
	type ISeriesApi,
	type LineSeriesOptions,
	type SeriesMarker,
	type Time,
	createChart,
} from "lightweight-charts";
import { useEffect, useRef } from "react";

interface CandlestickChartProps {
	steps: StepUpdate[];
	trades: TradeEvent[];
}

export function CandlestickChart({ steps, trades }: CandlestickChartProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const chartRef = useRef<IChartApi | null>(null);
	const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
	const volumeSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

	useEffect(() => {
		if (!containerRef.current) return;

		const chart = createChart(containerRef.current, {
			layout: {
				background: { type: ColorType.Solid, color: "#1e2235" },
				textColor: "#8b8fa3",
			},
			grid: {
				vertLines: { color: "#2a2e45" },
				horzLines: { color: "#2a2e45" },
			},
			crosshair: {
				vertLine: { color: "#3b82f6", width: 1, labelBackgroundColor: "#3b82f6" },
				horzLine: { color: "#3b82f6", width: 1, labelBackgroundColor: "#3b82f6" },
			},
			rightPriceScale: {
				borderColor: "#2a2e45",
			},
			timeScale: {
				borderColor: "#2a2e45",
				timeVisible: false,
			},
			width: containerRef.current.clientWidth,
			height: containerRef.current.clientHeight,
		});

		chartRef.current = chart;

		const candlestickOptions: DeepPartial<CandlestickSeriesOptions> = {
			upColor: "#22c55e",
			downColor: "#ef4444",
			borderUpColor: "#22c55e",
			borderDownColor: "#ef4444",
			wickUpColor: "#22c55e",
			wickDownColor: "#ef4444",
		};

		const candleSeries = chart.addCandlestickSeries(candlestickOptions);
		candleSeriesRef.current = candleSeries;

		const volumeOptions: DeepPartial<LineSeriesOptions> = {
			color: "#3b82f680",
			lineWidth: 1,
			priceScaleId: "volume",
		};

		const volumeSeries = chart.addLineSeries(volumeOptions);
		volumeSeriesRef.current = volumeSeries;

		chart.priceScale("volume").applyOptions({
			scaleMargins: { top: 0.8, bottom: 0 },
			visible: false,
		});

		const handleResize = () => {
			if (containerRef.current && chartRef.current) {
				chartRef.current.applyOptions({
					width: containerRef.current.clientWidth,
					height: containerRef.current.clientHeight,
				});
			}
		};

		const resizeObserver = new ResizeObserver(handleResize);
		resizeObserver.observe(containerRef.current);

		return () => {
			resizeObserver.disconnect();
			chart.remove();
			chartRef.current = null;
			candleSeriesRef.current = null;
			volumeSeriesRef.current = null;
		};
	}, []);

	useEffect(() => {
		if (!candleSeriesRef.current || !volumeSeriesRef.current) return;
		if (steps.length === 0) return;

		const candleData: CandlestickData<Time>[] = steps.map((s) => ({
			time: s.step as Time,
			open: s.open,
			high: s.high,
			low: s.low,
			close: s.close,
		}));

		const volumeData = steps.map((s) => ({
			time: s.step as Time,
			value: s.volume,
		}));

		candleSeriesRef.current.setData(candleData);
		volumeSeriesRef.current.setData(volumeData);

		if (trades.length > 0) {
			const markers: SeriesMarker<Time>[] = trades
				.map((t) => ({
					time: t.step as Time,
					position: (t.side === "buy" ? "belowBar" : "aboveBar") as "belowBar" | "aboveBar",
					color: t.side === "buy" ? "#22c55e" : "#ef4444",
					shape: (t.side === "buy" ? "arrowUp" : "arrowDown") as "arrowUp" | "arrowDown",
					text: `${t.side.toUpperCase()} ${t.size.toFixed(4)} @ ${t.price.toFixed(2)}`,
				}))
				.sort((a, b) => (a.time as number) - (b.time as number));

			candleSeriesRef.current.setMarkers(markers);
		}

		if (chartRef.current) {
			chartRef.current.timeScale().fitContent();
		}
	}, [steps, trades]);

	if (steps.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-[var(--text-secondary)]">
				No price data available
			</div>
		);
	}

	return <div ref={containerRef} className="h-full w-full" />;
}
