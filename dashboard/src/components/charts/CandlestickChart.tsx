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

const VISIBLE_WINDOW = 50;

export function CandlestickChart({ steps, trades }: CandlestickChartProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const chartRef = useRef<IChartApi | null>(null);
	const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
	const volumeSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
	const prevStepCountRef = useRef(0);

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
				timeVisible: true,
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
		if (!candleSeriesRef.current || !volumeSeriesRef.current || !chartRef.current) return;
		if (steps.length === 0) {
			prevStepCountRef.current = 0;
			return;
		}

		const timeFor = (s: { step: number; timestamp?: number }) => (s.timestamp ?? s.step) as Time;

		const prevCount = prevStepCountRef.current;

		if (prevCount === 0) {
			// First data or reset — full setData
			const candleData: CandlestickData<Time>[] = steps.map((s) => ({
				time: timeFor(s),
				open: s.open,
				high: s.high,
				low: s.low,
				close: s.close,
			}));
			const volumeData = steps.map((s) => ({
				time: timeFor(s),
				value: s.volume,
			}));
			candleSeriesRef.current.setData(candleData);
			volumeSeriesRef.current.setData(volumeData);
		} else {
			// Incremental — only update new steps (time must be strictly increasing)
			let lastTime = prevCount > 0 ? (timeFor(steps[prevCount - 1]) as number) : -1;
			for (let i = prevCount; i < steps.length; i++) {
				const s = steps[i];
				const t = timeFor(s) as number;
				if (t <= lastTime) continue; // skip non-monotonic timestamps
				lastTime = t;
				candleSeriesRef.current.update({
					time: t as Time,
					open: s.open,
					high: s.high,
					low: s.low,
					close: s.close,
				});
				volumeSeriesRef.current.update({
					time: t as Time,
					value: s.volume,
				});
			}
		}

		prevStepCountRef.current = steps.length;

		// Trade markers (must be set on full array each time)
		if (trades.length > 0) {
			const markers: SeriesMarker<Time>[] = trades
				.map((t) => ({
					time: (t.timestamp ?? t.step) as Time,
					position: (t.side === "buy" ? "belowBar" : "aboveBar") as "belowBar" | "aboveBar",
					color: t.side === "buy" ? "#22c55e" : "#ef4444",
					shape: (t.side === "buy" ? "arrowUp" : "arrowDown") as "arrowUp" | "arrowDown",
					text: `${t.side.toUpperCase()} ${t.size.toFixed(4)} @ ${t.price.toFixed(2)}`,
				}))
				.sort((a, b) => (a.time as number) - (b.time as number));

			candleSeriesRef.current.setMarkers(markers);
		}

		// Rolling window: show last VISIBLE_WINDOW candles, scrolling right
		const last = steps[steps.length - 1];
		const lastTime = timeFor(last) as number;
		const firstVisible = steps[Math.max(0, steps.length - VISIBLE_WINDOW)];
		const fromTime = timeFor(firstVisible) as number;
		chartRef.current.timeScale().setVisibleRange({
			from: fromTime as Time,
			to: (lastTime + 2) as Time,
		});
	}, [steps, trades]);

	return (
		<div className="relative h-full w-full">
			<div ref={containerRef} className="h-full w-full" />
			{steps.length === 0 && (
				<div className="absolute inset-0 flex items-center justify-center text-[var(--text-secondary)]">
					No price data available
				</div>
			)}
		</div>
	);
}
