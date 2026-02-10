"use client";

import type { GenomeChannelConfig, GenomeTooltipData } from "@/lib/cortex-types";
import type { TrainingUpdate } from "@/lib/types";
import { useEffect, useMemo, useRef, useState } from "react";

const CELL_W = 5;
const CELL_H = 22;
const GAP = 1;
const LABEL_W = 70;

const CHANNELS: GenomeChannelConfig[] = [
	{ key: "episode_return_mean", label: "Return", colorMode: "diverging" },
	{ key: "pnl_mean", label: "PnL", colorMode: "diverging" },
	{ key: "net_worth_mean", label: "Net Worth", colorMode: "sequential" },
	{ key: "buyPressure", label: "Buy Press", colorMode: "sequential" },
	{ key: "sellPressure", label: "Sell Press", colorMode: "sequential" },
	{ key: "holdDominance", label: "Hold Dom", colorMode: "sequential" },
	{ key: "tradeRatio", label: "Trade Ratio", colorMode: "sequential" },
	{ key: "buySellImbalance", label: "BSI", colorMode: "diverging" },
	{ key: "pnlPerTrade", label: "PnL/Trade", colorMode: "diverging" },
];

function lerpColor(a: [number, number, number], b: [number, number, number], t: number): string {
	const r = Math.round(a[0] + (b[0] - a[0]) * t);
	const g = Math.round(a[1] + (b[1] - a[1]) * t);
	const bl = Math.round(a[2] + (b[2] - a[2]) * t);
	return `rgb(${r},${g},${bl})`;
}

const RED: [number, number, number] = [239, 68, 68];
const GREEN: [number, number, number] = [34, 197, 94];
const NEUTRAL: [number, number, number] = [30, 33, 50];
const DIM: [number, number, number] = [20, 22, 35];
const BRIGHT: [number, number, number] = [99, 202, 255];

function divergingColor(normalized: number): string {
	// normalized: -1..1 → red..neutral..green
	if (normalized < 0) {
		return lerpColor(RED, NEUTRAL, normalized + 1);
	}
	return lerpColor(NEUTRAL, GREEN, normalized);
}

function sequentialColor(normalized: number): string {
	// normalized: 0..1 → dim..bright
	return lerpColor(DIM, BRIGHT, normalized);
}

function extractValue(update: TrainingUpdate, key: string): number {
	const tc = update.trade_count_mean ?? 0;
	const hc = update.hold_count_mean ?? 0;
	const total = tc + hc;
	const bc = update.buy_count_mean ?? 0;
	const sc = update.sell_count_mean ?? 0;

	switch (key) {
		case "episode_return_mean":
			return update.episode_return_mean;
		case "pnl_mean":
			return update.pnl_mean;
		case "net_worth_mean":
			return update.net_worth_mean;
		case "buyPressure":
			return total > 0 ? bc / total : 0;
		case "sellPressure":
			return total > 0 ? sc / total : 0;
		case "holdDominance":
			return update.hold_ratio_mean ?? (total > 0 ? hc / total : 0);
		case "tradeRatio":
			return update.trade_ratio_mean ?? (total > 0 ? tc / total : 0);
		case "buySellImbalance":
			return update.buy_sell_imbalance_mean ?? (tc > 0 ? (bc - sc) / tc : 0);
		case "pnlPerTrade":
			return update.pnl_per_trade_mean ?? (tc > 0 ? update.pnl_mean / tc : 0);
		default:
			return 0;
	}
}

interface StrategyGenomeProps {
	iterations: TrainingUpdate[];
}

export function StrategyGenome({ iterations }: StrategyGenomeProps) {
	const scrollRef = useRef<HTMLDivElement>(null);
	const [tooltip, setTooltip] = useState<GenomeTooltipData | null>(null);

	const { columns, channelRanges } = useMemo(() => {
		const ranges = CHANNELS.map(() => ({
			min: Number.POSITIVE_INFINITY,
			max: Number.NEGATIVE_INFINITY,
		}));
		const cols = iterations.map((it) => {
			const values = CHANNELS.map((ch, ci) => {
				const v = extractValue(it, ch.key);
				if (v < ranges[ci].min) ranges[ci].min = v;
				if (v > ranges[ci].max) ranges[ci].max = v;
				return v;
			});
			return { iteration: it.iteration, values };
		});
		return { columns: cols, channelRanges: ranges };
	}, [iterations]);

	// Auto-scroll right as new columns arrive
	useEffect(() => {
		if (scrollRef.current) {
			scrollRef.current.scrollLeft = scrollRef.current.scrollWidth;
		}
	}, [columns.length]);

	const svgWidth = columns.length * (CELL_W + GAP);
	const svgHeight = CHANNELS.length * (CELL_H + GAP);

	function getCellColor(channelIdx: number, value: number): string {
		const range = channelRanges[channelIdx];
		const ch = CHANNELS[channelIdx];
		if (range.min === range.max)
			return CHANNELS[channelIdx].colorMode === "diverging"
				? divergingColor(0)
				: sequentialColor(0.5);

		if (ch.colorMode === "diverging") {
			const absMax = Math.max(Math.abs(range.min), Math.abs(range.max));
			const normalized = absMax === 0 ? 0 : value / absMax;
			return divergingColor(Math.max(-1, Math.min(1, normalized)));
		}
		const normalized = (value - range.min) / (range.max - range.min);
		return sequentialColor(Math.max(0, Math.min(1, normalized)));
	}

	function handleMouseMove(e: React.MouseEvent<SVGSVGElement>) {
		const rect = e.currentTarget.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const y = e.clientY - rect.top;
		const colIdx = Math.floor(x / (CELL_W + GAP));
		if (colIdx < 0 || colIdx >= columns.length) {
			setTooltip(null);
			return;
		}
		const col = columns[colIdx];
		setTooltip({
			iteration: col.iteration,
			x: e.clientX - (scrollRef.current?.getBoundingClientRect().left ?? 0) + 12,
			y: y - 10,
			values: CHANNELS.map((ch, ci) => ({
				label: ch.label,
				value: col.values[ci],
				color: getCellColor(ci, col.values[ci]),
			})),
		});
	}

	if (iterations.length === 0) {
		return (
			<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
				No training data yet. Start a training run to see the Strategy Genome.
			</div>
		);
	}

	return (
		<div className="relative flex">
			{/* Fixed labels */}
			<div className="flex-shrink-0" style={{ width: LABEL_W }}>
				{CHANNELS.map((ch) => (
					<div
						key={ch.key}
						className="flex items-center text-[10px] text-[var(--text-secondary)]"
						style={{ height: CELL_H + GAP }}
					>
						{ch.label}
					</div>
				))}
			</div>
			{/* Scrollable genome strip */}
			<div
				ref={scrollRef}
				className="flex-1 overflow-x-auto overflow-y-hidden"
				style={{ scrollBehavior: "smooth" }}
			>
				<svg
					width={Math.max(svgWidth, 1)}
					height={svgHeight}
					className="block"
					role="img"
					aria-label="Strategy genome heatmap showing metric channels across training iterations"
					onMouseMove={handleMouseMove}
					onMouseLeave={() => setTooltip(null)}
				>
					{columns.map((col, colIdx) =>
						col.values.map((val, ri) => (
							<rect
								key={`${col.iteration}-${CHANNELS[ri].key}`}
								x={colIdx * (CELL_W + GAP)}
								y={ri * (CELL_H + GAP)}
								width={CELL_W}
								height={CELL_H}
								fill={getCellColor(ri, val)}
								rx={1}
							/>
						)),
					)}
				</svg>
			</div>
			{/* Tooltip */}
			{tooltip && (
				<div
					className="pointer-events-none absolute z-20 rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg"
					style={{ left: tooltip.x, top: tooltip.y }}
				>
					<p className="mb-1 font-medium text-[var(--text-primary)]">
						Iteration {tooltip.iteration}
					</p>
					{tooltip.values.map((v) => (
						<div key={v.label} className="flex items-center gap-2">
							<span
								className="inline-block h-2 w-2 rounded-sm"
								style={{ backgroundColor: v.color }}
							/>
							<span className="text-[var(--text-secondary)]">{v.label}:</span>
							<span className="font-mono text-[var(--text-primary)]">{v.value.toFixed(3)}</span>
						</div>
					))}
				</div>
			)}
		</div>
	);
}
