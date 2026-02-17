"use client";

import type { PulseHealthScore, PulseRing } from "@/lib/cortex-types";
import type { EpisodeMetrics, TrainingProgress, TrainingUpdate } from "@/lib/types";
import { useEffect, useMemo, useRef, useState } from "react";

const CENTER_RADIUS = 40;
const RING_WIDTH = 3;
const RING_GAP = 1;

function describeArc(
	cx: number,
	cy: number,
	radius: number,
	startAngle: number,
	endAngle: number,
): string {
	const startRad = ((startAngle - 90) * Math.PI) / 180;
	const endRad = ((endAngle - 90) * Math.PI) / 180;
	const x1 = cx + radius * Math.cos(startRad);
	const y1 = cy + radius * Math.sin(startRad);
	const x2 = cx + radius * Math.cos(endRad);
	const y2 = cy + radius * Math.sin(endRad);
	const largeArc = endAngle - startAngle > 180 ? 1 : 0;
	return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`;
}

function brightnessScale(base: [number, number, number], brightness: number): string {
	const min = 0.15;
	const t = min + (1 - min) * Math.max(0, Math.min(1, brightness));
	const r = Math.round(base[0] * t);
	const g = Math.round(base[1] * t);
	const b = Math.round(base[2] * t);
	return `rgb(${r},${g},${b})`;
}

const BUY_BASE: [number, number, number] = [34, 197, 94]; // emerald
const SELL_BASE: [number, number, number] = [244, 63, 94]; // rose
const HOLD_BASE: [number, number, number] = [56, 189, 248]; // sky

interface LearningPulseProps {
	iterations: TrainingUpdate[];
	episodes: EpisodeMetrics[];
	progress: TrainingProgress | null;
}

export function LearningPulse({ iterations, episodes, progress }: LearningPulseProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const [dims, setDims] = useState({ width: 400, height: 400 });

	useEffect(() => {
		if (!containerRef.current) return;
		const update = () => {
			if (containerRef.current) {
				setDims({
					width: containerRef.current.clientWidth,
					height: containerRef.current.clientHeight,
				});
			}
		};
		update();
		const ro = new ResizeObserver(update);
		ro.observe(containerRef.current);
		return () => ro.disconnect();
	}, []);

	const { rings, rewardRange } = useMemo(() => {
		let rMin = Number.POSITIVE_INFINITY;
		let rMax = Number.NEGATIVE_INFINITY;
		for (const it of iterations) {
			if (it.episode_return_mean < rMin) rMin = it.episode_return_mean;
			if (it.episode_return_mean > rMax) rMax = it.episode_return_mean;
		}
		if (!Number.isFinite(rMin)) {
			rMin = 0;
			rMax = 1;
		}
		const range = rMax - rMin || 1;

		const builtRings: PulseRing[] = iterations.map((it) => {
			const tc = it.trade_count_mean ?? 0;
			const hc = it.hold_count_mean ?? 0;
			const bc = it.buy_count_mean ?? 0;
			const sc = it.sell_count_mean ?? 0;
			const total = bc + sc + hc || 1;
			const buyFrac = bc / total;
			const sellFrac = sc / total;
			const holdFrac = hc / total;
			const brightness = (it.episode_return_mean - rMin) / range;

			return {
				iteration: it.iteration,
				arcs: [
					{ label: "Buy", fraction: buyFrac, color: "buy" },
					{ label: "Sell", fraction: sellFrac, color: "sell" },
					{ label: "Hold", fraction: holdFrac, color: "hold" },
				],
				brightness,
			};
		});

		return { rings: builtRings, rewardRange: { min: rMin, max: rMax } };
	}, [iterations]);

	const healthScore: PulseHealthScore = useMemo(() => {
		if (iterations.length < 2) return { score: 50, trend: "stable", trendLabel: "—" };

		const recent = iterations.slice(-10);
		const half = Math.floor(recent.length / 2);
		const firstHalf = recent.slice(0, half);
		const secondHalf = recent.slice(half);

		const avgFirst =
			firstHalf.reduce((s, it) => s + it.episode_return_mean, 0) / (firstHalf.length || 1);
		const avgSecond =
			secondHalf.reduce((s, it) => s + it.episode_return_mean, 0) / (secondHalf.length || 1);

		// Reward component (60%)
		const rewardRange_ = rewardRange.max - rewardRange.min || 1;
		const rewardNorm = ((avgSecond - rewardRange.min) / rewardRange_) * 100;

		// Action stability (40%): low variance in trade_ratio over last 10
		const trs = recent.map((it) => {
			const tc = it.trade_count_mean ?? 0;
			const hc = it.hold_count_mean ?? 0;
			const total = tc + hc;
			return it.trade_ratio_mean ?? (total > 0 ? tc / total : 0);
		});
		const trMean = trs.reduce((a, b) => a + b, 0) / trs.length;
		const trVar = trs.reduce((a, v) => a + (v - trMean) ** 2, 0) / trs.length;
		const stability = Math.max(0, 100 - trVar * 1000); // low variance → high stability

		const score = Math.round(rewardNorm * 0.6 + stability * 0.4);

		const diff = avgSecond - avgFirst;
		const threshold = rewardRange_ * 0.02;
		let trend: PulseHealthScore["trend"] = "stable";
		let trendLabel = "→";
		if (diff > threshold) {
			trend = "improving";
			trendLabel = "↑";
		} else if (diff < -threshold) {
			trend = "declining";
			trendLabel = "↓";
		}

		return { score: Math.max(0, Math.min(100, score)), trend, trendLabel };
	}, [iterations, rewardRange]);

	const cx = dims.width / 2;
	const cy = dims.height / 2;
	const maxRadius = Math.min(cx, cy) - 10;
	const maxRings = Math.floor((maxRadius - CENTER_RADIUS) / (RING_WIDTH + RING_GAP));
	const visibleRings = rings.slice(-maxRings);

	if (iterations.length === 0) {
		return (
			<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
				Waiting for training data...
			</div>
		);
	}

	const trendColor =
		healthScore.trend === "improving"
			? "var(--accent-green)"
			: healthScore.trend === "declining"
				? "var(--accent-red)"
				: "var(--text-secondary)";
	const scoreColor =
		healthScore.score >= 70
			? "var(--accent-green)"
			: healthScore.score >= 40
				? "var(--accent-amber)"
				: "var(--accent-red)";

	return (
		<div ref={containerRef} className="relative h-full w-full">
			<svg
				width={dims.width}
				height={dims.height}
				className="select-none"
				role="img"
				aria-label="Learning pulse radial diagram showing training health"
			>
				{/* Rings */}
				{visibleRings.map((ring, ri) => {
					const radius = CENTER_RADIUS + ri * (RING_WIDTH + RING_GAP);
					let startAngle = 0;
					return ring.arcs.map((arc) => {
						const sweep = arc.fraction * 360;
						if (sweep < 0.5) {
							startAngle += sweep;
							return null;
						}
						const endAngle = startAngle + sweep;
						const path = describeArc(
							cx,
							cy,
							radius,
							startAngle,
							Math.min(endAngle, startAngle + 359.9),
						);
						const base =
							arc.color === "buy" ? BUY_BASE : arc.color === "sell" ? SELL_BASE : HOLD_BASE;
						const color = brightnessScale(base, ring.brightness);
						startAngle = endAngle;
						return (
							<path
								key={`${ring.iteration}-${arc.label}`}
								d={path}
								fill="none"
								stroke={color}
								strokeWidth={RING_WIDTH}
								strokeLinecap="round"
							/>
						);
					});
				})}

				{/* Center circle background */}
				<circle
					cx={cx}
					cy={cy}
					r={CENTER_RADIUS - 2}
					fill="var(--bg-card)"
					stroke="#2a2e45"
					strokeWidth={1}
				/>

				{/* Center text: iteration count */}
				<text x={cx} y={cy - 14} fill="var(--text-secondary)" fontSize={10} textAnchor="middle">
					Iter
				</text>
				<text
					x={cx}
					y={cy + 2}
					fill="var(--text-primary)"
					fontSize={16}
					fontWeight="bold"
					textAnchor="middle"
				>
					{progress?.iteration ?? iterations[iterations.length - 1]?.iteration ?? 0}
				</text>

				{/* Health score */}
				<text
					x={cx}
					y={cy + 18}
					fill={scoreColor}
					fontSize={11}
					fontWeight="600"
					textAnchor="middle"
				>
					{healthScore.score}
				</text>
				{/* Trend arrow */}
				<text x={cx} y={cy + 30} fill={trendColor} fontSize={12} textAnchor="middle">
					{healthScore.trendLabel}
				</text>
			</svg>

			{/* Legend */}
			<div className="absolute bottom-2 left-2 flex gap-3 text-[10px] text-[var(--text-secondary)]">
				<span className="flex items-center gap-1">
					<span
						className="inline-block h-2 w-2 rounded-full"
						style={{ backgroundColor: `rgb(${BUY_BASE.join(",")})` }}
					/>
					Buy
				</span>
				<span className="flex items-center gap-1">
					<span
						className="inline-block h-2 w-2 rounded-full"
						style={{ backgroundColor: `rgb(${SELL_BASE.join(",")})` }}
					/>
					Sell
				</span>
				<span className="flex items-center gap-1">
					<span
						className="inline-block h-2 w-2 rounded-full"
						style={{ backgroundColor: `rgb(${HOLD_BASE.join(",")})` }}
					/>
					Hold
				</span>
			</div>
		</div>
	);
}
