"use client";

import type { OrbitTooltipData } from "@/lib/cortex-types";
import type { TrainingUpdate } from "@/lib/types";
import { useEffect, useMemo, useRef, useState } from "react";

const PADDING = { top: 30, right: 30, bottom: 36, left: 50 };
const MIN_DOT_R = 2;
const MAX_DOT_R = 8;
const HOVER_THRESHOLD = 15;

function lerpColor(t: number): string {
	// purple #8b5cf6 → cyan #06b6d4
	const r = Math.round(139 + (6 - 139) * t);
	const g = Math.round(92 + (182 - 92) * t);
	const b = Math.round(246 + (212 - 246) * t);
	return `rgb(${r},${g},${b})`;
}

interface BehavioralOrbitProps {
	iterations: TrainingUpdate[];
}

export function BehavioralOrbit({ iterations }: BehavioralOrbitProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const [dims, setDims] = useState({ width: 400, height: 400 });
	const [tooltip, setTooltip] = useState<OrbitTooltipData | null>(null);

	useEffect(() => {
		if (!containerRef.current) return;
		const update = () => {
			if (containerRef.current) {
				const { clientWidth, clientHeight } = containerRef.current;
				setDims({ width: clientWidth, height: clientHeight });
			}
		};
		update();
		const ro = new ResizeObserver(update);
		ro.observe(containerRef.current);
		return () => ro.disconnect();
	}, []);

	const points = useMemo(() => {
		if (iterations.length === 0) return [];
		const tc = iterations.map((it) => it.trade_count_mean ?? 0);
		const hc = iterations.map((it) => it.hold_count_mean ?? 0);
		return iterations.map((it, i) => {
			const total = tc[i] + hc[i];
			const tradeRatio = it.trade_ratio_mean ?? (total > 0 ? tc[i] / total : 0);
			const pnlPerTrade = it.pnl_per_trade_mean ?? (tc[i] > 0 ? it.pnl_mean / tc[i] : 0);
			const age = iterations.length > 1 ? i / (iterations.length - 1) : 1;
			const absReturn = Math.abs(it.episode_return_mean);
			return { iteration: it.iteration, x: tradeRatio, y: pnlPerTrade, age, absReturn };
		});
	}, [iterations]);

	const { xMin, xMax, yMin, yMax, maxAbsReturn } = useMemo(() => {
		if (points.length === 0) return { xMin: 0, xMax: 1, yMin: -1, yMax: 1, maxAbsReturn: 1 };
		let xMn = Number.POSITIVE_INFINITY;
		let xMx = Number.NEGATIVE_INFINITY;
		let yMn = Number.POSITIVE_INFINITY;
		let yMx = Number.NEGATIVE_INFINITY;
		let mAr = 0;
		for (const p of points) {
			if (p.x < xMn) xMn = p.x;
			if (p.x > xMx) xMx = p.x;
			if (p.y < yMn) yMn = p.y;
			if (p.y > yMx) yMx = p.y;
			if (p.absReturn > mAr) mAr = p.absReturn;
		}
		const yPad = (yMx - yMn) * 0.1 || 1;
		const xPad = (xMx - xMn) * 0.1 || 0.05;
		return {
			xMin: Math.max(0, xMn - xPad),
			xMax: Math.min(1, xMx + xPad),
			yMin: yMn - yPad,
			yMax: yMx + yPad,
			maxAbsReturn: mAr || 1,
		};
	}, [points]);

	const plotW = dims.width - PADDING.left - PADDING.right;
	const plotH = dims.height - PADDING.top - PADDING.bottom;

	function scaleX(v: number): number {
		return PADDING.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
	}
	function scaleY(v: number): number {
		return PADDING.top + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;
	}

	function handleMouseMove(e: React.MouseEvent<SVGSVGElement>) {
		const rect = e.currentTarget.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const my = e.clientY - rect.top;
		let closest: OrbitTooltipData | null = null;
		let minDist = HOVER_THRESHOLD;
		for (const p of points) {
			const px = scaleX(p.x);
			const py = scaleY(p.y);
			const d = Math.sqrt((mx - px) ** 2 + (my - py) ** 2);
			if (d < minDist) {
				minDist = d;
				closest = {
					iteration: p.iteration,
					tradeRatio: p.x,
					pnlPerTrade: p.y,
					episodeReturn: p.absReturn * (p.y >= 0 ? 1 : -1),
					x: px + 12,
					y: py - 12,
				};
			}
		}
		setTooltip(closest);
	}

	const trailPath = useMemo(() => {
		if (points.length < 2) return "";
		return points.map((p, i) => `${i === 0 ? "M" : "L"} ${scaleX(p.x)} ${scaleY(p.y)}`).join(" ");
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [points, xMin, xMax, yMin, yMax, plotW, plotH]);

	// Quadrant midpoints
	const midX = scaleX((xMin + xMax) / 2);
	const midY = scaleY((yMin + yMax) / 2);

	if (iterations.length === 0) {
		return (
			<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
				Waiting for training data...
			</div>
		);
	}

	const gradientId = "orbit-trail-grad";

	return (
		<div ref={containerRef} className="relative h-full w-full">
			<svg
				width={dims.width}
				height={dims.height}
				className="select-none"
				role="img"
				aria-label="Behavioral orbit trajectory plot showing agent strategy evolution"
				onMouseMove={handleMouseMove}
				onMouseLeave={() => setTooltip(null)}
			>
				<defs>
					<linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
						<stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.15} />
						<stop offset="100%" stopColor="#06b6d4" stopOpacity={0.6} />
					</linearGradient>
				</defs>

				{/* Quadrant backgrounds */}
				<rect
					x={PADDING.left}
					y={PADDING.top}
					width={midX - PADDING.left}
					height={midY - PADDING.top}
					fill="#22c55e"
					fillOpacity={0.03}
				/>
				<rect
					x={midX}
					y={PADDING.top}
					width={PADDING.left + plotW - midX}
					height={midY - PADDING.top}
					fill="#22c55e"
					fillOpacity={0.06}
				/>
				<rect
					x={PADDING.left}
					y={midY}
					width={midX - PADDING.left}
					height={PADDING.top + plotH - midY}
					fill="#ef4444"
					fillOpacity={0.03}
				/>
				<rect
					x={midX}
					y={midY}
					width={PADDING.left + plotW - midX}
					height={PADDING.top + plotH - midY}
					fill="#ef4444"
					fillOpacity={0.06}
				/>

				{/* Quadrant labels */}
				<text x={PADDING.left + 4} y={PADDING.top + 14} fill="#8b8fa3" fontSize={9} opacity={0.5}>
					Passive Profitable
				</text>
				<text
					x={PADDING.left + plotW - 4}
					y={PADDING.top + 14}
					fill="#8b8fa3"
					fontSize={9}
					opacity={0.5}
					textAnchor="end"
				>
					Active Profitable
				</text>
				<text
					x={PADDING.left + 4}
					y={PADDING.top + plotH - 4}
					fill="#8b8fa3"
					fontSize={9}
					opacity={0.5}
				>
					Passive Losing
				</text>
				<text
					x={PADDING.left + plotW - 4}
					y={PADDING.top + plotH - 4}
					fill="#8b8fa3"
					fontSize={9}
					opacity={0.5}
					textAnchor="end"
				>
					Active Losing
				</text>

				{/* Axes */}
				<line
					x1={PADDING.left}
					y1={PADDING.top + plotH}
					x2={PADDING.left + plotW}
					y2={PADDING.top + plotH}
					stroke="#2a2e45"
				/>
				<line
					x1={PADDING.left}
					y1={PADDING.top}
					x2={PADDING.left}
					y2={PADDING.top + plotH}
					stroke="#2a2e45"
				/>

				{/* Axis labels */}
				<text
					x={PADDING.left + plotW / 2}
					y={dims.height - 4}
					fill="#8b8fa3"
					fontSize={10}
					textAnchor="middle"
				>
					Trade Ratio
				</text>
				<text
					x={12}
					y={PADDING.top + plotH / 2}
					fill="#8b8fa3"
					fontSize={10}
					textAnchor="middle"
					transform={`rotate(-90, 12, ${PADDING.top + plotH / 2})`}
				>
					PnL / Trade
				</text>

				{/* Trail */}
				{trailPath && (
					<path d={trailPath} fill="none" stroke={`url(#${gradientId})`} strokeWidth={1.5} />
				)}

				{/* Points */}
				{points.map((p, i) => {
					const cx = scaleX(p.x);
					const cy = scaleY(p.y);
					const r = MIN_DOT_R + (MAX_DOT_R - MIN_DOT_R) * Math.min(1, p.absReturn / maxAbsReturn);
					const color = lerpColor(p.age);
					const isLatest = i === points.length - 1;
					return (
						<g key={p.iteration}>
							<circle cx={cx} cy={cy} r={r} fill={color} fillOpacity={0.3 + 0.7 * p.age} />
							{isLatest && (
								<circle cx={cx} cy={cy} r={r + 4} fill="none" stroke={color} strokeWidth={1.5}>
									<animate
										attributeName="opacity"
										values="1;0.2;1"
										dur="2s"
										repeatCount="indefinite"
									/>
								</circle>
							)}
						</g>
					);
				})}
			</svg>

			{tooltip && (
				<div
					className="pointer-events-none absolute z-20 rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg"
					style={{ left: tooltip.x, top: tooltip.y }}
				>
					<p className="font-medium text-[var(--text-primary)]">Iteration {tooltip.iteration}</p>
					<p className="text-[var(--text-secondary)]">
						Trade Ratio: {tooltip.tradeRatio.toFixed(3)}
					</p>
					<p className="text-[var(--text-secondary)]">
						PnL/Trade: {tooltip.pnlPerTrade.toFixed(3)}
					</p>
					<p className="text-[var(--text-secondary)]">Return: {tooltip.episodeReturn.toFixed(3)}</p>
				</div>
			)}
		</div>
	);
}
