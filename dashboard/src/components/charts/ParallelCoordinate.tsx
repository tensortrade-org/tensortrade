"use client";

import type { OptunaTrialRecord } from "@/lib/types";
import { useEffect, useMemo, useRef, useState } from "react";

interface ParallelCoordinateProps {
	trials: OptunaTrialRecord[];
	paramKeys: string[];
}

interface AxisBounds {
	min: number;
	max: number;
}

interface TrialLine {
	trialNumber: number;
	values: (number | null)[];
	objectiveValue: number | null;
	isBest: boolean;
}

interface TooltipState {
	visible: boolean;
	x: number;
	y: number;
	trial: TrialLine | null;
	paramKeys: string[];
}

const PADDING = { top: 40, right: 40, bottom: 30, left: 40 };

function valueToColor(normalized: number): string {
	const r = Math.round(239 * (1 - normalized) + 34 * normalized);
	const g = Math.round(68 * (1 - normalized) + 197 * normalized);
	const b = Math.round(68 * (1 - normalized) + 94 * normalized);
	return `rgb(${r}, ${g}, ${b})`;
}

function normalizeValue(value: number, bounds: AxisBounds): number {
	if (bounds.max === bounds.min) return 0.5;
	return (value - bounds.min) / (bounds.max - bounds.min);
}

export function ParallelCoordinate({ trials, paramKeys }: ParallelCoordinateProps) {
	const svgRef = useRef<SVGSVGElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const [dimensions, setDimensions] = useState({ width: 600, height: 400 });
	const [tooltip, setTooltip] = useState<TooltipState>({
		visible: false,
		x: 0,
		y: 0,
		trial: null,
		paramKeys: [],
	});

	useEffect(() => {
		if (!containerRef.current) return;

		const updateDimensions = () => {
			if (containerRef.current) {
				setDimensions({
					width: containerRef.current.clientWidth,
					height: containerRef.current.clientHeight,
				});
			}
		};

		updateDimensions();

		const resizeObserver = new ResizeObserver(updateDimensions);
		resizeObserver.observe(containerRef.current);

		return () => resizeObserver.disconnect();
	}, []);

	const { axisBounds, trialLines, axisXPositions } = useMemo(() => {
		if (trials.length === 0 || paramKeys.length === 0) {
			return { axisBounds: [], trialLines: [], axisXPositions: [] };
		}

		const plotWidth = dimensions.width - PADDING.left - PADDING.right;
		const axisSpacing = paramKeys.length > 1 ? plotWidth / (paramKeys.length - 1) : 0;
		const positions = paramKeys.map((_, i) => PADDING.left + i * axisSpacing);

		const bounds: AxisBounds[] = paramKeys.map((key) => {
			let min = Number.POSITIVE_INFINITY;
			let max = Number.NEGATIVE_INFINITY;
			for (const trial of trials) {
				const val = trial.params[key];
				if (typeof val === "number" && Number.isFinite(val)) {
					if (val < min) min = val;
					if (val > max) max = val;
				}
			}
			if (!Number.isFinite(min)) {
				min = 0;
				max = 1;
			}
			const range = max - min;
			const padding = range === 0 ? 0.5 : range * 0.05;
			return { min: min - padding, max: max + padding };
		});

		const completedTrials = trials.filter(
			(t): t is OptunaTrialRecord & { value: number } => t.value !== null,
		);
		let bestValue = Number.NEGATIVE_INFINITY;
		let bestTrialNumber = -1;
		for (const t of completedTrials) {
			if (t.value > bestValue) {
				bestValue = t.value;
				bestTrialNumber = t.trial_number;
			}
		}

		let minObj = Number.POSITIVE_INFINITY;
		let maxObj = Number.NEGATIVE_INFINITY;
		for (const t of completedTrials) {
			if (t.value < minObj) minObj = t.value;
			if (t.value > maxObj) maxObj = t.value;
		}

		const lines: TrialLine[] = trials.map((trial) => {
			const values = paramKeys.map((key) => {
				const val = trial.params[key];
				return typeof val === "number" && Number.isFinite(val) ? val : null;
			});

			let normalizedObj: number | null = null;
			if (trial.value !== null && Number.isFinite(minObj) && Number.isFinite(maxObj)) {
				normalizedObj = maxObj === minObj ? 0.5 : (trial.value - minObj) / (maxObj - minObj);
			}

			return {
				trialNumber: trial.trial_number,
				values,
				objectiveValue: normalizedObj,
				isBest: trial.trial_number === bestTrialNumber,
			};
		});

		return {
			axisBounds: bounds,
			trialLines: lines,
			axisXPositions: positions,
		};
	}, [trials, paramKeys, dimensions]);

	if (trials.length === 0 || paramKeys.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-[var(--text-secondary)]">
				No trial data available
			</div>
		);
	}

	const plotHeight = dimensions.height - PADDING.top - PADDING.bottom;

	function getY(value: number, axisIdx: number): number {
		const normalized = normalizeValue(value, axisBounds[axisIdx]);
		return PADDING.top + plotHeight * (1 - normalized);
	}

	function buildPath(line: TrialLine): string | null {
		const segments: string[] = [];
		let started = false;

		for (let i = 0; i < line.values.length; i++) {
			const val = line.values[i];
			if (val === null) continue;
			const x = axisXPositions[i];
			const y = getY(val, i);
			if (!started) {
				segments.push(`M ${x} ${y}`);
				started = true;
			} else {
				segments.push(`L ${x} ${y}`);
			}
		}

		return segments.length > 1 ? segments.join(" ") : null;
	}

	function formatAxisLabel(value: number): string {
		if (Math.abs(value) >= 1000) return value.toFixed(0);
		if (Math.abs(value) >= 1) return value.toFixed(2);
		return value.toPrecision(3);
	}

	const handleMouseMove = (event: React.MouseEvent<SVGPathElement>, line: TrialLine) => {
		const rect = svgRef.current?.getBoundingClientRect();
		if (!rect) return;
		setTooltip({
			visible: true,
			x: event.clientX - rect.left + 12,
			y: event.clientY - rect.top - 12,
			trial: line,
			paramKeys,
		});
	};

	const handleMouseLeave = () => {
		setTooltip((prev) => ({ ...prev, visible: false }));
	};

	const nonBestLines = trialLines.filter((l) => !l.isBest);
	const bestLine = trialLines.find((l) => l.isBest);

	return (
		<div ref={containerRef} className="relative h-full w-full">
			<svg
				ref={svgRef}
				width={dimensions.width}
				height={dimensions.height}
				className="select-none"
				role="img"
				aria-label="Parallel coordinate plot showing hyperparameter trial relationships"
			>
				{/* Axes */}
				{paramKeys.map((key, i) => {
					const x = axisXPositions[i];
					const bounds = axisBounds[i];
					return (
						<g key={key}>
							<line
								x1={x}
								y1={PADDING.top}
								x2={x}
								y2={PADDING.top + plotHeight}
								stroke="#2a2e45"
								strokeWidth={1}
							/>
							{/* Axis label */}
							<text
								x={x}
								y={PADDING.top - 12}
								textAnchor="middle"
								fill="#8b8fa3"
								fontSize={11}
								fontWeight={500}
							>
								{key.length > 12 ? `${key.slice(0, 11)}...` : key}
							</text>
							{/* Min label */}
							<text
								x={x}
								y={PADDING.top + plotHeight + 16}
								textAnchor="middle"
								fill="#8b8fa3"
								fontSize={9}
							>
								{formatAxisLabel(bounds.min)}
							</text>
							{/* Max label */}
							<text x={x} y={PADDING.top - 2} textAnchor="middle" fill="#8b8fa3" fontSize={9}>
								{formatAxisLabel(bounds.max)}
							</text>
						</g>
					);
				})}

				{/* Non-best trial lines */}
				{nonBestLines.map((line) => {
					const path = buildPath(line);
					if (!path) return null;
					const color =
						line.objectiveValue !== null ? valueToColor(line.objectiveValue) : "#8b8fa3";
					return (
						<path
							key={`line-${line.trialNumber}`}
							d={path}
							fill="none"
							stroke={color}
							strokeWidth={1.5}
							strokeOpacity={0.5}
							className="cursor-pointer transition-opacity hover:stroke-opacity-100"
							onMouseMove={(e) => handleMouseMove(e, line)}
							onMouseLeave={handleMouseLeave}
						/>
					);
				})}

				{/* Best trial line (rendered on top) */}
				{bestLine &&
					(() => {
						const path = buildPath(bestLine);
						if (!path) return null;
						return (
							<path
								d={path}
								fill="none"
								stroke="#22c55e"
								strokeWidth={3}
								strokeOpacity={1}
								className="cursor-pointer"
								onMouseMove={(e) => handleMouseMove(e, bestLine)}
								onMouseLeave={handleMouseLeave}
							/>
						);
					})()}

				{/* Legend */}
				<g>
					<rect
						x={dimensions.width - 150}
						y={4}
						width={140}
						height={28}
						rx={4}
						fill="#1a1d2e"
						fillOpacity={0.9}
						stroke="#2a2e45"
					/>
					<line
						x1={dimensions.width - 140}
						y1={18}
						x2={dimensions.width - 125}
						y2={18}
						stroke="#22c55e"
						strokeWidth={3}
					/>
					<text x={dimensions.width - 120} y={22} fill="#8b8fa3" fontSize={10}>
						Best Trial
					</text>
					{/* Gradient indicator */}
					<defs>
						<linearGradient id="pcGradient">
							<stop offset="0%" stopColor="rgb(239,68,68)" />
							<stop offset="100%" stopColor="rgb(34,197,94)" />
						</linearGradient>
					</defs>
					<line
						x1={dimensions.width - 62}
						y1={18}
						x2={dimensions.width - 20}
						y2={18}
						stroke="url(#pcGradient)"
						strokeWidth={3}
					/>
				</g>
			</svg>

			{/* Tooltip */}
			{tooltip.visible && tooltip.trial && (
				<div
					className="pointer-events-none absolute z-10 rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg"
					style={{ left: tooltip.x, top: tooltip.y }}
				>
					<p className="mb-1 font-medium text-[var(--text-primary)]">
						Trial {tooltip.trial.trialNumber}
						{tooltip.trial.isBest && (
							<span className="ml-1 text-[var(--accent-green)]">(Best)</span>
						)}
					</p>
					{tooltip.paramKeys.map((key, i) => {
						const val = tooltip.trial?.values[i];
						return (
							<p key={key} className="text-[var(--text-secondary)]">
								{key}: {val !== null && val !== undefined ? val.toFixed(4) : "N/A"}
							</p>
						);
					})}
				</div>
			)}
		</div>
	);
}
