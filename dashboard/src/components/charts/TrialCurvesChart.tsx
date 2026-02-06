"use client";

import type { TrialCurveData } from "@/lib/types";
import { useCallback, useMemo } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface TrialCurvesChartProps {
	trials: TrialCurveData[];
	metricKey: string;
	highlightedTrial: number | null;
	onTrialHover: (trialNumber: number | null) => void;
	onTrialClick: (trialNumber: number) => void;
}

interface CurveDataPoint {
	iteration: number;
	[key: string]: number | undefined;
}

interface TrialMeta {
	trialNumber: number;
	state: string;
	color: string;
}

const STATE_COLORS: Record<string, string> = {
	complete: "#22c55e",
	pruned: "#f59e0b",
	fail: "#6b7280",
};

interface TooltipPayloadEntry {
	dataKey: string;
	value: number;
	color: string;
}

interface CustomTooltipProps {
	active?: boolean;
	label?: number;
	payload?: TooltipPayloadEntry[];
}

function CurvesTooltip({ active, label, payload }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;

	const visibleEntries = payload.filter((e) => e.value !== undefined).slice(0, 5);

	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
			<p className="mb-1 text-[var(--text-secondary)]">Iteration {label}</p>
			{visibleEntries.map((entry) => (
				<p key={entry.dataKey} style={{ color: entry.color }}>
					{entry.dataKey}: {entry.value.toFixed(2)}
				</p>
			))}
		</div>
	);
}

export function TrialCurvesChart({
	trials,
	metricKey,
	highlightedTrial,
	onTrialHover,
	onTrialClick,
}: TrialCurvesChartProps) {
	const trialsWithData = useMemo(() => trials.filter((t) => t.iterations.length > 0), [trials]);

	const trialMetas: TrialMeta[] = useMemo(
		() =>
			trialsWithData.map((t) => ({
				trialNumber: t.trial_number,
				state: t.state,
				color: STATE_COLORS[t.state] ?? STATE_COLORS.fail,
			})),
		[trialsWithData],
	);

	const chartData: CurveDataPoint[] = useMemo(() => {
		if (trialsWithData.length === 0) return [];

		const maxIter = Math.max(
			...trialsWithData.flatMap((t) => t.iterations.map((i) => i.iteration)),
		);
		const data: CurveDataPoint[] = [];

		for (let iter = 1; iter <= maxIter; iter++) {
			const point: CurveDataPoint = { iteration: iter };
			for (const trial of trialsWithData) {
				const iterData = trial.iterations.find((i) => i.iteration === iter);
				if (iterData) {
					point[`trial_${trial.trial_number}`] = iterData.metrics[metricKey];
				}
			}
			data.push(point);
		}
		return data;
	}, [trialsWithData, metricKey]);

	const handleMouseMove = useCallback(
		(e: { activePayload?: TooltipPayloadEntry[] }) => {
			if (!e.activePayload || e.activePayload.length === 0) {
				onTrialHover(null);
				return;
			}
			// Find the first defined entry
			const firstDefined = e.activePayload.find((p) => p.value !== undefined);
			if (firstDefined) {
				const match = firstDefined.dataKey.match(/trial_(\d+)/);
				if (match) {
					onTrialHover(Number(match[1]));
				}
			}
		},
		[onTrialHover],
	);

	if (trialsWithData.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-sm text-[var(--text-secondary)]">
				No per-trial iteration data available. Run train_optuna.py with --dashboard to populate.
			</div>
		);
	}

	return (
		<ResponsiveContainer width="100%" height="100%">
			<LineChart
				data={chartData}
				margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
				onMouseMove={handleMouseMove}
			>
				<XAxis
					dataKey="iteration"
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
					label={{
						value: "Iteration",
						position: "insideBottom",
						offset: -2,
						fill: "#8b8fa3",
						fontSize: 11,
					}}
				/>
				<YAxis
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
					label={{
						value: metricKey,
						angle: -90,
						position: "insideLeft",
						fill: "#8b8fa3",
						fontSize: 11,
					}}
				/>
				<Tooltip content={<CurvesTooltip />} />
				{trialMetas.map((meta) => {
					const isHighlighted = highlightedTrial === null || highlightedTrial === meta.trialNumber;
					const isDashed = meta.state === "fail";
					return (
						<Line
							key={meta.trialNumber}
							type="monotone"
							dataKey={`trial_${meta.trialNumber}`}
							stroke={meta.color}
							strokeWidth={highlightedTrial === meta.trialNumber ? 2.5 : 1.5}
							strokeOpacity={isHighlighted ? 1 : 0.15}
							strokeDasharray={isDashed ? "4 3" : undefined}
							dot={false}
							activeDot={false}
							connectNulls={false}
							onClick={() => onTrialClick(meta.trialNumber)}
						/>
					);
				})}
			</LineChart>
		</ResponsiveContainer>
	);
}
