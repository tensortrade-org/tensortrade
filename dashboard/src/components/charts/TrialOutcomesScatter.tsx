"use client";

import type { OptunaTrialRecord } from "@/lib/types";
import { useMemo } from "react";
import {
	ComposedChart,
	Line,
	ResponsiveContainer,
	Scatter,
	Tooltip,
	XAxis,
	YAxis,
	ZAxis,
} from "recharts";

interface TrialOutcomesScatterProps {
	trials: OptunaTrialRecord[];
	onTrialClick: (trialNumber: number) => void;
	highlightedTrial?: number | null;
}

interface ScatterPoint {
	trial_number: number;
	value: number;
	state: string;
	fill: string;
	size: number;
}

interface BestLine {
	trial_number: number;
	bestValue: number;
}

const STATE_COLORS: Record<string, string> = {
	complete: "#22c55e",
	pruned: "#f59e0b",
	fail: "#ef4444",
};

interface TooltipPayloadEntry {
	payload: ScatterPoint | BestLine;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: TooltipPayloadEntry[];
}

function ScatterTooltip({ active, payload }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;

	const data = payload[0].payload;

	if ("state" in data) {
		const point = data as ScatterPoint;
		return (
			<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
				<p className="text-[var(--text-secondary)]">Trial {point.trial_number}</p>
				<p className="font-medium text-[var(--text-primary)]">P&L: ${point.value.toFixed(0)}</p>
				<p style={{ color: STATE_COLORS[point.state] }}>{point.state}</p>
			</div>
		);
	}

	return null;
}

export function TrialOutcomesScatter({ trials, onTrialClick }: TrialOutcomesScatterProps) {
	const trialsWithValues = useMemo(
		() => trials.filter((t): t is OptunaTrialRecord & { value: number } => t.value !== null),
		[trials],
	);

	const scatterData: ScatterPoint[] = useMemo(
		() =>
			trialsWithValues.map((t) => ({
				trial_number: t.trial_number,
				value: t.value,
				state: t.state,
				fill: STATE_COLORS[t.state] ?? "#6b7280",
				size: t.duration_seconds ? Math.min(Math.max(t.duration_seconds / 10, 30), 200) : 60,
			})),
		[trialsWithValues],
	);

	const bestSoFar: BestLine[] = useMemo(() => {
		const sorted = [...trialsWithValues].sort((a, b) => a.trial_number - b.trial_number);
		const result: BestLine[] = [];
		let best = Number.NEGATIVE_INFINITY;
		for (const t of sorted) {
			if (t.value > best) best = t.value;
			result.push({ trial_number: t.trial_number, bestValue: best });
		}
		return result;
	}, [trialsWithValues]);

	const combinedData = useMemo(() => {
		return scatterData.map((point) => {
			const bestPoint = bestSoFar.find((b) => b.trial_number === point.trial_number);
			return { ...point, bestValue: bestPoint?.bestValue };
		});
	}, [scatterData, bestSoFar]);

	if (trialsWithValues.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-sm text-[var(--text-secondary)]">
				No completed trials with values
			</div>
		);
	}

	return (
		<ResponsiveContainer width="100%" height="100%">
			<ComposedChart
				data={combinedData}
				margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
				onClick={(e) => {
					if (e?.activePayload?.[0]?.payload?.trial_number !== undefined) {
						onTrialClick(e.activePayload[0].payload.trial_number);
					}
				}}
			>
				<XAxis
					dataKey="trial_number"
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
					label={{
						value: "Trial",
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
				/>
				<ZAxis dataKey="size" range={[30, 200]} />
				<Tooltip content={<ScatterTooltip />} />
				<Scatter name="Trial P&L" dataKey="value" fill="#22c55e" shape="circle" />
				<Line
					name="Best So Far"
					type="stepAfter"
					dataKey="bestValue"
					stroke="#3b82f6"
					strokeWidth={2}
					dot={false}
					activeDot={false}
				/>
			</ComposedChart>
		</ResponsiveContainer>
	);
}
