"use client";

import type { OptunaTrialRecord } from "@/lib/types";
import {
	ComposedChart,
	Legend,
	Line,
	ResponsiveContainer,
	Scatter,
	Tooltip,
	XAxis,
	YAxis,
	ZAxis,
} from "recharts";

interface OptimizationHistoryProps {
	trials: OptunaTrialRecord[];
}

interface ScatterDataPoint {
	trial_number: number;
	value: number;
	state: "complete" | "pruned" | "fail";
	fill: string;
}

interface BestSoFarPoint {
	trial_number: number;
	bestValue: number;
}

interface TooltipPayloadEntry {
	name: string;
	value: number;
	payload: ScatterDataPoint | BestSoFarPoint;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: TooltipPayloadEntry[];
}

const STATE_COLORS: Record<string, string> = {
	complete: "#22c55e",
	pruned: "#f59e0b",
	fail: "#ef4444",
};

function ChartTooltip({ active, payload }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;

	const entry = payload[0];
	const data = entry.payload;

	if ("state" in data) {
		return (
			<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
				<p className="text-[var(--text-secondary)]">Trial {data.trial_number}</p>
				<p className="font-medium text-[var(--text-primary)]">Value: {data.value.toFixed(4)}</p>
				<p style={{ color: STATE_COLORS[data.state] }}>State: {data.state}</p>
			</div>
		);
	}

	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
			<p className="text-[var(--text-secondary)]">Trial {data.trial_number}</p>
			<p className="font-medium text-[var(--accent-blue)]">
				Best so far: {data.bestValue.toFixed(4)}
			</p>
		</div>
	);
}

export function OptimizationHistory({ trials }: OptimizationHistoryProps) {
	if (trials.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-[var(--text-secondary)]">
				No optimization trial data available
			</div>
		);
	}

	const trialsWithValues = trials.filter(
		(t): t is OptunaTrialRecord & { value: number } => t.value !== null,
	);

	if (trialsWithValues.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-[var(--text-secondary)]">
				No completed trials with values
			</div>
		);
	}

	const scatterData: ScatterDataPoint[] = trialsWithValues.map((t) => ({
		trial_number: t.trial_number,
		value: t.value,
		state: t.state,
		fill: STATE_COLORS[t.state],
	}));

	const sortedTrials = [...trialsWithValues].sort((a, b) => a.trial_number - b.trial_number);

	const bestSoFar: BestSoFarPoint[] = [];
	let currentBest = Number.NEGATIVE_INFINITY;

	for (const trial of sortedTrials) {
		if (trial.value > currentBest) {
			currentBest = trial.value;
		}
		bestSoFar.push({
			trial_number: trial.trial_number,
			bestValue: currentBest,
		});
	}

	const combinedData = scatterData.map((point) => {
		const bestPoint = bestSoFar.find((b) => b.trial_number === point.trial_number);
		return {
			...point,
			bestValue: bestPoint?.bestValue,
		};
	});

	return (
		<ResponsiveContainer width="100%" height="100%">
			<ComposedChart data={combinedData} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
				<XAxis
					dataKey="trial_number"
					name="Trial"
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
				/>
				<YAxis
					dataKey="value"
					name="Value"
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
				/>
				<ZAxis range={[40, 40]} />
				<Tooltip content={<ChartTooltip />} />
				<Legend wrapperStyle={{ color: "#8b8fa3", fontSize: 11 }} />
				<Scatter name="Trial Value" dataKey="value" fill="#22c55e" shape="circle" />
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
