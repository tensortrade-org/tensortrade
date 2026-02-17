"use client";

import type { TrainingUpdate } from "@/lib/types";
import {
	Bar,
	BarChart,
	Cell,
	ReferenceLine,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";

interface PnlChartProps {
	iterations: TrainingUpdate[];
}

interface PnlDataPoint {
	iteration: number;
	pnl_mean: number;
}

interface TooltipPayloadEntry {
	value: number;
	payload: PnlDataPoint;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: TooltipPayloadEntry[];
	label?: number;
}

function ChartTooltip({ active, payload }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;

	const entry = payload[0];
	const color = entry.value >= 0 ? "#22c55e" : "#ef4444";

	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
			<p className="text-[var(--text-secondary)]">Iteration {entry.payload.iteration}</p>
			<p className="font-medium" style={{ color }}>
				PnL: {entry.value >= 0 ? "+" : ""}
				{entry.value.toFixed(2)}
			</p>
		</div>
	);
}

export function PnlChart({ iterations }: PnlChartProps) {
	if (iterations.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-[var(--text-secondary)]">
				No PnL data available
			</div>
		);
	}

	const chartData: PnlDataPoint[] = iterations.map((iter) => ({
		iteration: iter.iteration,
		pnl_mean: iter.pnl_mean,
	}));

	return (
		<ResponsiveContainer width="100%" height="100%">
			<BarChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
				<XAxis
					dataKey="iteration"
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
				/>
				<YAxis
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
				/>
				<Tooltip content={<ChartTooltip />} />
				<ReferenceLine y={0} stroke="#8b8fa3" strokeDasharray="3 3" />
				<Bar dataKey="pnl_mean" radius={[2, 2, 0, 0]}>
					{chartData.map((entry) => (
						<Cell
							key={`cell-${entry.iteration}`}
							fill={entry.pnl_mean >= 0 ? "#22c55e" : "#ef4444"}
						/>
					))}
				</Bar>
			</BarChart>
		</ResponsiveContainer>
	);
}
