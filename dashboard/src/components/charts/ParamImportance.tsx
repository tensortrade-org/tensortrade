"use client";

import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface ParamImportanceProps {
	importance: Record<string, number>;
}

interface ImportanceDataPoint {
	name: string;
	value: number;
}

interface TooltipPayloadEntry {
	value: number;
	payload: ImportanceDataPoint;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: TooltipPayloadEntry[];
}

function ChartTooltip({ active, payload }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;

	const entry = payload[0];

	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
			<p className="font-medium text-[var(--text-primary)]">{entry.payload.name}</p>
			<p className="text-[var(--accent-blue)]">Importance: {(entry.value * 100).toFixed(1)}%</p>
		</div>
	);
}

export function ParamImportance({ importance }: ParamImportanceProps) {
	const entries = Object.entries(importance);

	if (entries.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-[var(--text-secondary)]">
				No parameter importance data available
			</div>
		);
	}

	const chartData: ImportanceDataPoint[] = entries
		.map(([name, value]) => ({ name, value }))
		.sort((a, b) => b.value - a.value);

	const barHeight = 32;
	const minHeight = chartData.length * barHeight + 40;

	return (
		<ResponsiveContainer width="100%" height="100%" minHeight={minHeight}>
			<BarChart
				data={chartData}
				layout="vertical"
				margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
			>
				<XAxis
					type="number"
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
					domain={[0, "auto"]}
				/>
				<YAxis
					type="category"
					dataKey="name"
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
					width={120}
				/>
				<Tooltip content={<ChartTooltip />} />
				<Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
					{chartData.map((entry) => (
						<Cell key={entry.name} fill="#3b82f6" />
					))}
				</Bar>
			</BarChart>
		</ResponsiveContainer>
	);
}
