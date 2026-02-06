"use client";

import type { IterationRecord } from "@/lib/types";
import { Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface MetricsLineChartProps {
	data: IterationRecord[];
	metricKeys: string[];
}

interface MetricDataPoint {
	iteration: number;
	[key: string]: number;
}

interface TooltipPayloadEntry {
	name: string;
	value: number;
	color: string;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: TooltipPayloadEntry[];
	label?: number;
}

const LINE_COLORS = [
	"#3b82f6",
	"#22c55e",
	"#ef4444",
	"#8b5cf6",
	"#f59e0b",
	"#ec4899",
	"#06b6d4",
	"#f97316",
];

function ChartTooltip({ active, payload, label }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;

	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
			<p className="mb-1 font-medium text-[var(--text-primary)]">Iteration {label}</p>
			{payload.map((entry) => (
				<p key={entry.name} style={{ color: entry.color }}>
					{entry.name}: {entry.value.toFixed(4)}
				</p>
			))}
		</div>
	);
}

export function MetricsLineChart({ data, metricKeys }: MetricsLineChartProps) {
	if (data.length === 0 || metricKeys.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-[var(--text-secondary)]">
				No metrics data available
			</div>
		);
	}

	const chartData: MetricDataPoint[] = data.map((record) => {
		const point: MetricDataPoint = { iteration: record.iteration };
		for (const key of metricKeys) {
			point[key] = record.metrics[key] ?? 0;
		}
		return point;
	});

	return (
		<ResponsiveContainer width="100%" height="100%">
			<LineChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
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
				<Legend wrapperStyle={{ color: "#8b8fa3", fontSize: 11 }} />
				{metricKeys.map((key, idx) => (
					<Line
						key={key}
						type="monotone"
						dataKey={key}
						stroke={LINE_COLORS[idx % LINE_COLORS.length]}
						strokeWidth={2}
						dot={false}
						activeDot={{ r: 4, fill: LINE_COLORS[idx % LINE_COLORS.length] }}
					/>
				))}
			</LineChart>
		</ResponsiveContainer>
	);
}
