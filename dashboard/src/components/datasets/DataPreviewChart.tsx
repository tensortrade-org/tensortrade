"use client";

import type { OHLCVSample } from "@/lib/types";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface DataPreviewChartProps {
	data: OHLCVSample[];
}

interface TooltipPayloadEntry {
	value: number;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: TooltipPayloadEntry[];
	label?: string;
}

function ChartTooltip({ active, payload, label }: CustomTooltipProps) {
	if (!active || !payload?.length) return null;
	return (
		<div className="rounded border border-[var(--border-color)] bg-[var(--bg-card)] px-3 py-2 text-xs shadow-lg">
			<p className="text-[var(--text-secondary)]">{label}</p>
			<p className="font-mono text-[var(--accent-blue)]">
				${payload[0].value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
			</p>
		</div>
	);
}

export function DataPreviewChart({ data }: DataPreviewChartProps) {
	if (data.length === 0) {
		return (
			<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
				No preview data available
			</div>
		);
	}

	return (
		<div className="h-full w-full">
			<h4 className="mb-2 text-sm font-medium text-[var(--text-secondary)]">Price Preview</h4>
			<ResponsiveContainer width="100%" height="90%">
				<LineChart data={data}>
					<XAxis
						dataKey="date"
						tick={{ fontSize: 10, fill: "var(--text-secondary)" }}
						tickLine={false}
						axisLine={{ stroke: "var(--border-color)" }}
					/>
					<YAxis
						domain={["auto", "auto"]}
						tick={{ fontSize: 10, fill: "var(--text-secondary)" }}
						tickLine={false}
						axisLine={{ stroke: "var(--border-color)" }}
						tickFormatter={(v: number) => `$${v.toLocaleString()}`}
					/>
					<Tooltip content={<ChartTooltip />} />
					<Line
						type="monotone"
						dataKey="close"
						stroke="var(--accent-blue)"
						strokeWidth={1.5}
						dot={false}
					/>
				</LineChart>
			</ResponsiveContainer>
		</div>
	);
}
