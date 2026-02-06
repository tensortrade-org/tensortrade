"use client";

import type { StepUpdate } from "@/lib/types";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface PortfolioChartProps {
	steps: StepUpdate[];
}

interface PortfolioDataPoint {
	step: number;
	net_worth: number;
}

interface TooltipPayloadEntry {
	value: number;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: TooltipPayloadEntry[];
	label?: number;
}

function ChartTooltip({ active, payload, label }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;

	const value = payload[0].value;

	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
			<p className="text-[var(--text-secondary)]">Step {label}</p>
			<p className="font-medium text-[var(--text-primary)]">Net Worth: ${value.toFixed(2)}</p>
		</div>
	);
}

export function PortfolioChart({ steps }: PortfolioChartProps) {
	if (steps.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-[var(--text-secondary)]">
				No portfolio data available
			</div>
		);
	}

	const initialNetWorth = steps[0].net_worth;

	const chartData: PortfolioDataPoint[] = steps.map((s) => ({
		step: s.step,
		net_worth: s.net_worth,
	}));

	const isAboveInitial = steps[steps.length - 1].net_worth >= initialNetWorth;
	const gradientColor = isAboveInitial ? "#22c55e" : "#ef4444";
	const strokeColor = isAboveInitial ? "#22c55e" : "#ef4444";

	return (
		<ResponsiveContainer width="100%" height="100%">
			<AreaChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
				<defs>
					<linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
						<stop offset="5%" stopColor={gradientColor} stopOpacity={0.3} />
						<stop offset="95%" stopColor={gradientColor} stopOpacity={0.02} />
					</linearGradient>
				</defs>
				<XAxis
					dataKey="step"
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
					domain={["auto", "auto"]}
				/>
				<Tooltip content={<ChartTooltip />} />
				<Area
					type="monotone"
					dataKey="net_worth"
					stroke={strokeColor}
					strokeWidth={2}
					fill="url(#portfolioGradient)"
				/>
			</AreaChart>
		</ResponsiveContainer>
	);
}
