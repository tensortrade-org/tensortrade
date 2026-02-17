"use client";

import { useMemo } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

interface ActionDistributionProps {
	buyCount: number;
	sellCount: number;
	holdCount: number;
}

interface SliceData {
	name: string;
	value: number;
	color: string;
}

interface TooltipPayloadEntry {
	name: string;
	value: number;
	payload: SliceData;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: TooltipPayloadEntry[];
}

function DistTooltip({ active, payload }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;
	const entry = payload[0];
	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
			<p style={{ color: entry.payload.color }} className="font-medium">
				{entry.name}: {entry.value}
			</p>
		</div>
	);
}

export function ActionDistribution({ buyCount, sellCount, holdCount }: ActionDistributionProps) {
	const data: SliceData[] = useMemo(
		() => [
			{ name: "Buy", value: buyCount, color: "#22c55e" },
			{ name: "Sell", value: sellCount, color: "#ef4444" },
			{ name: "Hold", value: holdCount, color: "#3b82f6" },
		],
		[buyCount, sellCount, holdCount],
	);

	const total = buyCount + sellCount + holdCount;

	if (total === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-sm text-[var(--text-secondary)]">
				No actions yet
			</div>
		);
	}

	return (
		<div className="flex flex-col items-center">
			<ResponsiveContainer width="100%" height={160}>
				<PieChart>
					<Pie
						data={data}
						cx="50%"
						cy="50%"
						innerRadius={40}
						outerRadius={65}
						paddingAngle={2}
						dataKey="value"
					>
						{data.map((entry) => (
							<Cell key={entry.name} fill={entry.color} stroke="transparent" />
						))}
					</Pie>
					<Tooltip content={<DistTooltip />} />
				</PieChart>
			</ResponsiveContainer>
			<div className="flex gap-4 text-xs">
				{data.map((d) => (
					<span key={d.name} className="flex items-center gap-1">
						<span
							className="inline-block h-2 w-2 rounded-full"
							style={{ backgroundColor: d.color }}
						/>
						<span className="text-[var(--text-secondary)]">
							{d.name} {total > 0 ? `${((d.value / total) * 100).toFixed(0)}%` : "0%"}
						</span>
					</span>
				))}
			</div>
		</div>
	);
}
