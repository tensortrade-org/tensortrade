"use client";

import type { OptunaTrialRecord } from "@/lib/types";
import { useMemo } from "react";
import { ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";

interface ParamHeatmapProps {
	trials: OptunaTrialRecord[];
	paramX: string;
	paramY: string;
}

interface HeatmapPoint {
	x: number;
	y: number;
	value: number;
	trial_number: number;
	fill: string;
}

interface TooltipPayloadEntry {
	payload: HeatmapPoint;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: TooltipPayloadEntry[];
}

function pnlToColor(value: number, min: number, max: number): string {
	const range = max - min || 1;
	const normalized = (value - min) / range;
	// Red (bad) -> Yellow (mid) -> Green (good)
	if (normalized < 0.5) {
		const t = normalized * 2;
		const r = 239;
		const g = Math.round(68 + t * (163 - 68));
		const b = Math.round(68 + t * (15 - 68));
		return `rgb(${r},${g},${b})`;
	}
	const t = (normalized - 0.5) * 2;
	const r = Math.round(245 - t * (245 - 34));
	const g = Math.round(158 + t * (197 - 158));
	const b = Math.round(11 + t * (94 - 11));
	return `rgb(${r},${g},${b})`;
}

function HeatmapTooltip({ active, payload }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;
	const point = payload[0].payload;
	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
			<p className="text-[var(--text-secondary)]">Trial {point.trial_number}</p>
			<p className="text-[var(--text-primary)]">
				X: {point.x.toFixed(4)} | Y: {point.y.toFixed(4)}
			</p>
			<p className="font-medium" style={{ color: point.fill }}>
				P&L: ${point.value.toFixed(0)}
			</p>
		</div>
	);
}

export function ParamHeatmap({ trials, paramX, paramY }: ParamHeatmapProps) {
	const data: HeatmapPoint[] = useMemo(() => {
		const completed = trials.filter(
			(t) =>
				t.state === "complete" &&
				t.value !== null &&
				paramX in t.params &&
				paramY in t.params &&
				typeof t.params[paramX] === "number" &&
				typeof t.params[paramY] === "number",
		);

		if (completed.length === 0) return [];

		const values = completed.map((t) => t.value as number);
		const min = Math.min(...values);
		const max = Math.max(...values);

		return completed.map((t) => ({
			x: t.params[paramX] as number,
			y: t.params[paramY] as number,
			value: t.value as number,
			trial_number: t.trial_number,
			fill: pnlToColor(t.value as number, min, max),
		}));
	}, [trials, paramX, paramY]);

	if (data.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-sm text-[var(--text-secondary)]">
				No data for selected parameters
			</div>
		);
	}

	return (
		<ResponsiveContainer width="100%" height="100%">
			<ScatterChart margin={{ top: 8, right: 16, left: 8, bottom: 24 }}>
				<XAxis
					dataKey="x"
					name={paramX}
					type="number"
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
					label={{
						value: paramX,
						position: "insideBottom",
						offset: -8,
						fill: "#8b8fa3",
						fontSize: 11,
					}}
				/>
				<YAxis
					dataKey="y"
					name={paramY}
					type="number"
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
					label={{
						value: paramY,
						angle: -90,
						position: "insideLeft",
						fill: "#8b8fa3",
						fontSize: 11,
					}}
				/>
				<ZAxis range={[60, 60]} />
				<Tooltip content={<HeatmapTooltip />} />
				<Scatter data={data} shape="circle">
					{data.map((point) => (
						<circle
							key={point.trial_number}
							cx={0}
							cy={0}
							r={6}
							fill={point.fill}
							stroke="#1a1d2e"
							strokeWidth={1}
						/>
					))}
				</Scatter>
			</ScatterChart>
		</ResponsiveContainer>
	);
}
