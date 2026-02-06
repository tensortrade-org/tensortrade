"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { useTrainingStore } from "@/stores/trainingStore";
import {
	CartesianGrid,
	Line,
	LineChart,
	ReferenceLine,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";

interface PnLDataPoint {
	episode: number;
	pnl: number;
}

interface PnLDotProps {
	cx?: number;
	cy?: number;
	payload?: PnLDataPoint;
}

function PnLDot({ cx, cy, payload }: PnLDotProps) {
	if (cx === undefined || cy === undefined || !payload) return null;
	const color = payload.pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)";
	return <circle cx={cx} cy={cy} r={3} fill={color} stroke="none" />;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: ReadonlyArray<{ value: number; payload: PnLDataPoint }>;
	label?: number;
}

function PnLTooltip({ active, payload, label }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;
	const pnl = payload[0].value;
	const color = pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)";
	const sign = pnl >= 0 ? "+" : "";
	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-card)] px-3 py-2 text-xs shadow-lg">
			<p className="text-[var(--text-secondary)]">Episode {label}</p>
			<p className="font-medium" style={{ color }}>
				PnL: {sign}${pnl.toFixed(2)}
			</p>
		</div>
	);
}

export function EpisodePnLChart() {
	const episodes = useTrainingStore((s) => s.episodes);

	const data: PnLDataPoint[] = episodes.map((ep) => ({
		episode: ep.episode,
		pnl: ep.pnl,
	}));

	return (
		<Card>
			<CardHeader title="Episode P&L" />
			{data.length === 0 ? (
				<div className="flex h-48 items-center justify-center text-sm text-[var(--text-secondary)]">
					Waiting for episode data...
				</div>
			) : (
				<ResponsiveContainer width="100%" height={240}>
					<LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
						<CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
						<XAxis
							dataKey="episode"
							tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
							stroke="var(--border-color)"
							label={{
								value: "Episode",
								position: "insideBottomRight",
								offset: -5,
								fill: "var(--text-secondary)",
								fontSize: 11,
							}}
						/>
						<YAxis
							tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
							stroke="var(--border-color)"
							width={70}
							tickFormatter={(v: number) => `$${v.toFixed(0)}`}
						/>
						<ReferenceLine y={0} stroke="var(--text-secondary)" strokeDasharray="3 3" />
						<Tooltip content={<PnLTooltip />} />
						<Line
							type="monotone"
							dataKey="pnl"
							stroke="var(--text-primary)"
							strokeWidth={2}
							dot={<PnLDot />}
							animationDuration={300}
						/>
					</LineChart>
				</ResponsiveContainer>
			)}
		</Card>
	);
}
