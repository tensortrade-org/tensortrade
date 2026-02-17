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
	pnl_pos: number | null;
	pnl_neg: number | null;
	pnl_avg: number;
}

const MAX_PNL_POINTS = 500;

function downsampleData(data: PnLDataPoint[], maxPoints = MAX_PNL_POINTS): PnLDataPoint[] {
	if (data.length <= maxPoints) return data;
	const stride = Math.ceil(data.length / maxPoints);
	const sampled = data.filter((point) => point.episode % stride === 0);
	if (sampled[sampled.length - 1]?.episode !== data[data.length - 1]?.episode) {
		sampled.push(data[data.length - 1]);
	}
	return sampled.length > 0 ? sampled : data.filter((_, idx) => idx % stride === 0);
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: ReadonlyArray<{ name: string; value?: number; color: string }>;
	label?: number;
}

function PnLTooltip({ active, payload, label }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;
	const pnlValue =
		payload.find((entry) => entry.name === "pnl_pos")?.value ??
		payload.find((entry) => entry.name === "pnl_neg")?.value;
	const pnlAvgValue = payload.find((entry) => entry.name === "pnl_avg")?.value;
	const pnlColor =
		typeof pnlValue === "number" && pnlValue >= 0 ? "var(--accent-green)" : "var(--accent-red)";
	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-card)] px-3 py-2 text-xs shadow-lg">
			<p className="text-[var(--text-secondary)]">Episode {label}</p>
			{typeof pnlValue === "number" && (
				<p className="font-medium" style={{ color: pnlColor }}>
					PnL: {pnlValue >= 0 ? "+" : ""}${pnlValue.toFixed(2)}
				</p>
			)}
			{typeof pnlAvgValue === "number" && (
				<p className="font-medium" style={{ color: "var(--accent-blue)" }}>
					PnL MA(5): ${pnlAvgValue.toFixed(2)}
				</p>
			)}
		</div>
	);
}

export function EpisodePnLChart() {
	const episodes = useTrainingStore((s) => s.episodes);

	let rolling = 0;
	const fullData: PnLDataPoint[] = episodes.map((ep, idx) => {
		rolling += ep.pnl;
		if (idx >= 5) {
			rolling -= episodes[idx - 5].pnl;
		}
		const windowLen = idx < 5 ? idx + 1 : 5;
		return {
			episode: ep.episode,
			pnl_pos: ep.pnl >= 0 ? ep.pnl : null,
			pnl_neg: ep.pnl < 0 ? ep.pnl : null,
			pnl_avg: rolling / windowLen,
		};
	});
	const data = downsampleData(fullData);
	const enableAnimation = data.length <= 600;
	const animationDuration = enableAnimation ? 180 : 0;
	const showDots = data.length <= 120;

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
							dataKey="pnl_pos"
							stroke="var(--accent-green)"
							strokeWidth={2}
							dot={showDots}
							connectNulls={false}
							isAnimationActive={enableAnimation}
							animationDuration={animationDuration}
							name="pnl_pos"
						/>
						<Line
							type="monotone"
							dataKey="pnl_neg"
							stroke="var(--accent-red)"
							strokeWidth={2}
							dot={showDots}
							connectNulls={false}
							isAnimationActive={enableAnimation}
							animationDuration={animationDuration}
							name="pnl_neg"
						/>
						<Line
							type="monotone"
							dataKey="pnl_avg"
							stroke="var(--accent-blue)"
							strokeWidth={1.5}
							strokeDasharray="4 4"
							dot={false}
							isAnimationActive={enableAnimation}
							animationDuration={animationDuration}
							name="pnl_avg"
						/>
					</LineChart>
				</ResponsiveContainer>
			)}
		</Card>
	);
}
