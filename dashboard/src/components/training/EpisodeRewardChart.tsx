"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { useTrainingStore } from "@/stores/trainingStore";
import {
	CartesianGrid,
	Line,
	LineChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";

interface RewardDataPoint {
	episode: number;
	reward: number;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: ReadonlyArray<{ value: number }>;
	label?: number;
}

function RewardTooltip({ active, payload, label }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;
	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-card)] px-3 py-2 text-xs shadow-lg">
			<p className="text-[var(--text-secondary)]">Episode {label}</p>
			<p className="font-medium text-[var(--accent-blue)]">Reward: {payload[0].value.toFixed(2)}</p>
		</div>
	);
}

export function EpisodeRewardChart() {
	const episodes = useTrainingStore((s) => s.episodes);

	const data: RewardDataPoint[] = episodes.map((ep) => ({
		episode: ep.episode,
		reward: ep.reward_total,
	}));

	return (
		<Card>
			<CardHeader title="Episode Reward" />
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
							width={60}
						/>
						<Tooltip content={<RewardTooltip />} />
						<Line
							type="monotone"
							dataKey="reward"
							stroke="var(--accent-blue)"
							strokeWidth={2}
							dot={false}
							animationDuration={300}
						/>
					</LineChart>
				</ResponsiveContainer>
			)}
		</Card>
	);
}
