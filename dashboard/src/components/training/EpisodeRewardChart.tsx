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
	reward_avg: number;
}

const MAX_REWARD_POINTS = 500;

function downsampleData(data: RewardDataPoint[], maxPoints = MAX_REWARD_POINTS): RewardDataPoint[] {
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
	payload?: ReadonlyArray<{ name: string; value: number; color: string }>;
	label?: number;
}

function RewardTooltip({ active, payload, label }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;
	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-card)] px-3 py-2 text-xs shadow-lg">
			<p className="text-[var(--text-secondary)]">Episode {label}</p>
			{payload.map((entry) => (
				<p key={entry.name} className="font-medium" style={{ color: entry.color }}>
					{entry.name === "reward" ? "Reward" : "Reward MA(5)"}: {entry.value.toFixed(2)}
				</p>
			))}
		</div>
	);
}

export function EpisodeRewardChart() {
	const episodes = useTrainingStore((s) => s.episodes);

	let rolling = 0;
	const fullData: RewardDataPoint[] = episodes.map((ep, idx) => {
		rolling += ep.reward_total;
		if (idx >= 5) {
			rolling -= episodes[idx - 5].reward_total;
		}
		const windowLen = idx < 5 ? idx + 1 : 5;
		return {
			episode: ep.episode,
			reward: ep.reward_total,
			reward_avg: rolling / windowLen,
		};
	});
	const data = downsampleData(fullData);
	const enableAnimation = data.length <= 600;
	const animationDuration = enableAnimation ? 180 : 0;

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
							isAnimationActive={enableAnimation}
							animationDuration={animationDuration}
						/>
						<Line
							type="monotone"
							dataKey="reward_avg"
							stroke="var(--accent-amber)"
							strokeWidth={1.5}
							strokeDasharray="4 4"
							dot={false}
							isAnimationActive={enableAnimation}
							animationDuration={animationDuration}
						/>
					</LineChart>
				</ResponsiveContainer>
			)}
		</Card>
	);
}
