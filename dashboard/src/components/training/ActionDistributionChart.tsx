"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { useTrainingStore } from "@/stores/trainingStore";
import {
	Area,
	AreaChart,
	CartesianGrid,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";

interface ActionDataPoint {
	episode: number;
	buy: number;
	sell: number;
	hold: number;
}

interface CustomTooltipProps {
	active?: boolean;
	payload?: ReadonlyArray<{ name: string; value: number; color: string }>;
	label?: number;
}

function ActionTooltip({ active, payload, label }: CustomTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;
	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-card)] px-3 py-2 text-xs shadow-lg">
			<p className="mb-1 text-[var(--text-secondary)]">Episode {label}</p>
			{payload.map((entry) => (
				<p key={entry.name} className="font-medium" style={{ color: entry.color }}>
					{entry.name}: {entry.value}
				</p>
			))}
		</div>
	);
}

export function ActionDistributionChart() {
	const episodes = useTrainingStore((s) => s.episodes);

	const data: ActionDataPoint[] = episodes.map((ep) => ({
		episode: ep.episode,
		buy: ep.buy_count,
		sell: ep.sell_count,
		hold: ep.hold_count,
	}));

	return (
		<Card>
			<CardHeader title="Action Distribution" />
			{data.length === 0 ? (
				<div className="flex h-48 items-center justify-center text-sm text-[var(--text-secondary)]">
					Waiting for episode data...
				</div>
			) : (
				<>
					<div className="mb-2 flex items-center gap-4 px-1">
						<div className="flex items-center gap-1.5">
							<div className="h-2.5 w-2.5 rounded-sm bg-[var(--accent-green)]" />
							<span className="text-xs text-[var(--text-secondary)]">Buy</span>
						</div>
						<div className="flex items-center gap-1.5">
							<div className="h-2.5 w-2.5 rounded-sm bg-[var(--accent-red)]" />
							<span className="text-xs text-[var(--text-secondary)]">Sell</span>
						</div>
						<div className="flex items-center gap-1.5">
							<div className="h-2.5 w-2.5 rounded-sm bg-[var(--accent-blue)]" />
							<span className="text-xs text-[var(--text-secondary)]">Hold</span>
						</div>
					</div>
					<ResponsiveContainer width="100%" height={240}>
						<AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
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
								width={50}
							/>
							<Tooltip content={<ActionTooltip />} />
							<Area
								type="monotone"
								dataKey="buy"
								stackId="actions"
								stroke="var(--accent-green)"
								fill="var(--accent-green)"
								fillOpacity={0.6}
								animationDuration={300}
							/>
							<Area
								type="monotone"
								dataKey="sell"
								stackId="actions"
								stroke="var(--accent-red)"
								fill="var(--accent-red)"
								fillOpacity={0.6}
								animationDuration={300}
							/>
							<Area
								type="monotone"
								dataKey="hold"
								stackId="actions"
								stroke="var(--accent-blue)"
								fill="var(--accent-blue)"
								fillOpacity={0.6}
								animationDuration={300}
							/>
						</AreaChart>
					</ResponsiveContainer>
				</>
			)}
		</Card>
	);
}
