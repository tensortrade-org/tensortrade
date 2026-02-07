"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { formatNumber, formatPercent } from "@/lib/formatters";
import type { IterationRecord } from "@/lib/types";
import { useMemo } from "react";

interface TradingBehaviorCardProps {
	iterations: IterationRecord[];
}

interface ActionStats {
	buyCount: number;
	sellCount: number;
	holdCount: number;
	tradeCount: number;
	totalActions: number;
	buyPct: number;
	sellPct: number;
	holdPct: number;
}

function extractLastStats(iterations: IterationRecord[]): ActionStats | null {
	if (iterations.length === 0) return null;

	const last = iterations[iterations.length - 1].metrics;
	const buyCount = last.buy_count_mean ?? 0;
	const sellCount = last.sell_count_mean ?? 0;
	const holdCount = last.hold_count_mean ?? 0;
	const tradeCount = last.trade_count_mean ?? buyCount + sellCount;
	const totalActions = buyCount + sellCount + holdCount;

	if (totalActions === 0) return null;

	return {
		buyCount,
		sellCount,
		holdCount,
		tradeCount,
		totalActions,
		buyPct: (buyCount / totalActions) * 100,
		sellPct: (sellCount / totalActions) * 100,
		holdPct: (holdCount / totalActions) * 100,
	};
}

function DistributionBar({ stats }: { stats: ActionStats }) {
	return (
		<div className="space-y-2">
			<div className="flex h-3 overflow-hidden rounded-full bg-[var(--bg-secondary)]">
				{stats.holdPct > 0 && (
					<div
						className="bg-[var(--text-secondary)] transition-all"
						style={{ width: `${stats.holdPct}%` }}
						title={`Hold: ${stats.holdPct.toFixed(1)}%`}
					/>
				)}
				{stats.buyPct > 0 && (
					<div
						className="bg-[var(--accent-green)] transition-all"
						style={{ width: `${stats.buyPct}%` }}
						title={`Buy: ${stats.buyPct.toFixed(1)}%`}
					/>
				)}
				{stats.sellPct > 0 && (
					<div
						className="bg-[var(--accent-red)] transition-all"
						style={{ width: `${stats.sellPct}%` }}
						title={`Sell: ${stats.sellPct.toFixed(1)}%`}
					/>
				)}
			</div>
			<div className="flex items-center gap-4 text-xs">
				<span className="flex items-center gap-1.5">
					<span className="inline-block h-2.5 w-2.5 rounded-sm bg-[var(--text-secondary)]" />
					Hold {stats.holdPct.toFixed(1)}%
				</span>
				<span className="flex items-center gap-1.5">
					<span className="inline-block h-2.5 w-2.5 rounded-sm bg-[var(--accent-green)]" />
					Buy {stats.buyPct.toFixed(1)}%
				</span>
				<span className="flex items-center gap-1.5">
					<span className="inline-block h-2.5 w-2.5 rounded-sm bg-[var(--accent-red)]" />
					Sell {stats.sellPct.toFixed(1)}%
				</span>
			</div>
		</div>
	);
}

function TradeCountSparkline({ iterations }: { iterations: IterationRecord[] }) {
	const data = useMemo(() => {
		return iterations.map((it) => it.metrics.trade_count_mean ?? 0);
	}, [iterations]);

	if (data.length < 2) return null;

	const max = Math.max(...data, 1);
	const width = 240;
	const height = 40;
	const stepX = width / (data.length - 1);

	const points = data.map((v, i) => `${i * stepX},${height - (v / max) * height}`).join(" ");

	return (
		<div className="flex items-center gap-3">
			<span className="shrink-0 text-xs text-[var(--text-secondary)]">Trade freq.</span>
			<svg
				width={width}
				height={height}
				className="overflow-visible"
				role="img"
				aria-label="Trade frequency sparkline"
			>
				<title>Trade frequency over iterations</title>
				<polyline
					fill="none"
					stroke="var(--accent-blue)"
					strokeWidth="1.5"
					strokeLinejoin="round"
					points={points}
				/>
			</svg>
		</div>
	);
}

function StatBox({
	label,
	value,
	subtext,
}: {
	label: string;
	value: string;
	subtext?: string;
}) {
	return (
		<div className="flex flex-col gap-0.5">
			<span className="text-xs text-[var(--text-secondary)]">{label}</span>
			<span className="text-sm font-semibold text-[var(--text-primary)]">{value}</span>
			{subtext && <span className="text-xs text-[var(--text-secondary)]">{subtext}</span>}
		</div>
	);
}

export function TradingBehaviorCard({ iterations }: TradingBehaviorCardProps) {
	const stats = useMemo(() => extractLastStats(iterations), [iterations]);

	const trendStats = useMemo(() => {
		if (iterations.length < 2) return null;

		const tradeCounts = iterations.map((it) => it.metrics.trade_count_mean ?? 0);
		const first5 = tradeCounts.slice(0, Math.min(5, tradeCounts.length));
		const last5 = tradeCounts.slice(-Math.min(5, tradeCounts.length));
		const earlyAvg = first5.reduce((a, b) => a + b, 0) / first5.length;
		const lateAvg = last5.reduce((a, b) => a + b, 0) / last5.length;
		const trendPct = earlyAvg > 0 ? ((lateAvg - earlyAvg) / earlyAvg) * 100 : 0;

		return {
			earlyAvg,
			lateAvg,
			trendPct,
		};
	}, [iterations]);

	if (!stats) {
		return (
			<Card>
				<CardHeader title="Trading Behavior" />
				<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
					No trading activity data available.
				</div>
			</Card>
		);
	}

	return (
		<Card>
			<CardHeader title="Trading Behavior" />
			<div className="space-y-5">
				{/* Action Distribution */}
				<div className="space-y-2">
					<span className="text-xs font-medium text-[var(--text-secondary)]">
						Action Distribution (last iteration avg)
					</span>
					<DistributionBar stats={stats} />
				</div>

				{/* Stat Grid */}
				<div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
					<StatBox
						label="Avg Trades / Episode"
						value={formatNumber(stats.tradeCount, 1)}
						subtext={`${formatNumber(stats.buyCount, 1)} buys, ${formatNumber(stats.sellCount, 1)} sells`}
					/>
					<StatBox label="Avg Holds / Episode" value={formatNumber(stats.holdCount, 1)} />
					<StatBox
						label="Trade Ratio"
						value={`${((stats.tradeCount / stats.totalActions) * 100).toFixed(1)}%`}
						subtext="of all actions"
					/>
					{trendStats && (
						<StatBox
							label="Trade Trend"
							value={formatPercent(trendStats.trendPct)}
							subtext={`${formatNumber(trendStats.earlyAvg, 1)} early, ${formatNumber(trendStats.lateAvg, 1)} late`}
						/>
					)}
				</div>

				{/* Sparkline */}
				{iterations.length >= 3 && <TradeCountSparkline iterations={iterations} />}
			</div>
		</Card>
	);
}
