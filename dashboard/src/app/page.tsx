"use client";

import { MetricsLineChart } from "@/components/charts/MetricsLineChart";
import { StatusBadge } from "@/components/common/Badge";
import { Card, CardHeader } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { MetricCards } from "@/components/experiments/MetricCards";
import { ActionDistributionChart } from "@/components/training/ActionDistributionChart";
import { EpisodePnLChart } from "@/components/training/EpisodePnLChart";
import { EpisodeRewardChart } from "@/components/training/EpisodeRewardChart";
import { ProgressBar } from "@/components/training/ProgressBar";
import { StatusIndicator } from "@/components/training/StatusIndicator";
import { TrainingControls } from "@/components/training/TrainingControls";
import { useApi } from "@/hooks/useApi";
import { getExperiments } from "@/lib/api";
import { formatCurrency } from "@/lib/formatters";
import type { ExperimentSummary, TrainingUpdate } from "@/lib/types";
import { useTrainingStore } from "@/stores/trainingStore";
import { useCallback, useMemo } from "react";

function buildLatestMetrics(iterations: TrainingUpdate[]): Record<string, number> {
	const latest = iterations[iterations.length - 1];
	if (!latest) return {};
	return {
		episode_return_mean: latest.episode_return_mean,
		pnl_mean: latest.pnl_mean,
		pnl_pct_mean: latest.pnl_pct_mean,
		net_worth_mean: latest.net_worth_mean,
		trade_count_mean: latest.trade_count_mean,
	};
}

function RecentExperimentsTable({
	experiments,
}: {
	experiments: ExperimentSummary[];
}) {
	return (
		<div className="overflow-x-auto">
			<table className="w-full text-sm">
				<thead>
					<tr className="border-b border-[var(--border-color)] text-left text-xs text-[var(--text-secondary)]">
						<th className="pb-2 pr-4 font-medium">Name</th>
						<th className="pb-2 pr-4 font-medium">Script</th>
						<th className="pb-2 pr-4 font-medium">Status</th>
						<th className="pb-2 font-medium text-right">PnL</th>
					</tr>
				</thead>
				<tbody>
					{experiments.map((exp) => (
						<tr key={exp.id} className="border-b border-[var(--border-color)] last:border-0">
							<td className="py-2 pr-4">
								<a
									href={`/experiments/${exp.id}`}
									className="text-[var(--accent-blue)] hover:underline"
								>
									{exp.name}
								</a>
							</td>
							<td className="py-2 pr-4 text-[var(--text-secondary)]">{exp.script}</td>
							<td className="py-2 pr-4">
								<StatusBadge status={exp.status} />
							</td>
							<td className="py-2 text-right font-mono">
								{exp.final_metrics.pnl_mean !== undefined ? (
									<span
										className={
											exp.final_metrics.pnl_mean >= 0
												? "text-[var(--accent-green)]"
												: "text-[var(--accent-red)]"
										}
									>
										{formatCurrency(exp.final_metrics.pnl_mean)}
									</span>
								) : (
									<span className="text-[var(--text-secondary)]">--</span>
								)}
							</td>
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
}

export default function TrainingOverviewPage() {
	const store = useTrainingStore();

	const recentFetcher = useCallback(() => getExperiments({ limit: 5 }), []);
	const {
		data: recentExperiments,
		loading: expLoading,
		error: expError,
	} = useApi<ExperimentSummary[]>(recentFetcher, []);

	const latestMetrics = useMemo(() => buildLatestMetrics(store.iterations), [store.iterations]);

	const iterationRecords = useMemo(
		() =>
			store.iterations.map((it, idx) => ({
				id: idx,
				experiment_id: store.status?.experiment_id ?? "",
				iteration: it.iteration,
				metrics: {
					episode_return_mean: it.episode_return_mean,
					pnl_mean: it.pnl_mean,
					pnl_pct_mean: it.pnl_pct_mean,
					net_worth_mean: it.net_worth_mean,
					trade_count_mean: it.trade_count_mean,
				},
				timestamp: new Date().toISOString(),
			})),
		[store.iterations, store.status?.experiment_id],
	);

	return (
		<div className="space-y-6">
			<div className="flex items-center justify-between">
				<h1 className="text-xl font-semibold text-[var(--text-primary)]">Training Overview</h1>
				<div className="flex items-center gap-4">
					<StatusIndicator />
					<TrainingControls />
				</div>
			</div>

			{/* Training Progress Bar */}
			<Card>
				<CardHeader title="Training Progress" />
				<ProgressBar />
			</Card>

			{/* Metric Cards */}
			<MetricCards metrics={latestMetrics} />

			{/* Training Metrics Chart */}
			<Card>
				<CardHeader title="Iteration Metrics" />
				<div className="h-80">
					{iterationRecords.length > 0 ? (
						<MetricsLineChart
							data={iterationRecords}
							metricKeys={["episode_return_mean", "pnl_mean"]}
						/>
					) : (
						<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
							No training iterations yet. Start a training run to see metrics.
						</div>
					)}
				</div>
			</Card>

			{/* Episode Charts */}
			<div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
				<EpisodeRewardChart />
				<EpisodePnLChart />
			</div>

			{/* Action Distribution */}
			<ActionDistributionChart />

			{/* Recent Experiments */}
			<Card>
				<CardHeader title="Recent Experiments" />
				{expLoading ? (
					<LoadingState message="Loading experiments..." />
				) : expError ? (
					<div className="py-6 text-center text-sm text-[var(--accent-red)]">
						Failed to load experiments: {expError.message}
					</div>
				) : recentExperiments && recentExperiments.length > 0 ? (
					<RecentExperimentsTable experiments={recentExperiments} />
				) : (
					<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
						No experiments found.
					</div>
				)}
			</Card>
		</div>
	);
}
