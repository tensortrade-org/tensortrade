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

export default function TrainingMonitorPage() {
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

	const isWarmingUp = useTrainingStore((s) => s.isWarmingUp);
	const completedExperiment = useTrainingStore((s) => s.completedExperiment);
	const dismissCompleted = useTrainingStore((s) => s.dismissCompleted);

	return (
		<div className="space-y-6">
			<div className="flex items-center justify-between">
				<h1 className="text-xl font-semibold text-[var(--text-primary)]">Training Monitor</h1>
				<div className="flex items-center gap-4">
					<StatusIndicator />
					<TrainingControls />
				</div>
			</div>

			{/* Training Complete Banner */}
			{completedExperiment && (
				<div className="flex items-center justify-between rounded-lg border border-[var(--accent-green)]/30 bg-[var(--accent-green)]/10 px-4 py-3">
					<div className="flex items-center gap-3">
						<div className="flex h-8 w-8 items-center justify-center rounded-full bg-[var(--accent-green)]/20">
							<svg
								className="h-5 w-5 text-[var(--accent-green)]"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
								strokeWidth={2}
								aria-label="Checkmark"
								role="img"
							>
								<path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
							</svg>
						</div>
						<div>
							<p className="text-sm font-medium text-[var(--accent-green)]">
								Training {completedExperiment.status === "completed" ? "Complete" : "Ended"}
							</p>
							<p className="text-xs text-[var(--text-secondary)]">
								Experiment {completedExperiment.experimentId.slice(0, 8)} finished.{" "}
								<a
									href={`/experiments/${completedExperiment.experimentId}`}
									className="text-[var(--accent-blue)] hover:underline"
								>
									View results
								</a>
							</p>
						</div>
					</div>
					<button
						type="button"
						onClick={dismissCompleted}
						className="rounded p-1 text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)]"
					>
						<svg
							className="h-4 w-4"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
							strokeWidth={2}
							aria-label="Dismiss"
							role="img"
						>
							<path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
						</svg>
					</button>
				</div>
			)}

			{/* Warming Up State */}
			{isWarmingUp && (
				<Card>
					<div className="flex flex-col items-center gap-4 py-12">
						<div className="relative">
							<div className="h-12 w-12 rounded-full border-4 border-[var(--border-color)]" />
							<div className="absolute inset-0 h-12 w-12 animate-spin rounded-full border-4 border-transparent border-t-[var(--accent-amber)]" />
						</div>
						<div className="text-center">
							<p className="text-sm font-medium text-[var(--accent-amber)]">
								Training is starting up...
							</p>
							<p className="mt-1 text-xs text-[var(--text-secondary)]">
								Initializing Ray, loading data, and building the environment. This may take a
								minute.
							</p>
						</div>
					</div>
				</Card>
			)}

			{/* Training Progress Bar */}
			{!isWarmingUp && (
				<Card>
					<CardHeader title="Training Progress" />
					<ProgressBar />
				</Card>
			)}

			{/* Metric Cards */}
			{!isWarmingUp && <MetricCards metrics={latestMetrics} />}

			{/* Training Metrics Chart */}
			{!isWarmingUp && (
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
			)}

			{/* Episode Charts */}
			{!isWarmingUp && (
				<div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
					<EpisodeRewardChart />
					<EpisodePnLChart />
				</div>
			)}

			{/* Action Distribution */}
			{!isWarmingUp && <ActionDistributionChart />}

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
