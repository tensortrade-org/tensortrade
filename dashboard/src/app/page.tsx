"use client";

import { StatusBadge } from "@/components/common/Badge";
import { Card, CardHeader } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { StatusIndicator } from "@/components/training/StatusIndicator";
import { TrainingControls } from "@/components/training/TrainingControls";
import { useApi } from "@/hooks/useApi";
import { getDashboardStats, getExperiments, getLeaderboard } from "@/lib/api";
import { formatCurrency, formatNumber, formatPercent, formatPnl } from "@/lib/formatters";
import type { DashboardStats, ExperimentSummary, LeaderboardEntry } from "@/lib/types";
import { useTrainingStore } from "@/stores/trainingStore";
import Link from "next/link";
import { useCallback } from "react";

// --- Stat Card ---

interface StatCardProps {
	label: string;
	value: string;
	accent: "blue" | "green" | "red" | "amber" | "neutral";
}

const accentClasses: Record<
	StatCardProps["accent"],
	{ border: string; text: string; glow: string }
> = {
	blue: {
		border: "border-l-[var(--accent-blue)]",
		text: "text-[var(--accent-blue)]",
		glow: "shadow-[inset_0_0_20px_-12px_var(--accent-blue)]",
	},
	green: {
		border: "border-l-[var(--accent-green)]",
		text: "text-[var(--accent-green)]",
		glow: "shadow-[inset_0_0_20px_-12px_var(--accent-green)]",
	},
	red: {
		border: "border-l-[var(--accent-red)]",
		text: "text-[var(--accent-red)]",
		glow: "shadow-[inset_0_0_20px_-12px_var(--accent-red)]",
	},
	amber: {
		border: "border-l-[var(--accent-amber)]",
		text: "text-[var(--accent-amber)]",
		glow: "shadow-[inset_0_0_20px_-12px_var(--accent-amber)]",
	},
	neutral: {
		border: "border-l-[var(--border-color)]",
		text: "text-[var(--text-primary)]",
		glow: "",
	},
};

function StatCard({ label, value, accent }: StatCardProps) {
	const styles = accentClasses[accent];
	return (
		<div
			className={`rounded-lg border border-[var(--border-color)] border-l-2 ${styles.border} bg-[var(--bg-card)] p-4 ${styles.glow}`}
		>
			<p className="text-xs font-medium uppercase tracking-wider text-[var(--text-secondary)]">
				{label}
			</p>
			<p className={`mt-1.5 font-mono text-xl font-semibold ${styles.text}`}>{value}</p>
		</div>
	);
}

// --- Top Performers Table ---

function TopPerformersTable({ entries }: { entries: LeaderboardEntry[] }) {
	return (
		<div className="overflow-x-auto">
			<table className="w-full text-sm">
				<thead>
					<tr className="border-b border-[var(--border-color)] text-left text-xs text-[var(--text-secondary)]">
						<th className="pb-2 pr-3 font-medium w-8">#</th>
						<th className="pb-2 pr-3 font-medium">Experiment</th>
						<th className="pb-2 font-medium text-right">PnL</th>
					</tr>
				</thead>
				<tbody>
					{entries.map((entry) => (
						<tr
							key={entry.experiment_id}
							className="border-b border-[var(--border-color)]/50 last:border-0"
						>
							<td className="py-2 pr-3 font-mono text-xs text-[var(--text-secondary)]">
								{entry.rank}
							</td>
							<td className="py-2 pr-3">
								<Link
									href={`/experiments/${entry.experiment_id}`}
									className="text-[var(--accent-blue)] hover:underline"
								>
									{entry.name}
								</Link>
							</td>
							<td className="py-2 text-right font-mono">
								<span
									className={
										entry.metric_value >= 0
											? "text-[var(--accent-green)]"
											: "text-[var(--accent-red)]"
									}
								>
									{formatPnl(entry.metric_value)}
								</span>
							</td>
						</tr>
					))}
					{entries.length === 0 && (
						<tr>
							<td colSpan={3} className="py-6 text-center text-xs text-[var(--text-secondary)]">
								No completed experiments yet
							</td>
						</tr>
					)}
				</tbody>
			</table>
		</div>
	);
}

// --- Recent Experiments Table ---

function RecentExperimentsTable({ experiments }: { experiments: ExperimentSummary[] }) {
	return (
		<div className="overflow-x-auto">
			<table className="w-full text-sm">
				<thead>
					<tr className="border-b border-[var(--border-color)] text-left text-xs text-[var(--text-secondary)]">
						<th className="pb-2 pr-3 font-medium">Name</th>
						<th className="pb-2 pr-3 font-medium">Status</th>
						<th className="pb-2 font-medium text-right">PnL</th>
					</tr>
				</thead>
				<tbody>
					{experiments.map((exp) => (
						<tr key={exp.id} className="border-b border-[var(--border-color)]/50 last:border-0">
							<td className="py-2 pr-3">
								<Link
									href={`/experiments/${exp.id}`}
									className="text-[var(--accent-blue)] hover:underline"
								>
									{exp.name}
								</Link>
							</td>
							<td className="py-2 pr-3">
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
										{formatPnl(exp.final_metrics.pnl_mean)}
									</span>
								) : (
									<span className="text-[var(--text-secondary)]">--</span>
								)}
							</td>
						</tr>
					))}
					{experiments.length === 0 && (
						<tr>
							<td colSpan={3} className="py-6 text-center text-xs text-[var(--text-secondary)]">
								No experiments yet
							</td>
						</tr>
					)}
				</tbody>
			</table>
		</div>
	);
}

// --- Quick Action Card ---

interface QuickActionProps {
	href: string;
	icon: string;
	title: string;
	description: string;
}

function QuickAction({ href, icon, title, description }: QuickActionProps) {
	return (
		<Link
			href={href}
			className="group flex items-center gap-3 rounded-lg border border-[var(--border-color)] bg-[var(--bg-card)] p-4 transition-colors hover:border-[var(--accent-blue)]/40 hover:bg-[var(--bg-tertiary)]"
		>
			<span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-[var(--accent-blue)]/10 text-lg text-[var(--accent-blue)] transition-colors group-hover:bg-[var(--accent-blue)]/20">
				{icon}
			</span>
			<div className="min-w-0">
				<p className="text-sm font-medium text-[var(--text-primary)]">{title}</p>
				<p className="text-xs text-[var(--text-secondary)]">{description}</p>
			</div>
		</Link>
	);
}

// --- Main Page ---

export default function MissionControlPage() {
	const isTraining = useTrainingStore((s) => s.status?.is_training ?? false);

	const statsFetcher = useCallback(() => getDashboardStats(), []);
	const leaderboardFetcher = useCallback(() => getLeaderboard({ metric: "pnl" }), []);
	const recentFetcher = useCallback(() => getExperiments({ limit: 5 }), []);

	const { data: stats, loading: statsLoading } = useApi<DashboardStats>(statsFetcher, []);
	const { data: leaderboard, loading: lbLoading } = useApi<LeaderboardEntry[]>(
		leaderboardFetcher,
		[],
	);
	const { data: recent, loading: recentLoading } = useApi<ExperimentSummary[]>(recentFetcher, []);

	const topPerformers = leaderboard?.slice(0, 5) ?? [];

	return (
		<div className="space-y-6">
			{/* Page Header */}
			<div className="flex items-center justify-between">
				<div>
					<h1 className="text-xl font-semibold text-[var(--text-primary)]">Mission Control</h1>
					<p className="mt-0.5 text-xs text-[var(--text-secondary)]">
						Aggregate performance across all training runs
					</p>
				</div>
				{stats && (
					<div className="text-right text-xs text-[var(--text-secondary)]">
						<span className="font-mono">{formatNumber(stats.total_experiments)}</span> experiments
						tracked
					</div>
				)}
			</div>

			{/* Active Training Banner */}
			{isTraining && (
				<div className="flex items-center justify-between rounded-lg border border-[var(--accent-blue)]/30 bg-[var(--accent-blue)]/5 px-4 py-3">
					<div className="flex items-center gap-4">
						<StatusIndicator />
					</div>
					<div className="flex items-center gap-3">
						<TrainingControls />
						<Link
							href="/training"
							className="rounded-md bg-[var(--accent-blue)]/15 px-3 py-1.5 text-xs font-medium text-[var(--accent-blue)] transition-colors hover:bg-[var(--accent-blue)]/25"
						>
							Open Monitor
						</Link>
					</div>
				</div>
			)}

			{/* Stats Grid */}
			{statsLoading ? (
				<LoadingState message="Loading stats..." />
			) : stats ? (
				<div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
					<StatCard
						label="Total Runs"
						value={formatNumber(stats.total_experiments)}
						accent="blue"
					/>
					<StatCard label="Completed" value={formatNumber(stats.completed)} accent="blue" />
					<StatCard
						label="Win Rate"
						value={formatPercent(stats.win_rate)}
						accent={stats.win_rate > 50 ? "green" : stats.win_rate > 0 ? "amber" : "neutral"}
					/>
					<StatCard
						label="Best PnL"
						value={stats.best_pnl !== null ? formatCurrency(stats.best_pnl) : "--"}
						accent={stats.best_pnl !== null && stats.best_pnl > 0 ? "green" : "neutral"}
					/>
					<StatCard
						label="Best Net Worth"
						value={stats.best_net_worth !== null ? formatCurrency(stats.best_net_worth) : "--"}
						accent={stats.best_net_worth !== null ? "green" : "neutral"}
					/>
					<StatCard
						label="Total Trades"
						value={formatNumber(stats.total_trades)}
						accent="neutral"
					/>
				</div>
			) : null}

			{/* Two-Column: Leaderboard + Recent */}
			<div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
				<Card>
					<CardHeader
						title="Top Performers"
						action={
							<Link
								href="/leaderboard"
								className="text-xs text-[var(--accent-blue)] hover:underline"
							>
								View all
							</Link>
						}
					/>
					{lbLoading ? (
						<LoadingState message="Loading leaderboard..." />
					) : (
						<TopPerformersTable entries={topPerformers} />
					)}
				</Card>

				<Card>
					<CardHeader
						title="Recent Experiments"
						action={
							<Link
								href="/experiments"
								className="text-xs text-[var(--accent-blue)] hover:underline"
							>
								View all
							</Link>
						}
					/>
					{recentLoading ? (
						<LoadingState message="Loading experiments..." />
					) : (
						<RecentExperimentsTable experiments={recent ?? []} />
					)}
				</Card>
			</div>

			{/* Quick Actions */}
			<div>
				<h2 className="mb-3 text-xs font-medium uppercase tracking-wider text-[var(--text-secondary)]">
					Quick Actions
				</h2>
				<div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
					<QuickAction
						href="/launch"
						icon={"\u25B6"}
						title="Launch Training"
						description="Start a new training run"
					/>
					<QuickAction
						href="/campaign"
						icon={"\u2694"}
						title="Alpha Search"
						description="Run Optuna HP optimization"
					/>
					<QuickAction
						href="/hyperparams"
						icon={"\u2692"}
						title="HP Studio"
						description="Edit hyperparameter packs"
					/>
				</div>
			</div>
		</div>
	);
}
