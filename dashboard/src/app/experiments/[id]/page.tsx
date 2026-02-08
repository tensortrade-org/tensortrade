"use client";

import { MetricsLineChart } from "@/components/charts/MetricsLineChart";
import { Badge, StatusBadge } from "@/components/common/Badge";
import { Card, CardHeader } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { MetricCards } from "@/components/experiments/MetricCards";
import { TradingBehaviorCard } from "@/components/experiments/TradingBehaviorCard";
import { InsightCard } from "@/components/insights/InsightCard";
import { InsightRequest } from "@/components/insights/InsightRequest";
import { useApi } from "@/hooks/useApi";
import { getExperiment } from "@/lib/api";
import { formatDate } from "@/lib/formatters";
import type { ExperimentDetail, InsightReport } from "@/lib/types";
import { useParams } from "next/navigation";
import { useCallback, useMemo, useState } from "react";

export default function ExperimentDetailPage() {
	const params = useParams();
	const experimentId = params.id as string;

	const [showInsightRequest, setShowInsightRequest] = useState(false);
	const [insightReport, setInsightReport] = useState<InsightReport | null>(null);

	const experimentFetcher = useCallback(() => getExperiment(experimentId), [experimentId]);

	const {
		data: detail,
		loading: detailLoading,
		error: detailError,
	} = useApi<ExperimentDetail>(experimentFetcher, [experimentId]);

	const metricKeys = useMemo(() => {
		if (!detail?.iterations || detail.iterations.length === 0) return [];
		const firstMetrics = detail.iterations[0].metrics;
		return Object.keys(firstMetrics);
	}, [detail?.iterations]);

	const handleInsightComplete = useCallback((report: InsightReport) => {
		setInsightReport(report);
		setShowInsightRequest(false);
	}, []);

	if (detailLoading) {
		return <LoadingState message="Loading experiment..." />;
	}

	if (detailError) {
		return (
			<Card>
				<div className="py-6 text-center text-sm text-[var(--accent-red)]">
					Failed to load experiment: {detailError.message}
				</div>
			</Card>
		);
	}

	if (!detail) {
		return (
			<Card>
				<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
					Experiment not found.
				</div>
			</Card>
		);
	}

	const { experiment, iterations } = detail;

	return (
		<div className="space-y-6">
			{/* Header */}
			<div className="flex items-center justify-between">
				<div className="space-y-1">
					<div className="flex items-center gap-3">
						<h1 className="text-xl font-semibold text-[var(--text-primary)]">{experiment.name}</h1>
						<StatusBadge status={experiment.status} />
					</div>
					<div className="flex items-center gap-4 text-sm text-[var(--text-secondary)]">
						<span>Script: {experiment.script}</span>
						<span>Started: {formatDate(experiment.started_at)}</span>
						{experiment.completed_at && (
							<span>Completed: {formatDate(experiment.completed_at)}</span>
						)}
					</div>
					{experiment.tags.length > 0 && (
						<div className="flex items-center gap-2 pt-1">
							{experiment.tags.map((tag) => (
								<Badge key={tag} label={tag} variant="purple" />
							))}
						</div>
					)}
				</div>
				<button
					type="button"
					onClick={() => setShowInsightRequest(true)}
					className="rounded-md bg-[var(--accent-purple)] px-4 py-2 text-sm font-medium text-white hover:opacity-90"
				>
					Analyze with AI
				</button>
			</div>

			{/* Insight Request Modal */}
			{showInsightRequest && (
				<Card>
					<CardHeader title="AI Analysis" />
					<InsightRequest experimentIds={[experimentId]} onComplete={handleInsightComplete} />
				</Card>
			)}

			{/* Insight Result */}
			{insightReport && <InsightCard insight={insightReport} />}

			{/* Final Metrics */}
			<MetricCards metrics={experiment.final_metrics} />

			{/* Trading Behavior */}
			<TradingBehaviorCard iterations={iterations} />

			{/* Iterations Chart */}
			<Card>
				<CardHeader title="Training Iterations" />
				<div className="h-80">
					{iterations.length > 0 && metricKeys.length > 0 ? (
						<MetricsLineChart data={iterations} metricKeys={metricKeys} />
					) : (
						<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
							No iteration data available.
						</div>
					)}
				</div>
			</Card>

			{/* Config */}
			<Card>
				<CardHeader title="Configuration" />
				<pre className="overflow-x-auto rounded-md bg-[var(--bg-secondary)] p-4 text-xs text-[var(--text-primary)]">
					{JSON.stringify(experiment.config, null, 2)}
				</pre>
			</Card>
		</div>
	);
}
