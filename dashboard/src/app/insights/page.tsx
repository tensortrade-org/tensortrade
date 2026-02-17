"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { InsightCard } from "@/components/insights/InsightCard";
import { InsightRequest } from "@/components/insights/InsightRequest";
import { StrategyAdvice } from "@/components/insights/StrategyAdvice";
import { useApi } from "@/hooks/useApi";
import { getInsights } from "@/lib/api";
import type { InsightReport } from "@/lib/types";
import { useCallback, useMemo } from "react";

export default function InsightsPage() {
	const insightsFetcher = useCallback(() => getInsights(), []);
	const { data: insights, loading, error, refresh } = useApi<InsightReport[]>(insightsFetcher, []);

	const latestStrategy = useMemo(() => {
		if (!insights) return null;
		const strategies = insights.filter((i) => i.analysis_type === "strategy");
		if (strategies.length === 0) return null;
		return strategies.reduce((latest, current) =>
			new Date(current.created_at) > new Date(latest.created_at) ? current : latest,
		);
	}, [insights]);

	const nonStrategyInsights = useMemo(
		() => (insights ?? []).filter((i) => i.analysis_type !== "strategy"),
		[insights],
	);

	const handleRequestComplete = useCallback(() => {
		refresh();
	}, [refresh]);

	return (
		<div className="space-y-6">
			<h1 className="text-xl font-semibold text-[var(--text-primary)]">AI Insights</h1>

			{/* Request Form */}
			<Card>
				<CardHeader title="Request Analysis" />
				<InsightRequest onComplete={handleRequestComplete} />
			</Card>

			{/* Strategy Advice - prominent */}
			{latestStrategy && <StrategyAdvice insight={latestStrategy} />}

			{/* Insight List */}
			{loading ? (
				<LoadingState message="Loading insights..." />
			) : error ? (
				<Card>
					<div className="py-6 text-center text-sm text-[var(--accent-red)]">
						Failed to load insights: {error.message}
					</div>
				</Card>
			) : nonStrategyInsights.length > 0 ? (
				<div className="space-y-4">
					<h2 className="text-sm font-medium text-[var(--text-secondary)]">Previous Analyses</h2>
					{nonStrategyInsights.map((insight) => (
						<InsightCard key={insight.id} insight={insight} />
					))}
				</div>
			) : (
				<Card>
					<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
						No insights yet. Use the form above to request an analysis.
					</div>
				</Card>
			)}
		</div>
	);
}
