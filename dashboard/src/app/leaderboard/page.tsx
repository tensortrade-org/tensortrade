"use client";

import { Card } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { LeaderboardFilters } from "@/components/leaderboard/LeaderboardFilters";
import { LeaderboardTable } from "@/components/leaderboard/LeaderboardTable";
import { useApi } from "@/hooks/useApi";
import { getExperiments, getLeaderboard } from "@/lib/api";
import type { ExperimentSummary, LeaderboardEntry } from "@/lib/types";
import { useCallback, useMemo, useState } from "react";

interface LeaderboardFilterState {
	script: string;
	metric: string;
}

export default function LeaderboardPage() {
	const [filters, setFilters] = useState<LeaderboardFilterState>({
		script: "",
		metric: "",
	});

	const leaderboardFetcher = useCallback(
		() =>
			getLeaderboard({
				script: filters.script || undefined,
				metric: filters.metric || undefined,
			}),
		[filters.script, filters.metric],
	);

	const scriptsFetcher = useCallback(() => getExperiments(), []);

	const {
		data: entries,
		loading,
		error,
	} = useApi<LeaderboardEntry[]>(leaderboardFetcher, [filters.script, filters.metric]);

	const { data: allExperiments } = useApi<ExperimentSummary[]>(scriptsFetcher, []);

	const scripts = useMemo(
		() => Array.from(new Set((allExperiments ?? []).map((e) => e.script))),
		[allExperiments],
	);

	const handleScriptChange = useCallback((script: string) => {
		setFilters((f) => ({ ...f, script }));
	}, []);

	const handleMetricChange = useCallback((metric: string) => {
		setFilters((f) => ({ ...f, metric }));
	}, []);

	return (
		<div className="space-y-6">
			<h1 className="text-xl font-semibold text-[var(--text-primary)]">Leaderboard</h1>

			<LeaderboardFilters
				scripts={scripts}
				onScriptChange={handleScriptChange}
				onMetricChange={handleMetricChange}
				currentScript={filters.script}
				currentMetric={filters.metric}
			/>

			{loading ? (
				<LoadingState message="Loading leaderboard..." />
			) : error ? (
				<Card>
					<div className="py-6 text-center text-sm text-[var(--accent-red)]">
						Failed to load leaderboard: {error.message}
					</div>
				</Card>
			) : entries && entries.length > 0 ? (
				<LeaderboardTable entries={entries} />
			) : (
				<Card>
					<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
						No leaderboard entries found.
					</div>
				</Card>
			)}
		</div>
	);
}
