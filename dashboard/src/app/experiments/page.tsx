"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { ExperimentCompare } from "@/components/experiments/ExperimentCompare";
import { ExperimentTable } from "@/components/experiments/ExperimentTable";
import { useApi } from "@/hooks/useApi";
import { getExperiment, getExperiments } from "@/lib/api";
import type { ExperimentDetail, ExperimentSummary } from "@/lib/types";
import { useCallback, useEffect, useState } from "react";

interface FilterState {
	script: string;
	status: string;
}

export default function ExperimentsPage() {
	const [filters, setFilters] = useState<FilterState>({
		script: "",
		status: "",
	});
	const [selectedIds, setSelectedIds] = useState<string[]>([]);
	const [isComparing, setIsComparing] = useState(false);
	const [compareDetails, setCompareDetails] = useState<ExperimentDetail[]>([]);
	const [compareLoading, setCompareLoading] = useState(false);
	const [compareError, setCompareError] = useState<Error | null>(null);

	const experimentFetcher = useCallback(
		() =>
			getExperiments({
				script: filters.script || undefined,
				status: filters.status || undefined,
			}),
		[filters.script, filters.status],
	);

	const {
		data: experiments,
		loading,
		error,
	} = useApi<ExperimentSummary[]>(experimentFetcher, [filters.script, filters.status]);

	const handleSelect = useCallback((id: string) => {
		setSelectedIds((prev) =>
			prev.includes(id) ? prev.filter((sid) => sid !== id) : [...prev, id],
		);
	}, []);

	const handleCompare = useCallback(async () => {
		if (selectedIds.length < 2) return;
		setIsComparing(true);
		setCompareLoading(true);
		setCompareError(null);

		try {
			const details = await Promise.all(selectedIds.map((id) => getExperiment(id)));
			setCompareDetails(details);
		} catch (err) {
			setCompareError(err instanceof Error ? err : new Error(String(err)));
		} finally {
			setCompareLoading(false);
		}
	}, [selectedIds]);

	const handleCloseCompare = useCallback(() => {
		setIsComparing(false);
		setCompareDetails([]);
		setCompareError(null);
	}, []);

	const scripts = Array.from(new Set((experiments ?? []).map((e) => e.script)));

	if (isComparing) {
		return (
			<div className="space-y-6">
				<div className="flex items-center justify-between">
					<h1 className="text-xl font-semibold text-[var(--text-primary)]">Compare Experiments</h1>
					<button
						type="button"
						onClick={handleCloseCompare}
						className="rounded-md border border-[var(--border-color)] px-4 py-2 text-sm text-[var(--text-secondary)] hover:bg-[var(--bg-secondary)]"
					>
						Back to List
					</button>
				</div>

				{compareLoading ? (
					<LoadingState message="Loading experiment details..." />
				) : compareError ? (
					<Card>
						<div className="py-6 text-center text-sm text-[var(--accent-red)]">
							Failed to load comparison: {compareError.message}
						</div>
					</Card>
				) : (
					<ExperimentCompare experiments={compareDetails} />
				)}
			</div>
		);
	}

	return (
		<div className="space-y-6">
			<div className="flex items-center justify-between">
				<h1 className="text-xl font-semibold text-[var(--text-primary)]">Experiments</h1>
				{selectedIds.length >= 2 && (
					<button
						type="button"
						onClick={handleCompare}
						className="rounded-md bg-[var(--accent-blue)] px-4 py-2 text-sm font-medium text-white hover:opacity-90"
					>
						Compare Selected ({selectedIds.length})
					</button>
				)}
			</div>

			{/* Filters */}
			<Card>
				<div className="flex flex-wrap items-center gap-4">
					<div className="flex items-center gap-2">
						<label htmlFor="script-filter" className="text-sm text-[var(--text-secondary)]">
							Script
						</label>
						<select
							id="script-filter"
							value={filters.script}
							onChange={(e) => setFilters((f) => ({ ...f, script: e.target.value }))}
							className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)]"
						>
							<option value="">All Scripts</option>
							{scripts.map((s) => (
								<option key={s} value={s}>
									{s}
								</option>
							))}
						</select>
					</div>

					<div className="flex items-center gap-2">
						<label htmlFor="status-filter" className="text-sm text-[var(--text-secondary)]">
							Status
						</label>
						<select
							id="status-filter"
							value={filters.status}
							onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))}
							className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)]"
						>
							<option value="">All Statuses</option>
							<option value="running">Running</option>
							<option value="completed">Completed</option>
							<option value="failed">Failed</option>
							<option value="pruned">Pruned</option>
						</select>
					</div>

					{selectedIds.length > 0 && (
						<button
							type="button"
							onClick={() => setSelectedIds([])}
							className="ml-auto text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
						>
							Clear Selection ({selectedIds.length})
						</button>
					)}
				</div>
			</Card>

			{/* Experiment Table */}
			{loading ? (
				<LoadingState message="Loading experiments..." />
			) : error ? (
				<Card>
					<div className="py-6 text-center text-sm text-[var(--accent-red)]">
						Failed to load experiments: {error.message}
					</div>
				</Card>
			) : experiments && experiments.length > 0 ? (
				<ExperimentTable
					experiments={experiments}
					selectable
					selectedIds={selectedIds}
					onSelect={handleSelect}
				/>
			) : (
				<Card>
					<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
						No experiments found matching filters.
					</div>
				</Card>
			)}
		</div>
	);
}
