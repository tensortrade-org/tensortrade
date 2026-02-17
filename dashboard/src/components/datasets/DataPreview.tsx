"use client";

import { Card } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { getDatasetPreview } from "@/lib/api";
import type { FeatureStats } from "@/lib/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useCallback, useState } from "react";

interface PreviewStatsRowProps {
	featureName: string;
	stats: FeatureStats;
}

function PreviewStatsRow({ featureName, stats }: PreviewStatsRowProps) {
	return (
		<tr className="border-t border-[var(--border-color)]">
			<td className="py-2 pr-4 text-sm font-medium text-[var(--text-primary)]">{featureName}</td>
			<td className="py-2 pr-4 text-right text-sm tabular-nums text-[var(--text-secondary)]">
				{stats.mean.toFixed(4)}
			</td>
			<td className="py-2 pr-4 text-right text-sm tabular-nums text-[var(--text-secondary)]">
				{stats.std.toFixed(4)}
			</td>
			<td className="py-2 pr-4 text-right text-sm tabular-nums text-[var(--text-secondary)]">
				{stats.min.toFixed(4)}
			</td>
			<td className="py-2 text-right text-sm tabular-nums text-[var(--text-secondary)]">
				{stats.max.toFixed(4)}
			</td>
		</tr>
	);
}

export function DataPreview() {
	const { selectedDatasetId, preview, setPreview } = useDatasetStore();
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<Error | null>(null);

	const handleFetchPreview = useCallback(async () => {
		if (!selectedDatasetId) return;
		setLoading(true);
		setError(null);
		setPreview(null);
		try {
			const result = await getDatasetPreview(selectedDatasetId);
			if ("error" in result) {
				throw new Error(String((result as Record<string, unknown>).error));
			}
			setPreview(result);
		} catch (err) {
			setError(err instanceof Error ? err : new Error(String(err)));
		} finally {
			setLoading(false);
		}
	}, [selectedDatasetId, setPreview]);

	if (!selectedDatasetId) {
		return (
			<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
				Save the dataset first to preview its data.
			</div>
		);
	}

	return (
		<div className="space-y-4">
			<div className="flex items-center justify-between">
				<p className="text-sm text-[var(--text-secondary)]">
					Preview the generated dataset with computed features.
				</p>
				<button
					type="button"
					onClick={handleFetchPreview}
					disabled={loading}
					className="rounded-md bg-[var(--accent-purple)] px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
				>
					{loading ? "Loading..." : "Generate Preview"}
				</button>
			</div>

			{loading && <LoadingState message="Generating dataset preview..." />}

			{error && (
				<div className="py-4 text-center text-sm text-[var(--accent-red)]">
					Preview failed: {error.message}
				</div>
			)}

			{preview && !loading && (
				<div className="space-y-4">
					{/* Summary stats */}
					<div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
						<Card>
							<p className="text-xs text-[var(--text-secondary)]">Rows</p>
							<p className="text-lg font-semibold text-[var(--text-primary)]">
								{preview.rows.toLocaleString()}
							</p>
						</Card>
						<Card>
							<p className="text-xs text-[var(--text-secondary)]">Columns</p>
							<p className="text-lg font-semibold text-[var(--text-primary)]">
								{preview.columns.length}
							</p>
						</Card>
						<Card>
							<p className="text-xs text-[var(--text-secondary)]">Date Range</p>
							<p className="text-sm font-medium text-[var(--text-primary)]">
								{preview.date_range.start}
							</p>
							<p className="text-sm font-medium text-[var(--text-primary)]">
								{preview.date_range.end}
							</p>
						</Card>
					</div>

					{/* Feature statistics table */}
					{Object.keys(preview.feature_stats).length > 0 && (
						<Card>
							<div className="overflow-x-auto">
								<table className="w-full">
									<thead>
										<tr className="text-xs text-[var(--text-secondary)]">
											<th className="pb-2 text-left font-medium">Feature</th>
											<th className="pb-2 text-right font-medium">Mean</th>
											<th className="pb-2 text-right font-medium">Std</th>
											<th className="pb-2 text-right font-medium">Min</th>
											<th className="pb-2 text-right font-medium">Max</th>
										</tr>
									</thead>
									<tbody>
										{Object.entries(preview.feature_stats).map(([name, stats]) => (
											<PreviewStatsRow key={name} featureName={name} stats={stats} />
										))}
									</tbody>
								</table>
							</div>
						</Card>
					)}
				</div>
			)}
		</div>
	);
}
