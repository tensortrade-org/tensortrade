"use client";

import { useApi } from "@/hooks/useApi";
import { getDatasets } from "@/lib/api";
import type { DatasetConfig } from "@/lib/types";
import { useLaunchStore } from "@/stores/launchStore";
import Link from "next/link";
import { useCallback } from "react";

export function DatasetStep() {
	const datasetId = useLaunchStore((s) => s.datasetId);
	const setDatasetId = useLaunchStore((s) => s.setDatasetId);

	const fetcher = useCallback(() => getDatasets(), []);
	const { data: datasets, loading, error } = useApi<DatasetConfig[]>(fetcher, []);

	const selectedDataset = datasets?.find((d) => d.id === datasetId) ?? null;

	return (
		<div className="space-y-4">
			<label
				htmlFor="dataset-select"
				className="block text-sm font-medium text-[var(--text-primary)]"
			>
				Select Dataset
			</label>

			{loading && <p className="text-sm text-[var(--text-secondary)]">Loading datasets...</p>}
			{error && (
				<p className="text-sm text-[var(--accent-red)]">Failed to load datasets: {error.message}</p>
			)}

			{datasets && datasets.length === 0 && (
				<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] p-4 text-center">
					<p className="text-sm text-[var(--text-secondary)]">No datasets available.</p>
					<Link
						href="/datasets"
						className="mt-2 inline-block text-sm text-[var(--accent-blue)] hover:underline"
					>
						+ Create New Dataset
					</Link>
				</div>
			)}

			{datasets && datasets.length > 0 && (
				<div className="space-y-2">
					{datasets.map((ds) => {
						const isSelected = ds.id === datasetId;
						return (
							<button
								key={ds.id}
								type="button"
								onClick={() => setDatasetId(ds.id)}
								className={`w-full rounded-md border p-3 text-left transition-colors ${
									isSelected
										? "border-[var(--accent-blue)] bg-[var(--accent-blue)]/10"
										: "border-[var(--border-color)] bg-[var(--bg-secondary)] hover:border-[var(--text-secondary)]"
								}`}
							>
								<div className="flex items-center justify-between">
									<span className="text-sm font-medium text-[var(--text-primary)]">{ds.name}</span>
									<span className="text-xs text-[var(--text-secondary)]">{ds.source_type}</span>
								</div>
								{ds.description && (
									<p className="mt-1 text-xs text-[var(--text-secondary)]">{ds.description}</p>
								)}
								<div className="mt-2 flex gap-4 text-xs text-[var(--text-secondary)]">
									<span>Features: {ds.features.length}</span>
									<span>
										Split: {ds.split_config.train_pct}/{ds.split_config.val_pct}/
										{ds.split_config.test_pct}
									</span>
								</div>
							</button>
						);
					})}
				</div>
			)}

			{selectedDataset && (
				<div className="rounded-md border border-[var(--accent-blue)]/30 bg-[var(--accent-blue)]/5 p-3">
					<p className="text-xs font-medium text-[var(--accent-blue)]">
						Selected: {selectedDataset.name}
					</p>
					<div className="mt-2 grid grid-cols-3 gap-2 text-xs">
						<div>
							<span className="text-[var(--text-secondary)]">Source: </span>
							<span className="text-[var(--text-primary)]">{selectedDataset.source_type}</span>
						</div>
						<div>
							<span className="text-[var(--text-secondary)]">Features: </span>
							<span className="text-[var(--text-primary)]">{selectedDataset.features.length}</span>
						</div>
						<div>
							<span className="text-[var(--text-secondary)]">Created: </span>
							<span className="text-[var(--text-primary)]">
								{new Date(selectedDataset.created_at).toLocaleDateString()}
							</span>
						</div>
					</div>
				</div>
			)}

			<Link
				href="/datasets"
				className="inline-block text-sm text-[var(--accent-blue)] hover:underline"
			>
				+ Create New Dataset
			</Link>
		</div>
	);
}
