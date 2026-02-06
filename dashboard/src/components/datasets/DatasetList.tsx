"use client";

import { Badge } from "@/components/common/Badge";
import { LoadingState } from "@/components/common/Spinner";
import { useApi } from "@/hooks/useApi";
import { getDatasets } from "@/lib/api";
import type { DatasetConfig } from "@/lib/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useCallback, useEffect, useMemo, useState } from "react";

const sourceTypeVariant: Record<string, "info" | "purple" | "success"> = {
	csv_upload: "info",
	crypto_download: "purple",
	synthetic: "success",
};

const sourceTypeLabel: Record<string, string> = {
	csv_upload: "CSV",
	crypto_download: "Crypto",
	synthetic: "Synthetic",
};

interface DatasetListProps {
	onNewDataset?: () => void;
}

export function DatasetList({ onNewDataset }: DatasetListProps) {
	const { datasets, selectedDatasetId, selectDataset, setDatasets, setLoading, setError } =
		useDatasetStore();
	const [search, setSearch] = useState("");

	const fetcher = useCallback(() => getDatasets(), []);
	const { data, loading, error } = useApi<DatasetConfig[]>(fetcher, []);

	useEffect(() => {
		if (data) {
			setDatasets(data);
		}
	}, [data, setDatasets]);

	useEffect(() => {
		setLoading(loading);
	}, [loading, setLoading]);

	useEffect(() => {
		if (error) setError(error.message);
	}, [error, setError]);

	const filtered = useMemo(() => {
		if (!search.trim()) return datasets;
		const q = search.toLowerCase();
		return datasets.filter(
			(d) =>
				d.name.toLowerCase().includes(q) ||
				d.source_type.toLowerCase().includes(q) ||
				d.description.toLowerCase().includes(q),
		);
	}, [datasets, search]);

	const handleNewDataset = useCallback(() => {
		if (onNewDataset) {
			onNewDataset();
		} else {
			selectDataset(null);
		}
	}, [onNewDataset, selectDataset]);

	if (loading) return <LoadingState message="Loading datasets..." />;
	if (error) {
		return <div className="p-4 text-sm text-[var(--accent-red)]">Failed to load datasets</div>;
	}

	return (
		<div className="flex h-full flex-col border-r border-[var(--border-color)]">
			<div className="p-3 space-y-2">
				<input
					type="text"
					placeholder="Search datasets..."
					value={search}
					onChange={(e) => setSearch(e.target.value)}
					className="w-full rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:border-[var(--accent-blue)] focus:outline-none"
				/>
				<button
					type="button"
					onClick={handleNewDataset}
					className="w-full rounded bg-[var(--accent-blue)] px-3 py-1.5 text-sm font-medium text-white hover:opacity-90 transition-opacity"
				>
					New Dataset
				</button>
			</div>
			<div className="flex-1 overflow-y-auto">
				{filtered.map((dataset) => (
					<button
						key={dataset.id}
						type="button"
						onClick={() => selectDataset(dataset.id)}
						className={`w-full text-left px-3 py-3 border-b border-[var(--border-color)] transition-colors hover:bg-[var(--bg-secondary)] ${
							selectedDatasetId === dataset.id
								? "border-l-2 border-l-[var(--accent-blue)] bg-[var(--bg-secondary)]"
								: "border-l-2 border-l-transparent"
						}`}
					>
						<div className="flex items-center gap-2 mb-1">
							<span className="text-sm font-medium text-[var(--text-primary)] truncate">
								{dataset.name}
							</span>
							<Badge
								label={sourceTypeLabel[dataset.source_type] ?? dataset.source_type}
								variant={sourceTypeVariant[dataset.source_type] ?? "default"}
							/>
						</div>
						<p className="text-xs text-[var(--text-secondary)] truncate">{dataset.description}</p>
					</button>
				))}
				{filtered.length === 0 && (
					<div className="p-4 text-center text-xs text-[var(--text-secondary)]">
						No datasets found
					</div>
				)}
			</div>
		</div>
	);
}
