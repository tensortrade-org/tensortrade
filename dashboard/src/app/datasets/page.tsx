"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { DataPreview } from "@/components/datasets/DataPreview";
import { DataPreviewChart } from "@/components/datasets/DataPreviewChart";
import { DatasetList } from "@/components/datasets/DatasetList";
import { FeatureBuilder } from "@/components/datasets/FeatureBuilder";
import { SourceConfig } from "@/components/datasets/SourceConfig";
import { SplitSlider } from "@/components/datasets/SplitSlider";
import { createDataset, deleteDataset, getDatasets, updateDataset } from "@/lib/api";
import { useDatasetStore } from "@/stores/datasetStore";
import { useCallback, useState } from "react";

type EditorTab = "source" | "features" | "split" | "preview";

interface TabConfig {
	key: EditorTab;
	label: string;
}

const TABS: TabConfig[] = [
	{ key: "source", label: "Source" },
	{ key: "features", label: "Features" },
	{ key: "split", label: "Split" },
	{ key: "preview", label: "Preview" },
];

export default function DatasetsPage() {
	const {
		selectedDatasetId,
		editingDataset,
		preview,
		isLoading,
		error,
		setDatasets,
		selectDataset,
		setEditingDataset,
		updateEditingField,
		setLoading,
		setError,
	} = useDatasetStore();

	const [activeTab, setActiveTab] = useState<EditorTab>("source");

	const reloadDatasets = useCallback(async () => {
		try {
			const data = await getDatasets();
			setDatasets(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to reload datasets");
		}
	}, [setDatasets, setError]);

	const handleNewDataset = useCallback(() => {
		selectDataset(null);
		setEditingDataset({
			name: "",
			description: "",
			source_type: "synthetic",
			source_config: {
				base_price: 50000,
				base_volume: 1000,
				num_candles: 5000,
				volatility: 0.02,
				drift: 0.0001,
			},
			features: [],
			split_config: { train_pct: 0.7, val_pct: 0.15, test_pct: 0.15 },
		});
		setActiveTab("source");
	}, [selectDataset, setEditingDataset]);

	const handleSave = useCallback(async () => {
		if (!editingDataset?.name) return;
		setLoading(true);
		setError(null);
		try {
			if (editingDataset.id) {
				await updateDataset(editingDataset.id, {
					name: editingDataset.name,
					description: editingDataset.description ?? "",
					source_type: editingDataset.source_type,
					source_config: editingDataset.source_config,
					features: editingDataset.features,
					split_config: editingDataset.split_config,
				});
			} else {
				await createDataset({
					name: editingDataset.name,
					description: editingDataset.description ?? "",
					source_type: editingDataset.source_type ?? "synthetic",
					source_config: editingDataset.source_config ?? {},
					features: editingDataset.features ?? [],
					split_config: editingDataset.split_config ?? {
						train_pct: 0.7,
						val_pct: 0.15,
						test_pct: 0.15,
					},
				});
			}
			await reloadDatasets();
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to save dataset");
		} finally {
			setLoading(false);
		}
	}, [editingDataset, reloadDatasets, setLoading, setError]);

	const handleDelete = useCallback(async () => {
		if (!selectedDatasetId) return;
		setLoading(true);
		setError(null);
		try {
			await deleteDataset(selectedDatasetId);
			selectDataset(null);
			setEditingDataset(null);
			await reloadDatasets();
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to delete dataset");
		} finally {
			setLoading(false);
		}
	}, [selectedDatasetId, selectDataset, setEditingDataset, reloadDatasets, setLoading, setError]);

	return (
		<div className="flex h-full flex-col">
			<div className="mb-4">
				<h1 className="text-xl font-semibold text-[var(--text-primary)]">Dataset Designer</h1>
			</div>

			{error && (
				<div className="mb-4 rounded-md border border-[var(--accent-red)] bg-[var(--bg-card)] p-3 text-sm text-[var(--accent-red)]">
					{error}
				</div>
			)}

			<div className="flex flex-1 gap-4 overflow-hidden">
				{/* Left sidebar - dataset list */}
				<div className="w-64 shrink-0 overflow-y-auto rounded-lg border border-[var(--border-color)] bg-[var(--bg-card)]">
					<DatasetList onNewDataset={handleNewDataset} />
				</div>

				{/* Right content area */}
				<div className="flex flex-1 flex-col gap-4 overflow-y-auto">
					{editingDataset ? (
						<>
							{/* Name / description header */}
							<Card>
								<CardHeader
									title="Dataset Info"
									action={
										selectedDatasetId ? (
											<button
												type="button"
												onClick={handleDelete}
												disabled={isLoading}
												className="text-xs text-[var(--accent-red)] hover:opacity-80 disabled:opacity-50"
											>
												Delete
											</button>
										) : undefined
									}
								/>
								<div className="space-y-3">
									<div>
										<label
											htmlFor="dataset-name"
											className="mb-1 block text-xs text-[var(--text-secondary)]"
										>
											Name
										</label>
										<input
											id="dataset-name"
											type="text"
											value={editingDataset.name ?? ""}
											onChange={(e) => updateEditingField("name", e.target.value)}
											className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-primary)] px-3 py-1.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent-blue)]"
											placeholder="Dataset name"
										/>
									</div>
									<div>
										<label
											htmlFor="dataset-description"
											className="mb-1 block text-xs text-[var(--text-secondary)]"
										>
											Description
										</label>
										<input
											id="dataset-description"
											type="text"
											value={editingDataset.description ?? ""}
											onChange={(e) => updateEditingField("description", e.target.value)}
											className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-primary)] px-3 py-1.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent-blue)]"
											placeholder="Describe this dataset configuration"
										/>
									</div>
								</div>
							</Card>

							{/* Tabs */}
							<div className="flex gap-1 border-b border-[var(--border-color)]">
								{TABS.map((tab) => (
									<button
										key={tab.key}
										type="button"
										onClick={() => setActiveTab(tab.key)}
										className={`px-4 py-2 text-sm font-medium transition-colors ${
											activeTab === tab.key
												? "border-b-2 border-[var(--accent-blue)] text-[var(--text-primary)]"
												: "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
										}`}
									>
										{tab.label}
									</button>
								))}
							</div>

							{/* Tab content */}
							<Card>
								<CardHeader title={TABS.find((t) => t.key === activeTab)?.label ?? "Source"} />
								{activeTab === "source" && <SourceConfig />}
								{activeTab === "features" && <FeatureBuilder />}
								{activeTab === "split" && <SplitSlider />}
								{activeTab === "preview" && (
									<div className="space-y-4">
										{preview?.ohlcv_sample && preview.ohlcv_sample.length > 0 && (
											<div className="h-48">
												<DataPreviewChart data={preview.ohlcv_sample} />
											</div>
										)}
										<DataPreview />
									</div>
								)}
							</Card>

							{/* Save action */}
							<div className="flex items-center gap-3">
								<button
									type="button"
									onClick={handleSave}
									disabled={isLoading || !editingDataset.name}
									className="rounded-md bg-[var(--accent-blue)] px-4 py-2 text-sm font-medium text-white transition-colors hover:opacity-90 disabled:opacity-50"
								>
									{isLoading
										? "Saving..."
										: editingDataset.id
											? "Update Dataset"
											: "Create Dataset"}
								</button>
								<button
									type="button"
									onClick={() => {
										selectDataset(null);
										setEditingDataset(null);
									}}
									className="rounded-md border border-[var(--border-color)] px-4 py-2 text-sm text-[var(--text-secondary)] hover:bg-[var(--bg-secondary)]"
								>
									Cancel
								</button>
							</div>
						</>
					) : (
						<div className="flex flex-1 items-center justify-center">
							<div className="text-center">
								<p className="mb-3 text-sm text-[var(--text-secondary)]">
									Select a dataset or create a new one to get started.
								</p>
								<button
									type="button"
									onClick={handleNewDataset}
									className="rounded-md bg-[var(--accent-green)] px-4 py-2 text-sm font-medium text-white hover:opacity-90"
								>
									+ New Dataset
								</button>
							</div>
						</div>
					)}
				</div>
			</div>
		</div>
	);
}
