"use client";

import { LoadingState } from "@/components/common/Spinner";
import { FeatureCard } from "@/components/datasets/FeatureCard";
import { useApi } from "@/hooks/useApi";
import { getFeatureCatalog } from "@/lib/api";
import type { FeatureCatalogEntry, FeatureSpec } from "@/lib/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useCallback, useEffect, useMemo } from "react";

interface FeatureParamState {
	[featureType: string]: Record<string, string | number | boolean>;
}

export function FeatureBuilder() {
	const { editingDataset, addFeature, removeFeature, setFeatureCatalog } = useDatasetStore();

	const catalogFetcher = useCallback(() => getFeatureCatalog(), []);
	const { data: catalog, loading, error } = useApi<FeatureCatalogEntry[]>(catalogFetcher, []);

	// Sync catalog to store when loaded
	useEffect(() => {
		if (catalog && catalog.length > 0) {
			setFeatureCatalog(catalog);
		}
	}, [catalog, setFeatureCatalog]);

	const enabledFeatures = useMemo(() => {
		const features = editingDataset?.features ?? [];
		const map: Record<string, FeatureSpec> = {};
		for (const f of features) {
			map[f.type] = f;
		}
		return map;
	}, [editingDataset?.features]);

	const paramState = useMemo<FeatureParamState>(() => {
		const state: FeatureParamState = {};
		for (const [type, spec] of Object.entries(enabledFeatures)) {
			const params: Record<string, string | number | boolean> = {};
			for (const [key, val] of Object.entries(spec)) {
				if (key !== "type" && val !== undefined) {
					params[key] = val as string | number | boolean;
				}
			}
			state[type] = params;
		}
		return state;
	}, [enabledFeatures]);

	const handleToggle = useCallback(
		(featureType: string, enabled: boolean) => {
			if (enabled) {
				const spec: FeatureSpec = { type: featureType };
				const entry = catalog?.find((c) => c.type === featureType);
				if (entry) {
					for (const param of entry.params) {
						spec[param.name] = param.default;
					}
				}
				addFeature(spec);
			} else {
				const features = editingDataset?.features ?? [];
				const idx = features.findIndex((f) => f.type === featureType);
				if (idx >= 0) {
					removeFeature(idx);
				}
			}
		},
		[catalog, editingDataset?.features, addFeature, removeFeature],
	);

	const handleParamChange = useCallback(
		(featureType: string, paramName: string, value: string | number | boolean) => {
			const features = editingDataset?.features ?? [];
			const idx = features.findIndex((f) => f.type === featureType);
			if (idx < 0) return;

			// Remove and re-add with updated params
			removeFeature(idx);
			const updatedSpec: FeatureSpec = {
				...features[idx],
				[paramName]: value,
			};
			addFeature(updatedSpec);
		},
		[editingDataset?.features, addFeature, removeFeature],
	);

	if (loading) {
		return <LoadingState message="Loading feature catalog..." />;
	}

	if (error) {
		return (
			<div className="py-6 text-center text-sm text-[var(--accent-red)]">
				Failed to load feature catalog: {error.message}
			</div>
		);
	}

	if (!catalog || catalog.length === 0) {
		return (
			<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
				No features available in the catalog.
			</div>
		);
	}

	const enabledCount = Object.keys(enabledFeatures).length;

	return (
		<div className="space-y-4">
			<div className="flex items-center justify-between">
				<p className="text-sm text-[var(--text-secondary)]">
					Toggle features to include in this dataset.
				</p>
				<span className="rounded-full bg-[var(--accent-blue)] px-2 py-0.5 text-xs font-medium text-white">
					{enabledCount} selected
				</span>
			</div>

			<div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
				{catalog.map((entry) => (
					<FeatureCard
						key={entry.type}
						entry={entry}
						enabled={entry.type in enabledFeatures}
						paramValues={paramState[entry.type] ?? {}}
						onToggle={handleToggle}
						onParamChange={handleParamChange}
					/>
				))}
			</div>
		</div>
	);
}
