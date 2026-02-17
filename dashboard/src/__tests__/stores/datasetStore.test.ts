import type { DatasetConfig, DatasetPreview, FeatureCatalogEntry, FeatureSpec } from "@/lib/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { beforeEach, describe, expect, it } from "vitest";

const makeDataset = (overrides?: Partial<DatasetConfig>): DatasetConfig => ({
	id: "ds-1",
	name: "Test Dataset",
	description: "A test dataset",
	source_type: "synthetic",
	source_config: { base_price: 50000 },
	features: [{ type: "rsi", period: 14 }],
	split_config: { train_pct: 0.7, val_pct: 0.15, test_pct: 0.15 },
	created_at: "2024-01-01T00:00:00",
	updated_at: "2024-01-01T00:00:00",
	...overrides,
});

const makePreview = (): DatasetPreview => ({
	rows: 5000,
	columns: ["open", "high", "low", "close", "volume", "rsi"],
	date_range: { start: "2020-01-01", end: "2024-01-01" },
	ohlcv_sample: [],
	feature_stats: {
		rsi: { mean: 0.02, std: 0.5, min: -1, max: 1 },
	},
});

const makeCatalogEntry = (): FeatureCatalogEntry => ({
	type: "rsi",
	name: "RSI",
	description: "Relative Strength Index",
	params: [
		{
			name: "period",
			type: "int",
			default: 14,
			min: 2,
			max: 100,
			description: "RSI period",
		},
	],
});

describe("useDatasetStore", () => {
	beforeEach(() => {
		useDatasetStore.getState().reset();
	});

	describe("initial state", () => {
		it("starts empty", () => {
			const state = useDatasetStore.getState();
			expect(state.datasets).toEqual([]);
			expect(state.selectedDatasetId).toBeNull();
			expect(state.editingDataset).toBeNull();
			expect(state.preview).toBeNull();
			expect(state.featureCatalog).toEqual([]);
			expect(state.isLoading).toBe(false);
			expect(state.error).toBeNull();
		});
	});

	describe("setDatasets", () => {
		it("sets datasets list", () => {
			const datasets = [makeDataset(), makeDataset({ id: "ds-2", name: "DS 2" })];
			useDatasetStore.getState().setDatasets(datasets);
			expect(useDatasetStore.getState().datasets).toHaveLength(2);
		});
	});

	describe("selectDataset", () => {
		it("sets selectedDatasetId and editingDataset", () => {
			useDatasetStore.getState().setDatasets([makeDataset()]);
			useDatasetStore.getState().selectDataset("ds-1");

			const state = useDatasetStore.getState();
			expect(state.selectedDatasetId).toBe("ds-1");
			expect(state.editingDataset).not.toBeNull();
			expect(state.editingDataset?.id).toBe("ds-1");
		});

		it("clears selection with null", () => {
			useDatasetStore.getState().setDatasets([makeDataset()]);
			useDatasetStore.getState().selectDataset("ds-1");
			useDatasetStore.getState().selectDataset(null);

			const state = useDatasetStore.getState();
			expect(state.selectedDatasetId).toBeNull();
			expect(state.editingDataset).toBeNull();
		});

		it("sets editingDataset null for non-existent id", () => {
			useDatasetStore.getState().setDatasets([makeDataset()]);
			useDatasetStore.getState().selectDataset("nonexistent");
			expect(useDatasetStore.getState().editingDataset).toBeNull();
		});
	});

	describe("setPreview", () => {
		it("sets preview data", () => {
			const preview = makePreview();
			useDatasetStore.getState().setPreview(preview);
			expect(useDatasetStore.getState().preview).toEqual(preview);
		});

		it("clears preview with null", () => {
			useDatasetStore.getState().setPreview(makePreview());
			useDatasetStore.getState().setPreview(null);
			expect(useDatasetStore.getState().preview).toBeNull();
		});
	});

	describe("setFeatureCatalog", () => {
		it("sets feature catalog", () => {
			const catalog = [makeCatalogEntry()];
			useDatasetStore.getState().setFeatureCatalog(catalog);
			expect(useDatasetStore.getState().featureCatalog).toEqual(catalog);
			expect(useDatasetStore.getState().featureCatalog).toHaveLength(1);
		});
	});

	describe("addFeature", () => {
		it("appends feature to editingDataset", () => {
			useDatasetStore.getState().setDatasets([makeDataset({ features: [] })]);
			useDatasetStore.getState().selectDataset("ds-1");

			const spec: FeatureSpec = { type: "volatility", period: 24 };
			useDatasetStore.getState().addFeature(spec);

			const editing = useDatasetStore.getState().editingDataset;
			expect(editing?.features).toHaveLength(1);
			expect(editing?.features?.[0].type).toBe("volatility");
		});

		it("does nothing when no editingDataset", () => {
			useDatasetStore.getState().addFeature({ type: "rsi" });
			expect(useDatasetStore.getState().editingDataset).toBeNull();
		});
	});

	describe("removeFeature", () => {
		it("removes feature by index", () => {
			const ds = makeDataset({
				features: [
					{ type: "rsi", period: 14 },
					{ type: "volatility", period: 24 },
				],
			});
			useDatasetStore.getState().setDatasets([ds]);
			useDatasetStore.getState().selectDataset("ds-1");
			useDatasetStore.getState().removeFeature(0);

			const editing = useDatasetStore.getState().editingDataset;
			expect(editing?.features).toHaveLength(1);
			expect(editing?.features?.[0].type).toBe("volatility");
		});
	});

	describe("updateSplit", () => {
		it("updates split_config", () => {
			useDatasetStore.getState().setDatasets([makeDataset()]);
			useDatasetStore.getState().selectDataset("ds-1");

			useDatasetStore.getState().updateSplit({
				train_pct: 0.8,
				val_pct: 0.1,
				test_pct: 0.1,
			});

			const editing = useDatasetStore.getState().editingDataset;
			expect(editing?.split_config?.train_pct).toBe(0.8);
		});
	});

	describe("reset", () => {
		it("resets to initial state", () => {
			useDatasetStore.getState().setDatasets([makeDataset()]);
			useDatasetStore.getState().selectDataset("ds-1");
			useDatasetStore.getState().setPreview(makePreview());
			useDatasetStore.getState().setLoading(true);
			useDatasetStore.getState().setError("Error");

			useDatasetStore.getState().reset();

			const state = useDatasetStore.getState();
			expect(state.datasets).toEqual([]);
			expect(state.selectedDatasetId).toBeNull();
			expect(state.editingDataset).toBeNull();
			expect(state.preview).toBeNull();
			expect(state.featureCatalog).toEqual([]);
			expect(state.isLoading).toBe(false);
			expect(state.error).toBeNull();
		});
	});
});
