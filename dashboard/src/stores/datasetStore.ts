import type {
	DatasetConfig,
	DatasetPreview,
	FeatureCatalogEntry,
	FeatureSpec,
	SplitConfig,
} from "@/lib/types";
import { create } from "zustand";

interface DatasetState {
	datasets: DatasetConfig[];
	selectedDatasetId: string | null;
	editingDataset: Partial<DatasetConfig> | null;
	preview: DatasetPreview | null;
	featureCatalog: FeatureCatalogEntry[];
	isLoading: boolean;
	error: string | null;

	setDatasets: (datasets: DatasetConfig[]) => void;
	selectDataset: (id: string | null) => void;
	setEditingDataset: (dataset: Partial<DatasetConfig> | null) => void;
	updateEditingField: <K extends keyof DatasetConfig>(key: K, value: DatasetConfig[K]) => void;
	setPreview: (preview: DatasetPreview | null) => void;
	setFeatureCatalog: (catalog: FeatureCatalogEntry[]) => void;
	addFeature: (spec: FeatureSpec) => void;
	removeFeature: (index: number) => void;
	updateSplit: (split: SplitConfig) => void;
	setLoading: (loading: boolean) => void;
	setError: (error: string | null) => void;
	reset: () => void;
}

const initialState: Pick<
	DatasetState,
	| "datasets"
	| "selectedDatasetId"
	| "editingDataset"
	| "preview"
	| "featureCatalog"
	| "isLoading"
	| "error"
> = {
	datasets: [],
	selectedDatasetId: null,
	editingDataset: null,
	preview: null,
	featureCatalog: [],
	isLoading: false,
	error: null,
};

export const useDatasetStore = create<DatasetState>((set) => ({
	...initialState,

	setDatasets: (datasets) => set({ datasets }),

	selectDataset: (id) =>
		set((state) => {
			const dataset = id ? (state.datasets.find((d) => d.id === id) ?? null) : null;
			return { selectedDatasetId: id, editingDataset: dataset ? { ...dataset } : null };
		}),

	setEditingDataset: (dataset) => set({ editingDataset: dataset }),

	updateEditingField: (key, value) =>
		set((state) => {
			if (!state.editingDataset) return state;
			return { editingDataset: { ...state.editingDataset, [key]: value } };
		}),

	setPreview: (preview) => set({ preview }),
	setFeatureCatalog: (catalog) => set({ featureCatalog: catalog }),

	addFeature: (spec) =>
		set((state) => {
			if (!state.editingDataset) return state;
			const features = [...(state.editingDataset.features ?? []), spec];
			return { editingDataset: { ...state.editingDataset, features } };
		}),

	removeFeature: (index) =>
		set((state) => {
			if (!state.editingDataset?.features) return state;
			const features = state.editingDataset.features.filter((_, i) => i !== index);
			return { editingDataset: { ...state.editingDataset, features } };
		}),

	updateSplit: (split) =>
		set((state) => {
			if (!state.editingDataset) return state;
			return { editingDataset: { ...state.editingDataset, split_config: split } };
		}),

	setLoading: (loading) => set({ isLoading: loading }),
	setError: (error) => set({ error }),
	reset: () => set(initialState),
}));
