import type { HyperparameterPack, TrainingConfig } from "@/lib/types";
import { create } from "zustand";

interface HyperparamState {
	packs: HyperparameterPack[];
	selectedPackId: string | null;
	editingPack: HyperparameterPack | null;
	comparePackIds: [string, string] | null;
	isLoading: boolean;
	error: string | null;

	setPacks: (packs: HyperparameterPack[]) => void;
	selectPack: (id: string | null) => void;
	setEditingPack: (pack: HyperparameterPack | null) => void;
	updateEditingConfig: (
		key: keyof TrainingConfig,
		value: TrainingConfig[keyof TrainingConfig],
	) => void;
	setComparePackIds: (ids: [string, string] | null) => void;
	setLoading: (loading: boolean) => void;
	setError: (error: string | null) => void;
	reset: () => void;
}

const initialState: Pick<
	HyperparamState,
	"packs" | "selectedPackId" | "editingPack" | "comparePackIds" | "isLoading" | "error"
> = {
	packs: [],
	selectedPackId: null,
	editingPack: null,
	comparePackIds: null,
	isLoading: false,
	error: null,
};

export const useHyperparamStore = create<HyperparamState>((set) => ({
	...initialState,

	setPacks: (packs) => set({ packs }),

	selectPack: (id) =>
		set((state) => {
			const pack = id ? (state.packs.find((p) => p.id === id) ?? null) : null;
			return {
				selectedPackId: id,
				editingPack: pack
					? { ...pack, config: { ...pack.config, model: { ...pack.config.model } } }
					: null,
			};
		}),

	setEditingPack: (pack) => set({ editingPack: pack }),

	updateEditingConfig: (key, value) =>
		set((state) => {
			if (!state.editingPack) return state;
			return {
				editingPack: {
					...state.editingPack,
					config: { ...state.editingPack.config, [key]: value },
				},
			};
		}),

	setComparePackIds: (ids) => set({ comparePackIds: ids }),
	setLoading: (loading) => set({ isLoading: loading }),
	setError: (error) => set({ error }),
	reset: () => set(initialState),
}));
