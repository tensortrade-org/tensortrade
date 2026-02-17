import type { ExperimentDetail, ExperimentSummary } from "@/lib/types";
import { create } from "zustand";

interface ExperimentState {
	experiments: ExperimentSummary[];
	selectedIds: string[];
	currentDetail: ExperimentDetail | null;

	setExperiments: (experiments: ExperimentSummary[]) => void;
	setCurrentDetail: (detail: ExperimentDetail | null) => void;
	toggleSelected: (id: string) => void;
	clearSelected: () => void;
}

export const useExperimentStore = create<ExperimentState>((set) => ({
	experiments: [],
	selectedIds: [],
	currentDetail: null,

	setExperiments: (experiments: ExperimentSummary[]) => set({ experiments }),

	setCurrentDetail: (detail: ExperimentDetail | null) => set({ currentDetail: detail }),

	toggleSelected: (id: string) =>
		set((state) => {
			const exists = state.selectedIds.includes(id);
			return {
				selectedIds: exists
					? state.selectedIds.filter((selectedId) => selectedId !== id)
					: [...state.selectedIds, id],
			};
		}),

	clearSelected: () => set({ selectedIds: [] }),
}));
