import type { TrainingConfig } from "@/lib/types";
import { create } from "zustand";

type LaunchStep = "name" | "dataset" | "hyperparams" | "environment" | "review";

interface LaunchState {
	currentStep: LaunchStep;
	name: string;
	tags: string[];
	datasetId: string | null;
	hpPackId: string | null;
	overrides: Partial<TrainingConfig>;
	isLaunching: boolean;
	launchError: string | null;
	launchedExperimentId: string | null;

	setStep: (step: LaunchStep) => void;
	setName: (name: string) => void;
	setTags: (tags: string[]) => void;
	setDatasetId: (id: string | null) => void;
	setHpPackId: (id: string | null) => void;
	setOverride: <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => void;
	clearOverrides: () => void;
	setLaunching: (launching: boolean) => void;
	setLaunchError: (error: string | null) => void;
	setLaunchedExperimentId: (id: string | null) => void;
	reset: () => void;
}

const STEP_ORDER: LaunchStep[] = ["name", "dataset", "hyperparams", "environment", "review"];

const initialState: Pick<
	LaunchState,
	| "currentStep"
	| "name"
	| "tags"
	| "datasetId"
	| "hpPackId"
	| "overrides"
	| "isLaunching"
	| "launchError"
	| "launchedExperimentId"
> = {
	currentStep: "name",
	name: "",
	tags: [],
	datasetId: null,
	hpPackId: null,
	overrides: {},
	isLaunching: false,
	launchError: null,
	launchedExperimentId: null,
};

export const useLaunchStore = create<LaunchState>((set) => ({
	...initialState,

	setStep: (step) => set({ currentStep: step }),
	setName: (name) => set({ name }),
	setTags: (tags) => set({ tags }),
	setDatasetId: (id) => set({ datasetId: id }),
	setHpPackId: (id) => set({ hpPackId: id }),

	setOverride: (key, value) =>
		set((state) => ({
			overrides: { ...state.overrides, [key]: value },
		})),

	clearOverrides: () => set({ overrides: {} }),
	setLaunching: (launching) => set({ isLaunching: launching }),
	setLaunchError: (error) => set({ launchError: error }),
	setLaunchedExperimentId: (id) => set({ launchedExperimentId: id }),
	reset: () => set(initialState),
}));

export { STEP_ORDER };
export type { LaunchStep };
