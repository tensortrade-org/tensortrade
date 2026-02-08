import type { EpisodeSummary, InferenceStatus, StepUpdate, TradeEvent } from "@/lib/types";
import { create } from "zustand";

interface InferenceState {
	status: InferenceStatus["state"];
	experimentId: string | null;
	datasetName: string | null;
	steps: StepUpdate[];
	trades: TradeEvent[];
	episodeSummary: EpisodeSummary | null;
	totalSteps: number;
	currentStep: number;
	error: string | null;

	setStatus: (msg: InferenceStatus) => void;
	addStep: (step: StepUpdate) => void;
	addTrade: (trade: TradeEvent) => void;
	reset: () => void;
}

const initialState: Pick<
	InferenceState,
	| "status"
	| "experimentId"
	| "datasetName"
	| "steps"
	| "trades"
	| "episodeSummary"
	| "totalSteps"
	| "currentStep"
	| "error"
> = {
	status: "idle",
	experimentId: null,
	datasetName: null,
	steps: [],
	trades: [],
	episodeSummary: null,
	totalSteps: 0,
	currentStep: 0,
	error: null,
};

export const useInferenceStore = create<InferenceState>((set) => ({
	...initialState,

	setStatus: (msg: InferenceStatus) =>
		set({
			status: msg.state,
			experimentId: msg.experiment_id,
			datasetName: msg.dataset_name ?? null,
			totalSteps: msg.total_steps,
			currentStep: msg.current_step,
			episodeSummary: msg.episode_summary ?? null,
			error: msg.error ?? null,
		}),

	addStep: (step: StepUpdate) =>
		set((state) => ({
			steps: [...state.steps, step],
			currentStep: step.step,
		})),

	addTrade: (trade: TradeEvent) =>
		set((state) => ({
			trades: [...state.trades, trade],
		})),

	reset: () => set(initialState),
}));
