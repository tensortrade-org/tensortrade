import type { StepUpdate, TradeEvent, TrainingStatus, TrainingUpdate } from "@/lib/types";
import { create } from "zustand";

const MAX_STEPS = 1000;
const MAX_ITERATIONS = 500;
const MAX_TRADES = 500;

interface TrainingState {
	status: TrainingStatus | null;
	currentIteration: number;
	iterations: TrainingUpdate[];
	steps: StepUpdate[];
	trades: TradeEvent[];
	isConnected: boolean;

	setStatus: (status: TrainingStatus) => void;
	addIteration: (update: TrainingUpdate) => void;
	addStep: (step: StepUpdate) => void;
	addTrade: (trade: TradeEvent) => void;
	setConnected: (connected: boolean) => void;
	reset: () => void;
}

const initialState: Pick<
	TrainingState,
	"status" | "currentIteration" | "iterations" | "steps" | "trades" | "isConnected"
> = {
	status: null,
	currentIteration: 0,
	iterations: [],
	steps: [],
	trades: [],
	isConnected: false,
};

export const useTrainingStore = create<TrainingState>((set) => ({
	...initialState,

	setStatus: (status: TrainingStatus) =>
		set({
			status,
			currentIteration: status.current_iteration,
		}),

	addIteration: (update: TrainingUpdate) =>
		set((state) => {
			const next = [...state.iterations, update];
			return {
				iterations: next.length > MAX_ITERATIONS ? next.slice(-MAX_ITERATIONS) : next,
				currentIteration: update.iteration,
			};
		}),

	addStep: (step: StepUpdate) =>
		set((state) => {
			const next = [...state.steps, step];
			return {
				steps: next.length > MAX_STEPS ? next.slice(-MAX_STEPS) : next,
			};
		}),

	addTrade: (trade: TradeEvent) =>
		set((state) => {
			const next = [...state.trades, trade];
			return {
				trades: next.length > MAX_TRADES ? next.slice(-MAX_TRADES) : next,
			};
		}),

	setConnected: (connected: boolean) => set({ isConnected: connected }),

	reset: () => set(initialState),
}));
