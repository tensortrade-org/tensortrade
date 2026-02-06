import type {
	EpisodeMetrics,
	StepUpdate,
	TradeEvent,
	TrainingProgress,
	TrainingStatus,
	TrainingUpdate,
} from "@/lib/types";
import { create } from "zustand";

const MAX_STEPS = 1000;
const MAX_ITERATIONS = 500;
const MAX_TRADES = 500;
const MAX_EPISODES = 500;

interface TrainingState {
	status: TrainingStatus | null;
	currentIteration: number;
	iterations: TrainingUpdate[];
	steps: StepUpdate[];
	trades: TradeEvent[];
	episodes: EpisodeMetrics[];
	progress: TrainingProgress | null;
	isConnected: boolean;

	setStatus: (status: TrainingStatus) => void;
	addIteration: (update: TrainingUpdate) => void;
	addStep: (step: StepUpdate) => void;
	addTrade: (trade: TradeEvent) => void;
	addEpisode: (episode: EpisodeMetrics) => void;
	setProgress: (progress: TrainingProgress | null) => void;
	setConnected: (connected: boolean) => void;
	reset: () => void;
}

const initialState: Pick<
	TrainingState,
	| "status"
	| "currentIteration"
	| "iterations"
	| "steps"
	| "trades"
	| "episodes"
	| "progress"
	| "isConnected"
> = {
	status: null,
	currentIteration: 0,
	iterations: [],
	steps: [],
	trades: [],
	episodes: [],
	progress: null,
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

	addEpisode: (episode: EpisodeMetrics) =>
		set((state) => {
			const next = [...state.episodes, episode];
			return {
				episodes: next.length > MAX_EPISODES ? next.slice(-MAX_EPISODES) : next,
			};
		}),

	setProgress: (progress: TrainingProgress | null) => set({ progress }),

	setConnected: (connected: boolean) => set({ isConnected: connected }),

	reset: () => set(initialState),
}));
