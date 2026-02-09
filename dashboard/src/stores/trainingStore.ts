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

interface CompletedExperiment {
	experimentId: string;
	status: string;
	completedAt: string;
}

interface TrainingState {
	status: TrainingStatus | null;
	currentIteration: number;
	iterations: TrainingUpdate[];
	steps: StepUpdate[];
	trades: TradeEvent[];
	episodes: EpisodeMetrics[];
	progress: TrainingProgress | null;
	isConnected: boolean;
	isWarmingUp: boolean;
	pendingExperimentId: string | null;
	completedExperiment: CompletedExperiment | null;

	setStatus: (status: TrainingStatus) => void;
	addIteration: (update: TrainingUpdate) => void;
	addStep: (step: StepUpdate) => void;
	addTrade: (trade: TradeEvent) => void;
	addEpisode: (episode: EpisodeMetrics) => void;
	setProgress: (progress: TrainingProgress | null) => void;
	setConnected: (connected: boolean) => void;
	startWarmingUp: (experimentId: string) => void;
	clearWarmingUp: () => void;
	markCompleted: (experimentId: string, status: string) => void;
	dismissCompleted: () => void;
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
	| "isWarmingUp"
	| "pendingExperimentId"
	| "completedExperiment"
> = {
	status: null,
	currentIteration: 0,
	iterations: [],
	steps: [],
	trades: [],
	episodes: [],
	progress: null,
	isConnected: false,
	isWarmingUp: false,
	pendingExperimentId: null,
	completedExperiment: null,
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
			const current = state.iterations;
			let next: TrainingUpdate[];
			if (current.length > 0 && current[current.length - 1].iteration === update.iteration) {
				next = [...current.slice(0, -1), update];
			} else {
				next = [...current, update];
			}
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
			const current = state.episodes;
			let next: EpisodeMetrics[];
			if (current.length > 0 && current[current.length - 1].episode === episode.episode) {
				next = [...current.slice(0, -1), episode];
			} else {
				next = [...current, episode];
			}
			return {
				episodes: next.length > MAX_EPISODES ? next.slice(-MAX_EPISODES) : next,
			};
		}),

	setProgress: (progress: TrainingProgress | null) => set({ progress }),

	setConnected: (connected: boolean) => set({ isConnected: connected }),

	startWarmingUp: (experimentId: string) =>
		set({ isWarmingUp: true, pendingExperimentId: experimentId }),

	clearWarmingUp: () => set({ isWarmingUp: false, pendingExperimentId: null }),

	markCompleted: (experimentId: string, status: string) =>
		set({
			completedExperiment: {
				experimentId,
				status,
				completedAt: new Date().toISOString(),
			},
			isWarmingUp: false,
			pendingExperimentId: null,
		}),

	dismissCompleted: () => set({ completedExperiment: null }),

	reset: () => set(initialState),
}));
