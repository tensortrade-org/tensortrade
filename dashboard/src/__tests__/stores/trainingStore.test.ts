import type {
	EpisodeMetrics,
	StepUpdate,
	TradeEvent,
	TrainingProgress,
	TrainingStatus,
	TrainingUpdate,
} from "@/lib/types";
import { useTrainingStore } from "@/stores/trainingStore";
import { beforeEach, describe, expect, it } from "vitest";

const makeTrainingUpdate = (iteration: number): TrainingUpdate => ({
	type: "training_update",
	iteration,
	episode_return_mean: 100 + iteration,
	pnl_mean: 50 + iteration,
	pnl_pct_mean: 0.5 + iteration * 0.01,
	net_worth_mean: 10000 + iteration * 10,
	trade_count_mean: 5,
	hold_count_mean: 10,
});

const makeEpisode = (episode: number): EpisodeMetrics => ({
	type: "episode_metrics",
	episode,
	reward_total: 100 + episode,
	pnl: 50 + episode,
	pnl_pct: 0.5,
	net_worth: 10050,
	trade_count: 5,
	hold_count: 10,
	buy_count: 3,
	sell_count: 2,
	action_distribution: { buy: 0.3, sell: 0.2, hold: 0.5 },
});

const makeStep = (step: number): StepUpdate => ({
	type: "step_update",
	step,
	price: 100 + step,
	open: 100,
	high: 105,
	low: 95,
	close: 100 + step,
	volume: 1000,
	net_worth: 10000 + step,
});

const makeTrade = (step: number): TradeEvent => ({
	type: "trade",
	step,
	side: step % 2 === 0 ? "buy" : "sell",
	price: 100 + step,
	size: 0.5,
	commission: 0.01,
});

const makeProgress = (iteration: number): TrainingProgress => ({
	type: "training_progress",
	experiment_id: "exp-1",
	iteration,
	total_iterations: 100,
	elapsed_seconds: iteration * 10,
	eta_seconds: (100 - iteration) * 10,
});

const makeStatus = (): TrainingStatus => ({
	type: "status",
	is_training: true,
	is_paused: false,
	experiment_id: "exp-1",
	current_iteration: 5,
	dashboard_clients: 1,
	training_producers: 1,
});

describe("useTrainingStore", () => {
	beforeEach(() => {
		useTrainingStore.getState().reset();
	});

	describe("initial state", () => {
		it("starts empty", () => {
			const state = useTrainingStore.getState();
			expect(state.status).toBeNull();
			expect(state.currentIteration).toBe(0);
			expect(state.iterations).toEqual([]);
			expect(state.steps).toEqual([]);
			expect(state.trades).toEqual([]);
			expect(state.episodes).toEqual([]);
			expect(state.progress).toBeNull();
			expect(state.isConnected).toBe(false);
		});
	});

	describe("setStatus", () => {
		it("sets status and currentIteration", () => {
			const status = makeStatus();
			useTrainingStore.getState().setStatus(status);

			const state = useTrainingStore.getState();
			expect(state.status).toEqual(status);
			expect(state.currentIteration).toBe(5);
		});
	});

	describe("addIteration", () => {
		it("accumulates iterations", () => {
			useTrainingStore.getState().addIteration(makeTrainingUpdate(1));
			useTrainingStore.getState().addIteration(makeTrainingUpdate(2));
			useTrainingStore.getState().addIteration(makeTrainingUpdate(3));

			const state = useTrainingStore.getState();
			expect(state.iterations).toHaveLength(3);
			expect(state.currentIteration).toBe(3);
		});

		it("caps at MAX_ITERATIONS (500)", () => {
			for (let i = 0; i < 510; i++) {
				useTrainingStore.getState().addIteration(makeTrainingUpdate(i));
			}

			const state = useTrainingStore.getState();
			expect(state.iterations.length).toBeLessThanOrEqual(500);
			// Should retain the most recent entries
			expect(state.iterations[state.iterations.length - 1].iteration).toBe(509);
		});
	});

	describe("addStep", () => {
		it("accumulates steps", () => {
			useTrainingStore.getState().addStep(makeStep(1));
			useTrainingStore.getState().addStep(makeStep(2));
			expect(useTrainingStore.getState().steps).toHaveLength(2);
		});

		it("caps at MAX_STEPS (1000)", () => {
			for (let i = 0; i < 1010; i++) {
				useTrainingStore.getState().addStep(makeStep(i));
			}
			expect(useTrainingStore.getState().steps.length).toBeLessThanOrEqual(1000);
		});
	});

	describe("addTrade", () => {
		it("accumulates trades", () => {
			useTrainingStore.getState().addTrade(makeTrade(1));
			useTrainingStore.getState().addTrade(makeTrade(2));
			expect(useTrainingStore.getState().trades).toHaveLength(2);
		});

		it("caps at MAX_TRADES (500)", () => {
			for (let i = 0; i < 510; i++) {
				useTrainingStore.getState().addTrade(makeTrade(i));
			}
			expect(useTrainingStore.getState().trades.length).toBeLessThanOrEqual(500);
		});
	});

	describe("addEpisode", () => {
		it("accumulates episodes", () => {
			useTrainingStore.getState().addEpisode(makeEpisode(1));
			useTrainingStore.getState().addEpisode(makeEpisode(2));
			useTrainingStore.getState().addEpisode(makeEpisode(3));

			expect(useTrainingStore.getState().episodes).toHaveLength(3);
		});

		it("caps at MAX_EPISODES (500)", () => {
			for (let i = 0; i < 510; i++) {
				useTrainingStore.getState().addEpisode(makeEpisode(i));
			}

			const state = useTrainingStore.getState();
			expect(state.episodes.length).toBeLessThanOrEqual(500);
			// Should retain the most recent entries
			expect(state.episodes[state.episodes.length - 1].episode).toBe(509);
		});
	});

	describe("setProgress", () => {
		it("updates progress", () => {
			const progress = makeProgress(10);
			useTrainingStore.getState().setProgress(progress);
			expect(useTrainingStore.getState().progress).toEqual(progress);
		});

		it("clears progress with null", () => {
			useTrainingStore.getState().setProgress(makeProgress(5));
			useTrainingStore.getState().setProgress(null);
			expect(useTrainingStore.getState().progress).toBeNull();
		});
	});

	describe("setConnected", () => {
		it("sets connection state", () => {
			useTrainingStore.getState().setConnected(true);
			expect(useTrainingStore.getState().isConnected).toBe(true);

			useTrainingStore.getState().setConnected(false);
			expect(useTrainingStore.getState().isConnected).toBe(false);
		});
	});

	describe("reset", () => {
		it("clears episodes and progress and all state", () => {
			useTrainingStore.getState().setStatus(makeStatus());
			useTrainingStore.getState().addIteration(makeTrainingUpdate(1));
			useTrainingStore.getState().addStep(makeStep(1));
			useTrainingStore.getState().addTrade(makeTrade(1));
			useTrainingStore.getState().addEpisode(makeEpisode(1));
			useTrainingStore.getState().setProgress(makeProgress(1));
			useTrainingStore.getState().setConnected(true);

			useTrainingStore.getState().reset();

			const state = useTrainingStore.getState();
			expect(state.status).toBeNull();
			expect(state.currentIteration).toBe(0);
			expect(state.iterations).toEqual([]);
			expect(state.steps).toEqual([]);
			expect(state.trades).toEqual([]);
			expect(state.episodes).toEqual([]);
			expect(state.progress).toBeNull();
			expect(state.isConnected).toBe(false);
		});
	});
});
