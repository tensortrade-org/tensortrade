import type { HyperparameterPack, TrainingConfig } from "@/lib/types";
import { useHyperparamStore } from "@/stores/hyperparamStore";
import { beforeEach, describe, expect, it } from "vitest";

const makePack = (overrides?: Partial<HyperparameterPack>): HyperparameterPack => ({
	id: "pack-1",
	name: "Test Pack",
	description: "A test pack",
	config: {
		algorithm: "PPO",
		learning_rate: 5e-5,
		gamma: 0.99,
		lambda_: 0.9,
		clip_param: 0.1,
		entropy_coeff: 0.1,
		vf_loss_coeff: 0.5,
		num_sgd_iter: 5,
		sgd_minibatch_size: 128,
		train_batch_size: 4000,
		num_rollout_workers: 4,
		rollout_fragment_length: 200,
		model: { fcnet_hiddens: [32, 32], fcnet_activation: "tanh" },
		action_scheme: "BSH",
		reward_scheme: "PBR",
		reward_params: {},
		commission: 0.0005,
		initial_cash: 10000,
		window_size: 10,
		max_allowed_loss: 0.3,
		max_episode_steps: null,
		num_iterations: 100,
	},
	created_at: "2024-01-01T00:00:00",
	updated_at: "2024-01-01T00:00:00",
	...overrides,
});

describe("useHyperparamStore", () => {
	beforeEach(() => {
		useHyperparamStore.getState().reset();
	});

	describe("initial state", () => {
		it("starts with empty packs", () => {
			const state = useHyperparamStore.getState();
			expect(state.packs).toEqual([]);
			expect(state.selectedPackId).toBeNull();
			expect(state.editingPack).toBeNull();
			expect(state.comparePackIds).toBeNull();
			expect(state.isLoading).toBe(false);
			expect(state.error).toBeNull();
		});
	});

	describe("setPacks", () => {
		it("updates packs list", () => {
			const packs = [makePack(), makePack({ id: "pack-2", name: "Pack 2" })];
			useHyperparamStore.getState().setPacks(packs);
			expect(useHyperparamStore.getState().packs).toEqual(packs);
			expect(useHyperparamStore.getState().packs).toHaveLength(2);
		});

		it("replaces existing packs", () => {
			useHyperparamStore.getState().setPacks([makePack()]);
			const newPacks = [makePack({ id: "pack-new", name: "New Pack" })];
			useHyperparamStore.getState().setPacks(newPacks);
			expect(useHyperparamStore.getState().packs).toHaveLength(1);
			expect(useHyperparamStore.getState().packs[0].id).toBe("pack-new");
		});
	});

	describe("selectPack", () => {
		it("sets selectedPackId and editingPack", () => {
			const pack = makePack();
			useHyperparamStore.getState().setPacks([pack]);
			useHyperparamStore.getState().selectPack("pack-1");

			const state = useHyperparamStore.getState();
			expect(state.selectedPackId).toBe("pack-1");
			expect(state.editingPack).not.toBeNull();
			expect(state.editingPack?.id).toBe("pack-1");
		});

		it("clears editingPack when selecting null", () => {
			useHyperparamStore.getState().setPacks([makePack()]);
			useHyperparamStore.getState().selectPack("pack-1");
			useHyperparamStore.getState().selectPack(null);

			const state = useHyperparamStore.getState();
			expect(state.selectedPackId).toBeNull();
			expect(state.editingPack).toBeNull();
		});

		it("sets editingPack to null for non-existent id", () => {
			useHyperparamStore.getState().setPacks([makePack()]);
			useHyperparamStore.getState().selectPack("nonexistent");

			const state = useHyperparamStore.getState();
			expect(state.selectedPackId).toBe("nonexistent");
			expect(state.editingPack).toBeNull();
		});
	});

	describe("updateEditingConfig", () => {
		it("updates a config field on editingPack", () => {
			useHyperparamStore.getState().setPacks([makePack()]);
			useHyperparamStore.getState().selectPack("pack-1");
			useHyperparamStore.getState().updateEditingConfig("learning_rate", 1e-3);

			const state = useHyperparamStore.getState();
			expect(state.editingPack?.config.learning_rate).toBe(1e-3);
		});

		it("does nothing when no editingPack", () => {
			useHyperparamStore.getState().updateEditingConfig("learning_rate", 1e-3);
			expect(useHyperparamStore.getState().editingPack).toBeNull();
		});
	});

	describe("setComparePackIds", () => {
		it("sets comparison pack ids", () => {
			useHyperparamStore.getState().setComparePackIds(["pack-1", "pack-2"]);
			expect(useHyperparamStore.getState().comparePackIds).toEqual(["pack-1", "pack-2"]);
		});

		it("clears comparison with null", () => {
			useHyperparamStore.getState().setComparePackIds(["pack-1", "pack-2"]);
			useHyperparamStore.getState().setComparePackIds(null);
			expect(useHyperparamStore.getState().comparePackIds).toBeNull();
		});
	});

	describe("setLoading / setError", () => {
		it("sets loading state", () => {
			useHyperparamStore.getState().setLoading(true);
			expect(useHyperparamStore.getState().isLoading).toBe(true);
		});

		it("sets error state", () => {
			useHyperparamStore.getState().setError("Something went wrong");
			expect(useHyperparamStore.getState().error).toBe("Something went wrong");
		});
	});

	describe("reset", () => {
		it("resets to initial state", () => {
			useHyperparamStore.getState().setPacks([makePack()]);
			useHyperparamStore.getState().selectPack("pack-1");
			useHyperparamStore.getState().setLoading(true);
			useHyperparamStore.getState().setError("Error");

			useHyperparamStore.getState().reset();

			const state = useHyperparamStore.getState();
			expect(state.packs).toEqual([]);
			expect(state.selectedPackId).toBeNull();
			expect(state.editingPack).toBeNull();
			expect(state.comparePackIds).toBeNull();
			expect(state.isLoading).toBe(false);
			expect(state.error).toBeNull();
		});
	});
});
