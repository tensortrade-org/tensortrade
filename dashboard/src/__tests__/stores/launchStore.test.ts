import { STEP_ORDER, useLaunchStore } from "@/stores/launchStore";
import type { LaunchStep } from "@/stores/launchStore";
import { beforeEach, describe, expect, it } from "vitest";

describe("useLaunchStore", () => {
	beforeEach(() => {
		useLaunchStore.getState().reset();
	});

	describe("initial state", () => {
		it("starts at name step with empty fields", () => {
			const state = useLaunchStore.getState();
			expect(state.currentStep).toBe("name");
			expect(state.name).toBe("");
			expect(state.tags).toEqual([]);
			expect(state.datasetId).toBeNull();
			expect(state.hpPackId).toBeNull();
			expect(state.overrides).toEqual({});
			expect(state.isLaunching).toBe(false);
			expect(state.launchError).toBeNull();
			expect(state.launchedExperimentId).toBeNull();
		});
	});

	describe("setStep", () => {
		it("navigates to different steps", () => {
			for (const step of STEP_ORDER) {
				useLaunchStore.getState().setStep(step);
				expect(useLaunchStore.getState().currentStep).toBe(step);
			}
		});
	});

	describe("setName", () => {
		it("sets experiment name", () => {
			useLaunchStore.getState().setName("My Experiment");
			expect(useLaunchStore.getState().name).toBe("My Experiment");
		});

		it("can set empty name", () => {
			useLaunchStore.getState().setName("Test");
			useLaunchStore.getState().setName("");
			expect(useLaunchStore.getState().name).toBe("");
		});
	});

	describe("setTags", () => {
		it("sets tags array", () => {
			useLaunchStore.getState().setTags(["tag1", "tag2", "tag3"]);
			expect(useLaunchStore.getState().tags).toEqual(["tag1", "tag2", "tag3"]);
		});

		it("can set empty tags", () => {
			useLaunchStore.getState().setTags(["tag1"]);
			useLaunchStore.getState().setTags([]);
			expect(useLaunchStore.getState().tags).toEqual([]);
		});
	});

	describe("setDatasetId / setHpPackId", () => {
		it("selects dataset", () => {
			useLaunchStore.getState().setDatasetId("ds-123");
			expect(useLaunchStore.getState().datasetId).toBe("ds-123");
		});

		it("selects hp pack", () => {
			useLaunchStore.getState().setHpPackId("hp-456");
			expect(useLaunchStore.getState().hpPackId).toBe("hp-456");
		});

		it("clears selections with null", () => {
			useLaunchStore.getState().setDatasetId("ds-123");
			useLaunchStore.getState().setHpPackId("hp-456");
			useLaunchStore.getState().setDatasetId(null);
			useLaunchStore.getState().setHpPackId(null);
			expect(useLaunchStore.getState().datasetId).toBeNull();
			expect(useLaunchStore.getState().hpPackId).toBeNull();
		});
	});

	describe("setOverride", () => {
		it("sets an override value", () => {
			useLaunchStore.getState().setOverride("learning_rate", 1e-3);
			expect(useLaunchStore.getState().overrides.learning_rate).toBe(1e-3);
		});

		it("accumulates multiple overrides", () => {
			useLaunchStore.getState().setOverride("learning_rate", 1e-3);
			useLaunchStore.getState().setOverride("gamma", 0.95);
			const overrides = useLaunchStore.getState().overrides;
			expect(overrides.learning_rate).toBe(1e-3);
			expect(overrides.gamma).toBe(0.95);
		});
	});

	describe("clearOverrides", () => {
		it("clears all overrides", () => {
			useLaunchStore.getState().setOverride("learning_rate", 1e-3);
			useLaunchStore.getState().setOverride("gamma", 0.95);
			useLaunchStore.getState().clearOverrides();
			expect(useLaunchStore.getState().overrides).toEqual({});
		});
	});

	describe("launch state", () => {
		it("setLaunching updates launching state", () => {
			useLaunchStore.getState().setLaunching(true);
			expect(useLaunchStore.getState().isLaunching).toBe(true);
		});

		it("setLaunchError updates error", () => {
			useLaunchStore.getState().setLaunchError("Failed to launch");
			expect(useLaunchStore.getState().launchError).toBe("Failed to launch");
		});

		it("setLaunchedExperimentId stores result", () => {
			useLaunchStore.getState().setLaunchedExperimentId("exp-789");
			expect(useLaunchStore.getState().launchedExperimentId).toBe("exp-789");
		});
	});

	describe("reset", () => {
		it("resets all state to initial values", () => {
			useLaunchStore.getState().setStep("review");
			useLaunchStore.getState().setName("Test");
			useLaunchStore.getState().setTags(["tag"]);
			useLaunchStore.getState().setDatasetId("ds-1");
			useLaunchStore.getState().setHpPackId("hp-1");
			useLaunchStore.getState().setOverride("gamma", 0.99);
			useLaunchStore.getState().setLaunching(true);
			useLaunchStore.getState().setLaunchError("Error");
			useLaunchStore.getState().setLaunchedExperimentId("exp-1");

			useLaunchStore.getState().reset();

			const state = useLaunchStore.getState();
			expect(state.currentStep).toBe("name");
			expect(state.name).toBe("");
			expect(state.tags).toEqual([]);
			expect(state.datasetId).toBeNull();
			expect(state.hpPackId).toBeNull();
			expect(state.overrides).toEqual({});
			expect(state.isLaunching).toBe(false);
			expect(state.launchError).toBeNull();
			expect(state.launchedExperimentId).toBeNull();
		});
	});
});
