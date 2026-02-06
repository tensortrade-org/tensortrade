import type {
	CampaignEndMessage,
	CampaignImportanceMessage,
	CampaignProgressMessage,
	CampaignSearchParam,
	CampaignStartMessage,
	TrialCompleteMessage,
	TrialIterationRecord,
	TrialPrunedMessage,
	TrialStartMessage,
	TrialUpdateMessage,
} from "@/lib/types";
import { create } from "zustand";

interface TrialLiveData {
	trialNumber: number;
	params: Record<string, number | string>;
	state: "running" | "complete" | "pruned";
	iterations: TrialIterationRecord[];
	value: number | null;
	durationSeconds: number | null;
	isBest: boolean;
}

interface ActivityEntry {
	id: number;
	trialNumber: number;
	event: "complete" | "pruned" | "started";
	value: number | null;
	isBest: boolean;
	iteration: number | null;
	timestamp: number;
}

interface CampaignState {
	studyName: string | null;
	totalTrials: number;
	isActive: boolean;
	searchSpace: Record<string, CampaignSearchParam>;
	trials: TrialLiveData[];
	currentTrialNumber: number | null;
	bestValue: number | null;
	bestParams: Record<string, number | string>;
	completedCount: number;
	prunedCount: number;
	importance: Record<string, number>;
	elapsedSeconds: number;
	etaSeconds: number | null;
	endStatus: string | null;
	activityFeed: ActivityEntry[];

	// Actions
	onCampaignStart: (msg: CampaignStartMessage) => void;
	onTrialStart: (msg: TrialStartMessage) => void;
	onTrialUpdate: (msg: TrialUpdateMessage) => void;
	onTrialPruned: (msg: TrialPrunedMessage) => void;
	onTrialComplete: (msg: TrialCompleteMessage) => void;
	onImportance: (msg: CampaignImportanceMessage) => void;
	onProgress: (msg: CampaignProgressMessage) => void;
	onCampaignEnd: (msg: CampaignEndMessage) => void;
	reset: () => void;
}

const MAX_ACTIVITY = 200;
let activityIdCounter = 0;

const initialState: Pick<
	CampaignState,
	| "studyName"
	| "totalTrials"
	| "isActive"
	| "searchSpace"
	| "trials"
	| "currentTrialNumber"
	| "bestValue"
	| "bestParams"
	| "completedCount"
	| "prunedCount"
	| "importance"
	| "elapsedSeconds"
	| "etaSeconds"
	| "endStatus"
	| "activityFeed"
> = {
	studyName: null,
	totalTrials: 0,
	isActive: false,
	searchSpace: {},
	trials: [],
	currentTrialNumber: null,
	bestValue: null,
	bestParams: {},
	completedCount: 0,
	prunedCount: 0,
	importance: {},
	elapsedSeconds: 0,
	etaSeconds: null,
	endStatus: null,
	activityFeed: [],
};

export const useCampaignStore = create<CampaignState>((set) => ({
	...initialState,

	onCampaignStart: (msg: CampaignStartMessage) =>
		set({
			studyName: msg.study_name,
			totalTrials: msg.total_trials,
			isActive: true,
			searchSpace: msg.search_space,
			trials: [],
			currentTrialNumber: null,
			bestValue: null,
			bestParams: {},
			completedCount: 0,
			prunedCount: 0,
			importance: {},
			elapsedSeconds: 0,
			etaSeconds: null,
			endStatus: null,
			activityFeed: [],
		}),

	onTrialStart: (msg: TrialStartMessage) =>
		set((state) => {
			const newTrial: TrialLiveData = {
				trialNumber: msg.trial_number,
				params: msg.params,
				state: "running",
				iterations: [],
				value: null,
				durationSeconds: null,
				isBest: false,
			};

			const entry: ActivityEntry = {
				id: ++activityIdCounter,
				trialNumber: msg.trial_number,
				event: "started",
				value: null,
				isBest: false,
				iteration: null,
				timestamp: Date.now(),
			};

			const feed = [entry, ...state.activityFeed].slice(0, MAX_ACTIVITY);

			return {
				trials: [...state.trials, newTrial],
				currentTrialNumber: msg.trial_number,
				activityFeed: feed,
			};
		}),

	onTrialUpdate: (msg: TrialUpdateMessage) =>
		set((state) => {
			const trials = state.trials.map((t) => {
				if (t.trialNumber !== msg.trial_number) return t;
				const iterEntry: TrialIterationRecord = {
					iteration: msg.iteration,
					metrics: msg.metrics,
				};
				return {
					...t,
					iterations: [...t.iterations, iterEntry],
				};
			});
			return { trials };
		}),

	onTrialPruned: (msg: TrialPrunedMessage) =>
		set((state) => {
			const trials = state.trials.map((t) => {
				if (t.trialNumber !== msg.trial_number) return t;
				return {
					...t,
					state: "pruned" as const,
					value: msg.intermediate_value,
				};
			});

			const entry: ActivityEntry = {
				id: ++activityIdCounter,
				trialNumber: msg.trial_number,
				event: "pruned",
				value: msg.intermediate_value,
				isBest: false,
				iteration: msg.pruned_at_iteration,
				timestamp: Date.now(),
			};

			const feed = [entry, ...state.activityFeed].slice(0, MAX_ACTIVITY);

			return {
				trials,
				prunedCount: state.prunedCount + 1,
				currentTrialNumber: null,
				activityFeed: feed,
			};
		}),

	onTrialComplete: (msg: TrialCompleteMessage) =>
		set((state) => {
			const trials = state.trials.map((t) => {
				if (t.trialNumber !== msg.trial_number) return t;
				return {
					...t,
					state: "complete" as const,
					value: msg.value,
					durationSeconds: msg.duration_seconds,
					isBest: msg.is_best,
				};
			});

			const newBest = msg.is_best ? msg.value : state.bestValue;
			const newBestParams = msg.is_best ? msg.params : state.bestParams;

			const entry: ActivityEntry = {
				id: ++activityIdCounter,
				trialNumber: msg.trial_number,
				event: "complete",
				value: msg.value,
				isBest: msg.is_best,
				iteration: null,
				timestamp: Date.now(),
			};

			const feed = [entry, ...state.activityFeed].slice(0, MAX_ACTIVITY);

			return {
				trials,
				completedCount: state.completedCount + 1,
				bestValue: newBest,
				bestParams: newBestParams,
				currentTrialNumber: null,
				activityFeed: feed,
			};
		}),

	onImportance: (msg: CampaignImportanceMessage) => set({ importance: msg.importance }),

	onProgress: (msg: CampaignProgressMessage) =>
		set({
			completedCount: msg.trials_completed,
			prunedCount: msg.trials_pruned,
			totalTrials: msg.total_trials,
			bestValue: msg.best_value,
			elapsedSeconds: msg.elapsed_seconds,
			etaSeconds: msg.eta_seconds,
		}),

	onCampaignEnd: (msg: CampaignEndMessage) =>
		set({
			isActive: false,
			endStatus: msg.status,
			bestValue: msg.best_value,
			bestParams: msg.best_params,
			completedCount: msg.total_completed,
			prunedCount: msg.total_pruned,
			currentTrialNumber: null,
		}),

	reset: () => set(initialState),
}));
