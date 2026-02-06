/** TypeScript interfaces matching Python protocol types */

export interface StepUpdate {
	type: "step_update";
	step: number;
	price: number;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
	net_worth: number;
	action?: number;
	reward?: number;
	source?: "inference" | "training";
}

export interface TradeEvent {
	type: "trade";
	step: number;
	side: "buy" | "sell";
	price: number;
	size: number;
	commission: number;
	source?: "inference" | "training";
}

export interface TrainingUpdate {
	type: "training_update";
	iteration: number;
	episode_return_mean: number;
	pnl_mean: number;
	pnl_pct_mean: number;
	net_worth_mean: number;
	trade_count_mean: number;
	hold_count_mean: number;
}

export interface EpisodeEvent {
	type: "episode_start" | "episode_end";
	episode: number;
	initial_net_worth?: number;
	final_net_worth?: number;
	pnl?: number;
	trade_count?: number;
}

export interface ExperimentSummary {
	id: string;
	name: string;
	script: string;
	status: "running" | "completed" | "failed" | "pruned";
	started_at: string;
	completed_at: string | null;
	config: Record<string, unknown>;
	final_metrics: Record<string, number>;
	tags: string[];
}

export interface ExperimentDetail {
	experiment: ExperimentSummary;
	iterations: IterationRecord[];
}

export interface IterationRecord {
	id: number;
	experiment_id: string;
	iteration: number;
	metrics: Record<string, number>;
	timestamp: string;
}

export interface TradeRecord {
	id: number;
	experiment_id: string;
	episode: number;
	step: number;
	side: "buy" | "sell";
	price: number;
	size: number;
	commission: number;
	experiment_name?: string;
	script?: string;
}

export interface LeaderboardEntry {
	experiment_id: string;
	name: string;
	script: string;
	rank: number;
	metric_name: string;
	metric_value: number;
	final_metrics: Record<string, number>;
	started_at: string;
	tags: string[];
}

export interface OptunaStudySummary {
	study_name: string;
	total_trials: number;
	completed_trials: number;
	pruned_trials: number;
	best_value: number | null;
	worst_value: number | null;
	avg_value: number | null;
}

export interface OptunaTrialRecord {
	id: number;
	study_name: string;
	trial_number: number;
	experiment_id: string | null;
	params: Record<string, unknown>;
	value: number | null;
	state: "complete" | "pruned" | "fail";
	duration_seconds: number | null;
}

export interface OptunaStudyDetail {
	study_name: string;
	trials: OptunaTrialRecord[];
	total: number;
	completed: number;
	pruned: number;
}

export interface InsightReport {
	id: string;
	experiment_ids: string[];
	analysis_type: "experiment" | "comparison" | "strategy" | "trades";
	summary: string;
	findings: string[];
	suggestions: string[];
	confidence: "high" | "medium" | "low";
	raw_response?: string;
	created_at: string;
}

export interface TrainingStatus {
	type: "status";
	is_training: boolean;
	is_paused: boolean;
	experiment_id: string | null;
	current_iteration: number;
	dashboard_clients: number;
	training_producers: number;
}

export interface ParamImportance {
	importance: Record<string, number>;
	note?: string;
}

// --- Optuna Curves ---

export interface TrialIterationRecord {
	iteration: number;
	metrics: Record<string, number>;
}

export interface TrialCurveData {
	trial_number: number;
	state: "complete" | "pruned" | "fail";
	params: Record<string, number>;
	value: number | null;
	duration_seconds: number | null;
	iterations: TrialIterationRecord[];
}

export interface StudyCurvesResponse {
	study_name: string;
	trials: TrialCurveData[];
}

// --- Inference ---

export interface InferenceRequest {
	experiment_id: string;
	use_random_agent?: boolean;
}

export interface InferenceStatus {
	type: "inference_status";
	state: "idle" | "running" | "completed" | "error";
	experiment_id: string | null;
	total_steps: number;
	current_step: number;
	source?: "inference";
	episode_summary?: EpisodeSummary;
	error?: string;
}

export interface EpisodeSummary {
	total_steps: number;
	final_net_worth: number;
	initial_net_worth: number;
	pnl: number;
	pnl_pct: number;
	total_trades: number;
	buy_count: number;
	sell_count: number;
	hold_count: number;
}

export type WebSocketMessage =
	| StepUpdate
	| TradeEvent
	| TrainingUpdate
	| EpisodeEvent
	| TrainingStatus
	| InferenceStatus
	| { type: "training_disconnected" }
	| { type: "experiment_start"; experiment_id: string; name: string }
	| { type: "experiment_end"; experiment_id: string; status: string };
