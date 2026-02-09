/** TypeScript interfaces matching Python protocol types */

export interface StepUpdate {
	type: "step_update";
	step: number;
	timestamp?: number;
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
	timestamp?: number;
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
	buy_count_mean?: number;
	sell_count_mean?: number;
	hold_ratio_mean?: number;
	trade_ratio_mean?: number;
	pnl_per_trade_mean?: number;
	buy_sell_imbalance_mean?: number;
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
	analysis_type: "experiment" | "comparison" | "strategy" | "trades" | "campaign_analysis";
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
	start_date?: string;
	end_date?: string;
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
	dataset_name?: string;
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
	hold_ratio?: number;
	trade_ratio?: number;
	pnl_per_trade?: number;
	max_drawdown_pct?: number;
}

// --- Hyperparameter Packs ---

export interface ModelConfig {
	fcnet_hiddens: number[];
	fcnet_activation: string;
}

export interface TrainingConfig {
	algorithm: string;
	learning_rate: number;
	gamma: number;
	lambda_: number;
	clip_param: number;
	entropy_coeff: number;
	vf_loss_coeff: number;
	num_sgd_iter: number;
	sgd_minibatch_size: number;
	train_batch_size: number;
	num_rollout_workers: number;
	rollout_fragment_length: number;
	model: ModelConfig;
	action_scheme:
		| "BSH"
		| "TrailingStopBSH"
		| "BracketBSH"
		| "DrawdownBudgetBSH"
		| "CooldownBSH"
		| "HoldMinimumBSH"
		| "ConfirmationBSH"
		| "ScaledEntryBSH"
		| "PartialTakeProfitBSH"
		| "VolatilitySizedBSH"
		| "SimpleOrders"
		| "ManagedRiskOrders";
	reward_scheme:
		| "SimpleProfit"
		| "RiskAdjustedReturns"
		| "PBR"
		| "AdvancedPBR"
		| "FractionalPBR"
		| "MaxDrawdownPenalty"
		| "AdaptiveProfitSeeker";
	reward_params: Record<string, number>;
	commission: number;
	initial_cash: number;
	window_size: number;
	max_allowed_loss: number;
	max_episode_steps: number | null;
	num_iterations: number;
}

export interface HyperparameterPack {
	id: string;
	name: string;
	description: string;
	config: TrainingConfig;
	created_at: string;
	updated_at: string;
}

// --- Dataset Configuration ---

export interface FeatureSpec {
	type: string;
	[key: string]: string | number | boolean | number[] | undefined;
}

export interface SplitConfig {
	train_pct: number;
	val_pct: number;
	test_pct: number;
}

export interface DatasetConfig {
	id: string;
	name: string;
	description: string;
	source_type: "csv_upload" | "crypto_download" | "synthetic" | "alpaca_crypto";
	source_config: Record<string, string | number | boolean>;
	features: FeatureSpec[];
	split_config: SplitConfig;
	created_at: string;
	updated_at: string;
}

export interface OHLCVSample {
	date: string;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
}

export interface FeatureStats {
	mean: number;
	std: number;
	min: number;
	max: number;
}

export interface DatasetPreview {
	rows: number;
	columns: string[];
	date_range: { start: string; end: string };
	ohlcv_sample: OHLCVSample[];
	feature_stats: Record<string, FeatureStats>;
}

export interface FeatureParamDef {
	name: string;
	type: string;
	default: string | number | boolean;
	min?: number;
	max?: number;
	description: string;
}

export interface FeatureCatalogEntry {
	type: string;
	name: string;
	description: string;
	params: FeatureParamDef[];
}

// --- Dashboard Stats ---

export interface DashboardStatsExperiment {
	id: string;
	name: string;
}

export interface DashboardStats {
	total_experiments: number;
	completed: number;
	failed: number;
	running: number;
	best_pnl: number | null;
	best_pnl_experiment: DashboardStatsExperiment | null;
	best_net_worth: number | null;
	avg_pnl: number | null;
	total_trades: number;
	win_rate: number;
	total_studies: number;
	total_optuna_trials: number;
}

// --- Training Launch ---

export interface LaunchRequest {
	name: string;
	hp_pack_id: string;
	dataset_id: string;
	tags: string[];
	overrides?: Partial<TrainingConfig>;
}

export interface LaunchResponse {
	experiment_id: string;
	status: string;
}

export interface RunningExperiment {
	experiment_id: string;
	name: string;
	started_at: string;
	iteration: number;
	total_iterations: number;
}

// --- Enhanced Training Messages ---

export interface EpisodeMetrics {
	type: "episode_metrics";
	episode: number;
	reward_total: number;
	pnl: number;
	pnl_pct: number;
	net_worth: number;
	trade_count: number;
	hold_count: number;
	buy_count: number;
	sell_count: number;
	hold_ratio?: number;
	trade_ratio?: number;
	pnl_per_trade?: number;
	buy_sell_imbalance?: number;
	action_distribution: Record<string, number>;
}

export interface TrainingProgress {
	type: "training_progress";
	experiment_id: string;
	iteration: number;
	total_iterations: number;
	elapsed_seconds: number;
	eta_seconds: number | null;
}

// --- Campaign (Live Optuna) Messages ---

export interface CampaignStartMessage {
	type: "campaign_start";
	study_name: string;
	total_trials: number;
	search_space: Record<string, CampaignSearchParam>;
}

export interface CampaignSearchParam {
	type: "float" | "int" | "categorical";
	low?: number;
	high?: number;
	log?: boolean;
	choices?: (number | string)[];
}

export interface TrialStartMessage {
	type: "trial_start";
	trial_number: number;
	params: Record<string, number | string>;
	total_trials: number;
}

export interface TrialUpdateMessage {
	type: "trial_update";
	trial_number: number;
	iteration: number;
	total_iterations: number;
	metrics: Record<string, number>;
}

export interface TrialPrunedMessage {
	type: "trial_pruned";
	trial_number: number;
	pruned_at_iteration: number;
	intermediate_value: number | null;
}

export interface TrialCompleteMessage {
	type: "trial_complete";
	trial_number: number;
	value: number | null;
	params: Record<string, number | string>;
	duration_seconds: number;
	is_best: boolean;
}

export interface CampaignImportanceMessage {
	type: "campaign_importance";
	importance: Record<string, number>;
}

export interface CampaignProgressMessage {
	type: "campaign_progress";
	trials_completed: number;
	trials_pruned: number;
	total_trials: number;
	best_value: number | null;
	elapsed_seconds: number;
	eta_seconds: number | null;
}

export interface CampaignEndMessage {
	type: "campaign_end";
	status: string;
	best_value: number | null;
	best_params: Record<string, number | string>;
	total_completed: number;
	total_pruned: number;
}

// --- Live Paper Trading ---

export interface LiveSession {
	id: string;
	experiment_id: string;
	config: Record<string, unknown>;
	status: "running" | "stopped" | "error";
	started_at: string;
	stopped_at: string | null;
	symbol: string;
	timeframe: string;
	initial_equity: number | null;
	final_equity: number | null;
	total_trades: number;
	total_bars: number;
	pnl: number;
	max_drawdown_pct: number;
	model_version: number;
}

export interface LiveStatusMessage {
	type: "live_status";
	state: "idle" | "running" | "stopped" | "error";
	session_id: string | null;
	symbol: string;
	equity: number;
	pnl: number;
	pnl_pct: number;
	position: "cash" | "asset";
	total_bars: number;
	total_trades: number;
	drawdown_pct: number;
}

export interface LiveBar {
	type: "live_bar";
	timestamp: number;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
	step: number;
}

export interface LiveActionEvent {
	type: "live_action";
	step: number;
	action: number;
	action_label: "hold" | "buy" | "sell";
	price: number;
	position: number; // 0=cash, 1=asset
	timestamp: number;
}

export interface LiveTradeEvent {
	type: "live_trade";
	step: number;
	timestamp: number;
	side: "buy" | "sell";
	symbol: string;
	price: number;
	size: number;
	commission: number;
	alpaca_order_id: string | null;
}

export interface LivePortfolioMessage {
	type: "live_portfolio";
	equity: number;
	cash: number;
	position_value: number;
	pnl: number;
	pnl_pct: number;
	drawdown_pct: number;
}

export interface LiveTradingStartRequest {
	experiment_id: string;
	symbol: string;
	timeframe: string;
}

export type WebSocketMessage =
	| StepUpdate
	| TradeEvent
	| TrainingUpdate
	| EpisodeEvent
	| TrainingStatus
	| InferenceStatus
	| EpisodeMetrics
	| TrainingProgress
	| CampaignStartMessage
	| TrialStartMessage
	| TrialUpdateMessage
	| TrialPrunedMessage
	| TrialCompleteMessage
	| CampaignImportanceMessage
	| CampaignProgressMessage
	| CampaignEndMessage
	| LiveStatusMessage
	| LiveBar
	| LiveActionEvent
	| LiveTradeEvent
	| LivePortfolioMessage
	| { type: "training_disconnected" }
	| { type: "experiment_start"; experiment_id: string; name: string }
	| { type: "experiment_end"; experiment_id: string; status: string };

// --- Campaign Launch Types ---

export interface CampaignLaunchRequest {
	study_name: string;
	dataset_id: string;
	n_trials: number;
	iterations_per_trial: number;
	action_schemes?: string[];
	reward_schemes?: string[];
	search_space?: Record<string, CampaignParamSpec>;
}

export interface CampaignParamSpec {
	mode?: "tune" | "fixed";
	type: "float" | "int" | "categorical";
	low?: number;
	high?: number;
	log?: boolean;
	value?: number | string;
	choices?: Array<number | string>;
}

export interface CampaignLaunchResponse {
	study_name: string;
	status: string;
}

export interface RunningCampaign {
	study_name: string;
	started_at: string;
	n_trials: number;
	dataset_id: string;
}
