import type {
	ExperimentDetail,
	ExperimentSummary,
	InsightReport,
	LeaderboardEntry,
	OptunaStudyDetail,
	OptunaStudySummary,
	ParamImportance,
	StudyCurvesResponse,
	TradeRecord,
	TrainingStatus,
} from "@/lib/types";

const API_BASE = "/api";

interface ExperimentsParams {
	script?: string;
	status?: string;
	limit?: number;
}

interface TradesParams {
	episode?: number;
}

interface LeaderboardParams {
	metric?: string;
	script?: string;
}

interface AllTradesParams {
	experiment_id?: string;
	side?: string;
	limit?: number;
}

interface AnalysisRequest {
	experiment_ids?: string[];
	analysis_type?: string;
	prompt?: string;
	[key: string]: unknown;
}

interface ActionResponse {
	status: string;
	message: string;
}

function buildQuery<T extends object>(params?: T): string {
	if (!params) return "";
	const entries = Object.entries(params).filter(
		(entry): entry is [string, string | number] =>
			entry[1] !== undefined && (typeof entry[1] === "string" || typeof entry[1] === "number"),
	);
	if (entries.length === 0) return "";
	const searchParams = new URLSearchParams();
	for (const [key, value] of entries) {
		searchParams.set(key, String(value));
	}
	return `?${searchParams.toString()}`;
}

async function fetchJSON<T>(url: string): Promise<T> {
	const res = await fetch(`${API_BASE}${url}`);
	if (!res.ok) throw new Error(`API error: ${res.status}`);
	return res.json();
}

async function postJSON<T>(url: string, body: Record<string, unknown>): Promise<T> {
	const res = await fetch(`${API_BASE}${url}`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	});
	if (!res.ok) throw new Error(`API error: ${res.status}`);
	return res.json();
}

export async function getExperiments(params?: ExperimentsParams): Promise<ExperimentSummary[]> {
	return fetchJSON<ExperimentSummary[]>(`/experiments${buildQuery(params)}`);
}

export async function getExperiment(id: string): Promise<ExperimentDetail> {
	return fetchJSON<ExperimentDetail>(`/experiments/${id}`);
}

export async function getExperimentTrades(
	id: string,
	params?: TradesParams,
): Promise<TradeRecord[]> {
	return fetchJSON<TradeRecord[]>(`/experiments/${id}/trades${buildQuery(params)}`);
}

export async function getLeaderboard(params?: LeaderboardParams): Promise<LeaderboardEntry[]> {
	return fetchJSON<LeaderboardEntry[]>(`/leaderboard${buildQuery(params)}`);
}

export async function getOptunaStudies(): Promise<OptunaStudySummary[]> {
	return fetchJSON<OptunaStudySummary[]>("/optuna/studies");
}

export async function getOptunaStudy(name: string): Promise<OptunaStudyDetail> {
	return fetchJSON<OptunaStudyDetail>(`/optuna/studies/${encodeURIComponent(name)}`);
}

export async function getParamImportance(name: string): Promise<ParamImportance> {
	return fetchJSON<ParamImportance>(`/optuna/studies/${encodeURIComponent(name)}/importance`);
}

export async function getInsights(): Promise<InsightReport[]> {
	return fetchJSON<InsightReport[]>("/insights");
}

export async function getInsight(id: string): Promise<InsightReport> {
	return fetchJSON<InsightReport>(`/insights/${id}`);
}

export async function requestAnalysis(body: AnalysisRequest): Promise<InsightReport> {
	return postJSON<InsightReport>("/insights/analyze", body);
}

export async function getAllTrades(params?: AllTradesParams): Promise<TradeRecord[]> {
	return fetchJSON<TradeRecord[]>(`/trades${buildQuery(params)}`);
}

export async function getStatus(): Promise<TrainingStatus> {
	return fetchJSON<TrainingStatus>("/status");
}

export async function stopTraining(): Promise<ActionResponse> {
	return postJSON<ActionResponse>("/training/stop", {});
}

export async function pauseTraining(): Promise<ActionResponse> {
	return postJSON<ActionResponse>("/training/pause", {});
}

export async function resumeTraining(): Promise<ActionResponse> {
	return postJSON<ActionResponse>("/training/resume", {});
}

export async function getStudyCurves(name: string): Promise<StudyCurvesResponse> {
	return fetchJSON<StudyCurvesResponse>(`/optuna/studies/${encodeURIComponent(name)}/curves`);
}

interface InferenceRunRequest {
	experiment_id: string;
	use_random_agent?: boolean;
}

export async function startInference(
	experimentId: string,
	useRandomAgent = true,
): Promise<ActionResponse> {
	return postJSON<ActionResponse>("/inference/run", {
		experiment_id: experimentId,
		use_random_agent: useRandomAgent,
	} satisfies InferenceRunRequest);
}
