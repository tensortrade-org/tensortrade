import type {
	CampaignLaunchRequest,
	CampaignLaunchResponse,
	DashboardStats,
	DatasetConfig,
	DatasetPreview,
	ExperimentDetail,
	ExperimentSummary,
	FeatureCatalogEntry,
	FeatureSpec,
	HyperparameterPack,
	InsightReport,
	LaunchRequest,
	LaunchResponse,
	LeaderboardEntry,
	LiveSession,
	LiveStatusMessage,
	LiveTradeEvent,
	LiveTradingStartRequest,
	OptunaStudyDetail,
	OptunaStudySummary,
	ParamImportance,
	RunningCampaign,
	RunningExperiment,
	SplitConfig,
	StudyCurvesResponse,
	TrainingConfig,
	TrainingStatus,
} from "@/lib/types";

const API_BASE = "/api";

// SSE streams bypass the Next.js rewrite proxy (which buffers responses)
// and hit the FastAPI backend directly.
const STREAM_API_BASE = process.env.NEXT_PUBLIC_STREAM_API_URL ?? "http://localhost:8000/api";

interface ExperimentsParams {
	script?: string;
	status?: string;
	limit?: number;
}

interface LeaderboardParams {
	metric?: string;
	script?: string;
}

interface AnalysisRequest {
	experiment_ids?: string[];
	analysis_type?: string;
	study_name?: string;
	prompt?: string;
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

async function putJSON<T>(url: string, body: Record<string, unknown>): Promise<T> {
	const res = await fetch(`${API_BASE}${url}`, {
		method: "PUT",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	});
	if (!res.ok) throw new Error(`API error: ${res.status}`);
	return res.json();
}

async function deleteJSON<T>(url: string): Promise<T> {
	const res = await fetch(`${API_BASE}${url}`, { method: "DELETE" });
	if (!res.ok) throw new Error(`API error: ${res.status}`);
	return res.json();
}

export async function getExperiments(params?: ExperimentsParams): Promise<ExperimentSummary[]> {
	return fetchJSON<ExperimentSummary[]>(`/experiments${buildQuery(params)}`);
}

export async function getExperiment(id: string): Promise<ExperimentDetail> {
	return fetchJSON<ExperimentDetail>(`/experiments/${id}`);
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

export async function getStudyInsight(studyName: string): Promise<InsightReport | null> {
	const result = await fetchJSON<InsightReport & { error?: string }>(
		`/insights/study/${encodeURIComponent(studyName)}`,
	);
	if (result.error) return null;
	return result;
}

export async function requestAnalysis(body: AnalysisRequest): Promise<InsightReport> {
	return postJSON<InsightReport>("/insights/analyze", body);
}

export async function getDashboardStats(): Promise<DashboardStats> {
	return fetchJSON<DashboardStats>("/stats");
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
	dataset_id?: string;
	start_date?: string;
	end_date?: string;
	test_only?: boolean;
}

export async function startInference(
	experimentId: string,
	datasetId?: string,
	startDate?: string,
	endDate?: string,
	testOnly?: boolean,
): Promise<ActionResponse> {
	const body: InferenceRunRequest = {
		experiment_id: experimentId,
	};
	if (datasetId) {
		body.dataset_id = datasetId;
	}
	if (startDate) {
		body.start_date = startDate;
	}
	if (endDate) {
		body.end_date = endDate;
	}
	if (testOnly) {
		body.test_only = true;
	}
	return postJSON<ActionResponse>("/inference/run", body as unknown as Record<string, unknown>);
}

// --- Hyperparameter Packs ---

export async function getHyperparamPacks(): Promise<HyperparameterPack[]> {
	return fetchJSON<HyperparameterPack[]>("/packs");
}

export async function getHyperparamPack(id: string): Promise<HyperparameterPack> {
	return fetchJSON<HyperparameterPack>(`/packs/${id}`);
}

interface CreateHyperparamPackRequest {
	name: string;
	description: string;
	config: TrainingConfig;
}

export async function createHyperparamPack(
	pack: CreateHyperparamPackRequest,
): Promise<HyperparameterPack> {
	return postJSON<HyperparameterPack>("/packs", pack as unknown as Record<string, unknown>);
}

interface UpdateHyperparamPackRequest {
	name?: string;
	description?: string;
	config?: TrainingConfig;
}

export async function updateHyperparamPack(
	id: string,
	pack: UpdateHyperparamPackRequest,
): Promise<HyperparameterPack> {
	return putJSON<HyperparameterPack>(`/packs/${id}`, pack as unknown as Record<string, unknown>);
}

export async function deleteHyperparamPack(id: string): Promise<{ status: string }> {
	return deleteJSON<{ status: string }>(`/packs/${id}`);
}

export async function duplicateHyperparamPack(id: string): Promise<HyperparameterPack> {
	return postJSON<HyperparameterPack>(`/packs/${id}/duplicate`, {});
}

// --- Datasets ---

export async function getDatasets(): Promise<DatasetConfig[]> {
	return fetchJSON<DatasetConfig[]>("/datasets");
}

export async function getDataset(id: string): Promise<DatasetConfig> {
	return fetchJSON<DatasetConfig>(`/datasets/${id}`);
}

interface CreateDatasetRequest {
	name: string;
	description: string;
	source_type: string;
	source_config: Record<string, string | number | boolean>;
	features: FeatureSpec[];
	split_config: SplitConfig;
}

export async function createDataset(dataset: CreateDatasetRequest): Promise<DatasetConfig> {
	return postJSON<DatasetConfig>("/datasets", dataset as unknown as Record<string, unknown>);
}

export async function updateDataset(
	id: string,
	dataset: Record<string, unknown>,
): Promise<DatasetConfig> {
	return putJSON<DatasetConfig>(`/datasets/${id}`, dataset);
}

export async function deleteDataset(id: string): Promise<{ status: string }> {
	return deleteJSON<{ status: string }>(`/datasets/${id}`);
}

export async function getDatasetPreview(id: string): Promise<DatasetPreview> {
	return fetchJSON<DatasetPreview>(`/datasets/${id}/preview`);
}

export async function getFeatureCatalog(): Promise<FeatureCatalogEntry[]> {
	return fetchJSON<FeatureCatalogEntry[]>("/datasets/features");
}

// --- Training Launcher ---

export async function launchTraining(request: LaunchRequest): Promise<LaunchResponse> {
	return postJSON<LaunchResponse>(
		"/training/launch",
		request as unknown as Record<string, unknown>,
	);
}

export async function getRunningExperiments(): Promise<RunningExperiment[]> {
	return fetchJSON<RunningExperiment[]>("/training/running");
}

export async function cancelTraining(experimentId: string): Promise<{ status: string }> {
	return postJSON<{ status: string }>(`/training/${experimentId}/cancel`, {});
}

// --- Campaign (Live Optuna) ---

export async function launchCampaign(
	request: CampaignLaunchRequest,
): Promise<CampaignLaunchResponse> {
	return postJSON<CampaignLaunchResponse>(
		"/campaign/launch",
		request as unknown as Record<string, unknown>,
	);
}

export async function getRunningCampaign(): Promise<RunningCampaign | null> {
	const result = await fetchJSON<{ is_active: boolean; study_name?: string }>("/campaign/running");
	if (!result.is_active) return null;
	return result as unknown as RunningCampaign;
}

// --- Generate HP Pack from Insights ---

interface GenerateHpPackRequest {
	experiment_id: string;
	insight_id: string;
	prompt?: string;
}

export async function generateHpPack(
	experimentId: string,
	insightId: string,
	prompt?: string,
): Promise<HyperparameterPack> {
	const body: GenerateHpPackRequest = {
		experiment_id: experimentId,
		insight_id: insightId,
	};
	if (prompt) {
		body.prompt = prompt;
	}
	return postJSON<HyperparameterPack>(
		"/insights/generate-pack",
		body as unknown as Record<string, unknown>,
	);
}

// --- Live Paper Trading ---

interface LiveTradingStartResponse {
	session_id?: string;
	status?: string;
	error?: string;
}

export async function startLiveTrading(
	request: LiveTradingStartRequest,
): Promise<LiveTradingStartResponse> {
	return postJSON<LiveTradingStartResponse>(
		"/live/start",
		request as unknown as Record<string, unknown>,
	);
}

export async function stopLiveTrading(): Promise<ActionResponse> {
	return postJSON<ActionResponse>("/live/stop", {});
}

export async function getLiveStatus(): Promise<LiveStatusMessage> {
	return fetchJSON<LiveStatusMessage>("/live/status");
}

export async function getLiveSessions(): Promise<LiveSession[]> {
	return fetchJSON<LiveSession[]>("/live/sessions");
}

export async function getLiveSession(id: string): Promise<LiveSession> {
	return fetchJSON<LiveSession>(`/live/sessions/${id}`);
}

export async function getLiveSessionTrades(id: string): Promise<LiveTradeEvent[]> {
	return fetchJSON<LiveTradeEvent[]>(`/live/sessions/${id}/trades`);
}

// --- Streaming Analysis (SSE) ---

interface SSEChunkData {
	text: string;
}

interface SSEErrorData {
	error: string;
}

interface StreamAnalysisCallbacks {
	onChunk: (text: string) => void;
	onComplete: (report: InsightReport) => void;
	onError: (message: string) => void;
}

export async function streamAnalysis(
	body: AnalysisRequest,
	callbacks: StreamAnalysisCallbacks,
): Promise<void> {
	const res = await fetch(`${STREAM_API_BASE}/insights/analyze/stream`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	});

	if (!res.ok) {
		callbacks.onError(`API error: ${res.status}`);
		return;
	}

	if (!res.body) {
		callbacks.onError("No response body");
		return;
	}

	const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
	let buffer = "";

	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += value;

			// Parse SSE events from buffer — events are separated by double newlines
			const parts = buffer.split("\n\n");
			// Keep the last part as incomplete buffer
			buffer = parts.pop() ?? "";

			for (const part of parts) {
				const trimmed = part.trim();
				if (!trimmed) continue;

				let eventType = "message";
				let data = "";

				for (const line of trimmed.split("\n")) {
					if (line.startsWith("event: ")) {
						eventType = line.slice(7);
					} else if (line.startsWith("data: ")) {
						data = line.slice(6);
					}
				}

				if (!data) continue;

				try {
					if (eventType === "chunk") {
						const parsed = JSON.parse(data) as SSEChunkData;
						callbacks.onChunk(parsed.text);
					} else if (eventType === "complete") {
						const report = JSON.parse(data) as InsightReport;
						callbacks.onComplete(report);
					} else if (eventType === "error") {
						const parsed = JSON.parse(data) as SSEErrorData;
						callbacks.onError(parsed.error);
					}
				} catch {
					// Skip malformed SSE data
				}
			}
		}
	} finally {
		reader.releaseLock();
	}
}
