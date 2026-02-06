"use client";

import { Spinner } from "@/components/common/Spinner";
import { useApi } from "@/hooks/useApi";
import { getExperiments, getOptunaStudies, requestAnalysis } from "@/lib/api";
import type { ExperimentSummary, InsightReport, OptunaStudySummary } from "@/lib/types";
import { useCallback, useState } from "react";

type AnalysisType = "experiment" | "comparison" | "strategy" | "trades";

interface InsightRequestProps {
	onComplete?: (report: InsightReport) => void;
}

interface AnalysisOption {
	value: AnalysisType;
	label: string;
	description: string;
}

const ANALYSIS_OPTIONS: AnalysisOption[] = [
	{
		value: "experiment",
		label: "Experiment Analysis",
		description: "Analyze metrics and performance of a selected experiment",
	},
	{
		value: "comparison",
		label: "Comparison",
		description: "Compare experiments side by side to find patterns",
	},
	{
		value: "strategy",
		label: "Strategy Advice",
		description: "Get HP suggestions based on an Optuna study",
	},
	{
		value: "trades",
		label: "Trade Analysis",
		description: "Deep dive into trade patterns and execution",
	},
];

const NEEDS_EXPERIMENT: Set<AnalysisType> = new Set(["experiment", "trades"]);
const NEEDS_MULTI_EXPERIMENT: Set<AnalysisType> = new Set(["comparison"]);
const NEEDS_STUDY: Set<AnalysisType> = new Set(["strategy"]);

interface RequestState {
	loading: boolean;
	error: string | null;
}

export function InsightRequest({ onComplete }: InsightRequestProps) {
	const [analysisType, setAnalysisType] = useState<AnalysisType>("experiment");
	const [selectedExperimentId, setSelectedExperimentId] = useState<string>("");
	const [selectedExperimentIds, setSelectedExperimentIds] = useState<Set<string>>(new Set());
	const [selectedStudy, setSelectedStudy] = useState<string>("");
	const [customPrompt, setCustomPrompt] = useState<string>("");
	const [requestState, setRequestState] = useState<RequestState>({
		loading: false,
		error: null,
	});

	const experimentsFetcher = useCallback(() => getExperiments(), []);
	const studiesFetcher = useCallback(() => getOptunaStudies(), []);
	const { data: experiments } = useApi<ExperimentSummary[]>(experimentsFetcher, []);
	const { data: studies } = useApi<OptunaStudySummary[]>(studiesFetcher, []);

	const needsExperiment = NEEDS_EXPERIMENT.has(analysisType);
	const needsMultiExperiment = NEEDS_MULTI_EXPERIMENT.has(analysisType);
	const needsStudy = NEEDS_STUDY.has(analysisType);

	const toggleExperimentId = (id: string) => {
		setSelectedExperimentIds((prev) => {
			const next = new Set(prev);
			if (next.has(id)) {
				next.delete(id);
			} else {
				next.add(id);
			}
			return next;
		});
	};

	const canSubmit = (): boolean => {
		if (requestState.loading) return false;
		if (needsExperiment && !selectedExperimentId) return false;
		if (needsMultiExperiment && selectedExperimentIds.size < 2) return false;
		if (needsStudy && !selectedStudy) return false;
		return true;
	};

	const handleSubmit = async () => {
		setRequestState({ loading: true, error: null });
		try {
			const payload: Record<string, unknown> = {
				analysis_type: analysisType,
			};

			if (needsExperiment) {
				payload.experiment_ids = [selectedExperimentId];
			} else if (needsMultiExperiment) {
				payload.experiment_ids = Array.from(selectedExperimentIds);
			}

			if (needsStudy) {
				payload.study_name = selectedStudy;
			}

			if (customPrompt.trim()) {
				payload.prompt = customPrompt.trim();
			}

			const report = await requestAnalysis(payload);
			setRequestState({ loading: false, error: null });
			onComplete?.(report);
		} catch (err) {
			const message = err instanceof Error ? err.message : "Analysis request failed";
			setRequestState({ loading: false, error: message });
		}
	};

	return (
		<div className="flex flex-col gap-4">
			{/* Analysis type selector */}
			<div className="grid gap-2 sm:grid-cols-2">
				{ANALYSIS_OPTIONS.map((option) => (
					<button
						key={option.value}
						type="button"
						onClick={() => setAnalysisType(option.value)}
						className={`rounded-lg border p-3 text-left transition-colors ${
							analysisType === option.value
								? "border-[var(--accent-blue)] bg-[var(--accent-blue)]/10"
								: "border-[var(--border-color)] bg-[var(--bg-secondary)] hover:border-[var(--text-secondary)]"
						}`}
					>
						<p
							className={`text-sm font-medium ${
								analysisType === option.value
									? "text-[var(--accent-blue)]"
									: "text-[var(--text-primary)]"
							}`}
						>
							{option.label}
						</p>
						<p className="mt-0.5 text-xs text-[var(--text-secondary)]">{option.description}</p>
					</button>
				))}
			</div>

			{/* Single experiment selector (experiment / trades) */}
			{needsExperiment && (
				<div className="flex flex-col gap-1.5">
					<label
						htmlFor="experiment-select"
						className="text-xs font-medium text-[var(--text-secondary)]"
					>
						Select Experiment
					</label>
					<select
						id="experiment-select"
						value={selectedExperimentId}
						onChange={(e) => setSelectedExperimentId(e.target.value)}
						className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)]"
					>
						<option value="">-- Choose an experiment --</option>
						{(experiments ?? []).map((exp) => (
							<option key={exp.id} value={exp.id}>
								{exp.name} ({exp.status})
							</option>
						))}
					</select>
				</div>
			)}

			{/* Multi experiment selector (comparison) */}
			{needsMultiExperiment && (
				<fieldset className="flex flex-col gap-1.5 border-none p-0 m-0">
					<legend className="text-xs font-medium text-[var(--text-secondary)] p-0 mb-1.5">
						Select Experiments to Compare (2+)
					</legend>
					<div className="max-h-48 overflow-y-auto rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)]">
						{(experiments ?? []).length === 0 ? (
							<p className="p-3 text-xs text-[var(--text-secondary)]">No experiments found</p>
						) : (
							(experiments ?? []).map((exp) => (
								<label
									key={exp.id}
									className="flex cursor-pointer items-center gap-2 px-3 py-2 text-sm hover:bg-[var(--bg-primary)]"
								>
									<input
										type="checkbox"
										checked={selectedExperimentIds.has(exp.id)}
										onChange={() => toggleExperimentId(exp.id)}
										className="accent-[var(--accent-blue)]"
									/>
									<span className="text-[var(--text-primary)]">{exp.name}</span>
									<span className="text-xs text-[var(--text-secondary)]">({exp.status})</span>
								</label>
							))
						)}
					</div>
					{selectedExperimentIds.size > 0 && (
						<p className="text-xs text-[var(--text-secondary)]">
							{selectedExperimentIds.size} selected
							{selectedExperimentIds.size < 2 && " (need at least 2)"}
						</p>
					)}
				</fieldset>
			)}

			{/* Study selector (strategy) */}
			{needsStudy && (
				<div className="flex flex-col gap-1.5">
					<label
						htmlFor="study-select"
						className="text-xs font-medium text-[var(--text-secondary)]"
					>
						Select Optuna Study
					</label>
					<select
						id="study-select"
						value={selectedStudy}
						onChange={(e) => setSelectedStudy(e.target.value)}
						className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)]"
					>
						<option value="">-- Choose a study --</option>
						{(studies ?? []).map((s) => (
							<option key={s.study_name} value={s.study_name}>
								{s.study_name} ({s.completed_trials}/{s.total_trials} trials)
							</option>
						))}
					</select>
				</div>
			)}

			{/* Custom prompt */}
			<div className="flex flex-col gap-1.5">
				<label htmlFor="custom-prompt" className="text-xs font-medium text-[var(--text-secondary)]">
					Ask a question (optional)
				</label>
				<textarea
					id="custom-prompt"
					value={customPrompt}
					onChange={(e) => setCustomPrompt(e.target.value)}
					placeholder="e.g. Why is the reward declining after episode 50?"
					rows={2}
					className="resize-y rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)]/50"
				/>
			</div>

			{/* Submit */}
			<button
				type="button"
				onClick={handleSubmit}
				disabled={!canSubmit()}
				className="flex items-center justify-center gap-2 rounded-md bg-[var(--accent-blue)] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[var(--accent-blue)]/80 disabled:cursor-not-allowed disabled:opacity-50"
			>
				{requestState.loading ? (
					<>
						<Spinner size="sm" />
						Analyzing...
					</>
				) : (
					"Request Analysis"
				)}
			</button>

			{requestState.error && (
				<p className="text-xs text-[var(--accent-red)]">{requestState.error}</p>
			)}
		</div>
	);
}
