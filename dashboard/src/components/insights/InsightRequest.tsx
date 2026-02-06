"use client";

import { Spinner } from "@/components/common/Spinner";
import { requestAnalysis } from "@/lib/api";
import type { InsightReport } from "@/lib/types";
import { useState } from "react";

type AnalysisType = "experiment" | "comparison" | "strategy" | "trades";

interface InsightRequestProps {
	experimentIds?: string[];
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
		description: "Analyze metrics and performance of selected experiments",
	},
	{
		value: "comparison",
		label: "Comparison",
		description: "Compare experiments side by side to find patterns",
	},
	{
		value: "strategy",
		label: "Strategy Advice",
		description: "Get strategic suggestions for improving results",
	},
	{
		value: "trades",
		label: "Trade Analysis",
		description: "Deep dive into trade patterns and execution",
	},
];

interface RequestState {
	loading: boolean;
	error: string | null;
}

export function InsightRequest({ experimentIds = [], onComplete }: InsightRequestProps) {
	const [analysisType, setAnalysisType] = useState<AnalysisType>("experiment");
	const [requestState, setRequestState] = useState<RequestState>({
		loading: false,
		error: null,
	});

	const handleSubmit = async () => {
		setRequestState({ loading: true, error: null });
		try {
			const report = await requestAnalysis({
				experiment_ids: experimentIds,
				analysis_type: analysisType,
			});
			setRequestState({ loading: false, error: null });
			onComplete?.(report);
		} catch (err) {
			const message = err instanceof Error ? err.message : "Analysis request failed";
			setRequestState({ loading: false, error: message });
		}
	};

	return (
		<div className="flex flex-col gap-4">
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

			{experimentIds.length > 0 && (
				<p className="text-xs text-[var(--text-secondary)]">
					{experimentIds.length} experiment{experimentIds.length !== 1 ? "s" : ""} selected
				</p>
			)}

			<button
				type="button"
				onClick={handleSubmit}
				disabled={requestState.loading}
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
