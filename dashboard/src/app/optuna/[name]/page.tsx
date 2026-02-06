"use client";

import { ParallelCoordinate } from "@/components/charts/ParallelCoordinate";
import { ParamHeatmap } from "@/components/charts/ParamHeatmap";
import { ParamImportance } from "@/components/charts/ParamImportance";
import { TrialCurvesChart } from "@/components/charts/TrialCurvesChart";
import { TrialOutcomesScatter } from "@/components/charts/TrialOutcomesScatter";
import { Badge } from "@/components/common/Badge";
import { Card, CardHeader } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { InsightRequest } from "@/components/insights/InsightRequest";
import { useApi } from "@/hooks/useApi";
import { getOptunaStudy, getParamImportance, getStudyCurves } from "@/lib/api";
import { formatDuration, formatNumber } from "@/lib/formatters";
import type {
	OptunaStudyDetail,
	OptunaTrialRecord,
	ParamImportance as ParamImportanceType,
	StudyCurvesResponse,
} from "@/lib/types";
import { useParams } from "next/navigation";
import { useCallback, useMemo, useState } from "react";

interface TrialStateVariant {
	variant: "success" | "warning" | "danger";
}

const TRIAL_STATE_VARIANTS: Record<string, TrialStateVariant> = {
	complete: { variant: "success" },
	pruned: { variant: "warning" },
	fail: { variant: "danger" },
};

function BestTrialsTable({
	trials,
	highlightedTrial,
	onTrialClick,
}: {
	trials: OptunaTrialRecord[];
	highlightedTrial: number | null;
	onTrialClick: (trialNumber: number) => void;
}) {
	const bestTrials = useMemo(() => {
		const completed = trials.filter((t) => t.state === "complete" && t.value !== null);
		return [...completed].sort((a, b) => (b.value ?? 0) - (a.value ?? 0)).slice(0, 10);
	}, [trials]);

	if (bestTrials.length === 0) {
		return (
			<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
				No completed trials.
			</div>
		);
	}

	return (
		<div className="overflow-x-auto">
			<table className="w-full text-sm">
				<thead>
					<tr className="border-b border-[var(--border-color)] text-left text-xs text-[var(--text-secondary)]">
						<th className="pb-2 pr-4 font-medium">Trial #</th>
						<th className="pb-2 pr-4 font-medium text-right">Val P&amp;L</th>
						<th className="pb-2 pr-4 font-medium text-right">Duration</th>
						<th className="pb-2 pr-4 font-medium">State</th>
						<th className="pb-2 font-medium">Parameters</th>
					</tr>
				</thead>
				<tbody>
					{bestTrials.map((trial) => {
						const stateInfo = TRIAL_STATE_VARIANTS[trial.state] ?? TRIAL_STATE_VARIANTS.complete;
						const isHighlighted = highlightedTrial === trial.trial_number;
						return (
							<tr
								key={trial.id}
								onClick={() => onTrialClick(trial.trial_number)}
								onKeyDown={(e) => {
									if (e.key === "Enter") onTrialClick(trial.trial_number);
								}}
								tabIndex={0}
								className={`cursor-pointer border-b border-[var(--border-color)] last:border-0 hover:bg-[var(--bg-secondary)] ${
									isHighlighted ? "bg-[var(--accent-blue)]/5" : ""
								}`}
							>
								<td className="py-2 pr-4 font-mono text-[var(--text-primary)]">
									{trial.trial_number}
								</td>
								<td className="py-2 pr-4 text-right font-mono text-[var(--accent-green)]">
									{trial.value !== null ? `$${trial.value.toFixed(0)}` : "--"}
								</td>
								<td className="py-2 pr-4 text-right font-mono text-[var(--text-secondary)]">
									{trial.duration_seconds !== null ? formatDuration(trial.duration_seconds) : "--"}
								</td>
								<td className="py-2 pr-4">
									<Badge label={trial.state} variant={stateInfo.variant} />
								</td>
								<td className="py-2">
									<code className="text-xs text-[var(--text-secondary)]">
										{Object.entries(trial.params)
											.map(([k, v]) => {
												const numVal = typeof v === "number" ? v : Number(v);
												return `${k}=${Number.isFinite(numVal) ? numVal.toPrecision(3) : v}`;
											})
											.join(", ")}
									</code>
								</td>
							</tr>
						);
					})}
				</tbody>
			</table>
		</div>
	);
}

export default function OptunaStudyDetailPage() {
	const params = useParams();
	const studyName = decodeURIComponent(params.name as string);

	const [showInsight, setShowInsight] = useState(false);
	const [highlightedTrial, setHighlightedTrial] = useState<number | null>(null);
	const [curveMetric, setCurveMetric] = useState("pnl_mean");
	const [heatmapParamX, setHeatmapParamX] = useState<string>("");
	const [heatmapParamY, setHeatmapParamY] = useState<string>("");

	// Three parallel API calls
	const studyFetcher = useCallback(() => getOptunaStudy(studyName), [studyName]);
	const curvesFetcher = useCallback(() => getStudyCurves(studyName), [studyName]);
	const importanceFetcher = useCallback(() => getParamImportance(studyName), [studyName]);

	const {
		data: study,
		loading: studyLoading,
		error: studyError,
	} = useApi<OptunaStudyDetail>(studyFetcher, [studyName]);

	const { data: curves, loading: curvesLoading } = useApi<StudyCurvesResponse>(curvesFetcher, [
		studyName,
	]);

	const {
		data: importance,
		loading: importanceLoading,
		error: importanceError,
	} = useApi<ParamImportanceType>(importanceFetcher, [studyName]);

	const paramKeys = useMemo(() => {
		if (!study?.trials || study.trials.length === 0) return [];
		const keySet = new Set<string>();
		for (const trial of study.trials) {
			for (const key of Object.keys(trial.params)) {
				keySet.add(key);
			}
		}
		return Array.from(keySet);
	}, [study?.trials]);

	const numericParamKeys = useMemo(() => {
		if (!study?.trials || study.trials.length === 0) return [];
		const keySet = new Set<string>();
		for (const trial of study.trials) {
			for (const [key, val] of Object.entries(trial.params)) {
				if (typeof val === "number") keySet.add(key);
			}
		}
		return Array.from(keySet);
	}, [study?.trials]);

	// Auto-select heatmap params
	useMemo(() => {
		if (numericParamKeys.length >= 2) {
			if (!heatmapParamX || !numericParamKeys.includes(heatmapParamX)) {
				setHeatmapParamX(numericParamKeys[0]);
			}
			if (!heatmapParamY || !numericParamKeys.includes(heatmapParamY)) {
				setHeatmapParamY(numericParamKeys[1]);
			}
		}
	}, [numericParamKeys, heatmapParamX, heatmapParamY]);

	// Available metrics from curve data
	const availableMetrics = useMemo(() => {
		if (!curves?.trials) return ["pnl_mean", "episode_return_mean"];
		const metricSet = new Set<string>();
		for (const trial of curves.trials) {
			for (const iter of trial.iterations) {
				for (const key of Object.keys(iter.metrics)) {
					metricSet.add(key);
				}
			}
		}
		return metricSet.size > 0 ? Array.from(metricSet) : ["pnl_mean", "episode_return_mean"];
	}, [curves?.trials]);

	const failedTrials = useMemo(() => {
		if (!study?.trials) return 0;
		return study.trials.filter((t) => t.state === "fail").length;
	}, [study?.trials]);

	const handleTrialClick = useCallback((trialNumber: number) => {
		setHighlightedTrial((prev) => (prev === trialNumber ? null : trialNumber));
	}, []);

	const handleInsightComplete = useCallback(() => {
		setShowInsight(false);
	}, []);

	if (studyLoading) {
		return <LoadingState message="Loading study..." />;
	}

	if (studyError) {
		return (
			<Card>
				<div className="py-6 text-center text-sm text-[var(--accent-red)]">
					Failed to load study: {studyError.message}
				</div>
			</Card>
		);
	}

	if (!study) {
		return (
			<Card>
				<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
					Study not found.
				</div>
			</Card>
		);
	}

	return (
		<div className="space-y-6">
			{/* Header */}
			<div className="flex items-center justify-between">
				<div className="space-y-1">
					<h1 className="text-xl font-semibold text-[var(--text-primary)]">{study.study_name}</h1>
					<div className="flex items-center gap-4 text-sm text-[var(--text-secondary)]">
						<span>Total: {formatNumber(study.total)}</span>
						<span className="text-[var(--accent-green)]">
							Completed: {formatNumber(study.completed)}
						</span>
						<span className="text-[var(--accent-amber)]">Pruned: {formatNumber(study.pruned)}</span>
						{failedTrials > 0 && (
							<span className="text-[var(--accent-red)]">Failed: {formatNumber(failedTrials)}</span>
						)}
					</div>
				</div>
				<button
					type="button"
					onClick={() => setShowInsight(true)}
					className="rounded-md bg-[var(--accent-purple)] px-4 py-2 text-sm font-medium text-white hover:opacity-90"
				>
					Suggest Strategy
				</button>
			</div>

			{/* Insight Request */}
			{showInsight && (
				<Card>
					<CardHeader title="Strategy Suggestion" />
					<InsightRequest onComplete={handleInsightComplete} />
				</Card>
			)}

			{/* Trial Training Curves */}
			<Card>
				<CardHeader
					title="Trial Training Curves (Training Set)"
					action={
						<select
							value={curveMetric}
							onChange={(e) => setCurveMetric(e.target.value)}
							className="rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] px-2 py-1 text-xs text-[var(--text-primary)]"
						>
							{availableMetrics.map((m) => (
								<option key={m} value={m}>
									{m}
								</option>
							))}
						</select>
					}
				/>
				<div className="h-[400px]">
					{curvesLoading ? (
						<LoadingState message="Loading curves..." />
					) : curves?.trials ? (
						<TrialCurvesChart
							trials={curves.trials}
							metricKey={curveMetric}
							highlightedTrial={highlightedTrial}
							onTrialHover={setHighlightedTrial}
							onTrialClick={handleTrialClick}
						/>
					) : (
						<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
							No curve data available.
						</div>
					)}
				</div>
			</Card>

			{/* Trial Outcomes + Param Importance side by side */}
			<div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
				<Card>
					<CardHeader title="Trial Outcomes (Validation Set)" />
					<div className="h-80">
						{study.trials.length > 0 ? (
							<TrialOutcomesScatter
								trials={study.trials}
								onTrialClick={handleTrialClick}
								highlightedTrial={highlightedTrial}
							/>
						) : (
							<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
								No trials recorded.
							</div>
						)}
					</div>
				</Card>

				<Card>
					<CardHeader title="Parameter Importance" />
					<div className="h-80">
						{importanceLoading ? (
							<LoadingState message="Loading importance..." />
						) : importanceError ? (
							<div className="flex h-full items-center justify-center text-sm text-[var(--accent-red)]">
								Failed to load importance: {importanceError.message}
							</div>
						) : importance ? (
							<ParamImportance importance={importance.importance} />
						) : (
							<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
								No importance data available.
							</div>
						)}
					</div>
				</Card>
			</div>

			{/* Parameter Heatmap */}
			{numericParamKeys.length >= 2 && (
				<Card>
					<CardHeader
						title="Parameter Heatmap"
						action={
							<div className="flex items-center gap-2">
								<select
									value={heatmapParamX}
									onChange={(e) => setHeatmapParamX(e.target.value)}
									className="rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] px-2 py-1 text-xs text-[var(--text-primary)]"
								>
									{numericParamKeys.map((k) => (
										<option key={k} value={k}>
											{k}
										</option>
									))}
								</select>
								<span className="text-xs text-[var(--text-secondary)]">vs</span>
								<select
									value={heatmapParamY}
									onChange={(e) => setHeatmapParamY(e.target.value)}
									className="rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] px-2 py-1 text-xs text-[var(--text-primary)]"
								>
									{numericParamKeys.map((k) => (
										<option key={k} value={k}>
											{k}
										</option>
									))}
								</select>
							</div>
						}
					/>
					<div className="h-[350px]">
						<ParamHeatmap trials={study.trials} paramX={heatmapParamX} paramY={heatmapParamY} />
					</div>
				</Card>
			)}

			{/* Parallel Coordinate */}
			{paramKeys.length > 0 && (
				<Card>
					<CardHeader title="Parallel Coordinates" />
					<div className="h-96">
						<ParallelCoordinate trials={study.trials} paramKeys={paramKeys} />
					</div>
				</Card>
			)}

			{/* Best Trials Table */}
			<Card>
				<CardHeader title="Best Trials (Validation P&amp;L)" />
				<BestTrialsTable
					trials={study.trials}
					highlightedTrial={highlightedTrial}
					onTrialClick={handleTrialClick}
				/>
			</Card>
		</div>
	);
}
