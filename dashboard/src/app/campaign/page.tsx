"use client";

import { CampaignConfigForm } from "@/components/campaign/CampaignConfigForm";
import { CampaignProgressBar } from "@/components/campaign/CampaignProgressBar";
import { CampaignSummaryBanner } from "@/components/campaign/CampaignSummaryBanner";
import { CurrentTrialPanel } from "@/components/campaign/CurrentTrialPanel";
import { TrialActivityFeed } from "@/components/campaign/TrialActivityFeed";
import { ParamImportance } from "@/components/charts/ParamImportance";
import { TrialCurvesChart } from "@/components/charts/TrialCurvesChart";
import { TrialOutcomesScatter } from "@/components/charts/TrialOutcomesScatter";
import { Card, CardHeader } from "@/components/common/Card";
import type { OptunaTrialRecord, TrialCurveData } from "@/lib/types";
import { useCampaignStore } from "@/stores/campaignStore";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export default function CampaignPage() {
	const studyName = useCampaignStore((s) => s.studyName);
	const isActive = useCampaignStore((s) => s.isActive);
	const [launching, setLaunching] = useState(false);
	const launchStudyName = useRef<string | null>(null);

	// Clear launching state once the campaign actually starts via WebSocket
	useEffect(() => {
		if (isActive && launching) {
			setLaunching(false);
			launchStudyName.current = null;
		}
	}, [isActive, launching]);
	const endStatus = useCampaignStore((s) => s.endStatus);
	const totalTrials = useCampaignStore((s) => s.totalTrials);
	const trials = useCampaignStore((s) => s.trials);
	const currentTrialNumber = useCampaignStore((s) => s.currentTrialNumber);
	const bestValue = useCampaignStore((s) => s.bestValue);
	const bestParams = useCampaignStore((s) => s.bestParams);
	const completedCount = useCampaignStore((s) => s.completedCount);
	const prunedCount = useCampaignStore((s) => s.prunedCount);
	const importance = useCampaignStore((s) => s.importance);
	const elapsedSeconds = useCampaignStore((s) => s.elapsedSeconds);
	const etaSeconds = useCampaignStore((s) => s.etaSeconds);
	const activityFeed = useCampaignStore((s) => s.activityFeed);
	const reset = useCampaignStore((s) => s.reset);

	const trialCurves: TrialCurveData[] = useMemo(
		() =>
			trials.map((t) => ({
				trial_number: t.trialNumber,
				state: t.state === "running" ? ("complete" as const) : t.state,
				params: Object.fromEntries(
					Object.entries(t.params).filter(
						(entry): entry is [string, number] => typeof entry[1] === "number",
					),
				),
				value: t.value,
				duration_seconds: t.durationSeconds,
				iterations: t.iterations,
			})),
		[trials],
	);

	const scatterTrials: OptunaTrialRecord[] = useMemo(
		() =>
			trials
				.filter((t) => t.state !== "running")
				.map((t) => ({
					id: t.trialNumber,
					study_name: studyName ?? "",
					trial_number: t.trialNumber,
					experiment_id: null,
					params: t.params as Record<string, unknown>,
					value: t.value,
					state: t.state === "running" ? ("complete" as const) : t.state,
					duration_seconds: t.durationSeconds,
				})),
		[trials, studyName],
	);

	const [highlightedTrial, setHighlightedTrial] = useState<number | null>(null);

	const handleTrialHover = useCallback((trialNumber: number | null) => {
		setHighlightedTrial(trialNumber);
	}, []);

	const handleTrialClick = useCallback((_trialNumber: number) => {
		// Could navigate to trial detail in future
	}, []);

	const handleLaunched = useCallback((name: string) => {
		launchStudyName.current = name;
		setLaunching(true);
	}, []);

	const handleNewCampaign = useCallback(() => {
		reset();
	}, [reset]);

	// Get current trial data for the detail panel
	const currentTrial = useMemo(() => {
		if (currentTrialNumber == null) return null;
		return trials.find((t) => t.trialNumber === currentTrialNumber) ?? null;
	}, [trials, currentTrialNumber]);

	const currentIteration = currentTrial?.iterations.length ?? 0;

	// Determine the total iterations for current trial from last trial_update message
	const lastIterationEntry = currentTrial?.iterations[currentTrial.iterations.length - 1];
	const latestMetrics = lastIterationEntry?.metrics ?? {};

	// Determine state
	const isConfigure = !isActive && !launching && endStatus == null;
	const isLive = isActive;
	const isComplete = !isActive && !launching && endStatus != null;

	// For trial iteration total, we derive from trial_update data or use a default
	const trialIterTotal = useMemo(() => {
		// Find any trial with iteration data to determine total
		for (const t of trials) {
			if (t.iterations.length > 0) {
				// The total iterations comes from the last update — look at the store
				// Since trial_update includes total_iterations, but we only store metrics,
				// we estimate from completed trials
				if (t.state === "complete" || t.state === "pruned") {
					return t.iterations.length;
				}
			}
		}
		return null;
	}, [trials]);

	return (
		<div className="space-y-4">
			<div>
				<h1 className="text-2xl font-bold text-[var(--text-primary)]">Alpha Search</h1>
				<p className="mt-1 text-sm text-[var(--text-secondary)]">
					{isConfigure && "Launch an Optuna HP optimization campaign"}
					{launching && "Starting campaign..."}
					{isLive && studyName && (
						<>
							Study: <span className="font-medium text-[var(--text-primary)]">{studyName}</span>
							{" | "}Trial {completedCount + prunedCount}/{totalTrials}
						</>
					)}
					{isComplete && "Campaign finished — review results below"}
				</p>
			</div>

			{/* Configure State */}
			{isConfigure && (
				<Card className="py-8">
					<CampaignConfigForm onLaunched={handleLaunched} />
				</Card>
			)}

			{/* Launching State — waiting for backend to start */}
			{launching && (
				<Card className="py-16">
					<div className="flex flex-col items-center gap-4 text-center">
						<div className="relative h-12 w-12">
							<div className="absolute inset-0 animate-ping rounded-full bg-[var(--accent-blue)]/20" />
							<div className="absolute inset-1 animate-spin rounded-full border-2 border-transparent border-t-[var(--accent-blue)]" />
							<div className="absolute inset-3 rounded-full bg-[var(--accent-blue)]/60" />
						</div>
						<div>
							<p className="text-lg font-semibold text-[var(--text-primary)]">
								Starting Alpha Search
							</p>
							<p className="mt-1 text-sm text-[var(--text-secondary)]">
								Initializing Optuna study
								{launchStudyName.current && (
									<>
										{" "}
										<span className="font-medium text-[var(--text-primary)]">
											{launchStudyName.current}
										</span>
									</>
								)}
								...
							</p>
							<p className="mt-3 text-xs text-[var(--text-secondary)]">
								Building environment, compiling model, and preparing first trial
							</p>
						</div>
					</div>
				</Card>
			)}

			{/* Complete State — Summary Banner */}
			{isComplete && studyName && (
				<CampaignSummaryBanner
					studyName={studyName}
					bestValue={bestValue}
					bestParams={bestParams}
					completedCount={completedCount}
					prunedCount={prunedCount}
					onNewCampaign={handleNewCampaign}
				/>
			)}

			{/* Live or Complete State — Charts & Panels */}
			{(isLive || isComplete) && (
				<>
					{/* Progress bar (only in live state) */}
					{isLive && (
						<Card>
							<CampaignProgressBar
								trialsCompleted={completedCount}
								trialsPruned={prunedCount}
								totalTrials={totalTrials}
								currentTrialNumber={currentTrialNumber}
								currentIteration={currentIteration}
								totalIterations={trialIterTotal}
								elapsedSeconds={elapsedSeconds}
								etaSeconds={etaSeconds}
							/>
						</Card>
					)}

					{/* 2x2 grid of charts */}
					<div className="grid grid-cols-2 gap-4">
						{/* Trial Curves */}
						<Card>
							<CardHeader title="Live Trial Curves" />
							<div className="h-72">
								<TrialCurvesChart
									trials={trialCurves}
									metricKey="pnl_mean"
									highlightedTrial={highlightedTrial}
									onTrialHover={handleTrialHover}
									onTrialClick={handleTrialClick}
								/>
							</div>
						</Card>

						{/* Convergence Scatter */}
						<Card>
							<CardHeader title="Convergence (Best So Far)" />
							<div className="h-72">
								<TrialOutcomesScatter
									trials={scatterTrials}
									onTrialClick={handleTrialClick}
									highlightedTrial={highlightedTrial}
								/>
							</div>
						</Card>

						{/* Param Importance */}
						<Card>
							<CardHeader title="Parameter Importance" />
							<div className="h-72">
								<ParamImportance importance={importance} />
							</div>
						</Card>

						{/* Current Trial Detail */}
						<Card>
							<CardHeader title="Current Trial Detail" />
							<div className="h-72 overflow-y-auto">
								<CurrentTrialPanel
									trialNumber={currentTrialNumber}
									params={currentTrial?.params ?? {}}
									latestMetrics={latestMetrics}
									iteration={currentIteration}
									totalIterations={trialIterTotal ?? 0}
								/>
							</div>
						</Card>
					</div>

					{/* Activity Feed */}
					<Card>
						<CardHeader title="Trial Activity Feed" />
						<TrialActivityFeed entries={activityFeed} />
					</Card>
				</>
			)}
		</div>
	);
}
