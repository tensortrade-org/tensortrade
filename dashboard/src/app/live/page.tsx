"use client";

import { ActionDistribution } from "@/components/charts/ActionDistribution";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import { PortfolioChart } from "@/components/charts/PortfolioChart";
import { Card, CardHeader } from "@/components/common/Card";
import { EpisodeProgressBar } from "@/components/inference/EpisodeProgressBar";
import { EpisodeSummaryCard } from "@/components/inference/EpisodeSummaryCard";
import { ExperimentSelector } from "@/components/inference/ExperimentSelector";
import { InferenceControls } from "@/components/inference/InferenceControls";
import { startInference } from "@/lib/api";
import { formatCurrency, formatNumber } from "@/lib/formatters";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useCallback, useMemo, useState } from "react";

const ACTION_LABELS: Record<number, string> = {
	0: "Hold",
	1: "Buy",
	2: "Sell",
};

export default function InferencePlaybackPage() {
	const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);

	const status = useInferenceStore((s) => s.status);
	const steps = useInferenceStore((s) => s.steps);
	const trades = useInferenceStore((s) => s.trades);
	const episodeSummary = useInferenceStore((s) => s.episodeSummary);
	const currentStep = useInferenceStore((s) => s.currentStep);
	const totalSteps = useInferenceStore((s) => s.totalSteps);
	const reset = useInferenceStore((s) => s.reset);

	const latestStep = steps[steps.length - 1];

	const actionCounts = useMemo(() => {
		let buy = 0;
		let sell = 0;
		let hold = 0;
		for (const step of steps) {
			if (step.action === 1) buy++;
			else if (step.action === 2) sell++;
			else hold++;
		}
		return { buy, sell, hold };
	}, [steps]);

	const handleRun = useCallback(async () => {
		if (!selectedExperiment) return;
		reset();
		try {
			await startInference(selectedExperiment, true);
		} catch (err) {
			console.error("Failed to start inference:", err);
		}
	}, [selectedExperiment, reset]);

	const handleReset = useCallback(() => {
		reset();
	}, [reset]);

	return (
		<div className="space-y-4">
			{/* Header row */}
			<div className="flex flex-wrap items-center justify-between gap-3">
				<div className="flex items-center gap-3">
					<h1 className="text-xl font-semibold text-[var(--text-primary)]">Inference Playback</h1>
					<ExperimentSelector value={selectedExperiment} onChange={setSelectedExperiment} />
				</div>
				<InferenceControls
					status={status}
					onRun={handleRun}
					onReset={handleReset}
					disabled={!selectedExperiment}
				/>
			</div>

			{/* Main area: Candlestick + Side panel */}
			<div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
				{/* Candlestick Chart */}
				<div className="lg:col-span-3">
					<Card className="h-[500px]">
						<CardHeader title="Price Action" />
						<div className="h-[calc(100%-2rem)]">
							<CandlestickChart steps={steps} trades={trades} />
						</div>
					</Card>
				</div>

				{/* Side Panel */}
				<div className="space-y-4 lg:col-span-1">
					<Card>
						<CardHeader title="Portfolio" />
						<div className="space-y-3">
							<div className="flex items-center justify-between">
								<span className="text-sm text-[var(--text-secondary)]">Net Worth</span>
								<span className="font-mono text-sm font-medium text-[var(--text-primary)]">
									{latestStep ? formatCurrency(latestStep.net_worth) : "--"}
								</span>
							</div>
							<div className="flex items-center justify-between">
								<span className="text-sm text-[var(--text-secondary)]">Price</span>
								<span className="font-mono text-sm text-[var(--text-primary)]">
									{latestStep ? formatCurrency(latestStep.close) : "--"}
								</span>
							</div>
							<div className="flex items-center justify-between">
								<span className="text-sm text-[var(--text-secondary)]">Step</span>
								<span className="font-mono text-sm text-[var(--text-primary)]">
									{latestStep ? formatNumber(latestStep.step) : "--"}
								</span>
							</div>
							<div className="flex items-center justify-between">
								<span className="text-sm text-[var(--text-secondary)]">P&L</span>
								<span
									className="font-mono text-sm font-medium"
									style={{
										color:
											latestStep && latestStep.net_worth >= 10000
												? "var(--accent-green)"
												: "var(--accent-red)",
									}}
								>
									{latestStep ? formatCurrency(latestStep.net_worth - 10000) : "--"}
								</span>
							</div>
							<div className="flex items-center justify-between">
								<span className="text-sm text-[var(--text-secondary)]">Action</span>
								<span className="font-mono text-sm text-[var(--text-primary)]">
									{latestStep?.action !== undefined
										? (ACTION_LABELS[latestStep.action] ?? `#${latestStep.action}`)
										: "--"}
								</span>
							</div>
						</div>
					</Card>

					<Card>
						<CardHeader title="Action Distribution" />
						<ActionDistribution
							buyCount={actionCounts.buy}
							sellCount={actionCounts.sell}
							holdCount={actionCounts.hold}
						/>
					</Card>
				</div>
			</div>

			{/* Progress Bar */}
			{status === "running" && (
				<Card>
					<EpisodeProgressBar currentStep={currentStep} totalSteps={totalSteps} />
				</Card>
			)}

			{/* Portfolio Chart */}
			<Card className="h-64">
				<CardHeader title="Net Worth Over Time" />
				<div className="h-[calc(100%-2rem)]">
					<PortfolioChart steps={steps} />
				</div>
			</Card>

			{/* Episode Summary */}
			{episodeSummary && <EpisodeSummaryCard summary={episodeSummary} />}
		</div>
	);
}
