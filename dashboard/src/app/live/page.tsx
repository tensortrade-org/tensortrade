"use client";

import { ActionDistribution } from "@/components/charts/ActionDistribution";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import { PortfolioChart } from "@/components/charts/PortfolioChart";
import { Card, CardHeader } from "@/components/common/Card";
import { DatasetSelector } from "@/components/inference/DatasetSelector";
import { EpisodeProgressBar } from "@/components/inference/EpisodeProgressBar";
import { EpisodeSummaryCard } from "@/components/inference/EpisodeSummaryCard";
import { ExperimentSelector } from "@/components/inference/ExperimentSelector";
import { InferenceControls } from "@/components/inference/InferenceControls";
import { startInference } from "@/lib/api";
import { formatCurrency, formatNumber, formatPercent } from "@/lib/formatters";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

const ACTION_LABELS: Record<number, string> = {
	0: "Hold",
	1: "Buy",
	2: "Sell",
};

export default function InferencePlaybackPage() {
	const searchParams = useSearchParams();
	const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
	const [selectedDataset, setSelectedDataset] = useState("");
	const [startDate, setStartDate] = useState("");
	const [endDate, setEndDate] = useState("");

	// Auto-select experiment from URL query param (e.g. from Optuna "Run Inference")
	useEffect(() => {
		const expId = searchParams.get("experiment_id");
		if (expId && !selectedExperiment) {
			setSelectedExperiment(expId);
		}
	}, [searchParams, selectedExperiment]);

	const status = useInferenceStore((s) => s.status);
	const steps = useInferenceStore((s) => s.steps);
	const trades = useInferenceStore((s) => s.trades);
	const episodeSummary = useInferenceStore((s) => s.episodeSummary);
	const currentStep = useInferenceStore((s) => s.currentStep);
	const totalSteps = useInferenceStore((s) => s.totalSteps);
	const datasetName = useInferenceStore((s) => s.datasetName);
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

	const liveStats = useMemo(() => {
		if (steps.length === 0) {
			return {
				holdRatio: 0,
				tradeRatio: 0,
				pnlPerTrade: 0,
				maxDrawdownPct: 0,
			};
		}
		const totalActions = actionCounts.buy + actionCounts.sell + actionCounts.hold;
		const holdRatio = totalActions > 0 ? actionCounts.hold / totalActions : 0;
		const tradeRatio = totalActions > 0 ? (actionCounts.buy + actionCounts.sell) / totalActions : 0;

		const initialNetWorth = steps[0].net_worth;
		const latestNetWorth = steps[steps.length - 1].net_worth;
		const tradeCount = trades.length;
		const pnlPerTrade = tradeCount > 0 ? (latestNetWorth - initialNetWorth) / tradeCount : 0;

		let peak = steps[0].net_worth;
		let maxDrawdownPct = 0;
		for (const s of steps) {
			peak = Math.max(peak, s.net_worth);
			if (peak > 0) {
				const dd = ((peak - s.net_worth) / peak) * 100;
				maxDrawdownPct = Math.max(maxDrawdownPct, dd);
			}
		}

		return {
			holdRatio,
			tradeRatio,
			pnlPerTrade,
			maxDrawdownPct,
		};
	}, [steps, trades.length, actionCounts.buy, actionCounts.sell, actionCounts.hold]);

	const handleRun = useCallback(async () => {
		if (!selectedExperiment) return;
		reset();
		try {
			await startInference(
				selectedExperiment,
				selectedDataset || undefined,
				startDate || undefined,
				endDate || undefined,
			);
		} catch (err) {
			console.error("Failed to start inference:", err);
		}
	}, [selectedExperiment, selectedDataset, startDate, endDate, reset]);

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
					<DatasetSelector value={selectedDataset} onChange={setSelectedDataset} />
					<input
						type="date"
						value={startDate}
						onChange={(e) => setStartDate(e.target.value)}
						placeholder="Start"
						className="rounded-md border border-[var(--border-primary)] bg-[var(--bg-primary)] px-2 py-1.5 text-sm text-[var(--text-primary)]"
					/>
					<span className="text-sm text-[var(--text-secondary)]">to</span>
					<input
						type="date"
						value={endDate}
						onChange={(e) => setEndDate(e.target.value)}
						placeholder="End"
						className="rounded-md border border-[var(--border-primary)] bg-[var(--bg-primary)] px-2 py-1.5 text-sm text-[var(--text-primary)]"
					/>
				</div>
				<div className="flex items-center gap-3">
					{datasetName && status !== "idle" && (
						<span className="rounded-md bg-[var(--bg-secondary)] px-2 py-1 text-xs text-[var(--text-secondary)]">
							{datasetName}
						</span>
					)}
					<InferenceControls
						status={status}
						onRun={handleRun}
						onReset={handleReset}
						disabled={!selectedExperiment}
					/>
				</div>
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

					<Card>
						<CardHeader title="Execution Quality" />
						<div className="space-y-3">
							<div className="flex items-center justify-between">
								<span className="text-sm text-[var(--text-secondary)]">Trade Ratio</span>
								<span className="font-mono text-sm text-[var(--text-primary)]">
									{formatPercent(liveStats.tradeRatio * 100)}
								</span>
							</div>
							<div className="flex items-center justify-between">
								<span className="text-sm text-[var(--text-secondary)]">Hold Ratio</span>
								<span className="font-mono text-sm text-[var(--text-primary)]">
									{formatPercent(liveStats.holdRatio * 100)}
								</span>
							</div>
							<div className="flex items-center justify-between">
								<span className="text-sm text-[var(--text-secondary)]">PnL / Trade</span>
								<span
									className="font-mono text-sm font-medium"
									style={{
										color: liveStats.pnlPerTrade >= 0 ? "var(--accent-green)" : "var(--accent-red)",
									}}
								>
									{liveStats.pnlPerTrade >= 0 ? "+" : ""}
									{formatCurrency(liveStats.pnlPerTrade)}
								</span>
							</div>
							<div className="flex items-center justify-between">
								<span className="text-sm text-[var(--text-secondary)]">Max Drawdown</span>
								<span className="font-mono text-sm text-[var(--accent-red)]">
									{liveStats.maxDrawdownPct.toFixed(2)}%
								</span>
							</div>
						</div>
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
