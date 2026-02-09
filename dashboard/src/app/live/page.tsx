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
import { TradeList } from "@/components/inference/TradeList";
import { createHyperparamPack, getExperiment, startInference } from "@/lib/api";
import { formatCurrency, formatDate, formatNumber, formatPercent } from "@/lib/formatters";
import type { ExperimentDetail, TrainingConfig } from "@/lib/types";
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
	const [experimentDetail, setExperimentDetail] = useState<ExperimentDetail | null>(null);

	// Auto-select experiment from URL query param (e.g. from Optuna "Run Inference")
	useEffect(() => {
		const expId = searchParams.get("experiment_id");
		if (expId && !selectedExperiment) {
			setSelectedExperiment(expId);
		}
	}, [searchParams, selectedExperiment]);

	// Fetch experiment details when selection changes
	useEffect(() => {
		if (!selectedExperiment) {
			setExperimentDetail(null);
			return;
		}
		let cancelled = false;
		getExperiment(selectedExperiment).then((detail) => {
			if (!cancelled) setExperimentDetail(detail);
		});
		return () => {
			cancelled = true;
		};
	}, [selectedExperiment]);

	const status = useInferenceStore((s) => s.status);
	const steps = useInferenceStore((s) => s.steps);
	const trades = useInferenceStore((s) => s.trades);
	const episodeSummary = useInferenceStore((s) => s.episodeSummary);
	const currentStep = useInferenceStore((s) => s.currentStep);
	const totalSteps = useInferenceStore((s) => s.totalSteps);
	const datasetName = useInferenceStore((s) => s.datasetName);
	const setStarting = useInferenceStore((s) => s.setStarting);
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
		setStarting();
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
	}, [selectedExperiment, selectedDataset, startDate, endDate, reset, setStarting]);

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
					<Card className="h-[670px]">
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
			{(status === "starting" || status === "running") && (
				<Card>
					<EpisodeProgressBar currentStep={currentStep} totalSteps={totalSteps} />
				</Card>
			)}

			{/* Trade List + Portfolio Chart */}
			<div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
				<Card className="h-64">
					<CardHeader title={`Trades (${trades.length})`} />
					<div className="h-[calc(100%-2rem)]">
						<TradeList trades={trades} />
					</div>
				</Card>
				<Card className="h-64">
					<CardHeader title="Net Worth Over Time" />
					<div className="h-[calc(100%-2rem)]">
						<PortfolioChart steps={steps} />
					</div>
				</Card>
			</div>

			{/* Episode Summary */}
			{episodeSummary && <EpisodeSummaryCard summary={episodeSummary} />}

			{/* Experiment Config */}
			{experimentDetail && (
				<ExperimentConfigPanel detail={experimentDetail} showSave={status === "completed"} />
			)}
		</div>
	);
}

interface ConfigPanelProps {
	detail: ExperimentDetail;
	showSave: boolean;
}

interface ConfigRecord {
	training_config?: Record<string, unknown>;
	hp_pack_name?: string;
	dataset_name?: string;
	source_type?: string;
	source_config?: Record<string, unknown>;
	features?: Array<Record<string, unknown>>;
	split_config?: Record<string, number>;
	[key: string]: unknown;
}

function ConfigValue({ value }: { value: unknown }) {
	if (value === null || value === undefined) return <span>--</span>;
	if (typeof value === "boolean") return <span>{value ? "true" : "false"}</span>;
	if (typeof value === "number") return <span>{value}</span>;
	if (typeof value === "string") return <span>{value}</span>;
	if (Array.isArray(value)) return <span>{value.join(", ")}</span>;
	if (typeof value === "object") return <span>{JSON.stringify(value)}</span>;
	return <span>{String(value)}</span>;
}

function ConfigSection({ title, entries }: { title: string; entries: [string, unknown][] }) {
	if (entries.length === 0) return null;
	return (
		<div>
			<h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
				{title}
			</h4>
			<div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3 lg:grid-cols-4">
				{entries.map(([key, val]) => (
					<div key={key} className="flex items-baseline justify-between gap-2 py-0.5">
						<span className="text-xs text-[var(--text-secondary)]">{key}</span>
						<span className="font-mono text-xs text-[var(--text-primary)]">
							<ConfigValue value={val} />
						</span>
					</div>
				))}
			</div>
		</div>
	);
}

function ExperimentConfigPanel({ detail, showSave }: ConfigPanelProps) {
	const { experiment } = detail;
	const config = experiment.config as ConfigRecord;
	const tc = config.training_config ?? {};
	const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
	const [packName, setPackName] = useState(`${experiment.name} (inference)`);

	// Separate model from the rest of training config
	const { model, ...tcRest } = tc as Record<string, unknown> & { model?: Record<string, unknown> };
	const tcEntries = Object.entries(tcRest);
	const modelEntries = model ? Object.entries(model) : [];

	const metaEntries: [string, unknown][] = [
		["name", experiment.name],
		["status", experiment.status],
		["started", formatDate(experiment.started_at)],
		["hp_pack", config.hp_pack_name],
		["dataset", config.dataset_name],
		["source_type", config.source_type],
	];

	const sourceEntries = config.source_config ? Object.entries(config.source_config) : [];
	const featureEntries: [string, unknown][] = (config.features ?? []).map((f, i) => [
		`feature_${i + 1}`,
		f.type
			? `${f.type} (${Object.entries(f)
					.filter(([k]) => k !== "type")
					.map(([k, v]) => `${k}=${v}`)
					.join(", ")})`
			: JSON.stringify(f),
	]);
	const splitEntries = config.split_config ? Object.entries(config.split_config) : [];

	const handleSavePack = async () => {
		const trainingConfig = config.training_config;
		if (!trainingConfig || !packName.trim()) return;
		setSaveState("saving");
		try {
			await createHyperparamPack({
				name: packName.trim(),
				description: `Saved from inference run of ${experiment.name}`,
				config: trainingConfig as TrainingConfig,
			});
			setSaveState("saved");
		} catch {
			setSaveState("error");
		}
	};

	return (
		<Card>
			<div className="flex items-center justify-between">
				<CardHeader title="Experiment Configuration" />
				{showSave && config.training_config && (
					<div className="mr-1 flex items-center gap-2">
						<input
							type="text"
							value={packName}
							onChange={(e) => {
								setPackName(e.target.value);
								if (saveState === "saved") setSaveState("idle");
							}}
							disabled={saveState === "saving"}
							placeholder="HP pack name"
							className="w-56 rounded-md border border-[var(--border-primary)] bg-[var(--bg-primary)] px-2 py-1.5 text-xs text-[var(--text-primary)] disabled:opacity-50"
						/>
						<button
							type="button"
							onClick={handleSavePack}
							disabled={saveState === "saving" || saveState === "saved" || !packName.trim()}
							className="whitespace-nowrap rounded-md bg-[var(--accent-green)] px-3 py-1.5 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
						>
							{saveState === "saving"
								? "Saving..."
								: saveState === "saved"
									? "Saved"
									: saveState === "error"
										? "Failed — Retry"
										: "Save as HP Pack"}
						</button>
					</div>
				)}
			</div>
			<div className="space-y-4">
				<ConfigSection title="Experiment" entries={metaEntries} />
				<ConfigSection title="Training" entries={tcEntries} />
				{modelEntries.length > 0 && <ConfigSection title="Model" entries={modelEntries} />}
				<ConfigSection title="Data Source" entries={sourceEntries} />
				<ConfigSection title="Features" entries={featureEntries} />
				{splitEntries.length > 0 && <ConfigSection title="Split" entries={splitEntries} />}
			</div>
		</Card>
	);
}
