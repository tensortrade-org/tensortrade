"use client";

import { useApi } from "@/hooks/useApi";
import { getDatasets, getHyperparamPacks, launchTraining } from "@/lib/api";
import type { DatasetConfig, HyperparameterPack, LaunchRequest, TrainingConfig } from "@/lib/types";
import { useLaunchStore } from "@/stores/launchStore";
import { useTrainingStore } from "@/stores/trainingStore";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef } from "react";

interface ReviewCardProps {
	title: string;
	children: React.ReactNode;
}

function ReviewCard({ title, children }: ReviewCardProps) {
	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] p-4">
			<h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
				{title}
			</h4>
			{children}
		</div>
	);
}

interface ReviewRowProps {
	label: string;
	value: string;
}

function ReviewRow({ label, value }: ReviewRowProps) {
	return (
		<div className="flex justify-between py-1 text-sm">
			<span className="text-[var(--text-secondary)]">{label}</span>
			<span className="font-mono text-[var(--text-primary)]">{value}</span>
		</div>
	);
}

function formatOverrideValue(key: string, value: unknown): string {
	if (typeof value === "number") {
		if (key === "learning_rate") return (value as number).toExponential(1);
		if (key === "commission") return String(value);
		return String(value);
	}
	if (value && typeof value === "object") {
		return JSON.stringify(value);
	}
	return String(value);
}

export function ReviewStep() {
	const name = useLaunchStore((s) => s.name);
	const tags = useLaunchStore((s) => s.tags);
	const datasetId = useLaunchStore((s) => s.datasetId);
	const hpPackId = useLaunchStore((s) => s.hpPackId);
	const overrides = useLaunchStore((s) => s.overrides);
	const isLaunching = useLaunchStore((s) => s.isLaunching);
	const launchError = useLaunchStore((s) => s.launchError);
	const launchedExperimentId = useLaunchStore((s) => s.launchedExperimentId);
	const setLaunching = useLaunchStore((s) => s.setLaunching);
	const setLaunchError = useLaunchStore((s) => s.setLaunchError);
	const setLaunchedExperimentId = useLaunchStore((s) => s.setLaunchedExperimentId);
	const reset = useLaunchStore((s) => s.reset);

	const startWarmingUp = useTrainingStore((s) => s.startWarmingUp);
	const router = useRouter();
	const redirectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

	// Auto-navigate to homepage after successful launch
	useEffect(() => {
		if (launchedExperimentId) {
			redirectTimerRef.current = setTimeout(() => {
				router.push("/training");
			}, 3000);
		}
		return () => {
			if (redirectTimerRef.current) {
				clearTimeout(redirectTimerRef.current);
			}
		};
	}, [launchedExperimentId, router]);

	const datasetFetcher = useCallback(() => getDatasets(), []);
	const { data: datasets } = useApi<DatasetConfig[]>(datasetFetcher, []);
	const selectedDataset = datasets?.find((d) => d.id === datasetId) ?? null;

	const packFetcher = useCallback(() => getHyperparamPacks(), []);
	const { data: packs } = useApi<HyperparameterPack[]>(packFetcher, []);
	const selectedPack = packs?.find((p) => p.id === hpPackId) ?? null;

	const overrideEntries = Object.entries(overrides).filter(
		(entry): entry is [string, string | number] => entry[1] !== undefined,
	);

	const handleLaunch = async () => {
		if (!datasetId || !hpPackId) return;

		setLaunching(true);
		setLaunchError(null);

		try {
			const request: LaunchRequest = {
				name,
				hp_pack_id: hpPackId,
				dataset_id: datasetId,
				tags,
				overrides: overrideEntries.length > 0 ? (overrides as Partial<TrainingConfig>) : undefined,
			};
			const response = await launchTraining(request);
			setLaunchedExperimentId(response.experiment_id);
			startWarmingUp(response.experiment_id);
		} catch (err) {
			const message = err instanceof Error ? err.message : "Failed to launch training";
			setLaunchError(message);
		} finally {
			setLaunching(false);
		}
	};

	if (launchedExperimentId) {
		return (
			<div className="flex flex-col items-center gap-4 py-8">
				<div className="flex h-16 w-16 items-center justify-center rounded-full bg-[var(--accent-green)]/15">
					<span className="text-3xl text-[var(--accent-green)]">&#10003;</span>
				</div>
				<h3 className="text-lg font-semibold text-[var(--text-primary)]">Training Launched</h3>
				<p className="text-sm text-[var(--text-secondary)]">
					Experiment ID:{" "}
					<code className="rounded bg-[var(--bg-secondary)] px-2 py-0.5 font-mono text-xs text-[var(--accent-blue)]">
						{launchedExperimentId}
					</code>
				</p>
				<p className="text-xs text-[var(--text-secondary)]">
					Redirecting to dashboard in a few seconds...
				</p>
				<div className="flex gap-3">
					<button
						type="button"
						onClick={() => router.push("/training")}
						className="rounded-md bg-[var(--accent-blue)] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[var(--accent-blue)]/80"
					>
						View Dashboard Now
					</button>
					<button
						type="button"
						onClick={() => {
							if (redirectTimerRef.current) {
								clearTimeout(redirectTimerRef.current);
							}
							reset();
						}}
						className="rounded-md border border-[var(--border-color)] px-4 py-2 text-sm font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-secondary)]"
					>
						Launch Another
					</button>
				</div>
			</div>
		);
	}

	return (
		<div className="space-y-4">
			<h3 className="text-sm font-medium text-[var(--text-primary)]">Review Configuration</h3>

			<div className="grid grid-cols-2 gap-4">
				<ReviewCard title="Experiment">
					<ReviewRow label="Name" value={name} />
					<ReviewRow label="Tags" value={tags.length > 0 ? tags.join(", ") : "None"} />
				</ReviewCard>

				<ReviewCard title="Dataset">
					<ReviewRow label="Dataset" value={selectedDataset?.name ?? datasetId ?? "Unknown"} />
					{selectedDataset && (
						<>
							<ReviewRow label="Source" value={selectedDataset.source_type} />
							<ReviewRow label="Features" value={String(selectedDataset.features.length)} />
						</>
					)}
				</ReviewCard>

				<ReviewCard title="Hyperparameters">
					<ReviewRow label="Pack" value={selectedPack?.name ?? hpPackId ?? "Unknown"} />
					{selectedPack && (
						<>
							<ReviewRow label="Algorithm" value={selectedPack.config.algorithm} />
							<ReviewRow
								label="Learning Rate"
								value={selectedPack.config.learning_rate.toExponential(1)}
							/>
							<ReviewRow label="Iterations" value={String(selectedPack.config.num_iterations)} />
						</>
					)}
				</ReviewCard>

				<ReviewCard title="Overrides">
					{overrideEntries.length === 0 ? (
						<p className="text-sm text-[var(--text-secondary)]">No overrides set</p>
					) : (
						overrideEntries.map(([key, value]) => (
							<ReviewRow key={key} label={key} value={formatOverrideValue(key, value)} />
						))
					)}
				</ReviewCard>
			</div>

			{launchError && (
				<div className="rounded-md border border-[var(--accent-red)]/30 bg-[var(--accent-red)]/10 p-3">
					<p className="text-sm text-[var(--accent-red)]">{launchError}</p>
				</div>
			)}

			<div className="flex justify-end">
				<button
					type="button"
					onClick={handleLaunch}
					disabled={isLaunching || !datasetId || !hpPackId}
					className="rounded-md bg-[var(--accent-green)] px-6 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[var(--accent-green)]/80 disabled:cursor-not-allowed disabled:opacity-40"
				>
					{isLaunching ? "Launching..." : "Launch Training"}
				</button>
			</div>
		</div>
	);
}
