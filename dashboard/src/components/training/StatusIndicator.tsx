"use client";

import { formatNumber } from "@/lib/formatters";
import { useTrainingStore } from "@/stores/trainingStore";

interface StatusConfig {
	label: string;
	dotColor: string;
	textColor: string;
}

function getStatusConfig(isTraining: boolean, isPaused: boolean): StatusConfig {
	if (isTraining && isPaused) {
		return {
			label: "Paused",
			dotColor: "bg-[var(--accent-amber)]",
			textColor: "text-[var(--accent-amber)]",
		};
	}
	if (isTraining) {
		return {
			label: "Training",
			dotColor: "bg-[var(--accent-green)]",
			textColor: "text-[var(--accent-green)]",
		};
	}
	return {
		label: "Idle",
		dotColor: "bg-[var(--text-secondary)]",
		textColor: "text-[var(--text-secondary)]",
	};
}

export function StatusIndicator() {
	const status = useTrainingStore((state) => state.status);
	const currentIteration = useTrainingStore((state) => state.currentIteration);
	const isWarmingUp = useTrainingStore((state) => state.isWarmingUp);

	if (isWarmingUp) {
		return (
			<div className="flex flex-col gap-2">
				<div className="flex items-center gap-2">
					<div className="h-2 w-2 animate-pulse rounded-full bg-[var(--accent-amber)]" />
					<span className="text-sm font-medium text-[var(--accent-amber)]">Starting</span>
				</div>
				<p className="text-xs text-[var(--text-secondary)]">Initializing training environment...</p>
			</div>
		);
	}

	if (!status) {
		return (
			<div className="flex items-center gap-2">
				<div className="h-2 w-2 rounded-full bg-[var(--text-secondary)]" />
				<span className="text-sm text-[var(--text-secondary)]">Disconnected</span>
			</div>
		);
	}

	const config = getStatusConfig(status.is_training, status.is_paused);

	return (
		<div className="flex flex-col gap-2">
			<div className="flex items-center gap-2">
				<div
					className={`h-2 w-2 rounded-full ${config.dotColor} ${
						status.is_training && !status.is_paused ? "animate-pulse" : ""
					}`}
				/>
				<span className={`text-sm font-medium ${config.textColor}`}>{config.label}</span>
			</div>

			<div className="flex flex-wrap items-center gap-4 text-xs text-[var(--text-secondary)]">
				{status.is_training && (
					<span>
						Iteration:{" "}
						<span className="font-medium text-[var(--text-primary)]">
							{formatNumber(currentIteration)}
						</span>
					</span>
				)}
				<span>
					Producers:{" "}
					<span className="font-medium text-[var(--text-primary)]">
						{formatNumber(status.training_producers)}
					</span>
				</span>
				<span>
					Clients:{" "}
					<span className="font-medium text-[var(--text-primary)]">
						{formatNumber(status.dashboard_clients)}
					</span>
				</span>
			</div>
		</div>
	);
}
