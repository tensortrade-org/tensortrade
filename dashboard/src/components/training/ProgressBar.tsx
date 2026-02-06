"use client";

import { formatDuration } from "@/lib/formatters";
import { useTrainingStore } from "@/stores/trainingStore";

export function ProgressBar() {
	const progress = useTrainingStore((s) => s.progress);

	if (!progress) {
		return (
			<div className="space-y-2">
				<div className="flex items-center justify-between text-xs text-[var(--text-secondary)]">
					<span>No active training</span>
				</div>
				<div className="h-2 w-full rounded-full bg-[var(--bg-secondary)]" />
			</div>
		);
	}

	const { iteration, total_iterations, elapsed_seconds, eta_seconds } = progress;
	const percent = total_iterations > 0 ? Math.min((iteration / total_iterations) * 100, 100) : 0;

	return (
		<div className="space-y-2">
			<div className="flex items-center justify-between text-xs">
				<span className="font-medium text-[var(--text-primary)]">
					Iteration {iteration} / {total_iterations}
				</span>
				<div className="flex items-center gap-3 text-[var(--text-secondary)]">
					<span>Elapsed: {formatDuration(elapsed_seconds)}</span>
					{eta_seconds !== null && <span>ETA: {formatDuration(eta_seconds)}</span>}
				</div>
			</div>

			<div className="h-2 w-full overflow-hidden rounded-full bg-[var(--bg-secondary)]">
				<div
					className="h-full rounded-full bg-[var(--accent-blue)] transition-all duration-300"
					style={{ width: `${percent}%` }}
				/>
			</div>

			<div className="flex justify-end">
				<span className="text-xs font-mono text-[var(--text-secondary)]">
					{percent.toFixed(1)}%
				</span>
			</div>
		</div>
	);
}
