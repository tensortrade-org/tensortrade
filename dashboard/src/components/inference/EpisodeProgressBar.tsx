"use client";

interface EpisodeProgressBarProps {
	currentStep: number;
	totalSteps: number;
}

export function EpisodeProgressBar({ currentStep, totalSteps }: EpisodeProgressBarProps) {
	const pct = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;
	const displayPct = Math.min(pct, 100);

	return (
		<div className="flex items-center gap-3">
			<div className="relative h-2 flex-1 overflow-hidden rounded-full bg-[var(--bg-secondary)]">
				<div
					className="h-full rounded-full bg-[var(--accent-blue)] transition-all duration-300"
					style={{ width: `${displayPct}%` }}
				/>
			</div>
			<span className="whitespace-nowrap font-mono text-xs text-[var(--text-secondary)]">
				Step {currentStep}
				{totalSteps > 0 ? `/${totalSteps}` : ""} {totalSteps > 0 ? `${displayPct.toFixed(0)}%` : ""}
			</span>
		</div>
	);
}
