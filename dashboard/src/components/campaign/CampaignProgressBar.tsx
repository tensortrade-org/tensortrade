"use client";

interface CampaignProgressBarProps {
	trialsCompleted: number;
	trialsPruned: number;
	totalTrials: number;
	currentTrialNumber: number | null;
	currentIteration: number | null;
	totalIterations: number | null;
	elapsedSeconds: number;
	etaSeconds: number | null;
}

function formatDuration(seconds: number): string {
	if (seconds < 60) return `${Math.round(seconds)}s`;
	const m = Math.floor(seconds / 60);
	const s = Math.round(seconds % 60);
	if (m < 60) return `${m}m ${s}s`;
	const h = Math.floor(m / 60);
	return `${h}h ${m % 60}m`;
}

export function CampaignProgressBar({
	trialsCompleted,
	trialsPruned,
	totalTrials,
	currentTrialNumber,
	currentIteration,
	totalIterations,
	elapsedSeconds,
	etaSeconds,
}: CampaignProgressBarProps) {
	const trialsDone = trialsCompleted + trialsPruned;
	const campaignPct = totalTrials > 0 ? (trialsDone / totalTrials) * 100 : 0;
	const iterPct =
		currentIteration && totalIterations ? (currentIteration / totalIterations) * 100 : 0;

	return (
		<div className="space-y-3">
			{/* Campaign-level progress */}
			<div>
				<div className="mb-1 flex items-center justify-between text-xs">
					<span className="text-[var(--text-secondary)]">
						Campaign: {trialsDone}/{totalTrials} trials
					</span>
					<span className="text-[var(--text-secondary)]">
						Elapsed: {formatDuration(elapsedSeconds)}
						{etaSeconds != null && ` | ETA: ${formatDuration(etaSeconds)}`}
					</span>
				</div>
				<div className="h-2.5 w-full overflow-hidden rounded-full bg-[var(--bg-primary)]">
					<div
						className="h-full rounded-full bg-[var(--accent-blue)] transition-all duration-300"
						style={{ width: `${Math.min(campaignPct, 100)}%` }}
					/>
				</div>
				<div className="mt-1 flex gap-3 text-xs text-[var(--text-secondary)]">
					<span className="text-green-400">{trialsCompleted} completed</span>
					<span className="text-amber-400">{trialsPruned} pruned</span>
				</div>
			</div>

			{/* Current trial progress */}
			{currentTrialNumber != null && (
				<div>
					<div className="mb-1 flex items-center justify-between text-xs">
						<span className="text-[var(--text-secondary)]">
							Trial {currentTrialNumber}: iter {currentIteration ?? 0}/{totalIterations ?? "?"}
						</span>
						<span className="text-xs text-blue-400">Running</span>
					</div>
					<div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--bg-primary)]">
						<div
							className="h-full rounded-full bg-blue-500 transition-all duration-300"
							style={{ width: `${Math.min(iterPct, 100)}%` }}
						/>
					</div>
				</div>
			)}
		</div>
	);
}
