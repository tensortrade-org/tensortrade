"use client";

interface ActivityEntry {
	id: number;
	trialNumber: number;
	event: "complete" | "pruned" | "started";
	value: number | null;
	isBest: boolean;
	iteration: number | null;
	timestamp: number;
}

interface TrialActivityFeedProps {
	entries: ActivityEntry[];
}

export function TrialActivityFeed({ entries }: TrialActivityFeedProps) {
	if (entries.length === 0) {
		return (
			<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
				Waiting for trial results...
			</div>
		);
	}

	return (
		<div className="max-h-48 space-y-1 overflow-y-auto pr-1">
			{entries.map((entry) => (
				<div key={entry.id} className="flex items-center gap-2 rounded px-2 py-1 text-xs">
					{entry.event === "complete" && (
						<>
							<span className="text-green-400">&#10003;</span>
							<span className="text-[var(--text-secondary)]">Trial {entry.trialNumber}:</span>
							<span className="font-medium text-green-400">COMPLETE</span>
							{entry.value != null && (
								<span className="text-[var(--text-primary)]">
									val_pnl=${entry.value.toFixed(0)}
								</span>
							)}
							{entry.isBest && <span className="font-medium text-amber-400">New Best!</span>}
						</>
					)}
					{entry.event === "pruned" && (
						<>
							<span className="text-amber-400">&#9986;</span>
							<span className="text-[var(--text-secondary)]">Trial {entry.trialNumber}:</span>
							<span className="font-medium text-amber-400">PRUNED</span>
							{entry.iteration != null && (
								<span className="text-[var(--text-secondary)]">at iter {entry.iteration}</span>
							)}
						</>
					)}
					{entry.event === "started" && (
						<>
							<span className="text-blue-400">&#9654;</span>
							<span className="text-[var(--text-secondary)]">Trial {entry.trialNumber}:</span>
							<span className="text-blue-400">STARTED</span>
						</>
					)}
				</div>
			))}
		</div>
	);
}
