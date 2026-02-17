"use client";

import Link from "next/link";

interface CampaignSummaryBannerProps {
	studyName: string;
	bestValue: number | null;
	bestParams: Record<string, number | string>;
	completedCount: number;
	prunedCount: number;
	onNewCampaign: () => void;
}

export function CampaignSummaryBanner({
	studyName,
	bestValue,
	bestParams,
	completedCount,
	prunedCount,
	onNewCampaign,
}: CampaignSummaryBannerProps) {
	const paramEntries = Object.entries(bestParams);

	return (
		<div className="rounded-lg border border-green-500/30 bg-green-500/10 p-4">
			<div className="mb-3 flex items-center justify-between">
				<div className="flex items-center gap-2">
					<span className="text-lg text-green-400">&#10003;</span>
					<h3 className="text-sm font-semibold text-green-400">Campaign Complete</h3>
				</div>
				<div className="flex gap-2">
					<Link
						href={`/optuna/${encodeURIComponent(studyName)}`}
						className="rounded-md border border-[var(--border-color)] px-3 py-1.5 text-xs text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-primary)]"
					>
						View Full Analysis
					</Link>
					<button
						type="button"
						onClick={onNewCampaign}
						className="rounded-md bg-[var(--accent-blue)] px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-[var(--accent-blue)]/80"
					>
						New Campaign
					</button>
				</div>
			</div>

			<div className="grid grid-cols-3 gap-4 text-sm">
				<div>
					<p className="text-xs text-[var(--text-secondary)]">Best Value</p>
					<p className="font-mono text-lg font-bold text-[var(--text-primary)]">
						{bestValue != null ? `$${bestValue.toFixed(0)}` : "N/A"}
					</p>
				</div>
				<div>
					<p className="text-xs text-[var(--text-secondary)]">Trials Completed</p>
					<p className="font-mono text-lg font-bold text-green-400">{completedCount}</p>
				</div>
				<div>
					<p className="text-xs text-[var(--text-secondary)]">Trials Pruned</p>
					<p className="font-mono text-lg font-bold text-amber-400">{prunedCount}</p>
				</div>
			</div>

			{paramEntries.length > 0 && (
				<div className="mt-3 border-t border-green-500/20 pt-3">
					<p className="mb-1 text-xs font-medium text-[var(--text-secondary)]">
						Best Hyperparameters
					</p>
					<div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs">
						{paramEntries.map(([key, val]) => (
							<div key={key} className="flex justify-between">
								<span className="text-[var(--text-secondary)]">{key}</span>
								<span className="font-mono text-[var(--text-primary)]">
									{typeof val === "number"
										? Math.abs(val) < 0.01
											? val.toExponential(2)
											: Number.isInteger(val)
												? String(val)
												: val.toFixed(4)
										: val}
								</span>
							</div>
						))}
					</div>
				</div>
			)}
		</div>
	);
}
