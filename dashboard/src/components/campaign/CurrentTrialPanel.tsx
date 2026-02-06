"use client";

interface CurrentTrialPanelProps {
	trialNumber: number | null;
	params: Record<string, number | string>;
	latestMetrics: Record<string, number>;
	iteration: number;
	totalIterations: number;
}

function formatParamValue(value: number | string): string {
	if (typeof value === "string") return value;
	if (Number.isInteger(value)) return String(value);
	if (Math.abs(value) < 0.01) return value.toExponential(2);
	return value.toFixed(4);
}

export function CurrentTrialPanel({
	trialNumber,
	params,
	latestMetrics,
	iteration,
	totalIterations,
}: CurrentTrialPanelProps) {
	if (trialNumber == null) {
		return (
			<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
				Waiting for next trial...
			</div>
		);
	}

	const paramEntries = Object.entries(params);
	const metricEntries = Object.entries(latestMetrics).filter(
		([, v]) => v !== 0 && !Number.isNaN(v),
	);

	return (
		<div className="space-y-3">
			<div className="flex items-center justify-between">
				<span className="text-sm font-medium text-[var(--text-primary)]">Trial {trialNumber}</span>
				<span className="text-xs text-blue-400">
					iter {iteration}/{totalIterations}
				</span>
			</div>

			{/* Sampled parameters */}
			<div>
				<p className="mb-1 text-xs font-medium text-[var(--text-secondary)]">Sampled Parameters</p>
				<div className="grid grid-cols-2 gap-x-3 gap-y-1">
					{paramEntries.map(([key, val]) => (
						<div key={key} className="flex justify-between text-xs">
							<span className="text-[var(--text-secondary)] truncate mr-1">{key}</span>
							<span className="font-mono text-[var(--text-primary)]">{formatParamValue(val)}</span>
						</div>
					))}
				</div>
			</div>

			{/* Live metrics */}
			{metricEntries.length > 0 && (
				<div>
					<p className="mb-1 text-xs font-medium text-[var(--text-secondary)]">Latest Metrics</p>
					<div className="grid grid-cols-2 gap-x-3 gap-y-1">
						{metricEntries.slice(0, 6).map(([key, val]) => (
							<div key={key} className="flex justify-between text-xs">
								<span className="text-[var(--text-secondary)] truncate mr-1">{key}</span>
								<span className="font-mono text-[var(--text-primary)]">
									{Math.abs(val) > 100 ? val.toFixed(0) : val.toFixed(2)}
								</span>
							</div>
						))}
					</div>
				</div>
			)}
		</div>
	);
}
