"use client";

type ParamValue = number | string | boolean | null | undefined;

interface CurrentTrialPanelProps {
	trialNumber: number | null;
	params: Record<string, ParamValue>;
	latestMetrics: Record<string, number>;
	iteration: number;
	totalIterations: number;
}

function formatParamValue(value: ParamValue): string {
	if (value == null) return "—";
	if (typeof value === "boolean") return value ? "true" : "false";
	if (typeof value === "string") return value;
	if (typeof value !== "number" || Number.isNaN(value)) return String(value);
	if (Number.isInteger(value)) return String(value);
	if (Math.abs(value) < 0.01) return value.toExponential(2);
	return value.toFixed(4);
}

function formatMetricValue(value: unknown): string {
	if (typeof value !== "number" || Number.isNaN(value)) return String(value ?? "—");
	if (Math.abs(value) > 100) return value.toFixed(0);
	return value.toFixed(2);
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
									{formatMetricValue(val)}
								</span>
							</div>
						))}
					</div>
				</div>
			)}
		</div>
	);
}
