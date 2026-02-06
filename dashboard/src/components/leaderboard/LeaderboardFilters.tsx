"use client";

interface LeaderboardFiltersProps {
	scripts: string[];
	onScriptChange: (script: string) => void;
	onMetricChange: (metric: string) => void;
	currentScript: string;
	currentMetric: string;
}

interface MetricOption {
	value: string;
	label: string;
}

const METRIC_OPTIONS: MetricOption[] = [
	{ value: "pnl", label: "PnL" },
	{ value: "net_worth", label: "Net Worth" },
	{ value: "sharpe_ratio", label: "Sharpe Ratio" },
	{ value: "episode_return", label: "Episode Return" },
];

export function LeaderboardFilters({
	scripts,
	onScriptChange,
	onMetricChange,
	currentScript,
	currentMetric,
}: LeaderboardFiltersProps) {
	return (
		<div className="flex flex-wrap items-center gap-3">
			<div className="flex flex-col gap-1">
				<label htmlFor="script-filter" className="text-xs font-medium text-[var(--text-secondary)]">
					Script
				</label>
				<select
					id="script-filter"
					value={currentScript}
					onChange={(e) => onScriptChange(e.target.value)}
					className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent-blue)]"
				>
					<option value="">All Scripts</option>
					{scripts.map((script) => (
						<option key={script} value={script}>
							{script}
						</option>
					))}
				</select>
			</div>

			<div className="flex flex-col gap-1">
				<label htmlFor="metric-filter" className="text-xs font-medium text-[var(--text-secondary)]">
					Metric
				</label>
				<select
					id="metric-filter"
					value={currentMetric}
					onChange={(e) => onMetricChange(e.target.value)}
					className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent-blue)]"
				>
					{METRIC_OPTIONS.map((opt) => (
						<option key={opt.value} value={opt.value}>
							{opt.label}
						</option>
					))}
				</select>
			</div>
		</div>
	);
}
