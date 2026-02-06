import { Card } from "@/components/common/Card";
import { formatCurrency, formatNumber, formatPercent, formatPnl } from "@/lib/formatters";

interface MetricDefinition {
	key: string;
	label: string;
	format: (value: number) => string;
	colorCoded: boolean;
}

const METRIC_DEFINITIONS: MetricDefinition[] = [
	{ key: "pnl", label: "PnL", format: formatPnl, colorCoded: true },
	{
		key: "net_worth",
		label: "Net Worth",
		format: formatCurrency,
		colorCoded: false,
	},
	{
		key: "episode_return",
		label: "Episode Return",
		format: formatPercent,
		colorCoded: true,
	},
	{
		key: "trade_count",
		label: "Trades",
		format: (v: number) => formatNumber(v),
		colorCoded: false,
	},
	{
		key: "sharpe_ratio",
		label: "Sharpe Ratio",
		format: (v: number) => formatNumber(v, 2),
		colorCoded: true,
	},
	{
		key: "max_drawdown",
		label: "Max Drawdown",
		format: formatPercent,
		colorCoded: true,
	},
];

interface MetricCardsProps {
	metrics: Record<string, number>;
}

function getValueColor(key: string, value: number, colorCoded: boolean): string {
	if (!colorCoded) return "text-[var(--text-primary)]";
	if (key === "max_drawdown") {
		return value <= 0 ? "text-[var(--accent-red)]" : "text-[var(--accent-green)]";
	}
	if (value > 0) return "text-[var(--accent-green)]";
	if (value < 0) return "text-[var(--accent-red)]";
	return "text-[var(--text-primary)]";
}

export function MetricCards({ metrics }: MetricCardsProps) {
	const visibleMetrics = METRIC_DEFINITIONS.filter((def) => metrics[def.key] !== undefined);

	if (visibleMetrics.length === 0) return null;

	return (
		<div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
			{visibleMetrics.map((def) => {
				const value = metrics[def.key];
				const colorClass = getValueColor(def.key, value, def.colorCoded);

				return (
					<Card key={def.key} className="flex flex-col gap-1">
						<span className="text-xs font-medium text-[var(--text-secondary)]">{def.label}</span>
						<span className={`text-lg font-semibold ${colorClass}`}>{def.format(value)}</span>
					</Card>
				);
			})}
		</div>
	);
}
