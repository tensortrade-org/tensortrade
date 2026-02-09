"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { formatCurrency, formatNumber, formatPercent } from "@/lib/formatters";
import type { EpisodeSummary } from "@/lib/types";

interface EpisodeSummaryCardProps {
	summary: EpisodeSummary;
}

export function EpisodeSummaryCard({ summary }: EpisodeSummaryCardProps) {
	const pnlColor = summary.pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)";
	const totalActions = summary.buy_count + summary.sell_count + summary.hold_count;
	const holdRatio =
		summary.hold_ratio !== undefined
			? summary.hold_ratio
			: totalActions > 0
				? summary.hold_count / totalActions
				: 0;
	const tradeRatio =
		summary.trade_ratio !== undefined
			? summary.trade_ratio
			: totalActions > 0
				? summary.total_trades / totalActions
				: 0;
	const pnlPerTrade =
		summary.pnl_per_trade !== undefined
			? summary.pnl_per_trade
			: summary.total_trades > 0
				? summary.pnl / summary.total_trades
				: 0;

	return (
		<Card>
			<CardHeader title="Episode Summary" />
			<div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
				<div>
					<p className="text-xs text-[var(--text-secondary)]">Final P&L</p>
					<p className="font-mono text-lg font-semibold" style={{ color: pnlColor }}>
						{summary.pnl >= 0 ? "+" : ""}
						{formatCurrency(summary.pnl)}
					</p>
					<p className="font-mono text-xs" style={{ color: pnlColor }}>
						{formatPercent(summary.pnl_pct)}
					</p>
				</div>

				<div>
					<p className="text-xs text-[var(--text-secondary)]">Net Worth</p>
					<p className="font-mono text-sm font-medium text-[var(--text-primary)]">
						{formatCurrency(summary.final_net_worth)}
					</p>
					<p className="font-mono text-xs text-[var(--text-secondary)]">
						from {formatCurrency(summary.initial_net_worth)}
					</p>
				</div>

				<div>
					<p className="text-xs text-[var(--text-secondary)]">Total Trades</p>
					<p className="font-mono text-sm font-medium text-[var(--text-primary)]">
						{formatNumber(summary.total_trades)}
					</p>
					<p className="font-mono text-xs text-[var(--text-secondary)]">
						{formatNumber(summary.total_steps)} steps
					</p>
				</div>

				<div>
					<p className="text-xs text-[var(--text-secondary)]">Actions</p>
					<div className="flex gap-2 text-xs font-mono">
						<span className="text-[#22c55e]">B:{formatNumber(summary.buy_count)}</span>
						<span className="text-[#ef4444]">S:{formatNumber(summary.sell_count)}</span>
						<span className="text-[#3b82f6]">H:{formatNumber(summary.hold_count)}</span>
					</div>
				</div>

				<div>
					<p className="text-xs text-[var(--text-secondary)]">Trade Ratio</p>
					<p className="font-mono text-sm font-medium text-[var(--text-primary)]">
						{formatPercent(tradeRatio * 100)}
					</p>
				</div>

				<div>
					<p className="text-xs text-[var(--text-secondary)]">Hold Ratio</p>
					<p className="font-mono text-sm font-medium text-[var(--text-primary)]">
						{formatPercent(holdRatio * 100)}
					</p>
				</div>

				<div>
					<p className="text-xs text-[var(--text-secondary)]">PnL / Trade</p>
					<p
						className="font-mono text-sm font-medium"
						style={{ color: pnlPerTrade >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}
					>
						{pnlPerTrade >= 0 ? "+" : ""}
						{formatCurrency(pnlPerTrade)}
					</p>
				</div>

				<div>
					<p className="text-xs text-[var(--text-secondary)]">Max Drawdown</p>
					<p className="font-mono text-sm font-medium text-[var(--accent-red)]">
						{summary.max_drawdown_pct !== undefined
							? `${summary.max_drawdown_pct.toFixed(2)}%`
							: "--"}
					</p>
				</div>
			</div>
		</Card>
	);
}
