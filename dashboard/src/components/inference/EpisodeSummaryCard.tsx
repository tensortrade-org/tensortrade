"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { formatCurrency, formatNumber, formatPercent } from "@/lib/formatters";
import type { EpisodeSummary } from "@/lib/types";

interface EpisodeSummaryCardProps {
	summary: EpisodeSummary;
}

export function EpisodeSummaryCard({ summary }: EpisodeSummaryCardProps) {
	const pnlColor = summary.pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)";

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
			</div>
		</Card>
	);
}
