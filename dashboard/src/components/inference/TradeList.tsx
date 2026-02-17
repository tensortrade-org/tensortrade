import { formatCurrency, formatNumber } from "@/lib/formatters";
import type { TradeEvent } from "@/lib/types";

interface TradeListProps {
	trades: TradeEvent[];
}

interface TradeWithPnl extends TradeEvent {
	pnl: number | null;
}

function formatTimestamp(ts: number | undefined): string {
	if (!ts) return "--";
	const d = new Date(ts * 1000);
	return new Intl.DateTimeFormat("en-US", {
		month: "short",
		day: "numeric",
		hour: "2-digit",
		minute: "2-digit",
		hour12: false,
	}).format(d);
}

/** Pair buys with sells chronologically and compute PnL on the sell side.
 *  BUY size = dollar amount spent, SELL size = asset quantity sold. */
function computePnl(trades: TradeEvent[]): TradeWithPnl[] {
	const result: TradeWithPnl[] = [];
	let lastBuy: TradeEvent | null = null;

	for (const t of trades) {
		if (t.side === "buy") {
			lastBuy = t;
			result.push({ ...t, pnl: null });
		} else {
			// sell proceeds (price × qty) minus buy cost (already in dollars)
			const pnl =
				lastBuy != null
					? t.price * t.size - lastBuy.size - t.commission - lastBuy.commission
					: null;
			result.push({ ...t, pnl });
			lastBuy = null;
		}
	}
	return result;
}

export function TradeList({ trades }: TradeListProps) {
	const withPnl = computePnl(trades);
	const sorted = [...withPnl].reverse();

	if (sorted.length === 0) {
		return (
			<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
				No trades yet
			</div>
		);
	}

	return (
		<div className="h-full overflow-y-auto">
			<table className="w-full text-left text-xs">
				<thead className="sticky top-0 bg-[var(--bg-secondary)] text-[var(--text-secondary)]">
					<tr>
						<th className="px-2 py-1.5 font-medium">Side</th>
						<th className="px-2 py-1.5 font-medium text-right">Price</th>
						<th className="px-2 py-1.5 font-medium text-right">Size</th>
						<th className="px-2 py-1.5 font-medium text-right">P&L</th>
						<th className="px-2 py-1.5 font-medium text-right">Time</th>
						<th className="px-2 py-1.5 font-medium text-right">Step</th>
					</tr>
				</thead>
				<tbody>
					{sorted.map((t, i) => (
						<tr
							key={`${t.step}-${t.side}-${i}`}
							className="border-t border-[var(--border-primary)]"
						>
							<td className="px-2 py-1.5">
								<span
									className="inline-block rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase"
									style={{
										backgroundColor:
											t.side === "buy" ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)",
										color: t.side === "buy" ? "var(--accent-green)" : "var(--accent-red)",
									}}
								>
									{t.side}
								</span>
							</td>
							<td className="px-2 py-1.5 text-right font-mono text-[var(--text-primary)]">
								{formatCurrency(t.price)}
							</td>
							<td className="px-2 py-1.5 text-right font-mono text-[var(--text-primary)]">
								{formatNumber(t.size, 4)}
							</td>
							<td className="px-2 py-1.5 text-right font-mono font-medium">
								{t.pnl != null ? (
									<span
										style={{
											color: t.pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)",
										}}
									>
										{t.pnl >= 0 ? "+" : ""}
										{formatCurrency(t.pnl)}
									</span>
								) : (
									<span className="text-[var(--text-secondary)]">--</span>
								)}
							</td>
							<td className="px-2 py-1.5 text-right text-[var(--text-secondary)]">
								{formatTimestamp(t.timestamp)}
							</td>
							<td className="px-2 py-1.5 text-right font-mono text-[var(--text-secondary)]">
								{formatNumber(t.step)}
							</td>
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
}
