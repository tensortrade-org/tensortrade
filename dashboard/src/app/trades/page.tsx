"use client";

import { Card } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { TradeLogTable } from "@/components/trading/TradeLogTable";
import { useApi } from "@/hooks/useApi";
import { getAllTrades } from "@/lib/api";
import type { TradeRecord } from "@/lib/types";
import { useCallback, useState } from "react";

interface TradeFilterState {
	experimentId: string;
	side: string;
}

export default function TradesPage() {
	const [filters, setFilters] = useState<TradeFilterState>({
		experimentId: "",
		side: "",
	});

	const tradesFetcher = useCallback(
		() =>
			getAllTrades({
				experiment_id: filters.experimentId || undefined,
				side: filters.side || undefined,
			}),
		[filters.experimentId, filters.side],
	);

	const {
		data: trades,
		loading,
		error,
	} = useApi<TradeRecord[]>(tradesFetcher, [filters.experimentId, filters.side]);

	return (
		<div className="space-y-6">
			<h1 className="text-xl font-semibold text-[var(--text-primary)]">Trade Log</h1>

			{/* Filters */}
			<Card>
				<div className="flex flex-wrap items-center gap-4">
					<div className="flex items-center gap-2">
						<label htmlFor="experiment-filter" className="text-sm text-[var(--text-secondary)]">
							Experiment ID
						</label>
						<input
							id="experiment-filter"
							type="text"
							value={filters.experimentId}
							onChange={(e) => setFilters((f) => ({ ...f, experimentId: e.target.value }))}
							placeholder="Filter by experiment..."
							className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)]"
						/>
					</div>

					<div className="flex items-center gap-2">
						<label htmlFor="side-filter" className="text-sm text-[var(--text-secondary)]">
							Side
						</label>
						<select
							id="side-filter"
							value={filters.side}
							onChange={(e) => setFilters((f) => ({ ...f, side: e.target.value }))}
							className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)]"
						>
							<option value="">All Sides</option>
							<option value="buy">Buy</option>
							<option value="sell">Sell</option>
						</select>
					</div>

					{(filters.experimentId || filters.side) && (
						<button
							type="button"
							onClick={() => setFilters({ experimentId: "", side: "" })}
							className="ml-auto text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
						>
							Clear Filters
						</button>
					)}
				</div>
			</Card>

			{/* Trade Table */}
			{loading ? (
				<LoadingState message="Loading trades..." />
			) : error ? (
				<Card>
					<div className="py-6 text-center text-sm text-[var(--accent-red)]">
						Failed to load trades: {error.message}
					</div>
				</Card>
			) : trades && trades.length > 0 ? (
				<TradeLogTable trades={trades} showExperiment />
			) : (
				<Card>
					<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
						No trades found matching filters.
					</div>
				</Card>
			)}
		</div>
	);
}
