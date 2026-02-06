"use client";

import { Badge } from "@/components/common/Badge";
import { DataTable } from "@/components/common/DataTable";
import { formatCurrency, formatNumber } from "@/lib/formatters";
import type { TradeRecord } from "@/lib/types";
import { useCallback } from "react";

interface TradeLogTableProps {
	trades: TradeRecord[];
	showExperiment?: boolean;
}

interface TradeRow extends Record<string, unknown> {
	id: number;
	experiment_id: string;
	experiment_name: string;
	script: string;
	episode: number;
	step: number;
	side: "buy" | "sell";
	price: number;
	size: number;
	commission: number;
}

function toRow(trade: TradeRecord): TradeRow {
	return {
		id: trade.id,
		experiment_id: trade.experiment_id,
		experiment_name: trade.experiment_name ?? trade.experiment_id,
		script: trade.script ?? "",
		episode: trade.episode,
		step: trade.step,
		side: trade.side,
		price: trade.price,
		size: trade.size,
		commission: trade.commission,
	};
}

function buildColumns(showExperiment: boolean) {
	const cols = [];

	if (showExperiment) {
		cols.push({
			key: "experiment_name",
			label: "Experiment",
			sortable: true,
			render: (row: TradeRow) => (
				<span className="text-[var(--text-primary)]">{row.experiment_name}</span>
			),
		});
	}

	cols.push(
		{
			key: "episode",
			label: "Episode",
			sortable: true,
			align: "right" as const,
			render: (row: TradeRow) => <span>{formatNumber(row.episode)}</span>,
		},
		{
			key: "step",
			label: "Step",
			sortable: true,
			align: "right" as const,
			render: (row: TradeRow) => <span>{formatNumber(row.step)}</span>,
		},
		{
			key: "side",
			label: "Side",
			sortable: true,
			render: (row: TradeRow) => (
				<Badge label={row.side.toUpperCase()} variant={row.side === "buy" ? "success" : "danger"} />
			),
		},
		{
			key: "price",
			label: "Price",
			sortable: true,
			align: "right" as const,
			render: (row: TradeRow) => <span>{formatCurrency(row.price)}</span>,
		},
		{
			key: "size",
			label: "Size",
			sortable: true,
			align: "right" as const,
			render: (row: TradeRow) => <span>{formatNumber(row.size, 4)}</span>,
		},
		{
			key: "commission",
			label: "Commission",
			sortable: true,
			align: "right" as const,
			render: (row: TradeRow) => (
				<span className="text-[var(--text-secondary)]">{formatCurrency(row.commission)}</span>
			),
		},
	);

	return cols;
}

function exportToCsv(trades: TradeRecord[], showExperiment: boolean) {
	const headers = [
		...(showExperiment ? ["experiment_id", "experiment_name"] : []),
		"episode",
		"step",
		"side",
		"price",
		"size",
		"commission",
	];

	const csvRows = [headers.join(",")];
	for (const trade of trades) {
		const row = [
			...(showExperiment ? [trade.experiment_id, trade.experiment_name ?? ""] : []),
			String(trade.episode),
			String(trade.step),
			trade.side,
			String(trade.price),
			String(trade.size),
			String(trade.commission),
		];
		csvRows.push(row.join(","));
	}

	const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
	const url = URL.createObjectURL(blob);
	const anchor = document.createElement("a");
	anchor.href = url;
	anchor.download = "trades.csv";
	anchor.click();
	URL.revokeObjectURL(url);
}

export function TradeLogTable({ trades, showExperiment = false }: TradeLogTableProps) {
	const rows = trades.map(toRow);
	const columns = buildColumns(showExperiment);

	const handleExport = useCallback(() => {
		exportToCsv(trades, showExperiment);
	}, [trades, showExperiment]);

	return (
		<div className="flex flex-col gap-3">
			<div className="flex items-center justify-between">
				<span className="text-sm text-[var(--text-secondary)]">
					{formatNumber(trades.length)} trade{trades.length !== 1 ? "s" : ""}
				</span>
				<button
					type="button"
					onClick={handleExport}
					disabled={trades.length === 0}
					className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-xs font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--border-color)] disabled:cursor-not-allowed disabled:opacity-50"
				>
					Export CSV
				</button>
			</div>
			<DataTable<TradeRow>
				columns={columns}
				data={rows}
				rowKey={(row) => String(row.id)}
				emptyMessage="No trades recorded"
			/>
		</div>
	);
}
