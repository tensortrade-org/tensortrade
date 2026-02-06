"use client";

import { StatusBadge } from "@/components/common/Badge";
import { DataTable } from "@/components/common/DataTable";
import { formatCurrency, formatDate, formatNumber, formatPnl } from "@/lib/formatters";
import type { ExperimentSummary } from "@/lib/types";
import { useRouter } from "next/navigation";

interface ExperimentTableProps {
	experiments: ExperimentSummary[];
	onSelect?: (id: string) => void;
	selectable?: boolean;
	selectedIds?: string[];
}

interface ExperimentRow extends Record<string, unknown> {
	id: string;
	name: string;
	script: string;
	status: ExperimentSummary["status"];
	pnl: number | null;
	net_worth: number | null;
	trade_count: number | null;
	started_at: string;
	_original: ExperimentSummary;
}

function toRow(experiment: ExperimentSummary): ExperimentRow {
	return {
		id: experiment.id,
		name: experiment.name,
		script: experiment.script,
		status: experiment.status,
		pnl: experiment.final_metrics.pnl ?? null,
		net_worth: experiment.final_metrics.net_worth ?? null,
		trade_count: experiment.final_metrics.trade_count ?? null,
		started_at: experiment.started_at,
		_original: experiment,
	};
}

const columns = [
	{
		key: "name",
		label: "Name",
		sortable: true,
		render: (row: ExperimentRow) => (
			<span className="font-medium text-[var(--text-primary)]">{row.name}</span>
		),
	},
	{
		key: "script",
		label: "Script",
		sortable: true,
		render: (row: ExperimentRow) => (
			<span className="text-[var(--text-secondary)]">{row.script}</span>
		),
	},
	{
		key: "status",
		label: "Status",
		sortable: true,
		render: (row: ExperimentRow) => <StatusBadge status={row.status} />,
	},
	{
		key: "pnl",
		label: "PnL",
		sortable: true,
		align: "right" as const,
		render: (row: ExperimentRow) => {
			if (row.pnl === null) return <span className="text-[var(--text-secondary)]">--</span>;
			const color = row.pnl >= 0 ? "text-[var(--accent-green)]" : "text-[var(--accent-red)]";
			return <span className={color}>{formatPnl(row.pnl)}</span>;
		},
	},
	{
		key: "net_worth",
		label: "Net Worth",
		sortable: true,
		align: "right" as const,
		render: (row: ExperimentRow) =>
			row.net_worth !== null ? (
				<span>{formatCurrency(row.net_worth)}</span>
			) : (
				<span className="text-[var(--text-secondary)]">--</span>
			),
	},
	{
		key: "trade_count",
		label: "Trades",
		sortable: true,
		align: "right" as const,
		render: (row: ExperimentRow) =>
			row.trade_count !== null ? (
				<span>{formatNumber(row.trade_count)}</span>
			) : (
				<span className="text-[var(--text-secondary)]">--</span>
			),
	},
	{
		key: "started_at",
		label: "Started",
		sortable: true,
		render: (row: ExperimentRow) => (
			<span className="text-[var(--text-secondary)]">{formatDate(row.started_at)}</span>
		),
	},
];

export function ExperimentTable({
	experiments,
	onSelect,
	selectable = false,
	selectedIds = [],
}: ExperimentTableProps) {
	const router = useRouter();
	const rows = experiments.map(toRow);

	const handleRowClick = (row: ExperimentRow) => {
		router.push(`/experiments/${row.id}`);
	};

	return (
		<DataTable<ExperimentRow>
			columns={columns}
			data={rows}
			rowKey={(row) => row.id}
			onRowClick={handleRowClick}
			selectable={selectable}
			selectedKeys={selectedIds}
			onSelect={onSelect}
			emptyMessage="No experiments found"
		/>
	);
}
