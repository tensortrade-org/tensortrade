"use client";

import { Badge } from "@/components/common/Badge";
import { DataTable } from "@/components/common/DataTable";
import { formatDate, formatNumber } from "@/lib/formatters";
import type { LeaderboardEntry } from "@/lib/types";
import { useRouter } from "next/navigation";

interface LeaderboardTableProps {
	entries: LeaderboardEntry[];
}

interface LeaderboardRow extends Record<string, unknown> {
	experiment_id: string;
	rank: number;
	name: string;
	script: string;
	metric_value: number;
	metric_name: string;
	started_at: string;
	tags: string[];
}

function toRow(entry: LeaderboardEntry): LeaderboardRow {
	return {
		experiment_id: entry.experiment_id,
		rank: entry.rank,
		name: entry.name,
		script: entry.script,
		metric_value: entry.metric_value,
		metric_name: entry.metric_name,
		started_at: entry.started_at,
		tags: entry.tags,
	};
}

function RankCell({ rank }: { rank: number }) {
	const baseClasses =
		"inline-flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold";

	if (rank === 1) {
		return <span className={`${baseClasses} bg-amber-500/20 text-amber-400`}>{rank}</span>;
	}
	if (rank === 2) {
		return <span className={`${baseClasses} bg-slate-300/20 text-slate-300`}>{rank}</span>;
	}
	if (rank === 3) {
		return <span className={`${baseClasses} bg-orange-600/20 text-orange-400`}>{rank}</span>;
	}
	return <span className="text-xs text-[var(--text-secondary)]">{rank}</span>;
}

const columns = [
	{
		key: "rank",
		label: "#",
		sortable: true,
		width: "60px",
		render: (row: LeaderboardRow) => <RankCell rank={row.rank} />,
	},
	{
		key: "name",
		label: "Name",
		sortable: true,
		render: (row: LeaderboardRow) => (
			<div className="flex flex-col gap-0.5">
				<span className="font-medium text-[var(--text-primary)]">{row.name}</span>
				{row.tags.length > 0 && (
					<div className="flex flex-wrap gap-1">
						{row.tags.map((tag) => (
							<Badge key={tag} label={tag} variant="default" />
						))}
					</div>
				)}
			</div>
		),
	},
	{
		key: "script",
		label: "Script",
		sortable: true,
		render: (row: LeaderboardRow) => (
			<span className="text-[var(--text-secondary)]">{row.script}</span>
		),
	},
	{
		key: "metric_value",
		label: "Score",
		sortable: true,
		align: "right" as const,
		render: (row: LeaderboardRow) => (
			<span className="font-semibold text-[var(--text-primary)]">
				{formatNumber(row.metric_value, 4)}
			</span>
		),
	},
	{
		key: "started_at",
		label: "Started",
		sortable: true,
		render: (row: LeaderboardRow) => (
			<span className="text-[var(--text-secondary)]">{formatDate(row.started_at)}</span>
		),
	},
];

export function LeaderboardTable({ entries }: LeaderboardTableProps) {
	const router = useRouter();
	const rows = entries.map(toRow);

	const handleRowClick = (row: LeaderboardRow) => {
		router.push(`/experiments/${row.experiment_id}`);
	};

	return (
		<DataTable<LeaderboardRow>
			columns={columns}
			data={rows}
			rowKey={(row) => row.experiment_id}
			onRowClick={handleRowClick}
			emptyMessage="No leaderboard entries"
		/>
	);
}
