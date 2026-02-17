"use client";

import { useMemo, useState } from "react";

interface Column<T> {
	key: string;
	label: string;
	render?: (row: T) => React.ReactNode;
	sortable?: boolean;
	align?: "left" | "right" | "center";
	width?: string;
}

interface DataTableProps<T> {
	columns: Column<T>[];
	data: T[];
	rowKey: (row: T) => string;
	onRowClick?: (row: T) => void;
	emptyMessage?: string;
	selectable?: boolean;
	selectedKeys?: string[];
	onSelect?: (key: string) => void;
}

type SortDir = "asc" | "desc";

export function DataTable<T extends Record<string, unknown>>({
	columns,
	data,
	rowKey,
	onRowClick,
	emptyMessage = "No data",
	selectable = false,
	selectedKeys = [],
	onSelect,
}: DataTableProps<T>) {
	const [sortKey, setSortKey] = useState<string | null>(null);
	const [sortDir, setSortDir] = useState<SortDir>("desc");

	const handleSort = (key: string) => {
		if (sortKey === key) {
			setSortDir(sortDir === "asc" ? "desc" : "asc");
		} else {
			setSortKey(key);
			setSortDir("desc");
		}
	};

	const sorted = useMemo(() => {
		if (!sortKey) return data;
		return [...data].sort((a, b) => {
			const va = a[sortKey];
			const vb = b[sortKey];
			if (va == null && vb == null) return 0;
			if (va == null) return 1;
			if (vb == null) return -1;
			const cmp = va < vb ? -1 : va > vb ? 1 : 0;
			return sortDir === "asc" ? cmp : -cmp;
		});
	}, [data, sortKey, sortDir]);

	const alignClass = (align?: string) => {
		if (align === "right") return "text-right";
		if (align === "center") return "text-center";
		return "text-left";
	};

	return (
		<div className="overflow-x-auto">
			<table className="w-full text-sm">
				<thead>
					<tr className="border-b border-[var(--border-color)]">
						{selectable && <th className="w-8 px-3 py-2" />}
						{columns.map((col) => (
							<th
								key={col.key}
								className={`px-3 py-2 font-medium text-[var(--text-secondary)] ${alignClass(col.align)} ${
									col.sortable ? "cursor-pointer select-none hover:text-[var(--text-primary)]" : ""
								}`}
								style={col.width ? { width: col.width } : undefined}
								onClick={col.sortable ? () => handleSort(col.key) : undefined}
								onKeyDown={
									col.sortable
										? (e) => {
												if (e.key === "Enter" || e.key === " ") handleSort(col.key);
											}
										: undefined
								}
								tabIndex={col.sortable ? 0 : undefined}
							>
								{col.label}
								{sortKey === col.key && (
									<span className="ml-1">{sortDir === "asc" ? "\u25B2" : "\u25BC"}</span>
								)}
							</th>
						))}
					</tr>
				</thead>
				<tbody>
					{sorted.length === 0 ? (
						<tr>
							<td
								colSpan={columns.length + (selectable ? 1 : 0)}
								className="px-3 py-8 text-center text-[var(--text-secondary)]"
							>
								{emptyMessage}
							</td>
						</tr>
					) : (
						sorted.map((row) => {
							const key = rowKey(row);
							const isSelected = selectedKeys.includes(key);
							return (
								<tr
									key={key}
									className={`border-b border-[var(--border-color)]/50 transition-colors ${
										onRowClick ? "cursor-pointer hover:bg-[var(--bg-secondary)]" : ""
									} ${isSelected ? "bg-[var(--accent-blue)]/5" : ""}`}
									onClick={() => onRowClick?.(row)}
									onKeyDown={
										onRowClick
											? (e) => {
													if (e.key === "Enter") onRowClick(row);
												}
											: undefined
									}
									tabIndex={onRowClick ? 0 : undefined}
								>
									{selectable && (
										<td className="px-3 py-2">
											<input
												type="checkbox"
												checked={isSelected}
												onChange={(e) => {
													e.stopPropagation();
													onSelect?.(key);
												}}
												className="rounded border-[var(--border-color)]"
											/>
										</td>
									)}
									{columns.map((col) => (
										<td key={col.key} className={`px-3 py-2 ${alignClass(col.align)}`}>
											{col.render ? col.render(row) : String(row[col.key] ?? "")}
										</td>
									))}
								</tr>
							);
						})
					)}
				</tbody>
			</table>
		</div>
	);
}
