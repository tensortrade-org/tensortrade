"use client";

import { Card, CardHeader } from "@/components/common/Card";
import type { TrainingConfig } from "@/lib/types";
import { useHyperparamStore } from "@/stores/hyperparamStore";
import { useMemo } from "react";

interface ConfigRow {
	key: string;
	valueA: string;
	valueB: string;
	differs: boolean;
}

function flattenConfig(config: TrainingConfig, prefix = ""): Record<string, string> {
	const result: Record<string, string> = {};
	for (const [key, value] of Object.entries(config)) {
		const fullKey = prefix ? `${prefix}.${key}` : key;
		if (value !== null && typeof value === "object" && !Array.isArray(value)) {
			const nested = flattenConfig(value as TrainingConfig, fullKey);
			Object.assign(result, nested);
		} else if (Array.isArray(value)) {
			result[fullKey] = `[${value.join(", ")}]`;
		} else {
			result[fullKey] = String(value ?? "null");
		}
	}
	return result;
}

export function PackCompare() {
	const { packs, comparePackIds, setComparePackIds } = useHyperparamStore();

	const packA = useMemo(
		() => packs.find((p) => p.id === comparePackIds?.[0]) ?? null,
		[packs, comparePackIds],
	);
	const packB = useMemo(
		() => packs.find((p) => p.id === comparePackIds?.[1]) ?? null,
		[packs, comparePackIds],
	);

	const rows = useMemo((): ConfigRow[] => {
		if (!packA || !packB) return [];
		const flatA = flattenConfig(packA.config);
		const flatB = flattenConfig(packB.config);
		const allKeys = Array.from(new Set([...Object.keys(flatA), ...Object.keys(flatB)])).sort();
		return allKeys.map((key) => ({
			key,
			valueA: flatA[key] ?? "--",
			valueB: flatB[key] ?? "--",
			differs: flatA[key] !== flatB[key],
		}));
	}, [packA, packB]);

	const diffCount = useMemo(() => rows.filter((r) => r.differs).length, [rows]);

	return (
		<div className="space-y-4">
			{/* Pack Selectors */}
			<div className="flex items-end gap-4">
				<div className="flex-1">
					<span className="mb-1 block text-xs text-[var(--text-secondary)]">Pack A</span>
					<select
						aria-label="Pack A"
						value={comparePackIds?.[0] ?? ""}
						onChange={(e) => setComparePackIds([e.target.value, comparePackIds?.[1] ?? ""])}
						className="w-full rounded border border-[var(--border-color)] bg-[var(--bg-card)] px-3 py-1.5 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
					>
						<option value="">Select pack...</option>
						{packs.map((p) => (
							<option key={p.id} value={p.id}>
								{p.name}
							</option>
						))}
					</select>
				</div>
				<div className="pb-1.5 text-sm text-[var(--text-secondary)]">vs</div>
				<div className="flex-1">
					<span className="mb-1 block text-xs text-[var(--text-secondary)]">Pack B</span>
					<select
						aria-label="Pack B"
						value={comparePackIds?.[1] ?? ""}
						onChange={(e) => setComparePackIds([comparePackIds?.[0] ?? "", e.target.value])}
						className="w-full rounded border border-[var(--border-color)] bg-[var(--bg-card)] px-3 py-1.5 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
					>
						<option value="">Select pack...</option>
						{packs.map((p) => (
							<option key={p.id} value={p.id}>
								{p.name}
							</option>
						))}
					</select>
				</div>
			</div>

			{/* Diff Table */}
			{packA && packB ? (
				<Card>
					<CardHeader
						title="Configuration Diff"
						action={
							<span className="text-xs text-[var(--text-secondary)]">
								{diffCount} {diffCount === 1 ? "difference" : "differences"}
							</span>
						}
					/>
					<div className="overflow-x-auto">
						<table className="w-full text-sm">
							<thead>
								<tr className="border-b border-[var(--border-color)] text-xs text-[var(--text-secondary)]">
									<th className="pb-2 pr-4 text-left font-medium">Parameter</th>
									<th className="pb-2 pr-4 text-left font-medium">{packA.name}</th>
									<th className="pb-2 text-left font-medium">{packB.name}</th>
								</tr>
							</thead>
							<tbody>
								{rows.map((row) => (
									<tr
										key={row.key}
										className={`border-b border-[var(--border-color)]/50 ${
											row.differs ? "bg-[var(--accent-red)]/5" : ""
										}`}
									>
										<td className="py-1.5 pr-4 font-mono text-xs text-[var(--text-secondary)]">
											{row.key}
										</td>
										<td
											className={`py-1.5 pr-4 font-mono text-xs ${
												row.differs ? "text-[var(--accent-red)]" : "text-[var(--text-primary)]"
											}`}
										>
											{row.valueA}
										</td>
										<td
											className={`py-1.5 font-mono text-xs ${
												row.differs ? "text-[var(--accent-green)]" : "text-[var(--text-primary)]"
											}`}
										>
											{row.valueB}
										</td>
									</tr>
								))}
							</tbody>
						</table>
					</div>
				</Card>
			) : (
				<div className="flex h-60 items-center justify-center text-sm text-[var(--text-secondary)]">
					Select two packs to compare
				</div>
			)}
		</div>
	);
}
