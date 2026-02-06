"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { useApi } from "@/hooks/useApi";
import { getOptunaStudies } from "@/lib/api";
import { formatNumber } from "@/lib/formatters";
import type { OptunaStudySummary } from "@/lib/types";
import { useRouter } from "next/navigation";
import { useCallback } from "react";

function StudiesTable({
	studies,
	onSelect,
}: {
	studies: OptunaStudySummary[];
	onSelect: (name: string) => void;
}) {
	return (
		<div className="overflow-x-auto">
			<table className="w-full text-sm">
				<thead>
					<tr className="border-b border-[var(--border-color)] text-left text-xs text-[var(--text-secondary)]">
						<th className="pb-2 pr-4 font-medium">Study Name</th>
						<th className="pb-2 pr-4 font-medium text-right">Total Trials</th>
						<th className="pb-2 pr-4 font-medium text-right">Completed</th>
						<th className="pb-2 pr-4 font-medium text-right">Pruned</th>
						<th className="pb-2 pr-4 font-medium text-right">Pruning Rate</th>
						<th className="pb-2 font-medium text-right">Best P&L</th>
					</tr>
				</thead>
				<tbody>
					{studies.map((study) => {
						const pruningRate =
							study.total_trials > 0
								? ((study.pruned_trials / study.total_trials) * 100).toFixed(0)
								: "0";
						return (
							<tr
								key={study.study_name}
								onClick={() => onSelect(study.study_name)}
								onKeyDown={(e) => {
									if (e.key === "Enter") onSelect(study.study_name);
								}}
								tabIndex={0}
								className="cursor-pointer border-b border-[var(--border-color)] last:border-0 hover:bg-[var(--bg-secondary)]"
							>
								<td className="py-3 pr-4">
									<span className="font-medium text-[var(--accent-blue)]">{study.study_name}</span>
								</td>
								<td className="py-3 pr-4 text-right font-mono text-[var(--text-primary)]">
									{formatNumber(study.total_trials)}
								</td>
								<td className="py-3 pr-4 text-right font-mono text-[var(--accent-green)]">
									{formatNumber(study.completed_trials)}
								</td>
								<td className="py-3 pr-4 text-right font-mono text-[var(--accent-amber)]">
									{formatNumber(study.pruned_trials)}
								</td>
								<td className="py-3 pr-4 text-right font-mono text-[var(--text-secondary)]">
									{pruningRate}%
								</td>
								<td className="py-3 text-right font-mono text-[var(--text-primary)]">
									{study.best_value !== null ? `$${study.best_value.toFixed(0)}` : "--"}
								</td>
							</tr>
						);
					})}
				</tbody>
			</table>
		</div>
	);
}

export default function OptunaStudiesPage() {
	const router = useRouter();

	const studiesFetcher = useCallback(() => getOptunaStudies(), []);
	const { data: studies, loading, error } = useApi<OptunaStudySummary[]>(studiesFetcher, []);

	const handleSelect = useCallback(
		(name: string) => {
			router.push(`/optuna/${encodeURIComponent(name)}`);
		},
		[router],
	);

	return (
		<div className="space-y-6">
			<h1 className="text-xl font-semibold text-[var(--text-primary)]">Optuna Studies</h1>

			{loading ? (
				<LoadingState message="Loading studies..." />
			) : error ? (
				<Card>
					<div className="py-6 text-center text-sm text-[var(--accent-red)]">
						Failed to load studies: {error.message}
					</div>
				</Card>
			) : studies && studies.length > 0 ? (
				<Card>
					<StudiesTable studies={studies} onSelect={handleSelect} />
				</Card>
			) : (
				<Card>
					<div className="py-6 text-center text-sm text-[var(--text-secondary)]">
						No Optuna studies found.
					</div>
				</Card>
			)}
		</div>
	);
}
