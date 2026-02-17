"use client";

import { getExperiments } from "@/lib/api";
import type { ExperimentSummary } from "@/lib/types";
import { useCallback, useEffect, useState } from "react";

interface ExperimentSelectorProps {
	value: string | null;
	onChange: (experimentId: string) => void;
}

export function ExperimentSelector({ value, onChange }: ExperimentSelectorProps) {
	const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
	const [loading, setLoading] = useState(true);

	const fetchExperiments = useCallback(async () => {
		try {
			const data = await getExperiments({ status: "completed", limit: 50 });
			setExperiments(data);
		} catch {
			setExperiments([]);
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		fetchExperiments();
	}, [fetchExperiments]);

	return (
		<select
			value={value ?? ""}
			onChange={(e) => onChange(e.target.value)}
			disabled={loading}
			className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
		>
			<option value="" disabled>
				{loading ? "Loading..." : "Select experiment"}
			</option>
			{experiments.map((exp) => (
				<option key={exp.id} value={exp.id}>
					{exp.name} ({exp.script})
				</option>
			))}
		</select>
	);
}
