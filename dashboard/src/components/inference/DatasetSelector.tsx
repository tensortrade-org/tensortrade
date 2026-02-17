"use client";

import { getDatasets } from "@/lib/api";
import type { DatasetConfig } from "@/lib/types";
import { useCallback, useEffect, useState } from "react";

interface DatasetSelectorProps {
	value: string;
	onChange: (datasetId: string) => void;
}

export function DatasetSelector({ value, onChange }: DatasetSelectorProps) {
	const [datasets, setDatasets] = useState<DatasetConfig[]>([]);
	const [loading, setLoading] = useState(true);

	const fetchDatasets = useCallback(async () => {
		try {
			const data = await getDatasets();
			setDatasets(data);
		} catch {
			setDatasets([]);
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		fetchDatasets();
	}, [fetchDatasets]);

	return (
		<select
			value={value}
			onChange={(e) => onChange(e.target.value)}
			disabled={loading}
			className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
		>
			<option value="">Use experiment&apos;s dataset</option>
			{datasets.map((ds) => (
				<option key={ds.id} value={ds.id}>
					{ds.name}
				</option>
			))}
		</select>
	);
}
