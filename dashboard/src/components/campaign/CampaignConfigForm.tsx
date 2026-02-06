"use client";

import { getDatasets, launchCampaign } from "@/lib/api";
import type { DatasetConfig } from "@/lib/types";
import { useCallback, useEffect, useState } from "react";

interface CampaignConfigFormProps {
	onLaunched: (studyName: string) => void;
}

export function CampaignConfigForm({ onLaunched }: CampaignConfigFormProps) {
	const [studyName, setStudyName] = useState("");
	const [datasetId, setDatasetId] = useState("");
	const [nTrials, setNTrials] = useState(50);
	const [iterationsPerTrial, setIterationsPerTrial] = useState(40);
	const [datasets, setDatasets] = useState<DatasetConfig[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		getDatasets()
			.then(setDatasets)
			.catch(() => {});
	}, []);

	const handleLaunch = useCallback(async () => {
		if (!studyName.trim() || !datasetId) {
			setError("Study name and dataset are required");
			return;
		}
		setLoading(true);
		setError(null);
		try {
			const res = await launchCampaign({
				study_name: studyName.trim(),
				dataset_id: datasetId,
				n_trials: nTrials,
				iterations_per_trial: iterationsPerTrial,
			});
			if ("error" in res) {
				setError((res as unknown as { error: string }).error);
			} else {
				onLaunched(res.study_name);
			}
		} catch (err) {
			setError(err instanceof Error ? err.message : "Launch failed");
		} finally {
			setLoading(false);
		}
	}, [studyName, datasetId, nTrials, iterationsPerTrial, onLaunched]);

	return (
		<div className="mx-auto max-w-lg space-y-5">
			<div>
				<label
					htmlFor="campaign-study-name"
					className="mb-1 block text-sm font-medium text-[var(--text-secondary)]"
				>
					Study Name
				</label>
				<input
					id="campaign-study-name"
					type="text"
					value={studyName}
					onChange={(e) => setStudyName(e.target.value)}
					placeholder="btc_alpha_v1"
					className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-primary)] px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:border-[var(--accent-blue)] focus:outline-none"
				/>
			</div>

			<div>
				<label
					htmlFor="campaign-dataset"
					className="mb-1 block text-sm font-medium text-[var(--text-secondary)]"
				>
					Dataset
				</label>
				<select
					id="campaign-dataset"
					value={datasetId}
					onChange={(e) => setDatasetId(e.target.value)}
					className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-primary)] px-3 py-2 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
				>
					<option value="">Select a dataset...</option>
					{datasets.map((ds) => (
						<option key={ds.id} value={ds.id}>
							{ds.name} ({ds.source_type})
						</option>
					))}
				</select>
			</div>

			<div>
				<label
					htmlFor="campaign-n-trials"
					className="mb-1 block text-sm font-medium text-[var(--text-secondary)]"
				>
					Number of Trials: {nTrials}
				</label>
				<input
					id="campaign-n-trials"
					type="range"
					min={10}
					max={200}
					step={5}
					value={nTrials}
					onChange={(e) => setNTrials(Number(e.target.value))}
					className="w-full accent-[var(--accent-blue)]"
				/>
				<div className="flex justify-between text-xs text-[var(--text-secondary)]">
					<span>10</span>
					<span>200</span>
				</div>
			</div>

			<div>
				<label
					htmlFor="campaign-iters"
					className="mb-1 block text-sm font-medium text-[var(--text-secondary)]"
				>
					Iterations per Trial: {iterationsPerTrial}
				</label>
				<input
					id="campaign-iters"
					type="range"
					min={20}
					max={80}
					step={5}
					value={iterationsPerTrial}
					onChange={(e) => setIterationsPerTrial(Number(e.target.value))}
					className="w-full accent-[var(--accent-blue)]"
				/>
				<div className="flex justify-between text-xs text-[var(--text-secondary)]">
					<span>20</span>
					<span>80</span>
				</div>
			</div>

			{error && (
				<div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400">
					{error}
				</div>
			)}

			<button
				type="button"
				onClick={handleLaunch}
				disabled={loading || !studyName.trim() || !datasetId}
				className="w-full rounded-md bg-[var(--accent-blue)] px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[var(--accent-blue)]/80 disabled:opacity-50 disabled:cursor-not-allowed"
			>
				{loading ? "Launching..." : "Start Alpha Search"}
			</button>
		</div>
	);
}
