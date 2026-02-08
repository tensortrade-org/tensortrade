"use client";

import { getDatasets, launchCampaign } from "@/lib/api";
import {
	ACTION_GROUPS,
	ACTION_SCHEMES,
	type ActionSchemeInfo,
	REWARD_SCHEMES,
	type RewardSchemeInfo,
	getCompatibleRewardSchemes,
} from "@/lib/scheme-compat";
import type { DatasetConfig } from "@/lib/types";
import { useCallback, useEffect, useMemo, useState } from "react";

interface CampaignConfigFormProps {
	onLaunched: (studyName: string) => void;
}

function SchemeCheckbox({
	label,
	checked,
	disabled,
	onChange,
}: {
	label: string;
	checked: boolean;
	disabled?: boolean;
	onChange: (checked: boolean) => void;
}) {
	return (
		<label
			className={`flex items-center gap-2 rounded px-2 py-1 text-xs transition-colors ${
				disabled
					? "cursor-not-allowed text-[var(--text-secondary)] opacity-40"
					: "cursor-pointer text-[var(--text-primary)] hover:bg-[var(--bg-secondary)]"
			}`}
		>
			<input
				type="checkbox"
				checked={checked}
				disabled={disabled}
				onChange={(e) => onChange(e.target.checked)}
				className="accent-[var(--accent-blue)]"
			/>
			{label}
		</label>
	);
}

export function CampaignConfigForm({ onLaunched }: CampaignConfigFormProps) {
	const [studyName, setStudyName] = useState("");
	const [datasetId, setDatasetId] = useState("");
	const [nTrials, setNTrials] = useState(50);
	const [iterationsPerTrial, setIterationsPerTrial] = useState(40);
	const [datasets, setDatasets] = useState<DatasetConfig[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	// Scheme selection — default all selected
	const [selectedActions, setSelectedActions] = useState<Set<string>>(
		() => new Set(ACTION_SCHEMES.map((a) => a.value)),
	);
	const [selectedRewards, setSelectedRewards] = useState<Set<string>>(
		() => new Set(REWARD_SCHEMES.map((r) => r.value)),
	);

	useEffect(() => {
		getDatasets()
			.then(setDatasets)
			.catch(() => {});
	}, []);

	// Compute which rewards are compatible with ANY selected action
	const compatibleRewardValues = useMemo(() => {
		const result = new Set<string>();
		for (const actionVal of selectedActions) {
			for (const r of getCompatibleRewardSchemes(actionVal)) {
				result.add(r.value);
			}
		}
		return result;
	}, [selectedActions]);

	// Auto-deselect rewards that became incompatible
	useEffect(() => {
		setSelectedRewards((prev) => {
			const next = new Set<string>();
			for (const v of prev) {
				if (compatibleRewardValues.has(v)) next.add(v);
			}
			return next.size === prev.size ? prev : next;
		});
	}, [compatibleRewardValues]);

	const toggleAction = useCallback((value: string, checked: boolean) => {
		setSelectedActions((prev) => {
			const next = new Set(prev);
			if (checked) next.add(value);
			else next.delete(value);
			return next;
		});
	}, []);

	const toggleReward = useCallback((value: string, checked: boolean) => {
		setSelectedRewards((prev) => {
			const next = new Set(prev);
			if (checked) next.add(value);
			else next.delete(value);
			return next;
		});
	}, []);

	const selectAllActions = useCallback(() => {
		setSelectedActions(new Set(ACTION_SCHEMES.map((a) => a.value)));
	}, []);

	const clearActions = useCallback(() => {
		setSelectedActions(new Set<string>());
	}, []);

	const selectAllRewards = useCallback(() => {
		setSelectedRewards(
			new Set(
				REWARD_SCHEMES.filter((r) => compatibleRewardValues.has(r.value)).map((r) => r.value),
			),
		);
	}, [compatibleRewardValues]);

	const clearRewards = useCallback(() => {
		setSelectedRewards(new Set<string>());
	}, []);

	const comboCount = useMemo(() => {
		let count = 0;
		for (const actionVal of selectedActions) {
			const compat = getCompatibleRewardSchemes(actionVal);
			for (const r of compat) {
				if (selectedRewards.has(r.value)) count++;
			}
		}
		return count;
	}, [selectedActions, selectedRewards]);

	const handleLaunch = useCallback(async () => {
		if (!studyName.trim() || !datasetId) {
			setError("Study name and dataset are required");
			return;
		}
		if (selectedActions.size === 0 || selectedRewards.size === 0) {
			setError("Select at least one action and one reward scheme");
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
				action_schemes: [...selectedActions],
				reward_schemes: [...selectedRewards],
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
	}, [
		studyName,
		datasetId,
		nTrials,
		iterationsPerTrial,
		selectedActions,
		selectedRewards,
		onLaunched,
	]);

	return (
		<div className="mx-auto max-w-2xl space-y-5">
			{/* Row 1: Study Name + Dataset */}
			<div className="grid grid-cols-2 gap-4">
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
			</div>

			{/* Row 2: Scheme Selection */}
			<div className="grid grid-cols-2 gap-4">
				{/* Action Schemes */}
				<div>
					<div className="mb-1 flex items-center justify-between">
						<span className="text-sm font-medium text-[var(--text-secondary)]">
							Action Schemes ({selectedActions.size})
						</span>
						<span className="flex gap-2 text-xs">
							<button
								type="button"
								onClick={selectAllActions}
								className="text-[var(--accent-blue)] hover:underline"
							>
								All
							</button>
							<button
								type="button"
								onClick={clearActions}
								className="text-[var(--accent-blue)] hover:underline"
							>
								None
							</button>
						</span>
					</div>
					<div className="max-h-56 overflow-y-auto rounded-md border border-[var(--border-color)] bg-[var(--bg-primary)] p-2">
						{ACTION_GROUPS.map((group) => (
							<div key={group} className="mb-1.5">
								<div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
									{group}
								</div>
								{ACTION_SCHEMES.filter((a: ActionSchemeInfo) => a.group === group).map(
									(action: ActionSchemeInfo) => (
										<SchemeCheckbox
											key={action.value}
											label={action.label}
											checked={selectedActions.has(action.value)}
											onChange={(c) => toggleAction(action.value, c)}
										/>
									),
								)}
							</div>
						))}
					</div>
				</div>

				{/* Reward Schemes */}
				<div>
					<div className="mb-1 flex items-center justify-between">
						<span className="text-sm font-medium text-[var(--text-secondary)]">
							Reward Schemes ({selectedRewards.size})
						</span>
						<span className="flex gap-2 text-xs">
							<button
								type="button"
								onClick={selectAllRewards}
								className="text-[var(--accent-blue)] hover:underline"
							>
								All
							</button>
							<button
								type="button"
								onClick={clearRewards}
								className="text-[var(--accent-blue)] hover:underline"
							>
								None
							</button>
						</span>
					</div>
					<div className="max-h-56 overflow-y-auto rounded-md border border-[var(--border-color)] bg-[var(--bg-primary)] p-2">
						{REWARD_SCHEMES.map((reward: RewardSchemeInfo) => {
							const isCompat = compatibleRewardValues.has(reward.value);
							return (
								<SchemeCheckbox
									key={reward.value}
									label={reward.label + (reward.requiresBshSemantics ? " (BSH only)" : "")}
									checked={selectedRewards.has(reward.value)}
									disabled={!isCompat}
									onChange={(c) => toggleReward(reward.value, c)}
								/>
							);
						})}
					</div>
					{selectedActions.size > 0 && (
						<div className="mt-1 text-xs text-[var(--text-secondary)]">
							{comboCount} valid action+reward combo
							{comboCount !== 1 ? "s" : ""}
						</div>
					)}
				</div>
			</div>

			{/* Row 3: Sliders */}
			<div className="grid grid-cols-2 gap-4">
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
						max={200}
						step={5}
						value={iterationsPerTrial}
						onChange={(e) => setIterationsPerTrial(Number(e.target.value))}
						className="w-full accent-[var(--accent-blue)]"
					/>
					<div className="flex justify-between text-xs text-[var(--text-secondary)]">
						<span>20</span>
						<span>200</span>
					</div>
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
				disabled={
					loading ||
					!studyName.trim() ||
					!datasetId ||
					selectedActions.size === 0 ||
					selectedRewards.size === 0
				}
				className="w-full rounded-md bg-[var(--accent-blue)] px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[var(--accent-blue)]/80 disabled:opacity-50 disabled:cursor-not-allowed"
			>
				{loading ? "Launching..." : "Start Alpha Search"}
			</button>
		</div>
	);
}
