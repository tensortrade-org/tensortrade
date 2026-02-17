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
import type { CampaignParamSpec, DatasetConfig } from "@/lib/types";
import { useCallback, useEffect, useMemo, useState } from "react";

interface CampaignConfigFormProps {
	onLaunched: (studyName: string) => void;
}

type RewardParamKey =
	| "trade_penalty_multiplier"
	| "churn_penalty_multiplier"
	| "churn_window"
	| "reward_clip";

interface RewardParamSetting {
	label: string;
	type: "float" | "int";
	tune: boolean;
	value: number;
	low: number;
	high: number;
	log: boolean;
	step: number;
}

const DEFAULT_REWARD_PARAM_SETTINGS: Record<RewardParamKey, RewardParamSetting> = {
	trade_penalty_multiplier: {
		label: "Trade Penalty",
		type: "float",
		tune: false,
		value: 1.1,
		low: 0.6,
		high: 2.0,
		log: false,
		step: 0.05,
	},
	churn_penalty_multiplier: {
		label: "Churn Penalty",
		type: "float",
		tune: false,
		value: 1.0,
		low: 0.5,
		high: 2.0,
		log: false,
		step: 0.05,
	},
	churn_window: {
		label: "Churn Window",
		type: "int",
		tune: false,
		value: 6,
		low: 2,
		high: 24,
		log: false,
		step: 1,
	},
	reward_clip: {
		label: "Reward Clip",
		type: "float",
		tune: false,
		value: 200,
		low: 50,
		high: 500,
		log: false,
		step: 5,
	},
};

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
	const [rewardParamSettings, setRewardParamSettings] = useState<
		Record<RewardParamKey, RewardParamSetting>
	>(DEFAULT_REWARD_PARAM_SETTINGS);

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

	const usesPbrRewards = useMemo(
		() => selectedRewards.has("PBR") || selectedRewards.has("AdvancedPBR"),
		[selectedRewards],
	);

	const rewardParamSearchSpace = useMemo(() => {
		const space: Record<string, CampaignParamSpec> = {};
		(Object.keys(rewardParamSettings) as RewardParamKey[]).forEach((key) => {
			const setting = rewardParamSettings[key];
			space[key] = setting.tune
				? {
						mode: "tune",
						type: setting.type,
						low: setting.low,
						high: setting.high,
						log: setting.log,
					}
				: {
						mode: "fixed",
						type: setting.type,
						value: setting.value,
					};
		});
		return space;
	}, [rewardParamSettings]);

	const updateRewardParam = useCallback(
		<K extends keyof RewardParamSetting>(
			key: RewardParamKey,
			field: K,
			value: RewardParamSetting[K],
		) => {
			setRewardParamSettings((prev) => ({
				...prev,
				[key]: {
					...prev[key],
					[field]: value,
				},
			}));
		},
		[],
	);

	const handleLaunch = useCallback(async () => {
		if (!studyName.trim() || !datasetId) {
			setError("Study name and dataset are required");
			return;
		}
		if (selectedActions.size === 0 || selectedRewards.size === 0) {
			setError("Select at least one action and one reward scheme");
			return;
		}
		if (usesPbrRewards) {
			for (const [paramName, spec] of Object.entries(rewardParamSettings)) {
				if (spec.tune && spec.low >= spec.high) {
					setError(`Invalid range for ${paramName}: low must be less than high.`);
					return;
				}
			}
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
				search_space: usesPbrRewards ? rewardParamSearchSpace : undefined,
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
		usesPbrRewards,
		rewardParamSettings,
		rewardParamSearchSpace,
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

			{/* Row 3: Reward Param Search Space */}
			{usesPbrRewards && (
				<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-primary)] p-3">
					<div className="mb-2 flex items-center justify-between">
						<div>
							<p className="text-sm font-medium text-[var(--text-primary)]">
								PBR Reward Parameter Search
							</p>
							<p className="text-xs text-[var(--text-secondary)]">
								Set each parameter to fixed or tune it with Optuna.
							</p>
						</div>
						<span className="rounded bg-[var(--bg-secondary)] px-2 py-1 text-[10px] uppercase tracking-wide text-[var(--text-secondary)]">
							PBR / AdvancedPBR
						</span>
					</div>
					<div className="space-y-2">
						{(Object.keys(rewardParamSettings) as RewardParamKey[]).map((key) => {
							const setting = rewardParamSettings[key];
							return (
								<div
									key={key}
									className="grid grid-cols-12 items-center gap-2 rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] p-2"
								>
									<div className="col-span-3 text-xs font-medium text-[var(--text-primary)]">
										{setting.label}
									</div>
									<label className="col-span-2 flex items-center gap-1 text-xs text-[var(--text-secondary)]">
										<input
											type="checkbox"
											checked={setting.tune}
											onChange={(e) => updateRewardParam(key, "tune", e.target.checked)}
											className="accent-[var(--accent-blue)]"
										/>
										Tune
									</label>
									{setting.tune ? (
										<>
											<input
												type="number"
												step={setting.step}
												value={setting.low}
												onChange={(e) =>
													updateRewardParam(
														key,
														"low",
														setting.type === "int"
															? Number.parseInt(e.target.value || "0", 10)
															: Number(e.target.value),
													)
												}
												className="col-span-3 rounded border border-[var(--border-color)] bg-[var(--bg-primary)] px-2 py-1 text-xs text-[var(--text-primary)]"
												aria-label={`${setting.label} low`}
											/>
											<input
												type="number"
												step={setting.step}
												value={setting.high}
												onChange={(e) =>
													updateRewardParam(
														key,
														"high",
														setting.type === "int"
															? Number.parseInt(e.target.value || "0", 10)
															: Number(e.target.value),
													)
												}
												className="col-span-3 rounded border border-[var(--border-color)] bg-[var(--bg-primary)] px-2 py-1 text-xs text-[var(--text-primary)]"
												aria-label={`${setting.label} high`}
											/>
											<div className="col-span-1 text-[10px] text-[var(--text-secondary)]">
												lo/hi
											</div>
										</>
									) : (
										<>
											<input
												type="number"
												step={setting.step}
												value={setting.value}
												onChange={(e) =>
													updateRewardParam(
														key,
														"value",
														setting.type === "int"
															? Number.parseInt(e.target.value || "0", 10)
															: Number(e.target.value),
													)
												}
												className="col-span-6 rounded border border-[var(--border-color)] bg-[var(--bg-primary)] px-2 py-1 text-xs text-[var(--text-primary)]"
												aria-label={`${setting.label} fixed value`}
											/>
											<div className="col-span-2 text-[10px] text-[var(--text-secondary)]">
												fixed
											</div>
										</>
									)}
								</div>
							);
						})}
					</div>
				</div>
			)}

			{/* Row 4: Sliders */}
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
