"use client";

import { useApi } from "@/hooks/useApi";
import { getHyperparamPacks } from "@/lib/api";
import { ACTION_GROUPS, ACTION_SCHEMES, REWARD_SCHEMES, isCompatible } from "@/lib/scheme-compat";
import type { HyperparameterPack, TrainingConfig } from "@/lib/types";
import { useLaunchStore } from "@/stores/launchStore";
import { useCallback } from "react";

interface NumberFieldConfig {
	key: keyof TrainingConfig;
	label: string;
	min: number;
	max: number;
	step: number;
	description: string;
}

const NUMBER_FIELDS: NumberFieldConfig[] = [
	{
		key: "commission",
		label: "Commission",
		min: 0,
		max: 0.01,
		step: 0.0001,
		description: "Trading commission rate",
	},
	{
		key: "initial_cash",
		label: "Initial Cash ($)",
		min: 1000,
		max: 1000000,
		step: 1000,
		description: "Starting cash balance",
	},
	{
		key: "window_size",
		label: "Window Size",
		min: 5,
		max: 200,
		step: 1,
		description: "Observation window lookback steps",
	},
	{
		key: "max_allowed_loss",
		label: "Max Allowed Loss",
		min: 0.1,
		max: 1.0,
		step: 0.05,
		description: "Maximum allowed portfolio loss (fraction)",
	},
];

const ANTI_CHURN_REWARD_DEFAULTS = {
	trade_penalty_multiplier: 1.1,
	churn_penalty_multiplier: 1.0,
	churn_window: 6,
	reward_clip: 200.0,
} as const;

type AntiChurnRewardParamKey = keyof typeof ANTI_CHURN_REWARD_DEFAULTS;

interface AntiChurnRewardFieldConfig {
	key: AntiChurnRewardParamKey;
	label: string;
	description: string;
	min: number;
	max: number;
	step: number;
}

const ANTI_CHURN_REWARD_FIELDS: AntiChurnRewardFieldConfig[] = [
	{
		key: "trade_penalty_multiplier",
		label: "Trade Penalty Multiplier",
		description: "Base trade-cost penalty multiplier for each executed trade",
		min: 0.1,
		max: 5.0,
		step: 0.1,
	},
	{
		key: "churn_penalty_multiplier",
		label: "Churn Penalty Multiplier",
		description: "Extra penalty for quick flip trades within churn window",
		min: 0.0,
		max: 5.0,
		step: 0.1,
	},
	{
		key: "churn_window",
		label: "Churn Window (steps)",
		description: "Number of steps used to detect quick churn trades",
		min: 1,
		max: 48,
		step: 1,
	},
	{
		key: "reward_clip",
		label: "Reward Clip",
		description: "Clamps extreme reward values to stabilize training updates",
		min: 1,
		max: 2000,
		step: 1,
	},
];

export function EnvironmentStep() {
	const overrides = useLaunchStore((s) => s.overrides);
	const setOverride = useLaunchStore((s) => s.setOverride);
	const hpPackId = useLaunchStore((s) => s.hpPackId);

	const packFetcher = useCallback(() => getHyperparamPacks(), []);
	const { data: packs } = useApi<HyperparameterPack[]>(packFetcher, []);
	const selectedPack = packs?.find((p) => p.id === hpPackId) ?? null;

	const actionScheme = overrides.action_scheme as string | undefined;
	const rewardScheme = overrides.reward_scheme as string | undefined;
	const effectiveRewardScheme = rewardScheme ?? selectedPack?.config.reward_scheme;
	const showWarning = actionScheme && rewardScheme && !isCompatible(actionScheme, rewardScheme);
	const showAntiChurnParams =
		effectiveRewardScheme === "PBR" || effectiveRewardScheme === "AdvancedPBR";

	const baseRewardParams = (selectedPack?.config.reward_params ?? {}) as Record<string, number>;
	const overrideRewardParams = (overrides.reward_params ?? {}) as Record<string, number>;
	const effectiveAntiChurnParams: Record<AntiChurnRewardParamKey, number> = {
		...ANTI_CHURN_REWARD_DEFAULTS,
		...baseRewardParams,
		...overrideRewardParams,
	};

	const setAntiChurnParam = (key: AntiChurnRewardParamKey, value: number) => {
		const next = {
			...effectiveAntiChurnParams,
			[key]: value,
		};
		setOverride("reward_params", next);
	};

	return (
		<div className="space-y-6">
			<p className="text-sm text-[var(--text-secondary)]">
				Configure the trading environment parameters. These override the hyperparameter pack
				defaults.
			</p>

			<div className="grid grid-cols-2 gap-6">
				{/* Action Scheme */}
				<div>
					<label
						htmlFor="env-action_scheme"
						className="mb-1 block text-sm font-medium text-[var(--text-primary)]"
					>
						Action Scheme
					</label>
					<p className="mb-2 text-xs text-[var(--text-secondary)]">
						How the agent interacts with the market
					</p>
					<select
						id="env-action_scheme"
						value={actionScheme ?? ""}
						onChange={(e) => {
							const val = e.target.value;
							if (val) {
								setOverride("action_scheme", val as TrainingConfig["action_scheme"]);
							}
						}}
						className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
					>
						<option value="">-- Use pack default --</option>
						{ACTION_GROUPS.map((group) => (
							<optgroup key={group} label={group}>
								{ACTION_SCHEMES.filter((a) => a.group === group).map((a) => (
									<option key={a.value} value={a.value}>
										{a.label}
									</option>
								))}
							</optgroup>
						))}
					</select>
				</div>

				{/* Reward Scheme */}
				<div>
					<label
						htmlFor="env-reward_scheme"
						className="mb-1 block text-sm font-medium text-[var(--text-primary)]"
					>
						Reward Scheme
					</label>
					<p className="mb-2 text-xs text-[var(--text-secondary)]">
						How the agent is rewarded for its actions
					</p>
					<select
						id="env-reward_scheme"
						value={rewardScheme ?? ""}
						onChange={(e) => {
							const val = e.target.value;
							if (val) {
								setOverride("reward_scheme", val as TrainingConfig["reward_scheme"]);
								if (val === "PBR" || val === "AdvancedPBR") {
									setOverride("reward_params", { ...effectiveAntiChurnParams });
								} else {
									setOverride("reward_params", {});
								}
							}
						}}
						className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
					>
						<option value="">-- Use pack default --</option>
						{REWARD_SCHEMES.map((r) => (
							<option key={r.value} value={r.value}>
								{r.label}
							</option>
						))}
					</select>
					{showWarning && (
						<p className="text-xs text-amber-500 mt-1">
							{rewardScheme} requires Discrete(3) BSH-style actions. {actionScheme} is not
							compatible — use SimpleProfit or RiskAdjustedReturns.
						</p>
					)}
				</div>
			</div>

			{showAntiChurnParams && (
				<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] p-4">
					<div className="mb-3 flex items-center justify-between">
						<div>
							<h4 className="text-sm font-medium text-[var(--text-primary)]">
								{effectiveRewardScheme} Reward Parameters
							</h4>
							<p className="text-xs text-[var(--text-secondary)]">
								Anti-overtrading controls for {effectiveRewardScheme}.
							</p>
						</div>
						<button
							type="button"
							onClick={() => setOverride("reward_params", { ...ANTI_CHURN_REWARD_DEFAULTS })}
							className="rounded border border-[var(--border-color)] px-2 py-1 text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]"
						>
							Reset Defaults
						</button>
					</div>

					<div className="grid grid-cols-2 gap-4">
						{ANTI_CHURN_REWARD_FIELDS.map((field) => (
							<div key={field.key}>
								<label
									htmlFor={`env-${field.key}`}
									className="mb-1 block text-sm font-medium text-[var(--text-primary)]"
								>
									{field.label}
								</label>
								<p className="mb-2 text-xs text-[var(--text-secondary)]">{field.description}</p>
								<input
									id={`env-${field.key}`}
									type="number"
									min={field.min}
									max={field.max}
									step={field.step}
									value={effectiveAntiChurnParams[field.key]}
									onChange={(e) => {
										const raw = e.target.value;
										const num = Number.parseFloat(raw);
										if (!Number.isNaN(num)) {
											setAntiChurnParam(
												field.key,
												field.key === "churn_window" ? Math.round(num) : num,
											);
										}
									}}
									className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-primary)] px-3 py-2 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
								/>
							</div>
						))}
					</div>
				</div>
			)}

			<div className="grid grid-cols-2 gap-6">
				{NUMBER_FIELDS.map((field) => {
					const value = overrides[field.key] as number | undefined;
					return (
						<div key={field.key}>
							<label
								htmlFor={`env-${field.key}`}
								className="mb-1 block text-sm font-medium text-[var(--text-primary)]"
							>
								{field.label}
							</label>
							<p className="mb-2 text-xs text-[var(--text-secondary)]">{field.description}</p>
							<input
								id={`env-${field.key}`}
								type="number"
								min={field.min}
								max={field.max}
								step={field.step}
								value={value ?? ""}
								placeholder="Use pack default"
								onChange={(e) => {
									const raw = e.target.value;
									if (raw === "") {
										return;
									}
									const num = Number.parseFloat(raw);
									if (!Number.isNaN(num)) {
										setOverride(field.key, num as TrainingConfig[typeof field.key] & number);
									}
								}}
								className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)]/50 focus:border-[var(--accent-blue)] focus:outline-none"
							/>
						</div>
					);
				})}
			</div>
		</div>
	);
}
