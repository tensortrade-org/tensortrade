"use client";

import { ACTION_GROUPS, ACTION_SCHEMES, REWARD_SCHEMES, isCompatible } from "@/lib/scheme-compat";
import type { TrainingConfig } from "@/lib/types";
import { useLaunchStore } from "@/stores/launchStore";

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

export function EnvironmentStep() {
	const overrides = useLaunchStore((s) => s.overrides);
	const setOverride = useLaunchStore((s) => s.setOverride);

	const actionScheme = overrides.action_scheme as string | undefined;
	const rewardScheme = overrides.reward_scheme as string | undefined;
	const showWarning = actionScheme && rewardScheme && !isCompatible(actionScheme, rewardScheme);

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
