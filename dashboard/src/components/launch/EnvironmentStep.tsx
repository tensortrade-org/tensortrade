"use client";

import type { TrainingConfig } from "@/lib/types";
import { useLaunchStore } from "@/stores/launchStore";

interface SelectFieldConfig {
	key: keyof TrainingConfig;
	label: string;
	options: string[];
	description: string;
}

interface NumberFieldConfig {
	key: keyof TrainingConfig;
	label: string;
	min: number;
	max: number;
	step: number;
	description: string;
}

const ACTION_SCHEMES: string[] = ["BSH", "SimpleOrders", "ManagedRiskOrders"];
const REWARD_SCHEMES: string[] = ["SimpleProfit", "RiskAdjustedReturns", "PBR", "AdvancedPBR"];

const SELECT_FIELDS: SelectFieldConfig[] = [
	{
		key: "action_scheme",
		label: "Action Scheme",
		options: ACTION_SCHEMES,
		description: "How the agent interacts with the market",
	},
	{
		key: "reward_scheme",
		label: "Reward Scheme",
		options: REWARD_SCHEMES,
		description: "How the agent is rewarded for its actions",
	},
];

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

	return (
		<div className="space-y-6">
			<p className="text-sm text-[var(--text-secondary)]">
				Configure the trading environment parameters. These override the hyperparameter pack
				defaults.
			</p>

			<div className="grid grid-cols-2 gap-6">
				{SELECT_FIELDS.map((field) => {
					const value = overrides[field.key] as string | undefined;
					return (
						<div key={field.key}>
							<label
								htmlFor={`env-${field.key}`}
								className="mb-1 block text-sm font-medium text-[var(--text-primary)]"
							>
								{field.label}
							</label>
							<p className="mb-2 text-xs text-[var(--text-secondary)]">{field.description}</p>
							<select
								id={`env-${field.key}`}
								value={value ?? ""}
								onChange={(e) => {
									const val = e.target.value;
									if (val) {
										setOverride(field.key, val as TrainingConfig[typeof field.key] & string);
									}
								}}
								className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
							>
								<option value="">-- Use pack default --</option>
								{field.options.map((opt) => (
									<option key={opt} value={opt}>
										{opt}
									</option>
								))}
							</select>
						</div>
					);
				})}
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
