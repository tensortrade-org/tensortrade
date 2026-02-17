"use client";

import { useApi } from "@/hooks/useApi";
import { getHyperparamPacks } from "@/lib/api";
import type { HyperparameterPack, TrainingConfig } from "@/lib/types";
import { useLaunchStore } from "@/stores/launchStore";
import { useCallback } from "react";

interface SliderConfig {
	key: keyof TrainingConfig;
	label: string;
	min: number;
	max: number;
	step: number;
	format: (v: number) => string;
}

const OVERRIDE_SLIDERS: SliderConfig[] = [
	{
		key: "learning_rate",
		label: "Learning Rate",
		min: 0.000001,
		max: 0.01,
		step: 0.000001,
		format: (v: number) => v.toExponential(1),
	},
	{
		key: "gamma",
		label: "Gamma",
		min: 0.9,
		max: 0.9999,
		step: 0.0001,
		format: (v: number) => v.toFixed(4),
	},
	{
		key: "entropy_coeff",
		label: "Entropy Coeff",
		min: 0,
		max: 0.1,
		step: 0.001,
		format: (v: number) => v.toFixed(3),
	},
	{
		key: "num_iterations",
		label: "Iterations",
		min: 10,
		max: 2000,
		step: 10,
		format: (v: number) => String(v),
	},
	{
		key: "train_batch_size",
		label: "Train Batch Size",
		min: 500,
		max: 20000,
		step: 500,
		format: (v: number) => String(v),
	},
];

interface ParamDisplayEntry {
	label: string;
	value: string;
}

function formatParamDisplay(config: TrainingConfig): ParamDisplayEntry[] {
	return [
		{ label: "Algorithm", value: config.algorithm },
		{ label: "Learning Rate", value: config.learning_rate.toExponential(1) },
		{ label: "Gamma", value: String(config.gamma) },
		{ label: "Iterations", value: String(config.num_iterations) },
		{ label: "Batch Size", value: String(config.train_batch_size) },
		{ label: "Workers", value: String(config.num_rollout_workers) },
		{ label: "Clip Param", value: String(config.clip_param) },
		{ label: "Entropy Coeff", value: String(config.entropy_coeff) },
	];
}

export function HyperparamStep() {
	const hpPackId = useLaunchStore((s) => s.hpPackId);
	const setHpPackId = useLaunchStore((s) => s.setHpPackId);
	const overrides = useLaunchStore((s) => s.overrides);
	const setOverride = useLaunchStore((s) => s.setOverride);
	const clearOverrides = useLaunchStore((s) => s.clearOverrides);

	const fetcher = useCallback(() => getHyperparamPacks(), []);
	const { data: packs, loading, error } = useApi<HyperparameterPack[]>(fetcher, []);

	const selectedPack = packs?.find((p) => p.id === hpPackId) ?? null;

	const handlePackChange = (packId: string) => {
		setHpPackId(packId || null);
		clearOverrides();
	};

	return (
		<div className="space-y-6">
			<div>
				<label
					htmlFor="hp-pack-select"
					className="mb-2 block text-sm font-medium text-[var(--text-primary)]"
				>
					Hyperparameter Pack
				</label>

				{loading && <p className="text-sm text-[var(--text-secondary)]">Loading packs...</p>}
				{error && (
					<p className="text-sm text-[var(--accent-red)]">Failed to load packs: {error.message}</p>
				)}

				{packs && (
					<select
						id="hp-pack-select"
						value={hpPackId ?? ""}
						onChange={(e) => handlePackChange(e.target.value)}
						className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
					>
						<option value="">-- Choose a pack --</option>
						{packs.map((pack) => (
							<option key={pack.id} value={pack.id}>
								{pack.name}
							</option>
						))}
					</select>
				)}
			</div>

			{selectedPack && (
				<>
					<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] p-4">
						<div className="mb-3 flex items-center justify-between">
							<h4 className="text-sm font-medium text-[var(--text-primary)]">
								{selectedPack.name}
							</h4>
							<span className="text-xs text-[var(--text-secondary)]">
								{selectedPack.description}
							</span>
						</div>
						<div className="grid grid-cols-2 gap-x-6 gap-y-2">
							{formatParamDisplay(selectedPack.config).map((entry) => (
								<div key={entry.label} className="flex justify-between text-xs">
									<span className="text-[var(--text-secondary)]">{entry.label}</span>
									<span className="font-mono text-[var(--text-primary)]">{entry.value}</span>
								</div>
							))}
						</div>
					</div>

					<div>
						<h4 className="mb-3 text-sm font-medium text-[var(--text-primary)]">
							Override Parameters
						</h4>
						<div className="space-y-4">
							{OVERRIDE_SLIDERS.map((slider) => {
								const baseValue = selectedPack.config[slider.key] as number;
								const overrideValue = overrides[slider.key] as number | undefined;
								const currentValue = overrideValue ?? baseValue;
								const isOverridden = overrideValue !== undefined;

								return (
									<div key={slider.key} className="space-y-1">
										<div className="flex items-center justify-between">
											<label
												htmlFor={`slider-${slider.key}`}
												className="text-xs text-[var(--text-secondary)]"
											>
												{slider.label}
											</label>
											<div className="flex items-center gap-2">
												<span
													className={`font-mono text-xs ${
														isOverridden
															? "text-[var(--accent-amber)]"
															: "text-[var(--text-primary)]"
													}`}
												>
													{slider.format(currentValue)}
												</span>
												{isOverridden && (
													<button
														type="button"
														onClick={() => setOverride(slider.key, baseValue)}
														className="text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
													>
														Reset
													</button>
												)}
											</div>
										</div>
										<input
											id={`slider-${slider.key}`}
											type="range"
											min={slider.min}
											max={slider.max}
											step={slider.step}
											value={currentValue}
											onChange={(e) =>
												setOverride(
													slider.key,
													Number.parseFloat(e.target.value) as TrainingConfig[typeof slider.key] &
														number,
												)
											}
											className="w-full accent-[var(--accent-blue)]"
										/>
										<div className="flex justify-between text-[10px] text-[var(--text-secondary)]">
											<span>{slider.format(slider.min)}</span>
											<span>{slider.format(slider.max)}</span>
										</div>
									</div>
								);
							})}
						</div>
					</div>
				</>
			)}

			{!hpPackId && packs && packs.length > 0 && (
				<p className="text-xs text-[var(--text-secondary)]">
					Select a hyperparameter pack to continue
				</p>
			)}
		</div>
	);
}
