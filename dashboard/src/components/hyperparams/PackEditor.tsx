"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { ParamSelect } from "@/components/hyperparams/ParamSelect";
import { ParamSlider } from "@/components/hyperparams/ParamSlider";
import {
	createHyperparamPack,
	deleteHyperparamPack,
	duplicateHyperparamPack,
	getHyperparamPacks,
	updateHyperparamPack,
} from "@/lib/api";
import { ACTION_GROUPS, ACTION_SCHEMES, REWARD_SCHEMES, isCompatible } from "@/lib/scheme-compat";
import type { HyperparameterPack, ModelConfig, TrainingConfig } from "@/lib/types";
import { useHyperparamStore } from "@/stores/hyperparamStore";
import { useCallback, useEffect, useState } from "react";

interface SectionProps {
	title: string;
	children: React.ReactNode;
}

function Section({ title, children }: SectionProps) {
	return (
		<Card className="space-y-1">
			<CardHeader title={title} />
			{children}
		</Card>
	);
}

const DEFAULT_CONFIG: TrainingConfig = {
	algorithm: "PPO",
	learning_rate: 5e-5,
	gamma: 0.99,
	lambda_: 0.95,
	clip_param: 0.3,
	entropy_coeff: 0.01,
	vf_loss_coeff: 1.0,
	num_sgd_iter: 10,
	sgd_minibatch_size: 128,
	train_batch_size: 4000,
	num_rollout_workers: 2,
	rollout_fragment_length: 200,
	model: {
		fcnet_hiddens: [256, 256],
		fcnet_activation: "relu",
	},
	action_scheme: "BSH",
	reward_scheme: "SimpleProfit",
	reward_params: {},
	commission: 0.001,
	initial_cash: 10000,
	window_size: 50,
	max_allowed_loss: 0.5,
	max_episode_steps: null,
	num_iterations: 100,
};

const ANTI_CHURN_REWARD_DEFAULTS = {
	trade_penalty_multiplier: 1.1,
	churn_penalty_multiplier: 1.0,
	churn_window: 6,
	reward_clip: 200.0,
} as const;

interface LayerEditorProps {
	layers: number[];
	onChange: (layers: number[]) => void;
}

function LayerEditor({ layers, onChange }: LayerEditorProps) {
	const handleLayerChange = useCallback(
		(index: number, value: number) => {
			const updated = [...layers];
			updated[index] = value;
			onChange(updated);
		},
		[layers, onChange],
	);

	const addLayer = useCallback(() => {
		onChange([...layers, 128]);
	}, [layers, onChange]);

	const removeLayer = useCallback(
		(index: number) => {
			if (layers.length <= 1) return;
			const updated = layers.filter((_, i) => i !== index);
			onChange(updated);
		},
		[layers, onChange],
	);

	return (
		<div className="space-y-2 py-2">
			<div className="flex items-center justify-between">
				<span className="text-sm text-[var(--text-primary)]">Hidden Layers</span>
				<button
					type="button"
					onClick={addLayer}
					className="rounded bg-[var(--bg-secondary)] px-2 py-0.5 text-xs text-[var(--accent-blue)] hover:opacity-80 transition-opacity border border-[var(--border-color)]"
				>
					+ Add Layer
				</button>
			</div>
			<div className="flex flex-wrap gap-2">
				{layers.map((size, idx) => {
					const layerKey = `layer-${idx}`;
					return (
						<div
							key={layerKey}
							className="flex items-center gap-1 rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] px-2 py-1"
						>
							<input
								type="number"
								value={size}
								min={8}
								max={1024}
								step={8}
								onChange={(e) => {
									const v = Number(e.target.value);
									if (!Number.isNaN(v) && v > 0) handleLayerChange(idx, v);
								}}
								className="w-16 bg-transparent text-center text-xs font-mono text-[var(--text-primary)] focus:outline-none"
							/>
							{layers.length > 1 && (
								<button
									type="button"
									onClick={() => removeLayer(idx)}
									className="text-xs text-[var(--text-secondary)] hover:text-[var(--accent-red)] transition-colors"
								>
									x
								</button>
							)}
						</div>
					);
				})}
			</div>
			<p className="text-xs text-[var(--text-secondary)]">Sizes of fully-connected hidden layers</p>
		</div>
	);
}

interface StatusMessageState {
	text: string;
	variant: "success" | "error";
}

export function PackEditor() {
	const {
		editingPack,
		selectedPackId,
		packs,
		setPacks,
		selectPack,
		setEditingPack,
		updateEditingConfig,
	} = useHyperparamStore();
	const [saving, setSaving] = useState(false);
	const [statusMessage, setStatusMessage] = useState<StatusMessageState | null>(null);

	const isNewPack = selectedPackId === null;
	const pack = editingPack;
	const config = pack?.config ?? DEFAULT_CONFIG;
	const rewardParams = {
		...ANTI_CHURN_REWARD_DEFAULTS,
		...(config.reward_params ?? {}),
	};

	const updateModel = useCallback(
		(modelUpdate: Partial<ModelConfig>) => {
			updateEditingConfig("model", { ...config.model, ...modelUpdate });
		},
		[config.model, updateEditingConfig],
	);

	const updateRewardParam = useCallback(
		(key: keyof typeof ANTI_CHURN_REWARD_DEFAULTS, value: number) => {
			updateEditingConfig("reward_params", {
				...rewardParams,
				[key]: key === "churn_window" ? Math.round(value) : value,
			});
		},
		[rewardParams, updateEditingConfig],
	);

	const handleNameChange = useCallback(
		(e: React.ChangeEvent<HTMLInputElement>) => {
			if (!pack) return;
			setEditingPack({ ...pack, name: e.target.value });
		},
		[pack, setEditingPack],
	);

	const handleDescriptionChange = useCallback(
		(e: React.ChangeEvent<HTMLTextAreaElement>) => {
			if (!pack) return;
			setEditingPack({ ...pack, description: e.target.value });
		},
		[pack, setEditingPack],
	);

	const showStatus = useCallback((text: string, variant: "success" | "error") => {
		setStatusMessage({ text, variant });
		setTimeout(() => setStatusMessage(null), 3000);
	}, []);

	const refreshPacks = useCallback(async () => {
		const freshPacks = await getHyperparamPacks();
		setPacks(freshPacks);
	}, [setPacks]);

	const normalizeConfigForSave = useCallback((cfg: TrainingConfig): TrainingConfig => {
		if (
			cfg.reward_scheme !== "PBR" &&
			cfg.reward_scheme !== "AdvancedPBR" &&
			cfg.reward_scheme !== "TrendPBR"
		) {
			return {
				...cfg,
				reward_params: {},
			};
		}
		return {
			...cfg,
			reward_params: {
				...ANTI_CHURN_REWARD_DEFAULTS,
				...(cfg.reward_params ?? {}),
			},
		};
	}, []);

	const handleSave = useCallback(async () => {
		if (!pack) return;
		setSaving(true);
		try {
			const configToSave = normalizeConfigForSave(pack.config);
			if (isNewPack || !pack.id) {
				const created = await createHyperparamPack({
					name: pack.name || "New Pack",
					description: pack.description,
					config: configToSave,
				});
				await refreshPacks();
				selectPack(created.id);
				showStatus("Pack created", "success");
			} else {
				const updated = await updateHyperparamPack(pack.id, {
					name: pack.name,
					description: pack.description,
					config: configToSave,
				});
				setPacks(packs.map((p) => (p.id === updated.id ? updated : p)));
				showStatus("Pack saved", "success");
			}
		} catch {
			showStatus("Error saving pack", "error");
		} finally {
			setSaving(false);
		}
	}, [
		pack,
		isNewPack,
		showStatus,
		refreshPacks,
		selectPack,
		packs,
		setPacks,
		normalizeConfigForSave,
	]);

	const handleSaveAs = useCallback(async () => {
		if (!pack) return;
		setSaving(true);
		try {
			const configToSave = normalizeConfigForSave(pack.config);
			const created = await createHyperparamPack({
				name: `${pack.name} (copy)`,
				description: pack.description,
				config: configToSave,
			});
			await refreshPacks();
			selectPack(created.id);
			showStatus("Pack saved as copy", "success");
		} catch {
			showStatus("Error saving copy", "error");
		} finally {
			setSaving(false);
		}
	}, [pack, showStatus, refreshPacks, selectPack, normalizeConfigForSave]);

	const handleDuplicate = useCallback(async () => {
		if (!pack || isNewPack) return;
		setSaving(true);
		try {
			const dup = await duplicateHyperparamPack(pack.id);
			await refreshPacks();
			selectPack(dup.id);
			showStatus("Pack duplicated", "success");
		} catch {
			showStatus("Error duplicating pack", "error");
		} finally {
			setSaving(false);
		}
	}, [pack, isNewPack, showStatus, refreshPacks, selectPack]);

	const handleDelete = useCallback(async () => {
		if (!pack || isNewPack) return;
		setSaving(true);
		try {
			await deleteHyperparamPack(pack.id);
			const remaining = packs.filter((p) => p.id !== pack.id);
			setPacks(remaining);
			selectPack(remaining.length > 0 ? remaining[0].id : null);
			showStatus("Pack deleted", "success");
		} catch {
			showStatus("Error deleting pack", "error");
		} finally {
			setSaving(false);
		}
	}, [pack, isNewPack, showStatus, packs, setPacks, selectPack]);

	// Initialize editing pack for new pack mode (in effect to avoid setState during render)
	useEffect(() => {
		if (!editingPack && isNewPack) {
			setEditingPack({
				id: "",
				name: "New Pack",
				description: "",
				config: { ...DEFAULT_CONFIG, model: { ...DEFAULT_CONFIG.model } },
				created_at: new Date().toISOString(),
				updated_at: new Date().toISOString(),
			});
		}
	}, [editingPack, isNewPack, setEditingPack]);

	if (!pack) {
		return (
			<div className="flex h-full items-center justify-center">
				<p className="text-sm text-[var(--text-secondary)]">
					Select a pack to edit, or create a new one.
				</p>
			</div>
		);
	}

	return (
		<div className="space-y-4 overflow-y-auto pb-8">
			{/* Header: Name, Description, Action Buttons */}
			<Card>
				<div className="space-y-3">
					<div className="flex-1 space-y-2">
						<input
							type="text"
							value={pack.name}
							onChange={handleNameChange}
							placeholder="Pack name"
							className="w-full rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-lg font-semibold text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
						/>
						<textarea
							value={pack.description}
							onChange={handleDescriptionChange}
							placeholder="Description..."
							rows={2}
							className="w-full rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-secondary)] resize-none focus:border-[var(--accent-blue)] focus:outline-none"
						/>
					</div>
					<div className="flex items-center gap-2">
						<button
							type="button"
							onClick={handleSave}
							disabled={saving}
							className="rounded bg-[var(--accent-blue)] px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 transition-opacity disabled:opacity-50"
						>
							{saving ? "Saving..." : isNewPack ? "Create" : "Save"}
						</button>
						{!isNewPack && pack.id && (
							<>
								<button
									type="button"
									onClick={handleSaveAs}
									disabled={saving}
									className="rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] px-4 py-1.5 text-sm text-[var(--text-primary)] hover:bg-[var(--bg-card)] transition-colors disabled:opacity-50"
								>
									Save As
								</button>
								<button
									type="button"
									onClick={handleDuplicate}
									disabled={saving}
									className="rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] px-4 py-1.5 text-sm text-[var(--text-primary)] hover:bg-[var(--bg-card)] transition-colors disabled:opacity-50"
								>
									Duplicate
								</button>
								<div className="flex-1" />
								<button
									type="button"
									onClick={handleDelete}
									disabled={saving}
									className="rounded border border-[var(--accent-red)]/30 bg-[var(--accent-red)]/10 px-4 py-1.5 text-sm text-[var(--accent-red)] hover:bg-[var(--accent-red)]/20 transition-colors disabled:opacity-50"
								>
									Delete
								</button>
							</>
						)}
						{statusMessage && (
							<span
								className={`ml-2 text-xs ${
									statusMessage.variant === "success"
										? "text-[var(--accent-green)]"
										: "text-[var(--accent-red)]"
								}`}
							>
								{statusMessage.text}
							</span>
						)}
					</div>
				</div>
			</Card>

			{/* Algorithm Section */}
			<Section title="Algorithm">
				<ParamSelect
					label="Algorithm"
					value={config.algorithm}
					options={[{ value: "PPO", label: "PPO (Proximal Policy Optimization)" }]}
					onChange={(v) => updateEditingConfig("algorithm", v)}
					description="RL algorithm for training"
				/>
			</Section>

			{/* Network Section */}
			<Section title="Network Architecture">
				<LayerEditor
					layers={config.model.fcnet_hiddens}
					onChange={(layers) => updateModel({ fcnet_hiddens: layers })}
				/>
				<ParamSelect
					label="Activation"
					value={config.model.fcnet_activation}
					options={[
						{ value: "relu", label: "ReLU" },
						{ value: "tanh", label: "Tanh" },
					]}
					onChange={(v) => updateModel({ fcnet_activation: v })}
					description="Activation function for hidden layers"
				/>
			</Section>

			{/* Training Loop Section */}
			<Section title="Training Loop">
				<ParamSlider
					label="Learning Rate"
					value={config.learning_rate}
					min={0.000001}
					max={0.01}
					step={0.000001}
					onChange={(v) => updateEditingConfig("learning_rate", v)}
					description="Step size for gradient descent optimization"
					format={(v) => v.toExponential(1)}
				/>
				<ParamSlider
					label="Gamma (Discount)"
					value={config.gamma}
					min={0.9}
					max={0.9999}
					step={0.0001}
					onChange={(v) => updateEditingConfig("gamma", v)}
					description="Discount factor for future rewards"
				/>
				<ParamSlider
					label="Lambda (GAE)"
					value={config.lambda_}
					min={0.9}
					max={1.0}
					step={0.01}
					onChange={(v) => updateEditingConfig("lambda_", v)}
					description="GAE lambda for advantage estimation"
				/>
				<ParamSlider
					label="Clip Param"
					value={config.clip_param}
					min={0.1}
					max={0.5}
					step={0.01}
					onChange={(v) => updateEditingConfig("clip_param", v)}
					description="PPO clipping parameter for policy updates"
				/>
				<ParamSlider
					label="Entropy Coeff"
					value={config.entropy_coeff}
					min={0}
					max={0.1}
					step={0.001}
					onChange={(v) => updateEditingConfig("entropy_coeff", v)}
					description="Entropy bonus to encourage exploration"
				/>
				<ParamSlider
					label="VF Loss Coeff"
					value={config.vf_loss_coeff}
					min={0.1}
					max={2.0}
					step={0.1}
					onChange={(v) => updateEditingConfig("vf_loss_coeff", v)}
					description="Value function loss coefficient"
				/>
				<ParamSlider
					label="Num SGD Iterations"
					value={config.num_sgd_iter}
					min={1}
					max={80}
					step={1}
					onChange={(v) => updateEditingConfig("num_sgd_iter", v)}
					description="Number of SGD passes per training batch"
				/>
				<ParamSlider
					label="SGD Minibatch Size"
					value={config.sgd_minibatch_size}
					min={32}
					max={8192}
					step={32}
					onChange={(v) => updateEditingConfig("sgd_minibatch_size", v)}
					description="Minibatch size for SGD updates"
				/>
				<ParamSlider
					label="Train Batch Size"
					value={config.train_batch_size}
					min={256}
					max={65536}
					step={256}
					onChange={(v) => updateEditingConfig("train_batch_size", v)}
					description="Total training batch size per iteration"
				/>
			</Section>

			{/* Environment Section */}
			<Section title="Environment">
				<ParamSelect
					label="Action Scheme"
					value={config.action_scheme}
					options={ACTION_SCHEMES.map((a) => ({ value: a.value, label: a.label }))}
					groups={ACTION_GROUPS}
					optionGroup={(value) => ACTION_SCHEMES.find((a) => a.value === value)?.group}
					onChange={(v) =>
						updateEditingConfig("action_scheme", v as TrainingConfig["action_scheme"])
					}
					description="Trading action scheme for the environment"
				/>
				<ParamSelect
					label="Reward Scheme"
					value={config.reward_scheme}
					options={REWARD_SCHEMES.map((r) => ({ value: r.value, label: r.label }))}
					onChange={(v) => {
						const rewardScheme = v as TrainingConfig["reward_scheme"];
						updateEditingConfig("reward_scheme", rewardScheme);
						if (rewardScheme === "PBR" || rewardScheme === "AdvancedPBR") {
							updateEditingConfig("reward_params", {
								...ANTI_CHURN_REWARD_DEFAULTS,
								...(config.reward_params ?? {}),
							});
						} else {
							updateEditingConfig("reward_params", {});
						}
					}}
					description="Reward calculation scheme"
				/>
				{!isCompatible(config.action_scheme, config.reward_scheme) && (
					<p className="text-xs text-amber-500 mt-1 px-1">
						{config.reward_scheme} requires Discrete(3) BSH-style actions. {config.action_scheme} is
						not compatible — use SimpleProfit or RiskAdjustedReturns.
					</p>
				)}
				{(config.reward_scheme === "PBR" ||
					config.reward_scheme === "AdvancedPBR" ||
					config.reward_scheme === "TrendPBR") && (
					<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-tertiary)] p-3 mt-3 space-y-3">
						<div className="flex items-center justify-between">
							<p className="text-xs font-medium text-[var(--text-primary)]">
								{config.reward_scheme} Reward Parameters
							</p>
							<button
								type="button"
								onClick={() =>
									updateEditingConfig("reward_params", { ...ANTI_CHURN_REWARD_DEFAULTS })
								}
								className="rounded border border-[var(--border-color)] px-2 py-1 text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-secondary)]"
							>
								Reset Defaults
							</button>
						</div>

						<ParamSlider
							label="Trade Penalty Multiplier"
							value={rewardParams.trade_penalty_multiplier}
							min={0.1}
							max={5.0}
							step={0.1}
							onChange={(v) => updateRewardParam("trade_penalty_multiplier", v)}
							description="Base trade-cost penalty applied on each executed trade"
							format={(v) => v.toFixed(2)}
						/>
						<ParamSlider
							label="Churn Penalty Multiplier"
							value={rewardParams.churn_penalty_multiplier}
							min={0.0}
							max={5.0}
							step={0.1}
							onChange={(v) => updateRewardParam("churn_penalty_multiplier", v)}
							description="Extra penalty for quick flip trades in the churn window"
							format={(v) => v.toFixed(2)}
						/>
						<ParamSlider
							label="Churn Window"
							value={rewardParams.churn_window}
							min={1}
							max={48}
							step={1}
							onChange={(v) => updateRewardParam("churn_window", v)}
							description="Window size in steps used to detect churn trades"
							format={(v) => String(Math.round(v))}
						/>
						<ParamSlider
							label="Reward Clip"
							value={rewardParams.reward_clip}
							min={1}
							max={2000}
							step={1}
							onChange={(v) => updateRewardParam("reward_clip", v)}
							description="Clips extreme reward values for stability"
							format={(v) => String(Math.round(v))}
						/>
					</div>
				)}
				<ParamSlider
					label="Commission"
					value={config.commission}
					min={0}
					max={0.01}
					step={0.0001}
					onChange={(v) => updateEditingConfig("commission", v)}
					description="Trading commission per transaction"
					format={(v) => `${(v * 100).toFixed(2)}%`}
				/>
				<ParamSlider
					label="Initial Cash"
					value={config.initial_cash}
					min={1000}
					max={1000000}
					step={1000}
					onChange={(v) => updateEditingConfig("initial_cash", v)}
					description="Starting cash for each episode"
					format={(v) => `$${v.toLocaleString()}`}
				/>
				<ParamSlider
					label="Window Size"
					value={config.window_size}
					min={5}
					max={200}
					step={5}
					onChange={(v) => updateEditingConfig("window_size", v)}
					description="Observation window size (number of past candles)"
				/>
				<ParamSlider
					label="Max Allowed Loss"
					value={config.max_allowed_loss}
					min={0.1}
					max={1.0}
					step={0.05}
					onChange={(v) => updateEditingConfig("max_allowed_loss", v)}
					description="Maximum drawdown before episode terminates"
					format={(v) => `${(v * 100).toFixed(0)}%`}
				/>
				<ParamSlider
					label="Num Iterations"
					value={config.num_iterations}
					min={1}
					max={500}
					step={1}
					onChange={(v) => updateEditingConfig("num_iterations", v)}
					description="Number of training iterations to run"
				/>
			</Section>

			{/* Rollout Section */}
			<Section title="Rollout">
				<ParamSlider
					label="Num Rollout Workers"
					value={config.num_rollout_workers}
					min={0}
					max={16}
					step={1}
					onChange={(v) => updateEditingConfig("num_rollout_workers", v)}
					description="Number of parallel rollout workers for data collection"
				/>
				<ParamSlider
					label="Rollout Fragment Length"
					value={config.rollout_fragment_length}
					min={50}
					max={2000}
					step={50}
					onChange={(v) => updateEditingConfig("rollout_fragment_length", v)}
					description="Length of each rollout fragment"
				/>
			</Section>
		</div>
	);
}
