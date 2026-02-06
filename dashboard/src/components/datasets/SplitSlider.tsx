"use client";

import type { SplitConfig } from "@/lib/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useCallback } from "react";

const inputClass =
	"w-20 rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-2 py-1 text-center text-sm text-[var(--text-primary)]";

interface SplitSegmentConfig {
	key: keyof SplitConfig;
	label: string;
	color: string;
	bgColor: string;
}

const SEGMENTS: SplitSegmentConfig[] = [
	{
		key: "train_pct",
		label: "Train",
		color: "var(--accent-green)",
		bgColor: "var(--accent-green)",
	},
	{
		key: "val_pct",
		label: "Validation",
		color: "var(--accent-blue)",
		bgColor: "var(--accent-blue)",
	},
	{
		key: "test_pct",
		label: "Test",
		color: "var(--accent-red)",
		bgColor: "var(--accent-red)",
	},
];

export function SplitSlider() {
	const { editingDataset, updateSplit } = useDatasetStore();

	const split: SplitConfig = editingDataset?.split_config ?? {
		train_pct: 0.7,
		val_pct: 0.15,
		test_pct: 0.15,
	};

	const handleChange = useCallback(
		(key: keyof SplitConfig, rawValue: string) => {
			const value = Number.parseFloat(rawValue);
			if (Number.isNaN(value) || value < 0 || value > 1) return;

			const newSplit = { ...split, [key]: value };

			// Adjust the other values to keep the total at 1.0
			const total = newSplit.train_pct + newSplit.val_pct + newSplit.test_pct;
			if (Math.abs(total - 1.0) > 0.001) {
				// Distribute the remainder proportionally to the other two segments
				const remainder = 1.0 - value;
				const otherKeys = SEGMENTS.filter((s) => s.key !== key).map((s) => s.key);
				const otherTotal = otherKeys.reduce((sum, k) => sum + split[k], 0);

				if (otherTotal > 0) {
					for (const otherKey of otherKeys) {
						newSplit[otherKey] = Math.max(
							0,
							Math.round((split[otherKey] / otherTotal) * remainder * 100) / 100,
						);
					}
				} else {
					// If other two are both 0, split remainder evenly
					for (const otherKey of otherKeys) {
						newSplit[otherKey] = Math.round((remainder / 2) * 100) / 100;
					}
				}

				// Fix rounding error on last segment
				const adjustedTotal = newSplit.train_pct + newSplit.val_pct + newSplit.test_pct;
				const diff = Math.round((1.0 - adjustedTotal) * 100) / 100;
				if (diff !== 0) {
					const lastOther = otherKeys[otherKeys.length - 1];
					newSplit[lastOther] = Math.max(0, Math.round((newSplit[lastOther] + diff) * 100) / 100);
				}
			}

			updateSplit(newSplit);
		},
		[split, updateSplit],
	);

	const total = Math.round((split.train_pct + split.val_pct + split.test_pct) * 100) / 100;
	const isValid = Math.abs(total - 1.0) < 0.01;

	return (
		<div className="space-y-4">
			<p className="text-sm text-[var(--text-secondary)]">
				Configure the train / validation / test split percentages.
			</p>

			{/* Visual bar */}
			<div className="flex h-8 overflow-hidden rounded-lg">
				{SEGMENTS.map((seg) => {
					const pct = split[seg.key] * 100;
					if (pct <= 0) return null;
					return (
						<div
							key={seg.key}
							className="flex items-center justify-center text-xs font-medium text-white"
							style={{
								width: `${pct}%`,
								backgroundColor: seg.bgColor,
								minWidth: pct > 0 ? "2rem" : undefined,
							}}
						>
							{pct.toFixed(0)}%
						</div>
					);
				})}
			</div>

			{/* Numeric inputs */}
			<div className="flex flex-wrap items-center gap-6">
				{SEGMENTS.map((seg) => (
					<div key={seg.key} className="flex items-center gap-2">
						<span className="h-3 w-3 rounded-full" style={{ backgroundColor: seg.bgColor }} />
						<label htmlFor={`split-${seg.key}`} className="text-sm text-[var(--text-secondary)]">
							{seg.label}
						</label>
						<input
							id={`split-${seg.key}`}
							type="number"
							value={split[seg.key]}
							onChange={(e) => handleChange(seg.key, e.target.value)}
							min={0}
							max={1}
							step={0.05}
							className={inputClass}
						/>
					</div>
				))}
			</div>

			{/* Validation */}
			{!isValid && (
				<p className="text-xs text-[var(--accent-red)]">
					Split percentages must sum to 1.0 (currently {total.toFixed(2)}).
				</p>
			)}
		</div>
	);
}
