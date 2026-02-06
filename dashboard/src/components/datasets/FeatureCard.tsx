"use client";

import type { FeatureCatalogEntry, FeatureParamDef } from "@/lib/types";
import { useCallback } from "react";

interface FeatureCardProps {
	entry: FeatureCatalogEntry;
	enabled: boolean;
	paramValues: Record<string, string | number | boolean>;
	onToggle: (featureType: string, enabled: boolean) => void;
	onParamChange: (featureType: string, paramName: string, value: string | number | boolean) => void;
}

function ParamInput({
	param,
	value,
	onChange,
}: {
	param: FeatureParamDef;
	value: string | number | boolean;
	onChange: (name: string, value: string | number | boolean) => void;
}) {
	const handleChange = useCallback(
		(e: React.ChangeEvent<HTMLInputElement>) => {
			if (param.type === "int" || param.type === "float") {
				onChange(param.name, Number(e.target.value));
			} else if (param.type === "bool") {
				onChange(param.name, e.target.checked);
			} else {
				onChange(param.name, e.target.value);
			}
		},
		[param.name, param.type, onChange],
	);

	if (param.type === "bool") {
		return (
			<label className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
				<input
					type="checkbox"
					checked={Boolean(value)}
					onChange={handleChange}
					className="accent-[var(--accent-blue)]"
				/>
				{param.description}
			</label>
		);
	}

	return (
		<div className="flex flex-col gap-1">
			<label className="text-xs text-[var(--text-secondary)]" htmlFor={`param-${param.name}`}>
				{param.description}
			</label>
			<input
				id={`param-${param.name}`}
				type={param.type === "int" || param.type === "float" ? "number" : "text"}
				value={String(value)}
				min={param.min}
				max={param.max}
				step={param.type === "float" ? 0.01 : 1}
				onChange={handleChange}
				className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-2 py-1 text-xs text-[var(--text-primary)]"
			/>
		</div>
	);
}

export function FeatureCard({
	entry,
	enabled,
	paramValues,
	onToggle,
	onParamChange,
}: FeatureCardProps) {
	const handleToggle = useCallback(() => {
		onToggle(entry.type, !enabled);
	}, [entry.type, enabled, onToggle]);

	const handleParamChange = useCallback(
		(paramName: string, value: string | number | boolean) => {
			onParamChange(entry.type, paramName, value);
		},
		[entry.type, onParamChange],
	);

	return (
		<div
			className={`rounded-lg border p-3 transition-colors ${
				enabled
					? "border-[var(--accent-blue)] bg-[var(--bg-secondary)]"
					: "border-[var(--border-color)] bg-[var(--bg-card)]"
			}`}
		>
			<div className="flex items-start justify-between gap-2">
				<div className="min-w-0 flex-1">
					<p className="text-sm font-medium text-[var(--text-primary)]">{entry.name}</p>
					<p className="mt-0.5 text-xs text-[var(--text-secondary)]">{entry.description}</p>
				</div>
				<button
					type="button"
					onClick={handleToggle}
					className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${
						enabled ? "bg-[var(--accent-blue)]" : "bg-[var(--border-color)]"
					}`}
					aria-label={`Toggle ${entry.name}`}
				>
					<span
						className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
							enabled ? "translate-x-4" : "translate-x-0"
						}`}
					/>
				</button>
			</div>

			{enabled && entry.params.length > 0 && (
				<div className="mt-3 space-y-2 border-t border-[var(--border-color)] pt-3">
					{entry.params.map((param) => (
						<ParamInput
							key={param.name}
							param={param}
							value={paramValues[param.name] ?? param.default}
							onChange={handleParamChange}
						/>
					))}
				</div>
			)}
		</div>
	);
}
