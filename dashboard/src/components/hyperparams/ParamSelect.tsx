"use client";

import { useId } from "react";

interface ParamSelectOption {
	value: string;
	label: string;
}

interface ParamSelectProps {
	label: string;
	value: string;
	options: ParamSelectOption[];
	onChange: (value: string) => void;
	description?: string;
}

export function ParamSelect({ label, value, options, onChange, description }: ParamSelectProps) {
	const selectId = useId();
	return (
		<div className="flex flex-col gap-1 py-2">
			<div className="flex items-center justify-between gap-4">
				<label htmlFor={selectId} className="text-sm text-[var(--text-primary)] shrink-0">
					{label}
				</label>
				<select
					id={selectId}
					value={value}
					onChange={(e) => onChange(e.target.value)}
					className="rounded border border-[var(--border-color)] bg-[var(--bg-card)] px-3 py-1.5 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none cursor-pointer"
				>
					{options.map((opt) => (
						<option key={opt.value} value={opt.value}>
							{opt.label}
						</option>
					))}
				</select>
			</div>
			{description && <p className="text-xs text-[var(--text-secondary)]">{description}</p>}
		</div>
	);
}
