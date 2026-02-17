"use client";

import { useId } from "react";

interface ParamSliderProps {
	label: string;
	value: number;
	min: number;
	max: number;
	step: number;
	onChange: (value: number) => void;
	description?: string;
	format?: (v: number) => string;
}

export function ParamSlider({
	label,
	value,
	min,
	max,
	step,
	onChange,
	description,
	format,
}: ParamSliderProps) {
	const sliderId = useId();
	const displayValue = format ? format(value) : String(value);

	return (
		<div className="flex flex-col gap-1 py-2">
			<div className="flex items-center justify-between gap-4">
				<label htmlFor={sliderId} className="text-sm text-[var(--text-primary)] shrink-0">
					{label}
				</label>
				<div className="flex items-center gap-3">
					<span className="text-xs font-mono text-[var(--text-secondary)] min-w-[60px] text-right">
						{displayValue}
					</span>
					<input
						id={sliderId}
						type="range"
						min={min}
						max={max}
						step={step}
						value={value}
						onChange={(e) => onChange(Number(e.target.value))}
						className="w-40 h-1.5 rounded-full appearance-none cursor-pointer
							bg-[var(--border-color)]
							[&::-webkit-slider-thumb]:appearance-none
							[&::-webkit-slider-thumb]:w-3.5
							[&::-webkit-slider-thumb]:h-3.5
							[&::-webkit-slider-thumb]:rounded-full
							[&::-webkit-slider-thumb]:bg-[var(--accent-blue)]
							[&::-webkit-slider-thumb]:cursor-pointer"
					/>
					<input
						type="number"
						min={min}
						max={max}
						step={step}
						value={value}
						onChange={(e) => {
							const v = Number(e.target.value);
							if (!Number.isNaN(v)) onChange(v);
						}}
						aria-label={`${label} numeric input`}
						className="w-20 rounded border border-[var(--border-color)] bg-[var(--bg-card)] px-2 py-1 text-right text-xs font-mono text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none"
					/>
				</div>
			</div>
			{description && <p className="text-xs text-[var(--text-secondary)] pl-0">{description}</p>}
		</div>
	);
}
