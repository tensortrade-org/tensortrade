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
	/** Optional ordered list of group names for optgroup rendering. */
	groups?: string[];
	/** Return the group name for a given option value. */
	optionGroup?: (value: string) => string | undefined;
}

export function ParamSelect({
	label,
	value,
	options,
	onChange,
	description,
	groups,
	optionGroup,
}: ParamSelectProps) {
	const selectId = useId();
	const useGroups = groups && groups.length > 0 && optionGroup;

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
					{useGroups
						? groups.map((group) => (
								<optgroup key={group} label={group}>
									{options
										.filter((opt) => optionGroup(opt.value) === group)
										.map((opt) => (
											<option key={opt.value} value={opt.value}>
												{opt.label}
											</option>
										))}
								</optgroup>
							))
						: options.map((opt) => (
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
