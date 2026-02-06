"use client";

import { useLaunchStore } from "@/stores/launchStore";
import { useCallback, useState } from "react";

export function NameStep() {
	const name = useLaunchStore((s) => s.name);
	const setName = useLaunchStore((s) => s.setName);
	const tags = useLaunchStore((s) => s.tags);
	const setTags = useLaunchStore((s) => s.setTags);

	const [tagInput, setTagInput] = useState("");

	const handleAddTag = useCallback(() => {
		const trimmed = tagInput.trim();
		if (trimmed.length > 0 && !tags.includes(trimmed)) {
			setTags([...tags, trimmed]);
		}
		setTagInput("");
	}, [tagInput, tags, setTags]);

	const handleTagKeyDown = useCallback(
		(e: React.KeyboardEvent<HTMLInputElement>) => {
			if (e.key === "Enter" || e.key === ",") {
				e.preventDefault();
				handleAddTag();
			}
			if (e.key === "Backspace" && tagInput === "" && tags.length > 0) {
				setTags(tags.slice(0, -1));
			}
		},
		[handleAddTag, tagInput, tags, setTags],
	);

	const handleRemoveTag = useCallback(
		(tag: string) => {
			setTags(tags.filter((t) => t !== tag));
		},
		[tags, setTags],
	);

	return (
		<div className="space-y-6">
			<div>
				<label
					htmlFor="experiment-name"
					className="mb-2 block text-sm font-medium text-[var(--text-primary)]"
				>
					Experiment Name
				</label>
				<input
					id="experiment-name"
					type="text"
					value={name}
					onChange={(e) => setName(e.target.value)}
					placeholder="e.g., PPO BTC 1h Aggressive"
					className="w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)]/50 focus:border-[var(--accent-blue)] focus:outline-none"
				/>
				{name.trim().length === 0 && (
					<p className="mt-1 text-xs text-[var(--text-secondary)]">A name is required to proceed</p>
				)}
			</div>

			<div>
				<label
					htmlFor="tag-input"
					className="mb-2 block text-sm font-medium text-[var(--text-primary)]"
				>
					Tags
				</label>
				<div className="flex min-h-[42px] flex-wrap items-center gap-2 rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2">
					{tags.map((tag) => (
						<span
							key={tag}
							className="inline-flex items-center gap-1 rounded-full bg-[var(--accent-blue)]/15 px-2.5 py-0.5 text-xs font-medium text-[var(--accent-blue)]"
						>
							{tag}
							<button
								type="button"
								onClick={() => handleRemoveTag(tag)}
								className="ml-0.5 text-[var(--accent-blue)]/70 hover:text-[var(--accent-blue)]"
							>
								x
							</button>
						</span>
					))}
					<input
						id="tag-input"
						type="text"
						value={tagInput}
						onChange={(e) => setTagInput(e.target.value)}
						onKeyDown={handleTagKeyDown}
						onBlur={handleAddTag}
						placeholder={tags.length === 0 ? "Type a tag and press Enter..." : ""}
						className="min-w-[120px] flex-1 bg-transparent text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)]/50 focus:outline-none"
					/>
				</div>
				<p className="mt-1 text-xs text-[var(--text-secondary)]">
					Press Enter or comma to add a tag
				</p>
			</div>
		</div>
	);
}
