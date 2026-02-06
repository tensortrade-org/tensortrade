"use client";

import { Badge } from "@/components/common/Badge";
import type { InferenceStatus } from "@/lib/types";

interface InferenceControlsProps {
	status: InferenceStatus["state"];
	onRun: () => void;
	onReset: () => void;
	disabled: boolean;
}

const STATUS_VARIANTS: Record<string, "default" | "success" | "danger" | "warning" | "info"> = {
	idle: "default",
	running: "info",
	completed: "success",
	error: "danger",
};

export function InferenceControls({ status, onRun, onReset, disabled }: InferenceControlsProps) {
	return (
		<div className="flex items-center gap-3">
			<Badge label={status} variant={STATUS_VARIANTS[status] ?? "default"} />

			<button
				type="button"
				onClick={onRun}
				disabled={disabled || status === "running"}
				className="rounded-md bg-[var(--accent-blue)] px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
			>
				{status === "running" ? "Running..." : "Run Inference"}
			</button>

			<button
				type="button"
				onClick={onReset}
				disabled={status === "running"}
				className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-4 py-1.5 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-50"
			>
				Reset
			</button>
		</div>
	);
}
