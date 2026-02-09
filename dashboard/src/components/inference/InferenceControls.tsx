"use client";

import { Badge } from "@/components/common/Badge";
import type { InferenceStatusState } from "@/stores/inferenceStore";

interface InferenceControlsProps {
	status: InferenceStatusState;
	onRun: () => void;
	onReset: () => void;
	disabled: boolean;
}

const STATUS_VARIANTS: Record<string, "default" | "success" | "danger" | "warning" | "info"> = {
	idle: "default",
	starting: "warning",
	running: "info",
	completed: "success",
	error: "danger",
};

function Spinner() {
	return (
		<svg
			className="h-4 w-4 animate-spin"
			viewBox="0 0 24 24"
			fill="none"
			aria-label="Loading"
			role="img"
		>
			<circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
			<path
				className="opacity-75"
				fill="currentColor"
				d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
			/>
		</svg>
	);
}

export function InferenceControls({ status, onRun, onReset, disabled }: InferenceControlsProps) {
	const busy = status === "starting" || status === "running";

	return (
		<div className="flex items-center gap-3">
			<Badge label={status} variant={STATUS_VARIANTS[status] ?? "default"} />

			<button
				type="button"
				onClick={onRun}
				disabled={disabled || busy}
				className="inline-flex items-center gap-2 rounded-md bg-[var(--accent-blue)] px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
			>
				{busy && <Spinner />}
				{status === "starting"
					? "Starting..."
					: status === "running"
						? "Running..."
						: "Run Inference"}
			</button>

			<button
				type="button"
				onClick={onReset}
				disabled={busy}
				className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-4 py-1.5 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-50"
			>
				Reset
			</button>
		</div>
	);
}
