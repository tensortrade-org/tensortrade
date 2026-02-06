"use client";

import { pauseTraining, resumeTraining, stopTraining } from "@/lib/api";
import { useTrainingStore } from "@/stores/trainingStore";
import { useState } from "react";

interface ActionState {
	loading: boolean;
	error: string | null;
}

export function TrainingControls() {
	const status = useTrainingStore((state) => state.status);
	const [actionState, setActionState] = useState<ActionState>({
		loading: false,
		error: null,
	});

	const isTraining = status?.is_training ?? false;
	const isPaused = status?.is_paused ?? false;

	const handleAction = async (action: () => Promise<{ status: string; message: string }>) => {
		setActionState({ loading: true, error: null });
		try {
			await action();
			setActionState({ loading: false, error: null });
		} catch (err) {
			const message = err instanceof Error ? err.message : "Action failed";
			setActionState({ loading: false, error: message });
		}
	};

	const handleStop = () => handleAction(stopTraining);
	const handlePause = () => handleAction(pauseTraining);
	const handleResume = () => handleAction(resumeTraining);

	if (!status) {
		return (
			<div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
				No training status available
			</div>
		);
	}

	return (
		<div className="flex flex-col gap-2">
			<div className="flex items-center gap-2">
				{isTraining && !isPaused && (
					<>
						<button
							type="button"
							onClick={handlePause}
							disabled={actionState.loading}
							className="rounded-md bg-[var(--accent-amber)]/15 px-3 py-1.5 text-xs font-medium text-[var(--accent-amber)] transition-colors hover:bg-[var(--accent-amber)]/25 disabled:cursor-not-allowed disabled:opacity-50"
						>
							{actionState.loading ? "Pausing..." : "Pause"}
						</button>
						<button
							type="button"
							onClick={handleStop}
							disabled={actionState.loading}
							className="rounded-md bg-[var(--accent-red)]/15 px-3 py-1.5 text-xs font-medium text-[var(--accent-red)] transition-colors hover:bg-[var(--accent-red)]/25 disabled:cursor-not-allowed disabled:opacity-50"
						>
							{actionState.loading ? "Stopping..." : "Stop"}
						</button>
					</>
				)}

				{isTraining && isPaused && (
					<>
						<button
							type="button"
							onClick={handleResume}
							disabled={actionState.loading}
							className="rounded-md bg-[var(--accent-green)]/15 px-3 py-1.5 text-xs font-medium text-[var(--accent-green)] transition-colors hover:bg-[var(--accent-green)]/25 disabled:cursor-not-allowed disabled:opacity-50"
						>
							{actionState.loading ? "Resuming..." : "Resume"}
						</button>
						<button
							type="button"
							onClick={handleStop}
							disabled={actionState.loading}
							className="rounded-md bg-[var(--accent-red)]/15 px-3 py-1.5 text-xs font-medium text-[var(--accent-red)] transition-colors hover:bg-[var(--accent-red)]/25 disabled:cursor-not-allowed disabled:opacity-50"
						>
							{actionState.loading ? "Stopping..." : "Stop"}
						</button>
					</>
				)}

				{!isTraining && (
					<span className="text-xs text-[var(--text-secondary)]">No active training session</span>
				)}
			</div>

			{actionState.error && <p className="text-xs text-[var(--accent-red)]">{actionState.error}</p>}
		</div>
	);
}
