"use client";

import { useTrainingStore } from "@/stores/trainingStore";

export function Header() {
	const status = useTrainingStore((s) => s.status);
	const isConnected = useTrainingStore((s) => s.isConnected);

	return (
		<header className="flex h-14 items-center justify-between border-b border-[var(--border-color)] bg-[var(--bg-secondary)] px-6">
			<div className="flex items-center gap-4">
				{status?.is_training && (
					<div className="flex items-center gap-2">
						<div className="h-2 w-2 animate-pulse rounded-full bg-[var(--accent-green)]" />
						<span className="text-sm text-[var(--text-primary)]">
							Training
							{status.is_paused ? " (paused)" : ""}
							{status.experiment_id ? ` \u2014 iteration ${status.current_iteration}` : ""}
						</span>
					</div>
				)}
			</div>
			<div className="flex items-center gap-3">
				<div className="flex items-center gap-1.5">
					<div
						className={`h-2 w-2 rounded-full ${
							isConnected ? "bg-[var(--accent-green)]" : "bg-[var(--text-secondary)]"
						}`}
					/>
					<span className="text-xs text-[var(--text-secondary)]">
						{isConnected ? "Connected" : "Disconnected"}
					</span>
				</div>
				{status && (
					<span className="text-xs text-[var(--text-secondary)]">
						{status.dashboard_clients} client{status.dashboard_clients !== 1 ? "s" : ""}
					</span>
				)}
			</div>
		</header>
	);
}
