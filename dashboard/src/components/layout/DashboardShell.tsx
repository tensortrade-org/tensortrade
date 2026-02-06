"use client";

import { useWebSocket } from "@/hooks/useWebSocket";
import type { WebSocketMessage } from "@/lib/types";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useTrainingStore } from "@/stores/trainingStore";
import { useCallback, useEffect } from "react";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";

interface DashboardShellProps {
	children: React.ReactNode;
}

export function DashboardShell({ children }: DashboardShellProps) {
	const addStep = useTrainingStore((s) => s.addStep);
	const addTrade = useTrainingStore((s) => s.addTrade);
	const addIteration = useTrainingStore((s) => s.addIteration);
	const setStatus = useTrainingStore((s) => s.setStatus);
	const setConnected = useTrainingStore((s) => s.setConnected);
	const reset = useTrainingStore((s) => s.reset);
	const addEpisode = useTrainingStore((s) => s.addEpisode);
	const setProgress = useTrainingStore((s) => s.setProgress);

	const inferenceSetStatus = useInferenceStore((s) => s.setStatus);
	const inferenceAddStep = useInferenceStore((s) => s.addStep);
	const inferenceAddTrade = useInferenceStore((s) => s.addTrade);
	const inferenceReset = useInferenceStore((s) => s.reset);

	const handleMessage = useCallback(
		(msg: WebSocketMessage) => {
			// Route inference messages to inferenceStore
			if ("source" in msg && msg.source === "inference") {
				switch (msg.type) {
					case "step_update":
						inferenceAddStep(msg);
						break;
					case "trade":
						inferenceAddTrade(msg);
						break;
					case "inference_status":
						inferenceSetStatus(msg);
						if (msg.state === "running") {
							inferenceReset();
							inferenceSetStatus(msg);
						}
						break;
				}
				return;
			}

			// Route inference_status even without source field
			if (msg.type === "inference_status") {
				inferenceSetStatus(msg);
				return;
			}

			// Default: route to trainingStore
			switch (msg.type) {
				case "step_update":
					addStep(msg);
					break;
				case "trade":
					addTrade(msg);
					break;
				case "training_update":
					addIteration(msg);
					break;
				case "status":
					setStatus(msg);
					break;
				case "training_disconnected":
					reset();
					break;
				case "episode_metrics":
					addEpisode(msg);
					break;
				case "training_progress":
					setProgress(msg);
					break;
			}
		},
		[
			addStep,
			addTrade,
			addIteration,
			setStatus,
			reset,
			addEpisode,
			setProgress,
			inferenceSetStatus,
			inferenceAddStep,
			inferenceAddTrade,
			inferenceReset,
		],
	);

	const wsUrl = typeof window !== "undefined" ? `ws://${window.location.host}/ws/dashboard` : "";

	const { connected } = useWebSocket({
		url: wsUrl,
		onMessage: handleMessage,
	});

	useEffect(() => {
		setConnected(connected);
	}, [connected, setConnected]);

	return (
		<div className="flex h-screen overflow-hidden">
			<Sidebar />
			<div className="flex flex-1 flex-col overflow-hidden">
				<Header />
				<main className="flex-1 overflow-y-auto bg-[var(--bg-primary)] p-6">{children}</main>
			</div>
		</div>
	);
}
