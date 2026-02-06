"use client";

import { useWebSocket } from "@/hooks/useWebSocket";
import type { WebSocketMessage } from "@/lib/types";
import { useCampaignStore } from "@/stores/campaignStore";
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
	const addEpisode = useTrainingStore((s) => s.addEpisode);
	const setProgress = useTrainingStore((s) => s.setProgress);
	const clearWarmingUp = useTrainingStore((s) => s.clearWarmingUp);
	const markCompleted = useTrainingStore((s) => s.markCompleted);

	const inferenceSetStatus = useInferenceStore((s) => s.setStatus);
	const inferenceAddStep = useInferenceStore((s) => s.addStep);
	const inferenceAddTrade = useInferenceStore((s) => s.addTrade);
	const inferenceReset = useInferenceStore((s) => s.reset);

	const campaignOnStart = useCampaignStore((s) => s.onCampaignStart);
	const campaignOnTrialStart = useCampaignStore((s) => s.onTrialStart);
	const campaignOnTrialUpdate = useCampaignStore((s) => s.onTrialUpdate);
	const campaignOnTrialPruned = useCampaignStore((s) => s.onTrialPruned);
	const campaignOnTrialComplete = useCampaignStore((s) => s.onTrialComplete);
	const campaignOnImportance = useCampaignStore((s) => s.onImportance);
	const campaignOnProgress = useCampaignStore((s) => s.onProgress);
	const campaignOnEnd = useCampaignStore((s) => s.onCampaignEnd);

	const handleMessage = useCallback(
		(msg: WebSocketMessage) => {
			// Route campaign messages to campaignStore
			switch (msg.type) {
				case "campaign_start":
					campaignOnStart(msg);
					return;
				case "trial_start":
					campaignOnTrialStart(msg);
					return;
				case "trial_update":
					campaignOnTrialUpdate(msg);
					return;
				case "trial_pruned":
					campaignOnTrialPruned(msg);
					return;
				case "trial_complete":
					campaignOnTrialComplete(msg);
					return;
				case "campaign_importance":
					campaignOnImportance(msg);
					return;
				case "campaign_progress":
					campaignOnProgress(msg);
					return;
				case "campaign_end":
					campaignOnEnd(msg);
					return;
			}

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
					clearWarmingUp();
					break;
				case "status":
					setStatus(msg);
					if (msg.is_training) {
						clearWarmingUp();
					}
					break;
				case "training_disconnected":
					// Don't reset — preserve chart data from the completed run
					break;
				case "episode_metrics":
					addEpisode(msg);
					break;
				case "training_progress":
					setProgress(msg);
					clearWarmingUp();
					break;
				case "experiment_start":
					clearWarmingUp();
					break;
				case "experiment_end":
					if ("experiment_id" in msg && "status" in msg) {
						markCompleted(msg.experiment_id as string, msg.status as string);
					}
					break;
			}
		},
		[
			addStep,
			addTrade,
			addIteration,
			setStatus,
			addEpisode,
			setProgress,
			clearWarmingUp,
			markCompleted,
			inferenceSetStatus,
			inferenceAddStep,
			inferenceAddTrade,
			inferenceReset,
			campaignOnStart,
			campaignOnTrialStart,
			campaignOnTrialUpdate,
			campaignOnTrialPruned,
			campaignOnTrialComplete,
			campaignOnImportance,
			campaignOnProgress,
			campaignOnEnd,
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
