"use client";

import { Badge } from "@/components/common/Badge";
import { Card, CardHeader } from "@/components/common/Card";
import { generateHpPack, updateHyperparamPack } from "@/lib/api";
import { formatDate } from "@/lib/formatters";
import type { HyperparameterPack, InsightReport } from "@/lib/types";
import Link from "next/link";
import { useCallback, useState } from "react";

interface InsightCardProps {
	insight: InsightReport;
	experimentId?: string;
}

type AnalysisType = InsightReport["analysis_type"];
type Confidence = InsightReport["confidence"];
type BadgeVariant = "info" | "purple" | "success" | "warning" | "danger" | "default";

type GenerateState = "idle" | "form" | "generating" | "created" | "error";

const ANALYSIS_TYPE_VARIANT: Record<AnalysisType, BadgeVariant> = {
	experiment: "info",
	comparison: "purple",
	strategy: "success",
	trades: "warning",
	campaign_analysis: "info",
};

const CONFIDENCE_VARIANT: Record<Confidence, BadgeVariant> = {
	high: "success",
	medium: "warning",
	low: "danger",
};

export function InsightCard({ insight, experimentId }: InsightCardProps) {
	const [generateState, setGenerateState] = useState<GenerateState>("idle");
	const [guidance, setGuidance] = useState("");
	const [createdPack, setCreatedPack] = useState<HyperparameterPack | null>(null);
	const [packName, setPackName] = useState("");
	const [nameChanged, setNameChanged] = useState(false);
	const [errorMessage, setErrorMessage] = useState("");

	const handleOpenForm = useCallback(() => {
		setGuidance("");
		setErrorMessage("");
		setGenerateState("form");
	}, []);

	const handleGenerate = useCallback(async () => {
		if (!experimentId) return;
		setGenerateState("generating");
		setErrorMessage("");
		try {
			const pack = await generateHpPack(experimentId, insight.id, guidance.trim() || undefined);
			if ("error" in pack) {
				setErrorMessage((pack as unknown as { error: string }).error);
				setGenerateState("error");
				return;
			}
			setCreatedPack(pack);
			setPackName(pack.name);
			setNameChanged(false);
			setGenerateState("created");
		} catch (err) {
			setErrorMessage(err instanceof Error ? err.message : "Failed to generate pack");
			setGenerateState("error");
		}
	}, [experimentId, insight.id, guidance]);

	const handleCancel = useCallback(() => {
		setGenerateState("idle");
		setErrorMessage("");
	}, []);

	const handleRename = useCallback(async () => {
		if (!createdPack || !packName.trim()) return;
		try {
			await updateHyperparamPack(createdPack.id, { name: packName.trim() });
			setCreatedPack({ ...createdPack, name: packName.trim() });
			setNameChanged(false);
		} catch (err) {
			setErrorMessage(err instanceof Error ? err.message : "Failed to rename pack");
		}
	}, [createdPack, packName]);

	return (
		<Card>
			<CardHeader
				title={formatDate(insight.created_at)}
				action={
					<div className="flex items-center gap-2">
						<Badge
							label={insight.analysis_type}
							variant={ANALYSIS_TYPE_VARIANT[insight.analysis_type]}
						/>
						<Badge label={insight.confidence} variant={CONFIDENCE_VARIANT[insight.confidence]} />
					</div>
				}
			/>

			<p className="mb-4 text-sm leading-relaxed text-[var(--text-primary)]">{insight.summary}</p>

			{(insight.findings?.length ?? 0) > 0 && (
				<div className="mb-4">
					<h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--text-secondary)]">
						Findings
					</h4>
					<ul className="flex flex-col gap-1.5">
						{insight.findings.map((finding, i) => (
							<li
								key={`finding-${insight.id}-${i}`}
								className="flex items-start gap-2 text-sm text-[var(--text-primary)]"
							>
								<span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--accent-blue)]" />
								{finding}
							</li>
						))}
					</ul>
				</div>
			)}

			{(insight.suggestions?.length ?? 0) > 0 && (
				<div>
					<h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--text-secondary)]">
						Suggestions
					</h4>
					<ul className="flex flex-col gap-1.5">
						{insight.suggestions.map((suggestion, i) => (
							<li
								key={`suggestion-${insight.id}-${i}`}
								className="flex items-start gap-2 text-sm text-[var(--accent-green)]"
							>
								<span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--accent-green)]" />
								{suggestion}
							</li>
						))}
					</ul>
				</div>
			)}

			{/* Generate HP Pack Section */}
			{experimentId && (insight.suggestions?.length ?? 0) > 0 && (
				<div className="mt-4 border-t border-[var(--border-primary)] pt-4">
					{generateState === "idle" && (
						<button
							type="button"
							onClick={handleOpenForm}
							className="rounded-md bg-[var(--accent-green)] px-4 py-2 text-sm font-medium text-white hover:opacity-90"
						>
							Generate Improved HP Pack
						</button>
					)}

					{generateState === "form" && (
						<div className="space-y-3">
							<div>
								<label
									htmlFor={`guidance-${insight.id}`}
									className="mb-1 block text-xs font-medium text-[var(--text-secondary)]"
								>
									Additional guidance (optional)
								</label>
								<textarea
									id={`guidance-${insight.id}`}
									value={guidance}
									onChange={(e) => setGuidance(e.target.value)}
									placeholder="e.g. Focus on reducing drawdown, keep entropy high..."
									className="w-full rounded-md border border-[var(--border-primary)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)]"
									rows={2}
								/>
							</div>
							<div className="flex items-center gap-2">
								<button
									type="button"
									onClick={handleGenerate}
									className="rounded-md bg-[var(--accent-green)] px-4 py-2 text-sm font-medium text-white hover:opacity-90"
								>
									Generate
								</button>
								<button
									type="button"
									onClick={handleCancel}
									className="rounded-md border border-[var(--border-primary)] px-4 py-2 text-sm font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-secondary)]"
								>
									Cancel
								</button>
							</div>
						</div>
					)}

					{generateState === "generating" && (
						<div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
							<svg
								className="h-4 w-4 animate-spin"
								viewBox="0 0 24 24"
								fill="none"
								aria-hidden="true"
							>
								<circle
									className="opacity-25"
									cx="12"
									cy="12"
									r="10"
									stroke="currentColor"
									strokeWidth="4"
								/>
								<path
									className="opacity-75"
									fill="currentColor"
									d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
								/>
							</svg>
							Generating improved HP pack...
						</div>
					)}

					{generateState === "created" && createdPack && (
						<div className="space-y-2 rounded-md bg-[var(--accent-green)]/10 px-3 py-2">
							<div className="flex items-center gap-2">
								<label
									htmlFor={`pack-name-${insight.id}`}
									className="shrink-0 text-sm text-[var(--accent-green)]"
								>
									Created:
								</label>
								<input
									id={`pack-name-${insight.id}`}
									type="text"
									value={packName}
									onChange={(e) => {
										setPackName(e.target.value);
										setNameChanged(e.target.value.trim() !== createdPack.name);
									}}
									className="flex-1 rounded border border-[var(--border-primary)] bg-[var(--bg-secondary)] px-2 py-1 text-sm font-medium text-[var(--text-primary)]"
								/>
								{nameChanged && (
									<button
										type="button"
										onClick={handleRename}
										className="shrink-0 rounded bg-[var(--accent-blue)] px-3 py-1 text-xs font-medium text-white hover:opacity-90"
									>
										Save Name
									</button>
								)}
							</div>
							<Link
								href="/hyperparams"
								className="inline-block text-sm font-medium text-[var(--accent-blue)] hover:underline"
							>
								View in HP Studio
							</Link>
						</div>
					)}

					{generateState === "error" && (
						<div className="space-y-2">
							<div className="rounded-md bg-[var(--accent-red)]/10 px-3 py-2 text-sm text-[var(--accent-red)]">
								{errorMessage}
							</div>
							<button
								type="button"
								onClick={handleOpenForm}
								className="text-sm font-medium text-[var(--accent-blue)] hover:underline"
							>
								Try again
							</button>
						</div>
					)}
				</div>
			)}
		</Card>
	);
}
