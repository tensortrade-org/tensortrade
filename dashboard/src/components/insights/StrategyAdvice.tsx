import { Badge } from "@/components/common/Badge";
import { Card, CardHeader } from "@/components/common/Card";
import type { InsightReport } from "@/lib/types";

interface StrategyAdviceProps {
	insight: InsightReport | null;
}

type Confidence = InsightReport["confidence"];
type BadgeVariant = "success" | "warning" | "danger";

const CONFIDENCE_VARIANT: Record<Confidence, BadgeVariant> = {
	high: "success",
	medium: "warning",
	low: "danger",
};

const CONFIDENCE_LABEL: Record<Confidence, string> = {
	high: "High Confidence",
	medium: "Medium Confidence",
	low: "Low Confidence",
};

export function StrategyAdvice({ insight }: StrategyAdviceProps) {
	if (!insight) {
		return (
			<Card className="border-dashed">
				<div className="flex flex-col items-center gap-2 py-6 text-center">
					<p className="text-sm text-[var(--text-secondary)]">No strategy advice available yet</p>
					<p className="text-xs text-[var(--text-secondary)]">
						Request an analysis to get strategy suggestions
					</p>
				</div>
			</Card>
		);
	}

	return (
		<Card className="border-[var(--accent-purple)]/30 bg-gradient-to-br from-[var(--bg-card)] to-[var(--accent-purple)]/5">
			<CardHeader
				title="Strategy Advice"
				action={
					<Badge
						label={CONFIDENCE_LABEL[insight.confidence]}
						variant={CONFIDENCE_VARIANT[insight.confidence]}
					/>
				}
			/>

			<p className="mb-4 text-base font-medium leading-relaxed text-[var(--text-primary)]">
				{insight.summary}
			</p>

			{insight.suggestions.length > 0 && (
				<div className="flex flex-col gap-3">
					{insight.suggestions.map((suggestion, i) => (
						<div
							key={`strategy-${insight.id}-${i}`}
							className="flex items-start gap-3 rounded-lg border border-[var(--border-color)] bg-[var(--bg-secondary)] p-3"
						>
							<span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[var(--accent-purple)]/20 text-xs font-bold text-[var(--accent-purple)]">
								{i + 1}
							</span>
							<p className="text-sm leading-relaxed text-[var(--text-primary)]">{suggestion}</p>
						</div>
					))}
				</div>
			)}

			{insight.findings.length > 0 && (
				<div className="mt-4 border-t border-[var(--border-color)] pt-4">
					<h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--text-secondary)]">
						Supporting Findings
					</h4>
					<ul className="flex flex-col gap-1">
						{insight.findings.map((finding, i) => (
							<li
								key={`finding-${insight.id}-${i}`}
								className="text-xs leading-relaxed text-[var(--text-secondary)]"
							>
								- {finding}
							</li>
						))}
					</ul>
				</div>
			)}
		</Card>
	);
}
