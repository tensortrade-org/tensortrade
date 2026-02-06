import { Badge } from "@/components/common/Badge";
import { Card, CardHeader } from "@/components/common/Card";
import { formatDate } from "@/lib/formatters";
import type { InsightReport } from "@/lib/types";

interface InsightCardProps {
	insight: InsightReport;
}

type AnalysisType = InsightReport["analysis_type"];
type Confidence = InsightReport["confidence"];
type BadgeVariant = "info" | "purple" | "success" | "warning" | "danger" | "default";

const ANALYSIS_TYPE_VARIANT: Record<AnalysisType, BadgeVariant> = {
	experiment: "info",
	comparison: "purple",
	strategy: "success",
	trades: "warning",
};

const CONFIDENCE_VARIANT: Record<Confidence, BadgeVariant> = {
	high: "success",
	medium: "warning",
	low: "danger",
};

export function InsightCard({ insight }: InsightCardProps) {
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

			{insight.findings.length > 0 && (
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

			{insight.suggestions.length > 0 && (
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
		</Card>
	);
}
