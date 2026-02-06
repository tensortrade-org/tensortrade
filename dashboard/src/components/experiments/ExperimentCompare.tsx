"use client";

import { Badge } from "@/components/common/Badge";
import { Card, CardHeader } from "@/components/common/Card";
import { formatCurrency, formatNumber, formatPercent, formatPnl } from "@/lib/formatters";
import type { ExperimentDetail } from "@/lib/types";
import { useMemo } from "react";

interface ExperimentCompareProps {
	experiments: ExperimentDetail[];
}

interface ConfigDiffRow {
	key: string;
	values: (string | undefined)[];
	isDifferent: boolean;
}

interface MetricComparisonRow {
	key: string;
	values: (number | undefined)[];
	bestIndex: number | null;
}

function formatConfigValue(value: unknown): string {
	if (value === null || value === undefined) return "--";
	if (typeof value === "object") return JSON.stringify(value);
	return String(value);
}

function formatMetricValue(key: string, value: number | undefined): string {
	if (value === undefined) return "--";
	if (key === "pnl") return formatPnl(value);
	if (key === "net_worth") return formatCurrency(value);
	if (key.includes("ratio") || key.includes("return") || key.includes("drawdown")) {
		return formatPercent(value);
	}
	return formatNumber(value, 2);
}

function computeConfigDiff(experiments: ExperimentDetail[]): ConfigDiffRow[] {
	const allKeys = new Set<string>();
	for (const exp of experiments) {
		for (const key of Object.keys(exp.experiment.config)) {
			allKeys.add(key);
		}
	}

	const rows: ConfigDiffRow[] = [];
	for (const key of Array.from(allKeys).sort()) {
		const values = experiments.map((exp) => formatConfigValue(exp.experiment.config[key]));
		const isDifferent = values.some((v) => v !== values[0]);
		rows.push({ key, values, isDifferent });
	}
	return rows;
}

function computeMetricComparison(experiments: ExperimentDetail[]): MetricComparisonRow[] {
	const allKeys = new Set<string>();
	for (const exp of experiments) {
		for (const key of Object.keys(exp.experiment.final_metrics)) {
			allKeys.add(key);
		}
	}

	const rows: MetricComparisonRow[] = [];
	for (const key of Array.from(allKeys).sort()) {
		const values = experiments.map((exp) => exp.experiment.final_metrics[key]);
		const defined = values
			.map((v, i) => ({ v, i }))
			.filter((entry): entry is { v: number; i: number } => entry.v !== undefined);

		let bestIndex: number | null = null;
		if (defined.length > 0) {
			const isLowerBetter = key === "max_drawdown";
			const best = defined.reduce((prev, curr) =>
				isLowerBetter ? (curr.v < prev.v ? curr : prev) : curr.v > prev.v ? curr : prev,
			);
			bestIndex = best.i;
		}

		rows.push({ key, values, bestIndex });
	}
	return rows;
}

interface MiniIterationChartProps {
	iterations: ExperimentDetail["iterations"];
	metricKey: string;
	color: string;
}

function MiniIterationChart({ iterations, metricKey, color }: MiniIterationChartProps) {
	const points = iterations
		.map((iter) => iter.metrics[metricKey])
		.filter((v): v is number => v !== undefined);

	if (points.length === 0) {
		return (
			<div className="flex h-16 items-center justify-center text-xs text-[var(--text-secondary)]">
				No data
			</div>
		);
	}

	const min = Math.min(...points);
	const max = Math.max(...points);
	const range = max - min || 1;
	const width = 200;
	const height = 64;
	const stepX = width / Math.max(points.length - 1, 1);

	const pathData = points
		.map((v, i) => {
			const x = i * stepX;
			const y = height - ((v - min) / range) * (height - 8) - 4;
			return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
		})
		.join(" ");

	return (
		<svg
			viewBox={`0 0 ${width} ${height}`}
			className="h-16 w-full"
			preserveAspectRatio="none"
			role="img"
			aria-label="Metric sparkline"
		>
			<path d={pathData} fill="none" stroke={color} strokeWidth="2" />
		</svg>
	);
}

const CHART_COLORS = [
	"var(--accent-blue)",
	"var(--accent-purple)",
	"var(--accent-green)",
	"var(--accent-amber)",
	"var(--accent-red)",
];

export function ExperimentCompare({ experiments }: ExperimentCompareProps) {
	const configDiff = useMemo(() => computeConfigDiff(experiments), [experiments]);
	const metricComparison = useMemo(() => computeMetricComparison(experiments), [experiments]);

	if (experiments.length === 0) {
		return (
			<Card>
				<p className="text-center text-[var(--text-secondary)]">Select experiments to compare</p>
			</Card>
		);
	}

	return (
		<div className="flex flex-col gap-6">
			{/* Config Diff */}
			<Card>
				<CardHeader title="Configuration Diff" />
				<div className="overflow-x-auto">
					<table className="w-full text-sm">
						<thead>
							<tr className="border-b border-[var(--border-color)]">
								<th className="px-3 py-2 text-left font-medium text-[var(--text-secondary)]">
									Parameter
								</th>
								{experiments.map((exp) => (
									<th
										key={exp.experiment.id}
										className="px-3 py-2 text-left font-medium text-[var(--text-secondary)]"
									>
										{exp.experiment.name}
									</th>
								))}
							</tr>
						</thead>
						<tbody>
							{configDiff.map((row) => (
								<tr
									key={row.key}
									className={`border-b border-[var(--border-color)]/50 ${
										row.isDifferent ? "bg-[var(--accent-amber)]/5" : ""
									}`}
								>
									<td className="px-3 py-2 font-mono text-xs text-[var(--text-secondary)]">
										{row.key}
										{row.isDifferent && <Badge label="diff" variant="warning" />}
									</td>
									{row.values.map((value, i) => (
										<td
											key={experiments[i].experiment.id}
											className={`px-3 py-2 font-mono text-xs ${
												row.isDifferent
													? "text-[var(--text-primary)] font-semibold"
													: "text-[var(--text-secondary)]"
											}`}
										>
											{value ?? "--"}
										</td>
									))}
								</tr>
							))}
							{configDiff.length === 0 && (
								<tr>
									<td
										colSpan={experiments.length + 1}
										className="px-3 py-6 text-center text-[var(--text-secondary)]"
									>
										No configuration data available
									</td>
								</tr>
							)}
						</tbody>
					</table>
				</div>
			</Card>

			{/* Metric Comparison */}
			<Card>
				<CardHeader title="Metric Comparison" />
				<div className="overflow-x-auto">
					<table className="w-full text-sm">
						<thead>
							<tr className="border-b border-[var(--border-color)]">
								<th className="px-3 py-2 text-left font-medium text-[var(--text-secondary)]">
									Metric
								</th>
								{experiments.map((exp) => (
									<th
										key={exp.experiment.id}
										className="px-3 py-2 text-right font-medium text-[var(--text-secondary)]"
									>
										{exp.experiment.name}
									</th>
								))}
							</tr>
						</thead>
						<tbody>
							{metricComparison.map((row) => (
								<tr key={row.key} className="border-b border-[var(--border-color)]/50">
									<td className="px-3 py-2 font-mono text-xs text-[var(--text-secondary)]">
										{row.key}
									</td>
									{row.values.map((value, i) => (
										<td
											key={experiments[i].experiment.id}
											className={`px-3 py-2 text-right font-mono text-xs ${
												row.bestIndex === i
													? "text-[var(--accent-green)] font-semibold"
													: "text-[var(--text-primary)]"
											}`}
										>
											{formatMetricValue(row.key, value)}
										</td>
									))}
								</tr>
							))}
							{metricComparison.length === 0 && (
								<tr>
									<td
										colSpan={experiments.length + 1}
										className="px-3 py-6 text-center text-[var(--text-secondary)]"
									>
										No metric data available
									</td>
								</tr>
							)}
						</tbody>
					</table>
				</div>
			</Card>

			{/* Mini Charts */}
			<Card>
				<CardHeader title="Iteration Trends" />
				<div className="grid gap-4 sm:grid-cols-2">
					{["pnl_mean", "net_worth_mean", "episode_return_mean"].map((metricKey) => (
						<div key={metricKey} className="rounded border border-[var(--border-color)] p-3">
							<p className="mb-2 text-xs font-medium text-[var(--text-secondary)]">
								{metricKey.replace(/_/g, " ")}
							</p>
							<div className="flex flex-col gap-2">
								{experiments.map((exp, i) => (
									<div key={exp.experiment.id}>
										<p
											className="mb-1 text-xs"
											style={{ color: CHART_COLORS[i % CHART_COLORS.length] }}
										>
											{exp.experiment.name}
										</p>
										<MiniIterationChart
											iterations={exp.iterations}
											metricKey={metricKey}
											color={CHART_COLORS[i % CHART_COLORS.length]}
										/>
									</div>
								))}
							</div>
						</div>
					))}
				</div>
			</Card>
		</div>
	);
}
