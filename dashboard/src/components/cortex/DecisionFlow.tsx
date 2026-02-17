"use client";

import type {
	AgentBehavior,
	FlowLink,
	FlowNode,
	MarketRegime,
	TradeOutcome,
} from "@/lib/cortex-types";
import type { TrainingUpdate } from "@/lib/types";
import { useEffect, useMemo, useRef, useState } from "react";

const PADDING = { top: 30, right: 10, bottom: 10, left: 10 };
const NODE_WIDTH = 18;
const NODE_GAP = 8;
const WINDOW_SIZE = 20;

const REGIME_COLORS: Record<MarketRegime, string> = {
	Improving: "#22c55e",
	Declining: "#ef4444",
	Sideways: "#f59e0b",
};

const BEHAVIOR_COLORS: Record<AgentBehavior, string> = {
	Aggressive: "#ef4444",
	Balanced: "#3b82f6",
	Conservative: "#a855f7",
};

const OUTCOME_COLORS: Record<TradeOutcome, string> = {
	Profitable: "#22c55e",
	Breakeven: "#f59e0b",
	Losing: "#ef4444",
};

function classifyRegime(iterations: TrainingUpdate[]): MarketRegime {
	if (iterations.length < 2) return "Sideways";
	const half = Math.floor(iterations.length / 2);
	const first = iterations.slice(0, half);
	const second = iterations.slice(half);
	const avgFirst = first.reduce((s, it) => s + it.pnl_mean, 0) / (first.length || 1);
	const avgSecond = second.reduce((s, it) => s + it.pnl_mean, 0) / (second.length || 1);
	const diff = avgSecond - avgFirst;
	if (diff > 5) return "Improving";
	if (diff < -5) return "Declining";
	return "Sideways";
}

function classifyBehavior(it: TrainingUpdate): AgentBehavior {
	const tc = it.trade_count_mean ?? 0;
	const hc = it.hold_count_mean ?? 0;
	const total = tc + hc;
	const ratio = it.trade_ratio_mean ?? (total > 0 ? tc / total : 0);
	if (ratio > 0.3) return "Aggressive";
	if (ratio < 0.05) return "Conservative";
	return "Balanced";
}

function classifyOutcome(it: TrainingUpdate): TradeOutcome {
	if (it.pnl_mean > 5) return "Profitable";
	if (it.pnl_mean < -5) return "Losing";
	return "Breakeven";
}

interface IterationClassification {
	regime: MarketRegime;
	behavior: AgentBehavior;
	outcome: TradeOutcome;
}

function buildFlowState(
	classifications: IterationClassification[],
	plotWidth: number,
	plotHeight: number,
): { nodes: FlowNode[]; links: FlowLink[] } {
	if (classifications.length === 0) return { nodes: [], links: [] };

	const total = classifications.length;
	const colPositions = [
		PADDING.left,
		PADDING.left + plotWidth / 2 - NODE_WIDTH / 2,
		PADDING.left + plotWidth - NODE_WIDTH,
	];

	// Count each category
	const regimeCounts: Record<MarketRegime, number> = { Improving: 0, Declining: 0, Sideways: 0 };
	const behaviorCounts: Record<AgentBehavior, number> = {
		Aggressive: 0,
		Balanced: 0,
		Conservative: 0,
	};
	const outcomeCounts: Record<TradeOutcome, number> = { Profitable: 0, Breakeven: 0, Losing: 0 };

	// Count links
	const linkMap: Record<string, number> = {};

	for (const c of classifications) {
		regimeCounts[c.regime]++;
		behaviorCounts[c.behavior]++;
		outcomeCounts[c.outcome]++;

		const rb = `${c.regime}->${c.behavior}`;
		const bo = `${c.behavior}->${c.outcome}`;
		linkMap[rb] = (linkMap[rb] ?? 0) + 1;
		linkMap[bo] = (linkMap[bo] ?? 0) + 1;
	}

	function buildColumnNodes(
		counts: Record<string, number>,
		colors: Record<string, string>,
		colIdx: number,
	): FlowNode[] {
		const entries = Object.entries(counts).filter(([, v]) => v > 0);
		const totalInCol = entries.reduce((s, [, v]) => s + v, 0);
		const availH = plotHeight - NODE_GAP * (entries.length - 1);
		let y = PADDING.top;
		return entries.map(([label, value]) => {
			const height = Math.max(4, (value / totalInCol) * availH);
			const node: FlowNode = {
				id: label,
				label,
				color: colors[label] ?? "#666",
				column: colIdx,
				value,
				y,
				height,
			};
			y += height + NODE_GAP;
			return node;
		});
	}

	const regimeNodes = buildColumnNodes(regimeCounts, REGIME_COLORS, 0);
	const behaviorNodes = buildColumnNodes(behaviorCounts, BEHAVIOR_COLORS, 1);
	const outcomeNodes = buildColumnNodes(outcomeCounts, OUTCOME_COLORS, 2);

	const allNodes = [...regimeNodes, ...behaviorNodes, ...outcomeNodes];
	const nodeMap = new Map(allNodes.map((n) => [n.id, n]));

	// Build links: track consumed offsets per node side
	const sourceOffsets: Record<string, number> = {};
	const targetOffsets: Record<string, number> = {};

	const links: FlowLink[] = [];

	for (const [key, value] of Object.entries(linkMap)) {
		const [srcId, tgtId] = key.split("->");
		const src = nodeMap.get(srcId);
		const tgt = nodeMap.get(tgtId);
		if (!src || !tgt) continue;

		const srcTotal = src.value;
		const tgtTotal = tgt.value;
		const linkHeightSrc = (value / srcTotal) * src.height;
		const linkHeightTgt = (value / tgtTotal) * tgt.height;

		const srcOff = sourceOffsets[srcId] ?? 0;
		const tgtOff = targetOffsets[tgtId] ?? 0;

		const x0 = colPositions[src.column] + NODE_WIDTH;
		const y0 = src.y + srcOff + linkHeightSrc / 2;
		const x1 = colPositions[tgt.column];
		const y1 = tgt.y + tgtOff + linkHeightTgt / 2;

		sourceOffsets[srcId] = srcOff + linkHeightSrc;
		targetOffsets[tgtId] = tgtOff + linkHeightTgt;

		links.push({
			source: srcId,
			target: tgtId,
			value,
			color: src.color,
		});
	}

	return { nodes: allNodes.map((n) => ({ ...n, y: n.y })), links };
}

interface DecisionFlowProps {
	iterations: TrainingUpdate[];
}

export function DecisionFlow({ iterations }: DecisionFlowProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const [dims, setDims] = useState({ width: 400, height: 400 });

	useEffect(() => {
		if (!containerRef.current) return;
		const update = () => {
			if (containerRef.current) {
				setDims({
					width: containerRef.current.clientWidth,
					height: containerRef.current.clientHeight,
				});
			}
		};
		update();
		const ro = new ResizeObserver(update);
		ro.observe(containerRef.current);
		return () => ro.disconnect();
	}, []);

	const recent = useMemo(() => iterations.slice(-WINDOW_SIZE), [iterations]);

	const classifications = useMemo((): IterationClassification[] => {
		const regime = classifyRegime(recent);
		return recent.map((it) => ({
			regime,
			behavior: classifyBehavior(it),
			outcome: classifyOutcome(it),
		}));
	}, [recent]);

	const plotWidth = dims.width - PADDING.left - PADDING.right;
	const plotHeight = dims.height - PADDING.top - PADDING.bottom;
	const colPositions = [
		PADDING.left,
		PADDING.left + plotWidth / 2 - NODE_WIDTH / 2,
		PADDING.left + plotWidth - NODE_WIDTH,
	];

	const { nodes, links } = useMemo(
		() => buildFlowState(classifications, plotWidth, plotHeight),
		[classifications, plotWidth, plotHeight],
	);

	// Build link paths with offsets
	const linkPaths = useMemo(() => {
		if (nodes.length === 0) return [];
		const nodeMap = new Map(nodes.map((n) => [n.id, n]));
		const sourceOffsets: Record<string, number> = {};
		const targetOffsets: Record<string, number> = {};

		return links
			.map((link) => {
				const src = nodeMap.get(link.source);
				const tgt = nodeMap.get(link.target);
				if (!src || !tgt) return null;

				const srcTotal = src.value;
				const tgtTotal = tgt.value;
				const linkHeightSrc = (link.value / srcTotal) * src.height;
				const linkHeightTgt = (link.value / tgtTotal) * tgt.height;

				const srcOff = sourceOffsets[link.source] ?? 0;
				const tgtOff = targetOffsets[link.target] ?? 0;

				const x0 = colPositions[src.column] + NODE_WIDTH;
				const y0top = src.y + srcOff;
				const y0bot = y0top + linkHeightSrc;
				const x1 = colPositions[tgt.column];
				const y1top = tgt.y + tgtOff;
				const y1bot = y1top + linkHeightTgt;

				sourceOffsets[link.source] = srcOff + linkHeightSrc;
				targetOffsets[link.target] = tgtOff + linkHeightTgt;

				const cpx = (x0 + x1) / 2;
				const d = [
					`M ${x0} ${y0top}`,
					`C ${cpx} ${y0top}, ${cpx} ${y1top}, ${x1} ${y1top}`,
					`L ${x1} ${y1bot}`,
					`C ${cpx} ${y1bot}, ${cpx} ${y0bot}, ${x0} ${y0bot}`,
					"Z",
				].join(" ");

				return { d, color: link.color, key: `${link.source}-${link.target}` };
			})
			.filter(Boolean);
	}, [nodes, links, colPositions]);

	if (iterations.length === 0) {
		return (
			<div className="flex h-full items-center justify-center text-sm text-[var(--text-secondary)]">
				Waiting for training data...
			</div>
		);
	}

	// Column headers
	const colHeaders = ["Regime", "Behavior", "Outcome"];

	return (
		<div ref={containerRef} className="relative h-full w-full">
			<svg
				width={dims.width}
				height={dims.height}
				className="select-none"
				role="img"
				aria-label="Decision flow Sankey diagram showing market regime to outcome mapping"
			>
				{/* Column headers */}
				{colHeaders.map((h, i) => (
					<text
						key={h}
						x={colPositions[i] + NODE_WIDTH / 2}
						y={PADDING.top - 10}
						fill="#8b8fa3"
						fontSize={10}
						textAnchor="middle"
						fontWeight={500}
					>
						{h}
					</text>
				))}

				{/* Links (behind nodes) */}
				{linkPaths.map((lp) =>
					lp ? (
						<path
							key={lp.key}
							d={lp.d}
							fill={lp.color}
							fillOpacity={0.15}
							stroke={lp.color}
							strokeOpacity={0.3}
							strokeWidth={0.5}
						/>
					) : null,
				)}

				{/* Nodes */}
				{nodes.map((node) => (
					<g key={node.id}>
						<rect
							x={colPositions[node.column]}
							y={node.y}
							width={NODE_WIDTH}
							height={Math.max(node.height, 2)}
							fill={node.color}
							rx={3}
							fillOpacity={0.8}
						/>
						<text
							x={
								node.column === 2
									? colPositions[node.column] - 4
									: colPositions[node.column] + NODE_WIDTH + 4
							}
							y={node.y + node.height / 2 + 3}
							fill="#8b8fa3"
							fontSize={9}
							textAnchor={node.column === 2 ? "end" : "start"}
						>
							{node.label} ({node.value})
						</text>
					</g>
				))}
			</svg>
		</div>
	);
}
