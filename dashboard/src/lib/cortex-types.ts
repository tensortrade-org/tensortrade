/** Types for Agent Cortex visualizations */

// --- Strategy Genome (Heatmap Strip) ---

export type ColorMode = "diverging" | "sequential";

export interface GenomeChannelConfig {
	key: string;
	label: string;
	colorMode: ColorMode;
}

export interface GenomeColumn {
	iteration: number;
	values: number[];
}

export interface GenomeTooltipData {
	iteration: number;
	x: number;
	y: number;
	values: { label: string; value: number; color: string }[];
}

// --- Behavioral Orbit (Trajectory Plot) ---

export interface OrbitPoint {
	iteration: number;
	x: number; // trade_ratio
	y: number; // pnl_per_trade
	age: number; // 0..1 normalized
	radius: number; // dot size from |episode_return|
}

export interface OrbitQuadrant {
	label: string;
	description: string;
	x: "left" | "right";
	y: "top" | "bottom";
}

export interface OrbitTooltipData {
	iteration: number;
	tradeRatio: number;
	pnlPerTrade: number;
	episodeReturn: number;
	x: number;
	y: number;
}

// --- Learning Pulse (Radial Ring) ---

export interface PulseArc {
	label: string;
	fraction: number;
	color: string;
}

export interface PulseRing {
	iteration: number;
	arcs: PulseArc[];
	brightness: number; // 0..1 from reward normalization
}

export interface PulseHealthScore {
	score: number; // 0..100
	trend: "improving" | "declining" | "stable";
	trendLabel: string;
}

// --- Decision Flow (Sankey) ---

export type MarketRegime = "Improving" | "Declining" | "Sideways";
export type AgentBehavior = "Aggressive" | "Balanced" | "Conservative";
export type TradeOutcome = "Profitable" | "Breakeven" | "Losing";

export interface FlowNode {
	id: string;
	label: string;
	color: string;
	column: number; // 0=regime, 1=behavior, 2=outcome
	value: number;
	y: number;
	height: number;
}

export interface FlowLink {
	source: string;
	target: string;
	value: number;
	color: string;
}

export interface FlowState {
	nodes: FlowNode[];
	links: FlowLink[];
}
