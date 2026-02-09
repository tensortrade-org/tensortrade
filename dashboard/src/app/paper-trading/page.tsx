"use client";

import { Card, CardHeader } from "@/components/common/Card";
import { LoadingState } from "@/components/common/Spinner";
import { useApi } from "@/hooks/useApi";
import { getExperiments, getLiveSessions, startLiveTrading, stopLiveTrading } from "@/lib/api";
import { formatCurrency, formatDate, formatNumber, formatPercent } from "@/lib/formatters";
import type {
	ExperimentSummary,
	LiveActionEvent,
	LiveBar,
	LiveBarsHistoryMessage,
	LivePortfolioMessage,
	LiveSession,
	LiveStatusMessage,
	LiveTradeEvent,
	LiveTradesHistoryMessage,
	WebSocketMessage,
} from "@/lib/types";
import { useLiveStore } from "@/stores/liveStore";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

// --- Constants ---

const TIMEFRAME_OPTIONS = [
	{ value: "1m", label: "1 Min" },
	{ value: "5m", label: "5 Min" },
	{ value: "15m", label: "15 Min" },
	{ value: "1h", label: "1 Hour" },
	{ value: "4h", label: "4 Hour" },
	{ value: "1d", label: "1 Day" },
] as const;

const WS_URL = typeof window !== "undefined" ? `ws://${window.location.hostname}:8000/ws/live` : "";

// --- WebSocket Hook ---

function useLiveWebSocket() {
	const wsRef = useRef<WebSocket | null>(null);
	const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
	const pingTimer = useRef<ReturnType<typeof setInterval> | null>(null);

	useEffect(() => {
		function connect() {
			if (!WS_URL) return;
			const ws = new WebSocket(WS_URL);
			wsRef.current = ws;

			ws.onopen = () => {
				pingTimer.current = setInterval(() => {
					if (ws.readyState === WebSocket.OPEN) ws.send("ping");
				}, 30000);
			};

			ws.onmessage = (event) => {
				try {
					const msg = JSON.parse(event.data) as WebSocketMessage;
					const s = useLiveStore.getState();
					switch (msg.type) {
						case "live_status":
							s.setStatus(msg as LiveStatusMessage);
							break;
						case "live_bar":
							s.addBar(msg as LiveBar);
							break;
						case "live_action":
							s.addAction(msg as LiveActionEvent);
							break;
						case "live_trade":
							s.addTrade(msg as LiveTradeEvent);
							break;
						case "live_portfolio":
							s.updatePortfolio(msg as LivePortfolioMessage);
							break;
						case "live_bars_history": {
							const barsMsg = msg as LiveBarsHistoryMessage;
							s.setBars(
								barsMsg.bars.map((b, i) => ({
									...b,
									type: "live_bar" as const,
									step: i,
								})),
							);
							break;
						}
						case "live_trades_history": {
							const tradesMsg = msg as LiveTradesHistoryMessage;
							s.setTrades(
								tradesMsg.trades.map((t) => ({
									...t,
									type: "live_trade" as const,
								})),
							);
							break;
						}
					}
				} catch {
					// skip malformed messages
				}
			};

			ws.onclose = () => {
				wsRef.current = null;
				if (pingTimer.current) clearInterval(pingTimer.current);
				reconnectTimer.current = setTimeout(connect, 3000);
			};

			ws.onerror = () => ws.close();
		}

		connect();
		return () => {
			if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
			if (pingTimer.current) clearInterval(pingTimer.current);
			wsRef.current?.close();
		};
	}, []);
}

// --- Candlestick Chart (lightweight-charts) ---

interface LiveCandlestickChartProps {
	bars: LiveBar[];
	trades: LiveTradeEvent[];
}

const LiveCandlestickChart = memo(function LiveCandlestickChart({
	bars,
	trades,
}: LiveCandlestickChartProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const chartRef = useRef<ReturnType<typeof import("lightweight-charts").createChart> | null>(null);
	const candleSeriesRef = useRef<ReturnType<
		NonNullable<typeof chartRef.current>["addCandlestickSeries"]
	> | null>(null);
	const volumeSeriesRef = useRef<ReturnType<
		NonNullable<typeof chartRef.current>["addLineSeries"]
	> | null>(null);
	const [chartReady, setChartReady] = useState(false);

	useEffect(() => {
		if (!containerRef.current) return;

		let chart: ReturnType<typeof import("lightweight-charts").createChart>;

		import("lightweight-charts").then(({ createChart, ColorType }) => {
			if (!containerRef.current) return;

			chart = createChart(containerRef.current, {
				layout: {
					background: { type: ColorType.Solid, color: "#1e2235" },
					textColor: "#8b8fa3",
				},
				grid: {
					vertLines: { color: "#2a2e45" },
					horzLines: { color: "#2a2e45" },
				},
				crosshair: {
					vertLine: {
						color: "#3b82f6",
						width: 1,
						labelBackgroundColor: "#3b82f6",
					},
					horzLine: {
						color: "#3b82f6",
						width: 1,
						labelBackgroundColor: "#3b82f6",
					},
				},
				rightPriceScale: { borderColor: "#2a2e45" },
				timeScale: { borderColor: "#2a2e45", timeVisible: true },
				width: containerRef.current.clientWidth,
				height: containerRef.current.clientHeight,
			});

			chartRef.current = chart;

			const candleSeries = chart.addCandlestickSeries({
				upColor: "#22c55e",
				downColor: "#ef4444",
				borderUpColor: "#22c55e",
				borderDownColor: "#ef4444",
				wickUpColor: "#22c55e",
				wickDownColor: "#ef4444",
			});
			candleSeriesRef.current = candleSeries;

			const volumeSeries = chart.addLineSeries({
				color: "#3b82f680",
				lineWidth: 1,
				priceScaleId: "volume",
			});
			volumeSeriesRef.current = volumeSeries;

			chart.priceScale("volume").applyOptions({
				scaleMargins: { top: 0.8, bottom: 0 },
				visible: false,
			});

			setChartReady(true);

			const handleResize = () => {
				if (containerRef.current && chartRef.current) {
					chartRef.current.applyOptions({
						width: containerRef.current.clientWidth,
						height: containerRef.current.clientHeight,
					});
				}
			};

			const resizeObserver = new ResizeObserver(handleResize);
			resizeObserver.observe(containerRef.current);

			return () => {
				resizeObserver.disconnect();
			};
		});

		return () => {
			if (chartRef.current) {
				chartRef.current.remove();
				chartRef.current = null;
				candleSeriesRef.current = null;
				volumeSeriesRef.current = null;
			}
		};
	}, []);

	const prevBarCountRef = useRef(0);
	const prevTradeCountRef = useRef(0);

	useEffect(() => {
		if (!candleSeriesRef.current || !volumeSeriesRef.current || !chartRef.current) return;
		if (bars.length === 0) return;

		type Time = import("lightweight-charts").Time;

		const isInitialLoad = prevBarCountRef.current === 0;
		const newBarCount = bars.length - prevBarCountRef.current;

		if (isInitialLoad || newBarCount > 5) {
			// Full setData for initial load or after a big batch (e.g. reconnect)
			const deduped = new Map<number, (typeof bars)[number]>();
			for (const b of bars) deduped.set(b.timestamp, b);
			const sorted = [...deduped.values()].sort((a, b) => a.timestamp - b.timestamp);

			candleSeriesRef.current.setData(
				sorted.map((b) => ({
					time: b.timestamp as Time,
					open: b.open,
					high: b.high,
					low: b.low,
					close: b.close,
				})),
			);
			volumeSeriesRef.current.setData(
				sorted.map((b) => ({ time: b.timestamp as Time, value: b.volume })),
			);
		} else if (newBarCount > 0) {
			// Incremental update for streaming bars
			for (let i = bars.length - newBarCount; i < bars.length; i++) {
				const b = bars[i];
				candleSeriesRef.current.update({
					time: b.timestamp as Time,
					open: b.open,
					high: b.high,
					low: b.low,
					close: b.close,
				});
				volumeSeriesRef.current.update({
					time: b.timestamp as Time,
					value: b.volume,
				});
			}
		}
		prevBarCountRef.current = bars.length;

		// Only update markers when trades change
		if (trades.length > prevTradeCountRef.current) {
			type SeriesMarker = import("lightweight-charts").SeriesMarker<Time>;
			const validTrades = trades.filter((t) => t.timestamp != null);
			if (validTrades.length > 0) {
				const markers: SeriesMarker[] = validTrades
					.map((t) => ({
						time: t.timestamp as Time,
						position: (t.side === "buy" ? "belowBar" : "aboveBar") as "belowBar" | "aboveBar",
						color: t.side === "buy" ? "#22c55e" : "#ef4444",
						shape: (t.side === "buy" ? "arrowUp" : "arrowDown") as "arrowUp" | "arrowDown",
						text: `${t.side.toUpperCase()} @ ${t.price.toFixed(2)}`,
					}))
					.sort((a, b) => (a.time as number) - (b.time as number));
				candleSeriesRef.current.setMarkers(markers);
			}
		}
		prevTradeCountRef.current = trades.length;

		// Auto-scroll to latest bar
		chartRef.current.timeScale().scrollToPosition(2, false);
	}, [bars, trades, chartReady]);

	return (
		<div className="relative h-full w-full">
			<div ref={containerRef} className="h-full w-full" />
			{bars.length === 0 && (
				<div className="absolute inset-0 flex items-center justify-center text-[var(--text-secondary)]">
					No price data — start a session to stream bars
				</div>
			)}
		</div>
	);
});

// --- Equity Chart ---

interface EquityDataPoint {
	idx: number;
	equity: number;
}

interface EquityTooltipPayload {
	value: number;
}

interface EquityTooltipProps {
	active?: boolean;
	payload?: EquityTooltipPayload[];
}

function EquityTooltip({ active, payload }: EquityTooltipProps) {
	if (!active || !payload || payload.length === 0) return null;
	return (
		<div className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-2 text-xs shadow-lg">
			<p className="font-medium text-[var(--text-primary)]">
				Equity: ${payload[0].value.toFixed(2)}
			</p>
		</div>
	);
}

interface EquityChartProps {
	portfolioHistory: { equity: number }[];
	initialEquity: number;
}

const EquityChart = memo(function EquityChart({
	portfolioHistory,
	initialEquity,
}: EquityChartProps) {
	if (portfolioHistory.length === 0) {
		return (
			<div className="flex h-full w-full items-center justify-center text-[var(--text-secondary)]">
				No equity data yet
			</div>
		);
	}

	const latest = portfolioHistory[portfolioHistory.length - 1].equity;
	const isAbove = latest >= initialEquity;
	const color = isAbove ? "#22c55e" : "#ef4444";

	const data: EquityDataPoint[] = portfolioHistory.map((p, i) => ({
		idx: i,
		equity: p.equity,
	}));

	return (
		<ResponsiveContainer width="100%" height="100%">
			<AreaChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
				<defs>
					<linearGradient id="liveEquityGradient" x1="0" y1="0" x2="0" y2="1">
						<stop offset="5%" stopColor={color} stopOpacity={0.3} />
						<stop offset="95%" stopColor={color} stopOpacity={0.02} />
					</linearGradient>
				</defs>
				<XAxis
					dataKey="idx"
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
				/>
				<YAxis
					stroke="#8b8fa3"
					tick={{ fill: "#8b8fa3", fontSize: 11 }}
					tickLine={{ stroke: "#2a2e45" }}
					axisLine={{ stroke: "#2a2e45" }}
					domain={["auto", "auto"]}
				/>
				<Tooltip content={<EquityTooltip />} />
				<Area
					type="monotone"
					dataKey="equity"
					stroke={color}
					strokeWidth={2}
					fill="url(#liveEquityGradient)"
				/>
			</AreaChart>
		</ResponsiveContainer>
	);
});

// --- Portfolio Card ---

interface PortfolioCardProps {
	equity: number;
	pnl: number;
	pnlPct: number;
	drawdownPct: number;
	position: "cash" | "asset";
	totalBars: number;
	totalTrades: number;
}

const PortfolioCard = memo(function PortfolioCard({
	equity,
	pnl,
	pnlPct,
	drawdownPct,
	position,
	totalBars,
	totalTrades,
}: PortfolioCardProps) {
	return (
		<Card>
			<CardHeader title="Portfolio" />
			<div className="space-y-3">
				<div className="flex items-center justify-between">
					<span className="text-sm text-[var(--text-secondary)]">Equity</span>
					<span className="font-mono text-sm font-medium text-[var(--text-primary)]">
						{equity > 0 ? formatCurrency(equity) : "--"}
					</span>
				</div>
				<div className="flex items-center justify-between">
					<span className="text-sm text-[var(--text-secondary)]">PnL</span>
					<span
						className="font-mono text-sm font-medium"
						style={{
							color: pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)",
						}}
					>
						{equity > 0
							? `${pnl >= 0 ? "+" : ""}${formatCurrency(pnl)} (${formatPercent(pnlPct)})`
							: "--"}
					</span>
				</div>
				<div className="flex items-center justify-between">
					<span className="text-sm text-[var(--text-secondary)]">Max Drawdown</span>
					<span className="font-mono text-sm text-[var(--accent-red)]">
						{drawdownPct > 0 ? `${drawdownPct.toFixed(2)}%` : "--"}
					</span>
				</div>
				<div className="flex items-center justify-between">
					<span className="text-sm text-[var(--text-secondary)]">Position</span>
					<span
						className="rounded-full px-2 py-0.5 text-xs font-medium"
						style={{
							backgroundColor:
								position === "asset" ? "rgba(34,197,94,0.15)" : "rgba(139,143,163,0.15)",
							color: position === "asset" ? "var(--accent-green)" : "var(--text-secondary)",
						}}
					>
						{position === "asset" ? "LONG" : "FLAT"}
					</span>
				</div>
				<div className="border-t border-[var(--border-color)] pt-2">
					<div className="flex items-center justify-between">
						<span className="text-xs text-[var(--text-secondary)]">Bars</span>
						<span className="font-mono text-xs text-[var(--text-primary)]">
							{formatNumber(totalBars)}
						</span>
					</div>
					<div className="mt-1 flex items-center justify-between">
						<span className="text-xs text-[var(--text-secondary)]">Trades</span>
						<span className="font-mono text-xs text-[var(--text-primary)]">
							{formatNumber(totalTrades)}
						</span>
					</div>
				</div>
			</div>
		</Card>
	);
});

// --- Action Log ---

interface ActionLogProps {
	actions: LiveActionEvent[];
}

const ACTION_COLORS: Record<string, string> = {
	hold: "var(--text-secondary)",
	buy: "var(--accent-green)",
	sell: "var(--accent-red)",
};

const POSITION_LABEL: Record<number, string> = { 0: "Cash", 1: "Asset" };

function formatTime(ts: number): string {
	const d = new Date(ts * 1000);
	return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

const ActionLog = memo(function ActionLog({ actions }: ActionLogProps) {
	const recent = useMemo(() => actions.slice(-50).reverse(), [actions]);

	return (
		<div className="h-full overflow-y-auto">
			{recent.length === 0 ? (
				<div className="flex h-full items-center justify-center text-xs text-[var(--text-secondary)]">
					No actions yet
				</div>
			) : (
				<div className="space-y-0.5">
					{recent.map((a, i) => (
						<div
							key={`${a.step}-${i}`}
							className="grid grid-cols-[4.5rem_2.5rem_5rem_3rem] gap-2 px-1 py-0.5 text-xs font-mono"
						>
							<span className="text-[var(--text-secondary)]">
								{a.timestamp ? formatTime(a.timestamp) : `#${a.step}`}
							</span>
							<span
								className="font-semibold uppercase"
								style={{ color: ACTION_COLORS[a.action_label] }}
							>
								{a.action_label}
							</span>
							<span className="text-right text-[var(--text-primary)]">
								{a.price != null ? `$${a.price.toFixed(2)}` : "—"}
							</span>
							<span className="text-right text-[var(--text-secondary)]">
								{a.position != null ? (POSITION_LABEL[a.position] ?? "—") : "—"}
							</span>
						</div>
					))}
				</div>
			)}
		</div>
	);
});

// --- Trade Table ---

interface TradeTableProps {
	trades: LiveTradeEvent[];
}

const TradeTable = memo(function TradeTable({ trades }: TradeTableProps) {
	const recent = useMemo(() => trades.slice(-50).reverse(), [trades]);

	return (
		<div className="h-full overflow-y-auto">
			{recent.length === 0 ? (
				<div className="flex h-full items-center justify-center text-xs text-[var(--text-secondary)]">
					No trades yet
				</div>
			) : (
				<table className="w-full text-xs">
					<thead className="sticky top-0 bg-[var(--bg-card)]">
						<tr className="border-b border-[var(--border-color)] text-left text-[var(--text-secondary)]">
							<th className="pb-1.5 pr-2 font-medium">Side</th>
							<th className="pb-1.5 pr-2 font-medium">Price</th>
							<th className="pb-1.5 pr-2 font-medium">Size</th>
							<th className="pb-1.5 pr-2 font-medium">Commission</th>
							<th className="pb-1.5 font-medium">Alpaca ID</th>
						</tr>
					</thead>
					<tbody>
						{recent.map((t) => (
							<tr
								key={`${t.step}-${t.timestamp}`}
								className="border-b border-[var(--border-color)]/50 last:border-0"
							>
								<td className="py-1 pr-2">
									<span
										className="font-medium uppercase"
										style={{
											color: t.side === "buy" ? "var(--accent-green)" : "var(--accent-red)",
										}}
									>
										{t.side}
									</span>
								</td>
								<td className="py-1 pr-2 font-mono text-[var(--text-primary)]">
									{formatCurrency(t.price)}
								</td>
								<td className="py-1 pr-2 font-mono text-[var(--text-primary)]">
									{t.size.toFixed(4)}
								</td>
								<td className="py-1 pr-2 font-mono text-[var(--text-secondary)]">
									{formatCurrency(t.commission)}
								</td>
								<td className="py-1 font-mono text-[var(--text-secondary)]">
									{t.alpaca_order_id ? t.alpaca_order_id.slice(0, 8) : "--"}
								</td>
							</tr>
						))}
					</tbody>
				</table>
			)}
		</div>
	);
});

// --- Session History ---

interface SessionHistoryProps {
	sessions: LiveSession[];
	loading: boolean;
}

function SessionHistory({ sessions, loading }: SessionHistoryProps) {
	if (loading) return <LoadingState message="Loading sessions..." />;

	if (sessions.length === 0) {
		return (
			<div className="py-6 text-center text-xs text-[var(--text-secondary)]">
				No previous sessions
			</div>
		);
	}

	return (
		<div className="overflow-x-auto">
			<table className="w-full text-sm">
				<thead>
					<tr className="border-b border-[var(--border-color)] text-left text-xs text-[var(--text-secondary)]">
						<th className="pb-2 pr-3 font-medium">Started</th>
						<th className="pb-2 pr-3 font-medium">Symbol</th>
						<th className="pb-2 pr-3 font-medium">Timeframe</th>
						<th className="pb-2 pr-3 font-medium">Status</th>
						<th className="pb-2 pr-3 font-medium text-right">PnL</th>
						<th className="pb-2 pr-3 font-medium text-right">Trades</th>
						<th className="pb-2 font-medium text-right">Drawdown</th>
					</tr>
				</thead>
				<tbody>
					{sessions.map((s) => (
						<tr key={s.id} className="border-b border-[var(--border-color)]/50 last:border-0">
							<td className="py-2 pr-3 text-xs text-[var(--text-primary)]">
								{formatDate(s.started_at)}
							</td>
							<td className="py-2 pr-3 font-mono text-xs text-[var(--text-primary)]">{s.symbol}</td>
							<td className="py-2 pr-3 text-xs text-[var(--text-secondary)]">{s.timeframe}</td>
							<td className="py-2 pr-3">
								<span
									className="rounded-full px-2 py-0.5 text-xs font-medium"
									style={{
										backgroundColor:
											s.status === "running"
												? "rgba(59,130,246,0.15)"
												: s.status === "stopped"
													? "rgba(139,143,163,0.15)"
													: "rgba(239,68,68,0.15)",
										color:
											s.status === "running"
												? "var(--accent-blue)"
												: s.status === "stopped"
													? "var(--text-secondary)"
													: "var(--accent-red)",
									}}
								>
									{s.status}
								</span>
							</td>
							<td className="py-2 pr-3 text-right font-mono text-xs">
								<span
									style={{
										color: s.pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)",
									}}
								>
									{s.pnl >= 0 ? "+" : ""}
									{formatCurrency(s.pnl)}
								</span>
							</td>
							<td className="py-2 pr-3 text-right font-mono text-xs text-[var(--text-primary)]">
								{formatNumber(s.total_trades)}
							</td>
							<td className="py-2 text-right font-mono text-xs text-[var(--accent-red)]">
								{s.max_drawdown_pct.toFixed(2)}%
							</td>
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
}

// --- Main Page ---

export default function PaperTradingPage() {
	useLiveWebSocket();

	const [selectedExperiment, setSelectedExperiment] = useState("");
	const [symbol, setSymbol] = useState("BTC/USD");
	const [timeframe, setTimeframe] = useState("1h");

	const status = useLiveStore((s) => s.status);
	const sessionId = useLiveStore((s) => s.sessionId);
	const equity = useLiveStore((s) => s.equity);
	const pnl = useLiveStore((s) => s.pnl);
	const pnlPct = useLiveStore((s) => s.pnlPct);
	const position = useLiveStore((s) => s.position);
	const bars = useLiveStore((s) => s.bars);
	const trades = useLiveStore((s) => s.trades);
	const actions = useLiveStore((s) => s.actions);
	const portfolioHistory = useLiveStore((s) => s.portfolioHistory);
	const drawdownPct = useLiveStore((s) => s.drawdownPct);
	const totalBars = useLiveStore((s) => s.totalBars);
	const totalTrades = useLiveStore((s) => s.totalTrades);
	const error = useLiveStore((s) => s.error);
	const setStarting = useLiveStore((s) => s.setStarting);
	const reset = useLiveStore((s) => s.reset);

	const experimentsFetcher = useCallback(
		() => getExperiments({ status: "completed", limit: 50 }),
		[],
	);
	const sessionsFetcher = useCallback(() => getLiveSessions(), []);

	const { data: experiments, loading: experimentsLoading } = useApi<ExperimentSummary[]>(
		experimentsFetcher,
		[],
	);
	const {
		data: sessions,
		loading: sessionsLoading,
		refresh: refreshSessions,
	} = useApi<LiveSession[]>(sessionsFetcher, []);

	const isRunning = status === "running" || status === "starting";

	const setError = useLiveStore((s) => s.setError);

	const handleStart = useCallback(async () => {
		if (!selectedExperiment) return;
		reset();
		setStarting();
		try {
			const result = await startLiveTrading({
				experiment_id: selectedExperiment,
				symbol,
				timeframe,
			});
			if ("error" in result && result.error) {
				setError(result.error as string);
			}
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to start live trading");
		}
	}, [selectedExperiment, symbol, timeframe, reset, setStarting, setError]);

	const handleStop = useCallback(async () => {
		try {
			await stopLiveTrading();
			refreshSessions();
		} catch (err) {
			console.error("Failed to stop live trading:", err);
		}
	}, [refreshSessions]);

	const initialEquity = useMemo(() => {
		if (portfolioHistory.length > 0) return portfolioHistory[0].equity;
		return equity > 0 ? equity : 10000;
	}, [portfolioHistory, equity]);

	return (
		<div className="space-y-4">
			{/* Header */}
			<div className="flex items-center justify-between">
				<div>
					<h1 className="text-xl font-semibold text-[var(--text-primary)]">Paper Trading</h1>
					<p className="mt-0.5 text-xs text-[var(--text-secondary)]">
						Live paper trading with trained models
					</p>
				</div>
				{sessionId && (
					<span className="rounded-md bg-[var(--bg-secondary)] px-2 py-1 text-xs text-[var(--text-secondary)]">
						Session: {sessionId.slice(0, 8)}
					</span>
				)}
			</div>

			{/* Top Bar: Controls */}
			<div className="flex flex-wrap items-center gap-3 rounded-lg border border-[var(--border-color)] bg-[var(--bg-card)] p-3">
				<select
					value={selectedExperiment}
					onChange={(e) => setSelectedExperiment(e.target.value)}
					disabled={isRunning || experimentsLoading}
					className="min-w-[200px] rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none disabled:opacity-50"
				>
					<option value="" disabled>
						{experimentsLoading ? "Loading..." : "Select experiment"}
					</option>
					{(experiments ?? []).map((exp) => (
						<option key={exp.id} value={exp.id}>
							{exp.name}
						</option>
					))}
				</select>

				<input
					type="text"
					value={symbol}
					onChange={(e) => setSymbol(e.target.value)}
					disabled={isRunning}
					placeholder="Symbol (e.g. BTC/USD)"
					className="w-36 rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none disabled:opacity-50"
				/>

				<select
					value={timeframe}
					onChange={(e) => setTimeframe(e.target.value)}
					disabled={isRunning}
					className="rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none disabled:opacity-50"
				>
					{TIMEFRAME_OPTIONS.map((tf) => (
						<option key={tf.value} value={tf.value}>
							{tf.label}
						</option>
					))}
				</select>

				<div className="ml-auto flex items-center gap-2">
					{/* Status indicator */}
					<div className="flex items-center gap-2">
						<span
							className="h-2 w-2 rounded-full"
							style={{
								backgroundColor:
									status === "running"
										? "var(--accent-green)"
										: status === "error"
											? "var(--accent-red)"
											: status === "starting"
												? "var(--accent-amber)"
												: "var(--text-secondary)",
							}}
						/>
						<span className="text-xs capitalize text-[var(--text-secondary)]">{status}</span>
					</div>

					{!isRunning ? (
						<button
							type="button"
							onClick={handleStart}
							disabled={!selectedExperiment}
							className="rounded-md bg-[var(--accent-green)] px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
						>
							Start
						</button>
					) : (
						<button
							type="button"
							onClick={handleStop}
							className="rounded-md bg-[var(--accent-red)] px-4 py-1.5 text-sm font-medium text-white hover:opacity-90"
						>
							Stop
						</button>
					)}
				</div>
			</div>

			{/* Error Banner */}
			{error && (
				<div className="rounded-lg border border-[var(--accent-red)]/30 bg-[var(--accent-red)]/5 px-4 py-3 text-sm text-[var(--accent-red)]">
					{error}
				</div>
			)}

			{/* Main Grid: Charts + Portfolio */}
			<div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
				{/* Left column: Charts */}
				<div className="space-y-4 lg:col-span-3">
					<Card className="h-[440px]">
						<CardHeader title="Price Action" />
						<div className="h-[calc(100%-2rem)]">
							<LiveCandlestickChart bars={bars} trades={trades} />
						</div>
					</Card>

					<Card className="h-[240px]">
						<CardHeader title="Equity Curve" />
						<div className="h-[calc(100%-2rem)]">
							<EquityChart portfolioHistory={portfolioHistory} initialEquity={initialEquity} />
						</div>
					</Card>
				</div>

				{/* Right column: Portfolio + Actions + Trades */}
				<div className="space-y-4 lg:col-span-1">
					<PortfolioCard
						equity={equity}
						pnl={pnl}
						pnlPct={pnlPct}
						drawdownPct={drawdownPct}
						position={position}
						totalBars={totalBars}
						totalTrades={totalTrades}
					/>

					<Card className="h-[200px]">
						<CardHeader title="Action Log" />
						<div className="h-[calc(100%-2rem)]">
							<ActionLog actions={actions} />
						</div>
					</Card>

					<Card className="h-[240px]">
						<CardHeader title={`Trades (${trades.length})`} />
						<div className="h-[calc(100%-2rem)]">
							<TradeTable trades={trades} />
						</div>
					</Card>
				</div>
			</div>

			{/* Session History */}
			<Card>
				<CardHeader
					title="Session History"
					action={
						<button
							type="button"
							onClick={refreshSessions}
							className="text-xs text-[var(--accent-blue)] hover:underline"
						>
							Refresh
						</button>
					}
				/>
				<SessionHistory sessions={sessions ?? []} loading={sessionsLoading} />
			</Card>
		</div>
	);
}
