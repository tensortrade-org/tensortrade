import type {
	LiveActionEvent,
	LiveBar,
	LivePortfolioMessage,
	LiveStatusMessage,
	LiveTradeEvent,
} from "@/lib/types";
import { create } from "zustand";

type LiveState = "idle" | "running" | "stopped" | "error" | "starting";

interface PortfolioSnapshot {
	timestamp: number;
	equity: number;
	pnl: number;
	drawdown_pct: number;
}

interface LiveStore {
	status: LiveState;
	sessionId: string | null;
	symbol: string;
	equity: number;
	pnl: number;
	pnlPct: number;
	position: "cash" | "asset";
	entryPrice: number | null;
	bars: LiveBar[];
	trades: LiveTradeEvent[];
	actions: LiveActionEvent[];
	portfolioHistory: PortfolioSnapshot[];
	drawdownPct: number;
	error: string | null;
	totalBars: number;
	totalTrades: number;

	setStarting: () => void;
	setStatus: (msg: LiveStatusMessage) => void;
	addBar: (bar: LiveBar) => void;
	setBars: (bars: LiveBar[]) => void;
	addTrade: (trade: LiveTradeEvent) => void;
	setTrades: (trades: LiveTradeEvent[]) => void;
	addAction: (action: LiveActionEvent) => void;
	updatePortfolio: (msg: LivePortfolioMessage) => void;
	setError: (error: string) => void;
	reset: () => void;
}

const initialState: Pick<
	LiveStore,
	| "status"
	| "sessionId"
	| "symbol"
	| "equity"
	| "pnl"
	| "pnlPct"
	| "position"
	| "entryPrice"
	| "bars"
	| "trades"
	| "actions"
	| "portfolioHistory"
	| "drawdownPct"
	| "error"
	| "totalBars"
	| "totalTrades"
> = {
	status: "idle",
	sessionId: null,
	symbol: "",
	equity: 0,
	pnl: 0,
	pnlPct: 0,
	position: "cash",
	entryPrice: null,
	bars: [],
	trades: [],
	actions: [],
	portfolioHistory: [],
	drawdownPct: 0,
	error: null,
	totalBars: 0,
	totalTrades: 0,
};

export const useLiveStore = create<LiveStore>((set) => ({
	...initialState,

	setStarting: () => set({ status: "starting" }),

	setStatus: (msg: LiveStatusMessage) =>
		set({
			status: msg.state,
			sessionId: msg.session_id,
			symbol: msg.symbol,
			equity: msg.equity,
			pnl: msg.pnl,
			pnlPct: msg.pnl_pct,
			position: msg.position,
			entryPrice: msg.entry_price ?? null,
			totalBars: msg.total_bars,
			totalTrades: msg.total_trades,
			drawdownPct: msg.drawdown_pct,
		}),

	addBar: (bar: LiveBar) =>
		set((state) => {
			const bars =
				state.bars.length >= 1000 ? [...state.bars.slice(-500), bar] : [...state.bars, bar];
			return { bars, totalBars: state.totalBars + 1 };
		}),

	setBars: (bars: LiveBar[]) => set({ bars }),

	addTrade: (trade: LiveTradeEvent) =>
		set((state) => ({
			trades:
				state.trades.length >= 500
					? [...state.trades.slice(-250), trade]
					: [...state.trades, trade],
			totalTrades: state.totalTrades + 1,
			entryPrice: trade.entry_price ?? state.entryPrice,
		})),

	setTrades: (trades: LiveTradeEvent[]) => set({ trades }),

	addAction: (action: LiveActionEvent) =>
		set((state) => ({
			actions:
				state.actions.length >= 200
					? [...state.actions.slice(-100), action]
					: [...state.actions, action],
		})),

	updatePortfolio: (msg: LivePortfolioMessage) =>
		set((state) => {
			const history =
				state.portfolioHistory.length >= 1000
					? state.portfolioHistory.slice(-500)
					: state.portfolioHistory;
			return {
				equity: msg.equity,
				pnl: msg.pnl,
				pnlPct: msg.pnl_pct,
				drawdownPct: msg.drawdown_pct,
				entryPrice: msg.entry_price ?? state.entryPrice,
				portfolioHistory: [
					...history,
					{
						timestamp: Date.now(),
						equity: msg.equity,
						pnl: msg.pnl,
						drawdown_pct: msg.drawdown_pct,
					},
				],
			};
		}),

	setError: (error: string) => set({ status: "error", error }),

	reset: () => set(initialState),
}));
