"use client";

import type { DatasetConfig } from "@/lib/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useCallback } from "react";

type SourceType = DatasetConfig["source_type"];

interface SourceConfigRecord {
	[key: string]: string | number | boolean;
}

interface ExchangeOption {
	value: string;
	label: string;
	note?: string;
}

const EXCHANGES: readonly ExchangeOption[] = [
	{ value: "Bitfinex", label: "Bitfinex" },
	{ value: "Bitstamp", label: "Bitstamp" },
	{ value: "Binance", label: "Binance", note: "Uses USDT pairs" },
	{ value: "Gemini", label: "Gemini" },
] as const;

const TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"] as const;

const inputClass =
	"w-full rounded-md border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)]";
const labelClass = "text-sm text-[var(--text-secondary)]";

function CryptoDownloadForm({
	config,
	onChange,
}: { config: SourceConfigRecord; onChange: (config: SourceConfigRecord) => void }) {
	const selectedExchange = EXCHANGES.find((ex) => ex.value === config.exchange) ?? EXCHANGES[0];

	return (
		<div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
			<div className="flex flex-col gap-1">
				<label htmlFor="exchange" className={labelClass}>
					Exchange
				</label>
				<select
					id="exchange"
					value={String(config.exchange ?? "Bitfinex")}
					onChange={(e) => onChange({ ...config, exchange: e.target.value })}
					className={inputClass}
				>
					{EXCHANGES.map((ex) => (
						<option key={ex.value} value={ex.value}>
							{ex.label}
							{ex.note ? ` (${ex.note})` : ""}
						</option>
					))}
				</select>
				{selectedExchange.note && (
					<p className="text-xs text-[var(--text-secondary)]">{selectedExchange.note}</p>
				)}
			</div>

			<div className="flex flex-col gap-1">
				<label htmlFor="timeframe" className={labelClass}>
					Timeframe
				</label>
				<select
					id="timeframe"
					value={String(config.timeframe ?? "1h")}
					onChange={(e) => onChange({ ...config, timeframe: e.target.value })}
					className={inputClass}
				>
					{TIMEFRAMES.map((tf) => (
						<option key={tf} value={tf}>
							{tf}
						</option>
					))}
				</select>
			</div>

			<div className="flex flex-col gap-1">
				<label htmlFor="base" className={labelClass}>
					Base Currency
				</label>
				<input
					id="base"
					type="text"
					value={String(config.base ?? "BTC")}
					onChange={(e) => onChange({ ...config, base: e.target.value.toUpperCase() })}
					placeholder="BTC"
					className={inputClass}
				/>
			</div>

			<div className="flex flex-col gap-1">
				<label htmlFor="quote" className={labelClass}>
					Quote Currency
				</label>
				<input
					id="quote"
					type="text"
					value={String(config.quote ?? "USD")}
					onChange={(e) => onChange({ ...config, quote: e.target.value.toUpperCase() })}
					placeholder="USD"
					className={inputClass}
				/>
			</div>
		</div>
	);
}

function SyntheticForm({
	config,
	onChange,
}: { config: SourceConfigRecord; onChange: (config: SourceConfigRecord) => void }) {
	return (
		<div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
			<div className="flex flex-col gap-1">
				<label htmlFor="base_price" className={labelClass}>
					Base Price
				</label>
				<input
					id="base_price"
					type="number"
					value={Number(config.base_price ?? 100)}
					onChange={(e) => onChange({ ...config, base_price: Number(e.target.value) })}
					min={0}
					step={1}
					className={inputClass}
				/>
			</div>

			<div className="flex flex-col gap-1">
				<label htmlFor="base_volume" className={labelClass}>
					Base Volume
				</label>
				<input
					id="base_volume"
					type="number"
					value={Number(config.base_volume ?? 1000)}
					onChange={(e) => onChange({ ...config, base_volume: Number(e.target.value) })}
					min={0}
					step={100}
					className={inputClass}
				/>
			</div>

			<div className="flex flex-col gap-1">
				<label htmlFor="num_candles" className={labelClass}>
					Number of Candles
				</label>
				<input
					id="num_candles"
					type="number"
					value={Number(config.num_candles ?? 1000)}
					onChange={(e) => onChange({ ...config, num_candles: Number(e.target.value) })}
					min={100}
					step={100}
					className={inputClass}
				/>
			</div>

			<div className="flex flex-col gap-1">
				<label htmlFor="volatility" className={labelClass}>
					Volatility
				</label>
				<input
					id="volatility"
					type="number"
					value={Number(config.volatility ?? 0.02)}
					onChange={(e) => onChange({ ...config, volatility: Number(e.target.value) })}
					min={0}
					max={1}
					step={0.001}
					className={inputClass}
				/>
			</div>

			<div className="flex flex-col gap-1 sm:col-span-2">
				<label htmlFor="drift" className={labelClass}>
					Drift
				</label>
				<input
					id="drift"
					type="number"
					value={Number(config.drift ?? 0.0)}
					onChange={(e) => onChange({ ...config, drift: Number(e.target.value) })}
					min={-1}
					max={1}
					step={0.0001}
					className={inputClass}
				/>
			</div>
		</div>
	);
}

function CsvUploadForm({
	config,
	onChange,
}: { config: SourceConfigRecord; onChange: (config: SourceConfigRecord) => void }) {
	return (
		<div className="flex flex-col gap-1">
			<label htmlFor="file_path" className={labelClass}>
				CSV File Path
			</label>
			<input
				id="file_path"
				type="text"
				value={String(config.file_path ?? "")}
				onChange={(e) => onChange({ ...config, file_path: e.target.value })}
				placeholder="/path/to/data.csv"
				className={inputClass}
			/>
			<p className="text-xs text-[var(--text-secondary)]">
				Enter the path to a CSV file with OHLCV columns (date, open, high, low, close, volume).
			</p>
		</div>
	);
}

interface AlpacaSymbolOption {
	value: string;
	label: string;
}

const ALPACA_SYMBOLS: readonly AlpacaSymbolOption[] = [
	{ value: "BTC/USD", label: "BTC/USD" },
	{ value: "ETH/USD", label: "ETH/USD" },
	{ value: "SOL/USD", label: "SOL/USD" },
	{ value: "DOGE/USD", label: "DOGE/USD" },
	{ value: "AVAX/USD", label: "AVAX/USD" },
	{ value: "LINK/USD", label: "LINK/USD" },
] as const;

const ALPACA_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"] as const;

function AlpacaCryptoForm({
	config,
	onChange,
}: { config: SourceConfigRecord; onChange: (config: SourceConfigRecord) => void }) {
	const symbolValue = String(config.symbol ?? "BTC/USD");
	const isCustomSymbol = !ALPACA_SYMBOLS.some((s) => s.value === symbolValue);

	return (
		<div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
			<div className="flex flex-col gap-1">
				<label htmlFor="alpaca_symbol" className={labelClass}>
					Symbol
				</label>
				<select
					id="alpaca_symbol"
					value={isCustomSymbol ? "__custom__" : symbolValue}
					onChange={(e) => {
						if (e.target.value === "__custom__") {
							onChange({ ...config, symbol: "" });
						} else {
							onChange({ ...config, symbol: e.target.value });
						}
					}}
					className={inputClass}
				>
					{ALPACA_SYMBOLS.map((s) => (
						<option key={s.value} value={s.value}>
							{s.label}
						</option>
					))}
					<option value="__custom__">Custom...</option>
				</select>
				{(isCustomSymbol || symbolValue === "") && (
					<input
						type="text"
						value={symbolValue}
						onChange={(e) => onChange({ ...config, symbol: e.target.value.toUpperCase() })}
						placeholder="e.g. MATIC/USD"
						className={`${inputClass} mt-1`}
					/>
				)}
			</div>

			<div className="flex flex-col gap-1">
				<label htmlFor="alpaca_timeframe" className={labelClass}>
					Timeframe
				</label>
				<select
					id="alpaca_timeframe"
					value={String(config.timeframe ?? "1h")}
					onChange={(e) => onChange({ ...config, timeframe: e.target.value })}
					className={inputClass}
				>
					{ALPACA_TIMEFRAMES.map((tf) => (
						<option key={tf} value={tf}>
							{tf}
						</option>
					))}
				</select>
			</div>

			<div className="flex flex-col gap-1">
				<label htmlFor="alpaca_start_date" className={labelClass}>
					Start Date
				</label>
				<input
					id="alpaca_start_date"
					type="date"
					value={String(config.start_date ?? "")}
					onChange={(e) => onChange({ ...config, start_date: e.target.value })}
					className={inputClass}
				/>
				<p className="text-xs text-[var(--text-secondary)]">Leave empty for 2 years of history</p>
			</div>

			<div className="flex flex-col gap-1">
				<label htmlFor="alpaca_end_date" className={labelClass}>
					End Date
				</label>
				<input
					id="alpaca_end_date"
					type="date"
					value={String(config.end_date ?? "")}
					onChange={(e) => onChange({ ...config, end_date: e.target.value })}
					className={inputClass}
				/>
				<p className="text-xs text-[var(--text-secondary)]">Leave empty for latest data</p>
			</div>
		</div>
	);
}

const SOURCE_TYPE_LABELS: Record<SourceType, string> = {
	crypto_download: "Crypto Download",
	alpaca_crypto: "Alpaca Crypto",
	synthetic: "Synthetic Data",
	csv_upload: "CSV Upload",
};

export function SourceConfig() {
	const { editingDataset, updateEditingField } = useDatasetStore();

	const sourceType: SourceType = editingDataset?.source_type ?? "crypto_download";
	const sourceConfig: SourceConfigRecord = editingDataset?.source_config ?? {};

	const handleSourceTypeChange = useCallback(
		(type: SourceType) => {
			updateEditingField("source_type", type);
			// Reset source_config when type changes
			updateEditingField("source_config", {});
		},
		[updateEditingField],
	);

	const handleConfigChange = useCallback(
		(config: SourceConfigRecord) => {
			updateEditingField("source_config", config);
		},
		[updateEditingField],
	);

	const sourceTypes: SourceType[] = ["crypto_download", "alpaca_crypto", "synthetic", "csv_upload"];

	return (
		<div className="space-y-4">
			{/* Source type selector */}
			<div className="flex gap-2">
				{sourceTypes.map((type) => (
					<button
						key={type}
						type="button"
						onClick={() => handleSourceTypeChange(type)}
						className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
							sourceType === type
								? "bg-[var(--accent-blue)] text-white"
								: "border border-[var(--border-color)] text-[var(--text-secondary)] hover:bg-[var(--bg-secondary)]"
						}`}
					>
						{SOURCE_TYPE_LABELS[type]}
					</button>
				))}
			</div>

			{/* Source-specific form */}
			<div className="rounded-lg border border-[var(--border-color)] bg-[var(--bg-card)] p-4">
				{sourceType === "crypto_download" && (
					<CryptoDownloadForm config={sourceConfig} onChange={handleConfigChange} />
				)}
				{sourceType === "alpaca_crypto" && (
					<AlpacaCryptoForm config={sourceConfig} onChange={handleConfigChange} />
				)}
				{sourceType === "synthetic" && (
					<SyntheticForm config={sourceConfig} onChange={handleConfigChange} />
				)}
				{sourceType === "csv_upload" && (
					<CsvUploadForm config={sourceConfig} onChange={handleConfigChange} />
				)}
			</div>
		</div>
	);
}
