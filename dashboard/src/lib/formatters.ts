/**
 * Format a number as US currency: $1,234.56
 */
export function formatCurrency(value: number): string {
	return new Intl.NumberFormat("en-US", {
		style: "currency",
		currency: "USD",
		minimumFractionDigits: 2,
		maximumFractionDigits: 2,
	}).format(value);
}

/**
 * Format a PnL value with explicit sign: +$1,234 or -$1,234
 */
export function formatPnl(value: number): string {
	const sign = value >= 0 ? "+" : "";
	const formatted = new Intl.NumberFormat("en-US", {
		style: "currency",
		currency: "USD",
		minimumFractionDigits: 0,
		maximumFractionDigits: 0,
	}).format(Math.abs(value));
	return value < 0 ? `-${formatted}` : `${sign}${formatted}`;
}

/**
 * Format a decimal as a percentage with sign: +1.23% or -1.23%
 */
export function formatPercent(value: number): string {
	const sign = value >= 0 ? "+" : "";
	return `${sign}${value.toFixed(2)}%`;
}

/**
 * Format an ISO date string to a readable local format.
 * Example: "Jan 15, 2026, 3:45 PM"
 */
export function formatDate(dateStr: string): string {
	const date = new Date(dateStr);
	if (Number.isNaN(date.getTime())) return dateStr;
	return new Intl.DateTimeFormat("en-US", {
		month: "short",
		day: "numeric",
		year: "numeric",
		hour: "numeric",
		minute: "2-digit",
	}).format(date);
}

/**
 * Format a duration in seconds to a human-readable string.
 * Examples: "1h 23m", "45s", "2d 3h"
 */
export function formatDuration(seconds: number): string {
	if (seconds < 0) return "0s";

	const days = Math.floor(seconds / 86400);
	const hours = Math.floor((seconds % 86400) / 3600);
	const minutes = Math.floor((seconds % 3600) / 60);
	const secs = Math.floor(seconds % 60);

	if (days > 0) {
		return hours > 0 ? `${days}d ${hours}h` : `${days}d`;
	}
	if (hours > 0) {
		return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
	}
	if (minutes > 0) {
		return secs > 0 ? `${minutes}m ${secs}s` : `${minutes}m`;
	}
	return `${secs}s`;
}

/**
 * Format a number with commas and optional decimal places.
 * Example: formatNumber(1234567.891, 2) => "1,234,567.89"
 */
export function formatNumber(value: number, decimals = 0): string {
	return new Intl.NumberFormat("en-US", {
		minimumFractionDigits: decimals,
		maximumFractionDigits: decimals,
	}).format(value);
}
