"use client";

import { Badge } from "@/components/common/Badge";
import { LoadingState } from "@/components/common/Spinner";
import { useApi } from "@/hooks/useApi";
import { getHyperparamPacks } from "@/lib/api";
import type { HyperparameterPack } from "@/lib/types";
import { useHyperparamStore } from "@/stores/hyperparamStore";
import { useCallback, useEffect, useMemo, useState } from "react";

const algorithmVariant: Record<string, "info" | "purple" | "success" | "warning"> = {
	PPO: "info",
	DQN: "purple",
	SAC: "success",
	IMPALA: "warning",
};

export function PackList() {
	const { packs, selectedPackId, selectPack, setPacks, setLoading, setError } =
		useHyperparamStore();
	const [search, setSearch] = useState("");

	const fetcher = useCallback(() => getHyperparamPacks(), []);
	const { data, loading, error } = useApi<HyperparameterPack[]>(fetcher, []);

	useEffect(() => {
		if (data) {
			setPacks(data);
			if (!selectedPackId && data.length > 0) {
				selectPack(data[0].id);
			}
		}
	}, [data, setPacks, selectPack, selectedPackId]);

	useEffect(() => {
		setLoading(loading);
	}, [loading, setLoading]);

	useEffect(() => {
		if (error) setError(error.message);
	}, [error, setError]);

	const filtered = useMemo(() => {
		if (!search.trim()) return packs;
		const q = search.toLowerCase();
		return packs.filter(
			(p) =>
				p.name.toLowerCase().includes(q) ||
				p.config.algorithm.toLowerCase().includes(q) ||
				p.description.toLowerCase().includes(q),
		);
	}, [packs, search]);

	const handleNewPack = () => {
		selectPack(null);
	};

	if (loading) return <LoadingState message="Loading packs..." />;
	if (error) {
		return <div className="p-4 text-sm text-[var(--accent-red)]">Failed to load packs</div>;
	}

	return (
		<div className="flex h-full flex-col border-r border-[var(--border-color)]">
			<div className="p-3 space-y-2">
				<input
					type="text"
					placeholder="Search packs..."
					value={search}
					onChange={(e) => setSearch(e.target.value)}
					className="w-full rounded border border-[var(--border-color)] bg-[var(--bg-secondary)] px-3 py-1.5 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:border-[var(--accent-blue)] focus:outline-none"
				/>
				<button
					type="button"
					onClick={handleNewPack}
					className="w-full rounded bg-[var(--accent-blue)] px-3 py-1.5 text-sm font-medium text-white hover:opacity-90 transition-opacity"
				>
					New Pack
				</button>
			</div>
			<div className="flex-1 overflow-y-auto">
				{filtered.map((pack) => (
					<button
						key={pack.id}
						type="button"
						onClick={() => selectPack(pack.id)}
						className={`w-full text-left px-3 py-3 border-b border-[var(--border-color)] transition-colors hover:bg-[var(--bg-secondary)] ${
							selectedPackId === pack.id
								? "border-l-2 border-l-[var(--accent-blue)] bg-[var(--bg-secondary)]"
								: "border-l-2 border-l-transparent"
						}`}
					>
						<div className="flex items-center gap-2 mb-1">
							<span className="text-sm font-medium text-[var(--text-primary)] truncate">
								{pack.name}
							</span>
							<Badge
								label={pack.config.algorithm}
								variant={algorithmVariant[pack.config.algorithm] ?? "default"}
							/>
						</div>
						<p className="text-xs text-[var(--text-secondary)] truncate">{pack.description}</p>
					</button>
				))}
				{filtered.length === 0 && (
					<div className="p-4 text-center text-xs text-[var(--text-secondary)]">No packs found</div>
				)}
			</div>
		</div>
	);
}
