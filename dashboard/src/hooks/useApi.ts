"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface UseApiReturn<T> {
	data: T | null;
	error: Error | null;
	loading: boolean;
	refresh: () => void;
}

export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[] = []): UseApiReturn<T> {
	const [data, setData] = useState<T | null>(null);
	const [error, setError] = useState<Error | null>(null);
	const [loading, setLoading] = useState(true);
	const fetcherRef = useRef(fetcher);
	const mountedRef = useRef(true);

	// Keep fetcher ref current
	useEffect(() => {
		fetcherRef.current = fetcher;
	}, [fetcher]);

	const execute = useCallback(async () => {
		setLoading(true);
		setError(null);
		try {
			const result = await fetcherRef.current();
			if (mountedRef.current) {
				setData(result);
			}
		} catch (err) {
			if (mountedRef.current) {
				setError(err instanceof Error ? err : new Error(String(err)));
			}
		} finally {
			if (mountedRef.current) {
				setLoading(false);
			}
		}
	}, []);

	useEffect(() => {
		mountedRef.current = true;
		execute();
		return () => {
			mountedRef.current = false;
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, deps);

	const refresh = useCallback(() => {
		execute();
	}, [execute]);

	return { data, error, loading, refresh };
}
