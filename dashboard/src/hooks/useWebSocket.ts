"use client";

import type { WebSocketMessage } from "@/lib/types";
import { useCallback, useEffect, useRef, useState } from "react";

interface UseWebSocketOptions {
	url: string;
	onMessage: (msg: WebSocketMessage) => void;
	reconnectInterval?: number;
}

interface UseWebSocketReturn {
	connected: boolean;
	send: (data: Record<string, unknown>) => void;
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
	const { url, onMessage, reconnectInterval = 3000 } = options;
	const [connected, setConnected] = useState(false);
	const wsRef = useRef<WebSocket | null>(null);
	const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const onMessageRef = useRef(onMessage);
	const mountedRef = useRef(true);

	// Keep the onMessage callback ref current without triggering reconnects
	useEffect(() => {
		onMessageRef.current = onMessage;
	}, [onMessage]);

	const clearReconnectTimer = useCallback(() => {
		if (reconnectTimerRef.current !== null) {
			clearTimeout(reconnectTimerRef.current);
			reconnectTimerRef.current = null;
		}
	}, []);

	const connect = useCallback(() => {
		if (!mountedRef.current) return;
		clearReconnectTimer();

		// Close existing connection if any
		if (wsRef.current) {
			wsRef.current.onopen = null;
			wsRef.current.onclose = null;
			wsRef.current.onerror = null;
			wsRef.current.onmessage = null;
			wsRef.current.close();
			wsRef.current = null;
		}

		const ws = new WebSocket(url);

		ws.onopen = () => {
			if (!mountedRef.current) {
				ws.close();
				return;
			}
			setConnected(true);
		};

		ws.onclose = () => {
			if (!mountedRef.current) return;
			setConnected(false);
			wsRef.current = null;
			// Schedule reconnect
			reconnectTimerRef.current = setTimeout(() => {
				connect();
			}, reconnectInterval);
		};

		ws.onerror = () => {
			// onclose will fire after onerror, handling reconnect
		};

		ws.onmessage = (event: MessageEvent) => {
			try {
				const data = JSON.parse(event.data as string) as WebSocketMessage;
				onMessageRef.current(data);
			} catch {
				// Ignore malformed messages
			}
		};

		wsRef.current = ws;
	}, [url, reconnectInterval, clearReconnectTimer]);

	useEffect(() => {
		mountedRef.current = true;
		connect();

		return () => {
			mountedRef.current = false;
			clearReconnectTimer();
			if (wsRef.current) {
				wsRef.current.onopen = null;
				wsRef.current.onclose = null;
				wsRef.current.onerror = null;
				wsRef.current.onmessage = null;
				wsRef.current.close();
				wsRef.current = null;
			}
		};
	}, [connect, clearReconnectTimer]);

	const send = useCallback((data: Record<string, unknown>) => {
		if (wsRef.current?.readyState === WebSocket.OPEN) {
			wsRef.current.send(JSON.stringify(data));
		}
	}, []);

	return { connected, send };
}
