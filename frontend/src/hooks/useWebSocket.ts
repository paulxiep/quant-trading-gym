/**
 * WebSocket hook for real-time simulation updates (V4.2).
 *
 * Connects to the server's WebSocket endpoint and provides:
 * - Real-time tick data
 * - Connection state management
 * - Automatic reconnection
 */

import { useEffect, useRef, useState, useCallback } from 'react';

/** Per-symbol market data from server. */
export interface SymbolData {
  symbol: string;
  last_price: number | null;
  best_bid: number | null;
  best_ask: number | null;
  bid_depth: number;
  ask_depth: number;
}

/** Tick data received from server. */
export interface TickData {
  tick: number;
  timestamp: number;
  symbols: Record<string, SymbolData>;
  trades_this_tick: number;
  total_trades: number;
  total_orders: number;
  agents_called: number;
}

/** WebSocket connection state. */
export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error';

/** Options for useWebSocket hook. */
export interface UseWebSocketOptions {
  /** Server URL (default: ws://localhost:8001/ws) */
  url?: string;
  /** Auto-reconnect on disconnect (default: true) */
  autoReconnect?: boolean;
  /** Reconnect interval in ms (default: 3000) */
  reconnectInterval?: number;
}

/** Return type for useWebSocket hook. */
export interface UseWebSocketReturn {
  /** Latest tick data from server. */
  tickData: TickData | null;
  /** Current connection state. */
  connectionState: ConnectionState;
  /** Send command to server. */
  sendCommand: (command: 'Start' | 'Pause' | 'Toggle' | 'Step' | 'Quit') => void;
  /** Manually connect to server. */
  connect: () => void;
  /** Manually disconnect from server. */
  disconnect: () => void;
}

const DEFAULT_WS_URL = 'ws://localhost:8001/ws';

/**
 * Hook for WebSocket connection to simulation server.
 *
 * @example
 * ```tsx
 * const { tickData, connectionState, sendCommand } = useWebSocket();
 *
 * // Start simulation
 * sendCommand('Start');
 *
 * // Display current tick
 * if (tickData) {
 *   console.log(`Tick: ${tickData.tick}`);
 * }
 * ```
 */
export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const { url = DEFAULT_WS_URL, autoReconnect = true, reconnectInterval = 3000 } = options;

  const [tickData, setTickData] = useState<TickData | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionState('connecting');

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setConnectionState('connected');
        console.log('WebSocket connected to', url);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TickData;
          setTickData(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionState('error');
      };

      ws.onclose = () => {
        setConnectionState('disconnected');
        wsRef.current = null;

        if (autoReconnect) {
          console.log(`Reconnecting in ${reconnectInterval}ms...`);
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setConnectionState('error');
    }
  }, [url, autoReconnect, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionState('disconnected');
  }, []);

  const sendCommand = useCallback((command: 'Start' | 'Pause' | 'Toggle' | 'Step' | 'Quit') => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(command));
    } else {
      console.warn('Cannot send command: WebSocket not connected');
    }
  }, []);

  // Connect on mount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    tickData,
    connectionState,
    sendCommand,
    connect,
    disconnect,
  };
}
