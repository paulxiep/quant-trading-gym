/**
 * Data service hooks for REST API endpoints (V4.4).
 *
 * Provides React hooks for fetching simulation data from the REST API.
 * Follows declarative-modular-SoC principles with:
 * - Single responsibility per hook
 * - Consistent error/loading state handling
 * - Auto-refresh capability tied to tick updates
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  CandlesResponse,
  IndicatorsResponse,
  FactorsResponse,
  AgentsResponse,
  RiskMetricsResponse,
  ActiveNewsResponse,
  OrderDistributionResponse,
} from '../types/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Base API configuration. */
export interface ApiConfig {
  /** Base URL for REST API (default: http://localhost:8001) */
  baseUrl?: string;
  /** Symbol to fetch data for (passed as query param where applicable) */
  symbol?: string;
}

/** Generic fetch state for all data hooks. */
export interface FetchState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => void;
}

/** Options for auto-refresh hooks. */
export interface AutoRefreshOptions {
  /** Enable auto-refresh (default: true) */
  enabled?: boolean;
  /** Refresh interval in ms (default: 1000) */
  interval?: number;
}

const DEFAULT_BASE_URL = 'http://localhost:8001';

// ---------------------------------------------------------------------------
// Core fetch utility
// ---------------------------------------------------------------------------

/**
 * Generic fetch function with error handling.
 */
async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Generic data hook factory
// ---------------------------------------------------------------------------

/**
 * Build URL with optional symbol query parameter.
 */
function buildUrl(baseUrl: string, endpoint: string, symbol?: string, useSymbol?: boolean): string {
  const url = `${baseUrl}${endpoint}`;
  if (useSymbol && symbol) {
    const separator = endpoint.includes('?') ? '&' : '?';
    return `${url}${separator}symbol=${encodeURIComponent(symbol)}`;
  }
  return url;
}

/**
 * Factory for creating data fetch hooks with consistent patterns.
 * @param endpoint - API endpoint path
 * @param useSymbol - Whether to append symbol query param (default: false)
 */
function createDataHook<T>(endpoint: string, useSymbol = false) {
  return function useData(config: ApiConfig = {}, refresh: AutoRefreshOptions = {}): FetchState<T> {
    const { baseUrl = DEFAULT_BASE_URL, symbol } = config;
    const { enabled = true, interval = 1000 } = refresh;

    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<Error | null>(null);
    const mountedRef = useRef(true);

    const fetchData = useCallback(async () => {
      if (!enabled) return;

      setLoading(true);
      try {
        const url = buildUrl(baseUrl, endpoint, symbol, useSymbol);
        const result = await fetchJson<T>(url);
        if (mountedRef.current) {
          setData(result);
          setError(null);
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
    }, [baseUrl, symbol, enabled]);

    // Initial fetch
    useEffect(() => {
      mountedRef.current = true;
      fetchData();
      return () => {
        mountedRef.current = false;
      };
    }, [fetchData]);

    // Auto-refresh interval
    useEffect(() => {
      if (!enabled || interval <= 0) return;

      const timer = setInterval(fetchData, interval);
      return () => clearInterval(timer);
    }, [enabled, interval, fetchData]);

    return { data, loading, error, refetch: fetchData };
  };
}

// ---------------------------------------------------------------------------
// Specific data hooks
// ---------------------------------------------------------------------------

/**
 * Fetch OHLCV candle data for price charts.
 * Supports symbol parameter: `config.symbol` appended as `?symbol=X`
 *
 * @example
 * ```tsx
 * const { data, loading, error } = useCandles({ symbol: 'AAPL' });
 * if (data) {
 *   console.log(`${data.candles.length} candles for ${data.symbol}`);
 * }
 * ```
 */
export const useCandles = createDataHook<CandlesResponse>('/api/analytics/candles', true);

/**
 * Fetch technical indicator values.
 * Supports symbol parameter: `config.symbol` appended as `?symbol=X`
 *
 * @example
 * ```tsx
 * const { data } = useIndicators({ symbol: 'AAPL' });
 * if (data?.sma_20) {
 *   console.log(`SMA-20: ${data.sma_20}`);
 * }
 * ```
 */
export const useIndicators = createDataHook<IndicatorsResponse>('/api/analytics/indicators', true);

/**
 * Fetch macro factor values for gauges.
 * Supports symbol parameter: `config.symbol` appended as `?symbol=X`
 *
 * @example
 * ```tsx
 * const { data } = useFactors({ symbol: 'AAPL' });
 * data?.factors.forEach(f => console.log(`${f.name}: ${f.value}`));
 * ```
 */
export const useFactors = createDataHook<FactorsResponse>('/api/analytics/factors', true);

/**
 * Fetch agent data for explorer table.
 *
 * @example
 * ```tsx
 * const { data } = useAgents();
 * data?.agents.sort((a, b) => b.pnl - a.pnl);
 * ```
 */
export const useAgents = createDataHook<AgentsResponse>('/api/portfolio/agents');

/**
 * Fetch risk metrics for dashboard panel.
 *
 * @example
 * ```tsx
 * const { data } = useRiskMetrics();
 * if (data) {
 *   console.log(`VaR 95: ${data.var_95}`);
 * }
 * ```
 */
export const useRiskMetrics = createDataHook<RiskMetricsResponse>('/api/risk/aggregate');

/**
 * Fetch active news events for feed.
 *
 * @example
 * ```tsx
 * const { data } = useActiveNews();
 * data?.events.forEach(e => console.log(e.headline));
 * ```
 */
export const useActiveNews = createDataHook<ActiveNewsResponse>('/api/news/active');

/**
 * Fetch pre-auction order distribution for depth chart.
 * Supports symbol parameter: `config.symbol` appended as `?symbol=X`
 *
 * @example
 * ```tsx
 * const { data } = useOrderDistribution({ symbol: 'AAPL' });
 * if (data) {
 *   console.log(`${data.bids.length} bid levels`);
 * }
 * ```
 */
export const useOrderDistribution = createDataHook<OrderDistributionResponse>(
  '/api/analytics/order-distribution',
  true,
);

// ---------------------------------------------------------------------------
// Composite hook for dashboard
// ---------------------------------------------------------------------------

/** All dashboard data bundled together. */
export interface DashboardData {
  candles: FetchState<CandlesResponse>;
  indicators: FetchState<IndicatorsResponse>;
  factors: FetchState<FactorsResponse>;
  agents: FetchState<AgentsResponse>;
  risk: FetchState<RiskMetricsResponse>;
  news: FetchState<ActiveNewsResponse>;
  orderDistribution: FetchState<OrderDistributionResponse>;
}

/**
 * Composite hook that fetches all dashboard data.
 *
 * Use this when you need multiple data streams with synchronized refresh.
 *
 * @example
 * ```tsx
 * const dashboard = useDashboardData();
 * if (dashboard.candles.loading) return <Loading />;
 * ```
 */
export function useDashboardData(
  config: ApiConfig = {},
  refresh: AutoRefreshOptions = {},
): DashboardData {
  return {
    candles: useCandles(config, refresh),
    indicators: useIndicators(config, refresh),
    factors: useFactors(config, refresh),
    agents: useAgents(config, refresh),
    risk: useRiskMetrics(config, refresh),
    news: useActiveNews(config, refresh),
    orderDistribution: useOrderDistribution(config, refresh),
  };
}
