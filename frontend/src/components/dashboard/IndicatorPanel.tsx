/**
 * Technical indicator panel component (V4.4).
 *
 * Displays indicator values with sparkline-style visualization.
 * Shows SMA, EMA, RSI, MACD, Bollinger Bands, ATR.
 *
 * Declarative: Data-driven from IndicatorsResponse
 * Modular: Self-contained indicator display
 * SoC: Only handles indicator visualization
 */

import type { IndicatorsResponse } from '../../types/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface IndicatorPanelProps {
  /** Indicator data from API. */
  data: IndicatorsResponse | null;
  /** Loading state. */
  loading?: boolean;
  /** Error state. */
  error?: Error | null;
}

interface IndicatorRowProps {
  name: string;
  value: number | null | undefined;
  format?: 'price' | 'percent' | 'number';
  trend?: 'bullish' | 'bearish' | 'neutral';
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatValue(value: number | null | undefined, format: string): string {
  if (value === null || value === undefined) return '—';

  switch (format) {
    case 'price':
      return value.toFixed(2);
    case 'percent':
      return value.toFixed(1) + '%';
    case 'number':
    default:
      return value.toFixed(2);
  }
}

function getTrendColor(trend: string | undefined): string {
  switch (trend) {
    case 'bullish':
      return 'text-green-500';
    case 'bearish':
      return 'text-red-500';
    default:
      return 'text-gray-300';
  }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function IndicatorRow({ name, value, format = 'number', trend }: IndicatorRowProps) {
  return (
    <div className="flex justify-between items-center py-1.5 border-b border-gray-800 last:border-0">
      <span className="text-gray-400 text-sm">{name}</span>
      <span className={`font-mono text-sm ${getTrendColor(trend)}`}>
        {formatValue(value, format)}
      </span>
    </div>
  );
}

function IndicatorGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4 last:mb-0">
      <h4 className="text-gray-500 text-xs uppercase tracking-wider mb-2">{title}</h4>
      <div className="bg-gray-800/50 rounded-md px-3 py-1">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function IndicatorPanel({ data, loading = false, error = null }: IndicatorPanelProps) {
  // Loading state
  if (loading && !data) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 h-full">
        <div className="text-gray-400 animate-pulse">Loading indicators...</div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-gray-900 rounded-lg border border-red-700 p-4 h-full">
        <div className="text-red-400 text-sm">Error: {error.message}</div>
      </div>
    );
  }

  // No data state
  if (!data) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 h-full">
        <div className="text-gray-500">No indicator data</div>
      </div>
    );
  }

  // Extract nested data
  const ind = data.indicators;
  const rsi = ind.rsi_8;
  const macdData = ind.macd;

  // Determine RSI trend
  const rsiTrend =
    rsi !== null && rsi !== undefined
      ? rsi > 70
        ? 'bearish'
        : rsi < 30
          ? 'bullish'
          : 'neutral'
      : 'neutral';

  // Determine MACD trend
  const macdTrend =
    macdData?.histogram !== null && macdData?.histogram !== undefined
      ? macdData.histogram > 0
        ? 'bullish'
        : 'bearish'
      : 'neutral';

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 h-full">
      <h3 className="text-white font-semibold mb-3">Technical Indicators</h3>
      <p className="text-gray-500 text-xs mb-3">
        {data.symbol} • Tick {data.tick}
      </p>

      {/* Moving Averages */}
      <IndicatorGroup title="Moving Averages">
        <IndicatorRow name="SMA-8" value={ind.sma['8']} format="price" />
        <IndicatorRow name="SMA-16" value={ind.sma['16']} format="price" />
        <IndicatorRow name="EMA-8" value={ind.ema['8']} format="price" />
        <IndicatorRow name="EMA-16" value={ind.ema['16']} format="price" />
      </IndicatorGroup>

      {/* Oscillators */}
      <IndicatorGroup title="Oscillators">
        <IndicatorRow name="RSI-8" value={rsi} format="number" trend={rsiTrend} />
      </IndicatorGroup>

      {/* MACD */}
      <IndicatorGroup title="MACD">
        <IndicatorRow name="MACD Line" value={macdData?.macd_line} format="number" />
        <IndicatorRow name="Signal" value={macdData?.signal_line} format="number" />
        <IndicatorRow
          name="Histogram"
          value={macdData?.histogram}
          format="number"
          trend={macdTrend}
        />
      </IndicatorGroup>

      {/* Bollinger Bands */}
      <IndicatorGroup title="Bollinger Bands">
        <IndicatorRow name="Upper" value={ind.bollinger?.upper} format="price" />
        <IndicatorRow name="Middle" value={ind.bollinger?.middle} format="price" />
        <IndicatorRow name="Lower" value={ind.bollinger?.lower} format="price" />
      </IndicatorGroup>

      {/* Volatility */}
      <IndicatorGroup title="Volatility">
        <IndicatorRow name="ATR-8" value={ind.atr_8} format="number" />
      </IndicatorGroup>
    </div>
  );
}
