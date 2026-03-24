/**
 * Risk metrics panel component (V4.4).
 *
 * Displays portfolio risk metrics including VaR, drawdown, volatility, Sharpe ratio.
 * Uses visual indicators for risk levels.
 *
 * Declarative: Data-driven from RiskMetricsResponse
 * Modular: Self-contained risk display
 * SoC: Only handles risk visualization
 */

import type { RiskMetricsResponse } from '../../types/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface RiskPanelProps {
  /** Risk metrics data from API. */
  data: RiskMetricsResponse | null;
  /** Loading state. */
  loading?: boolean;
  /** Error state. */
  error?: Error | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format percentage value. */
function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return (value * 100).toFixed(2) + '%';
}

/** Format ratio value. */
function formatRatio(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return value.toFixed(3);
}

/** Get risk level indicator. */
function getRiskLevel(var95: number | null | undefined): 'low' | 'medium' | 'high' {
  if (var95 === null || var95 === undefined) return 'medium';
  if (var95 < 0.02) return 'low';
  if (var95 < 0.05) return 'medium';
  return 'high';
}

/** Get color for risk level. */
function getRiskColor(level: 'low' | 'medium' | 'high'): string {
  switch (level) {
    case 'low':
      return 'text-green-400';
    case 'medium':
      return 'text-yellow-400';
    case 'high':
      return 'text-red-400';
  }
}

/** Get background for risk level. */
function getRiskBg(level: 'low' | 'medium' | 'high'): string {
  switch (level) {
    case 'low':
      return 'bg-green-500/20 border-green-500/50';
    case 'medium':
      return 'bg-yellow-500/20 border-yellow-500/50';
    case 'high':
      return 'bg-red-500/20 border-red-500/50';
  }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface MetricRowProps {
  label: string;
  value: string;
  subValue?: string;
  variant?: 'normal' | 'highlight';
}

function MetricRow({ label, value, subValue, variant = 'normal' }: MetricRowProps) {
  return (
    <div
      className={`flex justify-between items-center py-2 px-3 rounded ${
        variant === 'highlight' ? 'bg-gray-800/50' : ''
      }`}
    >
      <span className="text-gray-400 text-sm">{label}</span>
      <div className="text-right">
        <span className="font-mono text-white text-sm">{value}</span>
        {subValue && <div className="text-gray-500 text-xs">{subValue}</div>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function RiskPanel({ data, loading = false, error = null }: RiskPanelProps) {
  // Loading state
  if (loading && !data) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 h-full">
        <div className="text-gray-400 animate-pulse">Loading risk metrics...</div>
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
        <div className="text-gray-500">No risk data</div>
      </div>
    );
  }

  const riskLevel = getRiskLevel(data.var_95);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 h-full">
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-white font-semibold">Risk Metrics</h3>
          <p className="text-gray-500 text-xs">Tick {data.tick}</p>
        </div>

        {/* Risk level badge */}
        <div
          className={`px-2 py-1 rounded border text-xs uppercase tracking-wider ${getRiskBg(riskLevel)} ${getRiskColor(riskLevel)}`}
        >
          {riskLevel} risk
        </div>
      </div>

      {/* Metrics */}
      <div className="space-y-1">
        {/* Value at Risk */}
        <MetricRow
          label="VaR (95%)"
          value={formatPercent(data.var_95)}
          subValue="1-day 95% confidence"
          variant="highlight"
        />

        {/* Max Drawdown */}
        <MetricRow
          label="Max Drawdown"
          value={formatPercent(data.max_drawdown)}
          variant="highlight"
        />

        {/* Current Drawdown */}
        <MetricRow label="Current Drawdown" value={formatPercent(data.current_drawdown)} />

        {/* Volatility */}
        <MetricRow label="Volatility (Ann.)" value={formatPercent(data.volatility)} />

        {/* Sharpe Ratio */}
        <MetricRow
          label="Sharpe Ratio"
          value={formatRatio(data.sharpe_ratio)}
          variant="highlight"
        />

        {/* Sortino Ratio */}
        <MetricRow label="Sortino Ratio" value={formatRatio(data.sortino_ratio)} />
      </div>
    </div>
  );
}
