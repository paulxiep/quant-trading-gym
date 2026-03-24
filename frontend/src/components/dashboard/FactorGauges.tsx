/**
 * Factor gauges component (V4.4).
 *
 * Displays macro factor values as visual gauge indicators.
 * Shows interest rate, volatility regime, momentum factor, market sentiment.
 *
 * Declarative: Data-driven from FactorsResponse
 * Modular: Self-contained factor display
 * SoC: Only handles factor visualization
 */

import type { FactorsResponse, FactorSnapshot } from '../../types/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FactorGaugesProps {
  /** Factor data from API. */
  data: FactorsResponse | null;
  /** Loading state. */
  loading?: boolean;
  /** Error state. */
  error?: Error | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Get color class based on value and thresholds. */
function getValueColor(value: number, low: number, high: number): string {
  if (value <= low) return 'text-red-500';
  if (value >= high) return 'text-green-500';
  return 'text-yellow-500';
}

/** Get gauge fill color based on value. */
function getGaugeColor(value: number, low: number, high: number): string {
  if (value <= low) return 'bg-red-500';
  if (value >= high) return 'bg-green-500';
  return 'bg-yellow-500';
}

/** Normalize value to 0-100 range for gauge display. */
function normalizeValue(value: number, min: number, max: number): number {
  return Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface GaugeProps {
  factor: FactorSnapshot;
}

function Gauge({ factor }: GaugeProps) {
  // Determine range based on factor type
  let min = -1;
  let max = 1;
  let low = -0.3;
  let high = 0.3;
  let format = (v: number) => v.toFixed(3);

  // Adjust ranges for specific factor types
  if (factor.name.toLowerCase().includes('rate')) {
    min = 0;
    max = 0.1;
    low = 0.02;
    high = 0.05;
    format = (v) => (v * 100).toFixed(2) + '%';
  } else if (factor.name.toLowerCase().includes('volatility')) {
    min = 0;
    max = 0.5;
    low = 0.1;
    high = 0.3;
    format = (v) => (v * 100).toFixed(1) + '%';
  }

  const normalized = normalizeValue(factor.value, min, max);

  return (
    <div className="bg-gray-800/50 rounded-lg p-3">
      <div className="flex justify-between items-center mb-2">
        <span className="text-gray-400 text-sm">{factor.name}</span>
        <span className={`font-mono text-sm ${getValueColor(factor.value, low, high)}`}>
          {format(factor.value)}
        </span>
      </div>

      {/* Gauge bar */}
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${getGaugeColor(factor.value, low, high)}`}
          style={{ width: `${normalized}%` }}
        />
      </div>

      {/* Min/Max labels */}
      <div className="flex justify-between text-xs text-gray-600 mt-1">
        <span>{format(min)}</span>
        <span>{format(max)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function FactorGauges({ data, loading = false, error = null }: FactorGaugesProps) {
  // Loading state
  if (loading && !data) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
        <div className="text-gray-400 animate-pulse">Loading factors...</div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-gray-900 rounded-lg border border-red-700 p-4">
        <div className="text-red-400 text-sm">Error: {error.message}</div>
      </div>
    );
  }

  // No data state
  if (!data || data.factors.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
        <div className="text-gray-500">No factor data</div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
      <h3 className="text-white font-semibold mb-1">Macro Factors</h3>
      <p className="text-gray-500 text-xs mb-3">Tick {data.tick}</p>

      <div className="grid grid-cols-1 gap-3">
        {data.factors.map((factor) => (
          <Gauge key={factor.name} factor={factor} />
        ))}
      </div>
    </div>
  );
}
