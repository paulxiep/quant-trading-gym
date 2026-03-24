/**
 * Order depth chart component (V4.4).
 *
 * Visualizes pre-auction bid/ask distribution captured before batch auction.
 * Shows order flow dynamics that would be invisible at tick-end (post-clearing).
 *
 * Declarative: Data-driven from OrderDistributionResponse
 * Modular: Self-contained depth visualization
 * SoC: Only handles order distribution display
 */

import type { OrderDistributionResponse, PriceLevel } from '../../types/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface OrderDepthChartProps {
  /** Order distribution data from API. */
  data: OrderDistributionResponse | null;
  /** Loading state. */
  loading?: boolean;
  /** Error state. */
  error?: Error | null;
  /** Chart height in pixels. */
  height?: number;
  /** Maximum price levels to show per side. */
  maxLevels?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Find max quantity for scaling. */
function getMaxQuantity(bids: PriceLevel[], asks: PriceLevel[]): number {
  let max = 0;
  for (const [, qty] of bids) {
    if (qty > max) max = qty;
  }
  for (const [, qty] of asks) {
    if (qty > max) max = qty;
  }
  return max || 1;
}

/** Format quantity for display. */
function formatQuantity(qty: number): string {
  if (qty >= 1_000_000) return (qty / 1_000_000).toFixed(1) + 'M';
  if (qty >= 1_000) return (qty / 1_000).toFixed(1) + 'K';
  return qty.toString();
}

/** Format price for display. */
function formatPrice(price: number): string {
  return price.toFixed(2);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface DepthBarProps {
  price: number;
  quantity: number;
  maxQuantity: number;
  side: 'bid' | 'ask';
}

function DepthBar({ price, quantity, maxQuantity, side }: DepthBarProps) {
  const widthPercent = Math.min(100, (quantity / maxQuantity) * 100);
  const isBid = side === 'bid';

  return (
    <div className="flex items-center h-6 gap-2">
      {/* Price label */}
      <div className="w-16 text-right">
        <span className={`font-mono text-xs ${isBid ? 'text-green-400' : 'text-red-400'}`}>
          {formatPrice(price)}
        </span>
      </div>

      {/* Bar */}
      <div className="flex-1 h-4 bg-gray-800 rounded relative overflow-hidden">
        <div
          className={`absolute h-full transition-all duration-200 ${
            isBid ? 'bg-green-600/60 left-0' : 'bg-red-600/60 left-0'
          }`}
          style={{ width: `${widthPercent}%` }}
        />
        <div className="absolute inset-0 flex items-center justify-end px-2">
          <span className="text-xs text-gray-300 font-mono">{formatQuantity(quantity)}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function OrderDepthChart({
  data,
  loading = false,
  error = null,
  height = 300,
  maxLevels = 10,
}: OrderDepthChartProps) {
  // Loading state
  if (loading && !data) {
    return (
      <div
        className="bg-gray-900 rounded-lg border border-gray-700 flex items-center justify-center"
        style={{ minHeight: height }}
      >
        <div className="text-gray-400 animate-pulse">Loading order depth...</div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div
        className="bg-gray-900 rounded-lg border border-red-700 flex items-center justify-center"
        style={{ minHeight: height }}
      >
        <div className="text-red-400">Error: {error.message}</div>
      </div>
    );
  }

  // No data state
  if (!data) {
    return (
      <div
        className="bg-gray-900 rounded-lg border border-gray-700 flex items-center justify-center"
        style={{ minHeight: height }}
      >
        <div className="text-gray-500">No order distribution data</div>
      </div>
    );
  }

  // Limit levels and calculate max
  const bids: PriceLevel[] = data.bids.slice(0, maxLevels);
  const asks: PriceLevel[] = data.asks.slice(0, maxLevels);
  const maxQty = getMaxQuantity(bids, asks);

  // Calculate totals
  const totalBidQty = bids.reduce((sum: number, [, qty]: PriceLevel) => sum + qty, 0);
  const totalAskQty = asks.reduce((sum: number, [, qty]: PriceLevel) => sum + qty, 0);
  const imbalance = totalBidQty - totalAskQty;
  const imbalancePercent = ((totalBidQty - totalAskQty) / (totalBidQty + totalAskQty || 1)) * 100;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
      {/* Header */}
      <div className="flex justify-between items-center mb-3">
        <div>
          <h3 className="text-white font-semibold">Order Depth</h3>
          <p className="text-gray-500 text-xs">Pre-auction order distribution</p>
        </div>
        <div className="text-right">
          <div
            className={`text-sm font-mono ${
              imbalance > 0 ? 'text-green-400' : imbalance < 0 ? 'text-red-400' : 'text-gray-400'
            }`}
          >
            {imbalance > 0 ? '+' : ''}
            {formatQuantity(imbalance)}
          </div>
          <div className="text-xs text-gray-500">Imbalance: {imbalancePercent.toFixed(1)}%</div>
        </div>
      </div>

      {/* Summary bar */}
      <div className="flex h-3 rounded overflow-hidden mb-4">
        <div
          className="bg-green-600 transition-all duration-300"
          style={{ width: `${(totalBidQty / (totalBidQty + totalAskQty || 1)) * 100}%` }}
        />
        <div
          className="bg-red-600 transition-all duration-300"
          style={{ width: `${(totalAskQty / (totalBidQty + totalAskQty || 1)) * 100}%` }}
        />
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-2 gap-4">
        {/* Bids (buy orders) */}
        <div>
          <div className="flex justify-between text-xs text-gray-500 mb-2">
            <span>BIDS</span>
            <span>{formatQuantity(totalBidQty)} total</span>
          </div>
          <div className="space-y-1">
            {bids.length > 0 ? (
              bids.map(([price, qty]: PriceLevel) => (
                <DepthBar
                  key={price}
                  price={price}
                  quantity={qty}
                  maxQuantity={maxQty}
                  side="bid"
                />
              ))
            ) : (
              <div className="text-gray-500 text-sm py-2">No bids</div>
            )}
          </div>
        </div>

        {/* Asks (sell orders) */}
        <div>
          <div className="flex justify-between text-xs text-gray-500 mb-2">
            <span>ASKS</span>
            <span>{formatQuantity(totalAskQty)} total</span>
          </div>
          <div className="space-y-1">
            {asks.length > 0 ? (
              asks.map(([price, qty]: PriceLevel) => (
                <DepthBar
                  key={price}
                  price={price}
                  quantity={qty}
                  maxQuantity={maxQty}
                  side="ask"
                />
              ))
            ) : (
              <div className="text-gray-500 text-sm py-2">No asks</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
