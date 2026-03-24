/**
 * News feed component (V4.4).
 *
 * Displays active news events affecting the simulation.
 * Shows impact magnitude, affected symbols, and remaining duration.
 *
 * Declarative: Data-driven from ActiveNewsResponse
 * Modular: Self-contained news display
 * SoC: Only handles news visualization
 */

import type { ActiveNewsResponse, NewsEventData } from '../../types/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface NewsFeedProps {
  /** News data from API. */
  data: ActiveNewsResponse | null;
  /** Loading state. */
  loading?: boolean;
  /** Error state. */
  error?: Error | null;
  /** Maximum events to display. */
  maxEvents?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Get impact color based on magnitude. */
function getImpactColor(impact: number): string {
  const absImpact = Math.abs(impact);
  if (absImpact < 0.01) return 'text-gray-400';
  if (absImpact < 0.03) return impact > 0 ? 'text-green-400' : 'text-red-400';
  return impact > 0 ? 'text-green-500' : 'text-red-500';
}

/** Get impact background for badge. */
function getImpactBg(impact: number): string {
  const absImpact = Math.abs(impact);
  if (absImpact < 0.01) return 'bg-gray-700';
  if (impact > 0) return 'bg-green-500/20';
  return 'bg-red-500/20';
}

/** Format impact as percentage with sign. */
function formatImpact(impact: number): string {
  const sign = impact >= 0 ? '+' : '';
  return `${sign}${(impact * 100).toFixed(2)}%`;
}

/** Get event type icon. */
function getEventIcon(eventType: string): string {
  switch (eventType.toLowerCase()) {
    case 'earnings':
      return '📊';
    case 'macro':
      return '🏛️';
    case 'sector':
      return '🏭';
    case 'geopolitical':
      return '🌍';
    case 'technical':
      return '📈';
    default:
      return '📰';
  }
}

/** Calculate remaining ticks as percentage. */
function getProgressPercent(startTick: number, duration: number, currentTick: number): number {
  if (duration === 0) return 100;
  const elapsed = currentTick - startTick;
  const remaining = Math.max(0, duration - elapsed);
  return Math.max(0, Math.min(100, (remaining / duration) * 100));
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface NewsCardProps {
  event: NewsEventData;
  currentTick: number;
}

function NewsCard({ event, currentTick }: NewsCardProps) {
  const elapsed = currentTick - event.start_tick;
  const remainingTicks = Math.max(0, event.duration_ticks - elapsed);
  const progress = getProgressPercent(event.start_tick, event.duration_ticks, currentTick);
  const affectedSymbol = event.symbol || event.sector || 'Market';

  return (
    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700 hover:border-gray-600 transition-colors">
      {/* Header */}
      <div className="flex items-start gap-2 mb-2">
        <span className="text-lg">{getEventIcon(event.event_type)}</span>
        <div className="flex-1 min-w-0">
          <h4 className="text-white text-sm font-medium truncate">{event.headline}</h4>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-gray-500 text-xs uppercase">{event.event_type}</span>
            <span className="text-gray-400 text-xs">• {affectedSymbol}</span>
          </div>
        </div>

        {/* Impact badge */}
        <div
          className={`px-2 py-0.5 rounded text-xs font-mono ${getImpactBg(event.impact)} ${getImpactColor(event.impact)}`}
        >
          {formatImpact(event.impact)}
        </div>
      </div>

      {/* Duration progress bar */}
      <div className="relative">
        <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-primary-500 transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>{remainingTicks} ticks left</span>
          <span>Started tick {event.start_tick}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function NewsFeed({ data, loading = false, error = null, maxEvents = 5 }: NewsFeedProps) {
  // Loading state
  if (loading && !data) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 h-full">
        <div className="text-gray-400 animate-pulse">Loading news...</div>
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

  // No data or empty events
  if (!data || data.events.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 h-full">
        <h3 className="text-white font-semibold mb-2">News Feed</h3>
        <div className="flex items-center justify-center py-8">
          <div className="text-center">
            <span className="text-2xl mb-2 block">📰</span>
            <p className="text-gray-500 text-sm">No active news events</p>
          </div>
        </div>
      </div>
    );
  }

  // Sort by impact magnitude and limit
  const sortedEvents = [...data.events]
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, maxEvents);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 h-full">
      {/* Header */}
      <div className="flex justify-between items-center mb-3">
        <div>
          <h3 className="text-white font-semibold">News Feed</h3>
          <p className="text-gray-500 text-xs">Active market events</p>
        </div>
        <div className="flex items-center gap-1">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
          </span>
          <span className="text-green-400 text-xs">{data.events.length} active</span>
        </div>
      </div>

      {/* Event cards */}
      <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
        {sortedEvents.map((event) => (
          <NewsCard
            key={`${event.headline}-${event.start_tick}`}
            event={event}
            currentTick={data.tick}
          />
        ))}
      </div>

      {/* Show more indicator */}
      {data.events.length > maxEvents && (
        <div className="text-center text-gray-500 text-xs mt-2 pt-2 border-t border-gray-800">
          +{data.events.length - maxEvents} more events
        </div>
      )}
    </div>
  );
}
