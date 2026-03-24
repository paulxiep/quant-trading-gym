/**
 * Symbol selector component (V4.4).
 *
 * Dropdown for selecting the active trading symbol.
 * Fetches available symbols from the candles API.
 *
 * Declarative: Controlled by parent via value/onChange
 * Modular: Self-contained symbol selection
 * SoC: Only handles symbol selection UI
 */

import { useState, useEffect } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SymbolSelectorProps {
  /** Currently selected symbol. */
  value: string;
  /** Callback when symbol changes. */
  onChange: (symbol: string) => void;
  /** Base URL for API (default: http://localhost:8001). */
  baseUrl?: string;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function SymbolSelector({
  value,
  onChange,
  baseUrl = 'http://localhost:8001',
}: SymbolSelectorProps) {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch available symbols from the symbols endpoint, polling until available
  useEffect(() => {
    let cancelled = false;
    let symbolsLoaded = false;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    async function fetchSymbols() {
      if (symbolsLoaded || cancelled) return;

      try {
        const response = await fetch(`${baseUrl}/api/symbols`);
        if (response.ok && !cancelled) {
          const data = await response.json();
          const symbolList: string[] = data.symbols || [];

          if (symbolList.length > 0) {
            symbolsLoaded = true;
            setSymbols(symbolList);

            // If value is empty, set it to the first available symbol
            if (!value) {
              onChange(symbolList[0]);
            }

            // Stop polling
            if (intervalId) {
              clearInterval(intervalId);
              intervalId = null;
            }
          }
        }
      } catch (error) {
        console.error('Failed to fetch symbols:', error);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchSymbols();

    // Poll until symbols are available (stops when symbolsLoaded becomes true)
    intervalId = setInterval(fetchSymbols, 1000);

    return () => {
      cancelled = true;
      symbolsLoaded = true;
      if (intervalId) clearInterval(intervalId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl]);

  // Always show dropdown when there are symbols (even if just one)
  return (
    <div className="flex items-center gap-2">
      <span className="text-gray-400 text-sm">Symbol:</span>
      {loading ? (
        <span className="text-gray-500 text-sm">Loading...</span>
      ) : symbols.length > 0 ? (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="bg-gray-800 text-white text-sm rounded px-2 py-1 border border-gray-700 focus:outline-none focus:border-primary-500"
        >
          {symbols.map((sym) => (
            <option key={sym} value={sym}>
              {sym}
            </option>
          ))}
        </select>
      ) : (
        <span className="text-white font-medium">{value || 'Unknown'}</span>
      )}
    </div>
  );
}
