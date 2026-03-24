/**
 * Agent table component (V4.4).
 *
 * Displays all agents with sortable columns.
 * Shows ID, name, cash, equity, position count, PnL, tier.
 *
 * Declarative: Data-driven from AgentsResponse
 * Modular: Self-contained agent display
 * SoC: Only handles agent visualization
 */

import { useState, useMemo } from 'react';
import type { AgentsResponse, AgentData } from '../../types/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AgentTableProps {
  /** Agent data from API. */
  data: AgentsResponse | null;
  /** Loading state. */
  loading?: boolean;
  /** Error state. */
  error?: Error | null;
  /** Maximum agents to display per page. */
  pageSize?: number;
  /** Selected symbol to show position for. */
  selectedSymbol?: string;
}

type SortField = 'agent_id' | 'name' | 'cash' | 'equity' | 'total_pnl' | 'tier';
type SortDirection = 'asc' | 'desc';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format currency value. */
function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  if (Math.abs(value) >= 1_000_000) {
    return (value / 1_000_000).toFixed(2) + 'M';
  }
  if (Math.abs(value) >= 1_000) {
    return (value / 1_000).toFixed(2) + 'K';
  }
  return value.toFixed(2);
}

/** Get PnL color class. */
function getPnLColor(pnl: number | null | undefined): string {
  if (pnl === null || pnl === undefined) return 'text-gray-400';
  if (pnl > 0) return 'text-green-400';
  if (pnl < 0) return 'text-red-400';
  return 'text-gray-400';
}

/** Get PnL background for highlight. */
function getPnLBg(pnl: number | null | undefined): string {
  if (pnl === null || pnl === undefined) return '';
  if (pnl > 0) return 'bg-green-500/10';
  if (pnl < 0) return 'bg-red-500/10';
  return '';
}

/** Get position color (like TUI). */
function getPositionColor(position: number): string {
  if (position > 0) return 'text-green-400';
  if (position < 0) return 'text-red-400';
  return 'text-gray-400';
}

/** Get tier badge color. */
function getTierBadge(tier: number): { bg: string; text: string; label: string } {
  switch (tier) {
    case 1:
      return { bg: 'bg-yellow-500/20', text: 'text-yellow-400', label: 'T1' };
    case 2:
      return { bg: 'bg-blue-500/20', text: 'text-blue-400', label: 'T2' };
    case 3:
      return { bg: 'bg-purple-500/20', text: 'text-purple-400', label: 'T3' };
    default:
      return { bg: 'bg-gray-500/20', text: 'text-gray-400', label: `T${tier}` };
  }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface SortHeaderProps {
  label: string;
  field: SortField;
  currentSort: SortField;
  direction: SortDirection;
  onSort: (field: SortField) => void;
}

function SortHeader({ label, field, currentSort, direction, onSort }: SortHeaderProps) {
  const isActive = currentSort === field;

  return (
    <th
      className="px-3 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white transition-colors"
      onClick={() => onSort(field)}
    >
      <div className="flex items-center gap-1">
        {label}
        <span className="text-gray-600">{isActive ? (direction === 'asc' ? '↑' : '↓') : '↕'}</span>
      </div>
    </th>
  );
}

interface AgentRowProps {
  agent: AgentData;
  rank: number;
  selectedSymbol?: string;
}

function AgentRow({ agent, rank, selectedSymbol }: AgentRowProps) {
  const tierBadge = getTierBadge(agent.tier);

  // Get position for selected symbol only
  const position = selectedSymbol ? (agent.positions?.[selectedSymbol] ?? null) : null;

  return (
    <tr className={`border-b border-gray-800 hover:bg-gray-800/50 ${getPnLBg(agent.total_pnl)}`}>
      {/* Rank */}
      <td className="px-3 py-2 text-gray-500 text-sm">{rank}</td>

      {/* ID */}
      <td className="px-3 py-2 font-mono text-sm text-white">{agent.agent_id}</td>

      {/* Name */}
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="text-gray-300 text-sm">{agent.name}</span>
          {agent.is_market_maker && (
            <span className="inline-flex px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-400 text-xs">
              MM
            </span>
          )}
        </div>
      </td>

      {/* Tier */}
      <td className="px-3 py-2">
        <span
          className={`inline-flex px-2 py-0.5 rounded ${tierBadge.bg} ${tierBadge.text} text-xs`}
        >
          {tierBadge.label}
        </span>
      </td>

      {/* Cash */}
      <td className="px-3 py-2 font-mono text-sm text-gray-300 text-right">
        ${formatCurrency(agent.cash)}
      </td>

      {/* Equity */}
      <td className="px-3 py-2 font-mono text-sm text-gray-300 text-right">
        ${formatCurrency(agent.equity)}
      </td>

      {/* Positions - show qty for selected symbol */}
      <td className="px-3 py-2 font-mono text-sm text-center">
        {position !== null ? (
          <span className={getPositionColor(position)}>
            {position > 0 ? '+' : ''}
            {position}
          </span>
        ) : (
          <span className="text-gray-600">—</span>
        )}
      </td>

      {/* Total PnL */}
      <td
        className={`px-3 py-2 font-mono text-sm text-right font-semibold ${getPnLColor(agent.total_pnl)}`}
      >
        {(agent.total_pnl ?? 0) >= 0 ? '+' : ''}${formatCurrency(agent.total_pnl)}
      </td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function AgentTable({
  data,
  loading = false,
  error = null,
  pageSize = 20,
  selectedSymbol,
}: AgentTableProps) {
  const [sortField, setSortField] = useState<SortField>('total_pnl');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [page, setPage] = useState(0);

  // Sort and paginate agents
  // Primary sort: ML agents at top, market makers at bottom
  // Secondary sort: by selected field
  const sortedAgents = useMemo(() => {
    if (!data?.agents) return [];

    const sorted = [...data.agents].sort((a, b) => {
      // ML agents always at top
      if (a.is_ml_agent && !b.is_ml_agent) return -1;
      if (!a.is_ml_agent && b.is_ml_agent) return 1;

      // Market makers always at bottom
      if (a.is_market_maker && !b.is_market_maker) return 1;
      if (!a.is_market_maker && b.is_market_maker) return -1;

      // Within same category, sort by selected field
      const aVal = a[sortField as keyof AgentData];
      const bVal = b[sortField as keyof AgentData];

      // Handle string comparison
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }

      // Handle boolean comparison
      if (typeof aVal === 'boolean' && typeof bVal === 'boolean') {
        return sortDirection === 'asc' ? Number(aVal) - Number(bVal) : Number(bVal) - Number(aVal);
      }

      // Numeric comparison
      const aNum = typeof aVal === 'number' ? aVal : 0;
      const bNum = typeof bVal === 'number' ? bVal : 0;

      return sortDirection === 'asc' ? aNum - bNum : bNum - aNum;
    });

    return sorted;
  }, [data, sortField, sortDirection]);

  // Paginate
  const totalPages = Math.ceil(sortedAgents.length / pageSize);
  const paginatedAgents = sortedAgents.slice(page * pageSize, (page + 1) * pageSize);

  // Handle sort click
  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
    setPage(0);
  };

  // Loading state
  if (loading && !data) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
        <div className="text-gray-400 animate-pulse">Loading agents...</div>
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

  // No data
  if (!data || data.agents.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
        <div className="text-gray-500">No agent data</div>
      </div>
    );
  }

  // Calculate summary stats
  const totalPnL = data.agents.reduce((sum, a) => sum + (a.total_pnl ?? 0), 0);
  const profitableAgents = data.agents.filter((a) => (a.total_pnl ?? 0) > 0).length;
  const marketMakers = data.agents.filter((a) => a.is_market_maker).length;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-700">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-white font-semibold">Agent Explorer</h3>
            <p className="text-gray-500 text-xs">
              {data.total_count.toLocaleString()} agents • Tick {data.tick}
            </p>
          </div>
          <div className="flex gap-4 text-sm">
            <div className="text-center">
              <div className={`font-mono ${getPnLColor(totalPnL)}`}>
                {totalPnL >= 0 ? '+' : ''}${formatCurrency(totalPnL)}
              </div>
              <div className="text-gray-500 text-xs">Total PnL</div>
            </div>
            <div className="text-center">
              <div className="text-white font-mono">
                {profitableAgents}/{data.agents.length}
              </div>
              <div className="text-gray-500 text-xs">Profitable</div>
            </div>
            <div className="text-center">
              <div className="text-orange-400 font-mono">{marketMakers}</div>
              <div className="text-gray-500 text-xs">MMs</div>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-800/50">
            <tr>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                #
              </th>
              <SortHeader
                label="ID"
                field="agent_id"
                currentSort={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <SortHeader
                label="Name"
                field="name"
                currentSort={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <SortHeader
                label="Tier"
                field="tier"
                currentSort={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <SortHeader
                label="Cash"
                field="cash"
                currentSort={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <SortHeader
                label="Equity"
                field="equity"
                currentSort={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <th className="px-3 py-2 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                Position
              </th>
              <SortHeader
                label="Total PnL"
                field="total_pnl"
                currentSort={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
            </tr>
          </thead>
          <tbody>
            {paginatedAgents.map((agent, idx) => (
              <AgentRow
                key={agent.agent_id}
                agent={agent}
                rank={page * pageSize + idx + 1}
                selectedSymbol={selectedSymbol}
              />
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="px-4 py-3 border-t border-gray-700 flex justify-between items-center">
          <button
            className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed text-white"
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
          >
            Previous
          </button>
          <span className="text-gray-400 text-sm">
            Page {page + 1} of {totalPages}
          </span>
          <button
            className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed text-white"
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
