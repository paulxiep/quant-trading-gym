/**
 * Data Service API response types (V4.4).
 *
 * TypeScript interfaces matching Rust server responses.
 * Used by useDataService hooks and simulation components.
 *
 * # Design Principles
 *
 * - **Declarative**: Types describe shape, pure data contracts
 * - **Modular**: Grouped by domain (analytics, portfolio, risk, news)
 * - **SoC**: All API types in one place, separate from UI types
 */

// =============================================================================
// Analytics Types
// =============================================================================

/** OHLCV candle data. */
export interface CandleData {
  symbol: string;
  tick: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/** Single candle for chart rendering. */
export interface Candle {
  tick: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/** Response for GET /api/candles */
export interface CandlesResponse {
  symbol: string;
  candles: Candle[];
  total: number;
}

/** MACD indicator data. */
export interface MacdData {
  macd_line: number;
  signal_line: number;
  histogram: number;
}

/** Bollinger Bands data. */
export interface BollingerData {
  upper: number;
  middle: number;
  lower: number;
}

/** MACD data from server. */
export interface MacdData {
  macd_line: number;
  signal_line: number;
  histogram: number;
}

/** Technical indicator values - nested structure matching server response. */
export interface IndicatorsResponse {
  symbol: string;
  tick: number;
  indicators: {
    sma: Record<string, number | null>;
    ema: Record<string, number | null>;
    rsi_8: number | null;
    macd: MacdData | null;
    bollinger: BollingerData | null;
    atr_8: number | null;
  };
}

/** Factor snapshot for gauge display. */
export interface FactorSnapshot {
  name: string;
  value: number;
  min: number;
  max: number;
  neutral: number;
}

/** Response for GET /api/factors */
export interface FactorsResponse {
  symbol: string;
  tick: number;
  factors: FactorSnapshot[];
}

// =============================================================================
// Portfolio Types
// =============================================================================

/** Agent position in a symbol. */
export interface AgentPosition {
  symbol: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
}

/** Agent data for explorer table (matches server AgentSummary). */
export interface AgentData {
  agent_id: number;
  name: string;
  total_pnl: number;
  cash: number;
  equity: number;
  /** Per-symbol positions (only non-zero). */
  positions: Record<string, number>;
  is_market_maker: boolean;
  is_ml_agent: boolean;
  tier: number;
}

/** Response for GET /api/agents */
export interface AgentsResponse {
  agents: AgentData[];
  total_count: number;
  tick: number;
}

/** Response for GET /api/portfolio/agents/:agent_id */
export interface AgentPortfolioResponse {
  agent_id: number;
  name: string;
  cash: number;
  equity: number;
  total_pnl: number;
  realized_pnl: number;
  unrealized_pnl: number;
  positions: AgentPosition[];
  equity_curve: number[];
  tick: number;
}

// =============================================================================
// Risk Types
// =============================================================================

/** Response for GET /api/risk/:agent_id (per-agent) */
export interface AgentRiskMetricsResponse {
  agent_id: number;
  name: string;
  var_95: number | null;
  var_99: number | null;
  current_drawdown: number;
  max_drawdown: number;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  beta: number | null;
  volatility: number | null;
  total_return: number;
  equity: number;
  tick: number;
}

/** Response for GET /api/risk/aggregate (portfolio-wide) */
export interface RiskMetricsResponse {
  var_95: number | null;
  var_99: number | null;
  current_drawdown: number;
  max_drawdown: number;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  volatility: number | null;
  tick: number;
}

// =============================================================================
// News Types
// =============================================================================

/** News event snapshot for feed. */
export interface NewsEventSnapshot {
  id: number;
  headline: string;
  event_type: string;
  symbol: string | null;
  sector: string | null;
  sentiment: number;
  magnitude: number;
  impact: number;
  start_tick: number;
  duration_ticks: number;
  ticks_remaining: number;
}

/** News event data. */
export interface NewsEventData {
  id: number;
  headline: string;
  event_type: string;
  symbol: string | null;
  sector: string | null;
  sentiment: number;
  magnitude: number;
  impact: number;
  start_tick: number;
  duration_ticks: number;
  effective_sentiment: number;
  decay_factor: number;
}

/** Response for GET /api/news */
export interface ActiveNewsResponse {
  events: NewsEventData[];
  count: number;
  tick: number;
}

// =============================================================================
// Order Distribution Types (Pre-auction)
// =============================================================================

/** Price level with quantity. */
export type PriceLevel = [number, number]; // [price, quantity]

/** Response for GET /api/order-distribution */
export interface OrderDistributionResponse {
  symbol: string;
  bids: PriceLevel[];
  asks: PriceLevel[];
  tick: number;
}

// =============================================================================
// API Status Types
// =============================================================================

/** Response for GET /api/status */
export interface StatusResponse {
  tick: number;
  total_agents: number;
  running: boolean;
  finished: boolean;
  uptime_secs: number;
}

/** Response for GET /health */
export interface HealthResponse {
  tick: number;
  agents: number;
  uptime_secs: number;
  ws_connections: number;
}
