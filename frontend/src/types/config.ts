/**
 * SimConfig type definitions matching Rust src/config.rs
 *
 * Declarative: Types describe shape, not behavior
 * SoC: All config-related types in one place
 */

/** Sector enum matching Rust types::Sector */
export type Sector =
  | 'Tech'
  | 'Finance'
  | 'Healthcare'
  | 'Consumer'
  | 'Industrials'
  | 'Utilities'
  | 'RealEstate'
  | 'Communications';

/** Market regime for Tier 3 background pool */
export type MarketRegime = 'Normal' | 'Calm' | 'Volatile' | 'Trending' | 'Crisis';

/** Symbol specification */
export interface SymbolSpec {
  symbol: string;
  initialPrice: number;
  sector: Sector;
}

/** Full simulation configuration matching Rust SimConfig */
export interface SimConfig {
  // Simulation Control
  symbols: SymbolSpec[];
  totalTicks: number;
  tickDelayMs: number;
  verbose: boolean;
  maxCpuPercent: number;

  // Tier 1 Agent Counts
  numMarketMakers: number;
  numNoiseTraders: number;
  numMomentumTraders: number;
  numTrendFollowers: number;
  numMacdTraders: number;
  numBollingerTraders: number;
  numVwapExecutors: number;
  numPairsTraders: number;
  numSectorRotators: number;
  minTier1Agents: number;

  // Tier 2 Reactive Agents
  numTier2Agents: number;
  t2InitialCash: number;
  t2MaxPosition: number;
  t2BuyThresholdMin: number;
  t2BuyThresholdMax: number;
  t2StopLossMin: number;
  t2StopLossMax: number;
  t2TakeProfitMin: number;
  t2TakeProfitMax: number;
  t2SellThresholdMin: number;
  t2SellThresholdMax: number;
  t2TakeProfitProb: number;
  t2NewsReactorProb: number;
  t2OrderSizeMin: number;
  t2OrderSizeMax: number;

  // Tier 3 Background Pool
  enableBackgroundPool: boolean;
  backgroundPoolSize: number;
  backgroundRegime: MarketRegime;
  t3MeanOrderSize: number;
  t3MaxOrderSize: number;
  t3OrderSizeStddev: number;
  t3BaseActivity: number | null;
  t3PriceSpreadLambda: number;
  t3MaxPriceDeviation: number;

  // Market Maker Parameters
  mmInitialCash: number;
  mmHalfSpread: number;
  mmQuoteSize: number;
  mmRefreshInterval: number;
  mmMaxInventory: number;
  mmInventorySkew: number;

  // Noise Trader Parameters
  ntInitialCash: number;
  ntInitialPosition: number;
  ntOrderProbability: number;
  ntPriceDeviation: number;
  ntMinQuantity: number;
  ntMaxQuantity: number;

  // Quant Strategy Parameters
  quantInitialCash: number;
  quantOrderSize: number;
  quantMaxPosition: number;

  // TUI Parameters
  maxPriceHistory: number;
  tuiFrameRate: number;
  dataUpdateRate: number;

  // Event/News Generation
  eventsEnabled: boolean;
  eventEarningsProb: number;
  eventEarningsInterval: number;
  eventGuidanceProb: number;
  eventGuidanceInterval: number;
  eventRateDecisionProb: number;
  eventRateDecisionInterval: number;
  eventSectorNewsProb: number;
  eventSectorNewsInterval: number;
}

/** Preset metadata */
export interface Preset {
  name: string;
  isBuiltin: boolean;
}

/** All available sectors for dropdown */
export const SECTORS: Sector[] = [
  'Tech',
  'Finance',
  'Healthcare',
  'Consumer',
  'Industrials',
  'Utilities',
  'RealEstate',
  'Communications',
];

/** All available market regimes for dropdown */
export const MARKET_REGIMES: MarketRegime[] = ['Normal', 'Calm', 'Volatile', 'Trending', 'Crisis'];
