/**
 * Default SimConfig matching Rust SimConfig::default()
 *
 * Declarative: Config is pure data
 * SoC: Default values separate from components
 */

import type { SimConfig } from '../types';

export const DEFAULT_CONFIG: SimConfig = {
  // Simulation Control
  symbols: [
    { symbol: 'Duck Delish', initialPrice: 100.0, sector: 'Consumer' },
    { symbol: 'Zephyr Zap', initialPrice: 100.0, sector: 'Utilities' },
    { symbol: 'Vraiment Villa', initialPrice: 100.0, sector: 'RealEstate' },
    { symbol: 'Quant Quotation', initialPrice: 100.0, sector: 'Finance' },
    { symbol: 'Hello Handy', initialPrice: 100.0, sector: 'Communications' },
    { symbol: 'Nubes Nexus', initialPrice: 100.0, sector: 'Tech' },
  ],
  totalTicks: 10000,
  tickDelayMs: 0,
  verbose: false,
  maxCpuPercent: 34,

  // Tier 1 Agent Counts
  numMarketMakers: 200,
  numNoiseTraders: 1500,
  numMomentumTraders: 500,
  numTrendFollowers: 500,
  numMacdTraders: 500,
  numBollingerTraders: 500,
  numVwapExecutors: 100,
  numPairsTraders: 1200,
  numSectorRotators: 2000,
  minTier1Agents: 5000,

  // Tier 2 Reactive Agents
  numTier2Agents: 18000,
  t2InitialCash: 100000.0,
  t2MaxPosition: 1000,
  t2BuyThresholdMin: 60.0,
  t2BuyThresholdMax: 90.0,
  t2StopLossMin: 0.25,
  t2StopLossMax: 0.5,
  t2TakeProfitMin: 0.25,
  t2TakeProfitMax: 0.5,
  t2SellThresholdMin: 105.0,
  t2SellThresholdMax: 130.0,
  t2TakeProfitProb: 0.5,
  t2NewsReactorProb: 0.1,
  t2OrderSizeMin: 0.2,
  t2OrderSizeMax: 0.5,

  // Tier 3 Background Pool
  enableBackgroundPool: true,
  backgroundPoolSize: 75000,
  backgroundRegime: 'Normal',
  t3MeanOrderSize: 25.0,
  t3MaxOrderSize: 100,
  t3OrderSizeStddev: 10.0,
  t3BaseActivity: 0.003,
  t3PriceSpreadLambda: 10.0,
  t3MaxPriceDeviation: 0.05,

  // Market Maker Parameters
  mmInitialCash: 1000000.0,
  mmHalfSpread: 0.005,
  mmQuoteSize: 100,
  mmRefreshInterval: 1,
  mmMaxInventory: 200,
  mmInventorySkew: 0.001,

  // Noise Trader Parameters
  ntInitialCash: 100000.0,
  ntInitialPosition: 0,
  ntOrderProbability: 0.3,
  ntPriceDeviation: 0.02,
  ntMinQuantity: 15,
  ntMaxQuantity: 50,

  // Quant Strategy Parameters
  quantInitialCash: 100000.0,
  quantOrderSize: 35,
  quantMaxPosition: 200,

  // TUI Parameters
  maxPriceHistory: 500,
  tuiFrameRate: 30,
  dataUpdateRate: 30,

  // Event/News Generation
  eventsEnabled: true,
  eventEarningsProb: 0.006,
  eventEarningsInterval: 25,
  eventGuidanceProb: 0.003,
  eventGuidanceInterval: 50,
  eventRateDecisionProb: 0.0015,
  eventRateDecisionInterval: 125,
  eventSectorNewsProb: 0.009,
  eventSectorNewsInterval: 12,
};

/** Demo preset - 10% agents, fewer ticks */
export const DEMO_CONFIG: SimConfig = {
  ...DEFAULT_CONFIG,
  totalTicks: 1000,
  tickDelayMs: 5,
  numMarketMakers: 10,
  numNoiseTraders: 40,
  numMomentumTraders: 5,
  numTrendFollowers: 5,
  numMacdTraders: 5,
  numBollingerTraders: 5,
  numVwapExecutors: 5,
  numPairsTraders: 0,
  numSectorRotators: 0,
  minTier1Agents: 100,
  numTier2Agents: 100,
  enableBackgroundPool: false,
  backgroundPoolSize: 0,
};

/** Stress test - 2x agents, many ticks */
export const STRESS_TEST_CONFIG: SimConfig = {
  ...DEFAULT_CONFIG,
  totalTicks: 100000,
  tickDelayMs: 0,
  minTier1Agents: 10000,
  numTier2Agents: 36000,
  backgroundPoolSize: 150000,
};

/** Low activity - conservative parameters */
export const LOW_ACTIVITY_CONFIG: SimConfig = {
  ...DEFAULT_CONFIG,
  numMarketMakers: 20,
  numNoiseTraders: 80,
  numMomentumTraders: 10,
  numTrendFollowers: 10,
  numMacdTraders: 10,
  numBollingerTraders: 10,
  numVwapExecutors: 10,
  ntOrderProbability: 0.1,
  minTier1Agents: 200,
  numTier2Agents: 500,
  backgroundPoolSize: 5000,
};

/** High volatility - aggressive noise traders */
export const HIGH_VOLATILITY_CONFIG: SimConfig = {
  ...DEFAULT_CONFIG,
  numNoiseTraders: 2250,
  ntOrderProbability: 0.5,
  ntInitialCash: 50000.0,
  mmHalfSpread: 0.005,
};

/** Quant heavy - more algorithmic traders */
export const QUANT_HEAVY_CONFIG: SimConfig = {
  ...DEFAULT_CONFIG,
  numNoiseTraders: 100,
  numMomentumTraders: 450,
  numTrendFollowers: 450,
  numMacdTraders: 450,
  numBollingerTraders: 450,
  numVwapExecutors: 300,
};

/** Built-in presets map */
export const BUILTIN_PRESETS: Record<string, SimConfig> = {
  Default: DEFAULT_CONFIG,
  Demo: DEMO_CONFIG,
  'Stress Test': STRESS_TEST_CONFIG,
  'Low Activity': LOW_ACTIVITY_CONFIG,
  'High Volatility': HIGH_VOLATILITY_CONFIG,
  'Quant Heavy': QUANT_HEAVY_CONFIG,
};
