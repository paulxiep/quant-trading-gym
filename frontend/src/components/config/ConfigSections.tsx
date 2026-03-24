/**
 * Config form sections - grouped by SimConfig categories
 *
 * SoC: Each section handles one category of config
 * Modular: Sections are independent, composable
 */

import type { SimConfig, MarketRegime } from '../../types';
import { MARKET_REGIMES } from '../../types';
import { Accordion, NumberInput, CheckboxInput, SelectInput } from '../ui';
import { SymbolsEditor } from './SymbolsEditor';

type ConfigUpdater = <K extends keyof SimConfig>(key: K, value: SimConfig[K]) => void;

interface SectionProps {
  config: SimConfig;
  updateConfig: ConfigUpdater;
}

/** Simulation Control section - always visible */
export function SimulationControlSection({ config, updateConfig }: SectionProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-100">Simulation Control</h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <NumberInput
          label="Total Ticks"
          value={config.totalTicks}
          onChange={(v) => updateConfig('totalTicks', v)}
          min={0}
        />
        <NumberInput
          label="Tick Delay (ms)"
          value={config.tickDelayMs}
          onChange={(v) => updateConfig('tickDelayMs', v)}
          min={0}
        />
        <NumberInput
          label="Max CPU %"
          value={config.maxCpuPercent}
          onChange={(v) => updateConfig('maxCpuPercent', v)}
          min={1}
          max={100}
        />
        <CheckboxInput
          label="Events Enabled"
          checked={config.eventsEnabled}
          onChange={(v) => updateConfig('eventsEnabled', v)}
          className="self-end pb-2"
        />
      </div>
    </div>
  );
}

/** Symbols section - always visible */
export function SymbolsSection({ config, updateConfig }: SectionProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-100">Symbols</h3>
      <SymbolsEditor
        symbols={config.symbols}
        onChange={(symbols) => updateConfig('symbols', symbols)}
      />
    </div>
  );
}

/** Tier 1 Agents section - always visible */
export function Tier1AgentsSection({ config, updateConfig }: SectionProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-100">Tier 1 Agents</h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <NumberInput
          label="Market Makers"
          value={config.numMarketMakers}
          onChange={(v) => updateConfig('numMarketMakers', v)}
          min={0}
        />
        <NumberInput
          label="Noise Traders"
          value={config.numNoiseTraders}
          onChange={(v) => updateConfig('numNoiseTraders', v)}
          min={0}
        />
        <NumberInput
          label="Momentum Traders"
          value={config.numMomentumTraders}
          onChange={(v) => updateConfig('numMomentumTraders', v)}
          min={0}
        />
        <NumberInput
          label="Trend Followers"
          value={config.numTrendFollowers}
          onChange={(v) => updateConfig('numTrendFollowers', v)}
          min={0}
        />
        <NumberInput
          label="MACD Traders"
          value={config.numMacdTraders}
          onChange={(v) => updateConfig('numMacdTraders', v)}
          min={0}
        />
        <NumberInput
          label="Bollinger Traders"
          value={config.numBollingerTraders}
          onChange={(v) => updateConfig('numBollingerTraders', v)}
          min={0}
        />
        <NumberInput
          label="VWAP Executors"
          value={config.numVwapExecutors}
          onChange={(v) => updateConfig('numVwapExecutors', v)}
          min={0}
        />
        <NumberInput
          label="Pairs Traders"
          value={config.numPairsTraders}
          onChange={(v) => updateConfig('numPairsTraders', v)}
          min={0}
        />
        <NumberInput
          label="Sector Rotators"
          value={config.numSectorRotators}
          onChange={(v) => updateConfig('numSectorRotators', v)}
          min={0}
        />
        <NumberInput
          label="Min Tier 1 Total"
          value={config.minTier1Agents}
          onChange={(v) => updateConfig('minTier1Agents', v)}
          min={0}
        />
      </div>
    </div>
  );
}

/** Tier 2 Agents section - collapsed by default */
export function Tier2AgentsSection({ config, updateConfig }: SectionProps) {
  return (
    <Accordion title="Tier 2 Reactive Agents">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <NumberInput
          label="Tier 2 Count"
          value={config.numTier2Agents}
          onChange={(v) => updateConfig('numTier2Agents', v)}
          min={0}
        />
        <NumberInput
          label="Initial Cash"
          value={config.t2InitialCash}
          onChange={(v) => updateConfig('t2InitialCash', v)}
          min={0}
        />
        <NumberInput
          label="Max Position"
          value={config.t2MaxPosition}
          onChange={(v) => updateConfig('t2MaxPosition', v)}
          min={0}
        />
        <NumberInput
          label="Buy Threshold Min"
          value={config.t2BuyThresholdMin}
          onChange={(v) => updateConfig('t2BuyThresholdMin', v)}
          step={0.1}
        />
        <NumberInput
          label="Buy Threshold Max"
          value={config.t2BuyThresholdMax}
          onChange={(v) => updateConfig('t2BuyThresholdMax', v)}
          step={0.1}
        />
        <NumberInput
          label="Stop Loss Min"
          value={config.t2StopLossMin}
          onChange={(v) => updateConfig('t2StopLossMin', v)}
          step={0.01}
        />
        <NumberInput
          label="Stop Loss Max"
          value={config.t2StopLossMax}
          onChange={(v) => updateConfig('t2StopLossMax', v)}
          step={0.01}
        />
        <NumberInput
          label="Take Profit Min"
          value={config.t2TakeProfitMin}
          onChange={(v) => updateConfig('t2TakeProfitMin', v)}
          step={0.01}
        />
        <NumberInput
          label="Take Profit Max"
          value={config.t2TakeProfitMax}
          onChange={(v) => updateConfig('t2TakeProfitMax', v)}
          step={0.01}
        />
        <NumberInput
          label="Sell Threshold Min"
          value={config.t2SellThresholdMin}
          onChange={(v) => updateConfig('t2SellThresholdMin', v)}
          step={0.1}
        />
        <NumberInput
          label="Sell Threshold Max"
          value={config.t2SellThresholdMax}
          onChange={(v) => updateConfig('t2SellThresholdMax', v)}
          step={0.1}
        />
        <NumberInput
          label="Take Profit Prob"
          value={config.t2TakeProfitProb}
          onChange={(v) => updateConfig('t2TakeProfitProb', v)}
          step={0.1}
          min={0}
          max={1}
        />
        <NumberInput
          label="News Reactor Prob"
          value={config.t2NewsReactorProb}
          onChange={(v) => updateConfig('t2NewsReactorProb', v)}
          step={0.1}
          min={0}
          max={1}
        />
        <NumberInput
          label="Order Size Min"
          value={config.t2OrderSizeMin}
          onChange={(v) => updateConfig('t2OrderSizeMin', v)}
          step={0.1}
          min={0}
          max={1}
        />
        <NumberInput
          label="Order Size Max"
          value={config.t2OrderSizeMax}
          onChange={(v) => updateConfig('t2OrderSizeMax', v)}
          step={0.1}
          min={0}
          max={1}
        />
      </div>
    </Accordion>
  );
}

/** Tier 3 Background Pool section - collapsed by default */
export function Tier3PoolSection({ config, updateConfig }: SectionProps) {
  return (
    <Accordion title="Tier 3 Background Pool">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <CheckboxInput
          label="Enable Background Pool"
          checked={config.enableBackgroundPool}
          onChange={(v) => updateConfig('enableBackgroundPool', v)}
        />
        <NumberInput
          label="Pool Size"
          value={config.backgroundPoolSize}
          onChange={(v) => updateConfig('backgroundPoolSize', v)}
          min={0}
        />
        <SelectInput<MarketRegime>
          label="Market Regime"
          value={config.backgroundRegime}
          options={MARKET_REGIMES}
          onChange={(v) => updateConfig('backgroundRegime', v)}
        />
        <NumberInput
          label="Mean Order Size"
          value={config.t3MeanOrderSize}
          onChange={(v) => updateConfig('t3MeanOrderSize', v)}
          step={0.1}
        />
        <NumberInput
          label="Max Order Size"
          value={config.t3MaxOrderSize}
          onChange={(v) => updateConfig('t3MaxOrderSize', v)}
          min={1}
        />
        <NumberInput
          label="Order Size Stddev"
          value={config.t3OrderSizeStddev}
          onChange={(v) => updateConfig('t3OrderSizeStddev', v)}
          step={0.1}
        />
        <NumberInput
          label="Base Activity"
          value={config.t3BaseActivity ?? 0}
          onChange={(v) => updateConfig('t3BaseActivity', v || null)}
          step={0.001}
          min={0}
          max={1}
        />
        <NumberInput
          label="Price Spread Lambda"
          value={config.t3PriceSpreadLambda}
          onChange={(v) => updateConfig('t3PriceSpreadLambda', v)}
          step={0.1}
        />
        <NumberInput
          label="Max Price Deviation"
          value={config.t3MaxPriceDeviation}
          onChange={(v) => updateConfig('t3MaxPriceDeviation', v)}
          step={0.01}
          min={0}
          max={1}
        />
      </div>
    </Accordion>
  );
}

/** Market Maker Parameters section - collapsed by default */
export function MarketMakerSection({ config, updateConfig }: SectionProps) {
  return (
    <Accordion title="Market Maker Parameters">
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        <NumberInput
          label="Initial Cash"
          value={config.mmInitialCash}
          onChange={(v) => updateConfig('mmInitialCash', v)}
          min={0}
        />
        <NumberInput
          label="Half Spread"
          value={config.mmHalfSpread}
          onChange={(v) => updateConfig('mmHalfSpread', v)}
          step={0.001}
          min={0}
        />
        <NumberInput
          label="Quote Size"
          value={config.mmQuoteSize}
          onChange={(v) => updateConfig('mmQuoteSize', v)}
          min={1}
        />
        <NumberInput
          label="Refresh Interval"
          value={config.mmRefreshInterval}
          onChange={(v) => updateConfig('mmRefreshInterval', v)}
          min={1}
        />
        <NumberInput
          label="Max Inventory"
          value={config.mmMaxInventory}
          onChange={(v) => updateConfig('mmMaxInventory', v)}
          min={0}
        />
        <NumberInput
          label="Inventory Skew"
          value={config.mmInventorySkew}
          onChange={(v) => updateConfig('mmInventorySkew', v)}
          step={0.0001}
        />
      </div>
    </Accordion>
  );
}

/** Noise Trader Parameters section - collapsed by default */
export function NoiseTraderSection({ config, updateConfig }: SectionProps) {
  return (
    <Accordion title="Noise Trader Parameters">
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        <NumberInput
          label="Initial Cash"
          value={config.ntInitialCash}
          onChange={(v) => updateConfig('ntInitialCash', v)}
          min={0}
        />
        <NumberInput
          label="Initial Position"
          value={config.ntInitialPosition}
          onChange={(v) => updateConfig('ntInitialPosition', v)}
        />
        <NumberInput
          label="Order Probability"
          value={config.ntOrderProbability}
          onChange={(v) => updateConfig('ntOrderProbability', v)}
          step={0.01}
          min={0}
          max={1}
        />
        <NumberInput
          label="Price Deviation"
          value={config.ntPriceDeviation}
          onChange={(v) => updateConfig('ntPriceDeviation', v)}
          step={0.01}
          min={0}
        />
        <NumberInput
          label="Min Quantity"
          value={config.ntMinQuantity}
          onChange={(v) => updateConfig('ntMinQuantity', v)}
          min={1}
        />
        <NumberInput
          label="Max Quantity"
          value={config.ntMaxQuantity}
          onChange={(v) => updateConfig('ntMaxQuantity', v)}
          min={1}
        />
      </div>
    </Accordion>
  );
}

/** Quant Strategy Parameters section - collapsed by default */
export function QuantStrategySection({ config, updateConfig }: SectionProps) {
  return (
    <Accordion title="Quant Strategy Parameters">
      <div className="grid grid-cols-3 gap-4">
        <NumberInput
          label="Initial Cash"
          value={config.quantInitialCash}
          onChange={(v) => updateConfig('quantInitialCash', v)}
          min={0}
        />
        <NumberInput
          label="Order Size"
          value={config.quantOrderSize}
          onChange={(v) => updateConfig('quantOrderSize', v)}
          min={1}
        />
        <NumberInput
          label="Max Position"
          value={config.quantMaxPosition}
          onChange={(v) => updateConfig('quantMaxPosition', v)}
          min={0}
        />
      </div>
    </Accordion>
  );
}

/** Event/News Generation section - collapsed by default */
export function EventsSection({ config, updateConfig }: SectionProps) {
  return (
    <Accordion title="Event/News Generation">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <NumberInput
          label="Earnings Prob"
          value={config.eventEarningsProb}
          onChange={(v) => updateConfig('eventEarningsProb', v)}
          step={0.001}
          min={0}
        />
        <NumberInput
          label="Earnings Interval"
          value={config.eventEarningsInterval}
          onChange={(v) => updateConfig('eventEarningsInterval', v)}
          min={1}
        />
        <NumberInput
          label="Guidance Prob"
          value={config.eventGuidanceProb}
          onChange={(v) => updateConfig('eventGuidanceProb', v)}
          step={0.001}
          min={0}
        />
        <NumberInput
          label="Guidance Interval"
          value={config.eventGuidanceInterval}
          onChange={(v) => updateConfig('eventGuidanceInterval', v)}
          min={1}
        />
        <NumberInput
          label="Rate Decision Prob"
          value={config.eventRateDecisionProb}
          onChange={(v) => updateConfig('eventRateDecisionProb', v)}
          step={0.0001}
          min={0}
        />
        <NumberInput
          label="Rate Decision Interval"
          value={config.eventRateDecisionInterval}
          onChange={(v) => updateConfig('eventRateDecisionInterval', v)}
          min={1}
        />
        <NumberInput
          label="Sector News Prob"
          value={config.eventSectorNewsProb}
          onChange={(v) => updateConfig('eventSectorNewsProb', v)}
          step={0.001}
          min={0}
        />
        <NumberInput
          label="Sector News Interval"
          value={config.eventSectorNewsInterval}
          onChange={(v) => updateConfig('eventSectorNewsInterval', v)}
          min={1}
        />
      </div>
    </Accordion>
  );
}
