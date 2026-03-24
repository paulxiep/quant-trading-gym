/**
 * Dashboard components barrel file (V4.4)
 *
 * Exports all dashboard-related components for simulation visualization.
 */

// Charts
export { PriceChart, PriceYAxis } from './PriceChart';
export type { PriceChartProps, PriceYAxisProps, ChartType } from './PriceChart';

export { IndicatorPanel } from './IndicatorPanel';
export type { IndicatorPanelProps } from './IndicatorPanel';

export { OrderDepthChart } from './OrderDepthChart';
export type { OrderDepthChartProps } from './OrderDepthChart';

// Panels
export { FactorGauges } from './FactorGauges';
export type { FactorGaugesProps } from './FactorGauges';

export { RiskPanel } from './RiskPanel';
export type { RiskPanelProps } from './RiskPanel';

export { NewsFeed } from './NewsFeed';
export type { NewsFeedProps } from './NewsFeed';

// Tables
export { AgentTable } from './AgentTable';
export type { AgentTableProps } from './AgentTable';

// Controls
export { TimeControls } from './TimeControls';
export type { TimeControlsProps } from './TimeControls';

export { SymbolSelector } from './SymbolSelector';
export type { SymbolSelectorProps } from './SymbolSelector';
