/**
 * Simulation dashboard page (V4.4).
 *
 * Composes all dashboard components into a responsive grid layout.
 * Connects WebSocket for real-time updates and REST API for detailed data.
 *
 * Declarative: Props flow from hooks to components
 * Modular: Each section is a self-contained component
 * SoC: Page only handles layout and data orchestration
 */

import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '../components';
import {
  PriceChart,
  IndicatorPanel,
  OrderDepthChart,
  FactorGauges,
  RiskPanel,
  NewsFeed,
  AgentTable,
  TimeControls,
  SymbolSelector,
} from '../components/dashboard';
import type { ChartType } from '../components/dashboard';
import { useWebSocket } from '../hooks/useWebSocket';
import { useDashboardData } from '../hooks/useDataService';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TabId = 'dashboard' | 'agents' | 'orders';

interface TabProps {
  id: TabId;
  label: string;
  activeTab: TabId;
  onClick: (id: TabId) => void;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Tab({ id, label, activeTab, onClick }: TabProps) {
  const isActive = activeTab === id;
  return (
    <button
      className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
        isActive
          ? 'bg-gray-800 text-white border-b-2 border-primary-500'
          : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
      }`}
      onClick={() => onClick(id)}
    >
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function SimulationPage() {
  const [activeTab, setActiveTab] = useState<TabId>('dashboard');
  const [isRunning, setIsRunning] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [chartType, setChartType] = useState<ChartType>('candlestick');

  // WebSocket connection for real-time tick data
  const { tickData, connectionState, sendCommand, connect } = useWebSocket({
    autoReconnect: true,
  });

  // REST API data hooks with auto-refresh (pass selected symbol)
  const dashboard = useDashboardData(
    { baseUrl: 'http://localhost:8001', symbol: selectedSymbol || undefined },
    { enabled: true, interval: 1000 },
  );

  // Auto-select symbol from first candles response
  useEffect(() => {
    if (!selectedSymbol && dashboard.candles.data?.symbol) {
      setSelectedSymbol(dashboard.candles.data.symbol);
    }
  }, [selectedSymbol, dashboard.candles.data?.symbol]);

  // Handle simulation commands
  const handleCommand = (cmd: 'Start' | 'Pause' | 'Toggle' | 'Step' | 'Quit') => {
    sendCommand(cmd);
    if (cmd === 'Start') setIsRunning(true);
    if (cmd === 'Pause' || cmd === 'Quit') setIsRunning(false);
    if (cmd === 'Toggle') setIsRunning((prev) => !prev);
  };

  return (
    <div className="min-h-screen bg-gray-950 overflow-x-hidden">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-full mx-auto px-6 py-3 flex items-center justify-between">
          <Link
            to="/"
            className="text-xl font-bold text-gray-100 hover:text-primary-400 transition-colors"
          >
            ← Quant Trading Gym
          </Link>
          <div className="flex items-center gap-4">
            <SymbolSelector value={selectedSymbol} onChange={setSelectedSymbol} />
            <Link to="/config">
              <Button variant="ghost" className="text-sm">
                Configure
              </Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Time Controls Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-2">
        <TimeControls
          tickData={tickData}
          connectionState={connectionState}
          isRunning={isRunning}
          onCommand={handleCommand}
          onConnect={connect}
        />
      </div>

      {/* Tab Navigation */}
      <div className="bg-gray-900/30 border-b border-gray-800 px-6">
        <div className="flex gap-1">
          <Tab id="dashboard" label="Dashboard" activeTab={activeTab} onClick={setActiveTab} />
          <Tab id="agents" label="Agents" activeTab={activeTab} onClick={setActiveTab} />
          <Tab id="orders" label="Order Flow" activeTab={activeTab} onClick={setActiveTab} />
        </div>
      </div>

      {/* Main Content */}
      <main className="p-6">
        {activeTab === 'dashboard' && (
          <DashboardTab
            dashboard={dashboard}
            chartType={chartType}
            onChartTypeChange={setChartType}
          />
        )}
        {activeTab === 'agents' && (
          <AgentsTab agents={dashboard.agents} selectedSymbol={selectedSymbol} />
        )}
        {activeTab === 'orders' && <OrderFlowTab orderDistribution={dashboard.orderDistribution} />}
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab content components
// ---------------------------------------------------------------------------

interface DashboardTabProps {
  dashboard: ReturnType<typeof useDashboardData>;
  chartType: ChartType;
  onChartTypeChange: (type: ChartType) => void;
}

function DashboardTab({ dashboard, chartType, onChartTypeChange }: DashboardTabProps) {
  return (
    <div className="grid grid-cols-12 gap-4">
      {/* Main chart area - spans 8 columns */}
      <div className="col-span-12 lg:col-span-8 space-y-4 min-w-0 overflow-hidden">
        {/* Price Chart */}
        <PriceChart
          data={dashboard.candles.data}
          loading={dashboard.candles.loading}
          error={dashboard.candles.error}
          height={350}
          chartType={chartType}
          onChartTypeChange={onChartTypeChange}
        />

        {/* Order Depth (below chart) */}
        <OrderDepthChart
          data={dashboard.orderDistribution.data}
          loading={dashboard.orderDistribution.loading}
          error={dashboard.orderDistribution.error}
          height={200}
        />
      </div>

      {/* Right sidebar - 4 columns */}
      <div className="col-span-12 lg:col-span-4 space-y-4">
        {/* Technical Indicators */}
        <IndicatorPanel
          data={dashboard.indicators.data}
          loading={dashboard.indicators.loading}
          error={dashboard.indicators.error}
        />

        {/* Factor Gauges */}
        <FactorGauges
          data={dashboard.factors.data}
          loading={dashboard.factors.loading}
          error={dashboard.factors.error}
        />
      </div>

      {/* Bottom row - full width */}
      <div className="col-span-12 lg:col-span-6">
        {/* Risk Metrics */}
        <RiskPanel
          data={dashboard.risk.data}
          loading={dashboard.risk.loading}
          error={dashboard.risk.error}
        />
      </div>

      <div className="col-span-12 lg:col-span-6">
        {/* News Feed */}
        <NewsFeed
          data={dashboard.news.data}
          loading={dashboard.news.loading}
          error={dashboard.news.error}
          maxEvents={5}
        />
      </div>
    </div>
  );
}

interface AgentsTabProps {
  agents: ReturnType<typeof useDashboardData>['agents'];
  selectedSymbol: string;
}

function AgentsTab({ agents, selectedSymbol }: AgentsTabProps) {
  return (
    <div>
      <AgentTable
        data={agents.data}
        loading={agents.loading}
        error={agents.error}
        pageSize={25}
        selectedSymbol={selectedSymbol}
      />
    </div>
  );
}

interface OrderFlowTabProps {
  orderDistribution: ReturnType<typeof useDashboardData>['orderDistribution'];
}

function OrderFlowTab({ orderDistribution }: OrderFlowTabProps) {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-4">
        <h2 className="text-xl font-semibold text-white">Pre-Auction Order Distribution</h2>
        <p className="text-gray-400 text-sm">
          Order flow captured before batch auction clearing. Shows demand/supply distribution that
          would be invisible at tick-end.
        </p>
      </div>
      <OrderDepthChart
        data={orderDistribution.data}
        loading={orderDistribution.loading}
        error={orderDistribution.error}
        height={400}
        maxLevels={20}
      />
    </div>
  );
}
