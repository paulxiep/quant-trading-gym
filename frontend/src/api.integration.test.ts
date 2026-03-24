/**
 * Frontend-Backend Integration Tests (V4.5)
 *
 * Tests API contract between React frontend and Axum server.
 * Validates response shapes match TypeScript types in src/types/api.ts.
 *
 * # Running
 *
 * Via docker-compose:
 *   docker compose -f docker-compose.frontend.yaml run --rm integration-test
 *
 * # Design Principles
 *
 * - **Declarative**: Test assertions describe expected contract
 * - **Modular**: Grouped by API domain (analytics, portfolio, risk, news)
 * - **SoC**: Tests only verify API contract, not business logic
 */

import { describe, it, expect, beforeAll } from 'vitest';

// Server URL from environment (set in docker-compose)
const SERVER_URL = process.env.SERVER_URL || 'http://localhost:8001';

// Helper for fetch with timeout
async function fetchWithRetry(url: string, retries = 10, delay = 1000): Promise<Response> {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url);
      return response;
    } catch (e) {
      if (i === retries - 1) throw e;
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw new Error(`Failed to fetch ${url} after ${retries} retries`);
}

// =============================================================================
// Health Check
// =============================================================================

describe('Health Check', () => {
  it('GET /health returns valid response', async () => {
    const response = await fetchWithRetry(`${SERVER_URL}/health`);
    expect(response.ok).toBe(true);

    const data = await response.json();

    // Validate shape matches HealthResponse
    expect(data).toHaveProperty('tick');
    expect(typeof data.tick).toBe('number');
    expect(data).toHaveProperty('agents');
    expect(typeof data.agents).toBe('number');
    expect(data).toHaveProperty('uptime_secs');
    expect(typeof data.uptime_secs).toBe('number');
    expect(data).toHaveProperty('ws_connections');
    expect(typeof data.ws_connections).toBe('number');
  });
});

// =============================================================================
// Analytics API
// =============================================================================

describe('Analytics API', () => {
  describe('GET /api/symbols', () => {
    it('returns array of symbol strings', async () => {
      const response = await fetchWithRetry(`${SERVER_URL}/api/symbols`);
      expect(response.ok).toBe(true);

      const data = await response.json();
      expect(data).toHaveProperty('symbols');
      expect(Array.isArray(data.symbols)).toBe(true);
      // Symbols may be empty if simulation just started - that's valid
      // Just verify the shape is correct
      for (const symbol of data.symbols) {
        expect(typeof symbol).toBe('string');
      }
    });
  });

  describe('GET /api/analytics/candles', () => {
    it('returns CandlesResponse shape', async () => {
      const response = await fetchWithRetry(`${SERVER_URL}/api/analytics/candles`);
      expect(response.ok).toBe(true);

      const data = await response.json();

      // Validate CandlesResponse shape
      expect(data).toHaveProperty('symbol');
      expect(typeof data.symbol).toBe('string');
      expect(data).toHaveProperty('candles');
      expect(Array.isArray(data.candles)).toBe(true);
      expect(data).toHaveProperty('total');
      expect(typeof data.total).toBe('number');
    });

    it('candles have correct OHLCV shape', async () => {
      const response = await fetchWithRetry(`${SERVER_URL}/api/analytics/candles`);
      const data = await response.json();

      if (data.candles.length > 0) {
        const candle = data.candles[0];
        expect(candle).toHaveProperty('tick');
        expect(typeof candle.tick).toBe('number');
        expect(candle).toHaveProperty('open');
        expect(typeof candle.open).toBe('number');
        expect(candle).toHaveProperty('high');
        expect(typeof candle.high).toBe('number');
        expect(candle).toHaveProperty('low');
        expect(typeof candle.low).toBe('number');
        expect(candle).toHaveProperty('close');
        expect(typeof candle.close).toBe('number');
        expect(candle).toHaveProperty('volume');
        expect(typeof candle.volume).toBe('number');

        // OHLC logic: low <= open,close <= high
        expect(candle.low).toBeLessThanOrEqual(candle.open);
        expect(candle.low).toBeLessThanOrEqual(candle.close);
        expect(candle.high).toBeGreaterThanOrEqual(candle.open);
        expect(candle.high).toBeGreaterThanOrEqual(candle.close);
      }
    });

    it('respects symbol query parameter', async () => {
      // First get list of symbols
      const symbolsResp = await fetchWithRetry(`${SERVER_URL}/api/symbols`);
      const { symbols } = await symbolsResp.json();

      if (symbols.length > 0) {
        const symbol = symbols[0];
        const response = await fetchWithRetry(
          `${SERVER_URL}/api/analytics/candles?symbol=${encodeURIComponent(symbol)}`,
        );
        const data = await response.json();

        expect(data.symbol).toBe(symbol);
      }
    });
  });

  describe('GET /api/analytics/indicators', () => {
    it('returns IndicatorsResponse shape', async () => {
      const response = await fetchWithRetry(`${SERVER_URL}/api/analytics/indicators`);
      expect(response.ok).toBe(true);

      const data = await response.json();

      // Validate IndicatorsResponse shape
      expect(data).toHaveProperty('symbol');
      expect(typeof data.symbol).toBe('string');
      expect(data).toHaveProperty('tick');
      expect(typeof data.tick).toBe('number');
      expect(data).toHaveProperty('indicators');
      expect(typeof data.indicators).toBe('object');

      // Validate nested indicators structure
      const ind = data.indicators;
      expect(ind).toHaveProperty('sma');
      expect(typeof ind.sma).toBe('object');
      expect(ind).toHaveProperty('ema');
      expect(typeof ind.ema).toBe('object');
      expect(ind).toHaveProperty('rsi_8');
      // rsi_8 can be number or null
      expect(ind.rsi_8 === null || typeof ind.rsi_8 === 'number').toBe(true);
    });
  });

  describe('GET /api/analytics/factors', () => {
    it('returns FactorsResponse shape', async () => {
      const response = await fetchWithRetry(`${SERVER_URL}/api/analytics/factors`);
      expect(response.ok).toBe(true);

      const data = await response.json();

      expect(data).toHaveProperty('symbol');
      expect(typeof data.symbol).toBe('string');
      expect(data).toHaveProperty('tick');
      expect(typeof data.tick).toBe('number');
      expect(data).toHaveProperty('factors');
      expect(Array.isArray(data.factors)).toBe(true);
    });
  });

  describe('GET /api/analytics/order-distribution', () => {
    it('returns OrderDistributionResponse shape', async () => {
      const response = await fetchWithRetry(`${SERVER_URL}/api/analytics/order-distribution`);
      expect(response.ok).toBe(true);

      const data = await response.json();

      expect(data).toHaveProperty('symbol');
      expect(typeof data.symbol).toBe('string');
      expect(data).toHaveProperty('bids');
      expect(Array.isArray(data.bids)).toBe(true);
      expect(data).toHaveProperty('asks');
      expect(Array.isArray(data.asks)).toBe(true);
      expect(data).toHaveProperty('tick');
      expect(typeof data.tick).toBe('number');

      // Each bid/ask is [price, quantity] tuple
      if (data.bids.length > 0) {
        const bid = data.bids[0];
        expect(Array.isArray(bid)).toBe(true);
        expect(bid.length).toBe(2);
        expect(typeof bid[0]).toBe('number'); // price
        expect(typeof bid[1]).toBe('number'); // quantity
      }
    });
  });
});

// =============================================================================
// Portfolio API
// =============================================================================

describe('Portfolio API', () => {
  describe('GET /api/portfolio/agents', () => {
    it('returns AgentsResponse shape', async () => {
      const response = await fetchWithRetry(`${SERVER_URL}/api/portfolio/agents`);
      expect(response.ok).toBe(true);

      const data = await response.json();

      expect(data).toHaveProperty('agents');
      expect(Array.isArray(data.agents)).toBe(true);
      expect(data).toHaveProperty('total_count');
      expect(typeof data.total_count).toBe('number');
      expect(data).toHaveProperty('tick');
      expect(typeof data.tick).toBe('number');
    });

    it('agents have correct AgentData shape', async () => {
      const response = await fetchWithRetry(`${SERVER_URL}/api/portfolio/agents`);
      const data = await response.json();

      if (data.agents.length > 0) {
        const agent = data.agents[0];
        expect(agent).toHaveProperty('agent_id');
        expect(typeof agent.agent_id).toBe('number');
        expect(agent).toHaveProperty('name');
        expect(typeof agent.name).toBe('string');
        expect(agent).toHaveProperty('total_pnl');
        expect(typeof agent.total_pnl).toBe('number');
        expect(agent).toHaveProperty('cash');
        expect(typeof agent.cash).toBe('number');
        expect(agent).toHaveProperty('equity');
        expect(typeof agent.equity).toBe('number');
        expect(agent).toHaveProperty('positions');
        expect(typeof agent.positions).toBe('object');
        expect(agent).toHaveProperty('is_market_maker');
        expect(typeof agent.is_market_maker).toBe('boolean');
        expect(agent).toHaveProperty('tier');
        expect(typeof agent.tier).toBe('number');
      }
    });
  });

  describe('GET /api/portfolio/agents/:agent_id', () => {
    it('returns AgentPortfolioResponse for valid agent', async () => {
      // First get list of agents
      const agentsResp = await fetchWithRetry(`${SERVER_URL}/api/portfolio/agents`);
      const { agents } = await agentsResp.json();

      if (agents.length > 0) {
        const agentId = agents[0].agent_id;
        const response = await fetchWithRetry(`${SERVER_URL}/api/portfolio/agents/${agentId}`);
        expect(response.ok).toBe(true);

        const data = await response.json();
        expect(data).toHaveProperty('agent_id');
        expect(data.agent_id).toBe(agentId);
        expect(data).toHaveProperty('name');
        expect(data).toHaveProperty('cash');
        expect(data).toHaveProperty('equity');
        expect(data).toHaveProperty('total_pnl');
        expect(data).toHaveProperty('positions');
        expect(Array.isArray(data.positions)).toBe(true);
      }
    });
  });
});

// =============================================================================
// Risk API
// =============================================================================

describe('Risk API', () => {
  describe('GET /api/risk/:agent_id', () => {
    it('returns AgentRiskMetricsResponse for valid agent', async () => {
      // First get list of agents
      const agentsResp = await fetchWithRetry(`${SERVER_URL}/api/portfolio/agents`);
      const { agents } = await agentsResp.json();

      if (agents.length > 0) {
        const agentId = agents[0].agent_id;
        const response = await fetchWithRetry(`${SERVER_URL}/api/risk/${agentId}`);
        expect(response.ok).toBe(true);

        const data = await response.json();
        expect(data).toHaveProperty('agent_id');
        expect(data.agent_id).toBe(agentId);
        expect(data).toHaveProperty('current_drawdown');
        expect(typeof data.current_drawdown).toBe('number');
        expect(data).toHaveProperty('max_drawdown');
        expect(typeof data.max_drawdown).toBe('number');
      }
    });
  });
});

// =============================================================================
// News API
// =============================================================================

describe('News API', () => {
  describe('GET /api/news/active', () => {
    it('returns ActiveNewsResponse shape', async () => {
      const response = await fetchWithRetry(`${SERVER_URL}/api/news/active`);
      expect(response.ok).toBe(true);

      const data = await response.json();

      expect(data).toHaveProperty('events');
      expect(Array.isArray(data.events)).toBe(true);
      expect(data).toHaveProperty('count');
      expect(typeof data.count).toBe('number');
      expect(data).toHaveProperty('tick');
      expect(typeof data.tick).toBe('number');
    });
  });
});

// =============================================================================
// Status API
// =============================================================================

describe('Status API', () => {
  describe('GET /api/status', () => {
    it('returns StatusResponse shape', async () => {
      const response = await fetchWithRetry(`${SERVER_URL}/api/status`);
      expect(response.ok).toBe(true);

      const data = await response.json();

      expect(data).toHaveProperty('tick');
      expect(typeof data.tick).toBe('number');
      expect(data).toHaveProperty('agents');
      expect(typeof data.agents).toBe('number');
      expect(data).toHaveProperty('running');
      expect(typeof data.running).toBe('boolean');
      expect(data).toHaveProperty('finished');
      expect(typeof data.finished).toBe('boolean');
    });
  });
});
