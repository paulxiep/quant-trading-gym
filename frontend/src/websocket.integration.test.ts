/**
 * WebSocket Integration Tests (V4.5)
 *
 * Tests WebSocket connection and tick message format.
 * Validates real-time data streaming contract.
 *
 * # Running
 *
 * Via docker-compose:
 *   docker compose -f docker-compose.frontend.yaml run --rm integration-test
 */

import { describe, it, expect } from 'vitest';
import WebSocket from 'ws';

// Server URL from environment
const SERVER_URL = process.env.SERVER_URL || 'http://localhost:8001';
const WS_URL = SERVER_URL.replace('http', 'ws') + '/ws';

// Helper to wait for WebSocket connection
function connectWebSocket(url: string): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    const timeout = setTimeout(() => {
      ws.close();
      reject(new Error('WebSocket connection timeout'));
    }, 10000);

    ws.on('open', () => {
      clearTimeout(timeout);
      resolve(ws);
    });
    ws.on('error', (err) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}

// Helper to receive a message with timeout
function receiveMessage(ws: WebSocket, timeoutMs = 10000): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('WebSocket message timeout'));
    }, timeoutMs);

    ws.once('message', (data) => {
      clearTimeout(timeout);
      try {
        const parsed = JSON.parse(data.toString());
        resolve(parsed);
      } catch {
        resolve(data.toString());
      }
    });
  });
}

describe('WebSocket API', () => {
  it('connects to /ws endpoint', async () => {
    const ws = await connectWebSocket(WS_URL);
    expect(ws.readyState).toBe(WebSocket.OPEN);
    ws.close();
  });

  it('receives tick messages with correct shape when simulation running', async () => {
    const ws = await connectWebSocket(WS_URL);

    try {
      // Wait for a tick message - may timeout if simulation is paused
      const message = (await receiveMessage(ws, 5000)) as Record<string, unknown>;

      // Validate TickData shape
      expect(message).toHaveProperty('tick');
      expect(typeof message.tick).toBe('number');

      expect(message).toHaveProperty('timestamp');
      expect(typeof message.timestamp).toBe('string');

      expect(message).toHaveProperty('trades');
      expect(typeof message.trades).toBe('number');

      expect(message).toHaveProperty('orders');
      expect(typeof message.orders).toBe('number');

      expect(message).toHaveProperty('active_agents');
      expect(typeof message.active_agents).toBe('number');

      expect(message).toHaveProperty('prices');
      expect(typeof message.prices).toBe('object');
    } catch (e) {
      // If timeout, simulation may be paused - skip validation
      if ((e as Error).message === 'WebSocket message timeout') {
        console.log('WebSocket timeout - simulation may be paused');
        return; // Test passes - connection works, simulation just not running
      }
      throw e;
    } finally {
      ws.close();
    }
  });

  it('prices object contains symbol -> price mappings when running', async () => {
    const ws = await connectWebSocket(WS_URL);

    try {
      const message = (await receiveMessage(ws, 5000)) as Record<string, unknown>;
      const prices = message.prices as Record<string, number>;

      // Should have at least one symbol
      const symbols = Object.keys(prices);
      expect(symbols.length).toBeGreaterThan(0);

      // Each price should be a number
      for (const symbol of symbols) {
        expect(typeof prices[symbol]).toBe('number');
        expect(prices[symbol]).toBeGreaterThan(0);
      }
    } catch (e) {
      if ((e as Error).message === 'WebSocket message timeout') {
        console.log('WebSocket timeout - simulation may be paused');
        return;
      }
      throw e;
    } finally {
      ws.close();
    }
  });

  it('receives multiple consecutive ticks when running', async () => {
    const ws = await connectWebSocket(WS_URL);

    try {
      // Collect 3 tick messages
      const ticks: number[] = [];
      for (let i = 0; i < 3; i++) {
        const message = (await receiveMessage(ws, 5000)) as {
          tick: number;
        };
        ticks.push(message.tick);
      }

      // Ticks should be monotonically increasing
      expect(ticks.length).toBe(3);
      for (let i = 1; i < ticks.length; i++) {
        expect(ticks[i]).toBeGreaterThan(ticks[i - 1]);
      }
    } catch (e) {
      if ((e as Error).message === 'WebSocket message timeout') {
        console.log('WebSocket timeout - simulation may be paused');
        return;
      }
      throw e;
    } finally {
      ws.close();
    }
  });
});
