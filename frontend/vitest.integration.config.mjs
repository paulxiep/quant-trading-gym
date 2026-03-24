/**
 * Vitest Integration Test Configuration
 *
 * Runs integration tests against a live server.
 * Expects SERVER_URL environment variable to be set.
 */
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.integration.test.ts'],
    testTimeout: 30000, // 30s timeout for integration tests
    hookTimeout: 30000,
    // Run tests sequentially to avoid overwhelming the server
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
  },
});
