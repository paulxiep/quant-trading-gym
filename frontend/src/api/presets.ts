/**
 * API client for preset and config endpoints
 *
 * SoC: All API communication in one module
 * Declarative: Functions describe what data to fetch, not how
 */

import type { Preset, SimConfig } from '../types';

const API_BASE = '/api';

/**
 * Fetch all available preset names
 */
export async function fetchPresets(): Promise<Preset[]> {
  const response = await fetch(`${API_BASE}/presets`);
  if (!response.ok) {
    throw new Error(`Failed to fetch presets: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch a specific preset's full configuration
 */
export async function fetchPreset(name: string): Promise<SimConfig> {
  const response = await fetch(`${API_BASE}/presets/${encodeURIComponent(name)}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch preset '${name}': ${response.statusText}`);
  }
  return response.json();
}

/**
 * Save a new custom preset
 */
export async function savePreset(name: string, config: SimConfig): Promise<void> {
  const response = await fetch(`${API_BASE}/presets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, config }),
  });
  if (!response.ok) {
    throw new Error(`Failed to save preset '${name}': ${response.statusText}`);
  }
}

/**
 * Submit configuration to start simulation
 */
export async function submitConfig(config: SimConfig): Promise<void> {
  const response = await fetch(`${API_BASE}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) {
    throw new Error(`Failed to submit config: ${response.statusText}`);
  }
}
