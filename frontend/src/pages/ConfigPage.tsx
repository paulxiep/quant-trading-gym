/**
 * Configuration page - simulation parameter editor
 *
 * SoC: Handles config state, preset loading/saving, form layout
 */

import { useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import type { SimConfig, Preset } from '../types';
import { BUILTIN_PRESETS, DEFAULT_CONFIG } from '../config';
import { Button, SelectInput, TextInput } from '../components';
import {
  SimulationControlSection,
  SymbolsSection,
  Tier1AgentsSection,
  Tier2AgentsSection,
  Tier3PoolSection,
  MarketMakerSection,
  NoiseTraderSection,
  QuantStrategySection,
  EventsSection,
} from '../components';

/** Built-in preset names */
const BUILTIN_PRESET_NAMES = Object.keys(BUILTIN_PRESETS);

export function ConfigPage() {
  // Config state
  const [config, setConfig] = useState<SimConfig>(DEFAULT_CONFIG);
  const [selectedPreset, setSelectedPreset] = useState<string>('Default');
  const [customPresets, setCustomPresets] = useState<Preset[]>([]);
  const [newPresetName, setNewPresetName] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  // All preset names for dropdown
  const allPresetNames = [...BUILTIN_PRESET_NAMES, ...customPresets.map((p) => p.name)];

  // Update single config field
  const updateConfig = useCallback(<K extends keyof SimConfig>(key: K, value: SimConfig[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }, []);

  // Load preset
  const handlePresetChange = (presetName: string) => {
    setSelectedPreset(presetName);

    // Check built-in presets first
    if (presetName in BUILTIN_PRESETS) {
      setConfig(BUILTIN_PRESETS[presetName]);
      return;
    }

    // TODO: Load custom preset from API when backend is ready
    // For now, custom presets would be loaded from localStorage as fallback
    const stored = localStorage.getItem(`preset:${presetName}`);
    if (stored) {
      try {
        setConfig(JSON.parse(stored));
      } catch {
        console.error(`Failed to parse preset: ${presetName}`);
      }
    }
  };

  // Save custom preset
  const handleSavePreset = async () => {
    if (!newPresetName.trim()) return;

    setIsSaving(true);
    try {
      // TODO: Save to API when backend is ready
      // For now, save to localStorage
      localStorage.setItem(`preset:${newPresetName}`, JSON.stringify(config));

      // Add to custom presets list
      if (!customPresets.some((p) => p.name === newPresetName)) {
        setCustomPresets((prev) => [...prev, { name: newPresetName, isBuiltin: false }]);
      }

      setSelectedPreset(newPresetName);
      setNewPresetName('');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <Link
            to="/"
            className="text-xl font-bold text-gray-100 hover:text-primary-400 transition-colors"
          >
            ← Quant Trading Gym
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Preset Section */}
        <section className="mb-8 p-6 bg-gray-900 rounded-xl border border-gray-800">
          <h2 className="text-xl font-semibold text-gray-100 mb-4">Presets</h2>
          <div className="flex flex-wrap gap-4 items-end">
            <SelectInput
              label="Load Preset"
              value={selectedPreset}
              options={allPresetNames}
              onChange={handlePresetChange}
              className="w-48"
            />
            <div className="flex-1 min-w-[200px]" />
            <TextInput
              label="New Preset Name"
              value={newPresetName}
              onChange={setNewPresetName}
              placeholder="My Custom Preset"
              className="w-48"
            />
            <Button
              variant="secondary"
              onClick={handleSavePreset}
              disabled={!newPresetName.trim() || isSaving}
            >
              {isSaving ? 'Saving...' : 'Save Preset'}
            </Button>
          </div>
        </section>

        {/* Config Form */}
        <div className="space-y-6">
          {/* Always-visible sections */}
          <section className="p-6 bg-gray-900 rounded-xl border border-gray-800">
            <SimulationControlSection config={config} updateConfig={updateConfig} />
          </section>

          <section className="p-6 bg-gray-900 rounded-xl border border-gray-800">
            <SymbolsSection config={config} updateConfig={updateConfig} />
          </section>

          <section className="p-6 bg-gray-900 rounded-xl border border-gray-800">
            <Tier1AgentsSection config={config} updateConfig={updateConfig} />
          </section>

          {/* Collapsible sections */}
          <div className="space-y-3">
            <Tier2AgentsSection config={config} updateConfig={updateConfig} />
            <Tier3PoolSection config={config} updateConfig={updateConfig} />
            <MarketMakerSection config={config} updateConfig={updateConfig} />
            <NoiseTraderSection config={config} updateConfig={updateConfig} />
            <QuantStrategySection config={config} updateConfig={updateConfig} />
            <EventsSection config={config} updateConfig={updateConfig} />
          </div>
        </div>
      </main>
    </div>
  );
}
