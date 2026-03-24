/**
 * Symbols list editor component
 *
 * SoC: Handles only symbol array editing
 */

import type { SymbolSpec, Sector } from '../../types';
import { SECTORS } from '../../types';
import { Button, TextInput, NumberInput, SelectInput } from '../ui';

interface SymbolsEditorProps {
  symbols: SymbolSpec[];
  onChange: (symbols: SymbolSpec[]) => void;
}

export function SymbolsEditor({ symbols, onChange }: SymbolsEditorProps) {
  const updateSymbol = (index: number, updates: Partial<SymbolSpec>) => {
    const newSymbols = [...symbols];
    newSymbols[index] = { ...newSymbols[index], ...updates };
    onChange(newSymbols);
  };

  const addSymbol = () => {
    onChange([...symbols, { symbol: 'NEW', initialPrice: 100.0, sector: 'Tech' }]);
  };

  const removeSymbol = (index: number) => {
    onChange(symbols.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-3">
      {symbols.map((sym, index) => (
        <div key={index} className="flex gap-3 items-end">
          <TextInput
            label="Symbol"
            value={sym.symbol}
            onChange={(value) => updateSymbol(index, { symbol: value })}
            className="flex-1"
          />
          <NumberInput
            label="Initial Price"
            value={sym.initialPrice}
            onChange={(value) => updateSymbol(index, { initialPrice: value })}
            step={0.01}
            min={0.01}
            className="w-32"
          />
          <SelectInput<Sector>
            label="Sector"
            value={sym.sector}
            options={SECTORS}
            onChange={(value) => updateSymbol(index, { sector: value })}
            className="w-40"
          />
          <Button
            variant="ghost"
            onClick={() => removeSymbol(index)}
            className="text-loss hover:text-red-400 px-2"
            disabled={symbols.length <= 1}
          >
            ✕
          </Button>
        </div>
      ))}
      <Button variant="secondary" onClick={addSymbol} className="mt-2">
        + Add Symbol
      </Button>
    </div>
  );
}
