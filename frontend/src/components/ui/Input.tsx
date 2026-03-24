/**
 * Form input components
 *
 * Modular: Each input type is self-contained
 * Declarative: Props describe field, not DOM manipulation
 */

import type { InputHTMLAttributes, SelectHTMLAttributes } from 'react';

interface NumberInputProps extends Omit<
  InputHTMLAttributes<HTMLInputElement>,
  'type' | 'onChange'
> {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
}

export function NumberInput({
  label,
  value,
  onChange,
  step = 1,
  className = '',
  ...props
}: NumberInputProps) {
  return (
    <label className={`block ${className}`}>
      <span className="text-sm text-gray-400 block mb-1">{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        step={step}
        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
        {...props}
      />
    </label>
  );
}

interface TextInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> {
  label: string;
  value: string;
  onChange: (value: string) => void;
}

export function TextInput({ label, value, onChange, className = '', ...props }: TextInputProps) {
  return (
    <label className={`block ${className}`}>
      <span className="text-sm text-gray-400 block mb-1">{label}</span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
        {...props}
      />
    </label>
  );
}

interface SelectInputProps<T extends string> extends Omit<
  SelectHTMLAttributes<HTMLSelectElement>,
  'onChange'
> {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (value: T) => void;
}

export function SelectInput<T extends string>({
  label,
  value,
  options,
  onChange,
  className = '',
  ...props
}: SelectInputProps<T>) {
  return (
    <label className={`block ${className}`}>
      <span className="text-sm text-gray-400 block mb-1">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
        {...props}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

interface CheckboxInputProps extends Omit<
  InputHTMLAttributes<HTMLInputElement>,
  'type' | 'onChange'
> {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export function CheckboxInput({
  label,
  checked,
  onChange,
  className = '',
  ...props
}: CheckboxInputProps) {
  return (
    <label className={`flex items-center gap-2 cursor-pointer ${className}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-4 h-4 rounded bg-gray-800 border-gray-700 text-primary-600 focus:ring-primary-500 focus:ring-offset-gray-900"
        {...props}
      />
      <span className="text-sm text-gray-300">{label}</span>
    </label>
  );
}
