/**
 * Time controls component (V4.4).
 *
 * Provides simulation playback controls via WebSocket commands.
 * Shows tick counter, play/pause/step buttons, speed control.
 *
 * Declarative: Props-driven state
 * Modular: Self-contained controls
 * SoC: Only handles time control UI
 */

import type { ConnectionState, TickData } from '../../hooks/useWebSocket';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TimeControlsProps {
  /** Current tick data from WebSocket. */
  tickData: TickData | null;
  /** WebSocket connection state. */
  connectionState: ConnectionState;
  /** Whether simulation is running. */
  isRunning?: boolean;
  /** Send command callback. */
  onCommand: (command: 'Start' | 'Pause' | 'Toggle' | 'Step' | 'Quit') => void;
  /** Connect callback. */
  onConnect?: () => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Get connection status color. */
function getConnectionColor(state: ConnectionState): string {
  switch (state) {
    case 'connected':
      return 'bg-green-500';
    case 'connecting':
      return 'bg-yellow-500';
    case 'error':
      return 'bg-red-500';
    default:
      return 'bg-gray-500';
  }
}

/** Format tick number with commas. */
function formatTick(tick: number): string {
  return tick.toLocaleString();
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface ControlButtonProps {
  onClick: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'secondary' | 'danger';
  title?: string;
  children: React.ReactNode;
}

function ControlButton({
  onClick,
  disabled = false,
  variant = 'secondary',
  title,
  children,
}: ControlButtonProps) {
  const baseStyles =
    'p-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed';
  const variantStyles = {
    primary: 'bg-primary-600 hover:bg-primary-700 text-white',
    secondary: 'bg-gray-700 hover:bg-gray-600 text-gray-100',
    danger: 'bg-red-600 hover:bg-red-700 text-white',
  };

  return (
    <button
      className={`${baseStyles} ${variantStyles[variant]}`}
      onClick={onClick}
      disabled={disabled}
      title={title}
    >
      {children}
    </button>
  );
}

// Icons as inline SVG
function PlayIcon() {
  return (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path d="M8 5v14l11-7z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
    </svg>
  );
}

function StepIcon() {
  return (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path d="M6 6h12v12H6z" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function TimeControls({
  tickData,
  connectionState,
  isRunning = false,
  onCommand,
  onConnect,
}: TimeControlsProps) {
  const isConnected = connectionState === 'connected';
  const isConnecting = connectionState === 'connecting';

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
      <div className="flex items-center justify-between gap-4">
        {/* Connection status */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${getConnectionColor(connectionState)}`} />
            <span className="text-gray-400 text-sm capitalize">{connectionState}</span>
          </div>

          {!isConnected && onConnect && (
            <button
              className="text-primary-400 hover:text-primary-300 text-sm"
              onClick={onConnect}
              disabled={isConnecting}
            >
              {isConnecting ? 'Connecting...' : 'Connect'}
            </button>
          )}
        </div>

        {/* Tick counter */}
        <div className="flex items-center gap-6">
          <div className="text-center">
            <div className="text-2xl font-mono text-white">
              {tickData ? formatTick(tickData.tick) : '—'}
            </div>
            <div className="text-gray-500 text-xs uppercase">Tick</div>
          </div>

          {/* Stats */}
          {tickData && (
            <>
              <div className="text-center">
                <div className="text-lg font-mono text-gray-300">
                  {tickData.total_trades.toLocaleString()}
                </div>
                <div className="text-gray-500 text-xs uppercase">Trades</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-mono text-gray-300">
                  {tickData.total_orders.toLocaleString()}
                </div>
                <div className="text-gray-500 text-xs uppercase">Total Orders</div>
              </div>
            </>
          )}
        </div>

        {/* Control buttons */}
        <div className="flex items-center gap-2">
          {/* Play/Pause toggle */}
          <ControlButton
            onClick={() => onCommand('Toggle')}
            disabled={!isConnected}
            variant={isRunning ? 'secondary' : 'primary'}
            title={isRunning ? 'Pause simulation' : 'Start simulation'}
          >
            {isRunning ? <PauseIcon /> : <PlayIcon />}
          </ControlButton>

          {/* Step */}
          <ControlButton
            onClick={() => onCommand('Step')}
            disabled={!isConnected || isRunning}
            variant="secondary"
            title="Step one tick"
          >
            <StepIcon />
          </ControlButton>

          {/* Stop */}
          <ControlButton
            onClick={() => onCommand('Quit')}
            disabled={!isConnected}
            variant="danger"
            title="Stop simulation"
          >
            <StopIcon />
          </ControlButton>
        </div>
      </div>

      {/* Progress bar placeholder */}
      {tickData && (
        <div className="mt-3 pt-3 border-t border-gray-800">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Simulation Progress</span>
            <span>{tickData.agents_called} agents active</span>
          </div>
          <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-500 transition-all duration-300"
              style={{
                width: `${Math.min(100, (tickData.tick / 1000) * 100)}%`,
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
