/**
 * Landing page - entry point with hero and navigation
 *
 * SoC: Only handles landing UI and navigation
 */

import { Link } from 'react-router-dom';
import { Button } from '../components';

export function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-950 flex flex-col">
      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center px-8">
        <div className="text-center max-w-3xl">
          {/* Logo/Title */}
          <h1 className="text-5xl md:text-6xl font-bold text-gray-100 mb-4">Quant Trading Gym</h1>
          <p className="text-xl md:text-2xl text-gray-400 mb-8">
            A high-performance market simulation with 100k+ AI agents
          </p>

          {/* Feature bullets */}
          <ul className="text-left text-gray-300 mb-12 space-y-3 max-w-md mx-auto">
            <li className="flex items-start gap-3">
              <span className="text-primary-500 mt-1">▸</span>
              <span>Multi-symbol trading with realistic market microstructure</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-primary-500 mt-1">▸</span>
              <span>
                Tiered agent architecture: Market Makers, Quant Strategies, Reactive Traders
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-primary-500 mt-1">▸</span>
              <span>Real-time indicators: RSI, MACD, Bollinger, ATR, and more</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-primary-500 mt-1">▸</span>
              <span>News events and fundamental-driven price discovery</span>
            </li>
          </ul>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/sim">
              <Button className="w-full sm:w-auto px-8 py-3 text-lg">Run Simulation</Button>
            </Link>
            <Link to="/config">
              <Button variant="secondary" className="w-full sm:w-auto px-8 py-3 text-lg">
                Configure
              </Button>
            </Link>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-6 text-center text-gray-500 text-sm">
        <p>Built with Rust • React • WebSocket</p>
      </footer>
    </div>
  );
}
