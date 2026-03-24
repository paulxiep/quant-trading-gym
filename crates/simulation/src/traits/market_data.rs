//! Market data provider trait.
//!
//! Defines the interface for accessing historical market data including
//! candles, trades, and technical indicators.

use std::collections::{HashMap, VecDeque};

use quant::IndicatorSnapshot;
use types::{Candle, Symbol, Trade};

/// Provides access to historical market data.
///
/// This trait abstracts market data access, enabling:
/// - Agents to use mock data in unit tests
/// - Historical replay from files
/// - Decoupling from `Simulation` concrete type
pub trait MarketDataProvider {
    /// Get historical candles for a specific symbol (first contiguous slice).
    ///
    /// For VecDeque, this returns only the first contiguous portion.
    /// Use `candles_for_mut` if you need all candles as contiguous slice.
    fn candles_for(&self, symbol: &Symbol) -> &[Candle];

    /// Get historical candles for a specific symbol, making them contiguous.
    ///
    /// This calls `make_contiguous()` on the underlying VecDeque to ensure
    /// all candles are returned as a single slice.
    fn candles_for_mut(&mut self, symbol: &Symbol) -> &[Candle];

    /// Get all candles across all symbols.
    fn all_candles(&self) -> &HashMap<Symbol, VecDeque<Candle>>;

    /// Get recent trades for a specific symbol.
    fn recent_trades_for(&self, symbol: &Symbol) -> &[Trade];

    /// Get all recent trades across all symbols.
    fn all_recent_trades(&self) -> &HashMap<Symbol, Vec<Trade>>;

    /// Build indicator snapshot for current tick (all symbols).
    ///
    /// Requires mutable access to make candles contiguous for computation.
    /// V5.5: Single source of truth - returns IndicatorSnapshot with enum keys.
    fn build_indicator_snapshot(&mut self) -> IndicatorSnapshot;
}
