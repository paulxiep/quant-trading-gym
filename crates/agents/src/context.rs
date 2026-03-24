//! Strategy context for multi-symbol trading (V2.3).
//!
//! This module provides [`StrategyContext`], the unified context passed to agents
//! each tick. It replaces the simpler `MarketData` struct with a more flexible
//! design that supports multi-symbol trading.
//!
//! # Architecture
//!
//! `StrategyContext` uses references to avoid cloning large data structures:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    StrategyContext<'a>                       │
//! │  ┌─────────────────┐  ┌─────────────────┐                   │
//! │  │ &'a dyn MarketView │  │ &'a IndicatorSnapshot │           │
//! │  └─────────────────┘  └─────────────────┘                   │
//! │  ┌─────────────────────────────────────────┐                │
//! │  │ &'a HashMap<Symbol, Vec<Candle>>         │                │
//! │  └─────────────────────────────────────────┘                │
//! │  ┌─────────────────────────────────────────┐                │
//! │  │ &'a HashMap<Symbol, Vec<Trade>>          │                │
//! │  └─────────────────────────────────────────┘                │
//! │  tick: Tick, timestamp: Timestamp                           │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Migration from MarketData
//!
//! ```ignore
//! // Old (MarketData)
//! let mid = market.mid_price();
//! let rsi = market.get_indicator(IndicatorType::Rsi(14));
//!
//! // New (StrategyContext)
//! let mid = ctx.mid_price(&symbol);
//! let rsi = ctx.get_indicator(&symbol, IndicatorType::Rsi(14));
//! ```
//!
//! # Borrow Checker Considerations
//!
//! The context holds references, so it must be dropped before the underlying
//! data can be modified. The simulation uses a two-phase tick:
//! 1. Build context, call agents, collect actions
//! 2. Drop context, process orders

use std::collections::HashMap;

use quant::IndicatorSnapshot;
use sim_core::MarketView;
use types::{BookSnapshot, Candle, IndicatorType, Price, Quantity, Symbol, Tick, Timestamp, Trade};

use crate::ml_cache::MlPredictionCache;
use crate::tier1::ml::ClassProbabilities;

// =============================================================================
// StrategyContext
// =============================================================================

/// Unified context passed to agents each tick.
///
/// Provides read-only access to:
/// - Market state via `MarketView` (prices, book snapshots)
/// - Historical candles per symbol
/// - Pre-computed indicators per symbol
/// - Recent trades per symbol
/// - Current tick and timestamp
///
/// # Lifetimes
///
/// The `'a` lifetime ties the context to the simulation's data structures.
/// Agents cannot store references from the context across ticks.
///
/// # Example
///
/// ```ignore
/// fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
///     let symbol = &self.config.symbol;
///     
///     // Access market prices
///     let mid = ctx.mid_price(symbol);
///     let spread = ctx.spread(symbol);
///     
///     // Access indicators
///     let rsi = ctx.get_indicator(symbol, IndicatorType::Rsi(14));
///     
///     // Access candles
///     let last_close = ctx.last_candle(symbol).map(|c| c.close);
///     
///     // Make trading decision...
///     AgentAction::none()
/// }
/// ```
pub struct StrategyContext<'a> {
    /// Current simulation tick.
    pub tick: Tick,

    /// Current timestamp (wall clock).
    pub timestamp: Timestamp,

    /// Read-only market view for price/book access.
    market: &'a dyn MarketView,

    /// Historical candles per symbol.
    candles: &'a HashMap<Symbol, Vec<Candle>>,

    /// Pre-computed indicators for current tick (V5.5).
    /// Single source of truth with enum keys for type safety.
    /// Includes all component-level values (MacdLine, MacdSignal, etc.).
    indicators: &'a IndicatorSnapshot,

    /// Recent trades per symbol (most recent first).
    recent_trades: &'a HashMap<Symbol, Vec<Trade>>,

    /// Active news events (V2.4).
    events: &'a [news::NewsEvent],

    /// Symbol fundamentals for fair value lookups (V2.4).
    fundamentals: &'a news::SymbolFundamentals,

    /// Centralized ML prediction cache (V5.6).
    /// When present, agents can retrieve cached predictions instead of computing locally.
    ml_cache: Option<&'a MlPredictionCache>,
}

impl<'a> StrategyContext<'a> {
    /// Create a new strategy context.
    ///
    /// This is typically called by the simulation runner at the start of each tick.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tick: Tick,
        timestamp: Timestamp,
        market: &'a dyn MarketView,
        candles: &'a HashMap<Symbol, Vec<Candle>>,
        indicators: &'a IndicatorSnapshot,
        recent_trades: &'a HashMap<Symbol, Vec<Trade>>,
        events: &'a [news::NewsEvent],
        fundamentals: &'a news::SymbolFundamentals,
    ) -> Self {
        Self {
            tick,
            timestamp,
            market,
            candles,
            indicators,
            recent_trades,
            events,
            fundamentals,
            ml_cache: None,
        }
    }

    /// Create a new strategy context with ML prediction cache (V5.6).
    ///
    /// This constructor enables centralized ML prediction caching, allowing
    /// agents to retrieve pre-computed predictions instead of computing locally.
    #[allow(clippy::too_many_arguments)]
    pub fn with_ml_cache(
        tick: Tick,
        timestamp: Timestamp,
        market: &'a dyn MarketView,
        candles: &'a HashMap<Symbol, Vec<Candle>>,
        indicators: &'a IndicatorSnapshot,
        recent_trades: &'a HashMap<Symbol, Vec<Trade>>,
        events: &'a [news::NewsEvent],
        fundamentals: &'a news::SymbolFundamentals,
        ml_cache: &'a MlPredictionCache,
    ) -> Self {
        Self {
            tick,
            timestamp,
            market,
            candles,
            indicators,
            recent_trades,
            events,
            fundamentals,
            ml_cache: Some(ml_cache),
        }
    }

    // =========================================================================
    // Market Access (delegated to MarketView)
    // =========================================================================

    /// Get all symbols available in the market.
    pub fn symbols(&self) -> Vec<Symbol> {
        self.market.symbols()
    }

    /// Check if a symbol exists in the market.
    pub fn has_symbol(&self, symbol: &Symbol) -> bool {
        self.market.has_symbol(symbol)
    }

    /// Get the mid price for a symbol.
    pub fn mid_price(&self, symbol: &Symbol) -> Option<Price> {
        self.market.mid_price(symbol)
    }

    /// Get the best bid price for a symbol.
    pub fn best_bid(&self, symbol: &Symbol) -> Option<Price> {
        self.market.best_bid(symbol)
    }

    /// Get the best ask price for a symbol.
    pub fn best_ask(&self, symbol: &Symbol) -> Option<Price> {
        self.market.best_ask(symbol)
    }

    /// Get the last traded price for a symbol.
    pub fn last_price(&self, symbol: &Symbol) -> Option<Price> {
        self.market.last_price(symbol)
    }

    /// Get the spread for a symbol.
    pub fn spread(&self, symbol: &Symbol) -> Option<Price> {
        self.market.spread(symbol)
    }

    /// Get a book snapshot for a symbol.
    pub fn book_snapshot(&self, symbol: &Symbol, depth: usize) -> Option<BookSnapshot> {
        self.market
            .snapshot(symbol, depth, self.timestamp, self.tick)
    }

    /// Get total bid volume for a symbol.
    pub fn total_bid_volume(&self, symbol: &Symbol) -> Quantity {
        self.market.total_bid_volume(symbol)
    }

    /// Get total ask volume for a symbol.
    pub fn total_ask_volume(&self, symbol: &Symbol) -> Quantity {
        self.market.total_ask_volume(symbol)
    }

    // =========================================================================
    // Candle Access
    // =========================================================================

    /// Get all candles for a symbol.
    pub fn candles(&self, symbol: &Symbol) -> &[Candle] {
        self.candles
            .get(symbol)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get the most recent candle for a symbol.
    pub fn last_candle(&self, symbol: &Symbol) -> Option<&Candle> {
        self.candles.get(symbol).and_then(|v| v.last())
    }

    /// Get the N most recent candles for a symbol.
    pub fn recent_candles(&self, symbol: &Symbol, n: usize) -> &[Candle] {
        self.candles
            .get(symbol)
            .map(|v| {
                let start = v.len().saturating_sub(n);
                &v[start..]
            })
            .unwrap_or(&[])
    }

    // =========================================================================
    // Indicator Access
    // =========================================================================

    /// Get a specific indicator value for a symbol.
    ///
    /// V5.5: Uses enum keys for type safety. Access MACD/BB components directly:
    /// ```ignore
    /// ctx.get_indicator(&symbol, IndicatorType::MACD_LINE_STANDARD)
    /// ctx.get_indicator(&symbol, IndicatorType::BOLLINGER_UPPER_STANDARD)
    /// ```
    pub fn get_indicator(&self, symbol: &Symbol, indicator_type: IndicatorType) -> Option<f64> {
        self.indicators.get(symbol, indicator_type)
    }

    /// Get all indicator values for a symbol.
    pub fn get_all_indicators(&self, symbol: &Symbol) -> Option<&HashMap<IndicatorType, f64>> {
        self.indicators.get_symbol(symbol)
    }

    /// Check if indicator data is available.
    pub fn has_indicators(&self) -> bool {
        self.indicators.tick > 0
    }

    // =========================================================================
    // ML Cache Access (V5.6)
    // =========================================================================

    /// Get the ML prediction cache if available.
    ///
    /// Returns `Some` if the simulation has a model registry and computed
    /// predictions for this tick. Returns `None` for backward compatibility
    /// when no registry is configured.
    pub fn ml_cache(&self) -> Option<&MlPredictionCache> {
        self.ml_cache
    }

    /// Get a cached ML prediction for a model-symbol pair.
    ///
    /// This is a convenience method that combines cache lookup.
    /// Returns `None` if either the cache is not available or the
    /// prediction is not cached for this model-symbol pair.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the ML model
    /// * `symbol` - Symbol to get prediction for
    ///
    /// # Returns
    ///
    /// `Some([p_sell, p_hold, p_buy])` if cached, `None` otherwise.
    pub fn get_ml_prediction(
        &self,
        model_name: &str,
        symbol: &Symbol,
    ) -> Option<ClassProbabilities> {
        self.ml_cache
            .and_then(|cache| cache.get_prediction(model_name, symbol))
    }

    /// Check if ML prediction cache is available.
    pub fn has_ml_cache(&self) -> bool {
        self.ml_cache.is_some()
    }

    // =========================================================================
    // Trade Access
    // =========================================================================

    /// Get recent trades for a symbol (most recent first).
    pub fn recent_trades(&self, symbol: &Symbol) -> &[Trade] {
        self.recent_trades
            .get(symbol)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get the most recent trade for a symbol.
    pub fn last_trade(&self, symbol: &Symbol) -> Option<&Trade> {
        self.recent_trades.get(symbol).and_then(|v| v.first())
    }

    // =========================================================================
    // Convenience Methods for Single-Symbol Strategies
    // =========================================================================

    /// Get the primary symbol (first symbol in the market).
    ///
    /// Useful for single-symbol strategies that don't want to track the symbol.
    pub fn primary_symbol(&self) -> Option<Symbol> {
        self.market.symbols().into_iter().next()
    }

    /// Get mid price for the primary symbol.
    pub fn primary_mid_price(&self) -> Option<Price> {
        self.primary_symbol().and_then(|s| self.mid_price(&s))
    }

    /// Get last price for the primary symbol.
    pub fn primary_last_price(&self) -> Option<Price> {
        self.primary_symbol().and_then(|s| self.last_price(&s))
    }

    // =========================================================================
    // News & Fundamentals Access (V2.4)
    // =========================================================================

    /// Get all active news events.
    pub fn active_events(&self) -> &[news::NewsEvent] {
        self.events
    }

    /// Get active events affecting a specific symbol.
    pub fn events_for_symbol(&self, symbol: &Symbol) -> Vec<&news::NewsEvent> {
        self.events
            .iter()
            .filter(|e| e.symbol() == Some(symbol))
            .collect()
    }

    /// Get active events affecting a specific sector.
    pub fn events_for_sector(&self, sector: types::Sector) -> Vec<&news::NewsEvent> {
        self.events
            .iter()
            .filter(|e| e.sector() == Some(sector))
            .collect()
    }

    /// Get the fair value for a symbol based on fundamentals.
    ///
    /// Returns `None` if no fundamentals exist for the symbol.
    pub fn fair_value(&self, symbol: &Symbol) -> Option<Price> {
        self.fundamentals.fair_value(symbol)
    }

    /// Get the fundamentals for a symbol.
    pub fn get_fundamentals(&self, symbol: &Symbol) -> Option<&news::Fundamentals> {
        self.fundamentals.get(symbol)
    }

    /// Get the macro environment (risk-free rate, equity risk premium).
    pub fn macro_env(&self) -> &news::MacroEnvironment {
        &self.fundamentals.macro_env
    }

    /// Calculate aggregate sentiment for a symbol from active events.
    ///
    /// Combines direct symbol events and sector events with decay.
    pub fn symbol_sentiment(&self, symbol: &Symbol) -> f64 {
        let direct: f64 = self
            .events
            .iter()
            .filter(|e| e.symbol() == Some(symbol))
            .map(|e| e.effective_sentiment(self.tick))
            .sum();

        // TODO: Add sector sentiment when sector model is available in context
        direct
    }
}

impl std::fmt::Debug for StrategyContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StrategyContext")
            .field("tick", &self.tick)
            .field("timestamp", &self.timestamp)
            .field("symbols", &self.market.symbols())
            .field("candle_symbols", &self.candles.keys().collect::<Vec<_>>())
            .field("active_events", &self.events.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use news::SymbolFundamentals;
    use quant::IndicatorSnapshot;
    use sim_core::SingleSymbolMarket;
    use types::{AgentId, Order, OrderId, OrderSide, Quantity};

    fn setup_test_book() -> sim_core::OrderBook {
        let mut book = sim_core::OrderBook::new("TEST");
        let mut bid = Order::limit(
            AgentId(1),
            "TEST",
            OrderSide::Buy,
            Price::from_float(99.0),
            Quantity(100),
        );
        bid.id = OrderId(1);
        let mut ask = Order::limit(
            AgentId(2),
            "TEST",
            OrderSide::Sell,
            Price::from_float(101.0),
            Quantity(100),
        );
        ask.id = OrderId(2);
        book.add_order(bid).unwrap();
        book.add_order(ask).unwrap();
        book
    }

    #[test]
    fn test_context_market_access() {
        let book = setup_test_book();
        let market = SingleSymbolMarket::new(&book);
        let candles = HashMap::new();
        let indicators = IndicatorSnapshot::new(100);
        let recent_trades = HashMap::new();
        let events = vec![];
        let fundamentals = SymbolFundamentals::default();

        let ctx = StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let symbol = "TEST".to_string();
        assert!(ctx.has_symbol(&symbol));
        assert_eq!(ctx.best_bid(&symbol), Some(Price::from_float(99.0)));
        assert_eq!(ctx.best_ask(&symbol), Some(Price::from_float(101.0)));
        assert_eq!(ctx.mid_price(&symbol), Some(Price::from_float(100.0)));
    }

    #[test]
    fn test_context_candles() {
        let book = setup_test_book();
        let market = SingleSymbolMarket::new(&book);
        let indicators = IndicatorSnapshot::new(100);
        let recent_trades = HashMap::new();
        let events = vec![];
        let fundamentals = SymbolFundamentals::default();

        let symbol = "TEST".to_string();
        let mut candles = HashMap::new();
        candles.insert(
            symbol.clone(),
            vec![
                Candle::new(
                    "TEST",
                    Price::from_float(100.0),
                    Price::from_float(102.0),
                    Price::from_float(98.0),
                    Price::from_float(101.0),
                    Quantity(1000),
                    900,
                    90,
                ),
                Candle::new(
                    "TEST",
                    Price::from_float(101.0),
                    Price::from_float(103.0),
                    Price::from_float(99.0),
                    Price::from_float(102.0),
                    Quantity(1200),
                    1000,
                    100,
                ),
            ],
        );

        let ctx = StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        assert_eq!(ctx.candles(&symbol).len(), 2);
        assert_eq!(
            ctx.last_candle(&symbol).map(|c| c.close),
            Some(Price::from_float(102.0))
        );
        assert_eq!(ctx.recent_candles(&symbol, 1).len(), 1);
    }

    #[test]
    fn test_context_primary_symbol() {
        let book = setup_test_book();
        let market = SingleSymbolMarket::new(&book);
        let candles = HashMap::new();
        let indicators = IndicatorSnapshot::new(100);
        let recent_trades = HashMap::new();
        let events = vec![];
        let fundamentals = SymbolFundamentals::default();

        let ctx = StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        assert_eq!(ctx.primary_symbol(), Some("TEST".to_string()));
        assert_eq!(ctx.primary_mid_price(), Some(Price::from_float(100.0)));
    }
}
