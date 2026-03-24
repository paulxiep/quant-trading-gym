//! Indicator engine for batch computation and caching.
//!
//! The engine manages registered indicators and provides efficient computation
//! with per-tick caching to avoid redundant calculations.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────┐
//! │   IndicatorEngine   │
//! │  ┌───────────────┐  │
//! │  │  Registered   │  │
//! │  │  Indicators   │  │
//! │  └───────────────┘  │
//! │         │           │
//! │         ▼           │
//! │  ┌───────────────┐  │
//! │  │ IndicatorCache│  │
//! │  │ (per-tick)    │  │
//! │  └───────────────┘  │
//! └─────────────────────┘
//! ```
//!
//! # Usage
//!
//! Strategies declare their required indicators. The engine lazily computes
//! indicators on first access and caches values for the current tick.
//!
//! ```ignore
//! let mut engine = IndicatorEngine::new();
//! engine.register(IndicatorType::Sma(20));
//! engine.register(IndicatorType::Rsi(14));
//!
//! // On each tick, create a fresh cache
//! let mut cache = engine.create_cache();
//! let sma_value = cache.get_or_compute(
//!     &symbol,
//!     IndicatorType::Sma(20),
//!     current_tick,
//!     &candles
//! );
//! ```

use std::collections::HashMap;

use types::{Candle, IndicatorType, Symbol, Tick};

use crate::indicators::{Indicator, create_indicator};

/// Engine for managing and computing technical indicators.
///
/// Maintains a registry of indicator types and provides factory methods
/// for creating caches.
#[derive(Default)]
pub struct IndicatorEngine {
    /// Registered indicators (keyed by type).
    indicators: HashMap<IndicatorType, Box<dyn Indicator>>,
}

impl IndicatorEngine {
    /// Create a new empty indicator engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an engine with common indicators pre-registered.
    /// Periods optimized for batch auction (1 tick = 1 session).
    ///
    /// V5.5: Registers component-level indicators for MACD and Bollinger Bands.
    /// Use `compute_all_indicators()` from the indicators module for single-pass
    /// computation of all standard indicators.
    pub fn with_common_indicators() -> Self {
        let mut engine = Self::new();

        // Moving averages (8/16 spread)
        engine.register(IndicatorType::Sma(8));
        engine.register(IndicatorType::Sma(16));
        engine.register(IndicatorType::Ema(8)); // MACD fast
        engine.register(IndicatorType::Ema(16)); // MACD slow

        // Momentum
        engine.register(IndicatorType::Rsi(8));
        // MACD components (8/16/4)
        engine.register(IndicatorType::MACD_LINE_STANDARD);
        engine.register(IndicatorType::MACD_SIGNAL_STANDARD);
        engine.register(IndicatorType::MACD_HISTOGRAM_STANDARD);

        // Volatility
        engine.register(IndicatorType::Atr(8));
        // Bollinger components (12 period, 200bp = 2.0 std dev)
        engine.register(IndicatorType::BOLLINGER_UPPER_STANDARD);
        engine.register(IndicatorType::BOLLINGER_MIDDLE_STANDARD);
        engine.register(IndicatorType::BOLLINGER_LOWER_STANDARD);

        engine
    }

    /// Register an indicator type with the engine.
    ///
    /// If the indicator is already registered, this is a no-op.
    pub fn register(&mut self, indicator_type: IndicatorType) {
        self.indicators
            .entry(indicator_type)
            .or_insert_with(|| create_indicator(indicator_type));
    }

    /// Check if an indicator type is registered.
    pub fn is_registered(&self, indicator_type: &IndicatorType) -> bool {
        self.indicators.contains_key(indicator_type)
    }

    /// Get the list of registered indicator types.
    pub fn registered_types(&self) -> Vec<IndicatorType> {
        self.indicators.keys().copied().collect()
    }

    /// Get a reference to a registered indicator.
    pub fn get(&self, indicator_type: &IndicatorType) -> Option<&dyn Indicator> {
        self.indicators.get(indicator_type).map(|b| b.as_ref())
    }

    /// Create a new indicator cache.
    pub fn create_cache(&self) -> IndicatorCache {
        IndicatorCache::new()
    }

    /// Compute an indicator value directly (without caching).
    pub fn compute(&self, indicator_type: &IndicatorType, candles: &[Candle]) -> Option<f64> {
        self.indicators
            .get(indicator_type)
            .and_then(|ind| ind.calculate(candles))
    }

    /// Compute all registered indicators for a symbol.
    ///
    /// Returns a map of indicator type to value.
    pub fn compute_all(&self, candles: &[Candle]) -> HashMap<IndicatorType, f64> {
        self.indicators
            .iter()
            .filter_map(|(itype, indicator)| indicator.calculate(candles).map(|v| (*itype, v)))
            .collect()
    }
}

impl std::fmt::Debug for IndicatorEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndicatorEngine")
            .field("registered_count", &self.indicators.len())
            .field("types", &self.indicators.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Cache for indicator values within a single tick.
///
/// Provides lazy computation with memoization. Values are cached by
/// (symbol, indicator_type, tick) key, ensuring indicators are computed
/// at most once per tick per symbol.
#[derive(Debug, Default)]
pub struct IndicatorCache {
    /// Cached values: (symbol, indicator_type) → (tick, value).
    cache: HashMap<(Symbol, IndicatorType), (Tick, f64)>,
}

impl IndicatorCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a cached value if it exists for the current tick.
    pub fn get(&self, symbol: &Symbol, indicator_type: IndicatorType, tick: Tick) -> Option<f64> {
        self.cache
            .get(&(symbol.clone(), indicator_type))
            .filter(|(cached_tick, _)| *cached_tick == tick)
            .map(|(_, value)| *value)
    }

    /// Insert a computed value into the cache.
    pub fn insert(
        &mut self,
        symbol: Symbol,
        indicator_type: IndicatorType,
        tick: Tick,
        value: f64,
    ) {
        self.cache.insert((symbol, indicator_type), (tick, value));
    }

    /// Get a cached value or compute it if not present.
    ///
    /// This is the primary interface for strategies to access indicators.
    /// Values are computed lazily and cached for the duration of the tick.
    pub fn get_or_compute(
        &mut self,
        symbol: &Symbol,
        indicator_type: IndicatorType,
        tick: Tick,
        candles: &[Candle],
        engine: &IndicatorEngine,
    ) -> Option<f64> {
        // Check cache first
        if let Some(value) = self.get(symbol, indicator_type, tick) {
            return Some(value);
        }

        // Compute and cache
        let value = engine.compute(&indicator_type, candles)?;
        self.insert(symbol.clone(), indicator_type, tick, value);
        Some(value)
    }

    /// Clear all cached values.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get all cached values for a symbol at a given tick.
    pub fn get_all_for_symbol(&self, symbol: &Symbol, tick: Tick) -> HashMap<IndicatorType, f64> {
        self.cache
            .iter()
            .filter(|((s, _), (t, _))| s == symbol && *t == tick)
            .map(|((_, itype), (_, value))| (*itype, *value))
            .collect()
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

/// Multi-symbol indicator snapshot for a single tick.
///
/// Stores computed indicator values for all symbols at a point in time.
/// Used to pass indicator data to strategies.
#[derive(Debug, Clone, Default)]
pub struct IndicatorSnapshot {
    /// Tick when this snapshot was taken.
    pub tick: Tick,
    /// Values by symbol and indicator type.
    values: HashMap<Symbol, HashMap<IndicatorType, f64>>,
}

impl IndicatorSnapshot {
    /// Create a new empty snapshot.
    pub fn new(tick: Tick) -> Self {
        Self {
            tick,
            values: HashMap::new(),
        }
    }

    /// Create a snapshot from a pre-built map.
    pub fn from_map(tick: Tick, values: HashMap<Symbol, HashMap<IndicatorType, f64>>) -> Self {
        Self { tick, values }
    }

    /// Add indicator values for a symbol.
    pub fn insert(&mut self, symbol: Symbol, indicators: HashMap<IndicatorType, f64>) {
        self.values.insert(symbol, indicators);
    }

    /// Get all indicators for a symbol.
    pub fn get_symbol(&self, symbol: &Symbol) -> Option<&HashMap<IndicatorType, f64>> {
        self.values.get(symbol)
    }

    /// Get a specific indicator value for a symbol.
    pub fn get(&self, symbol: &Symbol, indicator_type: IndicatorType) -> Option<f64> {
        self.values
            .get(symbol)
            .and_then(|m| m.get(&indicator_type))
            .copied()
    }

    /// Get all symbols in this snapshot.
    pub fn symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.values.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{Price, Quantity};

    fn make_candles(closes: &[f64]) -> Vec<Candle> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &close)| Candle {
                symbol: "TEST".to_string(),
                open: Price::from_float(close),
                high: Price::from_float(close + 1.0),
                low: Price::from_float(close - 1.0),
                close: Price::from_float(close),
                volume: Quantity(1000),
                timestamp: i as u64,
                tick: i as u64,
            })
            .collect()
    }

    #[test]
    fn test_engine_registration() {
        let mut engine = IndicatorEngine::new();
        engine.register(IndicatorType::Sma(20));
        engine.register(IndicatorType::Rsi(14));

        assert!(engine.is_registered(&IndicatorType::Sma(20)));
        assert!(engine.is_registered(&IndicatorType::Rsi(14)));
        assert!(!engine.is_registered(&IndicatorType::Ema(20)));
    }

    #[test]
    fn test_engine_compute() {
        let mut engine = IndicatorEngine::new();
        engine.register(IndicatorType::Sma(3));

        let candles = make_candles(&[10.0, 11.0, 12.0, 13.0, 14.0]);
        let result = engine.compute(&IndicatorType::Sma(3), &candles);

        assert!(result.is_some());
        assert!((result.unwrap() - 13.0).abs() < 0.001);
    }

    #[test]
    fn test_engine_with_common() {
        let engine = IndicatorEngine::with_common_indicators();

        // V5.5: Updated to geometric spread (8/16) optimized for batch auction
        // Now uses component-level indicators for MACD and Bollinger
        assert!(engine.is_registered(&IndicatorType::Sma(8)));
        assert!(engine.is_registered(&IndicatorType::Sma(16)));
        assert!(engine.is_registered(&IndicatorType::Rsi(8)));
        assert!(engine.is_registered(&IndicatorType::MACD_LINE_STANDARD));
        assert!(engine.is_registered(&IndicatorType::MACD_SIGNAL_STANDARD));
        assert!(engine.is_registered(&IndicatorType::BOLLINGER_UPPER_STANDARD));
    }

    #[test]
    fn test_cache_get_or_compute() {
        let mut engine = IndicatorEngine::new();
        engine.register(IndicatorType::Sma(3));

        let candles = make_candles(&[10.0, 11.0, 12.0, 13.0, 14.0]);
        let mut cache = engine.create_cache();
        let symbol = "TEST".to_string();
        let tick = 5;

        // First call computes
        let v1 = cache.get_or_compute(&symbol, IndicatorType::Sma(3), tick, &candles, &engine);
        assert!(v1.is_some());

        // Second call returns cached value
        let v2 = cache.get_or_compute(&symbol, IndicatorType::Sma(3), tick, &candles, &engine);
        assert_eq!(v1, v2);

        // Cache should have one entry
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_tick_invalidation() {
        let mut engine = IndicatorEngine::new();
        engine.register(IndicatorType::Sma(3));

        let candles = make_candles(&[10.0, 11.0, 12.0]);
        let mut cache = engine.create_cache();
        let symbol = "TEST".to_string();

        // Compute at tick 1
        let v1 = cache.get_or_compute(&symbol, IndicatorType::Sma(3), 1, &candles, &engine);
        assert!(v1.is_some());

        // Try to get at tick 2 - cache miss
        let v2 = cache.get(&symbol, IndicatorType::Sma(3), 2);
        assert!(v2.is_none());
    }

    #[test]
    fn test_indicator_snapshot() {
        let mut snapshot = IndicatorSnapshot::new(100);

        let mut aapl_indicators = HashMap::new();
        aapl_indicators.insert(IndicatorType::Sma(20), 150.5);
        aapl_indicators.insert(IndicatorType::Rsi(14), 65.3);
        snapshot.insert("AAPL".to_string(), aapl_indicators);

        assert_eq!(
            snapshot.get(&"AAPL".to_string(), IndicatorType::Sma(20)),
            Some(150.5)
        );
        assert_eq!(
            snapshot.get(&"AAPL".to_string(), IndicatorType::Rsi(14)),
            Some(65.3)
        );
        assert_eq!(
            snapshot.get(&"GOOGL".to_string(), IndicatorType::Sma(20)),
            None
        );
    }
}
