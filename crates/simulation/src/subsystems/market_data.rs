//! Market data manager subsystem.
//!
//! Manages historical market data including candles, recent trades, and
//! technical indicators.

use std::collections::{HashMap, VecDeque};

use quant::{IndicatorCache, IndicatorEngine, IndicatorSnapshot};
use sim_core::Market;
use types::{Candle, Price, Quantity, Symbol, Tick, Timestamp, Trade};

use crate::traits::MarketDataProvider;

/// Helper for building candles incrementally.
#[derive(Debug, Clone)]
pub(crate) struct CandleBuilder {
    pub symbol: String,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Quantity,
    #[allow(dead_code)]
    pub start_tick: Tick,
    pub trade_count: usize,
}

impl CandleBuilder {
    pub fn new(symbol: &str, price: Price, tick: Tick) -> Self {
        Self {
            symbol: symbol.to_string(),
            open: price,
            high: price,
            low: price,
            close: price,
            volume: Quantity::ZERO,
            start_tick: tick,
            trade_count: 0,
        }
    }

    pub fn update(&mut self, trade: &Trade) {
        self.high = self.high.max(trade.price);
        self.low = self.low.min(trade.price);
        self.close = trade.price;
        self.volume += trade.quantity;
        self.trade_count += 1;
    }

    pub fn finalize(self, end_tick: Tick, timestamp: Timestamp) -> Candle {
        Candle {
            symbol: self.symbol,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            timestamp,
            tick: end_tick,
        }
    }
}

/// Manages historical market data.
///
/// Owns candles, recent trades, and indicators. Provides the `MarketDataProvider`
/// trait implementation.
pub struct MarketDataManager {
    /// Historical candles per symbol.
    candles: HashMap<Symbol, VecDeque<Candle>>,

    /// Current candle being built per symbol.
    current_candles: HashMap<Symbol, CandleBuilder>,

    /// Recent trades per symbol.
    recent_trades: HashMap<Symbol, Vec<Trade>>,

    /// Indicator engine for computing technical indicators.
    indicator_engine: IndicatorEngine,

    /// Indicator cache (reserved for future per-tick caching).
    #[allow(dead_code)]
    indicator_cache: IndicatorCache,

    /// Candle interval in ticks.
    candle_interval: u64,

    /// Maximum candles to retain per symbol.
    max_candles: usize,

    /// Maximum recent trades to retain per symbol.
    max_recent_trades: usize,
}

impl MarketDataManager {
    /// Create a new market data manager.
    pub fn new(
        symbols: &[Symbol],
        candle_interval: u64,
        max_candles: usize,
        max_recent_trades: usize,
    ) -> Self {
        let mut candles = HashMap::new();
        let mut recent_trades = HashMap::new();

        for symbol in symbols {
            candles.insert(symbol.clone(), VecDeque::new());
            recent_trades.insert(symbol.clone(), Vec::new());
        }

        Self {
            candles,
            current_candles: HashMap::new(),
            recent_trades,
            indicator_engine: IndicatorEngine::with_common_indicators(),
            indicator_cache: IndicatorCache::new(),
            candle_interval,
            max_candles,
            max_recent_trades,
        }
    }

    /// Get mutable access to the indicator engine for registration.
    pub fn indicator_engine_mut(&mut self) -> &mut IndicatorEngine {
        &mut self.indicator_engine
    }

    /// Get reference to the indicator engine.
    pub fn indicator_engine(&self) -> &IndicatorEngine {
        &self.indicator_engine
    }

    /// Update candles with trade data.
    pub fn update_candles(
        &mut self,
        trades: &[Trade],
        tick: Tick,
        timestamp: Timestamp,
        market: &Market,
    ) {
        for trade in trades {
            let symbol = &trade.symbol;

            // Initialize candle builder if needed
            if !self.current_candles.contains_key(symbol) {
                self.current_candles.insert(
                    symbol.clone(),
                    CandleBuilder::new(symbol, trade.price, tick),
                );
            }

            // Update current candle for this symbol
            if let Some(builder) = self.current_candles.get_mut(symbol) {
                builder.update(trade);
            }
        }

        // Check if we should finalize candles (every candle_interval ticks)
        if tick > 0 && tick.is_multiple_of(self.candle_interval) {
            // Ensure ALL symbols have candles, even if no trades occurred.
            let all_symbols: Vec<Symbol> = market.symbols().cloned().collect();
            for symbol in &all_symbols {
                if !self.current_candles.contains_key(symbol) {
                    // Get last price from market book or last candle
                    let price = market
                        .get_book(symbol)
                        .and_then(|b| b.last_price())
                        .or_else(|| {
                            self.candles
                                .get(symbol)
                                .and_then(|c| c.back())
                                .map(|c| c.close)
                        })
                        .unwrap_or_else(|| Price::from_float(100.0));

                    self.current_candles
                        .insert(symbol.clone(), CandleBuilder::new(symbol, price, tick));
                }
            }

            // Finalize all current candles
            let builders: Vec<(Symbol, CandleBuilder)> = self.current_candles.drain().collect();
            for (symbol, builder) in builders {
                let candle = builder.finalize(tick, timestamp);
                let symbol_candles = self.candles.entry(symbol).or_default();
                symbol_candles.push_back(candle);

                // Limit candle history per symbol (O(1) with VecDeque)
                if symbol_candles.len() > self.max_candles {
                    symbol_candles.pop_front();
                }
            }
        }
    }

    /// Update recent trades storage.
    pub fn update_recent_trades(&mut self, tick_trades: &[Trade]) {
        // Add trades (newest first)
        tick_trades.iter().rev().for_each(|trade| {
            let symbol_trades = self.recent_trades.entry(trade.symbol.clone()).or_default();
            symbol_trades.insert(0, trade.clone());
        });

        // Trim to max
        self.recent_trades.values_mut().for_each(|symbol_trades| {
            if symbol_trades.len() > self.max_recent_trades {
                symbol_trades.truncate(self.max_recent_trades);
            }
        });
    }

    /// Build candles map for StrategyContext (converts VecDeque to Vec).
    pub fn build_candles_map(&self) -> HashMap<Symbol, Vec<Candle>> {
        self.candles
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect()
    }

    /// Build trades map for StrategyContext.
    pub fn build_trades_map(&self) -> HashMap<Symbol, Vec<Trade>> {
        self.recent_trades.clone()
    }
}

impl MarketDataProvider for MarketDataManager {
    fn candles_for(&self, symbol: &Symbol) -> &[Candle] {
        self.candles
            .get(symbol)
            .map(|v| v.as_slices().0) // Return first contiguous slice
            .unwrap_or(&[])
    }

    fn candles_for_mut(&mut self, symbol: &Symbol) -> &[Candle] {
        self.candles
            .get_mut(symbol)
            .map(|v| v.make_contiguous())
            .map(|s| &*s)
            .unwrap_or(&[])
    }

    fn all_candles(&self) -> &HashMap<Symbol, VecDeque<Candle>> {
        &self.candles
    }

    fn recent_trades_for(&self, symbol: &Symbol) -> &[Trade] {
        self.recent_trades
            .get(symbol)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    fn all_recent_trades(&self) -> &HashMap<Symbol, Vec<Trade>> {
        &self.recent_trades
    }

    fn build_indicator_snapshot(&mut self) -> IndicatorSnapshot {
        use quant::indicators::compute_all_indicators;

        // Make all candles contiguous for indicator computation
        for deque in self.candles.values_mut() {
            deque.make_contiguous();
        }

        let indicators = self
            .candles
            .iter()
            .filter(|(_, symbol_candles)| !symbol_candles.is_empty())
            .filter_map(|(symbol, symbol_candles)| {
                let (slice, _) = symbol_candles.as_slices();
                // V5.5: Single source of truth - compute_all_indicators returns
                // all component-level values (MacdLine, MacdSignal, etc.)
                let values = compute_all_indicators(slice);
                (!values.is_empty()).then(|| (symbol.clone(), values))
            })
            .collect();

        IndicatorSnapshot::from_map(0, indicators) // tick is set by caller
    }
}
