//! Rolling price history for computing returns and price changes.
//!
//! V5.3: Used by RecordingHook for feature extraction.

use std::collections::{HashMap, VecDeque};
use types::{Price, Symbol, Tick};

/// Maximum lookback period for price history.
const MAX_LOOKBACK: usize = 100;

/// Rolling price history for computing returns and price changes.
///
/// Maintains a sliding window of (tick, price) pairs per symbol,
/// enabling efficient computation of price changes and log returns
/// over various horizons.
#[derive(Debug, Default)]
pub struct PriceHistory {
    /// Symbol -> VecDeque of (tick, mid_price).
    history: HashMap<Symbol, VecDeque<(Tick, Price)>>,
}

impl PriceHistory {
    /// Create a new empty price history.
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
        }
    }

    /// Record a price observation.
    ///
    /// Maintains at most MAX_LOOKBACK entries per symbol.
    pub fn record(&mut self, tick: Tick, symbol: &Symbol, price: Price) {
        let deque = self.history.entry(symbol.clone()).or_default();

        // Only add if this is a new tick (avoid duplicates)
        if deque.back().is_none_or(|(t, _)| *t < tick) {
            deque.push_back((tick, price));

            // Trim to MAX_LOOKBACK
            while deque.len() > MAX_LOOKBACK {
                deque.pop_front();
            }
        }
    }

    /// Get the price from N ticks ago.
    ///
    /// Returns None if insufficient history.
    pub fn price_at(&self, symbol: &Symbol, ticks_ago: usize) -> Option<Price> {
        let deque = self.history.get(symbol)?;
        if ticks_ago >= deque.len() {
            return None;
        }
        let idx = deque.len() - 1 - ticks_ago;
        Some(deque[idx].1)
    }

    /// Get the current (most recent) price.
    pub fn current_price(&self, symbol: &Symbol) -> Option<Price> {
        self.price_at(symbol, 0)
    }

    /// Compute price change as percentage over N ticks.
    ///
    /// Returns `(current - past) / past * 100` as percentage.
    /// Returns None if insufficient history.
    pub fn price_change(&self, symbol: &Symbol, ticks_ago: usize) -> Option<f64> {
        let current = self.current_price(symbol)?;
        let past = self.price_at(symbol, ticks_ago)?;

        if past.0 == 0 {
            return None;
        }

        let change = (current.0 as f64 - past.0 as f64) / past.0 as f64 * 100.0;
        Some(change)
    }

    /// Compute log return over N ticks.
    ///
    /// Returns `ln(current / past)`.
    /// Returns None if insufficient history or invalid prices.
    pub fn log_return(&self, symbol: &Symbol, ticks_ago: usize) -> Option<f64> {
        let current = self.current_price(symbol)?;
        let past = self.price_at(symbol, ticks_ago)?;

        if past.0 == 0 || current.0 == 0 {
            return None;
        }

        let ratio = current.0 as f64 / past.0 as f64;
        Some(ratio.ln())
    }

    /// Get the number of price observations for a symbol.
    pub fn len(&self, symbol: &Symbol) -> usize {
        self.history.get(symbol).map_or(0, |d| d.len())
    }

    /// Check if there are any price observations for a symbol.
    pub fn is_empty(&self, symbol: &Symbol) -> bool {
        self.len(symbol) == 0
    }

    /// Clear all price history.
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn price(cents: i64) -> Price {
        Price(cents)
    }

    fn sym(s: &str) -> Symbol {
        Symbol::from(s)
    }

    #[test]
    fn test_record_and_retrieve() {
        let mut ph = PriceHistory::new();
        let symbol = sym("AAPL");

        ph.record(0, &symbol, price(10000));
        ph.record(1, &symbol, price(10100));
        ph.record(2, &symbol, price(10200));

        assert_eq!(ph.current_price(&symbol), Some(price(10200)));
        assert_eq!(ph.price_at(&symbol, 0), Some(price(10200)));
        assert_eq!(ph.price_at(&symbol, 1), Some(price(10100)));
        assert_eq!(ph.price_at(&symbol, 2), Some(price(10000)));
        assert_eq!(ph.price_at(&symbol, 3), None);
    }

    #[test]
    fn test_price_change() {
        let mut ph = PriceHistory::new();
        let symbol = sym("AAPL");

        ph.record(0, &symbol, price(10000)); // $100.00
        ph.record(1, &symbol, price(10100)); // $101.00

        // 1% increase
        let change = ph.price_change(&symbol, 1).unwrap();
        assert!((change - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_log_return() {
        let mut ph = PriceHistory::new();
        let symbol = sym("AAPL");

        ph.record(0, &symbol, price(10000));
        ph.record(1, &symbol, price(10100));

        let log_ret = ph.log_return(&symbol, 1).unwrap();
        // ln(101/100) ≈ 0.00995
        assert!((log_ret - 0.00995).abs() < 0.001);
    }

    #[test]
    fn test_max_lookback_trimming() {
        let mut ph = PriceHistory::new();
        let symbol = sym("AAPL");

        // Record more than MAX_LOOKBACK entries
        for i in 0..150 {
            ph.record(i as Tick, &symbol, price(10000 + i));
        }

        // Should be trimmed to MAX_LOOKBACK
        assert_eq!(ph.len(&symbol), MAX_LOOKBACK);

        // Oldest should be tick 50 (150 - 100)
        assert_eq!(ph.price_at(&symbol, MAX_LOOKBACK - 1), Some(price(10050)));
    }

    #[test]
    fn test_multiple_symbols() {
        let mut ph = PriceHistory::new();
        let aapl = sym("AAPL");
        let goog = sym("GOOG");

        ph.record(0, &aapl, price(10000));
        ph.record(0, &goog, price(200000));

        assert_eq!(ph.current_price(&aapl), Some(price(10000)));
        assert_eq!(ph.current_price(&goog), Some(price(200000)));
    }

    #[test]
    fn test_no_duplicate_ticks() {
        let mut ph = PriceHistory::new();
        let symbol = sym("AAPL");

        ph.record(0, &symbol, price(10000));
        ph.record(0, &symbol, price(10100)); // Same tick, should be ignored

        assert_eq!(ph.len(&symbol), 1);
        assert_eq!(ph.current_price(&symbol), Some(price(10000)));
    }
}
