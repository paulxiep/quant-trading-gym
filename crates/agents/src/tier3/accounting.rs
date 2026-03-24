//! Aggregate P&L tracking for background pool sanity checks (V3.4).
//!
//! The pool doesn't track individual agents, but we need to ensure
//! order generation parameters don't create unrealistic P&L (e.g.,
//! always buying high, selling low due to misconfigured sentiment).
//!
//! # Design Principles
//!
//! - **Append-only**: Fills recorded sequentially, never modified
//! - **SoC**: Pure accounting, no order generation logic

use std::collections::HashMap;
use types::{Cash, Price, Quantity, Symbol};

// =============================================================================
// BackgroundPoolAccounting
// =============================================================================

/// Aggregate accounting for background pool fills.
///
/// Tracks total buy/sell volume and value per symbol to compute
/// realized P&L and VWAP for sanity checking.
#[derive(Debug, Clone, Default)]
pub struct BackgroundPoolAccounting {
    /// Per-symbol tracking
    per_symbol: HashMap<Symbol, SymbolAccounting>,
    /// Total orders generated (for diagnostics)
    total_orders_generated: u64,
}

/// Per-symbol accounting data.
#[derive(Debug, Clone, Default)]
struct SymbolAccounting {
    /// Total shares bought
    buy_volume: u64,
    /// Total value of buys (in raw price units × quantity)
    buy_value: i64,
    /// Total shares sold
    sell_volume: u64,
    /// Total value of sells
    sell_value: i64,
}

impl BackgroundPoolAccounting {
    /// Create a new accounting instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record orders generated this tick (for diagnostics).
    pub fn record_orders_generated(&mut self, count: usize) {
        self.total_orders_generated += count as u64;
    }

    /// Get total orders generated.
    pub fn total_orders_generated(&self) -> u64 {
        self.total_orders_generated
    }

    /// Record a trade where pool is the BUYER.
    ///
    /// Called by simulation when a pool order is filled.
    pub fn record_trade_as_buyer(&mut self, symbol: &Symbol, price: Price, quantity: Quantity) {
        let acc = self.per_symbol.entry(symbol.clone()).or_default();
        acc.buy_volume += quantity.raw();
        acc.buy_value += price.0 * quantity.raw() as i64;
    }

    /// Record a trade where pool is the SELLER.
    pub fn record_trade_as_seller(&mut self, symbol: &Symbol, price: Price, quantity: Quantity) {
        let acc = self.per_symbol.entry(symbol.clone()).or_default();
        acc.sell_volume += quantity.raw();
        acc.sell_value += price.0 * quantity.raw() as i64;
    }

    /// Compute aggregate realized P&L across all symbols.
    ///
    /// P&L = (sell_value - buy_value) for matched volume.
    /// Positive = pool sold higher than bought (profit).
    pub fn realized_pnl(&self) -> Cash {
        let total: i64 = self
            .per_symbol
            .values()
            .map(|acc| acc.sell_value - acc.buy_value)
            .sum();
        Cash(total)
    }

    /// Get total notional volume traded (for sanity check denominator).
    pub fn total_notional(&self) -> i64 {
        self.per_symbol
            .values()
            .map(|acc| acc.buy_value.abs() + acc.sell_value.abs())
            .sum()
    }

    /// Get total buy volume across all symbols.
    pub fn total_buy_volume(&self) -> u64 {
        self.per_symbol.values().map(|acc| acc.buy_volume).sum()
    }

    /// Get total sell volume across all symbols.
    pub fn total_sell_volume(&self) -> u64 {
        self.per_symbol.values().map(|acc| acc.sell_volume).sum()
    }

    /// Check if P&L is within acceptable bounds.
    ///
    /// Returns `(passed, pnl, threshold)`:
    /// - `passed`: true if loss is within acceptable range
    /// - `pnl`: actual realized P&L
    /// - `threshold`: maximum acceptable loss
    pub fn sanity_check(&self, max_loss_fraction: f64) -> SanityCheckResult {
        let pnl = self.realized_pnl();
        let notional = self.total_notional();
        let threshold = Cash((notional as f64 * max_loss_fraction) as i64);

        // Fail if loss exceeds threshold (pnl is negative for loss)
        let passed = pnl.0 >= -threshold.0;

        SanityCheckResult {
            passed,
            pnl,
            threshold,
            buy_volume: self.total_buy_volume(),
            sell_volume: self.total_sell_volume(),
        }
    }

    /// Get per-symbol statistics for debugging.
    ///
    /// Returns `(buy_volume, vwap_buy, sell_volume, vwap_sell)` if symbol exists.
    pub fn symbol_stats(&self, symbol: &Symbol) -> Option<SymbolStats> {
        self.per_symbol.get(symbol).map(|acc| {
            let vwap_buy = if acc.buy_volume > 0 {
                Price(acc.buy_value / acc.buy_volume as i64)
            } else {
                Price(0)
            };
            let vwap_sell = if acc.sell_volume > 0 {
                Price(acc.sell_value / acc.sell_volume as i64)
            } else {
                Price(0)
            };
            SymbolStats {
                buy_volume: Quantity(acc.buy_volume),
                vwap_buy,
                sell_volume: Quantity(acc.sell_volume),
                vwap_sell,
            }
        })
    }

    /// Get all tracked symbols.
    pub fn symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.per_symbol.keys()
    }
}

// =============================================================================
// SanityCheckResult
// =============================================================================

/// Result of a sanity check.
#[derive(Debug, Clone)]
pub struct SanityCheckResult {
    /// Whether the check passed
    pub passed: bool,
    /// Actual realized P&L
    pub pnl: Cash,
    /// Maximum acceptable loss threshold
    pub threshold: Cash,
    /// Total buy volume
    pub buy_volume: u64,
    /// Total sell volume
    pub sell_volume: u64,
}

// =============================================================================
// SymbolStats
// =============================================================================

/// Per-symbol statistics.
#[derive(Debug, Clone)]
pub struct SymbolStats {
    /// Total shares bought
    pub buy_volume: Quantity,
    /// Volume-weighted average buy price
    pub vwap_buy: Price,
    /// Total shares sold
    pub sell_volume: Quantity,
    /// Volume-weighted average sell price
    pub vwap_sell: Price,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accounting_basic() {
        let mut acc = BackgroundPoolAccounting::new();

        // Buy 100 shares at $10
        acc.record_trade_as_buyer(&"TEST".to_string(), Price::from_float(10.0), Quantity(100));

        // Sell 100 shares at $11 (profit)
        acc.record_trade_as_seller(&"TEST".to_string(), Price::from_float(11.0), Quantity(100));

        let pnl = acc.realized_pnl();
        // P&L = (11 - 10) * 100 = $100 profit in raw units
        assert!(pnl.0 > 0, "Should have profit: {:?}", pnl);
    }

    #[test]
    fn test_sanity_check_pass() {
        let mut acc = BackgroundPoolAccounting::new();

        // Balanced trading with small profit
        acc.record_trade_as_buyer(
            &"TEST".to_string(),
            Price::from_float(100.0),
            Quantity(1000),
        );
        acc.record_trade_as_seller(
            &"TEST".to_string(),
            Price::from_float(101.0),
            Quantity(1000),
        );

        let result = acc.sanity_check(0.05);
        assert!(result.passed, "Profitable trading should pass");
    }

    #[test]
    fn test_sanity_check_fail() {
        let mut acc = BackgroundPoolAccounting::new();

        // Consistently buy high, sell low (misconfigured)
        acc.record_trade_as_buyer(
            &"TEST".to_string(),
            Price::from_float(110.0),
            Quantity(1000),
        );
        acc.record_trade_as_seller(&"TEST".to_string(), Price::from_float(90.0), Quantity(1000));

        let result = acc.sanity_check(0.05); // 5% threshold
        // Loss is 20% of notional, should fail
        assert!(!result.passed, "Large loss should fail sanity check");
    }

    #[test]
    fn test_multi_symbol() {
        let mut acc = BackgroundPoolAccounting::new();

        acc.record_trade_as_buyer(&"AAPL".to_string(), Price::from_float(150.0), Quantity(100));
        acc.record_trade_as_seller(&"AAPL".to_string(), Price::from_float(155.0), Quantity(100));

        acc.record_trade_as_buyer(&"GOOG".to_string(), Price::from_float(100.0), Quantity(50));
        acc.record_trade_as_seller(&"GOOG".to_string(), Price::from_float(98.0), Quantity(50));

        // AAPL profit offsets GOOG loss
        let pnl = acc.realized_pnl();
        assert!(pnl.0 > 0, "Net should be profitable");

        // Check per-symbol stats
        let aapl_stats = acc.symbol_stats(&"AAPL".to_string()).unwrap();
        assert_eq!(aapl_stats.buy_volume.raw(), 100);
        assert_eq!(aapl_stats.sell_volume.raw(), 100);
    }

    #[test]
    fn test_orders_generated_tracking() {
        let mut acc = BackgroundPoolAccounting::new();
        assert_eq!(acc.total_orders_generated(), 0);

        acc.record_orders_generated(100);
        acc.record_orders_generated(50);

        assert_eq!(acc.total_orders_generated(), 150);
    }
}
