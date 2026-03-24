//! Lightweight context for Tier 2 reactive agents.
//!
//! Unlike Tier 1's `StrategyContext` which provides full candle history,
//! indicators, and trade lists, `LightweightContext` provides only the
//! minimal data needed for reactive decision-making.
//!
//! # Design Rationale
//!
//! Tier 2 agents wake on specific conditions and need only:
//! - Current price (to evaluate threshold crossings)
//! - Why they woke (which condition triggered)
//! - Current tick (for time-based decisions)
//! - Fundamental value (for fundamental strategies)
//!
//! This keeps the context small and avoids expensive data gathering
//! for agents that may decide to take no action.

use crate::tiers::WakeCondition;
use news::NewsEvent;
use serde::{Deserialize, Serialize};
use types::{Price, Symbol, Tick};

/// Minimal context passed to Tier 2 agents on wake.
///
/// Contains only the data needed for reactive decision-making.
/// Full market context is not available - use Tier 1 for complex strategies.
#[derive(Debug, Clone)]
pub struct LightweightContext<'a> {
    /// Current simulation tick.
    pub tick: Tick,

    /// The condition(s) that triggered this wake.
    pub wake_reasons: &'a [WakeCondition],

    /// Current prices by symbol.
    ///
    /// Only symbols relevant to the agent are included.
    pub prices: &'a [(Symbol, PriceSnapshot)],

    /// News event if this wake was triggered by news.
    pub news_event: Option<&'a NewsEvent>,

    /// Fundamental values by symbol (if available).
    pub fundamentals: &'a [(Symbol, Price)],
}

impl<'a> LightweightContext<'a> {
    /// Get the current price for a symbol.
    pub fn price(&self, symbol: &str) -> Option<Price> {
        self.prices
            .iter()
            .find(|(s, _)| s == symbol)
            .map(|(_, snap)| snap.last)
    }

    /// Get the price snapshot for a symbol.
    pub fn price_snapshot(&self, symbol: &str) -> Option<&PriceSnapshot> {
        self.prices
            .iter()
            .find(|(s, _)| s == symbol)
            .map(|(_, snap)| snap)
    }

    /// Get the fundamental value for a symbol.
    pub fn fundamental(&self, symbol: &str) -> Option<Price> {
        self.fundamentals
            .iter()
            .find(|(s, _)| s == symbol)
            .map(|(_, p)| *p)
    }

    /// Check if this wake was triggered by a specific condition type.
    pub fn woke_on_price_cross(&self) -> bool {
        self.wake_reasons
            .iter()
            .any(|c| matches!(c, WakeCondition::PriceCross { .. }))
    }

    /// Check if this wake was triggered by a time condition.
    pub fn woke_on_time(&self) -> bool {
        self.wake_reasons.iter().any(|c| {
            matches!(
                c,
                WakeCondition::TimeExact { .. } | WakeCondition::TimeInterval { .. }
            )
        })
    }

    /// Check if this wake was triggered by a news event.
    pub fn woke_on_news(&self) -> bool {
        self.wake_reasons
            .iter()
            .any(|c| matches!(c, WakeCondition::NewsEvent { .. }))
    }
}

/// Snapshot of price data for a symbol.
///
/// Provides just enough price information for reactive decisions
/// without full candle history.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PriceSnapshot {
    /// Last traded price.
    pub last: Price,

    /// Best bid price (highest buy order).
    pub bid: Price,

    /// Best ask price (lowest sell order).
    pub ask: Price,

    /// Session open price.
    pub open: Price,

    /// Session high price.
    pub high: Price,

    /// Session low price.
    pub low: Price,
}

impl PriceSnapshot {
    /// Create a new price snapshot.
    pub fn new(last: Price, bid: Price, ask: Price, open: Price, high: Price, low: Price) -> Self {
        Self {
            last,
            bid,
            ask,
            open,
            high,
            low,
        }
    }

    /// Get the mid price (average of bid and ask).
    pub fn mid(&self) -> Price {
        Price((self.bid.0 + self.ask.0) / 2)
    }

    /// Get the spread (ask - bid).
    pub fn spread(&self) -> Price {
        Price(self.ask.0 - self.bid.0)
    }

    /// Get the session range (high - low).
    pub fn range(&self) -> Price {
        Price(self.high.0 - self.low.0)
    }

    /// Calculate return from open as a fraction.
    pub fn return_from_open(&self) -> f64 {
        if self.open.0 == 0 {
            0.0
        } else {
            (self.last.0 - self.open.0) as f64 / self.open.0 as f64
        }
    }
}

impl Default for PriceSnapshot {
    fn default() -> Self {
        Self {
            last: Price(0),
            bid: Price(0),
            ask: Price(0),
            open: Price(0),
            high: Price(0),
            low: Price(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_snapshot_calculations() {
        let snap = PriceSnapshot::new(
            Price(10050), // last
            Price(10000), // bid
            Price(10100), // ask
            Price(10000), // open
            Price(10200), // high
            Price(9800),  // low
        );

        assert_eq!(snap.mid(), Price(10050));
        assert_eq!(snap.spread(), Price(100));
        assert_eq!(snap.range(), Price(400));
        assert!((snap.return_from_open() - 0.005).abs() < 0.0001);
    }

    #[test]
    fn test_context_price_lookup() {
        let prices = vec![
            ("ACME".to_string(), PriceSnapshot::default()),
            (
                "BETA".to_string(),
                PriceSnapshot {
                    last: Price(5000),
                    ..Default::default()
                },
            ),
        ];
        let fundamentals = vec![("ACME".to_string(), Price(10000))];
        let wake_reasons = vec![];

        let ctx = LightweightContext {
            tick: 100,
            wake_reasons: &wake_reasons,
            prices: &prices,
            news_event: None,
            fundamentals: &fundamentals,
        };

        assert_eq!(ctx.price("ACME"), Some(Price(0)));
        assert_eq!(ctx.price("BETA"), Some(Price(5000)));
        assert_eq!(ctx.price("UNKNOWN"), None);
        assert_eq!(ctx.fundamental("ACME"), Some(Price(10000)));
    }
}
