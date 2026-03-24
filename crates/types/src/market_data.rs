//! Market data types for the trading simulation.
//!
//! This module contains types for representing market data including
//! OHLCV candles and order book snapshots.

use crate::ids::{Symbol, Tick, Timestamp};
use crate::money::{Price, Quantity};
use serde::{Deserialize, Serialize};

// =============================================================================
// OHLCV Candle
// =============================================================================

/// OHLCV candle data for a single time period.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candle {
    /// Stock symbol.
    pub symbol: Symbol,
    /// Opening price.
    pub open: Price,
    /// Highest price during the period.
    pub high: Price,
    /// Lowest price during the period.
    pub low: Price,
    /// Closing price.
    pub close: Price,
    /// Trading volume during the period.
    pub volume: Quantity,
    /// Wall clock timestamp.
    pub timestamp: Timestamp,
    /// Simulation tick at period end.
    pub tick: Tick,
}

impl Candle {
    /// Create a new candle.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbol: impl Into<Symbol>,
        open: Price,
        high: Price,
        low: Price,
        close: Price,
        volume: Quantity,
        timestamp: Timestamp,
        tick: Tick,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            open,
            high,
            low,
            close,
            volume,
            timestamp,
            tick,
        }
    }

    /// Get the typical price (HLC/3).
    #[inline]
    pub fn typical_price(&self) -> Price {
        Price((self.high.0 + self.low.0 + self.close.0) / 3)
    }

    /// Get the candle range (high - low).
    #[inline]
    pub fn range(&self) -> Price {
        self.high - self.low
    }

    /// Check if this is a bullish candle (close > open).
    #[inline]
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish candle (close < open).
    #[inline]
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

// =============================================================================
// Order Book Types
// =============================================================================

/// A single price level in the order book.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookLevel {
    /// Price at this level.
    pub price: Price,
    /// Total quantity available at this price.
    pub quantity: Quantity,
    /// Number of orders at this level.
    pub order_count: usize,
}

/// Snapshot of the order book at a point in time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BookSnapshot {
    /// Symbol this book is for.
    pub symbol: Symbol,
    /// Bid levels (highest first).
    pub bids: Vec<BookLevel>,
    /// Ask levels (lowest first).
    pub asks: Vec<BookLevel>,
    /// When snapshot was taken.
    pub timestamp: Timestamp,
    /// Simulation tick.
    pub tick: Tick,
}

impl BookSnapshot {
    /// Get the best bid price.
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.first().map(|l| l.price)
    }

    /// Get the best ask price.
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.first().map(|l| l.price)
    }

    /// Calculate the spread between best bid and ask.
    pub fn spread(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate the mid price.
    pub fn mid_price(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(Price((bid.0 + ask.0) / 2)),
            _ => None,
        }
    }
}
