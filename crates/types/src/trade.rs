//! Trade and fill types for the trading simulation.
//!
//! This module contains types for completed trades, individual fills (executions
//! at a single price level), and slippage tracking metrics.

use crate::ids::{AgentId, FillId, OrderId, Symbol, Tick, Timestamp, TradeId};
use crate::money::{Cash, Price, Quantity};
use crate::order::OrderSide;
use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Trade Type
// =============================================================================

/// A completed trade between two parties.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trade {
    /// Unique trade identifier.
    pub id: TradeId,
    /// Symbol traded.
    pub symbol: Symbol,
    /// Agent who bought.
    pub buyer_id: AgentId,
    /// Agent who sold.
    pub seller_id: AgentId,
    /// Order that was the buyer.
    pub buyer_order_id: OrderId,
    /// Order that was the seller.
    pub seller_order_id: OrderId,
    /// Execution price.
    pub price: Price,
    /// Number of shares traded.
    pub quantity: Quantity,
    /// When trade occurred (wall clock).
    pub timestamp: Timestamp,
    /// Simulation tick when trade occurred.
    pub tick: Tick,
}

impl Trade {
    /// Calculate the total value of this trade.
    pub fn value(&self) -> Cash {
        self.price * self.quantity
    }
}

impl fmt::Display for Trade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trade[{}]: {} {} shares @ {} (buyer: {}, seller: {})",
            self.id, self.symbol, self.quantity, self.price, self.buyer_id, self.seller_id
        )
    }
}

// =============================================================================
// Fill Type (V2.2)
// =============================================================================

/// A single execution at one price level.
///
/// Fills are the atomic unit of execution. A single order may generate multiple
/// fills when it crosses multiple price levels in the order book. Fills are
/// distinct from trades in that:
/// - A `Fill` represents execution at exactly one price
/// - A `Trade` may aggregate multiple fills for reporting purposes
/// - Fills track slippage relative to a reference price
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fill {
    /// Unique fill identifier.
    pub id: FillId,
    /// Symbol being traded.
    pub symbol: Symbol,
    /// Order this fill is part of.
    pub order_id: OrderId,
    /// Agent who initiated the aggressive order.
    pub aggressor_id: AgentId,
    /// Agent whose resting order was filled.
    pub resting_id: AgentId,
    /// Order that was resting in the book.
    pub resting_order_id: OrderId,
    /// Side of the aggressor (the order that crossed the spread).
    pub aggressor_side: OrderSide,
    /// Execution price for this fill.
    pub price: Price,
    /// Number of shares filled.
    pub quantity: Quantity,
    /// Reference price at time of order submission (e.g., mid price).
    /// Used to calculate slippage.
    pub reference_price: Option<Price>,
    /// When fill occurred (wall clock).
    pub timestamp: Timestamp,
    /// Simulation tick when fill occurred.
    pub tick: Tick,
}

impl Fill {
    /// Calculate the total value of this fill.
    #[inline]
    pub fn value(&self) -> Cash {
        self.price * self.quantity
    }

    /// Calculate slippage in price points.
    ///
    /// Positive slippage means the fill was worse than expected:
    /// - For buys: fill price > reference price
    /// - For sells: fill price < reference price
    pub fn slippage(&self) -> Option<Price> {
        self.reference_price
            .map(|ref_price| match self.aggressor_side {
                OrderSide::Buy => self.price - ref_price, // Positive = paid more
                OrderSide::Sell => ref_price - self.price, // Positive = received less
            })
    }

    /// Calculate slippage as basis points relative to reference price.
    ///
    /// Returns `None` if no reference price or reference price is zero.
    pub fn slippage_bps(&self) -> Option<i64> {
        self.reference_price.and_then(|ref_price| {
            if ref_price.raw() == 0 {
                return None;
            }
            let slip = self.slippage()?;
            // Basis points = (slip / ref_price) * 10000
            Some((slip.raw() * 10_000) / ref_price.raw())
        })
    }
}

impl fmt::Display for Fill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fill[{}]: {} {} {} shares @ {}",
            self.id, self.aggressor_side, self.symbol, self.quantity, self.price
        )
    }
}

// =============================================================================
// Slippage Configuration (V2.2)
// =============================================================================

/// Configuration for slippage and market impact modeling.
///
/// Controls how market orders and large orders experience price impact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlippageConfig {
    /// Whether slippage tracking is enabled.
    pub enabled: bool,
    /// Minimum order size (as fraction of available liquidity in bps) before
    /// impact modeling kicks in. Orders smaller than this are considered
    /// "small" and experience no additional impact beyond natural spread.
    /// Default: 100 bps (1% of available liquidity).
    pub impact_threshold_bps: u32,
    /// Linear impact coefficient in basis points per percent of liquidity consumed.
    /// Impact = coefficient * (order_size / available_liquidity)
    /// Default: 10 bps per 1% liquidity consumed.
    pub linear_impact_bps: u32,
    /// Whether to use square-root impact model instead of linear.
    /// sqrt model: Impact = coefficient * sqrt(order_size / available_liquidity)
    /// More realistic for larger orders.
    pub use_sqrt_model: bool,
}

impl SlippageConfig {
    /// Create a new slippage configuration with tracking enabled.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            impact_threshold_bps: 100,
            linear_impact_bps: 10,
            use_sqrt_model: false,
        }
    }

    /// Disable slippage tracking entirely.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            impact_threshold_bps: 0,
            linear_impact_bps: 0,
            use_sqrt_model: false,
        }
    }

    /// Enable square-root impact model.
    pub fn with_sqrt_model(mut self) -> Self {
        self.use_sqrt_model = true;
        self
    }

    /// Set the linear impact coefficient.
    pub fn with_linear_impact_bps(mut self, bps: u32) -> Self {
        self.linear_impact_bps = bps;
        self
    }

    /// Set the impact threshold.
    pub fn with_impact_threshold_bps(mut self, bps: u32) -> Self {
        self.impact_threshold_bps = bps;
        self
    }
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self::enabled()
    }
}

// =============================================================================
// Slippage Metrics (V2.2)
// =============================================================================

/// Metrics tracking execution quality for an order.
///
/// Aggregates fill information to compute overall slippage and market impact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SlippageMetrics {
    /// Total quantity filled.
    pub filled_quantity: Quantity,
    /// Volume-weighted average fill price (VWAP) numerator.
    /// Stored as raw i128 to avoid floating point and overflow.
    vwap_numerator: i128,
    /// Reference price when order was submitted.
    pub reference_price: Option<Price>,
    /// Number of price levels crossed.
    pub levels_crossed: u32,
    /// Best price achieved.
    pub best_fill_price: Option<Price>,
    /// Worst price achieved.
    pub worst_fill_price: Option<Price>,
}

impl SlippageMetrics {
    /// Create new metrics with a reference price.
    pub fn new(reference_price: Option<Price>) -> Self {
        Self {
            reference_price,
            ..Default::default()
        }
    }

    /// Record a fill into the metrics.
    pub fn record_fill(&mut self, price: Price, quantity: Quantity) {
        if quantity.is_zero() {
            return;
        }

        // Update VWAP calculation
        self.vwap_numerator += price.raw() as i128 * quantity.raw() as i128;
        self.filled_quantity += quantity;

        // Track best/worst prices
        match self.best_fill_price {
            None => self.best_fill_price = Some(price),
            Some(best) => {
                // For buys, best = lowest; for sells, best = highest
                // Since we don't know side here, track both extremes
                if price < best {
                    self.best_fill_price = Some(price);
                }
            }
        }
        match self.worst_fill_price {
            None => self.worst_fill_price = Some(price),
            Some(worst) => {
                if price > worst {
                    self.worst_fill_price = Some(price);
                }
            }
        }

        self.levels_crossed += 1;
    }

    /// Get the volume-weighted average price (VWAP) of all fills.
    pub fn vwap(&self) -> Option<Price> {
        if self.filled_quantity.is_zero() {
            return None;
        }
        let vwap = self.vwap_numerator / self.filled_quantity.raw() as i128;
        Some(Price(vwap as i64))
    }

    /// Calculate total slippage in price points for a buy order.
    ///
    /// Positive means paid more than reference.
    pub fn slippage_buy(&self) -> Option<Price> {
        match (self.vwap(), self.reference_price) {
            (Some(vwap), Some(ref_price)) => Some(vwap - ref_price),
            _ => None,
        }
    }

    /// Calculate total slippage in price points for a sell order.
    ///
    /// Positive means received less than reference.
    pub fn slippage_sell(&self) -> Option<Price> {
        match (self.vwap(), self.reference_price) {
            (Some(vwap), Some(ref_price)) => Some(ref_price - vwap),
            _ => None,
        }
    }

    /// Calculate slippage in basis points for a given order side.
    pub fn slippage_bps(&self, side: OrderSide) -> Option<i64> {
        let slippage = match side {
            OrderSide::Buy => self.slippage_buy()?,
            OrderSide::Sell => self.slippage_sell()?,
        };
        let ref_price = self.reference_price?;
        if ref_price.raw() == 0 {
            return None;
        }
        Some((slippage.raw() * 10_000) / ref_price.raw())
    }

    /// Get the price range of all fills.
    pub fn fill_range(&self) -> Option<Price> {
        match (self.best_fill_price, self.worst_fill_price) {
            (Some(best), Some(worst)) => Some(worst - best),
            _ => None,
        }
    }
}
