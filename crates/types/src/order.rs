//! Order types for the trading simulation.
//!
//! This module defines all order-related types including order sides,
//! order types (market/limit), status tracking, and the Order struct itself.

use crate::ids::{AgentId, OrderId, Symbol, Tick, Timestamp};
use crate::money::{Price, Quantity};
use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Order Side
// =============================================================================

/// Which side of the market the order is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    /// Returns the opposite side.
    pub fn opposite(self) -> Self {
        match self {
            OrderSide::Buy => OrderSide::Sell,
            OrderSide::Sell => OrderSide::Buy,
        }
    }
}

impl fmt::Display for OrderSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

// =============================================================================
// Order Type
// =============================================================================

/// Type of order determining execution rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderType {
    /// Execute immediately at best available price.
    Market,
    /// Execute at specified price or better.
    Limit { price: Price },
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Market => write!(f, "MARKET"),
            OrderType::Limit { price } => write!(f, "LIMIT@{}", price),
        }
    }
}

// =============================================================================
// Order Status
// =============================================================================

/// Status of an order in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OrderStatus {
    /// Order created but not yet submitted.
    #[default]
    Pending,
    /// Order queued with latency, will execute at specified tick.
    Queued { execute_at: Tick },
    /// Order partially filled.
    PartialFill { filled: Quantity },
    /// Order completely filled.
    Filled,
    /// Order was cancelled.
    Cancelled,
}

// =============================================================================
// Order Struct
// =============================================================================

/// A trading order submitted by an agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Order {
    /// Unique order identifier (assigned by Market, use 0 as placeholder).
    pub id: OrderId,
    /// Agent who submitted the order.
    pub agent_id: AgentId,
    /// Symbol being traded.
    pub symbol: Symbol,
    /// Buy or Sell.
    pub side: OrderSide,
    /// Market or Limit order.
    pub order_type: OrderType,
    /// Number of shares.
    pub quantity: Quantity,
    /// Remaining quantity (for partial fills).
    pub remaining_quantity: Quantity,
    /// When order was created (wall clock).
    pub timestamp: Timestamp,
    /// Latency in ticks before order can be matched (0 = instant).
    pub latency_ticks: u64,
    /// Current status.
    pub status: OrderStatus,
}

impl Order {
    /// Create a new limit order.
    pub fn limit(
        agent_id: AgentId,
        symbol: impl Into<Symbol>,
        side: OrderSide,
        price: Price,
        quantity: Quantity,
    ) -> Self {
        Self {
            id: OrderId(0), // Placeholder, assigned by Market
            agent_id,
            symbol: symbol.into(),
            side,
            order_type: OrderType::Limit { price },
            quantity,
            remaining_quantity: quantity,
            timestamp: 0,
            latency_ticks: 0,
            status: OrderStatus::Pending,
        }
    }

    /// Create a new market order.
    pub fn market(
        agent_id: AgentId,
        symbol: impl Into<Symbol>,
        side: OrderSide,
        quantity: Quantity,
    ) -> Self {
        Self {
            id: OrderId(0),
            agent_id,
            symbol: symbol.into(),
            side,
            order_type: OrderType::Market,
            quantity,
            remaining_quantity: quantity,
            timestamp: 0,
            latency_ticks: 0,
            status: OrderStatus::Pending,
        }
    }

    /// Get the limit price if this is a limit order.
    pub fn limit_price(&self) -> Option<Price> {
        match self.order_type {
            OrderType::Limit { price } => Some(price),
            OrderType::Market => None,
        }
    }

    /// Check if order is fully filled.
    pub fn is_filled(&self) -> bool {
        self.remaining_quantity.is_zero()
    }

    /// Check if order is a buy order.
    pub fn is_buy(&self) -> bool {
        self.side == OrderSide::Buy
    }

    /// Check if order is a sell order.
    pub fn is_sell(&self) -> bool {
        self.side == OrderSide::Sell
    }
}
