//! Error types for sim-core operations.

use std::fmt;
use types::{OrderId, Symbol};

/// Result type for sim-core operations.
pub type Result<T> = std::result::Result<T, SimCoreError>;

/// Errors that can occur during market operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimCoreError {
    /// The requested symbol was not found.
    UnknownSymbol(Symbol),
    /// The requested order was not found.
    OrderNotFound(OrderId),
    /// Invalid order: zero quantity.
    ZeroQuantity,
    /// Invalid order: non-positive price for limit order.
    InvalidPrice,
    /// Order book is empty on the required side.
    EmptyBook,
    /// Order has already been filled or cancelled.
    OrderNotActive(OrderId),
}

impl fmt::Display for SimCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimCoreError::UnknownSymbol(s) => write!(f, "unknown symbol: {}", s),
            SimCoreError::OrderNotFound(id) => write!(f, "order not found: {}", id),
            SimCoreError::ZeroQuantity => write!(f, "order quantity cannot be zero"),
            SimCoreError::InvalidPrice => write!(f, "limit order price must be positive"),
            SimCoreError::EmptyBook => write!(f, "order book is empty"),
            SimCoreError::OrderNotActive(id) => {
                write!(f, "order {} is not active (filled or cancelled)", id)
            }
        }
    }
}

impl std::error::Error for SimCoreError {}
