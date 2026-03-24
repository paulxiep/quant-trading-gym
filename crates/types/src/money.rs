//! Fixed-point monetary types for the trading simulation.
//!
//! This module provides type-safe representations for prices, cash amounts,
//! and share quantities. All monetary values use fixed-point arithmetic with
//! 4 decimal places to avoid floating-point precision issues.

use crate::ids::PRICE_SCALE;
use derive_more::{Add, AddAssign, From, Into, Neg, Sub, SubAssign, Sum};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Mul;

// =============================================================================
// Quantity Type (Newtype for shares)
// =============================================================================

/// Number of shares (newtype for type safety).
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Default,
    Add,
    Sub,
    AddAssign,
    SubAssign,
    Sum,
    From,
    Into,
)]
pub struct Quantity(pub u64);

impl Quantity {
    pub const ZERO: Quantity = Quantity(0);

    /// Get raw value.
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Check if zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Saturating subtraction.
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Quantity(self.0.saturating_sub(rhs.0))
    }

    /// Minimum of two quantities.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Quantity(self.0.min(other.0))
    }
}

impl fmt::Debug for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Qty({})", self.0)
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Allow `quantity == 50` comparisons
impl PartialEq<u64> for Quantity {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

// =============================================================================
// Fixed-Point Price Type
// =============================================================================

/// Fixed-point price with 4 decimal places.
///
/// # Examples
/// - `Price(10000)` = $1.00
/// - `Price(15000)` = $1.50
/// - `Price(100)` = $0.01
/// - `Price(1)` = $0.0001
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Default,
    Add,
    Sub,
    Neg,
    AddAssign,
    SubAssign,
    From,
    Into,
)]
pub struct Price(pub i64);

impl Price {
    pub const ZERO: Price = Price(0);

    /// Create a Price from a floating-point value.
    #[inline]
    pub fn from_float(v: f64) -> Self {
        Self((v * PRICE_SCALE as f64).round() as i64)
    }

    /// Convert to floating-point for display/calculations.
    #[inline]
    pub fn to_float(self) -> f64 {
        self.0 as f64 / PRICE_SCALE as f64
    }

    /// Raw internal value.
    #[inline]
    pub fn raw(self) -> i64 {
        self.0
    }

    /// Check if price is positive.
    #[inline]
    pub fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Price(self.0.abs())
    }
}

impl fmt::Debug for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Price(${:.4})", self.to_float())
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${:.4}", self.to_float())
    }
}

// =============================================================================
// Fixed-Point Cash Type
// =============================================================================

/// Fixed-point cash/money with 4 decimal places.
///
/// Semantically identical to Price but represents account balances.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Default,
    Add,
    Sub,
    Neg,
    AddAssign,
    SubAssign,
    From,
    Into,
)]
pub struct Cash(pub i64);

impl Cash {
    pub const ZERO: Cash = Cash(0);

    /// Create Cash from a floating-point value.
    #[inline]
    pub fn from_float(v: f64) -> Self {
        Self((v * PRICE_SCALE as f64).round() as i64)
    }

    /// Convert to floating-point for display/calculations.
    #[inline]
    pub fn to_float(self) -> f64 {
        self.0 as f64 / PRICE_SCALE as f64
    }

    /// Raw internal value.
    #[inline]
    pub fn raw(self) -> i64 {
        self.0
    }

    /// Check if cash is positive.
    #[inline]
    pub fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Check if cash is negative.
    #[inline]
    pub fn is_negative(self) -> bool {
        self.0 < 0
    }
}

impl fmt::Debug for Cash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cash(${:.4})", self.to_float())
    }
}

impl fmt::Display for Cash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${:.4}", self.to_float())
    }
}

// =============================================================================
// Price-Quantity Operations
// =============================================================================

impl Mul<Quantity> for Price {
    type Output = Cash;

    /// Multiply price by quantity to get total cash value.
    fn mul(self, qty: Quantity) -> Cash {
        Cash(self.0 * qty.0 as i64)
    }
}

impl Mul<Price> for Quantity {
    type Output = Cash;

    fn mul(self, price: Price) -> Cash {
        Cash(price.0 * self.0 as i64)
    }
}
