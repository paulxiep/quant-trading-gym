//! Core identifier types for the trading simulation.
//!
//! This module defines all the fundamental ID types used throughout the system
//! to uniquely identify orders, agents, trades, and fills.

use derive_more::{Add, From, Into};
use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Constants
// =============================================================================

/// Price scale factor: 10,000 means 4 decimal places.
/// - `10000` = $1.00
/// - `1` = $0.0001 (smallest price increment)
pub const PRICE_SCALE: i64 = 10_000;

// =============================================================================
// Core ID Types
// =============================================================================

/// Unique identifier for an order.
#[derive(
    Debug,
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
    From,
    Into,
)]
pub struct OrderId(pub u64);

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Order#{}", self.0)
    }
}

/// Unique identifier for a trading agent.
#[derive(
    Debug,
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
    From,
    Into,
)]
pub struct AgentId(pub u64);

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Agent#{}", self.0)
    }
}

/// Unique identifier for a completed trade.
#[derive(
    Debug,
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
    From,
    Into,
)]
pub struct TradeId(pub u64);

impl fmt::Display for TradeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trade#{}", self.0)
    }
}

/// Unique identifier for a fill (individual execution at one price level).
#[derive(
    Debug,
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
    From,
    Into,
)]
pub struct FillId(pub u64);

impl fmt::Display for FillId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fill#{}", self.0)
    }
}

// =============================================================================
// Symbol Type
// =============================================================================

/// Stock/asset symbol (e.g., "AAPL", "SIM").
pub type Symbol = String;

// =============================================================================
// Time Types
// =============================================================================

/// Wall clock timestamp in milliseconds since epoch.
pub type Timestamp = u64;

/// Simulation tick (discrete time step).
pub type Tick = u64;
