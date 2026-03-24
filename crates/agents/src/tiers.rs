//! Shared types for the tiered agent architecture (V3.2).
//!
//! This module defines types shared between Tier 1 (Smart) and Tier 2 (Reactive) agents:
//! - [`TickFrequency`]: When agents should run
//! - [`WakeCondition`]: What triggers reactive agents
//! - [`CrossDirection`]: Direction for price threshold crossings
//! - [`PriceReference`]: Reference points for percentage-based calculations
//! - [`ConditionUpdate`]: Deferred wake condition mutations
//!
//! # Design Principles
//!
//! These types follow the "Declarative, Modular, SoC" mantra:
//! - **Declarative**: Agents declare wake conditions, not poll for them
//! - **Modular**: Types are independent of agent implementation details
//! - **SoC**: Condition tracking separated from agent logic

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use types::{AgentId, Price, Symbol, Tick};

// =============================================================================
// TickFrequency
// =============================================================================

/// Determines when a Tier 1 agent should execute its `on_tick()` method.
///
/// Most Tier 1 agents run every tick, but some may use reduced frequencies
/// for performance optimization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum TickFrequency {
    /// Run every tick (default for Tier 1).
    #[default]
    EveryTick,

    /// Run every N ticks (e.g., `EveryN(10)` runs on ticks 0, 10, 20, ...).
    EveryN(u64),

    /// Run with probability p per tick (0.0 to 1.0).
    /// Useful for simulating heterogeneous agent activity.
    Probabilistic(f64),
}

impl TickFrequency {
    /// Returns true if agent should run this tick.
    ///
    /// Uses provided RNG for probabilistic frequency to ensure determinism.
    pub fn should_run(&self, tick: Tick, rng: &mut impl rand::Rng) -> bool {
        match self {
            Self::EveryTick => true,
            Self::EveryN(n) => tick.is_multiple_of(*n),
            Self::Probabilistic(p) => rng.r#gen::<f64>() < *p,
        }
    }
}

// =============================================================================
// CrossDirection
// =============================================================================

/// Direction for price threshold crossings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrossDirection {
    /// Price crosses above threshold (was below, now above).
    Above,
    /// Price crosses below threshold (was above, now below).
    Below,
}

// =============================================================================
// WakeCondition
// =============================================================================

/// Conditions that trigger Tier 2 reactive agents to wake.
///
/// Agents register wake conditions with the `WakeConditionIndex`.
/// When conditions are met, the agent's `on_wake()` method is called.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WakeCondition {
    /// Price crosses a threshold in the specified direction.
    ///
    /// Triggered when price moves from one side of threshold to the other.
    PriceCross {
        symbol: Symbol,
        threshold: Price,
        direction: CrossDirection,
    },

    /// Wake at a specific tick (one-time).
    TimeExact { wake_tick: Tick },

    /// Wake periodically every N ticks.
    TimeInterval {
        /// Next tick to wake at (updated after each wake).
        next_wake: Tick,
        /// Interval between wakes.
        interval: u64,
    },

    /// Wake on news events matching specified types.
    ///
    /// Uses SmallVec to avoid heap allocation for typical 1-4 event subscriptions.
    NewsEvent {
        /// Event types to subscribe to (by symbol or sector).
        symbols: SmallVec<[Symbol; 4]>,
    },
}

impl WakeCondition {
    /// Create a price cross condition for buying dips.
    pub fn price_below(symbol: Symbol, threshold: Price) -> Self {
        Self::PriceCross {
            symbol,
            threshold,
            direction: CrossDirection::Below,
        }
    }

    /// Create a price cross condition for selling rallies.
    pub fn price_above(symbol: Symbol, threshold: Price) -> Self {
        Self::PriceCross {
            symbol,
            threshold,
            direction: CrossDirection::Above,
        }
    }

    /// Create a one-time wake at specific tick.
    pub fn at_tick(tick: Tick) -> Self {
        Self::TimeExact { wake_tick: tick }
    }

    /// Create a periodic wake condition.
    pub fn every_n_ticks(interval: u64, starting_at: Tick) -> Self {
        Self::TimeInterval {
            next_wake: starting_at,
            interval,
        }
    }

    /// Create a news subscription for specific symbols.
    pub fn on_news(symbols: impl IntoIterator<Item = Symbol>) -> Self {
        Self::NewsEvent {
            symbols: symbols.into_iter().collect(),
        }
    }
}

// =============================================================================
// PriceReference
// =============================================================================

/// Reference point for percentage-based price calculations.
///
/// Tier 2 agents use relative thresholds (e.g., "5% below reference")
/// rather than absolute prices for robustness across price levels.
///
/// # Tier 2 Constraint
///
/// Tier 2 agents cannot compute rolling indicators. All references must be
/// either agent-tracked or pre-computed externally.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PriceReference {
    /// Agent's cost basis (weighted average of entry prices).
    /// Updated automatically on fills.
    CostBasis,

    /// Session open price (stored once at session start).
    OpenPrice,

    /// Highest price since position entry (agent-tracked).
    /// Used for trailing stops.
    HighWaterMark,

    /// Lowest price since position entry (agent-tracked).
    LowWaterMark,

    /// Fundamental value from news crate (pre-computed externally).
    /// Updated when fundamentals change.
    FundamentalValue,

    /// Fixed price snapshot at strategy registration.
    /// Does not update - useful for initial reference points.
    Snapshot(Price),
}

// =============================================================================
// ConditionUpdate
// =============================================================================

/// Deferred wake condition mutation.
///
/// Collected during tick processing, applied after all agents have run.
/// This avoids borrow-checker issues from mutating the `WakeConditionIndex`
/// while iterating over triggered agents.
///
/// # Borrow-Checker Safety
///
/// The two-phase approach prevents this pattern:
/// ```ignore
/// // BAD: Mutating index while iterating
/// for agent_id in index.triggered_agents() {  // borrows index
///     let agent = agents.get_mut(agent_id);
///     agent.on_wake();
///     index.update_condition(agent_id, new_cond);  // ERROR: already borrowed
/// }
///
/// // GOOD: Collect updates, apply after
/// let mut updates = Vec::new();
/// for agent_id in index.triggered_agents() {
///     let agent = agents.get_mut(agent_id);
///     if let Some(update) = agent.on_wake() {
///         updates.push(update);
///     }
/// }
/// index.apply_updates(updates);  // Safe: iteration complete
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionUpdate {
    /// Agent requesting the update.
    pub agent_id: AgentId,

    /// Conditions to remove from index.
    pub remove: Vec<WakeCondition>,

    /// Conditions to add to index.
    pub add: Vec<WakeCondition>,
}

impl ConditionUpdate {
    /// Create a new condition update.
    pub fn new(agent_id: AgentId) -> Self {
        Self {
            agent_id,
            remove: Vec::new(),
            add: Vec::new(),
        }
    }

    /// Add a condition to remove.
    pub fn remove_condition(mut self, condition: WakeCondition) -> Self {
        self.remove.push(condition);
        self
    }

    /// Add a condition to add.
    pub fn add_condition(mut self, condition: WakeCondition) -> Self {
        self.add.push(condition);
        self
    }

    /// Check if this update has any changes.
    pub fn is_empty(&self) -> bool {
        self.remove.is_empty() && self.add.is_empty()
    }
}

// =============================================================================
// OrderedPrice
// =============================================================================

/// Wrapper for `Price` that implements `Ord` for use in `BTreeMap` keys.
///
/// The underlying `Price` type may not implement `Ord` directly (to prevent
/// accidental comparisons of prices from different symbols or contexts).
/// This wrapper explicitly opts into ordering for index lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderedPrice(pub i64);

impl OrderedPrice {
    /// Create from a Price.
    pub fn from_price(price: Price) -> Self {
        Self(price.0)
    }

    /// Convert back to Price.
    pub fn to_price(self) -> Price {
        Price(self.0)
    }
}

impl PartialOrd for OrderedPrice {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedPrice {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl From<Price> for OrderedPrice {
    fn from(price: Price) -> Self {
        Self(price.0)
    }
}

impl From<OrderedPrice> for Price {
    fn from(ordered: OrderedPrice) -> Self {
        Price(ordered.0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_frequency_every_tick() {
        let freq = TickFrequency::EveryTick;
        let mut rng = rand::thread_rng();
        for i in 0u64..100 {
            assert!(freq.should_run(i, &mut rng));
        }
    }

    #[test]
    fn test_tick_frequency_every_n() {
        let freq = TickFrequency::EveryN(10);
        let mut rng = rand::thread_rng();
        assert!(freq.should_run(0, &mut rng));
        assert!(!freq.should_run(1, &mut rng));
        assert!(freq.should_run(10, &mut rng));
        assert!(!freq.should_run(15, &mut rng));
        assert!(freq.should_run(20, &mut rng));
    }

    #[test]
    fn test_wake_condition_constructors() {
        let symbol = "ACME".to_string();

        let below = WakeCondition::price_below(symbol.clone(), Price(1000));
        assert!(matches!(
            below,
            WakeCondition::PriceCross {
                direction: CrossDirection::Below,
                ..
            }
        ));

        let above = WakeCondition::price_above(symbol.clone(), Price(2000));
        assert!(matches!(
            above,
            WakeCondition::PriceCross {
                direction: CrossDirection::Above,
                ..
            }
        ));

        let at_tick = WakeCondition::at_tick(100);
        assert!(matches!(at_tick, WakeCondition::TimeExact { wake_tick } if wake_tick == 100));
    }

    #[test]
    fn test_ordered_price() {
        let p1 = OrderedPrice(100);
        let p2 = OrderedPrice(200);
        let p3 = OrderedPrice(100);

        assert!(p1 < p2);
        assert!(p2 > p1);
        assert_eq!(p1, p3);

        // Round-trip through Price
        let price = Price(1500);
        let ordered = OrderedPrice::from_price(price);
        assert_eq!(ordered.to_price(), price);
    }

    #[test]
    fn test_condition_update_builder() {
        let update = ConditionUpdate::new(AgentId(42))
            .remove_condition(WakeCondition::at_tick(100))
            .add_condition(WakeCondition::at_tick(200));

        assert_eq!(update.agent_id, AgentId(42));
        assert_eq!(update.remove.len(), 1);
        assert_eq!(update.add.len(), 1);
        assert!(!update.is_empty());
    }
}
