//! Reactive strategy types for Tier 2 agents.
//!
//! This module defines the strategy variants available for Tier 2 reactive agents.
//! Unlike Tier 1 strategies (which are separate structs), Tier 2 strategies are
//! enum variants for compact storage and efficient dispatch.
//!
//! # Strategy Categories
//!
//! ## Entry Strategy (Absolute Price)
//! - [`ReactiveStrategyType::ThresholdBuyer`] - Buy when price drops to absolute level
//!
//! ## Exit Strategies
//! - [`ReactiveStrategyType::StopLoss`] - Exit when price drops X% below cost basis
//! - [`ReactiveStrategyType::TakeProfit`] - Exit when price rises X% above cost basis
//! - [`ReactiveStrategyType::ThresholdSeller`] - Exit when price rises to absolute level
//!
//! ## Optional Modifier
//! - [`ReactiveStrategyType::NewsReactor`] - React to news events (entry or exit)
//!
//! # Typical Agent Composition
//!
//! Each agent has 1 entry + 1-2 exits + optionally NewsReactor:
//! - `ThresholdBuyer($95) + StopLoss(5%) + TakeProfit(10%)`
//! - `ThresholdBuyer($95) + StopLoss(5%) + ThresholdSeller($110)`
//! - Any of the above + `NewsReactor` (20% probability)
//!
//! # WakeCondition Compatibility
//!
//! All active strategies work with [`WakeConditionIndex`] for O(log n) lookups:
//! - ThresholdBuyer: `PriceCross(Below)` registered at spawn
//! - ThresholdSeller: `PriceCross(Above)` registered at spawn
//! - StopLoss/TakeProfit: `PriceCross` registered on fill (computed from cost_basis)
//! - NewsReactor: `NewsEvent` registered at spawn
//!
//! # Deferred Strategies
//!
//! The following strategies are commented out because they need rolling price
//! references (e.g., "3% below recent high") which cannot be efficiently
//! represented as static wake conditions:
//! - DipBuyer, BreakoutEntry (need rolling high/low)
//! - TrailingStop (needs high-water-mark updates)
//! - TimedEntry/TimedExit (mechanical, not truly reactive)
//! - FundamentalExit (needs fair_value lookup each tick)

// PriceReference is unused now that DipBuyer/BreakoutEntry are deferred
#[allow(unused_imports)]
use crate::tiers::PriceReference;
use serde::{Deserialize, Serialize};
use types::Price;

/// Strategy variants for Tier 2 reactive agents.
///
/// Each variant is a self-contained strategy configuration that determines
/// wake conditions and trade behavior. Agents typically hold 2-3 strategies
/// (one entry + one or more exits).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReactiveStrategyType {
    // =========================================================================
    // Entry Strategy (Absolute Price Threshold)
    // =========================================================================
    /// Buy when price drops to or below an absolute threshold.
    ///
    /// Registers `WakeCondition::PriceCross` with `CrossDirection::Below` at spawn.
    /// Good for: Value buying, limit orders, accumulating at support levels.
    ThresholdBuyer {
        /// Absolute price at which to buy.
        buy_price: Price,
        /// Order size as fraction of max position (0.0 to 1.0).
        size_fraction: f64,
    },

    // =========================================================================
    // Exit Strategies
    // =========================================================================
    /// Exit when price drops below stop loss threshold (percentage).
    ///
    /// Registers `WakeCondition::PriceCross` with `CrossDirection::Below` on fill.
    /// Threshold computed as: cost_basis * (1 - stop_pct)
    StopLoss {
        /// How far below cost basis to trigger (e.g., 0.05 = 5% loss).
        stop_pct: f64,
    },

    /// Exit when price rises above profit target (percentage).
    ///
    /// Registers `WakeCondition::PriceCross` with `CrossDirection::Above` on fill.
    /// Threshold computed as: cost_basis * (1 + target_pct)
    TakeProfit {
        /// How far above cost basis to trigger (e.g., 0.10 = 10% profit).
        target_pct: f64,
    },

    /// Exit when price rises to an absolute threshold.
    ///
    /// Registers `WakeCondition::PriceCross` with `CrossDirection::Above` at spawn.
    /// Good for: Selling at resistance, profit target at specific price level.
    ThresholdSeller {
        /// Absolute price at which to sell.
        sell_price: Price,
        /// Order size as fraction of position to sell (0.0 to 1.0).
        size_fraction: f64,
    },

    // =========================================================================
    // Optional Modifier (can be added to any agent)
    // =========================================================================
    /// React to news events with directional trades.
    ///
    /// Registers `WakeCondition::NewsEvent` for subscribed symbols at spawn.
    /// Trade direction and size depend on news sentiment and magnitude.
    /// Can trigger entry (positive sentiment) or exit (negative sentiment).
    NewsReactor {
        /// Minimum sentiment magnitude to trigger (0.0 to 1.0).
        /// Higher values mean only react to significant news.
        min_magnitude: f64,
        /// Position size multiplier for sentiment (base * sentiment * multiplier).
        sentiment_multiplier: f64,
    },
    // =========================================================================
    // DEFERRED: Strategies requiring rolling references or periodic polling
    // =========================================================================
    // These are commented out because they cannot efficiently use WakeConditionIndex.
    // They would need either:
    // - Rolling price tracking (high/low over N ticks)
    // - Periodic polling via TimeInterval (defeats the purpose of reactive agents)
    // - Condition updates on every price change (expensive)
    //
    // /// Buy when price drops by threshold percentage below reference.
    // DipBuyer { threshold_pct: f64, reference: PriceReference, size_fraction: f64 },
    //
    // /// Buy when price breaks above resistance level.
    // BreakoutEntry { threshold_pct: f64, reference: PriceReference, size_fraction: f64 },
    //
    // /// Enter position at scheduled intervals (mechanical, not reactive).
    // TimedEntry { interval_ticks: u64, size_fraction: f64 },
    //
    // /// Exit when price drops below trailing high water mark.
    // TrailingStop { trail_pct: f64 },
    //
    // /// Exit after holding for specified duration (mechanical).
    // TimedExit { hold_ticks: u64 },
    //
    // /// Exit when fundamental value diverges significantly from price.
    // FundamentalExit { overvalued_pct: f64, undervalued_pct: f64 },
}

impl ReactiveStrategyType {
    /// Returns true if this is an entry strategy (can open positions).
    pub fn is_entry(&self) -> bool {
        matches!(self, Self::ThresholdBuyer { .. } | Self::NewsReactor { .. })
    }

    /// Returns true if this is an exit strategy (can close positions).
    pub fn is_exit(&self) -> bool {
        matches!(
            self,
            Self::StopLoss { .. }
                | Self::TakeProfit { .. }
                | Self::ThresholdSeller { .. }
                | Self::NewsReactor { .. }
        )
    }

    /// Returns true if this strategy needs high water mark tracking.
    pub fn needs_high_water_mark(&self) -> bool {
        // TrailingStop is deferred
        false
    }

    /// Returns true if this strategy is time-based.
    pub fn is_time_based(&self) -> bool {
        // TimedEntry/TimedExit are deferred
        false
    }

    /// Returns the PriceReference this strategy uses, if any.
    #[allow(dead_code)]
    pub fn price_reference(&self) -> Option<&PriceReference> {
        // DipBuyer/BreakoutEntry with references are deferred
        None
    }

    /// Compute the absolute trigger price for exit strategies given cost basis.
    ///
    /// Returns `None` for ThresholdSeller (already absolute), ThresholdBuyer (entry),
    /// or NewsReactor (event-driven).
    pub fn compute_exit_trigger(&self, cost_basis: Price) -> Option<Price> {
        let basis = cost_basis.0 as f64;
        match self {
            Self::StopLoss { stop_pct } => {
                let trigger = basis * (1.0 - stop_pct);
                Some(Price(trigger as i64))
            }
            Self::TakeProfit { target_pct } => {
                let trigger = basis * (1.0 + target_pct);
                Some(Price(trigger as i64))
            }
            // ThresholdSeller already has absolute price, doesn't need computation
            // ThresholdBuyer is entry, NewsReactor is event-driven
            Self::ThresholdBuyer { .. }
            | Self::ThresholdSeller { .. }
            | Self::NewsReactor { .. } => None,
        }
    }

    /// Get the absolute entry price for ThresholdBuyer.
    pub fn entry_price(&self) -> Option<Price> {
        match self {
            Self::ThresholdBuyer { buy_price, .. } => Some(*buy_price),
            _ => None,
        }
    }

    /// Get the absolute exit price for ThresholdSeller.
    pub fn exit_price(&self) -> Option<Price> {
        match self {
            Self::ThresholdSeller { sell_price, .. } => Some(*sell_price),
            _ => None,
        }
    }

    /// Get the size fraction for this strategy.
    pub fn size_fraction(&self) -> Option<f64> {
        match self {
            Self::ThresholdBuyer { size_fraction, .. }
            | Self::ThresholdSeller { size_fraction, .. } => Some(*size_fraction),
            _ => None,
        }
    }
}

/// Validation result for strategy combinations.
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyValidation {
    /// Strategy set is valid.
    Valid,
    /// No entry strategy - agent can never open positions.
    MissingEntry,
    /// No exit strategy - agent can never close positions.
    MissingExit,
}

/// Validate that a set of strategies can both enter and exit positions.
///
/// Returns `StrategyValidation::Valid` if the set has at least one entry
/// and at least one exit capability.
pub fn validate_strategies(strategies: &[ReactiveStrategyType]) -> StrategyValidation {
    let has_entry = strategies.iter().any(|s| s.is_entry());
    let has_exit = strategies.iter().any(|s| s.is_exit());

    match (has_entry, has_exit) {
        (true, true) => StrategyValidation::Valid,
        (false, _) => StrategyValidation::MissingEntry,
        (_, false) => StrategyValidation::MissingExit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_classification() {
        let buyer = ReactiveStrategyType::ThresholdBuyer {
            buy_price: Price(950000),
            size_fraction: 0.5,
        };
        assert!(buyer.is_entry());
        assert!(!buyer.is_exit());

        let stop = ReactiveStrategyType::StopLoss { stop_pct: 0.05 };
        assert!(!stop.is_entry());
        assert!(stop.is_exit());

        let seller = ReactiveStrategyType::ThresholdSeller {
            sell_price: Price(1100000),
            size_fraction: 0.5,
        };
        assert!(!seller.is_entry());
        assert!(seller.is_exit());

        let news = ReactiveStrategyType::NewsReactor {
            min_magnitude: 0.3,
            sentiment_multiplier: 1.0,
        };
        assert!(news.is_entry());
        assert!(news.is_exit());
    }

    #[test]
    fn test_compute_exit_trigger() {
        let stop = ReactiveStrategyType::StopLoss { stop_pct: 0.10 };
        let trigger = stop.compute_exit_trigger(Price(1000000)).unwrap();
        assert_eq!(trigger, Price(900000)); // 10% below 1000000

        let profit = ReactiveStrategyType::TakeProfit { target_pct: 0.05 };
        let trigger = profit.compute_exit_trigger(Price(1000000)).unwrap();
        assert_eq!(trigger, Price(1050000)); // 5% above 1000000

        // ThresholdSeller already has absolute price
        let seller = ReactiveStrategyType::ThresholdSeller {
            sell_price: Price(1100000),
            size_fraction: 0.5,
        };
        assert!(seller.compute_exit_trigger(Price(1000000)).is_none());
    }

    #[test]
    fn test_validate_strategies() {
        // Valid: entry + exit
        let valid = vec![
            ReactiveStrategyType::ThresholdBuyer {
                buy_price: Price(950000),
                size_fraction: 0.5,
            },
            ReactiveStrategyType::StopLoss { stop_pct: 0.03 },
        ];
        assert_eq!(validate_strategies(&valid), StrategyValidation::Valid);

        // Missing entry
        let no_entry = vec![ReactiveStrategyType::StopLoss { stop_pct: 0.03 }];
        assert_eq!(
            validate_strategies(&no_entry),
            StrategyValidation::MissingEntry
        );

        // Missing exit
        let no_exit = vec![ReactiveStrategyType::ThresholdBuyer {
            buy_price: Price(950000),
            size_fraction: 0.5,
        }];
        assert_eq!(
            validate_strategies(&no_exit),
            StrategyValidation::MissingExit
        );

        // NewsReactor alone is valid (bidirectional)
        let news_only = vec![ReactiveStrategyType::NewsReactor {
            min_magnitude: 0.3,
            sentiment_multiplier: 1.0,
        }];
        assert_eq!(validate_strategies(&news_only), StrategyValidation::Valid);
    }
}
