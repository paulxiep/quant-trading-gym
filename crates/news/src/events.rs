//! Event types for the news system (V2.4).
//!
//! This module defines:
//! - [`FundamentalEvent`]: The underlying cause of a market-moving event
//! - [`NewsEvent`]: A time-bounded event with sentiment and magnitude
//!
//! # Event Lifecycle
//!
//! Events have a start tick and duration. During their active window:
//! - Sentiment affects agent behavior
//! - Magnitude determines impact strength
//! - Decay factor reduces impact over time

use serde::{Deserialize, Serialize};
use types::{Sector, Symbol, Tick};

// =============================================================================
// FundamentalEvent
// =============================================================================

/// The underlying cause of a market-moving event.
///
/// These events can permanently modify fundamentals or create temporary sentiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FundamentalEvent {
    /// Earnings beat or miss expectations.
    /// `surprise_pct`: Percentage deviation (e.g., 0.10 = 10% beat, -0.05 = 5% miss)
    EarningsSurprise { symbol: Symbol, surprise_pct: f64 },

    /// Company revises growth guidance.
    /// `new_growth`: New growth estimate (e.g., 0.08 = 8%)
    GuidanceChange { symbol: Symbol, new_growth: f64 },

    /// Central bank rate decision.
    /// `new_rate`: New risk-free rate (e.g., 0.05 = 5%)
    RateDecision { new_rate: f64 },

    /// Sector-wide news affecting sentiment.
    /// `sentiment`: Impact direction and strength (-1.0 to +1.0)
    SectorNews { sector: Sector, sentiment: f64 },
}

impl FundamentalEvent {
    /// Get the primary symbol affected, if any.
    pub fn symbol(&self) -> Option<&Symbol> {
        match self {
            FundamentalEvent::EarningsSurprise { symbol, .. } => Some(symbol),
            FundamentalEvent::GuidanceChange { symbol, .. } => Some(symbol),
            FundamentalEvent::RateDecision { .. } => None,
            FundamentalEvent::SectorNews { .. } => None,
        }
    }

    /// Get the sector affected, if any.
    pub fn sector(&self) -> Option<Sector> {
        match self {
            FundamentalEvent::SectorNews { sector, .. } => Some(*sector),
            _ => None,
        }
    }

    /// Whether this event permanently modifies fundamentals.
    pub fn is_permanent(&self) -> bool {
        matches!(
            self,
            FundamentalEvent::EarningsSurprise { .. }
                | FundamentalEvent::GuidanceChange { .. }
                | FundamentalEvent::RateDecision { .. }
        )
    }
}

// =============================================================================
// NewsEvent
// =============================================================================

/// A time-bounded market event with sentiment and decay.
///
/// Events are active from `start_tick` for `duration_ticks` ticks.
/// During this window, agents can react to the event's sentiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewsEvent {
    /// Unique event identifier.
    pub id: u64,

    /// The underlying fundamental event.
    pub event: FundamentalEvent,

    /// Overall sentiment direction (-1.0 = very negative, +1.0 = very positive).
    pub sentiment: f64,

    /// Impact magnitude (0.0 to 1.0, how significant the event is).
    pub magnitude: f64,

    /// Tick when the event starts.
    pub start_tick: Tick,

    /// How long the event remains active.
    pub duration_ticks: u64,
}

impl NewsEvent {
    /// Create a new news event.
    pub fn new(
        id: u64,
        event: FundamentalEvent,
        sentiment: f64,
        magnitude: f64,
        start_tick: Tick,
        duration_ticks: u64,
    ) -> Self {
        Self {
            id,
            event,
            sentiment: sentiment.clamp(-1.0, 1.0),
            magnitude: magnitude.clamp(0.0, 1.0),
            start_tick,
            duration_ticks,
        }
    }

    /// Check if the event is active at the given tick.
    pub fn is_active(&self, tick: Tick) -> bool {
        tick >= self.start_tick && tick < self.start_tick + self.duration_ticks
    }

    /// Get the decay factor at the given tick (1.0 at start, 0.0 at end).
    ///
    /// Uses linear decay. Returns 0.0 if event is not active.
    pub fn decay_factor(&self, tick: Tick) -> f64 {
        if !self.is_active(tick) {
            return 0.0;
        }
        let elapsed = (tick - self.start_tick) as f64;
        let total = self.duration_ticks as f64;
        1.0 - (elapsed / total)
    }

    /// Get the effective sentiment at the given tick (sentiment × decay).
    pub fn effective_sentiment(&self, tick: Tick) -> f64 {
        self.sentiment * self.decay_factor(tick)
    }

    /// Get the effective magnitude at the given tick (magnitude × decay).
    pub fn effective_magnitude(&self, tick: Tick) -> f64 {
        self.magnitude * self.decay_factor(tick)
    }

    /// Get the primary symbol affected, if any.
    pub fn symbol(&self) -> Option<&Symbol> {
        self.event.symbol()
    }

    /// Get the sector affected, if any.
    pub fn sector(&self) -> Option<Sector> {
        self.event.sector()
    }

    /// Whether this event permanently modifies fundamentals.
    pub fn is_permanent(&self) -> bool {
        self.event.is_permanent()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_is_active() {
        let event = NewsEvent::new(
            1,
            FundamentalEvent::EarningsSurprise {
                symbol: "AAPL".to_string(),
                surprise_pct: 0.10,
            },
            0.8,
            0.5,
            100,
            50,
        );

        assert!(!event.is_active(99));
        assert!(event.is_active(100));
        assert!(event.is_active(125));
        assert!(event.is_active(149));
        assert!(!event.is_active(150));
    }

    #[test]
    fn test_decay_factor() {
        let event = NewsEvent::new(
            1,
            FundamentalEvent::RateDecision { new_rate: 0.05 },
            -0.5,
            0.8,
            100,
            100,
        );

        // At start: decay = 1.0
        assert!((event.decay_factor(100) - 1.0).abs() < 1e-10);

        // At midpoint: decay = 0.5
        assert!((event.decay_factor(150) - 0.5).abs() < 1e-10);

        // Near end: decay ≈ 0.01
        assert!((event.decay_factor(199) - 0.01).abs() < 1e-10);

        // After end: decay = 0.0
        assert!((event.decay_factor(200) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_effective_sentiment() {
        let event = NewsEvent::new(
            1,
            FundamentalEvent::SectorNews {
                sector: Sector::Tech,
                sentiment: 0.6,
            },
            0.8, // sentiment
            0.5, // magnitude
            100,
            100,
        );

        // At start: effective = 0.8 × 1.0 = 0.8
        assert!((event.effective_sentiment(100) - 0.8).abs() < 1e-10);

        // At midpoint: effective = 0.8 × 0.5 = 0.4
        assert!((event.effective_sentiment(150) - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_fundamental_event_properties() {
        let earnings = FundamentalEvent::EarningsSurprise {
            symbol: "AAPL".to_string(),
            surprise_pct: 0.10,
        };
        assert_eq!(earnings.symbol(), Some(&"AAPL".to_string()));
        assert!(earnings.is_permanent());

        let sector = FundamentalEvent::SectorNews {
            sector: Sector::Tech,
            sentiment: 0.5,
        };
        assert_eq!(sector.sector(), Some(Sector::Tech));
        assert!(!sector.is_permanent()); // Sector news is temporary sentiment
    }
}
