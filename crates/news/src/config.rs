//! Configuration types for the news generator (V2.4).
//!
//! This module provides declarative configuration for event generation.

use serde::{Deserialize, Serialize};

// =============================================================================
// EventFrequency
// =============================================================================

/// Configures how often a type of event occurs.
///
/// Frequency is expressed as probability per tick, allowing
/// declarative control over event generation rates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventFrequency {
    /// Probability of event occurring each tick (0.0 to 1.0).
    pub probability_per_tick: f64,

    /// Minimum ticks between events of this type.
    /// Prevents event spam even if probability rolls succeed.
    pub min_interval: u64,

    /// Whether this event type is enabled.
    pub enabled: bool,
}

impl EventFrequency {
    /// Create a new frequency configuration.
    pub fn new(probability_per_tick: f64, min_interval: u64) -> Self {
        Self {
            probability_per_tick: probability_per_tick.clamp(0.0, 1.0),
            min_interval,
            enabled: true,
        }
    }

    /// Create a disabled frequency (event type won't generate).
    pub fn disabled() -> Self {
        Self {
            probability_per_tick: 0.0,
            min_interval: 0,
            enabled: false,
        }
    }

    /// Set enabled state.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// =============================================================================
// MagnitudeConfig
// =============================================================================

/// Configures magnitude distribution for events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagnitudeConfig {
    /// Minimum magnitude (0.0 to 1.0).
    pub min: f64,

    /// Maximum magnitude (0.0 to 1.0).
    pub max: f64,
}

impl MagnitudeConfig {
    /// Create a new magnitude configuration.
    pub fn new(min: f64, max: f64) -> Self {
        Self {
            min: min.clamp(0.0, 1.0),
            max: max.clamp(0.0, 1.0),
        }
    }
}

impl Default for MagnitudeConfig {
    fn default() -> Self {
        Self { min: 0.2, max: 0.8 }
    }
}

// =============================================================================
// EarningsConfig
// =============================================================================

/// Configuration for earnings surprise events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EarningsConfig {
    /// How often earnings events occur.
    pub frequency: EventFrequency,

    /// Range for surprise percentage (e.g., -0.20 to +0.20).
    pub surprise_range: (f64, f64),

    /// Event duration in ticks.
    pub duration_ticks: u64,

    /// Magnitude distribution.
    pub magnitude: MagnitudeConfig,
}

impl Default for EarningsConfig {
    fn default() -> Self {
        Self {
            // Low probability: ~1 per 500 ticks per symbol
            frequency: EventFrequency::new(0.002, 100),
            surprise_range: (-0.15, 0.15),
            duration_ticks: 50,
            magnitude: MagnitudeConfig::new(0.4, 0.9),
        }
    }
}

// =============================================================================
// GuidanceConfig
// =============================================================================

/// Configuration for guidance change events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuidanceConfig {
    /// How often guidance events occur.
    pub frequency: EventFrequency,

    /// Range for new growth estimate (e.g., -0.05 to +0.15).
    pub growth_range: (f64, f64),

    /// Event duration in ticks.
    pub duration_ticks: u64,

    /// Magnitude distribution.
    pub magnitude: MagnitudeConfig,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            frequency: EventFrequency::new(0.001, 200),
            // Range capped at 7% to stay below required return (9%)
            // This prevents Gordon Growth Model breakdown
            growth_range: (-0.02, 0.07),
            duration_ticks: 30,
            magnitude: MagnitudeConfig::new(0.3, 0.7),
        }
    }
}

// =============================================================================
// RateDecisionConfig
// =============================================================================

/// Configuration for rate decision events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateDecisionConfig {
    /// How often rate decisions occur.
    pub frequency: EventFrequency,

    /// Range for rate change (basis points, e.g., -50 to +50).
    pub change_range_bps: (i32, i32),

    /// Event duration in ticks.
    pub duration_ticks: u64,

    /// Magnitude distribution.
    pub magnitude: MagnitudeConfig,
}

impl Default for RateDecisionConfig {
    fn default() -> Self {
        Self {
            // Very rare: ~1 per 2000 ticks
            frequency: EventFrequency::new(0.0005, 500),
            change_range_bps: (-25, 25),
            duration_ticks: 100,
            magnitude: MagnitudeConfig::new(0.6, 1.0),
        }
    }
}

// =============================================================================
// SectorNewsConfig
// =============================================================================

/// Configuration for sector news events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectorNewsConfig {
    /// How often sector news occurs.
    pub frequency: EventFrequency,

    /// Event duration in ticks.
    pub duration_ticks: u64,

    /// Magnitude distribution.
    pub magnitude: MagnitudeConfig,
}

impl Default for SectorNewsConfig {
    fn default() -> Self {
        Self {
            frequency: EventFrequency::new(0.003, 50),
            duration_ticks: 40,
            magnitude: MagnitudeConfig::new(0.2, 0.6),
        }
    }
}

// =============================================================================
// FairValueDriftConfig
// =============================================================================

/// Configuration for fair value drift (V2.5).
///
/// Adds a bounded random walk to fair value between news events,
/// simulating continuous uncertainty about fundamentals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FairValueDriftConfig {
    /// Enable fair value drift (default: true for sim, false for tests).
    pub enabled: bool,

    /// Max drift per tick as fraction (e.g., 0.005 = 0.5%).
    pub drift_pct: f64,

    /// Floor as fraction of initial fair value (e.g., 0.1 = 10%).
    pub min_pct: f64,

    /// Cap as multiple of initial fair value (e.g., 10.0 = 10x).
    pub max_multiple: f64,
}

impl Default for FairValueDriftConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            drift_pct: 0.01,    // 1% per tick
            min_pct: 0.1,       // Floor at 10% of initial
            max_multiple: 10.0, // Cap at 10x initial
        }
    }
}

impl FairValueDriftConfig {
    /// Create a disabled drift config (for deterministic tests).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a config with custom drift percentage.
    pub fn with_drift_pct(drift_pct: f64) -> Self {
        Self {
            enabled: true,
            drift_pct,
            ..Default::default()
        }
    }
}

// =============================================================================
// NewsGeneratorConfig
// =============================================================================

/// Top-level configuration for the news generator.
///
/// Declaratively specifies event frequencies, magnitudes, and durations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct NewsGeneratorConfig {
    /// Earnings surprise configuration.
    pub earnings: EarningsConfig,

    /// Guidance change configuration.
    pub guidance: GuidanceConfig,

    /// Rate decision configuration.
    pub rate_decision: RateDecisionConfig,

    /// Sector news configuration.
    pub sector_news: SectorNewsConfig,
}

impl NewsGeneratorConfig {
    /// Create a config with no events (all disabled).
    pub fn disabled() -> Self {
        Self {
            earnings: EarningsConfig {
                frequency: EventFrequency::disabled(),
                ..Default::default()
            },
            guidance: GuidanceConfig {
                frequency: EventFrequency::disabled(),
                ..Default::default()
            },
            rate_decision: RateDecisionConfig {
                frequency: EventFrequency::disabled(),
                ..Default::default()
            },
            sector_news: SectorNewsConfig {
                frequency: EventFrequency::disabled(),
                ..Default::default()
            },
        }
    }

    /// Create a config with high-frequency events (for testing).
    pub fn high_frequency() -> Self {
        Self {
            earnings: EarningsConfig {
                frequency: EventFrequency::new(0.05, 10),
                ..Default::default()
            },
            guidance: GuidanceConfig {
                frequency: EventFrequency::new(0.03, 20),
                ..Default::default()
            },
            rate_decision: RateDecisionConfig {
                frequency: EventFrequency::new(0.01, 100),
                ..Default::default()
            },
            sector_news: SectorNewsConfig {
                frequency: EventFrequency::new(0.08, 5),
                ..Default::default()
            },
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_frequency_clamp() {
        let freq = EventFrequency::new(1.5, 10);
        assert!((freq.probability_per_tick - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_disabled_config() {
        let config = NewsGeneratorConfig::disabled();
        assert!(!config.earnings.frequency.enabled);
        assert!(!config.guidance.frequency.enabled);
        assert!(!config.rate_decision.frequency.enabled);
        assert!(!config.sector_news.frequency.enabled);
    }
}
