//! Configuration for the Tier 3 Background Agent Pool (V3.4).
//!
//! The pool simulates 90k+ agents statistically without individual instances.
//! Behavior is controlled declaratively via [`BackgroundPoolConfig`] with
//! [`MarketRegime`] presets for common scenarios.
//!
//! # Design Principles
//!
//! - **Declarative**: All behavior controlled by config, not code
//! - **Modular**: Regime presets encapsulate parameter tuning
//! - **SoC**: Config is pure data; pool.rs handles behavior

use serde::{Deserialize, Serialize};
use types::Symbol;

// =============================================================================
// MarketRegime
// =============================================================================

/// Market regime presets that control pool behavior.
///
/// Each regime defines activity levels, sentiment volatility, and contrarian
/// behavior that together produce realistic order flow patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Low activity, stable sentiment. Typical quiet trading day.
    Calm,
    /// Moderate activity (default). Normal market conditions.
    #[default]
    Normal,
    /// High activity, wider price swings. Earnings season, macro uncertainty.
    Volatile,
    /// Very high activity, extreme sentiment. Market crisis, flash crash.
    Crisis,
}

impl MarketRegime {
    /// Get preset values for this regime.
    pub fn preset(&self) -> RegimePreset {
        match self {
            MarketRegime::Calm => RegimePreset {
                base_activity: 0.1,
                sentiment_volatility: 0.05,
                contrarian_fraction: 0.3,
            },
            MarketRegime::Normal => RegimePreset {
                base_activity: 0.3,
                sentiment_volatility: 0.15,
                contrarian_fraction: 0.25,
            },
            MarketRegime::Volatile => RegimePreset {
                base_activity: 0.6,
                sentiment_volatility: 0.3,
                contrarian_fraction: 0.2,
            },
            MarketRegime::Crisis => RegimePreset {
                base_activity: 0.9,
                sentiment_volatility: 0.5,
                contrarian_fraction: 0.15,
            },
        }
    }
}

// =============================================================================
// RegimePreset
// =============================================================================

/// Preset values derived from a market regime.
#[derive(Debug, Clone, Copy)]
pub struct RegimePreset {
    /// Orders per tick as fraction of pool size (0.0-1.0).
    /// At pool_size=90k and base_activity=0.3, generates ~27k orders/tick.
    pub base_activity: f64,

    /// How much news events swing sentiment (multiplier on event magnitude).
    pub sentiment_volatility: f64,

    /// Fraction of orders that go against current sentiment.
    /// Provides natural mean reversion and prevents runaway trends.
    pub contrarian_fraction: f64,
}

// =============================================================================
// BackgroundPoolConfig
// =============================================================================

/// Configuration for the Tier 3 background agent pool.
///
/// A single pool instance trades all configured symbols, selecting randomly
/// per-order based on activity and sentiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundPoolConfig {
    /// Target number of simulated background agents (for order rate scaling).
    /// Memory cost is O(1) regardless of this value.
    pub pool_size: usize,

    /// Market regime preset (provides default parameter values).
    pub regime: MarketRegime,

    /// Symbols the pool trades. Pool randomly selects symbol per order.
    pub symbols: Vec<Symbol>,

    // ─── Order Size Distribution ───────────────────────────────────────────
    /// Mean order size (log-normal distribution).
    pub mean_order_size: f64,

    /// Order size standard deviation (log-normal).
    pub order_size_stddev: f64,

    /// Maximum single order size (hard cap).
    pub max_order_size: u64,

    /// Minimum order size (floor).
    pub min_order_size: u64,

    // ─── Price Distribution ────────────────────────────────────────────────
    /// Price spread lambda (exponential decay parameter).
    /// Higher = orders cluster tighter around mid price.
    /// λ=20 gives most orders within 5% of mid.
    pub price_spread_lambda: f64,

    /// Maximum price deviation from mid (as fraction, e.g., 0.05 = 5%).
    pub max_price_deviation: f64,

    // ─── Sentiment Parameters ──────────────────────────────────────────────
    /// Sentiment decay per tick (0.995 = 0.5% decay toward neutral).
    pub sentiment_decay: f64,

    /// Maximum absolute sentiment (clamped to prevent runaway).
    pub max_sentiment: f64,

    /// Sentiment impact multiplier from news events.
    pub news_sentiment_scale: f64,

    // ─── Sanity Check Parameters ───────────────────────────────────────────
    /// Enable P&L sanity checking (warns if pool loses unrealistically).
    pub enable_sanity_check: bool,

    /// Maximum allowed loss as fraction of notional volume.
    /// Exceeding this triggers a warning (misconfigured params).
    pub max_pnl_loss_fraction: f64,

    /// Override base activity rate (None = use regime default).
    /// Fraction of pool that trades each tick (0.0-1.0).
    pub base_activity_override: Option<f64>,
}

impl Default for BackgroundPoolConfig {
    fn default() -> Self {
        Self {
            pool_size: 90_000,
            regime: MarketRegime::Normal,
            symbols: vec!["ACME".to_string()],

            // Log-normal size: many small, few large
            mean_order_size: 15.0,
            order_size_stddev: 10.0,
            max_order_size: 100,
            min_order_size: 1,

            // Exponential price spread: tight around mid
            price_spread_lambda: 20.0,
            max_price_deviation: 0.02, // 2% max from mid

            // Sentiment mechanics
            sentiment_decay: 0.995,
            max_sentiment: 0.8,
            news_sentiment_scale: 0.5,

            // Sanity checking
            enable_sanity_check: true,
            max_pnl_loss_fraction: 0.05, // 5% of volume

            // Activity override
            base_activity_override: None, // Use regime default
        }
    }
}

impl BackgroundPoolConfig {
    /// Create a new config with specified symbols.
    pub fn new(symbols: Vec<Symbol>) -> Self {
        Self {
            symbols,
            ..Default::default()
        }
    }

    /// Set the pool size.
    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    /// Set the market regime.
    pub fn with_regime(mut self, regime: MarketRegime) -> Self {
        self.regime = regime;
        self
    }

    /// Disable sanity checking (for testing extreme scenarios).
    pub fn without_sanity_check(mut self) -> Self {
        self.enable_sanity_check = false;
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_presets() {
        let calm = MarketRegime::Calm.preset();
        let normal = MarketRegime::Normal.preset();
        let volatile = MarketRegime::Volatile.preset();
        let crisis = MarketRegime::Crisis.preset();

        // Activity increases with regime severity
        assert!(calm.base_activity < normal.base_activity);
        assert!(normal.base_activity < volatile.base_activity);
        assert!(volatile.base_activity < crisis.base_activity);

        // Contrarian fraction decreases (less mean reversion in crisis)
        assert!(calm.contrarian_fraction > crisis.contrarian_fraction);
    }

    #[test]
    fn test_default_config() {
        let config = BackgroundPoolConfig::default();
        assert_eq!(config.pool_size, 90_000);
        assert_eq!(config.regime, MarketRegime::Normal);
        assert!(config.enable_sanity_check);
    }

    #[test]
    fn test_config_builder() {
        let config = BackgroundPoolConfig::new(vec!["AAPL".to_string(), "GOOG".to_string()])
            .with_pool_size(50_000)
            .with_regime(MarketRegime::Volatile)
            .without_sanity_check();

        assert_eq!(config.pool_size, 50_000);
        assert_eq!(config.regime, MarketRegime::Volatile);
        assert_eq!(config.symbols.len(), 2);
        assert!(!config.enable_sanity_check);
    }
}
