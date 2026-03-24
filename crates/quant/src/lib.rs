//! Quantitative analysis crate for the Quant Trading Gym.
//!
//! This crate provides technical indicators, risk metrics, and statistical
//! utilities for market analysis and strategy development.
//!
//! # Modules
//!
//! - [`indicators`] - Technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR)
//! - [`engine`] - Indicator engine and caching
//! - [`rolling`] - Rolling window data structures
//! - [`risk`] - Risk metrics (VaR, Sharpe, drawdown)
//! - [`stats`] - Statistical utilities
//! - [`tracker`] - Per-agent risk tracking
//!
//! # Example
//!
//! ```
//! use quant::{IndicatorEngine, indicators::Sma};
//! use types::IndicatorType;
//!
//! // Create engine with common indicators
//! let mut engine = IndicatorEngine::with_common_indicators();
//!
//! // Or register specific indicators
//! engine.register(IndicatorType::Sma(50));
//! engine.register(IndicatorType::Rsi(14));
//!
//! // Create cache for efficient per-tick access
//! let mut cache = engine.create_cache();
//! ```
//!
//! # Design Notes
//!
//! - All indicator calculations use `f64` for statistical precision
//! - Monetary values (`Price`, `Cash`) are converted from/to `f64` as needed
//! - Indicators are designed to be thread-safe (`Send + Sync`)
//! - The caching system ensures indicators are computed at most once per tick

pub mod engine;
pub mod indicators;
pub mod risk;
pub mod rolling;
pub mod stats;
pub mod tracker;

// Re-export main types at crate root for convenience
pub use engine::{IndicatorCache, IndicatorEngine, IndicatorSnapshot};
pub use indicators::{Atr, BollingerBands, Ema, Indicator, Macd, Rsi, Sma, create_indicator};
pub use risk::{
    RiskMetrics, annualized_volatility, historical_var, max_drawdown, sharpe_ratio, sortino_ratio,
};
pub use rolling::RollingWindow;
pub use tracker::{AgentRiskSnapshot, AgentRiskTracker, RiskTrackerConfig};

// V3.3: Multi-symbol strategy support
pub use stats::{
    CointegrationResult, CointegrationTracker, NewsEventLike, SectorSentiment,
    SectorSentimentAggregator,
};
