//! Unified feature schema for ML training and inference.
//!
//! This module provides a single source of truth for feature extraction,
//! eliminating training-serving skew risk by sharing constants and pure
//! computation functions between storage (training) and agents (inference).
//!
//! # Design Philosophy
//!
//! - **Declarative**: Feature schema defined via `FeatureDescriptor` + `FeatureRegistry`
//! - **Modular**: Features grouped by signal type via `FeatureGroup` for ablation
//! - **SoC**: Declaration (types crate), computation (agents crate), imputation (runner)
//! - **Pure Functions**: All computations are side-effect-free for testability
//! - **Type-Safe Indices**: Named constants prevent magic number bugs
//!
//! # Architecture
//!
//! Each feature is described by a `FeatureDescriptor` (index, name, group, neutral, valid_range).
//! Descriptors are collected in static `FeatureRegistry` instances:
//! - `MINIMAL_REGISTRY` (42 features, V5 compat)
//! - `FULL_REGISTRY` (55 features, V6.1)
//!
//! ```ignore
//! use types::features::{FULL_REGISTRY, FeatureGroup, extended_idx};
//!
//! let registry = &FULL_REGISTRY;
//! let vol_indices = registry.group_indices(FeatureGroup::Volatility);
//! let neutrals = registry.neutrals();
//! ```

use crate::{Candle, IndicatorType};

// =============================================================================
// Feature Groups
// =============================================================================

/// Signal groups for feature ablation and modular extraction.
///
/// Groups have contiguous index ranges enabling clean group-level operations:
/// disable one group, measure accuracy impact, aggregate SHAP values per group.
///
/// V5 groups (0-41): Price, TechnicalIndicator, News
/// V6.1 groups (42-54): Microstructure, Volatility, Fundamental, MomentumQuality, VolumeCross
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureGroup {
    /// Raw price and returns (indices 0-24).
    Price,
    /// Technical indicators from quant crate (indices 25-37).
    TechnicalIndicator,
    /// News and sentiment features (indices 38-41).
    News,
    /// Microstructure: spread, book imbalance, order flow (indices 42-44).
    Microstructure,
    /// Volatility regime: realized vol, vol ratio (indices 45-47).
    Volatility,
    /// Fundamental value: fair value deviation, price-to-fair (indices 48-49).
    Fundamental,
    /// Momentum quality: trend strength, RSI divergence (indices 50-51).
    MomentumQuality,
    /// Volume dynamics and cross-feature interactions (indices 52-54).
    VolumeCross,
}

impl FeatureGroup {
    /// All feature groups in index order.
    pub const ALL: [FeatureGroup; 8] = [
        Self::Price,
        Self::TechnicalIndicator,
        Self::News,
        Self::Microstructure,
        Self::Volatility,
        Self::Fundamental,
        Self::MomentumQuality,
        Self::VolumeCross,
    ];

    /// V5 groups only (first 42 features).
    pub const MINIMAL: [FeatureGroup; 3] = [Self::Price, Self::TechnicalIndicator, Self::News];

    /// V6.3 canonical groups (28 SHAP-validated features).
    pub const CANONICAL: [FeatureGroup; 5] = [
        Self::Price,
        Self::TechnicalIndicator,
        Self::Volatility,
        Self::Fundamental,
        Self::MomentumQuality,
    ];
}

// =============================================================================
// Feature Descriptors
// =============================================================================

/// Complete description of a single feature.
///
/// Co-locates all metadata that was previously scattered across
/// `idx` constants, `MARKET_FEATURE_NAMES`, and `MINIMAL_FEATURE_NEUTRALS`.
/// This is the authoritative definition — parallel arrays are verified against it.
#[derive(Debug, Clone, Copy)]
pub struct FeatureDescriptor {
    /// Positional index in the feature vector. Must equal array position.
    pub index: usize,
    /// Machine-readable name (e.g., "f_spread_bps"). Used in Parquet, SHAP, PyO3.
    pub name: &'static str,
    /// Signal group for ablation testing and modular extraction.
    pub group: FeatureGroup,
    /// Imputation value when data is missing (NaN). Semantically "no signal."
    pub neutral: f64,
    /// Expected value range `(min, max)` for validation and NN normalization (V7.2).
    /// `f64::NEG_INFINITY`/`f64::INFINITY` for unbounded features (need z-score normalization).
    pub valid_range: (f64, f64),
}

// =============================================================================
// Feature Registry
// =============================================================================

/// Central registry of feature metadata.
///
/// Wraps static descriptor, name, and neutral arrays. Provides derived accessors
/// for downstream consumers (SHAP analysis, gym observation space, PyO3 export,
/// NN normalization, ablation testing).
///
/// Two static instances exist:
/// - `MINIMAL_REGISTRY` — 42 features (V5 compat, neutrals all -1.0)
/// - `FULL_REGISTRY` — 55 features (V6.1, semantic neutrals)
pub struct FeatureRegistry {
    descriptors: &'static [FeatureDescriptor],
    /// Pre-computed name slice (Rust can't derive &[&str] from &[FeatureDescriptor] at const time).
    names: &'static [&'static str],
    /// Pre-computed neutral slice.
    neutrals: &'static [f64],
}

impl FeatureRegistry {
    /// Total number of features in this registry.
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Whether this registry is empty.
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Get descriptor by index. Panics if out of bounds.
    pub fn get(&self, index: usize) -> &FeatureDescriptor {
        &self.descriptors[index]
    }

    /// All descriptors.
    pub fn descriptors(&self) -> &[FeatureDescriptor] {
        self.descriptors
    }

    /// Feature names in index order. Zero-alloc access.
    pub fn names(&self) -> &[&str] {
        self.names
    }

    /// Neutral (imputation) values in index order. Zero-alloc access.
    pub fn neutrals(&self) -> &[f64] {
        self.neutrals
    }

    /// Indices belonging to a specific group (for ablation testing).
    pub fn group_indices(&self, group: FeatureGroup) -> Vec<usize> {
        self.descriptors
            .iter()
            .filter(|d| d.group == group)
            .map(|d| d.index)
            .collect()
    }

    /// Valid ranges as `(min, max)` tuples (for V7.2 NN normalization).
    pub fn valid_ranges(&self) -> Vec<(f64, f64)> {
        self.descriptors.iter().map(|d| d.valid_range).collect()
    }

    /// Validate a feature vector against expected ranges.
    /// Returns indices of out-of-range features (for monitoring/logging).
    pub fn validate(&self, features: &[f64]) -> Vec<usize> {
        self.descriptors
            .iter()
            .filter(|d| {
                if d.index >= features.len() {
                    return false;
                }
                let val = features[d.index];
                val.is_finite() && (val < d.valid_range.0 || val > d.valid_range.1)
            })
            .map(|d| d.index)
            .collect()
    }
}

// Allow FeatureRegistry to be shared across threads (all data is 'static).
unsafe impl Sync for FeatureRegistry {}
unsafe impl Send for FeatureRegistry {}

// =============================================================================
// Constants
// =============================================================================

/// Lookback periods for price changes and log returns.
/// Geometric spread optimized for batch auction tick intervals.
pub const LOOKBACKS: &[usize] = &[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

/// Number of lookback periods.
pub const N_LOOKBACKS: usize = 12;

/// Total number of V5 market-level features.
pub const N_MARKET_FEATURES: usize = 42;

/// Total number of V6.1 full features (42 base + 13 new).
pub const N_FULL_FEATURES: usize = 55;

/// Trade intensity baseline (average trades per tick for normalization).
pub const TRADE_INTENSITY_BASELINE: f64 = 8.0;

// =============================================================================
// Feature Indices — V5 Base (Type-Safe Access)
// =============================================================================

/// Named feature indices for type-safe array access (V5 base, 0-41).
///
/// Use these instead of magic numbers to prevent training-serving skew.
pub mod idx {
    /// Mid price at current tick.
    pub const MID_PRICE: usize = 0;

    /// Start of price change features (12 values for each lookback).
    pub const PRICE_CHANGE_START: usize = 1;

    /// Start of log return features (12 values for each lookback).
    pub const LOG_RETURN_START: usize = 13;

    /// SMA with 8-tick period.
    pub const SMA_8: usize = 25;
    /// SMA with 16-tick period.
    pub const SMA_16: usize = 26;
    /// EMA with 8-tick period.
    pub const EMA_8: usize = 27;
    /// EMA with 16-tick period.
    pub const EMA_16: usize = 28;
    /// RSI with 8-tick period.
    pub const RSI_8: usize = 29;

    /// MACD line (fast EMA - slow EMA).
    pub const MACD_LINE: usize = 30;
    /// MACD signal line (EMA of MACD line).
    pub const MACD_SIGNAL: usize = 31;
    /// MACD histogram (line - signal).
    pub const MACD_HISTOGRAM: usize = 32;

    /// Bollinger upper band.
    pub const BB_UPPER: usize = 33;
    /// Bollinger middle band (SMA).
    pub const BB_MIDDLE: usize = 34;
    /// Bollinger lower band.
    pub const BB_LOWER: usize = 35;
    /// Bollinger %B (normalized position within bands).
    pub const BB_PERCENT_B: usize = 36;

    /// ATR with 8-tick period.
    pub const ATR_8: usize = 37;

    /// Binary indicator for active news event.
    pub const HAS_ACTIVE_NEWS: usize = 38;
    /// News sentiment (-1 to +1).
    pub const NEWS_SENTIMENT: usize = 39;
    /// News magnitude (impact strength).
    pub const NEWS_MAGNITUDE: usize = 40;
    /// Ticks remaining in news event.
    pub const NEWS_TICKS_REMAINING: usize = 41;
}

// =============================================================================
// Feature Indices — V6.1 Extensions (Type-Safe Access)
// =============================================================================

/// Named feature indices for V6.1 new features (42-54).
///
/// Indices are contiguous by group to enable clean ablation testing.
pub mod extended_idx {
    // Microstructure (42-44)
    /// Spread in basis points: (ask - bid) / mid * 10000.
    pub const SPREAD_BPS: usize = 42;
    /// Book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).
    pub const BOOK_IMBALANCE: usize = 43;
    /// Net order flow: (n_buyers - n_sellers) / (n_buyers + n_sellers).
    pub const NET_ORDER_FLOW: usize = 44;

    // Volatility (45-47)
    /// Realized volatility over 8 ticks: std(sequential log returns).
    pub const REALIZED_VOL_8: usize = 45;
    /// Realized volatility over 32 ticks: std(sequential log returns).
    pub const REALIZED_VOL_32: usize = 46;
    /// Volatility ratio: vol_8 / vol_32. >1 = expanding, <1 = contracting.
    pub const VOL_RATIO: usize = 47;

    // Fundamental (48-49)
    /// Fair value deviation: (mid - fair_value) / fair_value.
    pub const FAIR_VALUE_DEV: usize = 48;
    /// Price-to-fair ratio: mid / fair_value.
    pub const PRICE_TO_FAIR: usize = 49;

    // Momentum quality (50-51)
    /// Trend strength: abs(ema_8 - ema_16) / atr_8.
    pub const TREND_STRENGTH: usize = 50;
    /// RSI divergence: rsi_8 - 50.0. Range [-50, +50].
    pub const RSI_DIVERGENCE: usize = 51;

    // Volume/cross (52-54)
    /// Volume surge: latest_volume / avg_volume_8.
    pub const VOLUME_SURGE: usize = 52;
    /// Trade intensity: n_recent_trades / baseline.
    pub const TRADE_INTENSITY: usize = 53;
    /// Sentiment-price gap: symbol_sentiment * fair_value_dev.
    pub const SENTIMENT_PRICE_GAP: usize = 54;
}

// =============================================================================
// Feature Names
// =============================================================================

/// V5 market feature names (42 total).
///
/// Order matches feature indices for direct array indexing.
pub const MARKET_FEATURE_NAMES: &[&str] = &[
    // Price features (25) - geometric lookbacks: 1,2,3,4,6,8,12,16,24,32,48,64
    "f_mid_price",
    "f_price_change_1",
    "f_price_change_2",
    "f_price_change_3",
    "f_price_change_4",
    "f_price_change_6",
    "f_price_change_8",
    "f_price_change_12",
    "f_price_change_16",
    "f_price_change_24",
    "f_price_change_32",
    "f_price_change_48",
    "f_price_change_64",
    "f_log_return_1",
    "f_log_return_2",
    "f_log_return_3",
    "f_log_return_4",
    "f_log_return_6",
    "f_log_return_8",
    "f_log_return_12",
    "f_log_return_16",
    "f_log_return_24",
    "f_log_return_32",
    "f_log_return_48",
    "f_log_return_64",
    // Technical indicators (13) - from quant crate, 8/16 spread
    "f_sma_8",
    "f_sma_16",
    "f_ema_8",
    "f_ema_16",
    "f_rsi_8",
    "f_macd_line",
    "f_macd_signal",
    "f_macd_histogram",
    "f_bb_upper",
    "f_bb_middle",
    "f_bb_lower",
    "f_bb_percent_b",
    "f_atr_8",
    // News/sentiment features (4)
    "f_has_active_news",
    "f_news_sentiment",
    "f_news_magnitude",
    "f_news_ticks_remaining",
];

/// V6.1 full feature names (55 total = 42 base + 13 new).
pub const FULL_FEATURE_NAMES: &[&str] = &[
    // Base 42 (same as MARKET_FEATURE_NAMES)
    "f_mid_price",
    "f_price_change_1",
    "f_price_change_2",
    "f_price_change_3",
    "f_price_change_4",
    "f_price_change_6",
    "f_price_change_8",
    "f_price_change_12",
    "f_price_change_16",
    "f_price_change_24",
    "f_price_change_32",
    "f_price_change_48",
    "f_price_change_64",
    "f_log_return_1",
    "f_log_return_2",
    "f_log_return_3",
    "f_log_return_4",
    "f_log_return_6",
    "f_log_return_8",
    "f_log_return_12",
    "f_log_return_16",
    "f_log_return_24",
    "f_log_return_32",
    "f_log_return_48",
    "f_log_return_64",
    "f_sma_8",
    "f_sma_16",
    "f_ema_8",
    "f_ema_16",
    "f_rsi_8",
    "f_macd_line",
    "f_macd_signal",
    "f_macd_histogram",
    "f_bb_upper",
    "f_bb_middle",
    "f_bb_lower",
    "f_bb_percent_b",
    "f_atr_8",
    "f_has_active_news",
    "f_news_sentiment",
    "f_news_magnitude",
    "f_news_ticks_remaining",
    // V6.1 new features (13) - contiguous by group
    "f_spread_bps",
    "f_book_imbalance",
    "f_net_order_flow",
    "f_realized_vol_8",
    "f_realized_vol_32",
    "f_vol_ratio",
    "f_fair_value_dev",
    "f_price_to_fair",
    "f_trend_strength",
    "f_rsi_divergence",
    "f_volume_surge",
    "f_trade_intensity",
    "f_sentiment_price_gap",
];

// =============================================================================
// Neutral (Imputation) Values
// =============================================================================

/// Per-feature neutral values for NaN imputation (V5 MinimalFeatures).
///
/// V5 training used `nan_to_num(X, nan=-1.0)` uniformly. All 42 features
/// impute to -1.0 for backward compatibility with trained V5 models.
pub const MINIMAL_FEATURE_NEUTRALS: [f64; N_MARKET_FEATURES] = [-1.0; N_MARKET_FEATURES];

/// Per-feature neutral values for NaN imputation (V6.1 FullFeatures).
///
/// Each value represents "no signal" for that feature type. Semantic neutrals
/// instead of uniform -1.0. Important for V7.2 neural networks where -1.0
/// falls within the normal range of many features.
#[rustfmt::skip]
pub const FULL_FEATURE_NEUTRALS: [f64; N_FULL_FEATURES] = [
    // Price (25): 0.0 = no change / no price info
    0.0,                                                                // f_mid_price
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,     // f_price_change_*
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,     // f_log_return_*
    // Technical indicators (13)
    0.0, 0.0,           // SMA 8/16 (raw price scale)
    0.0, 0.0,           // EMA 8/16
    50.0,                // RSI 8 (neutral midpoint)
    0.0, 0.0, 0.0,      // MACD line/signal/histogram (no trend)
    0.0, 0.0, 0.0,      // BB upper/middle/lower (raw price scale)
    0.5,                 // BB %B (middle of band)
    0.0,                 // ATR 8 (no volatility info)
    // News (4): 0.0 = no news
    0.0, 0.0, 0.0, 0.0,
    // Microstructure (3)
    0.0,                 // spread_bps (tight spread assumed)
    0.0,                 // book_imbalance (balanced)
    0.0,                 // net_order_flow (balanced)
    // Volatility (3)
    0.0,                 // realized_vol_8 (zero vol during warmup)
    0.0,                 // realized_vol_32
    1.0,                 // vol_ratio (no regime change)
    // Fundamental (2)
    0.0,                 // fair_value_dev (at fair value)
    1.0,                 // price_to_fair (at fair value)
    // Momentum quality (2)
    0.0,                 // trend_strength (no trend)
    0.0,                 // rsi_divergence (neutral RSI)
    // Volume/cross (3)
    1.0,                 // volume_surge (normal volume)
    0.0,                 // trade_intensity (no trades)
    0.0,                 // sentiment_price_gap (no signal)
];

// =============================================================================
// Descriptor Arrays (Authoritative Definitions)
// =============================================================================

use FeatureGroup::*;

/// Shorthand for FeatureDescriptor construction.
const fn desc(
    index: usize,
    name: &'static str,
    group: FeatureGroup,
    neutral: f64,
    valid_range: (f64, f64),
) -> FeatureDescriptor {
    FeatureDescriptor {
        index,
        name,
        group,
        neutral,
        valid_range,
    }
}

/// INF range shorthand for unbounded features (need z-score normalization for V7.2).
const INF: (f64, f64) = (f64::NEG_INFINITY, f64::INFINITY);

/// V5 minimal feature descriptors (42 features, neutrals all -1.0).
#[rustfmt::skip]
pub static MINIMAL_DESCRIPTORS: [FeatureDescriptor; N_MARKET_FEATURES] = [
    // Price (25)
    desc( 0, "f_mid_price",        Price, -1.0, INF),
    desc( 1, "f_price_change_1",   Price, -1.0, (-100.0, 100.0)),
    desc( 2, "f_price_change_2",   Price, -1.0, (-100.0, 100.0)),
    desc( 3, "f_price_change_3",   Price, -1.0, (-100.0, 100.0)),
    desc( 4, "f_price_change_4",   Price, -1.0, (-100.0, 100.0)),
    desc( 5, "f_price_change_6",   Price, -1.0, (-100.0, 100.0)),
    desc( 6, "f_price_change_8",   Price, -1.0, (-100.0, 100.0)),
    desc( 7, "f_price_change_12",  Price, -1.0, (-100.0, 100.0)),
    desc( 8, "f_price_change_16",  Price, -1.0, (-100.0, 100.0)),
    desc( 9, "f_price_change_24",  Price, -1.0, (-100.0, 100.0)),
    desc(10, "f_price_change_32",  Price, -1.0, (-100.0, 100.0)),
    desc(11, "f_price_change_48",  Price, -1.0, (-100.0, 100.0)),
    desc(12, "f_price_change_64",  Price, -1.0, (-100.0, 100.0)),
    desc(13, "f_log_return_1",     Price, -1.0, (-5.0, 5.0)),
    desc(14, "f_log_return_2",     Price, -1.0, (-5.0, 5.0)),
    desc(15, "f_log_return_3",     Price, -1.0, (-5.0, 5.0)),
    desc(16, "f_log_return_4",     Price, -1.0, (-5.0, 5.0)),
    desc(17, "f_log_return_6",     Price, -1.0, (-5.0, 5.0)),
    desc(18, "f_log_return_8",     Price, -1.0, (-5.0, 5.0)),
    desc(19, "f_log_return_12",    Price, -1.0, (-5.0, 5.0)),
    desc(20, "f_log_return_16",    Price, -1.0, (-5.0, 5.0)),
    desc(21, "f_log_return_24",    Price, -1.0, (-5.0, 5.0)),
    desc(22, "f_log_return_32",    Price, -1.0, (-5.0, 5.0)),
    desc(23, "f_log_return_48",    Price, -1.0, (-5.0, 5.0)),
    desc(24, "f_log_return_64",    Price, -1.0, (-5.0, 5.0)),
    // Technical Indicators (13)
    desc(25, "f_sma_8",            TechnicalIndicator, -1.0, INF),
    desc(26, "f_sma_16",           TechnicalIndicator, -1.0, INF),
    desc(27, "f_ema_8",            TechnicalIndicator, -1.0, INF),
    desc(28, "f_ema_16",           TechnicalIndicator, -1.0, INF),
    desc(29, "f_rsi_8",            TechnicalIndicator, -1.0, (0.0, 100.0)),
    desc(30, "f_macd_line",        TechnicalIndicator, -1.0, INF),
    desc(31, "f_macd_signal",      TechnicalIndicator, -1.0, INF),
    desc(32, "f_macd_histogram",   TechnicalIndicator, -1.0, INF),
    desc(33, "f_bb_upper",         TechnicalIndicator, -1.0, INF),
    desc(34, "f_bb_middle",        TechnicalIndicator, -1.0, INF),
    desc(35, "f_bb_lower",         TechnicalIndicator, -1.0, INF),
    desc(36, "f_bb_percent_b",     TechnicalIndicator, -1.0, (-2.0, 3.0)),
    desc(37, "f_atr_8",            TechnicalIndicator, -1.0, (0.0, f64::INFINITY)),
    // News (4)
    desc(38, "f_has_active_news",      News, -1.0, (0.0, 1.0)),
    desc(39, "f_news_sentiment",       News, -1.0, (-1.0, 1.0)),
    desc(40, "f_news_magnitude",       News, -1.0, (0.0, 10.0)),
    desc(41, "f_news_ticks_remaining", News, -1.0, (0.0, 1000.0)),
];

/// V6.1 full feature descriptors (55 features, semantic neutrals).
///
/// First 42 entries have semantic neutrals (different from MINIMAL which uses -1.0).
/// Models trained on V5 data use MinimalFeatures. FullFeatures requires new training.
#[rustfmt::skip]
pub static FULL_DESCRIPTORS: [FeatureDescriptor; N_FULL_FEATURES] = [
    // Price (25)
    desc( 0, "f_mid_price",        Price, 0.0, INF),
    desc( 1, "f_price_change_1",   Price, 0.0, (-100.0, 100.0)),
    desc( 2, "f_price_change_2",   Price, 0.0, (-100.0, 100.0)),
    desc( 3, "f_price_change_3",   Price, 0.0, (-100.0, 100.0)),
    desc( 4, "f_price_change_4",   Price, 0.0, (-100.0, 100.0)),
    desc( 5, "f_price_change_6",   Price, 0.0, (-100.0, 100.0)),
    desc( 6, "f_price_change_8",   Price, 0.0, (-100.0, 100.0)),
    desc( 7, "f_price_change_12",  Price, 0.0, (-100.0, 100.0)),
    desc( 8, "f_price_change_16",  Price, 0.0, (-100.0, 100.0)),
    desc( 9, "f_price_change_24",  Price, 0.0, (-100.0, 100.0)),
    desc(10, "f_price_change_32",  Price, 0.0, (-100.0, 100.0)),
    desc(11, "f_price_change_48",  Price, 0.0, (-100.0, 100.0)),
    desc(12, "f_price_change_64",  Price, 0.0, (-100.0, 100.0)),
    desc(13, "f_log_return_1",     Price, 0.0, (-5.0, 5.0)),
    desc(14, "f_log_return_2",     Price, 0.0, (-5.0, 5.0)),
    desc(15, "f_log_return_3",     Price, 0.0, (-5.0, 5.0)),
    desc(16, "f_log_return_4",     Price, 0.0, (-5.0, 5.0)),
    desc(17, "f_log_return_6",     Price, 0.0, (-5.0, 5.0)),
    desc(18, "f_log_return_8",     Price, 0.0, (-5.0, 5.0)),
    desc(19, "f_log_return_12",    Price, 0.0, (-5.0, 5.0)),
    desc(20, "f_log_return_16",    Price, 0.0, (-5.0, 5.0)),
    desc(21, "f_log_return_24",    Price, 0.0, (-5.0, 5.0)),
    desc(22, "f_log_return_32",    Price, 0.0, (-5.0, 5.0)),
    desc(23, "f_log_return_48",    Price, 0.0, (-5.0, 5.0)),
    desc(24, "f_log_return_64",    Price, 0.0, (-5.0, 5.0)),
    // Technical Indicators (13)
    desc(25, "f_sma_8",            TechnicalIndicator,  0.0,  INF),
    desc(26, "f_sma_16",           TechnicalIndicator,  0.0,  INF),
    desc(27, "f_ema_8",            TechnicalIndicator,  0.0,  INF),
    desc(28, "f_ema_16",           TechnicalIndicator,  0.0,  INF),
    desc(29, "f_rsi_8",            TechnicalIndicator, 50.0,  (0.0, 100.0)),
    desc(30, "f_macd_line",        TechnicalIndicator,  0.0,  INF),
    desc(31, "f_macd_signal",      TechnicalIndicator,  0.0,  INF),
    desc(32, "f_macd_histogram",   TechnicalIndicator,  0.0,  INF),
    desc(33, "f_bb_upper",         TechnicalIndicator,  0.0,  INF),
    desc(34, "f_bb_middle",        TechnicalIndicator,  0.0,  INF),
    desc(35, "f_bb_lower",         TechnicalIndicator,  0.0,  INF),
    desc(36, "f_bb_percent_b",     TechnicalIndicator,  0.5,  (-2.0, 3.0)),
    desc(37, "f_atr_8",            TechnicalIndicator,  0.0,  (0.0, f64::INFINITY)),
    // News (4)
    desc(38, "f_has_active_news",      News, 0.0, (0.0, 1.0)),
    desc(39, "f_news_sentiment",       News, 0.0, (-1.0, 1.0)),
    desc(40, "f_news_magnitude",       News, 0.0, (0.0, 10.0)),
    desc(41, "f_news_ticks_remaining", News, 0.0, (0.0, 1000.0)),
    // Microstructure (3) — V6.1 new
    desc(42, "f_spread_bps",       Microstructure, 0.0, (0.0, 1000.0)),
    desc(43, "f_book_imbalance",   Microstructure, 0.0, (-1.0, 1.0)),
    desc(44, "f_net_order_flow",   Microstructure, 0.0, (-1.0, 1.0)),
    // Volatility (3) — V6.1 new
    desc(45, "f_realized_vol_8",   Volatility, 0.0, (0.0, 1.0)),
    desc(46, "f_realized_vol_32",  Volatility, 0.0, (0.0, 1.0)),
    desc(47, "f_vol_ratio",        Volatility, 1.0, (0.0, 10.0)),
    // Fundamental (2) — V6.1 new
    desc(48, "f_fair_value_dev",   Fundamental, 0.0, (-1.0, 1.0)),
    desc(49, "f_price_to_fair",    Fundamental, 1.0, (0.0, 5.0)),
    // Momentum quality (2) — V6.1 new
    desc(50, "f_trend_strength",   MomentumQuality, 0.0, (0.0, 10.0)),
    desc(51, "f_rsi_divergence",   MomentumQuality, 0.0, (-50.0, 50.0)),
    // Volume/cross (3) — V6.1 new
    desc(52, "f_volume_surge",     VolumeCross, 1.0, (0.0, 100.0)),
    desc(53, "f_trade_intensity",  VolumeCross, 0.0, (0.0, 100.0)),
    desc(54, "f_sentiment_price_gap", VolumeCross, 0.0, (-1.0, 1.0)),
];

// =============================================================================
// Static Registries
// =============================================================================

/// V5 feature registry (42 features, backward compatible).
///
/// All neutrals are -1.0 matching V5 training convention `nan_to_num(X, nan=-1.0)`.
pub static MINIMAL_REGISTRY: FeatureRegistry = FeatureRegistry {
    descriptors: &MINIMAL_DESCRIPTORS,
    names: MARKET_FEATURE_NAMES,
    neutrals: &MINIMAL_FEATURE_NEUTRALS,
};

/// V6.1 full feature registry (55 features, semantic neutrals).
///
/// Requires new model training — V5 models are NOT compatible with these neutrals.
/// Use `MinimalFeatures` for V5 backward compat, `FullFeatures` for V6.1+.
pub static FULL_REGISTRY: FeatureRegistry = FeatureRegistry {
    descriptors: &FULL_DESCRIPTORS,
    names: FULL_FEATURE_NAMES,
    neutrals: &FULL_FEATURE_NEUTRALS,
};

// =============================================================================
// V6.3 Canonical Schema (28 SHAP-validated features)
// =============================================================================

/// Total number of V6.3 canonical features (SHAP-validated subset of V6.1).
pub const N_CANONICAL_FEATURES: usize = 28;

/// Lookback periods used by canonical Price features.
pub const CANONICAL_LOOKBACKS: &[usize] = &[1, 32, 48, 64];

/// Named feature indices for V6.3 canonical features (0-27).
///
/// 28 SHAP-validated features from 5 groups:
/// Price (8), Technical (13), Volatility (3), Fundamental (2), MomentumQuality (2).
pub mod canonical_idx {
    // Price (0-7)
    pub const MID_PRICE: usize = 0;
    pub const PRICE_CHANGE_1: usize = 1;
    pub const PRICE_CHANGE_32: usize = 2;
    pub const PRICE_CHANGE_48: usize = 3;
    pub const PRICE_CHANGE_64: usize = 4;
    pub const LOG_RETURN_32: usize = 5;
    pub const LOG_RETURN_48: usize = 6;
    pub const LOG_RETURN_64: usize = 7;

    // Technical (8-20)
    pub const SMA_8: usize = 8;
    pub const SMA_16: usize = 9;
    pub const EMA_8: usize = 10;
    pub const EMA_16: usize = 11;
    pub const RSI_8: usize = 12;
    pub const MACD_LINE: usize = 13;
    pub const MACD_SIGNAL: usize = 14;
    pub const MACD_HISTOGRAM: usize = 15;
    pub const BB_UPPER: usize = 16;
    pub const BB_MIDDLE: usize = 17;
    pub const BB_LOWER: usize = 18;
    pub const BB_PERCENT_B: usize = 19;
    pub const ATR_8: usize = 20;

    // Volatility (21-23)
    pub const REALIZED_VOL_8: usize = 21;
    pub const REALIZED_VOL_32: usize = 22;
    pub const VOL_RATIO: usize = 23;

    // Fundamental (24-25)
    pub const FAIR_VALUE_DEV: usize = 24;
    pub const PRICE_TO_FAIR: usize = 25;

    // MomentumQuality (26-27)
    pub const TREND_STRENGTH: usize = 26;
    pub const RSI_DIVERGENCE: usize = 27;
}

/// V6.3 canonical feature names (28 total).
pub const CANONICAL_FEATURE_NAMES: &[&str] = &[
    // Price (8)
    "f_mid_price",
    "f_price_change_1",
    "f_price_change_32",
    "f_price_change_48",
    "f_price_change_64",
    "f_log_return_32",
    "f_log_return_48",
    "f_log_return_64",
    // Technical (13)
    "f_sma_8",
    "f_sma_16",
    "f_ema_8",
    "f_ema_16",
    "f_rsi_8",
    "f_macd_line",
    "f_macd_signal",
    "f_macd_histogram",
    "f_bb_upper",
    "f_bb_middle",
    "f_bb_lower",
    "f_bb_percent_b",
    "f_atr_8",
    // Volatility (3)
    "f_realized_vol_8",
    "f_realized_vol_32",
    "f_vol_ratio",
    // Fundamental (2)
    "f_fair_value_dev",
    "f_price_to_fair",
    // MomentumQuality (2)
    "f_trend_strength",
    "f_rsi_divergence",
];

/// Per-feature neutral values for NaN imputation (V6.3 canonical).
#[rustfmt::skip]
pub const CANONICAL_FEATURE_NEUTRALS: [f64; N_CANONICAL_FEATURES] = [
    // Price (8): 0.0 = no change / no price info
    0.0,                            // f_mid_price
    0.0, 0.0, 0.0, 0.0,            // f_price_change_{1,32,48,64}
    0.0, 0.0, 0.0,                  // f_log_return_{32,48,64}
    // Technical (13)
    0.0, 0.0,                       // SMA 8/16
    0.0, 0.0,                       // EMA 8/16
    50.0,                            // RSI 8 (neutral midpoint)
    0.0, 0.0, 0.0,                  // MACD line/signal/histogram
    0.0, 0.0, 0.0,                  // BB upper/middle/lower
    0.5,                             // BB %B (middle of band)
    0.0,                             // ATR 8
    // Volatility (3)
    0.0, 0.0,                       // realized_vol 8/32
    1.0,                             // vol_ratio (no regime change)
    // Fundamental (2)
    0.0,                             // fair_value_dev (at fair value)
    1.0,                             // price_to_fair (at fair value)
    // MomentumQuality (2)
    0.0,                             // trend_strength (no trend)
    0.0,                             // rsi_divergence (neutral RSI)
];

/// V6.3 canonical feature descriptors (28 features, 5 groups).
#[rustfmt::skip]
pub static CANONICAL_DESCRIPTORS: [FeatureDescriptor; N_CANONICAL_FEATURES] = [
    // Price (8)
    desc( 0, "f_mid_price",        Price, 0.0, INF),
    desc( 1, "f_price_change_1",   Price, 0.0, (-100.0, 100.0)),
    desc( 2, "f_price_change_32",  Price, 0.0, (-100.0, 100.0)),
    desc( 3, "f_price_change_48",  Price, 0.0, (-100.0, 100.0)),
    desc( 4, "f_price_change_64",  Price, 0.0, (-100.0, 100.0)),
    desc( 5, "f_log_return_32",    Price, 0.0, (-5.0, 5.0)),
    desc( 6, "f_log_return_48",    Price, 0.0, (-5.0, 5.0)),
    desc( 7, "f_log_return_64",    Price, 0.0, (-5.0, 5.0)),
    // Technical (13)
    desc( 8, "f_sma_8",            TechnicalIndicator,  0.0,  INF),
    desc( 9, "f_sma_16",           TechnicalIndicator,  0.0,  INF),
    desc(10, "f_ema_8",            TechnicalIndicator,  0.0,  INF),
    desc(11, "f_ema_16",           TechnicalIndicator,  0.0,  INF),
    desc(12, "f_rsi_8",            TechnicalIndicator, 50.0,  (0.0, 100.0)),
    desc(13, "f_macd_line",        TechnicalIndicator,  0.0,  INF),
    desc(14, "f_macd_signal",      TechnicalIndicator,  0.0,  INF),
    desc(15, "f_macd_histogram",   TechnicalIndicator,  0.0,  INF),
    desc(16, "f_bb_upper",         TechnicalIndicator,  0.0,  INF),
    desc(17, "f_bb_middle",        TechnicalIndicator,  0.0,  INF),
    desc(18, "f_bb_lower",         TechnicalIndicator,  0.0,  INF),
    desc(19, "f_bb_percent_b",     TechnicalIndicator,  0.5,  (-2.0, 3.0)),
    desc(20, "f_atr_8",            TechnicalIndicator,  0.0,  (0.0, f64::INFINITY)),
    // Volatility (3)
    desc(21, "f_realized_vol_8",   Volatility, 0.0, (0.0, 1.0)),
    desc(22, "f_realized_vol_32",  Volatility, 0.0, (0.0, 1.0)),
    desc(23, "f_vol_ratio",        Volatility, 1.0, (0.0, 10.0)),
    // Fundamental (2)
    desc(24, "f_fair_value_dev",   Fundamental, 0.0, (-1.0, 1.0)),
    desc(25, "f_price_to_fair",    Fundamental, 1.0, (0.0, 5.0)),
    // MomentumQuality (2)
    desc(26, "f_trend_strength",   MomentumQuality, 0.0, (0.0, 10.0)),
    desc(27, "f_rsi_divergence",   MomentumQuality, 0.0, (-50.0, 50.0)),
];

/// V6.3 canonical feature registry (28 features, semantic neutrals).
pub static CANONICAL_REGISTRY: FeatureRegistry = FeatureRegistry {
    descriptors: &CANONICAL_DESCRIPTORS,
    names: CANONICAL_FEATURE_NAMES,
    neutrals: &CANONICAL_FEATURE_NEUTRALS,
};

// =============================================================================
// Pure Computation Functions
// =============================================================================

/// Compute price change percentage.
///
/// Returns `(current - past) / past * 100`, or NaN if past <= 0.
#[inline]
pub fn price_change_pct(current: f64, past: f64) -> f64 {
    if past > 0.0 {
        (current - past) / past * 100.0
    } else {
        f64::NAN
    }
}

/// Compute log return.
///
/// Returns `ln(current / past)`, or NaN if either price is non-positive.
#[inline]
pub fn log_return(current: f64, past: f64) -> f64 {
    if current > 0.0 && past > 0.0 {
        (current / past).ln()
    } else {
        f64::NAN
    }
}

/// Compute Bollinger %B (normalized position within bands).
///
/// Returns `(price - lower) / (upper - lower)`, or NaN if inputs invalid.
/// When bands converge (width < 1e-10), returns 0.5 (price at center).
#[inline]
pub fn bollinger_percent_b(price: f64, upper: f64, lower: f64) -> f64 {
    if upper.is_finite() && lower.is_finite() && price.is_finite() {
        let width = upper - lower;
        if width > 1e-10 {
            (price - lower) / width
        } else {
            0.5
        }
    } else {
        f64::NAN
    }
}

/// Compute price change from candle history.
///
/// Looks back `lookback` candles from the most recent and computes percentage change.
/// Returns NaN if insufficient history.
pub fn price_change_from_candles(candles: &[Candle], lookback: usize) -> f64 {
    if candles.len() < lookback + 1 {
        return f64::NAN;
    }
    let current = candles[candles.len() - 1].close.to_float();
    let past = candles[candles.len() - 1 - lookback].close.to_float();
    price_change_pct(current, past)
}

/// Compute log return from candle history.
///
/// Looks back `lookback` candles from the most recent and computes log return.
/// Returns NaN if insufficient history.
pub fn log_return_from_candles(candles: &[Candle], lookback: usize) -> f64 {
    if candles.len() < lookback + 1 {
        return f64::NAN;
    }
    let current = candles[candles.len() - 1].close.to_float();
    let past = candles[candles.len() - 1 - lookback].close.to_float();
    log_return(current, past)
}

/// Compute realized volatility (standard deviation of sequential 1-period log returns).
///
/// IMPORTANT: Uses sequential returns `ln(close[t] / close[t-1])` for consecutive candles,
/// NOT cumulative returns from `log_return_from_candles()` which computes overlapping
/// cumulative returns whose std dev mechanically increases with horizon.
///
/// Returns NaN if insufficient history or fewer than 2 valid returns.
pub fn realized_volatility(candles: &[Candle], lookback: usize) -> f64 {
    if candles.len() < lookback + 1 {
        return f64::NAN;
    }
    let n = candles.len();
    let returns: Vec<f64> = (0..lookback)
        .map(|i| {
            let current = candles[n - 1 - i].close.to_float();
            let previous = candles[n - 2 - i].close.to_float();
            if current > 0.0 && previous > 0.0 {
                (current / previous).ln()
            } else {
                f64::NAN
            }
        })
        .filter(|r| r.is_finite())
        .collect();
    if returns.len() < 2 {
        return f64::NAN;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
    variance.sqrt()
}

/// Compute spread in basis points.
///
/// Returns `(ask - bid) / mid * 10000`, or NaN if mid <= 0.
#[inline]
pub fn spread_bps(bid: f64, ask: f64, mid: f64) -> f64 {
    if mid > 0.0 && bid.is_finite() && ask.is_finite() {
        (ask - bid) / mid * 10000.0
    } else {
        f64::NAN
    }
}

// =============================================================================
// Required Indicators
// =============================================================================

/// Returns the set of technical indicators required for feature extraction.
///
/// Use this to configure the indicator engine with exactly what's needed.
pub fn required_indicators() -> [IndicatorType; 12] {
    [
        IndicatorType::Sma(8),
        IndicatorType::Sma(16),
        IndicatorType::Ema(8),
        IndicatorType::Ema(16),
        IndicatorType::Rsi(8),
        IndicatorType::MACD_LINE_STANDARD,
        IndicatorType::MACD_SIGNAL_STANDARD,
        IndicatorType::MACD_HISTOGRAM_STANDARD,
        IndicatorType::BOLLINGER_UPPER_STANDARD,
        IndicatorType::BOLLINGER_MIDDLE_STANDARD,
        IndicatorType::BOLLINGER_LOWER_STANDARD,
        IndicatorType::Atr(8),
    ]
}

// =============================================================================
// Compile-Time Assertions
// =============================================================================

/// Exhaustive compile-time assertions for schema consistency.
///
/// Verifies:
/// - Every descriptor index equals its array position (no gaps, duplicates, or mismatches)
/// - MINIMAL and FULL prefix consistency (first 42 indices match)
/// - Name and neutral array lengths match descriptor arrays
/// - Legacy constant consistency
const _: () = {
    // MINIMAL: every index == position
    assert!(MINIMAL_DESCRIPTORS[0].index == 0);
    assert!(MINIMAL_DESCRIPTORS[1].index == 1);
    assert!(MINIMAL_DESCRIPTORS[2].index == 2);
    assert!(MINIMAL_DESCRIPTORS[3].index == 3);
    assert!(MINIMAL_DESCRIPTORS[4].index == 4);
    assert!(MINIMAL_DESCRIPTORS[5].index == 5);
    assert!(MINIMAL_DESCRIPTORS[6].index == 6);
    assert!(MINIMAL_DESCRIPTORS[7].index == 7);
    assert!(MINIMAL_DESCRIPTORS[8].index == 8);
    assert!(MINIMAL_DESCRIPTORS[9].index == 9);
    assert!(MINIMAL_DESCRIPTORS[10].index == 10);
    assert!(MINIMAL_DESCRIPTORS[11].index == 11);
    assert!(MINIMAL_DESCRIPTORS[12].index == 12);
    assert!(MINIMAL_DESCRIPTORS[13].index == 13);
    assert!(MINIMAL_DESCRIPTORS[14].index == 14);
    assert!(MINIMAL_DESCRIPTORS[15].index == 15);
    assert!(MINIMAL_DESCRIPTORS[16].index == 16);
    assert!(MINIMAL_DESCRIPTORS[17].index == 17);
    assert!(MINIMAL_DESCRIPTORS[18].index == 18);
    assert!(MINIMAL_DESCRIPTORS[19].index == 19);
    assert!(MINIMAL_DESCRIPTORS[20].index == 20);
    assert!(MINIMAL_DESCRIPTORS[21].index == 21);
    assert!(MINIMAL_DESCRIPTORS[22].index == 22);
    assert!(MINIMAL_DESCRIPTORS[23].index == 23);
    assert!(MINIMAL_DESCRIPTORS[24].index == 24);
    assert!(MINIMAL_DESCRIPTORS[25].index == 25);
    assert!(MINIMAL_DESCRIPTORS[26].index == 26);
    assert!(MINIMAL_DESCRIPTORS[27].index == 27);
    assert!(MINIMAL_DESCRIPTORS[28].index == 28);
    assert!(MINIMAL_DESCRIPTORS[29].index == 29);
    assert!(MINIMAL_DESCRIPTORS[30].index == 30);
    assert!(MINIMAL_DESCRIPTORS[31].index == 31);
    assert!(MINIMAL_DESCRIPTORS[32].index == 32);
    assert!(MINIMAL_DESCRIPTORS[33].index == 33);
    assert!(MINIMAL_DESCRIPTORS[34].index == 34);
    assert!(MINIMAL_DESCRIPTORS[35].index == 35);
    assert!(MINIMAL_DESCRIPTORS[36].index == 36);
    assert!(MINIMAL_DESCRIPTORS[37].index == 37);
    assert!(MINIMAL_DESCRIPTORS[38].index == 38);
    assert!(MINIMAL_DESCRIPTORS[39].index == 39);
    assert!(MINIMAL_DESCRIPTORS[40].index == 40);
    assert!(MINIMAL_DESCRIPTORS[41].index == 41);

    // FULL: every index == position
    assert!(FULL_DESCRIPTORS[0].index == 0);
    assert!(FULL_DESCRIPTORS[1].index == 1);
    assert!(FULL_DESCRIPTORS[2].index == 2);
    assert!(FULL_DESCRIPTORS[3].index == 3);
    assert!(FULL_DESCRIPTORS[4].index == 4);
    assert!(FULL_DESCRIPTORS[5].index == 5);
    assert!(FULL_DESCRIPTORS[6].index == 6);
    assert!(FULL_DESCRIPTORS[7].index == 7);
    assert!(FULL_DESCRIPTORS[8].index == 8);
    assert!(FULL_DESCRIPTORS[9].index == 9);
    assert!(FULL_DESCRIPTORS[10].index == 10);
    assert!(FULL_DESCRIPTORS[11].index == 11);
    assert!(FULL_DESCRIPTORS[12].index == 12);
    assert!(FULL_DESCRIPTORS[13].index == 13);
    assert!(FULL_DESCRIPTORS[14].index == 14);
    assert!(FULL_DESCRIPTORS[15].index == 15);
    assert!(FULL_DESCRIPTORS[16].index == 16);
    assert!(FULL_DESCRIPTORS[17].index == 17);
    assert!(FULL_DESCRIPTORS[18].index == 18);
    assert!(FULL_DESCRIPTORS[19].index == 19);
    assert!(FULL_DESCRIPTORS[20].index == 20);
    assert!(FULL_DESCRIPTORS[21].index == 21);
    assert!(FULL_DESCRIPTORS[22].index == 22);
    assert!(FULL_DESCRIPTORS[23].index == 23);
    assert!(FULL_DESCRIPTORS[24].index == 24);
    assert!(FULL_DESCRIPTORS[25].index == 25);
    assert!(FULL_DESCRIPTORS[26].index == 26);
    assert!(FULL_DESCRIPTORS[27].index == 27);
    assert!(FULL_DESCRIPTORS[28].index == 28);
    assert!(FULL_DESCRIPTORS[29].index == 29);
    assert!(FULL_DESCRIPTORS[30].index == 30);
    assert!(FULL_DESCRIPTORS[31].index == 31);
    assert!(FULL_DESCRIPTORS[32].index == 32);
    assert!(FULL_DESCRIPTORS[33].index == 33);
    assert!(FULL_DESCRIPTORS[34].index == 34);
    assert!(FULL_DESCRIPTORS[35].index == 35);
    assert!(FULL_DESCRIPTORS[36].index == 36);
    assert!(FULL_DESCRIPTORS[37].index == 37);
    assert!(FULL_DESCRIPTORS[38].index == 38);
    assert!(FULL_DESCRIPTORS[39].index == 39);
    assert!(FULL_DESCRIPTORS[40].index == 40);
    assert!(FULL_DESCRIPTORS[41].index == 41);
    assert!(FULL_DESCRIPTORS[42].index == 42);
    assert!(FULL_DESCRIPTORS[43].index == 43);
    assert!(FULL_DESCRIPTORS[44].index == 44);
    assert!(FULL_DESCRIPTORS[45].index == 45);
    assert!(FULL_DESCRIPTORS[46].index == 46);
    assert!(FULL_DESCRIPTORS[47].index == 47);
    assert!(FULL_DESCRIPTORS[48].index == 48);
    assert!(FULL_DESCRIPTORS[49].index == 49);
    assert!(FULL_DESCRIPTORS[50].index == 50);
    assert!(FULL_DESCRIPTORS[51].index == 51);
    assert!(FULL_DESCRIPTORS[52].index == 52);
    assert!(FULL_DESCRIPTORS[53].index == 53);
    assert!(FULL_DESCRIPTORS[54].index == 54);

    // Prefix consistency: first 42 of FULL match MINIMAL indices
    assert!(FULL_DESCRIPTORS[0].index == MINIMAL_DESCRIPTORS[0].index);
    assert!(FULL_DESCRIPTORS[41].index == MINIMAL_DESCRIPTORS[41].index);

    // Array length checks
    assert!(MINIMAL_DESCRIPTORS.len() == N_MARKET_FEATURES);
    assert!(FULL_DESCRIPTORS.len() == N_FULL_FEATURES);
    assert!(MARKET_FEATURE_NAMES.len() == N_MARKET_FEATURES);
    assert!(FULL_FEATURE_NAMES.len() == N_FULL_FEATURES);
    assert!(FULL_FEATURE_NEUTRALS.len() == N_FULL_FEATURES);
    assert!(LOOKBACKS.len() == N_LOOKBACKS);

    // Legacy constant consistency
    assert!(idx::NEWS_TICKS_REMAINING == N_MARKET_FEATURES - 1);
    assert!(extended_idx::SENTIMENT_PRICE_GAP == N_FULL_FEATURES - 1);

    // CANONICAL: every index == position
    assert!(CANONICAL_DESCRIPTORS[0].index == 0);
    assert!(CANONICAL_DESCRIPTORS[1].index == 1);
    assert!(CANONICAL_DESCRIPTORS[2].index == 2);
    assert!(CANONICAL_DESCRIPTORS[3].index == 3);
    assert!(CANONICAL_DESCRIPTORS[4].index == 4);
    assert!(CANONICAL_DESCRIPTORS[5].index == 5);
    assert!(CANONICAL_DESCRIPTORS[6].index == 6);
    assert!(CANONICAL_DESCRIPTORS[7].index == 7);
    assert!(CANONICAL_DESCRIPTORS[8].index == 8);
    assert!(CANONICAL_DESCRIPTORS[9].index == 9);
    assert!(CANONICAL_DESCRIPTORS[10].index == 10);
    assert!(CANONICAL_DESCRIPTORS[11].index == 11);
    assert!(CANONICAL_DESCRIPTORS[12].index == 12);
    assert!(CANONICAL_DESCRIPTORS[13].index == 13);
    assert!(CANONICAL_DESCRIPTORS[14].index == 14);
    assert!(CANONICAL_DESCRIPTORS[15].index == 15);
    assert!(CANONICAL_DESCRIPTORS[16].index == 16);
    assert!(CANONICAL_DESCRIPTORS[17].index == 17);
    assert!(CANONICAL_DESCRIPTORS[18].index == 18);
    assert!(CANONICAL_DESCRIPTORS[19].index == 19);
    assert!(CANONICAL_DESCRIPTORS[20].index == 20);
    assert!(CANONICAL_DESCRIPTORS[21].index == 21);
    assert!(CANONICAL_DESCRIPTORS[22].index == 22);
    assert!(CANONICAL_DESCRIPTORS[23].index == 23);
    assert!(CANONICAL_DESCRIPTORS[24].index == 24);
    assert!(CANONICAL_DESCRIPTORS[25].index == 25);
    assert!(CANONICAL_DESCRIPTORS[26].index == 26);
    assert!(CANONICAL_DESCRIPTORS[27].index == 27);

    // Canonical array length checks
    assert!(CANONICAL_DESCRIPTORS.len() == N_CANONICAL_FEATURES);
    assert!(CANONICAL_FEATURE_NAMES.len() == N_CANONICAL_FEATURES);
    assert!(CANONICAL_FEATURE_NEUTRALS.len() == N_CANONICAL_FEATURES);
    assert!(canonical_idx::RSI_DIVERGENCE == N_CANONICAL_FEATURES - 1);
};

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Price;

    #[test]
    fn test_price_change_pct() {
        // 10% increase
        assert!((price_change_pct(110.0, 100.0) - 10.0).abs() < 1e-10);
        // 10% decrease
        assert!((price_change_pct(90.0, 100.0) - (-10.0)).abs() < 1e-10);
        // Zero past price returns NaN
        assert!(price_change_pct(100.0, 0.0).is_nan());
        // Negative past price returns NaN
        assert!(price_change_pct(100.0, -100.0).is_nan());
    }

    #[test]
    fn test_log_return() {
        // ln(1.1) ≈ 0.0953
        assert!((log_return(110.0, 100.0) - 0.1_f64.ln_1p()).abs() < 1e-10);
        // ln(0.9) ≈ -0.1054
        assert!((log_return(90.0, 100.0) - (-0.1_f64).ln_1p()).abs() < 1e-10);
        // Invalid inputs return NaN
        assert!(log_return(0.0, 100.0).is_nan());
        assert!(log_return(100.0, 0.0).is_nan());
        assert!(log_return(-100.0, 100.0).is_nan());
    }

    #[test]
    fn test_bollinger_percent_b() {
        // Price at lower band = 0.0
        assert!((bollinger_percent_b(100.0, 120.0, 100.0) - 0.0).abs() < 1e-10);
        // Price at upper band = 1.0
        assert!((bollinger_percent_b(120.0, 120.0, 100.0) - 1.0).abs() < 1e-10);
        // Price at middle = 0.5
        assert!((bollinger_percent_b(110.0, 120.0, 100.0) - 0.5).abs() < 1e-10);
        // Converged bands return 0.5 (price at center)
        assert!((bollinger_percent_b(100.0, 100.0, 100.0) - 0.5).abs() < 1e-10);
        // NaN inputs return NaN
        assert!(bollinger_percent_b(f64::NAN, 120.0, 100.0).is_nan());
    }

    #[test]
    fn test_price_change_from_candles() {
        use crate::Quantity;

        let candles: Vec<Candle> = (0..10)
            .map(|i| Candle {
                symbol: "TEST".to_string(),
                timestamp: i as u64,
                tick: i as u64,
                open: Price::from_float(100.0 + i as f64),
                high: Price::from_float(101.0 + i as f64),
                low: Price::from_float(99.0 + i as f64),
                close: Price::from_float(100.0 + i as f64),
                volume: Quantity(1000),
            })
            .collect();

        // Latest close = 109, lookback 1 = 108 => (109-108)/108 * 100
        let expected = (109.0 - 108.0) / 108.0 * 100.0;
        assert!((price_change_from_candles(&candles, 1) - expected).abs() < 1e-10);

        // Insufficient history
        assert!(price_change_from_candles(&candles, 100).is_nan());
    }

    #[test]
    fn test_feature_count_consistency() {
        assert_eq!(MARKET_FEATURE_NAMES.len(), N_MARKET_FEATURES);
        assert_eq!(FULL_FEATURE_NAMES.len(), N_FULL_FEATURES);
        assert_eq!(LOOKBACKS.len(), N_LOOKBACKS);
    }

    #[test]
    fn test_feature_name_prefix() {
        for name in MARKET_FEATURE_NAMES {
            assert!(
                name.starts_with("f_"),
                "Feature name '{}' should start with 'f_'",
                name
            );
        }
        for name in FULL_FEATURE_NAMES {
            assert!(
                name.starts_with("f_"),
                "Feature name '{}' should start with 'f_'",
                name
            );
        }
    }

    #[test]
    fn test_required_indicators() {
        let indicators = required_indicators();
        assert_eq!(indicators.len(), 12);

        // Check we have all expected types
        assert!(indicators.contains(&IndicatorType::Sma(8)));
        assert!(indicators.contains(&IndicatorType::Rsi(8)));
        assert!(indicators.contains(&IndicatorType::MACD_LINE_STANDARD));
        assert!(indicators.contains(&IndicatorType::BOLLINGER_UPPER_STANDARD));
    }

    #[test]
    fn test_realized_volatility() {
        use crate::Quantity;

        // Create candles with known log returns
        let candles: Vec<Candle> = (0..10)
            .map(|i| Candle {
                symbol: "TEST".to_string(),
                timestamp: i as u64,
                tick: i as u64,
                open: Price::from_float(100.0),
                high: Price::from_float(100.0),
                low: Price::from_float(100.0),
                close: Price::from_float(100.0 + i as f64),
                volume: Quantity(1000),
            })
            .collect();

        // With 10 candles, lookback 8 should work
        let vol = realized_volatility(&candles, 8);
        assert!(vol.is_finite() && vol >= 0.0);

        // Insufficient history
        assert!(realized_volatility(&candles, 100).is_nan());

        // All same price → zero vol
        let flat_candles: Vec<Candle> = (0..10)
            .map(|i| Candle {
                symbol: "TEST".to_string(),
                timestamp: i as u64,
                tick: i as u64,
                open: Price::from_float(100.0),
                high: Price::from_float(100.0),
                low: Price::from_float(100.0),
                close: Price::from_float(100.0),
                volume: Quantity(1000),
            })
            .collect();
        let flat_vol = realized_volatility(&flat_candles, 8);
        assert!((flat_vol - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_spread_bps() {
        // Normal spread: (101 - 99) / 100 * 10000 = 200 bps
        assert!((spread_bps(99.0, 101.0, 100.0) - 200.0).abs() < 1e-10);
        // Zero mid price
        assert!(spread_bps(99.0, 101.0, 0.0).is_nan());
        // NaN input
        assert!(spread_bps(f64::NAN, 101.0, 100.0).is_nan());
    }

    #[test]
    fn test_registry_len() {
        assert_eq!(MINIMAL_REGISTRY.len(), 42);
        assert_eq!(FULL_REGISTRY.len(), 55);
    }

    #[test]
    fn test_registry_names_match_descriptors() {
        for (i, desc) in MINIMAL_REGISTRY.descriptors().iter().enumerate() {
            assert_eq!(
                desc.name,
                MINIMAL_REGISTRY.names()[i],
                "MINIMAL name mismatch at index {i}"
            );
        }
        for (i, desc) in FULL_REGISTRY.descriptors().iter().enumerate() {
            assert_eq!(
                desc.name,
                FULL_REGISTRY.names()[i],
                "FULL name mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_registry_neutrals_match_descriptors() {
        for (i, desc) in MINIMAL_REGISTRY.descriptors().iter().enumerate() {
            assert_eq!(
                desc.neutral,
                MINIMAL_REGISTRY.neutrals()[i],
                "MINIMAL neutral mismatch at index {i}"
            );
        }
        for (i, desc) in FULL_REGISTRY.descriptors().iter().enumerate() {
            assert!(
                (desc.neutral - FULL_REGISTRY.neutrals()[i]).abs() < 1e-15,
                "FULL neutral mismatch at index {i}: descriptor={}, array={}",
                desc.neutral,
                FULL_REGISTRY.neutrals()[i]
            );
        }
    }

    #[test]
    fn test_registry_group_indices() {
        let price_indices = FULL_REGISTRY.group_indices(FeatureGroup::Price);
        assert_eq!(price_indices.len(), 25); // 0-24

        let microstructure_indices = FULL_REGISTRY.group_indices(FeatureGroup::Microstructure);
        assert_eq!(microstructure_indices, vec![42, 43, 44]);

        let volatility_indices = FULL_REGISTRY.group_indices(FeatureGroup::Volatility);
        assert_eq!(volatility_indices, vec![45, 46, 47]);

        let fundamental_indices = FULL_REGISTRY.group_indices(FeatureGroup::Fundamental);
        assert_eq!(fundamental_indices, vec![48, 49]);

        let momentum_indices = FULL_REGISTRY.group_indices(FeatureGroup::MomentumQuality);
        assert_eq!(momentum_indices, vec![50, 51]);

        let volume_indices = FULL_REGISTRY.group_indices(FeatureGroup::VolumeCross);
        assert_eq!(volume_indices, vec![52, 53, 54]);
    }

    #[test]
    fn test_registry_validate() {
        // All NaN → no out-of-range (NaN is not out of range, it's missing)
        let features = vec![f64::NAN; N_FULL_FEATURES];
        assert!(FULL_REGISTRY.validate(&features).is_empty());

        // In-range values → no violations
        let mut features = vec![0.0; N_FULL_FEATURES];
        features[extended_idx::SPREAD_BPS] = 100.0; // valid: [0, 1000]
        assert!(FULL_REGISTRY.validate(&features).is_empty());

        // Out-of-range value
        features[extended_idx::SPREAD_BPS] = 5000.0; // out of [0, 1000]
        let violations = FULL_REGISTRY.validate(&features);
        assert!(violations.contains(&extended_idx::SPREAD_BPS));
    }

    #[test]
    fn test_full_feature_names_prefix_consistency() {
        // First 42 names of FULL must equal MARKET_FEATURE_NAMES
        for i in 0..N_MARKET_FEATURES {
            assert_eq!(
                FULL_FEATURE_NAMES[i], MARKET_FEATURE_NAMES[i],
                "Name mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_feature_group_all() {
        assert_eq!(FeatureGroup::ALL.len(), 8);
        assert_eq!(FeatureGroup::MINIMAL.len(), 3);
        assert_eq!(FeatureGroup::CANONICAL.len(), 5);
    }

    // =========================================================================
    // V6.3 Canonical Registry Tests
    // =========================================================================

    #[test]
    fn test_canonical_registry_len() {
        assert_eq!(CANONICAL_REGISTRY.len(), 28);
    }

    #[test]
    fn test_canonical_names_match_descriptors() {
        for (i, desc) in CANONICAL_REGISTRY.descriptors().iter().enumerate() {
            assert_eq!(
                desc.name,
                CANONICAL_REGISTRY.names()[i],
                "CANONICAL name mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_canonical_neutrals_match_descriptors() {
        for (i, desc) in CANONICAL_REGISTRY.descriptors().iter().enumerate() {
            assert!(
                (desc.neutral - CANONICAL_REGISTRY.neutrals()[i]).abs() < 1e-15,
                "CANONICAL neutral mismatch at index {i}: descriptor={}, array={}",
                desc.neutral,
                CANONICAL_REGISTRY.neutrals()[i]
            );
        }
    }

    #[test]
    fn test_canonical_group_indices() {
        let price = CANONICAL_REGISTRY.group_indices(FeatureGroup::Price);
        assert_eq!(price.len(), 8); // 0-7

        let technical = CANONICAL_REGISTRY.group_indices(FeatureGroup::TechnicalIndicator);
        assert_eq!(technical.len(), 13); // 8-20

        let volatility = CANONICAL_REGISTRY.group_indices(FeatureGroup::Volatility);
        assert_eq!(volatility, vec![21, 22, 23]);

        let fundamental = CANONICAL_REGISTRY.group_indices(FeatureGroup::Fundamental);
        assert_eq!(fundamental, vec![24, 25]);

        let momentum = CANONICAL_REGISTRY.group_indices(FeatureGroup::MomentumQuality);
        assert_eq!(momentum, vec![26, 27]);

        // Dropped groups should have no canonical features
        let news = CANONICAL_REGISTRY.group_indices(FeatureGroup::News);
        assert!(news.is_empty());
        let micro = CANONICAL_REGISTRY.group_indices(FeatureGroup::Microstructure);
        assert!(micro.is_empty());
        let volcross = CANONICAL_REGISTRY.group_indices(FeatureGroup::VolumeCross);
        assert!(volcross.is_empty());
    }

    #[test]
    fn test_canonical_feature_name_prefix() {
        for name in CANONICAL_FEATURE_NAMES {
            assert!(
                name.starts_with("f_"),
                "Canonical feature name '{}' should start with 'f_'",
                name
            );
        }
    }

    #[test]
    fn test_canonical_names_exist_in_full() {
        // Every canonical feature name must exist in the full 55-feature set
        for name in CANONICAL_FEATURE_NAMES {
            assert!(
                FULL_FEATURE_NAMES.contains(name),
                "Canonical feature '{}' not found in FULL_FEATURE_NAMES",
                name
            );
        }
    }
}
