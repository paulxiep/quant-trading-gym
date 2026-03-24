//! Technical indicator types for the quant module.
//!
//! This module defines the types used for technical analysis indicators
//! including moving averages, RSI, MACD, Bollinger Bands, and ATR.

use crate::ids::{Symbol, Tick};
use serde::{Deserialize, Serialize};

// =============================================================================
// Indicator Type Enum
// =============================================================================

/// Type of technical indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndicatorType {
    /// Simple Moving Average with period.
    Sma(usize),
    /// Exponential Moving Average with period.
    Ema(usize),
    /// Relative Strength Index with period.
    Rsi(usize),
    /// Average True Range with period.
    Atr(usize),

    // =========================================================================
    // MACD Components (V5.5)
    // =========================================================================
    /// MACD Line (fast EMA - slow EMA).
    MacdLine {
        fast: usize,
        slow: usize,
        signal: usize,
    },
    /// MACD Signal Line (EMA of MACD line).
    MacdSignal {
        fast: usize,
        slow: usize,
        signal: usize,
    },
    /// MACD Histogram (MACD Line - Signal Line).
    MacdHistogram {
        fast: usize,
        slow: usize,
        signal: usize,
    },

    // =========================================================================
    // Bollinger Bands Components (V5.5)
    // =========================================================================
    /// Bollinger Upper Band.
    BollingerUpper {
        period: usize,
        /// Standard deviation multiplier * 100 (e.g., 200 = 2.0 std devs).
        std_dev_bp: u32,
    },
    /// Bollinger Middle Band (SMA).
    BollingerMiddle {
        period: usize,
        /// Standard deviation multiplier * 100 (e.g., 200 = 2.0 std devs).
        std_dev_bp: u32,
    },
    /// Bollinger Lower Band.
    BollingerLower {
        period: usize,
        /// Standard deviation multiplier * 100 (e.g., 200 = 2.0 std devs).
        std_dev_bp: u32,
    },

    // =========================================================================
    // Legacy variants (deprecated, use components instead)
    // =========================================================================
    /// MACD with fast, slow, and signal periods.
    /// Deprecated: Use MacdLine, MacdSignal, MacdHistogram instead.
    #[deprecated(note = "Use MacdLine, MacdSignal, MacdHistogram for component access")]
    Macd {
        fast: usize,
        slow: usize,
        signal: usize,
    },
    /// Bollinger Bands with period and standard deviation multiplier.
    /// Deprecated: Use BollingerUpper, BollingerMiddle, BollingerLower instead.
    #[deprecated(note = "Use BollingerUpper, BollingerMiddle, BollingerLower for component access")]
    BollingerBands {
        period: usize,
        /// Standard deviation multiplier * 100 (e.g., 200 = 2.0 std devs).
        std_dev_bp: u32,
    },
}

impl IndicatorType {
    /// Standard MACD Line configuration (8, 16, 4) - optimized for batch auction.
    pub const MACD_LINE_STANDARD: Self = Self::MacdLine {
        fast: 8,
        slow: 16,
        signal: 4,
    };

    /// Standard MACD Signal configuration (8, 16, 4) - optimized for batch auction.
    pub const MACD_SIGNAL_STANDARD: Self = Self::MacdSignal {
        fast: 8,
        slow: 16,
        signal: 4,
    };

    /// Standard MACD Histogram configuration (8, 16, 4) - optimized for batch auction.
    pub const MACD_HISTOGRAM_STANDARD: Self = Self::MacdHistogram {
        fast: 8,
        slow: 16,
        signal: 4,
    };

    /// Standard Bollinger Upper Band (12 period, 2 std devs) - optimized for batch auction.
    pub const BOLLINGER_UPPER_STANDARD: Self = Self::BollingerUpper {
        period: 12,
        std_dev_bp: 200,
    };

    /// Standard Bollinger Middle Band (12 period, 2 std devs) - optimized for batch auction.
    pub const BOLLINGER_MIDDLE_STANDARD: Self = Self::BollingerMiddle {
        period: 12,
        std_dev_bp: 200,
    };

    /// Standard Bollinger Lower Band (12 period, 2 std devs) - optimized for batch auction.
    pub const BOLLINGER_LOWER_STANDARD: Self = Self::BollingerLower {
        period: 12,
        std_dev_bp: 200,
    };

    /// Standard MACD configuration (8, 16, 4) - deprecated, use MACD_LINE_STANDARD etc.
    #[allow(deprecated)]
    #[deprecated(note = "Use MACD_LINE_STANDARD, MACD_SIGNAL_STANDARD, MACD_HISTOGRAM_STANDARD")]
    pub const MACD_STANDARD: Self = Self::Macd {
        fast: 8,
        slow: 16,
        signal: 4,
    };

    /// Standard Bollinger Bands (12 period, 2 std devs) - deprecated.
    #[allow(deprecated)]
    #[deprecated(
        note = "Use BOLLINGER_UPPER_STANDARD, BOLLINGER_MIDDLE_STANDARD, BOLLINGER_LOWER_STANDARD"
    )]
    pub const BOLLINGER_STANDARD: Self = Self::BollingerBands {
        period: 12,
        std_dev_bp: 200,
    };

    /// Get the number of periods required for this indicator to produce valid output.
    #[allow(deprecated)]
    pub fn required_periods(&self) -> usize {
        match self {
            Self::Sma(p) | Self::Ema(p) | Self::Rsi(p) | Self::Atr(p) => *p,
            Self::MacdLine { slow, signal, .. }
            | Self::MacdSignal { slow, signal, .. }
            | Self::MacdHistogram { slow, signal, .. }
            | Self::Macd { slow, signal, .. } => slow + signal,
            Self::BollingerUpper { period, .. }
            | Self::BollingerMiddle { period, .. }
            | Self::BollingerLower { period, .. }
            | Self::BollingerBands { period, .. } => *period,
        }
    }

    /// Convert to a canonical string key for serialization.
    #[allow(deprecated)]
    pub fn to_key(&self) -> String {
        match self {
            Self::Sma(p) => format!("SMA_{p}"),
            Self::Ema(p) => format!("EMA_{p}"),
            Self::Rsi(p) => format!("RSI_{p}"),
            Self::Atr(p) => format!("ATR_{p}"),
            Self::MacdLine { .. } => "MACD_line".to_string(),
            Self::MacdSignal { .. } => "MACD_signal".to_string(),
            Self::MacdHistogram { .. } => "MACD_histogram".to_string(),
            Self::BollingerUpper { .. } => "BB_upper".to_string(),
            Self::BollingerMiddle { .. } => "BB_middle".to_string(),
            Self::BollingerLower { .. } => "BB_lower".to_string(),
            Self::Macd { .. } => "MACD".to_string(),
            Self::BollingerBands { .. } => "BB".to_string(),
        }
    }
}

// =============================================================================
// Indicator Value
// =============================================================================

/// Computed indicator value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndicatorValue {
    /// Type of indicator.
    pub indicator_type: IndicatorType,
    /// Stock symbol.
    pub symbol: Symbol,
    /// Computed value (f64 for statistical precision).
    pub value: f64,
    /// Tick when computed.
    pub tick: Tick,
}

// =============================================================================
// MACD Output
// =============================================================================

/// MACD output values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct MacdOutput {
    /// MACD line (fast EMA - slow EMA).
    pub macd_line: f64,
    /// Signal line (EMA of MACD line).
    pub signal_line: f64,
    /// Histogram (MACD - Signal).
    pub histogram: f64,
}

// =============================================================================
// Bollinger Bands Output
// =============================================================================

/// Bollinger Bands output values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct BollingerOutput {
    /// Upper band.
    pub upper: f64,
    /// Middle band (SMA).
    pub middle: f64,
    /// Lower band.
    pub lower: f64,
    /// Band width as percentage of middle.
    pub bandwidth: f64,
    /// %B: where price is relative to bands (0 = lower, 1 = upper).
    pub percent_b: f64,
}
