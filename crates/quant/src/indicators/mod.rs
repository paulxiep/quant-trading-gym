//! Technical indicators for market analysis.
//!
//! This module provides a trait-based interface for computing technical indicators
//! on price candle data. All indicators are designed to work with [`Candle`] slices
//! and produce f64 values suitable for statistical analysis.
//!
//! # Supported Indicators
//! - **SMA** - Simple Moving Average
//! - **EMA** - Exponential Moving Average
//! - **RSI** - Relative Strength Index
//! - **MACD** - Moving Average Convergence Divergence
//! - **Bollinger Bands** - Volatility bands around SMA
//! - **ATR** - Average True Range
//!
//! # Example
//! ```
//! use quant::indicators::{Indicator, Sma};
//! use types::{Candle, Price, Quantity};
//!
//! let candles: Vec<Candle> = vec![/* ... */];
//! let sma = Sma::new(20);
//! if let Some(value) = sma.calculate(&candles) {
//!     println!("SMA(20) = {:.2}", value);
//! }
//! ```

use types::{Candle, IndicatorType};

// =============================================================================
// Indicator Modules
// =============================================================================

mod atr;
mod bollinger;
mod ema;
mod macd;
mod rsi;
mod sma;

// =============================================================================
// Re-exports
// =============================================================================

pub use atr::Atr;
pub use bollinger::BollingerBands;
pub use ema::Ema;
pub use macd::Macd;
pub use rsi::Rsi;
pub use sma::Sma;

// =============================================================================
// Indicator Trait
// =============================================================================

/// Trait for technical indicators.
///
/// Indicators consume candle data and produce a single f64 value.
/// They declare their type (for caching) and minimum required data periods.
pub trait Indicator: Send + Sync {
    /// The type of this indicator (for caching and identification).
    fn indicator_type(&self) -> IndicatorType;

    /// Calculate the indicator value from candle data.
    ///
    /// Returns `None` if there's insufficient data.
    /// Candles are expected to be ordered from oldest to newest.
    fn calculate(&self, candles: &[Candle]) -> Option<f64>;

    /// Minimum number of candles required for a valid calculation.
    fn required_periods(&self) -> usize;
}

// =============================================================================
// Factory Function
// =============================================================================

/// Create an indicator from its type specification.
#[allow(deprecated)]
pub fn create_indicator(indicator_type: IndicatorType) -> Box<dyn Indicator> {
    match indicator_type {
        IndicatorType::Sma(p) => Box::new(Sma::new(p)),
        IndicatorType::Ema(p) => Box::new(Ema::new(p)),
        IndicatorType::Rsi(p) => Box::new(Rsi::new(p)),
        IndicatorType::Atr(p) => Box::new(Atr::new(p)),
        // MACD components all use same underlying computation
        IndicatorType::MacdLine { fast, slow, signal }
        | IndicatorType::MacdSignal { fast, slow, signal }
        | IndicatorType::MacdHistogram { fast, slow, signal }
        | IndicatorType::Macd { fast, slow, signal } => Box::new(Macd::new(fast, slow, signal)),
        // Bollinger components all use same underlying computation
        IndicatorType::BollingerUpper { period, std_dev_bp }
        | IndicatorType::BollingerMiddle { period, std_dev_bp }
        | IndicatorType::BollingerLower { period, std_dev_bp }
        | IndicatorType::BollingerBands { period, std_dev_bp } => {
            Box::new(BollingerBands::new(period, std_dev_bp as f64 / 100.0))
        }
    }
}

/// Compute all standard indicators for given candles, returning component-level values.
///
/// This computes all common indicators and returns a map with separate entries for
/// each MACD and Bollinger component. This is the single computation point -
/// use this instead of calling compute_all() on registered indicators.
pub fn compute_all_indicators(candles: &[Candle]) -> HashMap<IndicatorType, f64> {
    let mut result = HashMap::new();

    if candles.is_empty() {
        return result;
    }

    // Simple indicators
    if let Some(v) = Sma::new(8).calculate(candles) {
        result.insert(IndicatorType::Sma(8), v);
    }
    if let Some(v) = Sma::new(16).calculate(candles) {
        result.insert(IndicatorType::Sma(16), v);
    }
    if let Some(v) = Ema::new(8).calculate(candles) {
        result.insert(IndicatorType::Ema(8), v);
    }
    if let Some(v) = Ema::new(16).calculate(candles) {
        result.insert(IndicatorType::Ema(16), v);
    }
    if let Some(v) = Rsi::new(8).calculate(candles) {
        result.insert(IndicatorType::Rsi(8), v);
    }
    if let Some(v) = Atr::new(8).calculate(candles) {
        result.insert(IndicatorType::Atr(8), v);
    }

    // MACD components (8, 16, 4) - compute once, store all components
    let macd = Macd::new(8, 16, 4);
    if let Some(output) = macd.calculate_full(candles) {
        result.insert(
            IndicatorType::MacdLine {
                fast: 8,
                slow: 16,
                signal: 4,
            },
            output.macd_line,
        );
        result.insert(
            IndicatorType::MacdSignal {
                fast: 8,
                slow: 16,
                signal: 4,
            },
            output.signal_line,
        );
        result.insert(
            IndicatorType::MacdHistogram {
                fast: 8,
                slow: 16,
                signal: 4,
            },
            output.histogram,
        );
    }

    // Bollinger Bands components (12, 2.0) - compute once, store all components
    let bb = BollingerBands::new(12, 2.0);
    if let Some(output) = bb.calculate_full(candles) {
        result.insert(
            IndicatorType::BollingerUpper {
                period: 12,
                std_dev_bp: 200,
            },
            output.upper,
        );
        result.insert(
            IndicatorType::BollingerMiddle {
                period: 12,
                std_dev_bp: 200,
            },
            output.middle,
        );
        result.insert(
            IndicatorType::BollingerLower {
                period: 12,
                std_dev_bp: 200,
            },
            output.lower,
        );
    }

    result
}

use std::collections::HashMap;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use types::{Price, Quantity};

    /// Helper to create test candles with given close prices.
    fn make_candles(closes: &[f64]) -> Vec<Candle> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &close)| Candle {
                symbol: "TEST".to_string(),
                open: Price::from_float(close),
                high: Price::from_float(close + 1.0),
                low: Price::from_float(close - 1.0),
                close: Price::from_float(close),
                volume: Quantity(1000),
                timestamp: i as u64,
                tick: i as u64,
            })
            .collect()
    }

    #[test]
    fn test_sma_calculation() {
        let candles = make_candles(&[10.0, 11.0, 12.0, 13.0, 14.0]);
        let sma = Sma::new(3);

        // SMA(3) of last 3 values: (12 + 13 + 14) / 3 = 13
        let result = sma.calculate(&candles);
        assert!((result.unwrap() - 13.0).abs() < 0.001);
    }

    #[test]
    fn test_sma_insufficient_data() {
        let candles = make_candles(&[10.0, 11.0]);
        let sma = Sma::new(5);
        assert!(sma.calculate(&candles).is_none());
    }

    #[test]
    fn test_ema_calculation() {
        let candles = make_candles(&[
            22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
        ]);
        let ema = Ema::new(10);

        let result = ema.calculate(&candles);
        // EMA(10) with these values should be around 22.22
        assert!(result.is_some());
        assert!((result.unwrap() - 22.221).abs() < 0.01);
    }

    #[test]
    fn test_rsi_calculation() {
        // Test data that should produce RSI around 70
        let prices: Vec<f64> = (0..20)
            .map(|i| 44.0 + i as f64 * 0.2 + (i % 3) as f64 * 0.1)
            .collect();
        let candles = make_candles(&prices);
        let rsi = Rsi::new(14);

        let result = rsi.calculate(&candles);
        assert!(result.is_some());
        // RSI should be positive and <= 100
        let rsi_val = result.unwrap();
        assert!(rsi_val >= 0.0 && rsi_val <= 100.0);
    }

    #[test]
    fn test_rsi_boundaries() {
        // Test with only gains (RSI should be 100)
        let increasing = make_candles(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let rsi = Rsi::new(14);
        let result = rsi.calculate(&increasing);
        assert!(result.is_some());
        assert!((result.unwrap() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_macd_standard() {
        // Need at least 26 + 9 = 35 candles
        let prices: Vec<f64> = (0..40)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0)
            .collect();
        let candles = make_candles(&prices);
        let macd = Macd::standard();

        let result = macd.calculate_full(&candles);
        assert!(result.is_some());

        let output = result.unwrap();
        // MACD line should be the difference of fast and slow EMAs
        // Histogram should be MACD - Signal
        assert!((output.histogram - (output.macd_line - output.signal_line)).abs() < 0.0001);
    }

    #[test]
    fn test_bollinger_bands() {
        let candles = make_candles(&[
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
            45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
        ]);
        let bb = BollingerBands::standard();

        let result = bb.calculate_full(&candles);
        assert!(result.is_some());

        let output = result.unwrap();
        // Upper band should be > middle > lower
        assert!(output.upper > output.middle);
        assert!(output.middle > output.lower);
        // %B should be between 0 and 1 for price within bands
        assert!(output.percent_b >= -0.5 && output.percent_b <= 1.5);
    }

    #[test]
    fn test_atr_calculation() {
        // Create candles with varying ranges
        let candles: Vec<Candle> = (0..20)
            .map(|i| {
                let base = 100.0 + i as f64;
                Candle {
                    symbol: "TEST".to_string(),
                    open: Price::from_float(base),
                    high: Price::from_float(base + 2.0),
                    low: Price::from_float(base - 1.0),
                    close: Price::from_float(base + 0.5),
                    volume: Quantity(1000),
                    timestamp: i as u64,
                    tick: i as u64,
                }
            })
            .collect();

        let atr = Atr::new(14);
        let result = atr.calculate(&candles);
        assert!(result.is_some());

        // ATR should be positive
        let atr_val = result.unwrap();
        assert!(atr_val > 0.0);
    }

    #[test]
    fn test_indicator_factory() {
        let sma = create_indicator(IndicatorType::Sma(20));
        assert_eq!(sma.required_periods(), 20);

        // V5.3: MACD_STANDARD is (8, 16, 4) → required = 16 + 4 = 20
        let macd = create_indicator(IndicatorType::MACD_LINE_STANDARD);
        assert_eq!(macd.required_periods(), 20);

        // V5.3: BOLLINGER_STANDARD is period=12
        let bb = create_indicator(IndicatorType::BOLLINGER_MIDDLE_STANDARD);
        assert_eq!(bb.required_periods(), 12);
    }
}
