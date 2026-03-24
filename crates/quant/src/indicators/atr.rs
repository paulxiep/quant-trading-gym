//! Average True Range (ATR) indicator.

use super::Indicator;
use types::{Candle, IndicatorType};

/// Average True Range indicator.
///
/// Measures market volatility by decomposing the entire range of price movement.
/// True Range is max of: High-Low, |High-PrevClose|, |Low-PrevClose|
#[derive(Debug, Clone)]
pub struct Atr {
    period: usize,
}

impl Atr {
    /// Create a new ATR indicator with the given period.
    ///
    /// # Panics
    /// Panics if period is 0.
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "ATR period must be > 0");
        Self { period }
    }

    /// Calculate True Range for a candle given the previous close.
    fn true_range(candle: &Candle, prev_close: f64) -> f64 {
        let high = candle.high.to_float();
        let low = candle.low.to_float();

        let hl = high - low;
        let hpc = (high - prev_close).abs();
        let lpc = (low - prev_close).abs();

        hl.max(hpc).max(lpc)
    }
}

impl Indicator for Atr {
    fn indicator_type(&self) -> IndicatorType {
        IndicatorType::Atr(self.period)
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        // Need period + 1 candles for period true ranges
        if candles.len() < self.period + 1 {
            return None;
        }

        // Calculate true ranges using iterator over window pairs
        let true_ranges: Vec<f64> = candles
            .windows(2)
            .map(|w| {
                let prev_close = w[0].close.to_float();
                Self::true_range(&w[1], prev_close)
            })
            .collect();

        // Calculate initial ATR as simple average
        let initial_atr: f64 =
            true_ranges.iter().take(self.period).sum::<f64>() / self.period as f64;

        // Apply Wilder's smoothing (same as RSI)
        let atr = true_ranges
            .iter()
            .skip(self.period)
            .fold(initial_atr, |prev_atr, &tr| {
                (prev_atr * (self.period as f64 - 1.0) + tr) / self.period as f64
            });

        Some(atr)
    }

    fn required_periods(&self) -> usize {
        self.period + 1
    }
}
