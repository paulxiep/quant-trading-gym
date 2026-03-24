//! Simple Moving Average (SMA) indicator.

use super::Indicator;
use types::{Candle, IndicatorType};

/// Simple Moving Average indicator.
///
/// Computes the arithmetic mean of the closing prices over a specified period.
#[derive(Debug, Clone)]
pub struct Sma {
    period: usize,
}

impl Sma {
    /// Create a new SMA indicator with the given period.
    ///
    /// # Panics
    /// Panics if period is 0.
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "SMA period must be > 0");
        Self { period }
    }

    /// Calculate SMA from a slice of closing prices (f64).
    pub fn calculate_from_prices(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period || period == 0 {
            return None;
        }
        let sum: f64 = prices.iter().rev().take(period).sum();
        Some(sum / period as f64)
    }
}

impl Indicator for Sma {
    fn indicator_type(&self) -> IndicatorType {
        IndicatorType::Sma(self.period)
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        if candles.len() < self.period {
            return None;
        }

        let sum: f64 = candles
            .iter()
            .rev()
            .take(self.period)
            .map(|c| c.close.to_float())
            .sum();

        Some(sum / self.period as f64)
    }

    fn required_periods(&self) -> usize {
        self.period
    }
}
