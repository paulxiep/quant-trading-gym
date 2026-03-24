//! Exponential Moving Average (EMA) indicator.

use super::Indicator;
use types::{Candle, IndicatorType};

/// Exponential Moving Average indicator.
///
/// Gives more weight to recent prices using exponential smoothing.
/// Multiplier = 2 / (period + 1)
#[derive(Debug, Clone)]
pub struct Ema {
    period: usize,
    multiplier: f64,
}

impl Ema {
    /// Create a new EMA indicator with the given period.
    ///
    /// # Panics
    /// Panics if period is 0.
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "EMA period must be > 0");
        Self {
            period,
            multiplier: 2.0 / (period as f64 + 1.0),
        }
    }

    /// Calculate EMA from a slice of prices.
    /// Uses SMA of first `period` values as the initial EMA value.
    pub fn calculate_from_prices(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period || period == 0 {
            return None;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);

        // Initial EMA is SMA of first `period` values
        let initial_sma: f64 = prices.iter().take(period).sum::<f64>() / period as f64;

        // Apply EMA formula to remaining values
        let ema = prices
            .iter()
            .skip(period)
            .fold(initial_sma, |prev_ema, price| {
                (price - prev_ema) * multiplier + prev_ema
            });

        Some(ema)
    }
}

impl Indicator for Ema {
    fn indicator_type(&self) -> IndicatorType {
        IndicatorType::Ema(self.period)
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        if candles.len() < self.period {
            return None;
        }

        let prices: Vec<f64> = candles.iter().map(|c| c.close.to_float()).collect();

        // Initial EMA is SMA of first `period` values
        let initial_sma: f64 = prices.iter().take(self.period).sum::<f64>() / self.period as f64;

        // Apply EMA formula to remaining values
        let ema = prices
            .iter()
            .skip(self.period)
            .fold(initial_sma, |prev_ema, price| {
                (price - prev_ema) * self.multiplier + prev_ema
            });

        Some(ema)
    }

    fn required_periods(&self) -> usize {
        self.period
    }
}
