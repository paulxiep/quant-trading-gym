//! Relative Strength Index (RSI) indicator.

use super::Indicator;
use types::{Candle, IndicatorType};

/// Relative Strength Index indicator.
///
/// Measures the speed and change of price movements on a 0-100 scale.
/// RSI > 70 is typically considered overbought, < 30 oversold.
#[derive(Debug, Clone)]
pub struct Rsi {
    period: usize,
}

impl Rsi {
    /// Create a new RSI indicator with the given period.
    ///
    /// # Panics
    /// Panics if period is 0.
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "RSI period must be > 0");
        Self { period }
    }

    /// Calculate RSI from a slice of prices using Wilder's smoothing method.
    pub fn calculate_from_prices(prices: &[f64], period: usize) -> Option<f64> {
        // Need at least period + 1 prices for period changes
        if prices.len() < period + 1 || period == 0 {
            return None;
        }

        // Calculate price changes
        let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

        // Separate gains and losses
        let (mut avg_gain, mut avg_loss) =
            changes
                .iter()
                .take(period)
                .fold((0.0, 0.0), |(g, l), &change| {
                    if change > 0.0 {
                        (g + change, l)
                    } else {
                        (g, l - change)
                    }
                });

        avg_gain /= period as f64;
        avg_loss /= period as f64;

        // Wilder's smoothing for remaining periods
        let (avg_gain, avg_loss) =
            changes
                .iter()
                .skip(period)
                .fold((avg_gain, avg_loss), |(ag, al), &change| {
                    let (gain, loss) = if change > 0.0 {
                        (change, 0.0)
                    } else {
                        (0.0, -change)
                    };
                    (
                        (ag * (period as f64 - 1.0) + gain) / period as f64,
                        (al * (period as f64 - 1.0) + loss) / period as f64,
                    )
                });

        // Calculate RSI
        if avg_loss == 0.0 {
            Some(100.0) // No losses = max RSI
        } else {
            let rs = avg_gain / avg_loss;
            Some(100.0 - (100.0 / (1.0 + rs)))
        }
    }
}

impl Indicator for Rsi {
    fn indicator_type(&self) -> IndicatorType {
        IndicatorType::Rsi(self.period)
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        // Need period + 1 candles for period price changes
        if candles.len() < self.period + 1 {
            return None;
        }

        let prices: Vec<f64> = candles.iter().map(|c| c.close.to_float()).collect();
        Rsi::calculate_from_prices(&prices, self.period)
    }

    fn required_periods(&self) -> usize {
        self.period + 1
    }
}
