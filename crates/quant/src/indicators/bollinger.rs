//! Bollinger Bands indicator.

use super::Indicator;
use types::{BollingerOutput, Candle, IndicatorType};

/// Bollinger Bands indicator.
///
/// Volatility bands placed above and below a moving average.
/// Default is 20-period SMA with 2 standard deviations.
#[derive(Debug, Clone)]
pub struct BollingerBands {
    period: usize,
    std_dev_multiplier: f64,
}

impl BollingerBands {
    /// Create new Bollinger Bands with custom parameters.
    ///
    /// # Arguments
    /// * `period` - SMA period for middle band
    /// * `std_dev_multiplier` - Number of standard deviations for bands (typically 2.0)
    ///
    /// # Panics
    /// Panics if period is 0.
    pub fn new(period: usize, std_dev_multiplier: f64) -> Self {
        assert!(period > 0, "Bollinger period must be > 0");
        Self {
            period,
            std_dev_multiplier,
        }
    }

    /// Create Bollinger Bands with standard (20, 2.0) configuration.
    pub fn standard() -> Self {
        Self::new(20, 2.0)
    }

    /// Calculate full Bollinger Bands output.
    pub fn calculate_full(&self, candles: &[Candle]) -> Option<BollingerOutput> {
        if candles.len() < self.period {
            return None;
        }

        let prices: Vec<f64> = candles
            .iter()
            .rev()
            .take(self.period)
            .map(|c| c.close.to_float())
            .collect();

        let mean = prices.iter().sum::<f64>() / self.period as f64;

        let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / self.period as f64;
        let std_dev = variance.sqrt();

        let upper = mean + (std_dev * self.std_dev_multiplier);
        let lower = mean - (std_dev * self.std_dev_multiplier);
        let current_price = candles.last()?.close.to_float();

        // Band width as percentage of middle band
        let bandwidth = if mean != 0.0 {
            (upper - lower) / mean * 100.0
        } else {
            0.0
        };

        // %B: where is price relative to bands (0 = lower, 1 = upper)
        let percent_b = if upper != lower {
            (current_price - lower) / (upper - lower)
        } else {
            0.5
        };

        Some(BollingerOutput {
            upper,
            middle: mean,
            lower,
            bandwidth,
            percent_b,
        })
    }
}

impl Indicator for BollingerBands {
    fn indicator_type(&self) -> IndicatorType {
        // Return BollingerMiddle as the primary type (calculate returns middle band)
        IndicatorType::BollingerMiddle {
            period: self.period,
            std_dev_bp: (self.std_dev_multiplier * 100.0) as u32,
        }
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        // Returns middle band (SMA) for trait compatibility
        self.calculate_full(candles).map(|b| b.middle)
    }

    fn required_periods(&self) -> usize {
        self.period
    }
}
