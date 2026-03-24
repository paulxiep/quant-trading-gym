//! MACD (Moving Average Convergence Divergence) indicator.

use super::Indicator;
use super::ema::Ema;
use types::{Candle, IndicatorType, MacdOutput};

/// MACD indicator.
///
/// Shows the relationship between two EMAs and includes a signal line.
/// Standard configuration is (12, 26, 9).
#[derive(Debug, Clone)]
pub struct Macd {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl Macd {
    /// Create a new MACD indicator with custom periods.
    ///
    /// # Panics
    /// Panics if any period is 0 or if fast_period >= slow_period.
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        assert!(fast_period > 0, "MACD fast period must be > 0");
        assert!(slow_period > 0, "MACD slow period must be > 0");
        assert!(signal_period > 0, "MACD signal period must be > 0");
        assert!(
            fast_period < slow_period,
            "MACD fast period must be < slow period"
        );
        Self {
            fast_period,
            slow_period,
            signal_period,
        }
    }

    /// Create MACD with standard (12, 26, 9) configuration.
    pub fn standard() -> Self {
        Self::new(12, 26, 9)
    }

    /// Calculate full MACD output including signal line and histogram.
    pub fn calculate_full(&self, candles: &[Candle]) -> Option<MacdOutput> {
        let prices: Vec<f64> = candles.iter().map(|c| c.close.to_float()).collect();
        self.calculate_full_from_prices(&prices)
    }

    /// Calculate full MACD output from price data.
    pub fn calculate_full_from_prices(&self, prices: &[f64]) -> Option<MacdOutput> {
        // Need enough data for slow EMA + signal EMA
        if prices.len() < self.slow_period + self.signal_period {
            return None;
        }

        // Calculate MACD line at each point after slow_period
        let macd_values: Vec<f64> = (self.slow_period..=prices.len())
            .filter_map(|i| {
                let slice = &prices[..i];
                let fast_ema = Ema::calculate_from_prices(slice, self.fast_period)?;
                let slow_ema = Ema::calculate_from_prices(slice, self.slow_period)?;
                Some(fast_ema - slow_ema)
            })
            .collect();

        if macd_values.len() < self.signal_period {
            return None;
        }

        // Calculate signal line (EMA of MACD values)
        let signal_line = Ema::calculate_from_prices(&macd_values, self.signal_period)?;
        let macd_line = *macd_values.last()?;
        let histogram = macd_line - signal_line;

        Some(MacdOutput {
            macd_line,
            signal_line,
            histogram,
        })
    }
}

impl Indicator for Macd {
    fn indicator_type(&self) -> IndicatorType {
        // Return MacdLine as the primary type (calculate returns macd_line)
        IndicatorType::MacdLine {
            fast: self.fast_period,
            slow: self.slow_period,
            signal: self.signal_period,
        }
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        // Returns just the MACD line value for trait compatibility
        self.calculate_full(candles).map(|m| m.macd_line)
    }

    fn required_periods(&self) -> usize {
        self.slow_period + self.signal_period
    }
}
