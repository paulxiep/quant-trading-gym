//! Statistical distributions for order generation (V3.4).
//!
//! Trait-based for extensibility — distributions can be swapped without
//! changing pool logic.
//!
//! # Distributions Used
//!
//! - **Order Size**: Log-normal (many small orders, few large)
//! - **Price Offset**: Exponential decay (most orders near mid)
//!
//! # Design Principles
//!
//! - **Modular**: Traits allow swapping distributions
//! - **SoC**: Pure math, no side effects

use rand::Rng;
use rand_distr::{Distribution, Exp, LogNormal};
use types::{Price, Quantity};

// =============================================================================
// PriceDistribution Trait
// =============================================================================

/// Trait for generating order price offsets from mid price.
pub trait PriceDistribution: Send + Sync {
    /// Generate a price offset from mid price.
    ///
    /// Returns signed offset in raw price units (positive = above mid).
    fn sample_offset(&self, rng: &mut impl Rng, mid_price: Price) -> i64;
}

// =============================================================================
// ExponentialPriceSpread
// =============================================================================

/// Exponential decay distribution for price offsets.
///
/// Orders cluster near mid price, with exponentially fewer at distance.
/// This matches empirical market microstructure observations.
#[derive(Debug, Clone)]
pub struct ExponentialPriceSpread {
    /// Lambda parameter (higher = tighter spread around mid).
    lambda: f64,
    /// Maximum offset as fraction of mid price.
    max_fraction: f64,
}

impl ExponentialPriceSpread {
    /// Create a new exponential price spread.
    ///
    /// # Arguments
    /// * `lambda` - Decay rate. λ=20 gives most orders within 5% of mid.
    /// * `max_fraction` - Maximum offset fraction (e.g., 0.02 = 2%).
    pub fn new(lambda: f64, max_fraction: f64) -> Self {
        Self {
            lambda: lambda.max(0.1), // Prevent division issues
            max_fraction: max_fraction.clamp(0.001, 0.5),
        }
    }
}

impl PriceDistribution for ExponentialPriceSpread {
    fn sample_offset(&self, rng: &mut impl Rng, mid_price: Price) -> i64 {
        // Sample from exponential distribution
        let exp = Exp::new(self.lambda).unwrap_or_else(|_| Exp::new(1.0).unwrap());
        let magnitude_frac = exp.sample(rng).min(self.max_fraction);

        // Convert to raw price offset
        let offset_raw = (mid_price.0 as f64 * magnitude_frac) as i64;

        // Randomly positive or negative (buy or sell side of book)
        if rng.r#gen_bool(0.5) {
            offset_raw
        } else {
            -offset_raw
        }
    }
}

// =============================================================================
// SizeDistribution Trait
// =============================================================================

/// Trait for generating order sizes.
pub trait SizeDistribution: Send + Sync {
    /// Generate an order size.
    fn sample(&self, rng: &mut impl Rng) -> Quantity;
}

// =============================================================================
// LogNormalSize
// =============================================================================

/// Log-normal distribution for order sizes.
///
/// Produces many small orders, few large ones — matching real market
/// microstructure where retail orders are small and institutional
/// orders are occasionally large.
#[derive(Debug, Clone)]
pub struct LogNormalSize {
    /// Log-normal distribution instance
    dist: LogNormal<f64>,
    /// Maximum order size (hard cap)
    max_size: u64,
    /// Minimum order size (floor)
    min_size: u64,
}

impl LogNormalSize {
    /// Create a log-normal size distribution.
    ///
    /// # Arguments
    /// * `mean` - Mean order size (e.g., 15 shares)
    /// * `stddev` - Standard deviation (e.g., 10)
    /// * `min_size` - Minimum order size
    /// * `max_size` - Maximum order size
    pub fn new(mean: f64, stddev: f64, min_size: u64, max_size: u64) -> Self {
        // Convert mean/stddev to log-normal μ and σ parameters
        // For log-normal: mean = exp(μ + σ²/2), var = (exp(σ²) - 1) * exp(2μ + σ²)
        let mean = mean.max(1.0);
        let variance = (stddev * stddev).max(0.1);

        let sigma_sq = (1.0 + variance / (mean * mean)).ln();
        let sigma = sigma_sq.sqrt();
        let mu = mean.ln() - sigma_sq / 2.0;

        Self {
            dist: LogNormal::new(mu, sigma).unwrap_or_else(|_| LogNormal::new(2.0, 0.5).unwrap()),
            max_size,
            min_size: min_size.max(1),
        }
    }
}

impl SizeDistribution for LogNormalSize {
    fn sample(&self, rng: &mut impl Rng) -> Quantity {
        let size = self.dist.sample(rng).round() as u64;
        Quantity(size.clamp(self.min_size, self.max_size))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_price_spread() {
        let dist = ExponentialPriceSpread::new(20.0, 0.05);
        let mut rng = rand::thread_rng();
        let mid = Price::from_float(100.0);

        let mut offsets = Vec::new();
        for _ in 0..1000 {
            offsets.push(dist.sample_offset(&mut rng, mid));
        }

        // Most offsets should be within max deviation
        let within_bounds = offsets
            .iter()
            .filter(|o| o.abs() <= (mid.0 as f64 * 0.05) as i64)
            .count();
        assert!(
            within_bounds > 900,
            "Most offsets should be within 5% of mid, got {} within bounds",
            within_bounds
        );

        // Should have both positive and negative
        let positive = offsets.iter().filter(|o| **o > 0).count();
        let negative = offsets.iter().filter(|o| **o < 0).count();
        assert!(positive > 300 && negative > 300, "Should be balanced");
    }

    #[test]
    fn test_log_normal_size() {
        let dist = LogNormalSize::new(15.0, 10.0, 1, 100);
        let mut rng = rand::thread_rng();

        let mut sizes = Vec::new();
        for _ in 0..1000 {
            sizes.push(dist.sample(&mut rng).raw());
        }

        // All within bounds
        assert!(sizes.iter().all(|s| *s >= 1 && *s <= 100));

        // Mean should be roughly around target
        let mean: f64 = sizes.iter().map(|s| *s as f64).sum::<f64>() / sizes.len() as f64;
        assert!(
            mean > 5.0 && mean < 50.0,
            "Mean {} should be reasonable",
            mean
        );

        // Should have more small orders than large
        let small = sizes.iter().filter(|s| **s < 20).count();
        let large = sizes.iter().filter(|s| **s > 50).count();
        assert!(
            small > large,
            "Log-normal should skew small: {} small, {} large",
            small,
            large
        );
    }
}
