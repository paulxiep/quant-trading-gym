//! Statistical utilities for quantitative analysis.
//!
//! This module provides common statistical functions used across
//! indicator calculations, risk metrics, and factor scoring.
//!
//! ## V3.3 Additions
//!
//! - [`CointegrationTracker`]: Rolling cointegration analysis for pairs trading
//! - [`SectorSentimentAggregator`]: Sector-level sentiment aggregation from news

use std::collections::HashMap;

use crate::rolling::RollingWindow;
use types::Sector;

// =============================================================================
// CointegrationTracker (V3.3)
// =============================================================================

/// Result of cointegration analysis for a pair of assets.
///
/// Used by pairs trading strategies to determine spread positions.
#[derive(Debug, Clone, Copy)]
pub struct CointegrationResult {
    /// Spread value: `price_a - hedge_ratio * price_b`.
    pub spread: f64,
    /// Z-score of spread: `(spread - mean) / std_dev`.
    pub z_score: f64,
    /// OLS hedge ratio (beta): `cov(A, B) / var(B)`.
    pub hedge_ratio: f64,
    /// Mean of historical spreads.
    pub mean: f64,
    /// Standard deviation of historical spreads.
    pub std_dev: f64,
}

/// Rolling cointegration tracker for pairs trading.
///
/// Maintains rolling windows of two price series and computes:
/// - Hedge ratio via OLS regression
/// - Spread series
/// - Z-score for mean reversion signals
///
/// # Design (Declarative, Modular, SoC)
///
/// - **Declarative**: Configure lookback; tracker handles computation
/// - **Modular**: Pure state machine, no external dependencies
/// - **SoC**: Computes statistics only; strategy decides trades
///
/// # Borrow-Checker Safety
///
/// - Owns two `RollingWindow`s internally (no external borrows)
/// - `update()` takes owned values, returns owned `CointegrationResult`
/// - Thread-safe: `Send + Sync` (no interior mutability)
///
/// # Example
///
/// ```
/// use quant::stats::CointegrationTracker;
///
/// let mut tracker = CointegrationTracker::new(100);
///
/// // Sample price data (asset A and asset B)
/// let prices: Vec<(f64, f64)> = (0..150)
///     .map(|i| (100.0 + i as f64 * 0.1, 50.0 + i as f64 * 0.05))
///     .collect();
///
/// // Feed price updates
/// for (price_a, price_b) in prices.iter() {
///     if let Some(result) = tracker.update(*price_a, *price_b) {
///         if result.z_score > 2.0 {
///             // Short spread: sell A, buy B
///         } else if result.z_score < -2.0 {
///             // Long spread: buy A, sell B
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CointegrationTracker {
    /// Rolling window for asset A prices.
    prices_a: RollingWindow,
    /// Rolling window for asset B prices.
    prices_b: RollingWindow,
    /// Rolling window for spread values.
    spreads: RollingWindow,
    /// Lookback period for all windows.
    lookback: usize,
    /// Minimum data points required for valid output.
    min_periods: usize,
}

impl CointegrationTracker {
    /// Create a new cointegration tracker with specified lookback.
    ///
    /// # Arguments
    /// * `lookback` - Number of periods for rolling calculations
    ///
    /// # Panics
    /// Panics if `lookback < 10` (need sufficient data for meaningful statistics).
    pub fn new(lookback: usize) -> Self {
        assert!(
            lookback >= 10,
            "CointegrationTracker requires lookback >= 10"
        );
        Self {
            prices_a: RollingWindow::new(lookback),
            prices_b: RollingWindow::new(lookback),
            spreads: RollingWindow::new(lookback),
            lookback,
            min_periods: lookback / 2, // Require at least half the lookback
        }
    }

    /// Create a tracker with custom minimum periods.
    pub fn with_min_periods(lookback: usize, min_periods: usize) -> Self {
        assert!(
            lookback >= 10,
            "CointegrationTracker requires lookback >= 10"
        );
        assert!(
            min_periods <= lookback,
            "min_periods cannot exceed lookback"
        );
        Self {
            prices_a: RollingWindow::new(lookback),
            prices_b: RollingWindow::new(lookback),
            spreads: RollingWindow::new(lookback),
            lookback,
            min_periods,
        }
    }

    /// Update with new prices and compute cointegration metrics.
    ///
    /// Returns `None` if insufficient data (< `min_periods`).
    ///
    /// # Borrow Safety
    /// Takes owned `f64` values, returns owned `CointegrationResult`.
    pub fn update(&mut self, price_a: f64, price_b: f64) -> Option<CointegrationResult> {
        self.prices_a.push(price_a);
        self.prices_b.push(price_b);

        // Need minimum periods for valid statistics
        if self.prices_a.len() < self.min_periods {
            return None;
        }

        // Compute hedge ratio using OLS: beta = cov(A, B) / var(B)
        let hedge_ratio = self.compute_hedge_ratio()?;

        // Compute spread
        let spread = price_a - hedge_ratio * price_b;
        self.spreads.push(spread);

        // Need spread history for z-score
        if self.spreads.len() < self.min_periods {
            return None;
        }

        // Compute z-score
        let spread_mean = self.spreads.mean()?;
        let spread_std = self.spreads.std_dev()?;

        // Avoid division by zero
        if spread_std < f64::EPSILON {
            return None;
        }

        let z_score = (spread - spread_mean) / spread_std;

        Some(CointegrationResult {
            spread,
            z_score,
            hedge_ratio,
            mean: spread_mean,
            std_dev: spread_std,
        })
    }

    /// Compute OLS hedge ratio: cov(A, B) / var(B).
    ///
    /// Uses simple inline formula (no matrix inversion needed for bivariate OLS).
    fn compute_hedge_ratio(&self) -> Option<f64> {
        let n = self.prices_a.len() as f64;
        if n < 2.0 {
            return None;
        }

        // Compute means
        let mean_a = self.prices_a.mean()?;
        let mean_b = self.prices_b.mean()?;

        // Compute covariance(A, B) and variance(B) via fold
        let (cov_ab, var_b) = self
            .prices_a
            .iter()
            .zip(self.prices_b.iter())
            .map(|(a, b)| ((a - mean_a) * (b - mean_b), (b - mean_b).powi(2)))
            .fold((0.0, 0.0), |(cov, var), (dc, dv)| (cov + dc, var + dv));

        // Population covariance/variance (divide by n)
        let cov_ab = cov_ab / n;
        let var_b = var_b / n;

        // Avoid division by zero
        (var_b >= f64::EPSILON).then_some(cov_ab / var_b)
    }

    /// Get the current lookback period.
    pub fn lookback(&self) -> usize {
        self.lookback
    }

    /// Get the number of data points collected.
    pub fn len(&self) -> usize {
        self.prices_a.len()
    }

    /// Check if tracker has enough data for output.
    pub fn is_ready(&self) -> bool {
        self.prices_a.len() >= self.min_periods
    }

    /// Check if tracker has no data.
    pub fn is_empty(&self) -> bool {
        self.prices_a.is_empty()
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.prices_a.clear();
        self.prices_b.clear();
        self.spreads.clear();
    }
}

// =============================================================================
// SectorSentimentAggregator (V3.3)
// =============================================================================

/// Aggregated sentiment for a sector.
#[derive(Debug, Clone, Copy, Default)]
pub struct SectorSentiment {
    /// Aggregate sentiment score (-1.0 to +1.0).
    pub sentiment: f64,
    /// Total weight (for normalization).
    pub total_weight: f64,
    /// Number of contributing events.
    pub event_count: usize,
}

/// Stateless sector sentiment aggregator.
///
/// Aggregates sentiment from active news events by sector.
/// Uses time-weighted decay: events closer to expiry have less weight.
///
/// # Design (Declarative, Modular, SoC)
///
/// - **Declarative**: Define aggregation rules; call `aggregate()`
/// - **Modular**: Pure function, no internal state
/// - **SoC**: Computes sentiment only; strategy decides allocation
///
/// # Borrow-Checker Safety
///
/// - Stateless: all methods take references, return owned values
/// - No mutation: `&self` methods only
/// - Thread-safe: `Send + Sync`
///
/// # Example
///
/// ```ignore
/// use quant::stats::SectorSentimentAggregator;
///
/// let aggregator = SectorSentimentAggregator::new();
/// let sentiments = aggregator.aggregate_all(active_events, current_tick);
///
/// for (sector, sentiment) in &sentiments {
///     if sentiment.sentiment > 0.3 {
///         // Overweight this sector
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct SectorSentimentAggregator {
    /// Minimum magnitude for event to contribute (filter noise).
    min_magnitude: f64,
}

impl SectorSentimentAggregator {
    /// Create a new aggregator with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an aggregator with minimum magnitude filter.
    ///
    /// Events with `magnitude < min_magnitude` are ignored.
    pub fn with_min_magnitude(min_magnitude: f64) -> Self {
        Self {
            min_magnitude: min_magnitude.clamp(0.0, 1.0),
        }
    }

    /// Aggregate sentiment for all sectors from active events.
    ///
    /// Returns a map of sector → aggregated sentiment.
    ///
    /// # Arguments
    /// * `events` - Slice of active news events
    /// * `current_tick` - Current simulation tick (for decay calculation)
    ///
    /// # Borrow Safety
    /// Takes `&[NewsEvent]` (immutable), returns owned `HashMap`.
    pub fn aggregate_all<E: NewsEventLike>(
        &self,
        events: &[E],
        current_tick: u64,
    ) -> HashMap<Sector, SectorSentiment> {
        // Filter valid events and compute contributions
        let contributions = events.iter().filter_map(|event| {
            // Skip events below magnitude threshold or without sector
            let sector = event.sector()?;
            if event.magnitude() < self.min_magnitude {
                return None;
            }

            // Skip expired events (decay <= 0)
            let decay = event.decay_factor(current_tick);
            if decay <= f64::EPSILON {
                return None;
            }

            let weight = event.magnitude() * decay;
            let contribution = event.sentiment() * weight;
            Some((sector, weight, contribution))
        });

        // Aggregate by sector
        let mut result: HashMap<Sector, SectorSentiment> =
            contributions.fold(HashMap::new(), |mut acc, (sector, weight, contribution)| {
                let entry = acc.entry(sector).or_default();
                entry.sentiment += contribution;
                entry.total_weight += weight;
                entry.event_count += 1;
                acc
            });

        // Normalize by total weight
        result.values_mut().for_each(|sentiment| {
            if sentiment.total_weight > f64::EPSILON {
                sentiment.sentiment /= sentiment.total_weight;
            }
        });

        result
    }

    /// Aggregate sentiment for a specific sector.
    ///
    /// Returns `None` if no events affect the sector.
    pub fn aggregate_sector<E: NewsEventLike>(
        &self,
        events: &[E],
        sector: Sector,
        current_tick: u64,
    ) -> Option<SectorSentiment> {
        // Filter and aggregate contributions for the target sector
        let sentiment = events
            .iter()
            .filter(|e| e.sector() == Some(sector) && e.magnitude() >= self.min_magnitude)
            .filter_map(|event| {
                let decay = event.decay_factor(current_tick);
                (decay > f64::EPSILON).then(|| {
                    let weight = event.magnitude() * decay;
                    (event.sentiment() * weight, weight)
                })
            })
            .fold(
                SectorSentiment::default(),
                |mut acc, (contribution, weight)| {
                    acc.sentiment += contribution;
                    acc.total_weight += weight;
                    acc.event_count += 1;
                    acc
                },
            );

        // Return None if no events contributed
        if sentiment.event_count == 0 {
            return None;
        }

        // Normalize and return
        Some(SectorSentiment {
            sentiment: if sentiment.total_weight > f64::EPSILON {
                sentiment.sentiment / sentiment.total_weight
            } else {
                sentiment.sentiment
            },
            ..sentiment
        })
    }
}

/// Trait for news event-like types to enable decoupling from `news` crate.
///
/// This allows `quant` to remain independent of the `news` crate while
/// still being able to aggregate sentiment from news events.
pub trait NewsEventLike {
    /// Get the sector this event affects, if any.
    fn sector(&self) -> Option<Sector>;
    /// Get the sentiment direction (-1.0 to +1.0).
    fn sentiment(&self) -> f64;
    /// Get the magnitude (0.0 to 1.0).
    fn magnitude(&self) -> f64;
    /// Get the decay factor at the given tick (1.0 at start, 0.0 at expiry).
    fn decay_factor(&self, current_tick: u64) -> f64;
}

// =============================================================================
// Original Statistical Functions
// =============================================================================

/// Calculate the mean of a slice of values.
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Calculate the variance of a slice of values (population variance).
pub fn variance(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 2 {
        return None;
    }

    let mean_val = mean(values)?;
    let sum_sq: f64 = values.iter().map(|v| (v - mean_val).powi(2)).sum();
    Some(sum_sq / n as f64)
}

/// Calculate the standard deviation (population).
pub fn std_dev(values: &[f64]) -> Option<f64> {
    variance(values).map(|v| v.sqrt())
}

/// Calculate the sample variance (n-1 denominator).
pub fn sample_variance(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 2 {
        return None;
    }

    let mean_val = mean(values)?;
    let sum_sq: f64 = values.iter().map(|v| (v - mean_val).powi(2)).sum();
    Some(sum_sq / (n - 1) as f64)
}

/// Calculate the sample standard deviation (n-1 denominator).
pub fn sample_std_dev(values: &[f64]) -> Option<f64> {
    sample_variance(values).map(|v| v.sqrt())
}

/// Calculate returns from a price series.
/// Returns (price[i] - price[i-1]) / price[i-1] for each consecutive pair.
pub fn returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .filter_map(|w| {
            if w[0] != 0.0 {
                Some((w[1] - w[0]) / w[0])
            } else {
                None
            }
        })
        .collect()
}

/// Calculate log returns from a price series.
/// Returns ln(price[i] / price[i-1]) for each consecutive pair.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .filter_map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                Some((w[1] / w[0]).ln())
            } else {
                None
            }
        })
        .collect()
}

/// Calculate covariance between two series.
pub fn covariance(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;
    let n = x.len();

    let sum: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    Some(sum / n as f64)
}

/// Calculate Pearson correlation coefficient.
pub fn correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    let cov = covariance(x, y)?;
    let std_x = std_dev(x)?;
    let std_y = std_dev(y)?;

    if std_x == 0.0 || std_y == 0.0 {
        return None;
    }

    Some(cov / (std_x * std_y))
}

/// Calculate beta (slope) of y with respect to x using linear regression.
pub fn beta(x: &[f64], y: &[f64]) -> Option<f64> {
    let cov = covariance(x, y)?;
    let var_x = variance(x)?;

    if var_x == 0.0 {
        return None;
    }

    Some(cov / var_x)
}

/// Calculate percentile value from a sorted slice.
/// Percentile should be between 0.0 and 1.0 (e.g., 0.95 for 95th percentile).
pub fn percentile(sorted_values: &[f64], pct: f64) -> Option<f64> {
    if sorted_values.is_empty() || !(0.0..=1.0).contains(&pct) {
        return None;
    }

    let n = sorted_values.len();
    if n == 1 {
        return Some(sorted_values[0]);
    }

    let idx = pct * (n - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if upper >= n {
        Some(sorted_values[n - 1])
    } else {
        Some(sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac)
    }
}

/// Calculate exponential weighted moving average.
/// Alpha is the smoothing factor (higher = more weight on recent values).
pub fn ewma(values: &[f64], alpha: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&alpha) {
        return None;
    }

    let initial = values[0];
    Some(
        values
            .iter()
            .skip(1)
            .fold(initial, |prev, curr| alpha * curr + (1.0 - alpha) * prev),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), Some(3.0));
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn test_std_dev() {
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = std_dev(&values).unwrap();
        assert!((std - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_returns() {
        let prices = [100.0, 110.0, 99.0, 121.0];
        let rets = returns(&prices);
        assert_eq!(rets.len(), 3);
        assert!((rets[0] - 0.1).abs() < 0.0001); // 10% gain
        assert!((rets[1] - (-0.1)).abs() < 0.0001); // 10% loss
    }

    #[test]
    fn test_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 0.0001); // Perfect positive correlation
    }

    #[test]
    fn test_percentile() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&sorted, 0.5).unwrap() - 5.5).abs() < 0.0001);
        assert!((percentile(&sorted, 0.0).unwrap() - 1.0).abs() < 0.0001);
        assert!((percentile(&sorted, 1.0).unwrap() - 10.0).abs() < 0.0001);
    }

    // =========================================================================
    // V3.3 Tests: CointegrationTracker
    // =========================================================================

    #[test]
    fn test_cointegration_tracker_new() {
        let tracker = CointegrationTracker::new(20);
        assert_eq!(tracker.lookback(), 20);
        assert!(tracker.is_empty());
        assert!(!tracker.is_ready());
    }

    #[test]
    #[should_panic(expected = "requires lookback >= 10")]
    fn test_cointegration_tracker_small_lookback() {
        CointegrationTracker::new(5);
    }

    #[test]
    fn test_cointegration_tracker_returns_none_until_ready() {
        let mut tracker = CointegrationTracker::new(10);
        // min_periods = 10 / 2 = 5

        // First 4 updates should return None
        for i in 0..4 {
            // Add small noise so spread has variance
            let noise = (i as f64 * 0.1).sin();
            let result = tracker.update(100.0 + i as f64 + noise, 50.0 + i as f64 * 0.5);
            assert!(result.is_none(), "iteration {} should be None", i);
        }

        // 5th update - prices ready but spreads not yet (need 5 spread values)
        let result = tracker.update(104.0, 52.0);
        assert!(
            result.is_none(),
            "5th should still be None (spread history building)"
        );

        // Continue until we have enough spread history (with noise for variance)
        for i in 5..10 {
            let noise = (i as f64 * 0.1).sin();
            tracker.update(100.0 + i as f64 + noise, 50.0 + i as f64 * 0.5);
        }

        // Now should be ready - the noise ensures spread has variance
        let result = tracker.update(110.0 + 0.5, 55.0);
        assert!(
            result.is_some(),
            "should be Some after enough data with spread variance"
        );
        assert!(tracker.is_ready());
    }

    #[test]
    fn test_cointegration_tracker_near_perfect_correlation() {
        let mut tracker = CointegrationTracker::new(10);

        // Price B ≈ 0.5 * Price A (near-perfect correlation, hedge ratio should be ~2.0)
        // Add small noise to ensure spread has variance
        for i in 0..20 {
            let noise = (i as f64 * 0.3).sin() * 0.5; // Small oscillation
            let price_a = 100.0 + i as f64 + noise;
            let price_b = 50.0 + i as f64 * 0.5; // B moves half as much (no noise)
            tracker.update(price_a, price_b);
        }

        // Now get a result
        let result = tracker.update(120.0, 60.0);
        assert!(
            result.is_some(),
            "Should have result after 20+ updates with spread variance"
        );
        let result = result.unwrap();

        // Hedge ratio should be close to 2.0 (A ≈ 2 * B)
        assert!(
            (result.hedge_ratio - 2.0).abs() < 0.2,
            "hedge_ratio: {}",
            result.hedge_ratio
        );
    }

    #[test]
    fn test_cointegration_tracker_detects_divergence() {
        let mut tracker = CointegrationTracker::new(20);

        // Build history with stable 1:1 relationship (A ≈ B)
        for i in 0..25 {
            let price_a = 100.0 + (i as f64 * 0.1);
            let price_b = 100.0 + (i as f64 * 0.1);
            tracker.update(price_a, price_b);
        }

        // Now A jumps up significantly while B stays stable
        // With hedge_ratio ≈ 1.0, spread = A - B = 110 - 102 = 8
        // Previous spreads were all ~0, so z_score should be large (positive or negative)
        let result = tracker.update(110.0, 102.0).unwrap();

        // Z-score should be significant (absolute value > 1)
        assert!(
            result.z_score.abs() > 1.0,
            "z_score should detect divergence (abs): {}",
            result.z_score
        );
    }

    #[test]
    fn test_cointegration_tracker_clear() {
        let mut tracker = CointegrationTracker::new(10);

        for i in 0..15 {
            tracker.update(100.0 + i as f64, 100.0 + i as f64);
        }
        assert!(tracker.is_ready());

        tracker.clear();
        assert!(tracker.is_empty());
        assert!(!tracker.is_ready());
    }

    // =========================================================================
    // V3.3 Tests: SectorSentimentAggregator
    // =========================================================================

    /// Mock news event for testing SectorSentimentAggregator
    struct MockNewsEvent {
        sector: Option<Sector>,
        sentiment: f64,
        magnitude: f64,
        start_tick: u64,
        duration: u64,
    }

    impl NewsEventLike for MockNewsEvent {
        fn sector(&self) -> Option<Sector> {
            self.sector
        }

        fn sentiment(&self) -> f64 {
            self.sentiment
        }

        fn magnitude(&self) -> f64 {
            self.magnitude
        }

        fn decay_factor(&self, current_tick: u64) -> f64 {
            if current_tick < self.start_tick || current_tick >= self.start_tick + self.duration {
                return 0.0;
            }
            let elapsed = (current_tick - self.start_tick) as f64;
            let total = self.duration as f64;
            1.0 - (elapsed / total)
        }
    }

    #[test]
    fn test_sector_sentiment_aggregator_empty() {
        let aggregator = SectorSentimentAggregator::new();
        let events: Vec<MockNewsEvent> = vec![];
        let result = aggregator.aggregate_all(&events, 100);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sector_sentiment_aggregator_single_event() {
        let aggregator = SectorSentimentAggregator::new();
        let events = vec![MockNewsEvent {
            sector: Some(Sector::Tech),
            sentiment: 0.8,
            magnitude: 0.5,
            start_tick: 0,
            duration: 100,
        }];

        let result = aggregator.aggregate_all(&events, 0);

        assert_eq!(result.len(), 1);
        let tech_sentiment = result.get(&Sector::Tech).unwrap();
        assert!((tech_sentiment.sentiment - 0.8).abs() < 0.01);
        assert_eq!(tech_sentiment.event_count, 1);
    }

    #[test]
    fn test_sector_sentiment_aggregator_multiple_events_same_sector() {
        let aggregator = SectorSentimentAggregator::new();
        let events = vec![
            MockNewsEvent {
                sector: Some(Sector::Tech),
                sentiment: 0.6,
                magnitude: 0.5,
                start_tick: 0,
                duration: 100,
            },
            MockNewsEvent {
                sector: Some(Sector::Tech),
                sentiment: -0.4,
                magnitude: 0.5,
                start_tick: 0,
                duration: 100,
            },
        ];

        let result = aggregator.aggregate_all(&events, 0);

        let tech_sentiment = result.get(&Sector::Tech).unwrap();
        // Average of 0.6 and -0.4 = 0.1 (equal weights)
        assert!(
            (tech_sentiment.sentiment - 0.1).abs() < 0.01,
            "sentiment: {}",
            tech_sentiment.sentiment
        );
        assert_eq!(tech_sentiment.event_count, 2);
    }

    #[test]
    fn test_sector_sentiment_aggregator_decay() {
        let aggregator = SectorSentimentAggregator::new();
        let events = vec![MockNewsEvent {
            sector: Some(Sector::Utilities),
            sentiment: 1.0,
            magnitude: 1.0,
            start_tick: 0,
            duration: 100,
        }];

        // At tick 0, decay = 1.0, full sentiment
        let result = aggregator.aggregate_all(&events, 0);
        let sentiment = result.get(&Sector::Utilities).unwrap().sentiment;
        assert!((sentiment - 1.0).abs() < 0.01);

        // At tick 50, decay = 0.5, half weight but sentiment unchanged (single event)
        let result = aggregator.aggregate_all(&events, 50);
        let sentiment = result.get(&Sector::Utilities).unwrap().sentiment;
        assert!((sentiment - 1.0).abs() < 0.01); // Still 1.0, just lower weight

        // At tick 100, event expired
        let result = aggregator.aggregate_all(&events, 100);
        assert!(result.get(&Sector::Utilities).is_none());
    }

    #[test]
    fn test_sector_sentiment_aggregator_min_magnitude_filter() {
        let aggregator = SectorSentimentAggregator::with_min_magnitude(0.3);
        let events = vec![
            MockNewsEvent {
                sector: Some(Sector::Healthcare),
                sentiment: 0.9,
                magnitude: 0.2, // Below threshold
                start_tick: 0,
                duration: 100,
            },
            MockNewsEvent {
                sector: Some(Sector::Healthcare),
                sentiment: -0.5,
                magnitude: 0.5, // Above threshold
                start_tick: 0,
                duration: 100,
            },
        ];

        let result = aggregator.aggregate_all(&events, 0);

        let sentiment = result.get(&Sector::Healthcare).unwrap();
        // Only the -0.5 event should contribute
        assert!(
            (sentiment.sentiment - (-0.5)).abs() < 0.01,
            "sentiment: {}",
            sentiment.sentiment
        );
        assert_eq!(sentiment.event_count, 1);
    }

    #[test]
    fn test_sector_sentiment_aggregator_specific_sector() {
        let aggregator = SectorSentimentAggregator::new();
        let events = vec![
            MockNewsEvent {
                sector: Some(Sector::Tech),
                sentiment: 0.5,
                magnitude: 0.5,
                start_tick: 0,
                duration: 100,
            },
            MockNewsEvent {
                sector: Some(Sector::Utilities),
                sentiment: -0.5,
                magnitude: 0.5,
                start_tick: 0,
                duration: 100,
            },
        ];

        let tech_result = aggregator.aggregate_sector(&events, Sector::Tech, 0);
        assert!(tech_result.is_some());
        assert!((tech_result.unwrap().sentiment - 0.5).abs() < 0.01);

        let finance_result = aggregator.aggregate_sector(&events, Sector::Finance, 0);
        assert!(finance_result.is_none());
    }
}
