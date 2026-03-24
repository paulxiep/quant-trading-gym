//! Rolling window data structure for efficient indicator calculations.
//!
//! Provides O(1) push operations and efficient window access for computing
//! moving averages, standard deviations, and other rolling statistics.

use std::collections::VecDeque;

/// A fixed-size rolling window of values.
///
/// Efficiently maintains the most recent `capacity` values, automatically
/// discarding old values when new ones are pushed. Supports incremental
/// computation of statistics like sum and mean.
///
/// # Example
/// ```
/// use quant::rolling::RollingWindow;
///
/// let mut window = RollingWindow::new(3);
/// window.push(1.0);
/// window.push(2.0);
/// window.push(3.0);
/// assert_eq!(window.mean(), Some(2.0));
///
/// window.push(4.0); // Drops 1.0
/// assert_eq!(window.mean(), Some(3.0));
/// ```
#[derive(Debug, Clone)]
pub struct RollingWindow {
    data: VecDeque<f64>,
    capacity: usize,
    /// Running sum for O(1) mean computation.
    sum: f64,
}

impl RollingWindow {
    /// Create a new rolling window with the given capacity.
    ///
    /// # Panics
    /// Panics if capacity is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RollingWindow capacity must be > 0");
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }

    /// Push a value into the window.
    ///
    /// If the window is full, the oldest value is removed and returned.
    pub fn push(&mut self, value: f64) -> Option<f64> {
        let removed = if self.data.len() >= self.capacity {
            let old = self.data.pop_front();
            if let Some(v) = old {
                self.sum -= v;
            }
            old
        } else {
            None
        };

        self.data.push_back(value);
        self.sum += value;
        removed
    }

    /// Get the number of values currently in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the window is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if the window is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.data.len() >= self.capacity
    }

    /// Get the capacity of the window.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the sum of all values in the window.
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Get the mean of values in the window.
    ///
    /// Returns `None` if the window is empty.
    #[inline]
    pub fn mean(&self) -> Option<f64> {
        if self.is_empty() {
            None
        } else {
            Some(self.sum / self.data.len() as f64)
        }
    }

    /// Compute the variance of values in the window.
    ///
    /// Returns `None` if the window has fewer than 2 values.
    pub fn variance(&self) -> Option<f64> {
        if self.data.len() < 2 {
            return None;
        }

        let mean = self.sum / self.data.len() as f64;
        let sum_sq: f64 = self.data.iter().map(|v| (v - mean).powi(2)).sum();
        Some(sum_sq / self.data.len() as f64)
    }

    /// Compute the standard deviation of values in the window.
    ///
    /// Returns `None` if the window has fewer than 2 values.
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    /// Get the most recent value.
    #[inline]
    pub fn last(&self) -> Option<f64> {
        self.data.back().copied()
    }

    /// Get the oldest value.
    #[inline]
    pub fn first(&self) -> Option<f64> {
        self.data.front().copied()
    }

    /// Get a value by index (0 = oldest).
    #[inline]
    pub fn get(&self, index: usize) -> Option<f64> {
        self.data.get(index).copied()
    }

    /// Iterate over values from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.data.iter().copied()
    }

    /// Get values as a slice-compatible iterator for calculations.
    /// Returns values from oldest to newest.
    pub fn as_slice(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.data.iter().copied()
    }

    /// Get the maximum value in the window.
    pub fn max(&self) -> Option<f64> {
        self.data
            .iter()
            .copied()
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
    }

    /// Get the minimum value in the window.
    pub fn min(&self) -> Option<f64> {
        self.data
            .iter()
            .copied()
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.min(v))))
    }

    /// Clear the window.
    pub fn clear(&mut self) {
        self.data.clear();
        self.sum = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_window_basic() {
        let mut window = RollingWindow::new(3);
        assert!(window.is_empty());
        assert!(!window.is_full());

        window.push(1.0);
        window.push(2.0);
        assert_eq!(window.len(), 2);
        assert!(!window.is_full());

        window.push(3.0);
        assert!(window.is_full());
        assert_eq!(window.len(), 3);
    }

    #[test]
    fn test_rolling_window_push_overflow() {
        let mut window = RollingWindow::new(3);
        window.push(1.0);
        window.push(2.0);
        window.push(3.0);

        // This should remove 1.0
        let removed = window.push(4.0);
        assert_eq!(removed, Some(1.0));
        assert_eq!(window.first(), Some(2.0));
        assert_eq!(window.last(), Some(4.0));
    }

    #[test]
    fn test_rolling_window_sum_and_mean() {
        let mut window = RollingWindow::new(4);
        window.push(10.0);
        window.push(20.0);
        window.push(30.0);
        window.push(40.0);

        assert_eq!(window.sum(), 100.0);
        assert_eq!(window.mean(), Some(25.0));

        window.push(50.0); // Removes 10
        assert_eq!(window.sum(), 140.0);
        assert_eq!(window.mean(), Some(35.0));
    }

    #[test]
    fn test_rolling_window_std_dev() {
        let mut window = RollingWindow::new(5);
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            window.push(v);
        }
        // Window now contains [5.0, 5.0, 7.0, 9.0] (only last 5 if capacity is 5)
        // Actually with capacity 5 and 8 pushes: [4.0, 5.0, 5.0, 7.0, 9.0]
        // Mean = 6.0, variance = ((4-6)^2 + (5-6)^2 + (5-6)^2 + (7-6)^2 + (9-6)^2) / 5
        //      = (4 + 1 + 1 + 1 + 9) / 5 = 16/5 = 3.2
        // std_dev = sqrt(3.2) ≈ 1.789

        let std = window.std_dev().unwrap();
        assert!((std - 1.7889).abs() < 0.001);
    }

    #[test]
    fn test_rolling_window_min_max() {
        let mut window = RollingWindow::new(5);
        window.push(3.0);
        window.push(1.0);
        window.push(4.0);
        window.push(1.0);
        window.push(5.0);

        assert_eq!(window.min(), Some(1.0));
        assert_eq!(window.max(), Some(5.0));
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_rolling_window_zero_capacity() {
        RollingWindow::new(0);
    }
}
