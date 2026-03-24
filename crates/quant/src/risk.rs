//! Risk metrics and calculations.
//!
//! This module provides risk assessment tools including VaR, Sharpe ratio,
//! drawdown analysis, and other portfolio risk metrics.
//!
//! Note: Full implementation planned for V1.2.

use crate::stats;

/// Risk metrics for a portfolio or strategy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RiskMetrics {
    /// Value at Risk at 95% confidence (as percentage loss).
    pub var_95: f64,
    /// Value at Risk at 99% confidence (as percentage loss).
    pub var_99: f64,
    /// Sharpe ratio (excess return / volatility).
    pub sharpe: f64,
    /// Sortino ratio (excess return / downside deviation).
    pub sortino: f64,
    /// Maximum drawdown as percentage.
    pub max_drawdown: f64,
    /// Annualized volatility.
    pub volatility: f64,
}

/// Calculate maximum drawdown from an equity curve.
///
/// Returns the largest peak-to-trough decline as a percentage.
pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let initial_peak = equity_curve[0];
    equity_curve
        .iter()
        .fold((0.0_f64, initial_peak), |(max_dd, peak), &value| {
            let new_peak = peak.max(value);
            let dd = if new_peak > 0.0 {
                (new_peak - value) / new_peak
            } else {
                0.0
            };
            (max_dd.max(dd), new_peak)
        })
        .0
}

/// Calculate Sharpe ratio from returns.
///
/// # Arguments
/// * `returns` - Periodic returns
/// * `risk_free_rate` - Risk-free rate for the same period (e.g., daily)
/// * `periods_per_year` - Number of periods in a year (252 for daily, 12 for monthly)
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> Option<f64> {
    if returns.len() < 2 {
        return None;
    }

    let mean_return = stats::mean(returns)?;
    let std = stats::sample_std_dev(returns)?;

    if std == 0.0 {
        return None;
    }

    let excess_return = mean_return - risk_free_rate;
    let annualization = periods_per_year.sqrt();

    Some((excess_return / std) * annualization)
}

/// Calculate Sortino ratio (uses only downside deviation).
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> Option<f64> {
    if returns.len() < 2 {
        return None;
    }

    let mean_return = stats::mean(returns)?;
    let downside_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < risk_free_rate)
        .map(|&r| (r - risk_free_rate).powi(2))
        .collect();

    if downside_returns.is_empty() {
        return Some(f64::INFINITY); // No downside deviation
    }

    let downside_dev = (stats::mean(&downside_returns)?).sqrt();
    if downside_dev == 0.0 {
        return None;
    }

    let excess_return = mean_return - risk_free_rate;
    let annualization = periods_per_year.sqrt();

    Some((excess_return / downside_dev) * annualization)
}

/// Calculate historical Value at Risk using percentile method.
///
/// Returns the loss at the specified confidence level (as positive number).
pub fn historical_var(returns: &[f64], confidence: f64) -> Option<f64> {
    if returns.is_empty() || !(0.0..1.0).contains(&confidence) {
        return None;
    }

    let mut sorted: Vec<f64> = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = 1.0 - confidence;
    let var = stats::percentile(&sorted, percentile)?;

    // Return as positive value (loss)
    Some(-var.min(0.0))
}

/// Calculate annualized volatility from returns.
pub fn annualized_volatility(returns: &[f64], periods_per_year: f64) -> Option<f64> {
    let std = stats::sample_std_dev(returns)?;
    Some(std * periods_per_year.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown() {
        let equity = [100.0, 110.0, 105.0, 120.0, 100.0, 115.0];
        let dd = max_drawdown(&equity);
        // Max drawdown is from 120 to 100 = 16.67%
        assert!((dd - (20.0 / 120.0)).abs() < 0.0001);
    }

    #[test]
    fn test_max_drawdown_no_drawdown() {
        let equity = [100.0, 110.0, 120.0, 130.0];
        assert_eq!(max_drawdown(&equity), 0.0);
    }

    #[test]
    fn test_sharpe_ratio() {
        // 12 months of returns averaging 1% with 2% std
        let returns = [
            0.01, 0.02, -0.01, 0.015, 0.005, 0.02, -0.005, 0.01, 0.015, 0.02, 0.005, 0.01,
        ];
        let sharpe = sharpe_ratio(&returns, 0.0, 12.0);
        assert!(sharpe.is_some());
        assert!(sharpe.unwrap() > 0.0);
    }

    #[test]
    fn test_historical_var() {
        let returns = [
            -0.05, -0.02, 0.01, 0.02, -0.01, 0.03, -0.03, 0.02, 0.01, -0.02,
        ];
        let var95 = historical_var(&returns, 0.95);
        assert!(var95.is_some());
        assert!(var95.unwrap() > 0.0);
    }
}
