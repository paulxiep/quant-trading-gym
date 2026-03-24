//! Per-agent risk tracking.
//!
//! This module provides `AgentRiskTracker` for tracking equity history
//! and computing rolling risk metrics per agent. Designed to be used
//! by the simulation crate to provide real-time risk monitoring.
//!
//! # Design Principles
//! - **Declarative**: Metrics are computed on-demand from equity history
//! - **Modular**: Self-contained, no knowledge of simulation internals
//! - **SoC**: Only handles equity tracking and risk computation

use std::collections::HashMap;

use crate::{risk, rolling::RollingWindow, stats};
use types::AgentId;

/// Risk metrics snapshot for a single agent.
#[derive(Debug, Clone, Copy, Default)]
pub struct AgentRiskSnapshot {
    /// Sharpe ratio (annualized, assuming daily periods).
    pub sharpe: Option<f64>,
    /// Sortino ratio (annualized, assuming daily periods).
    pub sortino: Option<f64>,
    /// Maximum drawdown as percentage (0.0 to 1.0).
    pub max_drawdown: f64,
    /// Value at Risk at 95% confidence.
    pub var_95: Option<f64>,
    /// Current equity value.
    pub equity: f64,
    /// Total return as percentage.
    pub total_return: f64,
    /// Annualized volatility.
    pub volatility: Option<f64>,
}

/// Configuration for risk tracker.
#[derive(Debug, Clone)]
pub struct RiskTrackerConfig {
    /// Number of equity observations to keep for rolling calculations.
    pub window_size: usize,
    /// Minimum observations required before computing metrics.
    pub min_observations: usize,
    /// Risk-free rate per period (default 0.0).
    pub risk_free_rate: f64,
    /// Periods per year for annualization (default 252 for daily).
    pub periods_per_year: f64,
}

impl Default for RiskTrackerConfig {
    fn default() -> Self {
        Self {
            window_size: 500,        // Keep ~500 ticks of history
            min_observations: 20,    // Need at least 20 points
            risk_free_rate: 0.0,     // Assume zero risk-free rate
            periods_per_year: 252.0, // Trading days
        }
    }
}

/// Tracks equity history and computes rolling risk metrics per agent.
///
/// # Usage
/// ```ignore
/// use quant::tracker::{AgentRiskTracker, RiskTrackerConfig};
/// use types::AgentId;
///
/// let mut tracker = AgentRiskTracker::new(RiskTrackerConfig::default());
///
/// // Each tick, record equity for each agent
/// tracker.record_equity(AgentId(1), 10500.0);
/// tracker.record_equity(AgentId(2), 9800.0);
///
/// // Get risk metrics
/// let metrics = tracker.compute_metrics(AgentId(1));
/// ```
pub struct AgentRiskTracker {
    /// Configuration.
    config: RiskTrackerConfig,
    /// Equity history per agent (using rolling window).
    equity_history: HashMap<AgentId, RollingWindow>,
}

impl AgentRiskTracker {
    /// Create a new risk tracker with the given configuration.
    pub fn new(config: RiskTrackerConfig) -> Self {
        Self {
            config,
            equity_history: HashMap::new(),
        }
    }

    /// Create a risk tracker with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RiskTrackerConfig::default())
    }

    /// Record an equity observation for an agent.
    pub fn record_equity(&mut self, agent_id: AgentId, equity: f64) {
        let window = self
            .equity_history
            .entry(agent_id)
            .or_insert_with(|| RollingWindow::new(self.config.window_size));
        window.push(equity);
    }

    /// Get the number of observations for an agent.
    pub fn observation_count(&self, agent_id: AgentId) -> usize {
        self.equity_history
            .get(&agent_id)
            .map(|w| w.len())
            .unwrap_or(0)
    }

    /// Compute risk metrics for a single agent.
    pub fn compute_metrics(&self, agent_id: AgentId) -> AgentRiskSnapshot {
        let Some(window) = self.equity_history.get(&agent_id) else {
            return AgentRiskSnapshot::default();
        };

        let equity_curve: Vec<f64> = window.iter().collect();

        if equity_curve.is_empty() {
            return AgentRiskSnapshot::default();
        }

        let current_equity = *equity_curve.last().unwrap();
        let initial_equity = equity_curve[0];

        let total_return = if initial_equity > 0.0 {
            (current_equity - initial_equity) / initial_equity
        } else {
            0.0
        };

        // Calculate returns for risk metrics
        let returns = stats::returns(&equity_curve);

        // Need minimum observations for meaningful metrics
        let has_enough_data = returns.len() >= self.config.min_observations;

        let sharpe = if has_enough_data {
            risk::sharpe_ratio(
                &returns,
                self.config.risk_free_rate,
                self.config.periods_per_year,
            )
        } else {
            None
        };

        let sortino = if has_enough_data {
            risk::sortino_ratio(
                &returns,
                self.config.risk_free_rate,
                self.config.periods_per_year,
            )
        } else {
            None
        };

        let var_95 = if has_enough_data {
            risk::historical_var(&returns, 0.95)
        } else {
            None
        };

        let volatility = if has_enough_data {
            risk::annualized_volatility(&returns, self.config.periods_per_year)
        } else {
            None
        };

        let max_drawdown = risk::max_drawdown(&equity_curve);

        AgentRiskSnapshot {
            sharpe,
            sortino,
            max_drawdown,
            var_95,
            equity: current_equity,
            total_return,
            volatility,
        }
    }

    /// Compute risk metrics for all tracked agents.
    pub fn compute_all_metrics(&self) -> HashMap<AgentId, AgentRiskSnapshot> {
        self.equity_history
            .keys()
            .map(|&id| (id, self.compute_metrics(id)))
            .collect()
    }

    /// Clear all tracking data.
    pub fn reset(&mut self) {
        self.equity_history.clear();
    }

    /// Remove tracking for a specific agent.
    pub fn remove_agent(&mut self, agent_id: AgentId) {
        self.equity_history.remove(&agent_id);
    }

    /// Get the equity history for an agent (for debugging/testing).
    pub fn equity_history(&self, agent_id: AgentId) -> Option<Vec<f64>> {
        self.equity_history
            .get(&agent_id)
            .map(|w| w.iter().collect())
    }
}

impl Default for AgentRiskTracker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_tracker_new() {
        let tracker = AgentRiskTracker::with_defaults();
        assert_eq!(tracker.observation_count(AgentId(1)), 0);
    }

    #[test]
    fn test_record_equity() {
        let mut tracker = AgentRiskTracker::with_defaults();
        tracker.record_equity(AgentId(1), 10000.0);
        tracker.record_equity(AgentId(1), 10100.0);
        tracker.record_equity(AgentId(1), 10050.0);

        assert_eq!(tracker.observation_count(AgentId(1)), 3);

        let history = tracker.equity_history(AgentId(1)).unwrap();
        assert_eq!(history, vec![10000.0, 10100.0, 10050.0]);
    }

    #[test]
    fn test_compute_metrics_empty() {
        let tracker = AgentRiskTracker::with_defaults();
        let metrics = tracker.compute_metrics(AgentId(1));

        assert_eq!(metrics.equity, 0.0);
        assert_eq!(metrics.max_drawdown, 0.0);
        assert!(metrics.sharpe.is_none());
    }

    #[test]
    fn test_compute_metrics_basic() {
        let mut tracker = AgentRiskTracker::with_defaults();
        let agent = AgentId(1);

        // Generate simple equity curve
        for i in 0..30 {
            let equity = 10000.0 + (i as f64 * 10.0);
            tracker.record_equity(agent, equity);
        }

        let metrics = tracker.compute_metrics(agent);

        // Should have positive equity
        assert!(metrics.equity > 10000.0);
        // No drawdown in monotonically increasing curve
        assert_eq!(metrics.max_drawdown, 0.0);
        // Positive return
        assert!(metrics.total_return > 0.0);
    }

    #[test]
    fn test_compute_metrics_with_drawdown() {
        let mut tracker = AgentRiskTracker::new(RiskTrackerConfig {
            min_observations: 5,
            ..Default::default()
        });
        let agent = AgentId(1);

        // Equity with drawdown: 100 -> 120 -> 100
        let equities = [
            100.0, 105.0, 110.0, 115.0, 120.0, // Up
            115.0, 110.0, 105.0, 100.0, // Down
            105.0, 110.0, // Recovery
        ];

        equities
            .iter()
            .for_each(|&equity| tracker.record_equity(agent, equity));

        let metrics = tracker.compute_metrics(agent);

        // Max drawdown from 120 to 100 = 16.67%
        let expected_dd = (120.0 - 100.0) / 120.0;
        assert!((metrics.max_drawdown - expected_dd).abs() < 0.01);

        // Final equity
        assert_eq!(metrics.equity, 110.0);
    }

    #[test]
    fn test_compute_all_metrics() {
        let mut tracker = AgentRiskTracker::with_defaults();

        // Track two agents
        tracker.record_equity(AgentId(1), 10000.0);
        tracker.record_equity(AgentId(1), 10100.0);
        tracker.record_equity(AgentId(2), 5000.0);
        tracker.record_equity(AgentId(2), 4900.0);

        let all_metrics = tracker.compute_all_metrics();

        assert_eq!(all_metrics.len(), 2);
        assert!(all_metrics.contains_key(&AgentId(1)));
        assert!(all_metrics.contains_key(&AgentId(2)));

        // Agent 1 has positive return
        assert!(all_metrics[&AgentId(1)].total_return > 0.0);
        // Agent 2 has negative return
        assert!(all_metrics[&AgentId(2)].total_return < 0.0);
    }

    #[test]
    fn test_reset_and_remove() {
        let mut tracker = AgentRiskTracker::with_defaults();

        tracker.record_equity(AgentId(1), 10000.0);
        tracker.record_equity(AgentId(2), 5000.0);

        tracker.remove_agent(AgentId(1));
        assert_eq!(tracker.observation_count(AgentId(1)), 0);
        assert_eq!(tracker.observation_count(AgentId(2)), 1);

        tracker.reset();
        assert_eq!(tracker.observation_count(AgentId(2)), 0);
    }

    #[test]
    fn test_rolling_window_limit() {
        let mut tracker = AgentRiskTracker::new(RiskTrackerConfig {
            window_size: 5,
            ..Default::default()
        });
        let agent = AgentId(1);

        // Add more than window size
        for i in 0..10 {
            tracker.record_equity(agent, i as f64 * 100.0);
        }

        // Should only keep last 5
        assert_eq!(tracker.observation_count(agent), 5);

        let history = tracker.equity_history(agent).unwrap();
        assert_eq!(history, vec![500.0, 600.0, 700.0, 800.0, 900.0]);
    }
}
