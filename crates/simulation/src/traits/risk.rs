//! Risk tracking and position validation traits.

use std::collections::HashMap;

use agents::BorrowLedger;
use quant::AgentRiskSnapshot;
use types::{AgentId, Cash, Order, Quantity, RiskViolation, Symbol};

/// Tracks position limits and borrow state.
///
/// Provides order validation and position tracking for risk management.
/// Requires `Sync` for parallel order validation.
pub trait PositionTracker: Sync {
    /// Get reference to the borrow ledger.
    fn borrow_ledger(&self) -> &BorrowLedger;

    /// Get total shares held for a specific symbol.
    fn total_shares_held_for(&self, symbol: &Symbol) -> Quantity;

    /// Get total shares held for all symbols.
    fn all_total_shares(&self) -> &HashMap<Symbol, Quantity>;

    /// Validate an order against position limits.
    fn validate_order(
        &self,
        order: &Order,
        agent_position: i64,
        agent_cash: Cash,
        is_market_maker: bool,
        enforce_limits: bool,
    ) -> Result<(), RiskViolation>;
}

/// Tracks agent risk metrics over time.
pub trait RiskTracker {
    /// Record an agent's current equity for risk calculations.
    fn record_equity(&mut self, agent_id: AgentId, equity: f64);

    /// Compute risk metrics for all agents.
    fn compute_all_metrics(&self) -> HashMap<AgentId, AgentRiskSnapshot>;

    /// Compute risk metrics for a specific agent.
    fn compute_metrics(&self, agent_id: AgentId) -> AgentRiskSnapshot;
}
