//! Agent execution coordinator trait.
//!
//! Defines the interface for managing agent execution across tiers.

use std::collections::HashMap;

use agents::{StrategyContext, WakeCondition};
use smallvec::SmallVec;
use types::{AgentId, Cash, Price, Symbol, Tick, Trade};

/// Agent action with captured state for order validation.
///
/// Tuple of (agent_id, action, per-symbol positions, cash, is_market_maker).
pub type AgentActionWithState = (
    AgentId,
    agents::AgentAction,
    HashMap<Symbol, i64>,
    Cash,
    bool,
);

/// Summary of an agent's current state.
#[derive(Debug, Clone)]
pub struct AgentSummary {
    /// Agent ID.
    pub id: AgentId,
    /// Agent display name.
    pub name: String,
    /// Per-symbol positions (positive = long, negative = short).
    pub positions: HashMap<Symbol, i64>,
    /// Current cash balance.
    pub cash: Cash,
    /// Total realized + unrealized P&L.
    pub total_pnl: Cash,
    /// Whether this is a market maker.
    pub is_market_maker: bool,
    /// Whether this is an ML agent.
    pub is_ml_agent: bool,
}

/// Coordinates agent execution across tiers.
///
/// This trait abstracts agent management, enabling:
/// - Testing strategies with mock coordinators
/// - Replay of recorded agent actions
/// - Separation of agent complexity from tick orchestration
pub trait AgentExecutionCoordinator {
    /// Add an agent to the simulation.
    fn add_agent(&mut self, agent: Box<dyn agents::Agent>, initial_tick: Tick);

    /// Get the total number of agents.
    fn agent_count(&self) -> usize;

    /// Determine which agents should be called this tick.
    ///
    /// Returns (indices to call, map of triggered T2 agents with their conditions).
    fn compute_agents_to_call(
        &mut self,
        tick: Tick,
        prices: &[(Symbol, Price)],
        news_symbols: &[Symbol],
    ) -> (Vec<usize>, HashMap<AgentId, SmallVec<[WakeCondition; 2]>>);

    /// Collect actions from the specified agent indices.
    fn collect_actions(
        &self,
        indices: &[usize],
        ctx: &StrategyContext<'_>,
        force_sequential: bool,
    ) -> Vec<AgentActionWithState>;

    /// Build a cache of all agent positions.
    fn build_position_cache(
        &self,
        force_sequential: bool,
    ) -> HashMap<AgentId, HashMap<Symbol, i64>>;

    /// Notify agents of trade fills.
    fn notify_fills(&mut self, fills: &[(AgentId, Trade, i64)], force_sequential: bool);

    /// Restore wake conditions for triggered T2 agents.
    fn restore_wake_conditions(
        &mut self,
        triggered: &HashMap<AgentId, SmallVec<[WakeCondition; 2]>>,
        force_sequential: bool,
    );

    /// Get summaries of all agents.
    fn agent_summaries(
        &self,
        prices: &HashMap<Symbol, Price>,
        force_sequential: bool,
    ) -> Vec<AgentSummary>;
}
