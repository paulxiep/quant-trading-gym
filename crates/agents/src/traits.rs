//! Agent trait and type definitions for the trading simulation.
//!
//! This module defines the core `Agent` trait that all trading agents must implement,
//! as well as the `StrategyContext` they receive each tick.
//!
//! # V3.1 Multi-Symbol Support
//!
//! Agents can now track positions across multiple symbols:
//! - `positions()` returns a map of symbol → position entry
//! - `watched_symbols()` declares symbols for Tier 2 wake conditions
//! - Backward-compatible: `position()` returns aggregate across all symbols
//!
//! # V2.3 Changes
//!
//! The `on_tick` method receives `StrategyContext<'_>` which provides
//! multi-symbol access via the `MarketView` trait.
//!
//! # State Management
//!
//! All agents must provide access to their `AgentState` via the `state()` method.
//! This enables automatic tracking of position, cash, and realized P&L.

use std::collections::HashMap;

use crate::state::{AgentState, PositionEntry};
use types::{AgentId, Order, OrderId, Symbol};

use crate::StrategyContext;

/// Result of an agent's decision each tick.
///
/// An agent may submit zero, one, or multiple orders each tick,
/// and optionally cancel existing orders.
#[derive(Debug, Clone, Default)]
pub struct AgentAction {
    /// Orders to submit this tick.
    pub orders: Vec<Order>,
    /// Order IDs to cancel this tick.
    pub cancellations: Vec<OrderId>,
}

impl AgentAction {
    /// Create an empty action (no orders, no cancellations).
    pub fn none() -> Self {
        Self {
            orders: vec![],
            cancellations: vec![],
        }
    }

    /// Create an action with a single order.
    pub fn single(order: Order) -> Self {
        Self {
            orders: vec![order],
            cancellations: vec![],
        }
    }

    /// Create an action with multiple orders.
    pub fn multiple(orders: Vec<Order>) -> Self {
        Self {
            orders,
            cancellations: vec![],
        }
    }

    /// Create an action that cancels orders and places new ones.
    pub fn cancel_and_replace(cancellations: Vec<OrderId>, orders: Vec<Order>) -> Self {
        Self {
            orders,
            cancellations,
        }
    }
}

/// The core trait that all trading agents must implement.
///
/// Agents are called once per tick with a `StrategyContext` providing
/// access to market state, indicators, and historical data.
///
/// # V2.3 Changes
///
/// The `on_tick` method now receives `StrategyContext<'_>` instead of `MarketData`.
/// This enables multi-symbol access through the `MarketView` trait.
///
/// # Lifetimes
///
/// The `StrategyContext` parameter borrows from the simulation state, so agents
/// cannot store references to it. They should extract any needed data during
/// the `on_tick` call.
///
/// # Example
/// ```ignore
/// struct SimpleAgent {
///     id: AgentId,
///     symbol: String,
/// }
///
/// impl Agent for SimpleAgent {
///     fn id(&self) -> AgentId { self.id }
///
///     fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
///         // Access market data for our symbol
///         let mid = ctx.mid_price(&self.symbol);
///         let rsi = ctx.get_indicator(&self.symbol, IndicatorType::Rsi(14));
///         AgentAction::none()
///     }
/// }
/// ```
pub trait Agent: Send {
    /// Get the unique identifier for this agent.
    fn id(&self) -> AgentId;

    /// Called each simulation tick with the current market state.
    ///
    /// The agent should analyze the context and return any orders
    /// it wishes to submit.
    ///
    /// # Arguments
    /// * `ctx` - Read-only context with market state, indicators, and historical data
    ///
    /// # Returns
    /// An `AgentAction` containing zero or more orders to submit
    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction;

    /// Called when one of this agent's orders is filled (fully or partially).
    ///
    /// This allows agents to update their internal state when trades occur.
    /// Default implementation does nothing.
    ///
    /// # Arguments
    /// * `trade` - The trade that occurred involving this agent's order
    fn on_fill(&mut self, _trade: &types::Trade) {
        // Default: no-op
    }

    /// Called when a limit order is placed on the order book.
    ///
    /// This allows agents to track their resting orders for later cancellation.
    /// The order_id is assigned by the simulation and can be used to cancel.
    ///
    /// # Arguments
    /// * `order_id` - The assigned order ID for the resting order
    /// * `order` - The original order (now resting on the book)
    fn on_order_resting(&mut self, _order_id: OrderId, _order: &types::Order) {
        // Default: no-op
    }

    /// Get a human-readable name for this agent (for logging/debugging).
    fn name(&self) -> &str {
        "Agent"
    }

    /// Whether this agent is a market maker (exempt from short position limits).
    /// Market makers need flexibility to provide two-sided liquidity.
    fn is_market_maker(&self) -> bool {
        false
    }

    /// Whether this agent is an ML-based agent (tree models, neural nets, etc.).
    /// Used for leaderboard sorting (ML agents shown at top).
    fn is_ml_agent(&self) -> bool {
        false
    }

    /// Get a reference to the agent's state.
    /// Required for all agents - enables automatic position/cash/P&L tracking.
    fn state(&self) -> &AgentState;

    /// Get the agent's current aggregate position (shares held across all symbols).
    /// Positive = long, negative = short, zero = flat.
    fn position(&self) -> i64 {
        self.state().position()
    }

    /// Get position for a specific symbol.
    fn position_for(&self, symbol: &str) -> i64 {
        self.state().position_for(symbol)
    }

    /// Get all positions as a map of symbol → position entry.
    ///
    /// For single-symbol agents, this returns a map with one entry.
    /// For multi-symbol agents, this returns all tracked positions.
    fn positions(&self) -> &HashMap<Symbol, PositionEntry> {
        self.state().positions()
    }

    /// Get symbols this agent watches for Tier 2 wake conditions.
    ///
    /// Used by `WakeConditionIndex` to register price/event subscriptions.
    fn watched_symbols(&self) -> Vec<Symbol> {
        self.state().symbols()
    }

    /// Get the agent's current cash balance.
    fn cash(&self) -> types::Cash {
        self.state().cash()
    }

    /// Get the agent's realized P&L.
    fn realized_pnl(&self) -> types::Cash {
        self.state().realized_pnl()
    }

    /// Compute the agent's total equity given prices for all symbols.
    fn equity(&self, prices: &HashMap<Symbol, types::Price>) -> types::Cash {
        self.state().equity(prices)
    }

    /// Compute the agent's equity for a single symbol (convenience).
    fn equity_for(&self, symbol: &str, price: types::Price) -> types::Cash {
        self.state().equity_for(symbol, price)
    }

    /// Whether this agent uses reactive/event-driven wake conditions (Tier 2).
    ///
    /// Reactive agents are NOT called every tick via `on_tick()`. Instead, they
    /// register wake conditions and are only invoked when conditions trigger.
    /// Override this to return `true` for T2 agents.
    fn is_reactive(&self) -> bool {
        false
    }

    /// Get initial wake conditions for registration with WakeConditionIndex.
    ///
    /// Called once when agent is added to simulation. Override for T2 agents.
    /// T1 agents return empty (they're called every tick via on_tick).
    fn initial_wake_conditions(&self, _current_tick: types::Tick) -> Vec<crate::WakeCondition> {
        Vec::new()
    }

    /// Get wake conditions to register after a fill (for exit strategies).
    ///
    /// Called after on_fill() for T2 agents. Exit strategies like StopLoss/TakeProfit
    /// compute absolute thresholds from cost_basis at fill time.
    fn fill_wake_conditions(&self) -> Vec<crate::WakeCondition> {
        Vec::new()
    }

    /// Get ALL wake conditions that should currently be active based on agent state.
    ///
    /// Called after a trigger fires to restore correct conditions.
    /// Returns the complete set of conditions that should be registered.
    fn current_wake_conditions(&self) -> Vec<crate::WakeCondition> {
        Vec::new()
    }

    /// Generate condition updates after a fill to maintain wake index consistency.
    ///
    /// Called after on_fill() for T2 agents. Returns conditions to add/remove:
    /// - Remove entry conditions when at max capacity or out of cash
    /// - Add exit conditions when opening a position (was 0, now > 0)
    /// - Remove exit conditions when closing a position (was > 0, now 0)
    /// - Re-add entry conditions when position closed and has capacity
    ///
    /// # Arguments
    /// * `position_before` - Agent's position before the fill
    fn post_fill_condition_update(
        &self,
        _position_before: i64,
    ) -> Option<crate::tiers::ConditionUpdate> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quant::IndicatorSnapshot;
    use sim_core::SingleSymbolMarket;
    use std::collections::HashMap;
    use types::{OrderId, Price, Quantity};

    #[test]
    fn test_agent_action_none() {
        let action = AgentAction::none();
        assert!(action.orders.is_empty());
    }

    #[test]
    fn test_agent_action_single() {
        let order = Order::market(
            AgentId(1),
            "AAPL",
            types::OrderSide::Buy,
            types::Quantity(100),
        );
        let action = AgentAction::single(order.clone());
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].agent_id, AgentId(1));
    }

    #[test]
    fn test_strategy_context_integration() {
        // Create a minimal order book
        let mut book = sim_core::OrderBook::new("TEST");
        let mut bid = Order::limit(
            AgentId(1),
            "TEST",
            types::OrderSide::Buy,
            Price::from_float(99.0),
            Quantity(100),
        );
        bid.id = OrderId(1);
        let mut ask = Order::limit(
            AgentId(2),
            "TEST",
            types::OrderSide::Sell,
            Price::from_float(101.0),
            Quantity(100),
        );
        ask.id = OrderId(2);
        book.add_order(bid).unwrap();
        book.add_order(ask).unwrap();

        // Create context
        let market = SingleSymbolMarket::new(&book);
        let candles = HashMap::new();
        let indicators = IndicatorSnapshot::new(100);
        let recent_trades = HashMap::new();
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();
        let ctx = StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        // Test access
        let symbol = "TEST".to_string();
        assert_eq!(ctx.mid_price(&symbol), Some(Price::from_float(100.0)));
        assert_eq!(ctx.best_bid(&symbol), Some(Price::from_float(99.0)));
        assert_eq!(ctx.best_ask(&symbol), Some(Price::from_float(101.0)));
    }
}
