//! Common agent state tracking.
//!
//! This module provides shared state management for agents that track
//! position, cash, and trading metrics. Using a shared struct reduces
//! code duplication across agent implementations.
//!
//! ## V3.1 Multi-Symbol Support
//!
//! All positions are tracked per-symbol using `HashMap<Symbol, PositionEntry>`.
//! Methods always require explicit symbol parameters — no implicit "primary symbol".
//!
//! ## P&L Tracking
//!
//! The state tracks realized P&L using weighted average cost basis:
//! - On buy: `new_avg_cost = (old_qty * old_avg + buy_qty * buy_price) / (old_qty + buy_qty)`
//! - On sell: `realized_pnl += (sell_price - avg_cost) * sell_qty`

use std::collections::HashMap;
use types::{Cash, Price, Symbol};

/// Per-symbol position tracking with cost basis.
///
/// # V3.1
/// Extracted from AgentState to enable multi-symbol portfolios.
#[derive(Debug, Clone, Default)]
pub struct PositionEntry {
    /// Current position in shares (positive = long, negative = short).
    pub quantity: i64,
    /// Weighted average cost basis per share (for P&L calculation).
    pub avg_cost: f64,
}

impl PositionEntry {
    /// Create a new position entry with initial quantity and cost.
    pub fn new(quantity: i64, avg_cost: f64) -> Self {
        Self { quantity, avg_cost }
    }

    /// Create an empty position entry.
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Common state shared across agent implementations.
///
/// Agents that track position and cash should embed this struct
/// rather than duplicating the fields. This ensures consistent
/// behavior and makes it easier to add new metrics.
///
/// # V3.1 Multi-Symbol API
///
/// All position operations require explicit symbol:
/// - `position_for(symbol)` — get position for symbol
/// - `set_position(symbol, qty)` — set position for symbol  
/// - `on_buy(symbol, qty, value)` — record buy fill
/// - `on_sell(symbol, qty, value)` — record sell fill
/// - `avg_cost_for(symbol)` — get cost basis for symbol
///
/// Aggregate methods:
/// - `position()` — sum of all positions (for risk checks)
/// - `equity(prices)` — total equity given price map
#[derive(Debug, Clone)]
pub struct AgentState {
    /// Per-symbol positions with cost basis.
    positions: HashMap<Symbol, PositionEntry>,
    /// Current cash balance.
    cash: Cash,
    /// Accumulated realized P&L from closed positions (all symbols).
    realized_pnl: Cash,
    /// Total number of orders placed.
    orders_placed: u64,
    /// Total number of fills received.
    fills_received: u64,
}

impl AgentState {
    /// Create a new agent state with initial cash for given symbols.
    pub fn new(initial_cash: Cash, symbols: &[&str]) -> Self {
        let positions = symbols
            .iter()
            .map(|symbol| (symbol.to_string(), PositionEntry::empty()))
            .collect();
        Self {
            positions,
            cash: initial_cash,
            realized_pnl: Cash::ZERO,
            orders_placed: 0,
            fills_received: 0,
        }
    }

    /// Create a new agent state with owned symbol strings.
    pub fn with_symbols(initial_cash: Cash, symbols: Vec<Symbol>) -> Self {
        let positions = symbols
            .into_iter()
            .map(|symbol| (symbol, PositionEntry::empty()))
            .collect();
        Self {
            positions,
            cash: initial_cash,
            realized_pnl: Cash::ZERO,
            orders_placed: 0,
            fills_received: 0,
        }
    }

    /// Get aggregate position across all symbols.
    /// For single-symbol agents, this equals the position in that symbol.
    pub fn position(&self) -> i64 {
        self.positions.values().map(|e| e.quantity).sum()
    }

    /// Get position for a specific symbol.
    pub fn position_for(&self, symbol: &str) -> i64 {
        self.positions.get(symbol).map(|e| e.quantity).unwrap_or(0)
    }

    /// Get all positions.
    pub fn positions(&self) -> &HashMap<Symbol, PositionEntry> {
        &self.positions
    }

    /// Get symbols this agent tracks.
    pub fn symbols(&self) -> Vec<Symbol> {
        self.positions.keys().cloned().collect()
    }

    /// Set position for a specific symbol (for initial allocation).
    pub fn set_position(&mut self, symbol: &Symbol, position: i64) {
        self.positions
            .entry(symbol.to_string())
            .or_insert_with(PositionEntry::empty)
            .quantity = position;
    }

    /// Get current cash balance.
    pub fn cash(&self) -> Cash {
        self.cash
    }

    /// Get total orders placed.
    pub fn orders_placed(&self) -> u64 {
        self.orders_placed
    }

    /// Get total fills received.
    pub fn fills_received(&self) -> u64 {
        self.fills_received
    }

    /// Get realized P&L (aggregate across all symbols).
    pub fn realized_pnl(&self) -> Cash {
        self.realized_pnl
    }

    /// Compute total P&L (realized + unrealized) given current prices.
    ///
    /// Unrealized P&L = Σ (current_price - avg_cost) * quantity for each position.
    pub fn total_pnl(&self, prices: &HashMap<Symbol, Price>) -> Cash {
        let unrealized: f64 = self
            .positions
            .iter()
            .map(|(symbol, entry)| {
                let current_price = prices
                    .get(symbol)
                    .map(|p| p.to_float())
                    .unwrap_or(entry.avg_cost); // Use cost basis if no price
                (current_price - entry.avg_cost) * entry.quantity as f64
            })
            .sum();
        Cash::from_float(self.realized_pnl.to_float() + unrealized)
    }

    /// Get average cost basis for a specific symbol.
    pub fn avg_cost_for(&self, symbol: &str) -> f64 {
        self.positions
            .get(symbol)
            .map(|e| e.avg_cost)
            .unwrap_or(0.0)
    }

    /// Update state after a buy fill.
    /// Uses weighted average cost basis calculation.
    /// Handles covering short positions with proper P&L calculation.
    pub fn on_buy(&mut self, symbol: &str, quantity: u64, value: Cash) {
        let entry = self
            .positions
            .entry(symbol.to_string())
            .or_insert_with(PositionEntry::empty);

        let buy_price = value.to_float() / quantity as f64;
        let mut remaining_qty = quantity as i64;

        // If we have a short position, buying covers it first
        if entry.quantity < 0 {
            let short_qty = -entry.quantity;
            let qty_to_cover = remaining_qty.min(short_qty);

            // Realized P&L for covering short = (short_sale_price - buy_price) * qty
            // Profit if price dropped since we shorted
            if entry.avg_cost > 0.0 {
                let pnl = (entry.avg_cost - buy_price) * qty_to_cover as f64;
                self.realized_pnl += Cash::from_float(pnl);
            }

            remaining_qty -= qty_to_cover;
            entry.quantity += qty_to_cover;

            // If fully covered, reset cost basis
            if entry.quantity == 0 {
                entry.avg_cost = 0.0;
            }
        }

        // Any remaining quantity opens/adds to long position
        if remaining_qty > 0 {
            let old_long_qty = entry.quantity.max(0) as f64;
            let new_long_qty = old_long_qty + remaining_qty as f64;

            // Update weighted average cost for long position
            entry.avg_cost =
                (old_long_qty * entry.avg_cost + remaining_qty as f64 * buy_price) / new_long_qty;

            entry.quantity += remaining_qty;
        }

        self.cash -= value;
        self.fills_received += 1;
    }

    /// Update state after a sell fill.
    /// Computes realized P&L as (sell_price - avg_cost) * quantity.
    /// Handles opening short positions with proper cost basis tracking.
    pub fn on_sell(&mut self, symbol: &str, quantity: u64, value: Cash) {
        let entry = self
            .positions
            .entry(symbol.to_string())
            .or_insert_with(PositionEntry::empty);

        let sell_price = value.to_float() / quantity as f64;
        let mut remaining_qty = quantity as i64;

        // If we have a long position, selling closes it first
        if entry.quantity > 0 {
            let long_qty = entry.quantity;
            let qty_to_close = remaining_qty.min(long_qty);

            // Realized P&L for closing long = (sell_price - avg_cost) * qty
            if entry.avg_cost > 0.0 {
                let pnl = (sell_price - entry.avg_cost) * qty_to_close as f64;
                self.realized_pnl += Cash::from_float(pnl);
            }

            remaining_qty -= qty_to_close;
            entry.quantity -= qty_to_close;

            // If fully closed, reset cost basis
            if entry.quantity == 0 {
                entry.avg_cost = 0.0;
            }
        }

        // Any remaining quantity opens/adds to short position
        if remaining_qty > 0 {
            let old_short_qty = (-entry.quantity).max(0) as f64;
            let new_short_qty = old_short_qty + remaining_qty as f64;

            // Update weighted average cost for short position (the price we sold at)
            entry.avg_cost = (old_short_qty * entry.avg_cost + remaining_qty as f64 * sell_price)
                / new_short_qty;

            entry.quantity -= remaining_qty;
        }

        self.cash += value;
        self.fills_received += 1;
    }

    /// Increment orders placed counter.
    pub fn record_order(&mut self) {
        self.orders_placed += 1;
    }

    /// Increment orders placed counter by count.
    pub fn record_orders(&mut self, count: u64) {
        self.orders_placed += count;
    }

    /// Compute total equity given prices for all symbols.
    pub fn equity(&self, prices: &HashMap<Symbol, Price>) -> Cash {
        self.positions
            .iter()
            .filter_map(|(symbol, entry)| {
                prices.get(symbol).map(|price| {
                    if entry.quantity >= 0 {
                        *price * types::Quantity(entry.quantity as u64)
                    } else {
                        -(*price * types::Quantity((-entry.quantity) as u64))
                    }
                })
            })
            .fold(self.cash, |acc, val| acc + val)
    }

    /// Compute equity for a single symbol (convenience for single-symbol agents).
    pub fn equity_for(&self, symbol: &str, price: Price) -> Cash {
        let position = self.position_for(symbol);
        let position_value = if position >= 0 {
            price * types::Quantity(position as u64)
        } else {
            -(price * types::Quantity((-position) as u64))
        };
        self.cash + position_value
    }
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            positions: HashMap::new(),
            cash: Cash::ZERO,
            realized_pnl: Cash::ZERO,
            orders_placed: 0,
            fills_received: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_new() {
        let state = AgentState::new(Cash::from_float(10_000.0), &["ACME"]);
        assert_eq!(state.position(), 0);
        assert_eq!(state.position_for("ACME"), 0);
        assert_eq!(state.cash(), Cash::from_float(10_000.0));
        assert_eq!(state.orders_placed(), 0);
        assert_eq!(state.fills_received(), 0);
        assert_eq!(state.realized_pnl(), Cash::ZERO);
        assert_eq!(state.avg_cost_for("ACME"), 0.0);
    }

    #[test]
    fn test_agent_state_on_buy() {
        let mut state = AgentState::new(Cash::from_float(10_000.0), &["ACME"]);
        state.on_buy("ACME", 100, Cash::from_float(1_000.0));
        assert_eq!(state.position_for("ACME"), 100);
        assert_eq!(state.position(), 100);
        assert_eq!(state.cash(), Cash::from_float(9_000.0));
        assert_eq!(state.fills_received(), 1);
        // avg_cost = 1000 / 100 = $10 per share
        assert!((state.avg_cost_for("ACME") - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_agent_state_on_sell() {
        let mut state = AgentState::new(Cash::from_float(10_000.0), &["ACME"]);
        // Buy first to establish cost basis
        state.on_buy("ACME", 100, Cash::from_float(1_000.0)); // $10/share
        state.on_sell("ACME", 50, Cash::from_float(600.0)); // Sell at $12/share
        assert_eq!(state.position_for("ACME"), 50);
        assert_eq!(state.cash(), Cash::from_float(9_600.0)); // 9000 + 600
        assert_eq!(state.fills_received(), 2);
        // Realized P&L = (12 - 10) * 50 = $100
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_agent_state_record_orders() {
        let mut state = AgentState::default();
        state.record_order();
        assert_eq!(state.orders_placed(), 1);
        state.record_orders(3);
        assert_eq!(state.orders_placed(), 4);
    }

    #[test]
    fn test_weighted_average_cost() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);
        // Buy 100 shares at $10
        state.on_buy("ACME", 100, Cash::from_float(1_000.0));
        assert!((state.avg_cost_for("ACME") - 10.0).abs() < 0.001);

        // Buy 100 more shares at $20
        state.on_buy("ACME", 100, Cash::from_float(2_000.0));
        // New avg = (100 * 10 + 100 * 20) / 200 = 3000 / 200 = $15
        assert!((state.avg_cost_for("ACME") - 15.0).abs() < 0.001);
        assert_eq!(state.position_for("ACME"), 200);
    }

    #[test]
    fn test_realized_pnl_profit() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);
        // Buy 100 at $50
        state.on_buy("ACME", 100, Cash::from_float(5_000.0));
        // Sell 100 at $60 (profit of $10/share)
        state.on_sell("ACME", 100, Cash::from_float(6_000.0));
        // Realized P&L = (60 - 50) * 100 = $1000
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 1000.0).abs() < 0.01);
        assert_eq!(state.position_for("ACME"), 0);
    }

    #[test]
    fn test_realized_pnl_loss() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);
        // Buy 100 at $50
        state.on_buy("ACME", 100, Cash::from_float(5_000.0));
        // Sell 100 at $40 (loss of $10/share)
        state.on_sell("ACME", 100, Cash::from_float(4_000.0));
        // Realized P&L = (40 - 50) * 100 = -$1000
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - (-1000.0)).abs() < 0.01);
    }

    #[test]
    fn test_partial_sell_pnl() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);
        // Buy 100 at $10
        state.on_buy("ACME", 100, Cash::from_float(1_000.0));
        // Sell 30 at $15 (profit of $5/share on 30 shares)
        state.on_sell("ACME", 30, Cash::from_float(450.0));
        // Realized P&L = (15 - 10) * 30 = $150
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 150.0).abs() < 0.01);
        assert_eq!(state.position_for("ACME"), 70);
        // avg_cost should remain $10
        assert!((state.avg_cost_for("ACME") - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_multi_symbol_state() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["AAPL", "GOOGL"]);

        // Buy AAPL
        state.on_buy("AAPL", 100, Cash::from_float(15_000.0));
        assert_eq!(state.position_for("AAPL"), 100);
        assert_eq!(state.position_for("GOOGL"), 0);
        assert_eq!(state.position(), 100); // Aggregate

        // Buy GOOGL
        state.on_buy("GOOGL", 50, Cash::from_float(7_500.0));
        assert_eq!(state.position_for("GOOGL"), 50);
        assert_eq!(state.position(), 150); // Aggregate

        // Cash should be reduced
        assert_eq!(state.cash(), Cash::from_float(77_500.0));
    }

    #[test]
    fn test_multi_symbol_pnl() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["AAPL", "GOOGL"]);

        // Buy AAPL at $150
        state.on_buy("AAPL", 100, Cash::from_float(15_000.0));
        // Sell AAPL at $160 (profit $10/share)
        state.on_sell("AAPL", 100, Cash::from_float(16_000.0));

        // Realized P&L = $1000
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_equity() {
        let mut state = AgentState::new(Cash::from_float(50_000.0), &["AAPL", "GOOGL"]);

        state.set_position(&"AAPL".to_string(), 100);
        state.set_position(&"GOOGL".to_string(), 20);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), Price::from_float(150.0));
        prices.insert("GOOGL".to_string(), Price::from_float(100.0));

        // Equity = 50000 + 100*150 + 20*100 = 50000 + 15000 + 2000 = 67000
        let equity = state.equity(&prices);
        assert!((equity.to_float() - 67_000.0).abs() < 0.01);
    }

    #[test]
    fn test_equity_for_single_symbol() {
        let mut state = AgentState::new(Cash::from_float(50_000.0), &["ACME"]);
        state.set_position(&"ACME".to_string(), 100);

        let equity = state.equity_for("ACME", Price::from_float(150.0));
        // Equity = 50000 + 100*150 = 65000
        assert!((equity.to_float() - 65_000.0).abs() < 0.01);
    }

    // ==================== Short Position Tests ====================

    #[test]
    fn test_short_position_open_and_cover_profit() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);

        // Short 100 shares at $50 (selling without owning)
        state.on_sell("ACME", 100, Cash::from_float(5_000.0));
        assert_eq!(state.position_for("ACME"), -100);
        assert_eq!(state.cash(), Cash::from_float(105_000.0)); // Received $5000
        assert!((state.avg_cost_for("ACME") - 50.0).abs() < 0.001); // Cost basis = $50

        // Cover at $40 (buy back) - profit of $10/share
        state.on_buy("ACME", 100, Cash::from_float(4_000.0));
        assert_eq!(state.position_for("ACME"), 0);
        assert_eq!(state.cash(), Cash::from_float(101_000.0)); // 105000 - 4000

        // Realized P&L = (50 - 40) * 100 = $1000 profit
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_short_position_open_and_cover_loss() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);

        // Short 100 shares at $50
        state.on_sell("ACME", 100, Cash::from_float(5_000.0));
        assert_eq!(state.position_for("ACME"), -100);

        // Cover at $60 (price went up) - loss of $10/share
        state.on_buy("ACME", 100, Cash::from_float(6_000.0));
        assert_eq!(state.position_for("ACME"), 0);

        // Realized P&L = (50 - 60) * 100 = -$1000 loss
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - (-1000.0)).abs() < 0.01);
    }

    #[test]
    fn test_short_position_partial_cover() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);

        // Short 100 shares at $50
        state.on_sell("ACME", 100, Cash::from_float(5_000.0));

        // Cover 40 shares at $45
        state.on_buy("ACME", 40, Cash::from_float(1_800.0));
        assert_eq!(state.position_for("ACME"), -60);

        // Realized P&L = (50 - 45) * 40 = $200
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 200.0).abs() < 0.01);

        // avg_cost should remain $50 for remaining short
        assert!((state.avg_cost_for("ACME") - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_short_position_add_to_short() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);

        // Short 100 shares at $50
        state.on_sell("ACME", 100, Cash::from_float(5_000.0));
        assert!((state.avg_cost_for("ACME") - 50.0).abs() < 0.001);

        // Short 100 more shares at $60
        state.on_sell("ACME", 100, Cash::from_float(6_000.0));
        assert_eq!(state.position_for("ACME"), -200);

        // Weighted avg cost = (100 * 50 + 100 * 60) / 200 = 11000 / 200 = $55
        assert!((state.avg_cost_for("ACME") - 55.0).abs() < 0.001);

        // No realized P&L yet
        assert_eq!(state.realized_pnl().to_float(), 0.0);
    }

    #[test]
    fn test_short_unrealized_pnl() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);

        // Short 100 shares at $50
        state.on_sell("ACME", 100, Cash::from_float(5_000.0));

        let mut prices = HashMap::new();
        prices.insert("ACME".to_string(), Price::from_float(40.0));

        // Unrealized P&L = (current - avg_cost) * qty = (40 - 50) * (-100) = $1000 profit
        let total_pnl = state.total_pnl(&prices).to_float();
        assert!((total_pnl - 1000.0).abs() < 0.01);

        // Price goes up (loss for short)
        prices.insert("ACME".to_string(), Price::from_float(60.0));
        // Unrealized P&L = (60 - 50) * (-100) = -$1000 loss
        let total_pnl = state.total_pnl(&prices).to_float();
        assert!((total_pnl - (-1000.0)).abs() < 0.01);
    }

    #[test]
    fn test_long_to_short_transition() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);

        // Buy 50 shares at $100
        state.on_buy("ACME", 50, Cash::from_float(5_000.0));
        assert_eq!(state.position_for("ACME"), 50);

        // Sell 80 shares at $120 (close 50 long + open 30 short)
        state.on_sell("ACME", 80, Cash::from_float(9_600.0));
        assert_eq!(state.position_for("ACME"), -30);

        // Realized P&L from closing long = (120 - 100) * 50 = $1000
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 1000.0).abs() < 0.01);

        // avg_cost should now be $120 (the short sale price)
        assert!((state.avg_cost_for("ACME") - 120.0).abs() < 0.001);
    }

    #[test]
    fn test_short_to_long_transition() {
        let mut state = AgentState::new(Cash::from_float(100_000.0), &["ACME"]);

        // Short 50 shares at $100
        state.on_sell("ACME", 50, Cash::from_float(5_000.0));
        assert_eq!(state.position_for("ACME"), -50);

        // Buy 80 shares at $90 (cover 50 short + open 30 long)
        state.on_buy("ACME", 80, Cash::from_float(7_200.0));
        assert_eq!(state.position_for("ACME"), 30);

        // Realized P&L from covering short = (100 - 90) * 50 = $500
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 500.0).abs() < 0.01);

        // avg_cost should now be $90 (the buy price for new long)
        assert!((state.avg_cost_for("ACME") - 90.0).abs() < 0.001);
    }
}
