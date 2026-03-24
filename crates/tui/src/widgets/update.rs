//! SimUpdate message type for TUI updates.
//!
//! This module defines the data structure sent from the simulation thread
//! to the TUI thread via channels.
//!
//! # V2.3 Multi-Symbol Support
//!
//! The update now supports multiple symbols with per-symbol data:
//! - `symbols`: List of tradeable symbols
//! - `selected_symbol`: Index of currently selected symbol for display
//! - `price_history`, `bids`, `asks`, `last_price`: Per-symbol data

use std::collections::HashMap;

use types::{BookLevel, Cash, Price, Symbol, Tick, Trade};

use crate::widgets::RiskInfo;
use serde::{Deserialize, Serialize};

/// Agent state summary for TUI display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Agent display name.
    pub name: String,
    /// Positions per symbol (positive = long, negative = short).
    pub positions: HashMap<Symbol, i64>,
    /// Total P&L (realized + unrealized).
    pub total_pnl: Cash,
    /// Current cash balance.
    pub cash: Cash,
    /// Whether this is a market maker (for sorting to bottom).
    pub is_market_maker: bool,
    /// Whether this is an ML agent (for sorting to top).
    pub is_ml_agent: bool,
    /// Current equity (cash + position value) for sorting.
    pub equity: f64,
}

impl AgentInfo {
    /// Get the net position across all symbols.
    pub fn net_position(&self) -> i64 {
        self.positions.values().sum()
    }

    /// Get position for a specific symbol.
    pub fn position(&self, symbol: &Symbol) -> i64 {
        self.positions.get(symbol).copied().unwrap_or(0)
    }
}

/// Update message sent from simulation to TUI.
///
/// Contains all data needed to render a single frame.
/// Designed for efficient channel transmission without
/// requiring the TUI to understand simulation internals.
///
/// # Multi-Symbol Support (V2.3)
///
/// Market data is now keyed by symbol. Use `selected_symbol` index
/// to determine which symbol to display, or show overlay for all.
///
/// # Serialization (V3.6)
///
/// Implements Serialize/Deserialize for network transmission
/// (WebSocket hooks, remote TUI clients).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimUpdate {
    /// Current simulation tick.
    pub tick: Tick,

    // ─────────────────────────────────────────────────────────────────────────
    // Multi-Symbol Data (V2.3)
    // ─────────────────────────────────────────────────────────────────────────
    /// List of tradeable symbols.
    pub symbols: Vec<Symbol>,
    /// Currently selected symbol index (for TUI display).
    pub selected_symbol: usize,
    /// Price history per symbol (most recent last).
    pub price_history: HashMap<Symbol, Vec<f64>>,
    /// Bid levels per symbol (highest first).
    pub bids: HashMap<Symbol, Vec<BookLevel>>,
    /// Ask levels per symbol (lowest first).
    pub asks: HashMap<Symbol, Vec<BookLevel>>,
    /// Last trade price per symbol.
    pub last_price: HashMap<Symbol, Price>,
    /// Latest trades this tick (may be empty).
    pub trades: Vec<Trade>,

    // ─────────────────────────────────────────────────────────────────────────
    // Portfolio-Level Data (unchanged)
    // ─────────────────────────────────────────────────────────────────────────
    /// Agent summaries for P&L table.
    pub agents: Vec<AgentInfo>,
    /// Number of Tier 1 agents.
    pub tier1_count: usize,
    /// Number of Tier 2 reactive agents (V3.2).
    pub tier2_count: usize,
    /// Number of Tier 3 background pool agents (V3.4).
    pub tier3_count: usize,
    /// Total trades executed.
    pub total_trades: u64,
    /// Total orders submitted.
    pub total_orders: u64,
    /// Agents called this tick (V3.2 debug).
    pub agents_called: usize,
    /// T2 agents triggered this tick (V3.2 debug).
    pub t2_triggered: usize,
    /// T3 background pool orders this tick (V3.4).
    pub t3_orders: usize,
    /// Simulation is complete.
    pub finished: bool,
    /// Per-agent risk metrics.
    pub risk_metrics: Vec<RiskInfo>,
}

impl SimUpdate {
    /// Create a "simulation finished" message.
    pub fn finished(tick: Tick, total_trades: u64) -> Self {
        Self {
            tick,
            total_trades,
            finished: true,
            ..Default::default()
        }
    }

    /// Get the currently selected symbol (if any).
    pub fn current_symbol(&self) -> Option<&Symbol> {
        self.symbols.get(self.selected_symbol)
    }

    /// Get price history for the selected symbol.
    pub fn current_price_history(&self) -> &[f64] {
        self.current_symbol()
            .and_then(|s| self.price_history.get(s))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get bids for the selected symbol.
    pub fn current_bids(&self) -> &[BookLevel] {
        self.current_symbol()
            .and_then(|s| self.bids.get(s))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get asks for the selected symbol.
    pub fn current_asks(&self) -> &[BookLevel] {
        self.current_symbol()
            .and_then(|s| self.asks.get(s))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get last price for the selected symbol.
    pub fn current_last_price(&self) -> Option<Price> {
        self.current_symbol()
            .and_then(|s| self.last_price.get(s))
            .copied()
    }
}
