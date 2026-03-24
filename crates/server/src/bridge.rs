//! Channel bridge types for simulation ↔ server communication (V4.2).
//!
//! Provides message types and channel abstractions for the sync-async bridge.
//!
//! # Architecture
//!
//! ```text
//! Simulation (sync)                    Server (async)
//!       │                                   │
//!       │──── SimUpdate ────────────────────▶│ (broadcast to WS clients)
//!       │                                   │
//!       │◀─── SimCommand ───────────────────│ (pause/resume/step)
//!       │                                   │
//! ```
//!
//! # Design Principles
//!
//! - **Declarative**: Message types are plain data, no behavior
//! - **Modular**: Bridge is independent of simulation/server internals
//! - **SoC**: Types here, senders/receivers in respective modules

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use types::{Price, Symbol, Tick};

/// Per-symbol market data snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolData {
    /// Symbol identifier.
    pub symbol: Symbol,
    /// Last traded price.
    pub last_price: Option<f64>,
    /// Best bid price.
    pub best_bid: Option<f64>,
    /// Best ask price.
    pub best_ask: Option<f64>,
    /// Bid depth (top 5 levels).
    pub bid_depth: u64,
    /// Ask depth (top 5 levels).
    pub ask_depth: u64,
}

/// Per-tick data snapshot for WebSocket broadcast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    /// Current tick number.
    pub tick: Tick,
    /// Timestamp (ms since sim start).
    pub timestamp: u64,
    /// Per-symbol market data.
    pub symbols: HashMap<Symbol, SymbolData>,
    /// Trade count this tick.
    pub trades_this_tick: u64,
    /// Total trades so far.
    pub total_trades: u64,
    /// Total orders so far.
    pub total_orders: u64,
    /// Number of agents called this tick.
    pub agents_called: usize,
}

/// Full simulation update (for REST/initial state).
///
/// Contains more data than TickData for full state sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimUpdate {
    /// Current tick number.
    pub tick: Tick,
    /// Whether simulation is running.
    pub running: bool,
    /// Whether simulation has finished.
    pub finished: bool,
    /// Per-symbol tick data.
    pub tick_data: TickData,
    /// Total agent count.
    pub total_agents: usize,
    /// Tier 1 agent count.
    pub tier1_count: usize,
    /// Tier 2 agent count.
    pub tier2_count: usize,
    /// Tier 3 (background pool) count.
    pub tier3_count: usize,
}

impl Default for SimUpdate {
    fn default() -> Self {
        Self {
            tick: 0,
            running: false,
            finished: false,
            tick_data: TickData {
                tick: 0,
                timestamp: 0,
                symbols: HashMap::new(),
                trades_this_tick: 0,
                total_trades: 0,
                total_orders: 0,
                agents_called: 0,
            },
            total_agents: 0,
            tier1_count: 0,
            tier2_count: 0,
            tier3_count: 0,
        }
    }
}

/// Commands from server to simulation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimCommand {
    /// Start/resume simulation.
    Start,
    /// Pause simulation.
    Pause,
    /// Toggle pause/resume.
    Toggle,
    /// Step one tick (when paused).
    Step,
    /// Quit simulation.
    Quit,
}

/// Build SymbolData from hook context.
impl SymbolData {
    /// Create from symbol and price info.
    pub fn new(symbol: Symbol) -> Self {
        Self {
            symbol,
            last_price: None,
            best_bid: None,
            best_ask: None,
            bid_depth: 0,
            ask_depth: 0,
        }
    }

    /// Set prices from Price types.
    pub fn with_prices(
        mut self,
        last: Option<Price>,
        bid: Option<Price>,
        ask: Option<Price>,
    ) -> Self {
        self.last_price = last.map(|p| p.to_float());
        self.best_bid = bid.map(|p| p.to_float());
        self.best_ask = ask.map(|p| p.to_float());
        self
    }

    /// Set depth.
    pub fn with_depth(mut self, bid_depth: u64, ask_depth: u64) -> Self {
        self.bid_depth = bid_depth;
        self.ask_depth = ask_depth;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_data_serialization() {
        let data = TickData {
            tick: 100,
            timestamp: 1000,
            symbols: HashMap::new(),
            trades_this_tick: 5,
            total_trades: 500,
            total_orders: 1000,
            agents_called: 25,
        };

        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("\"tick\":100"));
    }

    #[test]
    fn test_sim_command_variants() {
        let cmds = [
            SimCommand::Start,
            SimCommand::Pause,
            SimCommand::Toggle,
            SimCommand::Step,
            SimCommand::Quit,
        ];

        for cmd in cmds {
            let json = serde_json::to_string(&cmd).unwrap();
            let _: SimCommand = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn test_symbol_data_builder() {
        let data = SymbolData::new("AAPL".into())
            .with_prices(
                Some(Price::from_float(150.0)),
                Some(Price::from_float(149.5)),
                Some(Price::from_float(150.5)),
            )
            .with_depth(1000, 1200);

        assert_eq!(data.last_price, Some(150.0));
        assert_eq!(data.best_bid, Some(149.5));
        assert_eq!(data.best_ask, Some(150.5));
        assert_eq!(data.bid_depth, 1000);
    }
}
